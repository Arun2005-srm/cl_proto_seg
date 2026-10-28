"""Training (phase 1) and frozen inference (phase 2) for the domain-incremental schedule."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .config import Config, deep_merge
from .data.datasets import SegDataset, build_splits, build_tasks, filter_domains
from .data.transforms import build_transforms
from .losses import DRIFTLoss
from .metrics import (SegMetrics, bootstrap_miou, class_forgetting, continual_metrics, format_class_table,
                      miou_from_conf)
from .models.drift import DRIFT
from .utils import dump_json


# ============================================================================ data
class _Presence(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        lbl = self.ds.read_label(i)
        lbl = lbl[lbl != self.ds.ignore_index]
        return np.bincount(lbl.ravel(), minlength=self.ds.num_classes)[: self.ds.num_classes] > 0


def class_presence(cfg, samples, cache_dir: str, num_workers: int) -> np.ndarray:
    """[N, C] bool presence per sample, cached on disk by image path."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5((str(cfg.dataset.root) + str(cfg.dataset.get("label_map"))).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"presence_{cfg.dataset.name}_{key}.json")
    cache = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    todo = [s for s in samples if s.rel not in cache]
    if todo:
        ds = SegDataset(todo, cfg.dataset)
        dl = DataLoader(_Presence(ds), batch_size=16, num_workers=num_workers)
        rows = [r for b in dl for r in b.numpy()]
        for s, r in zip(todo, rows):
            cache[s.rel] = np.nonzero(r)[0].tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    C = len(cfg.dataset.classes)
    out = np.zeros((len(samples), C), dtype=bool)
    for i, s in enumerate(samples):
        out[i, cache[s.rel]] = True
    return out


def _collate(batch):
    return {"image": torch.stack([b["image"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "index": torch.tensor([b["index"] for b in batch]),
            "domain": torch.tensor([b["domain"] for b in batch])}


def make_loader(cfg, samples, split: str, shuffle: bool, batch_size: int, num_workers: int, drop_last=False,
                drop_classes=()):
    tf = build_transforms(cfg.transforms[split], cfg.dataset.ignore_index)
    ds = SegDataset(samples, cfg.dataset, transform=tf, drop_classes=drop_classes)
    sampler = None
    if shuffle and len(ds) < batch_size:
        # a task smaller than one batch: sample with replacement so every step sees a full batch
        # (batch size 1 breaks some fused kernels and makes statistics meaningless)
        sampler, shuffle = RandomSampler(ds, replacement=True, num_samples=batch_size), False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last and len(ds) >= batch_size,
                      collate_fn=_collate, persistent_workers=num_workers > 0)


# ============================================================================ optim
def build_optimizer(model: DRIFT, tcfg, lr_scale: float = 1.0):
    oc = tcfg.optimizer
    groups = []
    for name, params in model.param_groups().items():
        base = name.replace("_no_wd", "")
        mult = float(oc.get("lr_mult", {}).get(base, 1.0))
        lr = oc.lr * mult * lr_scale
        wd = 0.0 if name.endswith("_no_wd") else oc.weight_decay
        groups.append({"params": params, "lr": lr, "name": name, "initial_lr": lr, "weight_decay": wd})
    if oc.name == "adamw":
        return torch.optim.AdamW(groups, lr=oc.lr)
    if oc.name == "adam":
        return torch.optim.Adam(groups, lr=oc.lr)
    if oc.name == "sgd":
        return torch.optim.SGD(groups, lr=oc.lr, momentum=oc.momentum, nesterov=True)
    raise KeyError(f"unknown optimizer {oc.name}")


def build_scheduler(opt, scfg, total_iters: int, warmup: int):
    warm = min(int(warmup), max(0, total_iters // 10))   # never let warmup eat the whole task

    def f(it):
        w = min(1.0, (it + 1) / warm) if warm > 0 else 1.0
        p = min(it / max(total_iters, 1), 1.0)
        floor = float(scfg.get("min_lr_ratio", 0.0))
        if scfg.name == "poly":
            return w * max((1 - p) ** scfg.power, floor)
        if scfg.name == "cosine":
            return w * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))
        if scfg.name == "step":
            return w * scfg.gamma ** (it // max(1, scfg.step_size))
        return w
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ============================================================================ eval
def _forward_logits(model, x, out_size, tta, amp, device):
    scales = list(tta.get("scales", [1.0])) if tta else [1.0]
    flip = bool(tta.get("flip", False)) if tta else False
    acc, out0 = None, None
    for sc in scales:
        xs = x if sc == 1.0 else F.interpolate(x, scale_factor=sc, mode="bilinear", align_corners=False)
        for fl in ([False, True] if flip else [False]):
            xi = xs.flip(-1) if fl else xs
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(xi)
            lg = out["logits"].float()
            lg = lg.flip(-1) if fl else lg
            lg = F.interpolate(lg, size=out_size, mode="bilinear", align_corners=False).softmax(1)
            acc = lg if acc is None else acc + lg
            if out0 is None and sc == 1.0 and not fl:
                out0 = out
    return acc / (len(scales) * (2 if flip else 1)), out0


@torch.no_grad()
def evaluate(model, loader, cfg, device, with_loss: bool = True, max_batches=None, keep_per_image=False,
             collect_s=False, tta=None) -> dict:
    """Phase 2: everything frozen, softmax over all classes, no memory write, no domain id."""
    model.eval()
    C, ign = len(cfg.dataset.classes), cfg.dataset.ignore_index
    met = SegMetrics(C, ign, cfg.eval.boundary_dilation, device=device, keep_per_image=keep_per_image)
    per_domain: dict[int, SegMetrics] = {}
    loss_sum, loss_n = 0.0, 0
    thr = cfg.model.inference.get("unknown_threshold")
    amp = cfg.experiment.amp and device.type == "cuda"
    s_list, d_list = [], []
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
        prob, out = _forward_logits(model, x, y.shape[-2:], tta, amp, device)
        if with_loss:
            l = F.nll_loss(prob.clamp(min=1e-8).log(), y, ignore_index=ign)
            if torch.isfinite(l):
                loss_sum += l.item() * x.shape[0]
                loss_n += x.shape[0]
        conf, pred = prob.max(1)
        if thr is not None:
            pred[conf < thr] = ign
        met.update(pred, y)
        if collect_s and out is not None and out.get("s") is not None:
            s_list.append(out["s"].float().cpu())
            d_list.append(b["domain"])
        for d in b["domain"].unique().tolist():
            sel = b["domain"].to(device) == d
            per_domain.setdefault(d, SegMetrics(C, ign, device=device, compute_boundary=False)).update(pred[sel], y[sel])
    res = met.compute()
    res["loss"] = loss_sum / max(loss_n, 1)
    ds = loader.dataset
    idx2dom = {v: k for k, v in ds.domain_to_idx.items()}
    res["per_domain_mIoU"] = {idx2dom[d]: m.compute()["mIoU"] for d, m in sorted(per_domain.items())}
    if keep_per_image:
        res["per_image"] = met.per_image
    if collect_s and s_list:
        res["s"], res["s_domain"] = torch.cat(s_list), torch.cat(d_list)
    return res


def shift_diagnostics(s: torch.Tensor, dom: torch.Tensor, seed: int = 0) -> dict:
    """Is s collapsed? Reports per-dim std and nearest-centroid domain accuracy of s on a
    held-out half (chance = 1 / #domains). Low std + chance accuracy = no domain information."""
    s = s.float()
    out = {"s_std_mean": float(s.std(0).mean()) if len(s) > 1 else 0.0, "n": len(s),
           "num_domains": int(dom.unique().numel())}
    if out["num_domains"] < 2 or len(s) < 4:
        return out
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(s), generator=g)
    fit, held = perm[: len(s) // 2], perm[len(s) // 2:]
    sn = F.normalize(s, dim=-1)
    doms = dom[fit].unique()
    cents = torch.stack([sn[fit][dom[fit] == d].mean(0) for d in doms])
    pred = doms[(sn[held] @ F.normalize(cents, dim=-1).t()).argmax(1)]
    out["s_domain_acc"] = float((pred == dom[held]).float().mean())
    out["s_domain_chance"] = 1.0 / out["num_domains"]
    return out


# ============================================================================ trainer
class Trainer:
    def __init__(self, cfg: Config, run_dir: str):
        self.cfg, self.run_dir = cfg, run_dir
        dev = cfg.experiment.device
        self.device = torch.device(dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu")
        self.amp = bool(cfg.experiment.amp) and self.device.type == "cuda"
        self.classes = list(cfg.dataset.classes)
        self.C = len(self.classes)

        self.splits = build_splits(cfg)
        self.tasks = build_tasks(cfg, self.splits["train"])
        counts = {}
        for s in self.splits["train"]:
            counts[s.domain] = counts.get(s.domain, 0) + 1
        print(f"[cl] {len(self.tasks)} tasks: " + " | ".join(
            f"T{i}:{','.join(t) if len(t) <= 4 else ','.join(t[:2]) + f'..+{len(t) - 2}'} ({sum(counts[d] for d in t)} imgs)"
            for i, t in enumerate(self.tasks)))

        self.model = DRIFT(cfg).to(self.device)
        print("[model]\n" + self.model.module_report())
        self.criterion = DRIFTLoss(cfg).to(self.device)

        # controlled absence: {task: [class names]} removed from that task's training labels
        name2id = {c: i for i, c in enumerate(self.classes)}
        ab = cfg.continual.get("absent_classes", {}) or {}
        self.task_drop = [[name2id[c] for c in (ab.get(t, ab.get(str(t), [])) or [])] for t in range(len(self.tasks))]

        cache_dir = os.path.join(cfg.experiment.output_dir, "_cache")
        pres = class_presence(cfg, self.splits["train"], cache_dir, cfg.train.num_workers)
        self.task_valid, self.task_presence = [], []
        for t, tk in enumerate(self.tasks):
            sel = np.array([s.domain in set(tk) for s in self.splits["train"]])
            v = torch.from_numpy(pres[sel].any(0))
            v[self.task_drop[t]] = False
            self.task_valid.append(v)
            self.task_presence.append(torch.from_numpy(pres[sel].mean(0)).float())
        for i, v in enumerate(self.task_valid):
            absent = [self.classes[c] for c in range(self.C) if not v[c]]
            extra = f" (forced: {', '.join(self.classes[c] for c in self.task_drop[i])})" if self.task_drop[i] else ""
            print(f"[cl] T{i} valid set: {int(v.sum())}/{self.C} classes; absent: {', '.join(absent) or '-'}{extra}")

        self.history: list[dict] = []
        self.seen_curve: list[dict] = []       # per-epoch mIoU on every seen task (forgetting curve)
        self.test_curve: list[dict] = []       # held-out test after each task
        self.diag: list[dict] = []             # shift-estimator diagnostics per task
        T = len(self.tasks)
        self.R = np.full((T, T), np.nan)       # R[i, j]: val mIoU on task j after task i (classes present in j)
        self.R_common = np.full((T, T), np.nan)  # same, on the classes present in every task's val
        self.Rc = np.full((T, self.C), np.nan)  # per-class IoU on pooled val of tasks <= i (fixed class set)
        self.global_epoch = 0
        self._val_cache: dict[int, DataLoader | None] = {}
        self.common_classes = None
        self.task_hidden = None
        self.hide_candidates = None

    # ------------------------------------------------------------------ helpers
    def _task_cfg(self, t: int) -> Config:
        ov = self.cfg.continual.get("task_overrides", {}) or {}
        o = ov.get(t, ov.get(str(t)))
        return Config(deep_merge(self.cfg.to_dict(), o.to_dict() if isinstance(o, Config) else o)) if o else self.cfg

    def _epochs(self, tcfg, t, iters_per_epoch):
        ipt = tcfg.train.get("iters_per_task")
        if ipt:
            ipt = ipt[t] if isinstance(ipt, list) else ipt
            return max(1, math.ceil(int(ipt) / max(1, iters_per_epoch)))
        e = tcfg.train.epochs
        return int(e[t] if isinstance(e, list) else e)

    def _val_loader(self, cfg, t):
        if t not in self._val_cache:
            s = filter_domains(self.splits["val"], self.tasks[t]) if "val" in self.splits else []
            self._val_cache[t] = make_loader(cfg, s, "val", False, cfg.eval.batch_size, cfg.eval.num_workers) if s else None
        return self._val_cache[t]

    def _allowed(self, cfg, t, labels):
        vs = cfg.continual.valid_set
        if vs == "all":
            return torch.ones(self.C, dtype=torch.bool)
        if vs == "batch":
            a = torch.zeros(self.C, dtype=torch.bool)
            u = labels.unique()
            a[u[(u < self.C)].cpu()] = True
            return a
        return self.task_valid[t]

    def _lr_scale(self, cfg, t):
        oc = cfg.train.optimizer
        if t == 0:
            return 1.0
        scale = float(oc.get("later_task_lr_mult", 1.0)) * float(oc.get("lr_task_decay", 1.0)) ** (t - 1)
        return max(scale, float(oc.get("min_task_lr_mult", 0.0)))

    # ------------------------------------------------------------------ training
    def train_epoch(self, cfg, t, loader, opt, sched, scaler):
        self.model.train()
        tc = cfg.train
        tot, n, correct, count, skipped = 0.0, 0, 0, 0, 0
        comps: dict[str, float] = {}
        mem_mask = self.model.memory_mask()
        hidden_iters = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        max_it = tc.get("max_iters_per_epoch")
        for it, b in enumerate(loader):
            if max_it and it >= max_it:
                break
            x, y = b["image"].to(self.device, non_blocking=True), b["label"].to(self.device, non_blocking=True)
            allowed = self._allowed(cfg, t, y)
            th = self.task_hidden
            if th is not None and np.random.rand() >= float(cfg.loss.hidden.get("iter_prob", 1.0)):
                th = torch.zeros_like(th)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                out = self.model(x)
            loss, parts, (c_ok, c_n), hidden = self.criterion(self.model, out, y, allowed, mem_mask, th,
                                                              self.hide_candidates)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                if skipped > 20:
                    raise FloatingPointError("more than 20 non-finite losses in one epoch; lower the lr")
                continue
            hidden_iters += int(hidden.any())
            scaler.scale(loss / tc.accum_steps).backward()
            if (it + 1) % tc.accum_steps == 0:
                if tc.grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tc.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            tot += loss.item()
            n += 1
            correct += c_ok
            count += c_n
            for k, v in parts.items():
                comps[k] = comps.get(k, 0.0) + v
            if tc.log_interval and (it + 1) % tc.log_interval == 0:
                print(f"    it {it + 1:5d}/{len(loader)}  loss {tot / n:.4f}  acc {100 * correct / max(count, 1):.2f}"
                      f"  lr {opt.param_groups[0]['lr']:.2e}  {(time.time() - t0) / (it + 1):.2f}s/it", flush=True)
        if skipped:
            print(f"    [warn] skipped {skipped} iterations with a non-finite loss")
        return {"loss": tot / max(n, 1), "acc": correct / max(count, 1),
                **{f"loss_{k}": v / max(n, 1) for k, v in comps.items()}, "hidden_iter_frac": hidden_iters / max(n, 1)}

    @torch.no_grad()
    def _pixel_feats(self, model, loader, max_pix_per_img: int, seed: int):
        """Sampled normalised z(p) with labels, plus mean s, from a loader (no augmentation)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        model.eval()
        zs, ys, ss = [], [], []
        for b in loader:
            x, y = b["image"].to(self.device), b["label"].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                out = model(x)
            z = F.normalize(out["z"].float(), dim=1)
            yl = F.interpolate(y[:, None].float(), size=z.shape[-2:], mode="nearest")[:, 0].long()
            for k in range(x.shape[0]):
                zk = z[k].flatten(1).t()
                yk = yl[k].flatten()
                idx = torch.randperm(len(yk), generator=g)[:max_pix_per_img].to(self.device)
                zs.append(zk[idx].cpu())
                ys.append(yk[idx].cpu())
            if out.get("s") is not None:
                ss.append(out["s"].float().cpu())
        return torch.cat(zs), torch.cat(ys), (torch.cat(ss).mean(0) if ss else None)

    @torch.no_grad()
    def compensate_drift(self, cfg, t, old_model):
        """Semantic drift compensation (Yu et al., CVPR 2020) for the memory, using only the
        current task's images: each stored slot moves by the average z_new - z_old of current
        pixels, weighted by a Gaussian kernel around the slot (class-agnostic, so classes that are
        absent from this task are carried along with the feature space)."""
        sc = cfg.model.prototypes.memory.get("drift_compensation", {}) or {}
        if not sc.get("enabled", True) or old_model is None or not self.model.uses_memory:
            return
        pg = self.model.prototypes
        if not pg.slot_filled.any():
            return
        loader = make_loader(cfg, filter_domains(self.splits["train"], self.tasks[t]), "val", False,
                             cfg.eval.batch_size, cfg.eval.num_workers)
        ppi = int(sc.get("pixels_per_image", 512))
        z_old, y, _ = self._pixel_feats(old_model, loader, ppi, seed=t)
        z_new, _, _ = self._pixel_feats(self.model, loader, ppi, seed=t)
        valid = y != cfg.dataset.ignore_index
        z_old, z_new = z_old[valid].to(self.device), z_new[valid].to(self.device)
        delta = z_new - z_old
        sigma = float(sc.get("sigma", 0.3))
        mem = pg.memory                                            # [C,K,D]
        C, K, D = mem.shape
        flat = mem.reshape(C * K, D)
        drift = torch.zeros_like(flat)
        wsum = torch.zeros(C * K, device=self.device)
        for i in range(0, len(z_old), 8192):
            d2 = torch.cdist(flat, z_old[i:i + 8192]).pow(2)       # [CK, n]
            w = torch.exp(-d2 / (2 * sigma ** 2))
            drift += w @ delta[i:i + 8192]
            wsum += w.sum(1)
        drift = drift / wsum.clamp(min=1e-6).unsqueeze(1)
        drift[wsum < 1e-3] = 0
        before = flat.clone()
        pg.apply_drift(drift.reshape(C, K, D))
        moved = (1 - F.cosine_similarity(before, pg.memory.reshape(C * K, D), dim=-1))[pg.slot_filled.flatten()]
        print(f"[memory] drift compensation: mean slot shift {float(moved.mean()):.4f} (1 - cos) "
              f"from {len(z_old)} pixels")

    @torch.no_grad()
    def write_memory(self, cfg, t):
        """End of task: class means of z(p) per observed class, with the mean s as the slot key.
        Absent classes are untouched."""
        if not self.model.uses_memory:
            return
        self.model.eval()
        per = cfg.model.prototypes.memory.get("write_per", "domain")
        groups = [[d] for d in self.tasks[t]] if per == "domain" else [self.tasks[t]]
        for g in groups:
            loader = make_loader(cfg, filter_domains(self.splits["train"], g), "val", False,
                                 cfg.eval.batch_size, cfg.eval.num_workers, drop_classes=self.task_drop[t])
            D = cfg.model.embed_dim
            sums = torch.zeros(self.C, D, device=self.device, dtype=torch.float64)
            cnt = torch.zeros(self.C, device=self.device, dtype=torch.float64)
            s_sum, s_n = None, 0
            for b in loader:
                x, y = b["image"].to(self.device), b["label"].to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    out = self.model(x)
                z = F.normalize(out["z"].float(), dim=1)
                if out.get("s") is not None:
                    s_sum = out["s"].float().sum(0) if s_sum is None else s_sum + out["s"].float().sum(0)
                    s_n += x.shape[0]
                ys = F.interpolate(y[:, None].float(), size=z.shape[-2:], mode="nearest")[:, 0].long()
                v = (ys != cfg.dataset.ignore_index) & (ys < self.C)
                lab = ys[v]
                sums.index_add_(0, lab, z.permute(0, 2, 3, 1)[v].double())
                cnt += torch.bincount(lab, minlength=self.C).double()
            if self.task_hidden is not None and not cfg.loss.hidden.get("write_hidden", True):
                cnt[self.task_hidden.to(self.device)] = 0
            key = s_sum / s_n if s_sum is not None else None
            self.model.prototypes.memory_write((sums / cnt.clamp(min=1)[:, None]).float(), cnt.float(), key)
        print(f"[memory] T{t}: wrote {int((self.task_valid[t]).sum())} classes from {len(self.tasks[t])} domain(s); "
              f"{int(self.model.memory_mask().sum())}/{self.C} classes in memory, "
              f"{int(self.model.prototypes.slot_filled.sum())} slots filled")

    def evaluate_stage(self, cfg, t):
        """After task t: R / R_common / Rc and shift diagnostics."""
        T = len(self.tasks)
        confs = {}
        s_all, d_all = [], []
        for j in range(T):
            vl = self._val_loader(cfg, j)
            if vl is None:
                continue
            r = evaluate(self.model, vl, cfg, self.device, with_loss=False, collect_s=j <= t)
            self.R[t, j] = r["mIoU"]
            confs[j] = (np.asarray(r["confusion"]), np.asarray(r["unknown_fn"]))
            if "s" in r:
                s_all.append(r["s"])
                d_all.append(torch.full((len(r["s"]),), j))       # task id as the domain label
        if self.common_classes is None and confs:
            present = [c[0].sum(1) > 0 for c in confs.values()]
            self.common_classes = np.logical_and.reduce(present)
        for j, (cm, un) in confs.items():
            self.R_common[t, j] = miou_from_conf(cm, un, self.common_classes)
        seen = [j for j in confs if j <= t]
        if seen:
            pooled = sum(confs[j][0] for j in seen)
            tp = np.diag(pooled)
            with np.errstate(divide="ignore", invalid="ignore"):
                iou = tp / (pooled.sum(1) + pooled.sum(0) - tp)
            self.Rc[t] = np.where(pooled.sum(1) > 0, iou, np.nan)
        if s_all:
            d = shift_diagnostics(torch.cat(s_all), torch.cat(d_all), seed=t)
            d["task"] = t
            self.diag.append(d)
            if "s_domain_acc" in d:
                print(f"[shift] after T{t}: s std {d['s_std_mean']:.3f}, task-from-s accuracy "
                      f"{100 * d['s_domain_acc']:.1f}% (chance {100 * d['s_domain_chance']:.1f}%)")

    def fit(self):
        cfg0 = self.cfg
        T = len(self.tasks)
        opt = sched = None
        for t in range(T):
            cfg = self._task_cfg(t)
            self.criterion.set_cfg(cfg.loss)
            self.criterion.to(self.device)
            tr = filter_domains(self.splits["train"], self.tasks[t])
            train_loader = make_loader(cfg, tr, "train", True, cfg.train.batch_size, cfg.train.num_workers,
                                       drop_last=True, drop_classes=self.task_drop[t])
            ipe = min(len(train_loader), cfg.train.get("max_iters_per_epoch") or 10 ** 9)
            epochs = self._epochs(cfg, t, ipe)
            iters = epochs * ipe
            lr_scale = self._lr_scale(cfg, t)
            if opt is None or cfg.train.reset_optimizer_per_task:
                opt = build_optimizer(self.model, cfg.train, lr_scale)
            else:
                for gr in opt.param_groups:
                    gr["initial_lr"] = gr["lr"] = cfg.train.optimizer.lr * lr_scale * float(
                        cfg.train.optimizer.get("lr_mult", {}).get(gr["name"].replace("_no_wd", ""), 1.0))
            warm = cfg.train.scheduler.get("warmup_iters", 0) if (t == 0 or cfg.train.scheduler.get("warmup_every_task")) else 0
            sched = build_scheduler(opt, cfg.train.scheduler, max(1, iters // cfg.train.accum_steps), warm)
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
            self.model.mark_seen(self.task_valid[t])
            self.task_hidden = None
            # classes that may be hidden: valid here and not dominant (present in <= max_presence of the
            # task's images) - simulated absence targets the rarer classes the hypothesis is about
            self.hide_candidates = self.task_valid[t] & (
                self.task_presence[t] <= float(cfg.loss.hidden.get("max_presence", 1.0)))
            if cfg.loss.hidden.enabled and cfg.loss.hidden.get("scope", "iteration") == "task":
                self.task_hidden = self.criterion.sample_hidden(self.hide_candidates, self.model.memory_mask())
            old_model = copy.deepcopy(self.model).eval() if (t > 0 and self.model.uses_memory) else None
            hid = (", hidden for the task: " + ", ".join(self.classes[c] for c in torch.nonzero(self.task_hidden).flatten())
                   ) if self.task_hidden is not None and self.task_hidden.any() else ""
            print(f"\n=== Task {t + 1}/{T}  [{', '.join(self.tasks[t][:6])}{'...' if len(self.tasks[t]) > 6 else ''}]"
                  f"  train {len(tr)} imgs, {epochs} epochs x {ipe} it, lr {cfg.train.optimizer.lr * lr_scale:.2e}{hid} ===")

            for ep in range(epochs):
                self.global_epoch += 1
                t0 = time.time()
                trs = self.train_epoch(cfg, t, train_loader, opt, sched, scaler)
                row = {"task": t, "epoch": ep + 1, "global_epoch": self.global_epoch,
                       "lr": opt.param_groups[0]["lr"], **{f"train_{k}": v for k, v in trs.items()}}
                val_loader = self._val_loader(cfg, t)
                do_eval = val_loader is not None and ((ep + 1) % cfg.eval.every == 0 or ep + 1 == epochs)
                if do_eval:
                    vr = evaluate(self.model, val_loader, cfg, self.device)
                    row.update(val_loss=vr["loss"], val_acc=vr["pixel_acc"], mIoU=vr["mIoU"], IoU=vr["fwIoU"],
                               BIoU=vr["BIoU"], Dice=vr["Dice"], mean_acc=vr["mean_acc"],
                               IoU_per_class=vr["IoU_per_class"], F1_per_class=vr["F1_per_class"],
                               BIoU_per_class=vr["BIoU_per_class"])
                    if cfg.eval.track_seen_every_epoch:
                        seen = {t: vr["mIoU"]}
                        for j in range(max(0, t - int(cfg.eval.get("seen_max_tasks", 10 ** 6))), t):
                            vl = self._val_loader(cfg, j)
                            if vl is not None:
                                seen[j] = evaluate(self.model, vl, cfg, self.device, with_loss=False)["mIoU"]
                        row["seen_mIoU"] = {str(k): v for k, v in sorted(seen.items())}
                        row["seen_avg_mIoU"] = float(np.mean(list(seen.values())))
                        self.seen_curve.append({"global_epoch": self.global_epoch, "task": t, **row["seen_mIoU"]})
                row["time_s"] = time.time() - t0
                self.history.append(row)
                self._print_epoch(row, epochs, t, T, cfg)
                self._save_history()
                self.save_checkpoint("last.pt", t)

            self.compensate_drift(cfg, t, old_model)
            del old_model
            self.write_memory(cfg, t)
            self.evaluate_stage(cfg, t)
            cm = continual_metrics(self.R[: t + 1, : t + 1])
            cmc = continual_metrics(self.R_common[: t + 1, : t + 1])
            cf = class_forgetting(self.Rc[: t + 1])
            shown = [j for j in range(T) if not np.isnan(self.R[t, j])][:12]
            print(f"[cl] after T{t}: val mIoU " + " ".join(f"T{j}:{100 * self.R[t, j]:.1f}" for j in shown) +
                  (" ..." if T > 12 else "") +
                  f" | avg seen {100 * cm['avg_final']:.2f} | forgetting {100 * cm['avg_forgetting']:.2f}"
                  f" (common classes {100 * cmc['avg_forgetting']:.2f}, class-level {100 * cf['avg_class_forgetting']:.2f})"
                  f" | BWT {100 * cm['BWT']:.2f}")
            for name in [n for n in self.splits if n.startswith("val_") or (n == "test" and cfg.eval.test_after_each_task)]:
                tl = make_loader(cfg, self.splits[name], "test", False, cfg.eval.batch_size, cfg.eval.num_workers)
                te = evaluate(self.model, tl, cfg, self.device, with_loss=False)
                self.test_curve.append({"split": name, "task": t, "mIoU": te["mIoU"], "BIoU": te["BIoU"],
                                        "Dice": te["Dice"], "IoU": te["fwIoU"], "pixel_acc": te["pixel_acc"]})
                print(f"[cl] after T{t}: {name} mIoU {100 * te['mIoU']:.2f}  BIoU {100 * te['BIoU']:.2f}")
            self.save_checkpoint(f"task{t}.pt", t)
            self._save_history()
        self.cfg = cfg0
        return self.history

    # ------------------------------------------------------------------ logging
    def _print_epoch(self, r, epochs, t, T, cfg):
        s = (f"ep[{r['epoch']}/{epochs}] T[{t + 1}/{T}] | train loss {r['train_loss']:.4f} acc {100 * r['train_acc']:.2f}")
        if "val_loss" in r:
            s += (f" | val loss {r['val_loss']:.4f} acc {100 * r['val_acc']:.2f}"
                  f" | mIoU {100 * r['mIoU']:.2f} IoU(fw) {100 * r['IoU']:.2f} BIoU {100 * r['BIoU']:.2f}"
                  f" Dice {100 * r['Dice']:.2f}")
        if "seen_avg_mIoU" in r and t > 0:
            s += f" | seen-avg mIoU {100 * r['seen_avg_mIoU']:.2f}"
        parts = [f"{k[11:]} {v:.3f}" for k, v in r.items() if k.startswith("train_loss_")]
        s += f" | {r['time_s']:.0f}s"
        print(s)
        print(f"    loss terms: {', '.join(parts)}  hidden-iters {100 * r['train_hidden_iter_frac']:.0f}%  lr {r['lr']:.2e}")
        if "IoU_per_class" in r and cfg.eval.print_class_table:
            print(format_class_table(r, self.classes, cols=("IoU", "F1", "BIoU")))

    def _done_tasks(self):
        return int((~np.isnan(np.diag(self.R))).sum())

    def continual_summary(self) -> dict:
        d = self._done_tasks()
        if not d:
            return {}
        out = continual_metrics(self.R[:d, :d])
        out["common_classes"] = continual_metrics(self.R_common[:d, :d])
        out.update(class_forgetting(self.Rc[:d]))
        return out

    def _save_history(self):
        dump_json({"history": self.history, "seen_curve": self.seen_curve, "test_curve": self.test_curve,
                   "R": self.R, "R_common": self.R_common, "Rc": self.Rc, "tasks": self.tasks,
                   "classes": self.classes, "shift_diagnostics": self.diag,
                   "common_classes": None if self.common_classes is None else
                   [self.classes[c] for c in np.nonzero(self.common_classes)[0]],
                   "continual": self.continual_summary()},
                  os.path.join(self.run_dir, "history.json"))
        keys = ["task", "epoch", "global_epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
                "mIoU", "IoU", "BIoU", "Dice", "mean_acc", "seen_avg_mIoU", "time_s"]
        with open(os.path.join(self.run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys + [f"IoU_{c}" for c in self.classes] + [f"F1_{c}" for c in self.classes])
            for r in self.history:
                w.writerow([r.get(k, "") for k in keys] + list(r.get("IoU_per_class", [""] * self.C))
                           + list(r.get("F1_per_class", [""] * self.C)))

    def save_checkpoint(self, name, t):
        torch.save({"model": self.model.checkpoint_state(), "task": t, "R": self.R, "tasks": self.tasks,
                    "encoder_frozen": self.model.encoder_frozen, "cfg": self.cfg.to_dict()},
                   os.path.join(self.run_dir, name))


def load_model(cfg, ckpt_path, device):
    """Strict load. Only frozen encoder weights may be absent (checkpoints omit them;
    they are rebuilt from model.encoder.pretrained)."""
    model = DRIFT(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    frozen = {f"encoder.{n}" for n, p in model.encoder.named_parameters() if not p.requires_grad and "lora_" not in n}
    bad_missing = [k for k in missing if k not in frozen]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the config: {len(bad_missing)} missing "
                           f"(e.g. {bad_missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    return model, sd
