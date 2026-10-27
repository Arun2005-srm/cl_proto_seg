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
from torch.utils.data import DataLoader, Dataset

from .config import Config, deep_merge
from .data.datasets import SegDataset, build_splits, build_tasks, filter_domains
from .data.transforms import build_transforms
from .losses import DRIFTLoss, mask_logits
from .metrics import SegMetrics, continual_metrics, format_class_table
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
    key = hashlib.md5((cfg.dataset.root + str(cfg.dataset.get("label_map"))).encode()).hexdigest()[:10]
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


def make_loader(cfg, samples, split: str, shuffle: bool, batch_size: int, num_workers: int, drop_last=False):
    tf = build_transforms(cfg.transforms[split], cfg.dataset.ignore_index)
    ds = SegDataset(samples, cfg.dataset, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last and len(ds) > batch_size,
                      collate_fn=_collate, persistent_workers=False)


# ============================================================================ optim
def build_optimizer(model: DRIFT, tcfg):
    oc = tcfg.optimizer
    groups = []
    for name, params in model.param_groups().items():
        mult = float(oc.get("lr_mult", {}).get(name, 1.0))
        groups.append({"params": params, "lr": oc.lr * mult, "name": name, "initial_lr": oc.lr * mult})
    if oc.name == "adamw":
        return torch.optim.AdamW(groups, lr=oc.lr, weight_decay=oc.weight_decay)
    if oc.name == "adam":
        return torch.optim.Adam(groups, lr=oc.lr, weight_decay=oc.weight_decay)
    if oc.name == "sgd":
        return torch.optim.SGD(groups, lr=oc.lr, momentum=oc.momentum, weight_decay=oc.weight_decay, nesterov=True)
    raise KeyError(f"unknown optimizer {oc.name}")


def build_scheduler(opt, scfg, total_iters: int):
    warm = int(scfg.get("warmup_iters", 0))

    def f(it):
        w = min(1.0, (it + 1) / warm) if warm > 0 else 1.0
        p = min(it / max(total_iters, 1), 1.0)
        if scfg.name == "poly":
            return w * (1 - p) ** scfg.power
        if scfg.name == "cosine":
            return w * 0.5 * (1 + math.cos(math.pi * p))
        if scfg.name == "step":
            return w * scfg.gamma ** (it // max(1, scfg.step_size))
        return w
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ============================================================================ eval
@torch.no_grad()
def evaluate(model, loader, cfg, device, desc: str = "", with_loss: bool = True, max_batches=None) -> dict:
    """Phase 2: everything frozen, softmax over all classes, no memory write, no domain id."""
    model.eval()
    C, ign = len(cfg.dataset.classes), cfg.dataset.ignore_index
    met = SegMetrics(C, ign, cfg.eval.boundary_dilation, device=device)
    per_domain: dict[int, SegMetrics] = {}
    loss_sum, loss_n = 0.0, 0
    thr = cfg.model.inference.get("unknown_threshold")
    amp = cfg.experiment.amp and device.type == "cuda"
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        logits = F.interpolate(out["logits"].float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        if with_loss:
            l = F.cross_entropy(logits, y, ignore_index=ign)
            if torch.isfinite(l):
                loss_sum += l.item() * x.shape[0]
                loss_n += x.shape[0]
        prob = logits.softmax(1)
        conf, pred = prob.max(1)
        if thr is not None:
            pred[conf < thr] = ign
        met.update(pred, y)
        for d in b["domain"].unique().tolist():
            sel = b["domain"].to(device) == d
            per_domain.setdefault(d, SegMetrics(C, ign, device=device, compute_boundary=False)).update(pred[sel], y[sel])
    res = met.compute()
    res["loss"] = loss_sum / max(loss_n, 1)
    ds = loader.dataset
    idx2dom = {v: k for k, v in ds.domain_to_idx.items()}
    res["per_domain_mIoU"] = {idx2dom[d]: m.compute()["mIoU"] for d, m in sorted(per_domain.items())}
    return res


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
        print(f"[cl] {len(self.tasks)} tasks: " + " | ".join(f"T{i}:{','.join(t)}" for i, t in enumerate(self.tasks)))

        self.model = DRIFT(cfg).to(self.device)
        print("[model]\n" + self.model.module_report())
        self.criterion = DRIFTLoss(cfg).to(self.device)

        cache_dir = os.path.join(cfg.experiment.output_dir, "_cache")
        pres = class_presence(cfg, self.splits["train"], cache_dir, cfg.train.num_workers)
        self.task_valid = []
        for t in self.tasks:
            sel = np.array([s.domain in set(t) for s in self.splits["train"]])
            self.task_valid.append(torch.from_numpy(pres[sel].any(0)))
        for i, v in enumerate(self.task_valid):
            absent = [self.classes[c] for c in range(self.C) if not v[c]]
            print(f"[cl] T{i} valid set: {int(v.sum())}/{self.C} classes; absent: {', '.join(absent) or '-'}")

        self.history: list[dict] = []
        self.seen_curve: list[dict] = []       # per-epoch mIoU on every seen task (forgetting curve)
        self.test_curve: list[dict] = []       # held-out test after each task
        T = len(self.tasks)
        self.R = np.full((T, T), np.nan)       # R[i, j]: val mIoU on task j after task i
        self.global_epoch = 0

    # ------------------------------------------------------------------ helpers
    def _task_cfg(self, t: int) -> Config:
        ov = self.cfg.continual.get("task_overrides", {}) or {}
        o = ov.get(t, ov.get(str(t)))
        return Config(deep_merge(self.cfg.to_dict(), o.to_dict() if isinstance(o, Config) else o)) if o else self.cfg

    def _epochs(self, tcfg, t):
        e = tcfg.train.epochs
        return int(e[t] if isinstance(e, list) else e)

    def _val_loader(self, cfg, t):
        s = filter_domains(self.splits["val"], self.tasks[t]) if "val" in self.splits else []
        if not s:
            return None
        return make_loader(cfg, s, "val", False, cfg.eval.batch_size, cfg.eval.num_workers)

    def _allowed(self, cfg, t, labels):
        vs = cfg.continual.valid_set
        if vs == "all" or not cfg.loss.masked_softmax:
            return torch.ones(self.C, dtype=torch.bool)
        if vs == "batch":
            a = torch.zeros(self.C, dtype=torch.bool)
            u = labels.unique()
            a[u[(u < self.C)].cpu()] = True
            return a
        return self.task_valid[t]

    # ------------------------------------------------------------------ training
    def train_epoch(self, cfg, t, loader, opt, sched, scaler):
        self.model.train()
        tc = cfg.train
        tot, n, correct, count = 0.0, 0, 0, 0
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
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                out = self.model(x)
            loss, parts, (c_ok, c_n), hidden = self.criterion(self.model, out, y, allowed, mem_mask)
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
        return {"loss": tot / max(n, 1), "acc": correct / max(count, 1),
                **{f"loss_{k}": v / max(n, 1) for k, v in comps.items()}, "hidden_iter_frac": hidden_iters / max(n, 1)}

    @torch.no_grad()
    def write_memory(self, cfg, t):
        """End of task: EMA of z(p) per observed class into m_c. Absent classes untouched."""
        if not self.model.uses_memory:
            return
        self.model.eval()
        per = cfg.model.prototypes.memory.get("write_per", "task")
        groups = [[d] for d in self.tasks[t]] if per == "domain" else [self.tasks[t]]
        for g in groups:
            loader = make_loader(cfg, filter_domains(self.splits["train"], g), "val", False,
                                 cfg.eval.batch_size, cfg.eval.num_workers)
            D = cfg.model.embed_dim
            sums = torch.zeros(self.C, D, device=self.device, dtype=torch.float64)
            cnt = torch.zeros(self.C, device=self.device, dtype=torch.float64)
            for b in loader:
                x, y = b["image"].to(self.device), b["label"].to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    z = self.model(x)["z"]
                z = F.normalize(z.float(), dim=1)
                ys = F.interpolate(y[:, None].float(), size=z.shape[-2:], mode="nearest")[:, 0].long()
                v = (ys != cfg.dataset.ignore_index) & (ys < self.C)
                zf = z.permute(0, 2, 3, 1)[v].double()
                lab = ys[v]
                sums.index_add_(0, lab, zf)
                cnt += torch.bincount(lab, minlength=self.C).double()
            observed = (cnt > 0).cpu()
            self.model.prototypes.memory_write((sums / cnt.clamp(min=1)[:, None]).float(), observed.to(self.device))
            print(f"[memory] wrote {int(observed.sum())} classes from {','.join(g)}; "
                  f"{int(self.model.memory_mask().sum())}/{self.C} classes now in memory")

    def fit(self):
        cfg0 = self.cfg
        T = len(self.tasks)
        opt = sched = None
        for t in range(T):
            cfg = self._task_cfg(t)
            self.criterion.set_cfg(cfg.loss)
            self.criterion.to(self.device)
            epochs = self._epochs(cfg, t)
            tr = filter_domains(self.splits["train"], self.tasks[t])
            train_loader = make_loader(cfg, tr, "train", True, cfg.train.batch_size, cfg.train.num_workers, drop_last=True)
            val_loader = self._val_loader(cfg, t)
            iters = epochs * min(len(train_loader), cfg.train.get("max_iters_per_epoch") or 10 ** 9)
            if opt is None or cfg.train.reset_optimizer_per_task:
                opt = build_optimizer(self.model, cfg.train)
            sched = build_scheduler(opt, cfg.train.scheduler, max(1, iters // cfg.train.accum_steps))
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
            print(f"\n=== Task {t + 1}/{T}  [{', '.join(self.tasks[t])}]  train {len(tr)} imgs, "
                  f"{epochs} epochs, lr {cfg.train.optimizer.lr:g} ===")

            for ep in range(epochs):
                self.global_epoch += 1
                t0 = time.time()
                trs = self.train_epoch(cfg, t, train_loader, opt, sched, scaler)
                row = {"task": t, "epoch": ep + 1, "global_epoch": self.global_epoch,
                       "lr": opt.param_groups[0]["lr"], **{f"train_{k}": v for k, v in trs.items()}}
                do_eval = val_loader is not None and ((ep + 1) % cfg.eval.every == 0 or ep + 1 == epochs)
                if do_eval:
                    vr = evaluate(self.model, val_loader, cfg, self.device)
                    row.update(val_loss=vr["loss"], val_acc=vr["pixel_acc"], mIoU=vr["mIoU"], IoU=vr["fwIoU"],
                               BIoU=vr["BIoU"], Dice=vr["Dice"], mean_acc=vr["mean_acc"],
                               IoU_per_class=vr["IoU_per_class"], F1_per_class=vr["F1_per_class"],
                               BIoU_per_class=vr["BIoU_per_class"])
                    if cfg.eval.track_seen_every_epoch:
                        seen = {t: vr["mIoU"]}
                        for j in range(t):
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

            self.write_memory(cfg, t)
            for j in range(T):        # all tasks: past = forgetting, future = zero-shot transfer
                vl = self._val_loader(cfg, j)
                if vl is not None:
                    self.R[t, j] = evaluate(self.model, vl, cfg, self.device, with_loss=False)["mIoU"]
            cm = continual_metrics(self.R[: t + 1, : t + 1])
            print(f"[cl] after T{t}: val mIoU per task " +
                  " ".join(f"T{j}:{100 * self.R[t, j]:.1f}" for j in range(T) if not np.isnan(self.R[t, j])) +
                  f" | avg seen {100 * cm['avg_final']:.2f} | forgetting {100 * cm['avg_forgetting']:.2f}"
                  f" | BWT {100 * cm['BWT']:.2f}")
            if cfg.eval.test_after_each_task and "test" in self.splits:
                tl = make_loader(cfg, self.splits["test"], "test", False, cfg.eval.batch_size, cfg.eval.num_workers)
                te = evaluate(self.model, tl, cfg, self.device, with_loss=False)
                self.test_curve.append({"task": t, "mIoU": te["mIoU"], "BIoU": te["BIoU"], "Dice": te["Dice"],
                                        "IoU": te["fwIoU"], "pixel_acc": te["pixel_acc"]})
                print(f"[cl] after T{t}: held-out test mIoU {100 * te['mIoU']:.2f}  BIoU {100 * te['BIoU']:.2f}")
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

    def _save_history(self):
        dump_json({"history": self.history, "seen_curve": self.seen_curve, "test_curve": self.test_curve,
                   "R": self.R, "tasks": self.tasks, "classes": self.classes,
                   "continual": continual_metrics(self.R[: self._done_tasks(), : self._done_tasks()])
                   if self._done_tasks() else {}},
                  os.path.join(self.run_dir, "history.json"))
        keys = ["task", "epoch", "global_epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
                "mIoU", "IoU", "BIoU", "Dice", "mean_acc", "seen_avg_mIoU", "time_s"]
        with open(os.path.join(self.run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys + [f"IoU_{c}" for c in self.classes] + [f"F1_{c}" for c in self.classes])
            for r in self.history:
                w.writerow([r.get(k, "") for k in keys] + list(r.get("IoU_per_class", [""] * self.C))
                           + list(r.get("F1_per_class", [""] * self.C)))

    def _done_tasks(self):
        return int((~np.isnan(np.diag(self.R))).sum())

    def save_checkpoint(self, name, t):
        torch.save({"model": self.model.state_dict(), "task": t, "R": self.R, "tasks": self.tasks,
                    "cfg": self.cfg.to_dict()}, os.path.join(self.run_dir, name))


def load_model(cfg, ckpt_path, device):
    model = DRIFT(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if missing or unexpected:
        print(f"[ckpt] missing {len(missing)} / unexpected {len(unexpected)} keys")
    return model, sd
