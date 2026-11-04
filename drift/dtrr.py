"""Drift-Triggered Reactive Replay (DTRR), self-adjusting.

Old-task memory stays feature-level (no image or pixel of a finished task is kept):
per class a diagonal-Gaussian sketch of the decoder input u(p) (per-scale encoder
features upsampled to stride 4, concatenated), a histogram of logit margins, a pixel
count, the memory-slot snapshot and the image-level statistics needed to replay it.

During a task:
  probe set    N current-task pixels (images discarded at task end); every K steps
               eps_hat = mean ||norm(z_t) - norm(z_0)|| and delta_hat = max prototype shift
  estimator    F_meas[c]  = rise of the error rate of pseudo-features sampled from the sketch
               F_bound[c] = P(margin_c in [0, 2 (eps_hat + delta_hat) / tau))   (stored histogram)
               F_hat[c]   = max(F_meas, F_bound)
  controller   lam[c] <- clip(lam + alpha (F_hat - rho_c) + beta stale[c] 1[F_hat > 0], 0, lam_max)
  replay       classes drawn with p ~ lam, pseudo-features through decoder tail + cosine head,
               conflict-gated gradient (PCGrad-style projection when <g_cur, g_mem> < 0),
               LoRA lr scaled by 1 / (1 + mean lam)
lam = 0 everywhere means no replay and no extra backward pass. Inference is unchanged.

Every stage asserts that it actually ran (probe built, drift visible, sketches stored and
reloadable, minor classes queued, controller consumed eps_hat) and raises otherwise.
"""
from __future__ import annotations

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

from .models.components import FusionDecoder, GroupStatsTracker
from .models.lora import lora_layers
from .utils import dump_json


class DTRRError(RuntimeError):
    """A DTRR stage silently failed (no drift signal, missing memory, ...)."""


# ============================================================================ class statistics
def label_mapping_hash(ds) -> str:
    """Fingerprint of the dataset-label -> union-class mapping a SegDataset applies."""
    h = hashlib.sha1()
    h.update(json.dumps([ds.num_classes, ds.ignore_index, sorted(ds.drop_classes)]).encode())
    luts = {"": ds.lut} if not ds.luts else ds.luts
    for name in sorted(luts):
        h.update(name.encode())
        lut = luts[name]
        h.update(b"identity" if lut is None else np.ascontiguousarray(lut).tobytes())
    return h.hexdigest()[:16]


class _PixelCount(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        lbl = self.ds.read_label(i)
        lbl = lbl[lbl != self.ds.ignore_index]
        return torch.from_numpy(np.bincount(lbl.ravel(), minlength=self.ds.num_classes)[: self.ds.num_classes])


def class_pixel_stats(cfg, samples, drop_classes, num_workers: int) -> dict:
    """One pass over the training LABELS of a task (no image, no model): per-class pixel and image counts."""
    from .data.datasets import SegDataset

    ds = SegDataset(samples, cfg.dataset, drop_classes=drop_classes)
    C = ds.num_classes
    pix = torch.zeros(C, dtype=torch.float64)
    imgs = torch.zeros(C, dtype=torch.float64)
    # own generator: iterating a DataLoader draws its base seed from the global RNG otherwise
    for b in DataLoader(_PixelCount(ds), batch_size=16, num_workers=num_workers, generator=torch.Generator()):
        pix += b.double().sum(0)
        imgs += (b > 0).double().sum(0)
    return {"pixels": pix.numpy(), "images": imgs.numpy(), "n_images": len(ds), "mapping_hash": label_mapping_hash(ds)}


def classify_classes(stats: dict, percentile: float = 25.0, min_share: float = 0.005) -> dict:
    """q_c = pixel share. minor: present and (q_c < P-th percentile of q over present classes or q_c < min_share).
    absent: class of the union vocabulary with no pixel in this dataset."""
    pix = np.asarray(stats["pixels"], dtype=np.float64)
    q = pix / max(pix.sum(), 1.0)
    present = pix > 0
    thr = float(np.percentile(q[present], percentile)) if present.any() else 0.0
    minor = present & ((q < thr) | (q < min_share))
    return {"q": q, "image_share": np.asarray(stats["images"]) / max(stats["n_images"], 1), "present": present,
            "minor": minor, "absent": ~present, "q_threshold": thr}


def write_class_stats(path, t, classes, stats, info, extra=None):
    rows = [{"index": c, "class": classes[c], "pixels": int(stats["pixels"][c]), "q": float(info["q"][c]),
             "images": int(stats["images"][c]), "image_share": float(info["image_share"][c]),
             "minor": bool(info["minor"][c]), "absent": bool(info["absent"][c])} for c in range(len(classes))]
    dump_json({"task": t, "n_images": stats["n_images"], "q_threshold_p25": info["q_threshold"],
               "mapping_hash": stats["mapping_hash"], "minor": [classes[c] for c in np.nonzero(info["minor"])[0]],
               "absent": [classes[c] for c in np.nonzero(info["absent"])[0]], "classes": rows, **(extra or {})}, path)


def format_class_stats(classes, stats, info) -> str:
    lines = [f"    {'class':16s} {'q_c':>8s} {'img share':>9s} {'minor':>6s} {'absent':>7s}"]
    for c, name in enumerate(classes):
        lines.append(f"    {name:16s} {100 * info['q'][c]:7.3f}% {100 * info['image_share'][c]:8.1f}% "
                     f"{'yes' if info['minor'][c] else '-':>6s} {'yes' if info['absent'][c] else '-':>7s}")
    return "\n".join(lines)


class ClassTracker:
    """Per class, over the whole run: steps in which it received gradient and staleness
    (steps since it last did). Updated every optimizer step for every union-vocabulary class."""

    def __init__(self, C: int, ema: float = 0.01):
        self.C, self.ema = C, ema
        self.staleness = torch.zeros(C)
        self.mean_staleness = 0.0           # running mean over classes and steps (normaliser)
        self.reset_task()

    def reset_task(self):
        self.times_seen = torch.zeros(self.C, dtype=torch.long)
        self.max_stale = torch.zeros(self.C)
        self.sum_stale = torch.zeros(self.C)
        self.steps = 0

    def update(self, received: torch.Tensor):
        received = received.cpu().bool()
        self.staleness += 1
        self.staleness[received] = 0
        self.times_seen += received.long()
        self.max_stale = torch.maximum(self.max_stale, self.staleness)
        self.sum_stale += self.staleness
        self.steps += 1
        self.mean_staleness = (1 - self.ema) * self.mean_staleness + self.ema * float(self.staleness.mean())

    def normalised(self) -> torch.Tensor:
        return self.staleness / (self.mean_staleness + 1.0)


# ============================================================================ margins
def margin_edges(bins: int, lim: float) -> torch.Tensor:
    return torch.linspace(-lim, lim, bins + 1)


def margin_hist(margins: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Counts per bin; values outside the range go to the first / last bin."""
    m = margins.float().clamp(float(edges[0]), float(edges[-1]) - 1e-6).cpu()
    return torch.histc(m, bins=len(edges) - 1, min=float(edges[0]), max=float(edges[-1]))


def hist_cdf(hist: torch.Tensor, edges: torch.Tensor, x: float) -> float:
    """P(margin < x) under the histogram, linear inside a bin."""
    tot = float(hist.sum())
    if tot <= 0:
        return 0.0
    lo, hi = edges[:-1], edges[1:]
    frac = ((x - lo) / (hi - lo)).clamp(0, 1)
    return float((hist * frac).sum() / tot)


def margins_of(logits: torch.Tensor, y: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """gamma = l_y - max_{c != y, c allowed} l_c for logits [N, C]."""
    lg = logits.float().masked_fill(~allowed.to(logits.device).view(1, -1), -1e4)
    true = lg.gather(1, y.view(-1, 1)).squeeze(1)
    other = lg.scatter(1, y.view(-1, 1), -1e4).max(1).values
    return true - other


def class_cells(y: torch.Tensor, h: int, w: int, C: int, ign: int):
    """Per class, the z-grid cells (b, row, col) [K, 3] of labels y [B, H, W] on an h x w grid, plus the
    nearest-sampled label map [B, h, w]. A cell takes the label at its centre; a class that has pixels
    but no cell centre (thin or tiny objects) gets the cells covering its pixels, so every class that
    appears in the labels is represented."""
    H, W = y.shape[-2:]
    ys = ((torch.arange(h, device=y.device) + 0.5) * H / h).long().clamp(0, H - 1)
    xs = ((torch.arange(w, device=y.device) + 0.5) * W / w).long().clamp(0, W - 1)
    lz = y[:, ys][:, :, xs]
    out = {}
    full = y.unique().tolist()
    centre = set(lz.unique().tolist())
    for c in full:
        if c == ign or c >= C:
            continue
        if c in centre:
            out[c] = (lz == c).nonzero()
        else:
            px = (y == c).nonzero()
            cells = torch.stack([px[:, 0], px[:, 1] * h // H, px[:, 2] * w // W], 1)
            out[c] = cells.unique(dim=0)
    return out, lz


class RingQueue:
    """FIFO of the last `size` decoder-input features of one class, kept on the training device (fp16)."""

    def __init__(self, size: int, dim: int, device):
        self.buf = torch.zeros(size, dim, dtype=torch.float16, device=device)
        self.n, self.ptr = 0, 0

    def __len__(self):
        return self.n

    def push(self, u: torch.Tensor):
        u = u[-len(self.buf):].to(self.buf)
        k = len(u)
        idx = (self.ptr + torch.arange(k, device=self.buf.device)) % len(self.buf)
        self.buf[idx] = u
        self.ptr = (self.ptr + k) % len(self.buf)
        self.n = min(self.n + k, len(self.buf))

    def sample(self, k: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(self.n, (k,), generator=gen).to(self.buf.device)
        return self.buf[idx].float()


# ============================================================================ sketch store
class SketchStore:
    """Feature-level memory keyed by union-vocabulary class index: {class: [entry per task]}.
    No image or raw pixel is stored."""

    TENSORS = ("mean", "var", "hist", "slot", "slot_count", "stats_key", "gn_mean", "gn_var")

    def __init__(self, classes, edges: torch.Tensor):
        self.classes = list(classes)
        self.edges = edges
        self.entries: dict[int, list[dict]] = {}
        self.task_protos: dict[int, torch.Tensor] = {}     # prototypes [C, D] at the end of each task

    def add(self, c: int, entry: dict):
        self.entries.setdefault(int(c), []).append(entry)

    def has(self, c: int) -> bool:
        return int(c) in self.entries

    def class_mask(self, C: int) -> torch.Tensor:
        m = torch.zeros(C, dtype=torch.bool)
        for c in self.entries:
            m[c] = True
        return m

    def nbytes(self) -> int:
        n = sum(v.numel() * v.element_size() for es in self.entries.values() for e in es
                for v in e.values() if torch.is_tensor(v))
        return n + sum(p.numel() * p.element_size() for p in self.task_protos.values())

    def pick(self, c: int, gen: torch.Generator) -> dict:
        es = self.entries[int(c)]
        w = torch.tensor([float(e["count"]) for e in es])
        return es[int(torch.multinomial(w / w.sum(), 1, generator=gen))]

    @staticmethod
    def sample(entry: dict, n: int, gen: torch.Generator) -> torch.Tensor:
        noise = torch.randn(n, entry["mean"].numel(), generator=gen)
        return entry["mean"].float() + entry["var"].float().clamp(min=0).sqrt() * noise

    # ---------------------------------------------------------------- persistence
    def state(self) -> dict:
        return {"classes": self.classes, "edges": self.edges,
                "entries": {c: [dict(e) for e in es] for c, es in self.entries.items()},
                "task_protos": dict(self.task_protos)}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state(), path)

    @classmethod
    def load(cls, path: str) -> "SketchStore":
        st = torch.load(path, map_location="cpu", weights_only=False)
        s = cls(st["classes"], st["edges"])
        s.entries = {int(c): list(es) for c, es in st["entries"].items()}
        s.task_protos = {int(k): v for k, v in st["task_protos"].items()}
        return s

    def digest(self, c: int) -> str:
        h = hashlib.sha1()
        for e in self.entries[int(c)]:
            h.update(f"{e['task']}|{e['count']}".encode())
            for k in self.TENSORS:
                v = e.get(k)
                if v is not None:
                    h.update(v.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def audit(self) -> dict:
        out = {}
        for c, es in sorted(self.entries.items()):
            out[self.classes[c]] = {
                "index": c, "tasks": [int(e["task"]) for e in es], "counts": [int(e["count"]) for e in es],
                "low_confidence": [bool(e["low_confidence"]) for e in es],
                "mean_norm": [float(e["mean"].float().norm()) for e in es],
                "mean_var": [float(e["var"].float().mean()) for e in es],
                "err0": [float(e["err0"]) for e in es],
                "self_consistency": [float(e["self_consistency"]) for e in es],
                "digest": self.digest(c)}
        return out


def verify_reload(store: SketchStore, path: str, seen: list[int]) -> SketchStore:
    """Reload from disk into a fresh object; same keys, same shapes, finite, count > 0 for every seen class."""
    fresh = SketchStore.load(path)
    if sorted(fresh.entries) != sorted(store.entries):
        raise DTRRError(f"memory reload: keys differ ({sorted(fresh.entries)} vs {sorted(store.entries)})")
    for c, es in store.entries.items():
        fe = fresh.entries[c]
        if len(fe) != len(es):
            raise DTRRError(f"memory reload: class {store.classes[c]} has {len(fe)} entries, wrote {len(es)}")
        for a, b in zip(es, fe):
            for k in SketchStore.TENSORS:
                if (a.get(k) is None) != (b.get(k) is None):
                    raise DTRRError(f"memory reload: {store.classes[c]}.{k} presence differs")
                if a.get(k) is None:
                    continue
                if a[k].shape != b[k].shape:
                    raise DTRRError(f"memory reload: {store.classes[c]}.{k} shape {tuple(b[k].shape)} != {tuple(a[k].shape)}")
                if not torch.isfinite(b[k].float()).all():
                    raise DTRRError(f"memory reload: NaN/inf in {store.classes[c]}.{k}")
    for c in seen:
        if not fresh.has(c) or sum(e["count"] for e in fresh.entries[c]) <= 0:
            raise DTRRError(f"memory reload: seen class '{store.classes[c]}' (index {c}) has no stored sketch / count 0")
    return fresh


# ============================================================================ probe / drift
_PROBE_SETTINGS = ("eval", "no_grad", "fp32", "lora_unmerged")


@torch.no_grad()
def probe_embed(model, probe: dict, gn_tracker: GroupStatsTracker | None = None):
    """THE single code path for z0 and every z_t: model.eval(), no_grad, fp32 (no autocast),
    adapters unmerged, the stored (already preprocessed) probe images. Restores the training mode.
    Returns (normalised z at the probe pixels [N, D], prototypes [C, D], settings)."""
    if any(m.merged for m in lora_layers(model.encoder)):
        raise DTRRError("probe_embed: LoRA adapters are merged; drift must be measured with unmerged adapters")
    was_training = model.training
    dev = next(model.parameters()).device
    if gn_tracker is not None:
        gn_tracker.paused = True
    model.eval()
    try:
        N = len(probe["labels"])
        zs = None
        protos = []
        for i, img in enumerate(probe["images"]):
            out = model(img[None].to(dev).float())
            z = F.normalize(out["z"].float(), dim=1)[0]
            h, w = z.shape[-2:]
            sel = (probe["img"] == i).nonzero().flatten()
            if zs is None:
                zs = torch.zeros(N, z.shape[0], device=dev)
            ys = (probe["ny"][sel] * h).long().clamp(0, h - 1).to(dev)
            xs = (probe["nx"][sel] * w).long().clamp(0, w - 1).to(dev)
            zs[sel.to(dev)] = z[:, ys, xs].t()
            if out.get("protos") is not None:
                protos.append(out["protos"][0].float())
        if protos:
            p = F.normalize(torch.stack(protos).mean(0), dim=-1)
        else:
            p = F.normalize(model.head.cls.weight.flatten(1).float(), dim=-1)
    finally:
        model.train(was_training)
        if gn_tracker is not None:
            gn_tracker.paused = False
    return zs, p, _PROBE_SETTINGS


def build_probe(model, loader, C: int, ignore_index: int, num_pixels: int, min_per_class: int,
                max_images: int, scan_batches: int, gen: torch.Generator) -> dict:
    """Stratified probe pixels from the first `scan_batches` batches of the current task:
    every class present there contributes >= min_per_class pixels (all of them if it has fewer).
    Pixels live on the z grid (stride 4) and are stored as normalised coordinates."""
    cands = []           # per scanned image: (image tensor, {class: [K, 2] (y, x) on the z grid}, (h, w))
    for bi, b in enumerate(loader):
        if bi >= scan_batches:
            break
        for k in range(b["image"].shape[0]):
            img, lbl = b["image"][k], b["label"][k]
            H, W = lbl.shape
            h, w = math.ceil(H / 4), math.ceil(W / 4)
            cells, _ = class_cells(lbl[None], h, w, C, ignore_index)
            per = {c: v[:, 1:] for c, v in cells.items()}
            if per:
                cands.append((img, per, (h, w)))
    if not cands:
        raise DTRRError("probe: no labelled pixel in the first batches of the task")
    scanned = sorted({c for _, per, _ in cands for c in per})
    # greedy coverage, rarest class first, then fill with random images
    n_img_per_class = {c: sum(c in per for _, per, _ in cands) for c in scanned}
    chosen: list[int] = []
    for c in sorted(scanned, key=lambda c: n_img_per_class[c]):
        if any(c in cands[i][1] for i in chosen):
            continue
        best = max((i for i in range(len(cands)) if c in cands[i][1]),
                   key=lambda i: int(cands[i][1][c].shape[0]))
        chosen.append(best)
    rest = [i for i in torch.randperm(len(cands), generator=gen).tolist() if i not in chosen]
    chosen += rest[: max(0, max(max_images, len(chosen)) - len(chosen))]
    pool = []            # (slot, class, y, x, h, w)
    for slot, i in enumerate(chosen):
        _, per, (h, w) = cands[i]
        for c, yx in per.items():
            for y, x in yx.tolist():
                pool.append((slot, c, y, x, h, w))
    pool_t = torch.tensor([p[:4] for p in pool])
    take = torch.zeros(len(pool), dtype=torch.bool)
    for c in scanned:
        idx = (pool_t[:, 1] == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=gen)[:min_per_class]]
        take[idx] = True
    left = (~take).nonzero().flatten()
    extra = max(0, num_pixels - int(take.sum()))
    take[left[torch.randperm(len(left), generator=gen)[:extra]]] = True
    sel = take.nonzero().flatten()
    hw = torch.tensor([[pool[i][4], pool[i][5]] for i in sel.tolist()], dtype=torch.float32)
    probe = {"images": [cands[i][0].half() for i in chosen], "img": pool_t[sel, 0], "labels": pool_t[sel, 1],
             "ny": (pool_t[sel, 2].float() + 0.5) / hw[:, 0], "nx": (pool_t[sel, 3].float() + 0.5) / hw[:, 1],
             "scanned_classes": scanned}
    for c in scanned:
        have = int((probe["labels"] == c).sum())
        avail = int((pool_t[:, 1] == c).sum())
        if have < min(min_per_class, avail):
            raise DTRRError(f"probe: class {c} has {have} probe pixels, needs {min(min_per_class, avail)}")
    return probe


def project_conflict(g_cur: torch.Tensor, g_mem: torch.Tensor):
    """If <g_cur, g_mem> < 0, remove from g_cur its component along g_mem. Returns (g, dot, projected)."""
    dot = torch.dot(g_cur, g_mem)
    nm = torch.dot(g_mem, g_mem)
    if dot < 0 and nm > 0:
        return g_cur - dot / nm * g_mem, float(dot), True
    return g_cur, float(dot), False


class DualController:
    """lam[c] <- clip(lam[c] + alpha (F_hat[c] - rho_c) + beta clip(stale_norm[c] - 1, 0, 1) 1[F_hat[c] > 0], 0, lam_max).
    No trigger threshold: F_hat < rho_c drives lam down, F_hat > rho_c drives it up. Staleness above the
    running mean only adds a head start once forgetting is measured at all (F_hat > 0)."""

    def __init__(self, C: int, alpha=0.5, beta=0.05, lam_max=5.0, rho=0.02):
        self.C, self.alpha, self.beta, self.lam_max, self.rho = C, alpha, beta, lam_max, rho
        self.lam = torch.zeros(C)
        self.rho_c = torch.full((C,), float(rho))
        self.updates = 0
        self.eps_updates = 0

    def set_budget(self, share: torch.Tensor | None):
        """rho_c = rho (share / mean share)^0.5: rarer classes get a tighter budget."""
        if share is None:
            self.rho_c = torch.full((self.C,), float(self.rho))
            return
        s = share.float()
        m = s[s > 0].mean() if (s > 0).any() else torch.tensor(1.0)
        self.rho_c = torch.where(s > 0, self.rho * (s / m).sqrt(), torch.full_like(s, float(self.rho)))

    def update(self, F_hat: torch.Tensor, eligible: torch.Tensor, stale_norm: torch.Tensor, used_eps: bool):
        head = self.beta * (stale_norm - 1).clamp(0, 1) * (F_hat > 0).float()
        new = (self.lam + self.alpha * (F_hat - self.rho_c) + head).clamp(0, self.lam_max)
        self.lam = torch.where(eligible, new, torch.zeros_like(new))
        self.updates += 1
        self.eps_updates += int(used_eps)
        return self.lam


# ============================================================================ orchestrator
class DTRR:
    def __init__(self, cfg, trainer):
        self.cfg, self.tr = cfg, trainer
        self.dc = cfg.dtrr
        self.model = trainer.model
        self.device = trainer.device
        self.C = trainer.C
        self.classes = trainer.classes
        self.ign = cfg.dataset.ignore_index
        self.out_dir = os.path.join(trainer.run_dir, "dtrr")
        os.makedirs(self.out_dir, exist_ok=True)
        self.gen = torch.Generator().manual_seed(int(cfg.experiment.seed) + 7919)    # never the global RNG
        tau = float(self.model.head.log_tau.exp()) if hasattr(self.model.head, "log_tau") else 0.05
        self.tau = tau
        self.store = SketchStore(self.classes, margin_edges(int(self.dc.sketch.bins), 2.0 / tau))
        self.gn = GroupStatsTracker(self.model.decoder.norm_module(), float(self.dc.online.get("gn_momentum", 0.1)))
        c = self.dc.controller
        self.ctrl = DualController(self.C, float(c.alpha), float(c.beta), float(c.lam_max), float(c.rho))
        self.seen_share = torch.zeros(self.C)       # max pixel share of each class over the tasks seen so far
        self.audit_path = None
        self.mem_path = None
        self.report: dict = {"tasks": []}
        self.total_steps = 0
        self.replay_steps_total = 0

    # ---------------------------------------------------------------- helpers
    @property
    def protect_params(self):
        mods = [self.model.decoder, self.model.head]
        if self.model.shift_estimator is not None:
            mods.append(self.model.shift_estimator)
        if self.model.prototypes is not None:
            mods.append(self.model.prototypes)
        return [p for m in mods for p in m.parameters() if p.requires_grad]

    def _s_from_stats(self, stats_key):
        se = self.model.shift_estimator
        if se is None or stats_key is None:
            return None
        return se.head(stats_key.to(self.device).float().view(1, -1))

    def _logits(self, z: torch.Tensor, s, protos=None):
        """Cosine (or linear) logits [N, C] for pixel embeddings z [N, D]."""
        if self.model.prototypes is None:
            return F.linear(z, self.model.head.cls.weight.flatten(1), self.model.head.cls.bias)
        if protos is None:
            protos = self.model.prototypes(s, 1)[0]
        return F.normalize(z.float(), dim=-1) @ protos.float().t() / self.model.head.log_tau.exp().clamp(min=1e-3)

    def _seen_mask(self) -> torch.Tensor:
        m = self.store.class_mask(self.C)
        if self.model.prototypes is not None:
            m |= self.model.prototypes.class_seen.cpu()
        m |= self.valid
        return m

    # ---------------------------------------------------------------- task start
    def start_task(self, cfg, t: int, train_samples, train_loader, class_stats: dict, class_info: dict):
        self.cfg, self.dc, self.t = cfg, cfg.dtrr, t
        dc = self.dc
        self.valid = self.tr.task_valid[t].clone()
        self.minor = torch.from_numpy(class_info["minor"]) & self.valid
        self.absent = torch.from_numpy(class_info["absent"])
        self.seen_share = torch.maximum(self.seen_share, torch.from_numpy(class_info["q"]).float())
        self.ctrl.set_budget(self.seen_share if dc.controller.get("rho_by_share", False) else None)
        self.ctrl.lam.zero_()
        self.task_report = {"task": t, "minor": [self.classes[c] for c in torch.nonzero(self.minor).flatten()],
                            "absent": [self.classes[c] for c in torch.nonzero(self.absent).flatten()]}

        # single source of truth for dataset label -> union index
        h_train = label_mapping_hash(train_loader.dataset)
        if h_train != class_stats["mapping_hash"]:
            raise DTRRError(f"label mapping used for class statistics ({class_stats['mapping_hash']}) differs from "
                            f"the training loader's ({h_train})")
        if self.store.classes != self.classes:
            raise DTRRError("memory class vocabulary differs from the training vocabulary")
        self.task_report["mapping_hash"] = h_train

        # memory from the previous task, from disk, checked against its audit
        prev_ok = None
        if t > 0 and self.mem_path is not None:
            fresh = SketchStore.load(self.mem_path)
            with open(self.audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)["classes"]
            missing = [n for n, a in audit.items() if not fresh.has(a["index"])]
            if missing:
                raise DTRRError(f"T{t} start: previously seen classes missing from memory: {missing}")
            bad = [n for n, a in audit.items() if fresh.digest(a["index"]) != a["digest"]]
            if bad:
                raise DTRRError(f"T{t} start: memory of {bad} does not match memory_audit.json (hash)")
            self.store = fresh
            prev_ok = sorted(audit)
            print(f"[dtrr] T{t} start: memory loaded from {self.mem_path}: {len(prev_ok)} classes, hashes match the "
                  f"audit file ({', '.join(prev_ok)})")
        self.task_report["prev_classes_present"] = prev_ok

        # per-task online state
        du = sum(self.model.encoder.out_channels)
        self.queues = {c: RingQueue(int(dc.queue.size), du, self.device)
                       for c in torch.nonzero(self.minor).flatten().tolist()}
        self.pushes = torch.zeros(self.C, dtype=torch.long)
        self.online = {}                       # class -> (mean, var) EMA of u
        self.last_stats_key = None
        self.task_step = 0
        self.replay_steps = 0
        self.replay_px = 0
        self.conflicts = 0
        self.replay_time = 0.0
        self.step_time = 0.0
        self.batch_px = 0
        self.lam_max_seen = torch.zeros(self.C)
        self.corr_pairs = []
        self.best_err = {}                     # intra: lowest pseudo-error of each minor class so far this task
        self.curve_rows = []
        self.ctrl_rows = []
        self.lam_bar = 0.0
        self.lam_bar_inter = 0.0

        # probe set + z0 + self-test
        from .engine import make_loader

        sub = [train_samples[i] for i in torch.randperm(len(train_samples), generator=self.gen).tolist()]
        loader = make_loader(cfg, sub, "val", False, cfg.train.batch_size, cfg.eval.num_workers,
                             drop_classes=self.tr.task_drop[t], generator=torch.Generator())
        pc = dc.probe
        self.probe = build_probe(self.model, loader, self.C, self.ign, int(pc.num_pixels), int(pc.min_per_class),
                                 int(pc.max_images), int(pc.scan_batches), self.gen)
        self._embed = probe_embed
        self.z0, self.p0, self.settings0 = probe_embed(self.model, self.probe, self.gn)
        cnt = torch.bincount(self.probe["labels"], minlength=self.C)
        print(f"[dtrr] T{t} probe: {len(self.probe['labels'])} pixels from {len(self.probe['images'])} images; "
              f"per class " + ", ".join(f"{self.classes[c]} {int(cnt[c])}" for c in self.probe["scanned_classes"]))
        self._self_test()
        self.task_report.update(probe_pixels=len(self.probe["labels"]), probe_images=len(self.probe["images"]),
                                probe_per_class={self.classes[c]: int(cnt[c]) for c in self.probe["scanned_classes"]},
                                self_test=True)
        self.prev_val0 = self._prev_val_miou() if (t > 0 and int(dc.get("corr_val_batches", 0)) > 0) else None

    def _self_test(self):
        """Perturb LoRA -> eps_hat > 0; restore -> eps_hat == 0. Raises otherwise."""
        layers = lora_layers(self.model.encoder)
        params = [p for m in layers for p in (m.lora_A, m.lora_B)] or list(self.model.decoder.proj[0].parameters())
        what = "LoRA" if layers else "decoder.proj[0] (no LoRA in this model)"
        backup = [p.detach().clone() for p in params]
        noise = float(self.dc.probe.get("self_test_noise", 1e-3))
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn(p.shape, generator=self.gen).to(p) * noise)
        z1, _, _ = probe_embed(self.model, self.probe, self.gn)
        e1 = float((z1 - self.z0).norm(dim=1).mean())
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)
        z2, _, _ = probe_embed(self.model, self.probe, self.gn)
        e2 = float((z2 - self.z0).norm(dim=1).mean())
        ok = e1 > 0 and e2 <= 1e-6
        print(f"[dtrr] drift self-test ({what} noise {noise:g}): perturbed eps {e1:.3e} (> 0), "
              f"restored eps {e2:.3e} (<= 1e-6) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise DTRRError(f"drift self-test failed: perturbed eps {e1:.3e}, restored eps {e2:.3e}. The probe does "
                            f"not see {what} changes, or the probe path is not deterministic.")

    # ---------------------------------------------------------------- per step
    @torch.no_grad()
    def observe(self, out, y, received: torch.Tensor):
        """Online sketches of the current task's classes and FIFO queues of its minor classes."""
        feats = [f.detach().float() for f in out["feats"]]
        h, w = out["z"].shape[-2:]
        cells, lz = class_cells(y, h, w, self.C, self.ign)
        self.batch_px = int(((lz != self.ign) & (lz < self.C)).sum())
        if out.get("stats") is not None:
            self.last_stats_key = out["stats"].detach().float().mean(0).cpu()
        k = int(self.dc.online.pixels_per_class_step)
        kq = int(self.dc.queue.push_per_step)
        mom = float(self.dc.online.momentum)
        sel, owner = [], []
        for c in torch.nonzero(received).flatten().tolist():
            idx = cells.get(c)
            if idx is None or len(idx) == 0:
                continue
            n = min(len(idx), kq if c in self.queues else k)
            pick = torch.randperm(len(idx), generator=self.gen)[:n].to(idx.device)
            sel.append(idx[pick])
            owner.append((c, n))
        if not sel:
            return
        idx = torch.cat(sel)
        u = FusionDecoder.gather_inputs(feats, idx[:, 0], idx[:, 1:]).float()   # one gather for every class
        off = 0
        for c, n in owner:
            uc = u[off:off + n]
            off += n
            m, v = uc.mean(0), uc.var(0, unbiased=False)
            if c in self.online:
                om, ov = self.online[c]
                self.online[c] = ((1 - mom) * om + mom * m, (1 - mom) * ov + mom * v)
            else:
                self.online[c] = (m, v)
            if c in self.queues:
                self.queues[c].push(uc)
                self.pushes[c] += 1

    def current_lam(self) -> torch.Tensor:
        dc = self.dc
        mode = dc.get("mode", "adaptive")
        elig = self.eligible()
        lc = float(dc.replay.get("lam_const", 1.0))
        if mode == "adaptive":
            return self.ctrl.lam * elig.float()
        if mode == "always":
            return lc * elig.float()
        if mode == "periodic":
            return lc * elig.float() if self.task_step % int(dc.replay.period) == 0 else torch.zeros(self.C)
        if mode == "threshold":
            th = dc.threshold
            eps = getattr(self, "last_eps", 0.0)
            fire = (self.tr.tracker.staleness > float(th.staleness)) | torch.tensor(eps > float(th.eps))
            return lc * (elig & fire).float()
        raise KeyError(f"unknown dtrr.mode '{mode}' (adaptive | always | periodic | threshold)")

    def eligible(self) -> torch.Tensor:
        e = torch.zeros(self.C, dtype=torch.bool)
        if self.dc.get("inter", True):
            e |= self.store.class_mask(self.C)
        if self.dc.get("intra", True):
            e |= torch.tensor([c in self.queues and len(self.queues[c]) > 0 for c in range(self.C)])
        return e

    def _budget(self, lam: torch.Tensor) -> int:
        r = self.dc.replay
        mode = self.dc.get("mode", "adaptive")
        if mode in ("always", "periodic"):
            n = int(r.budget_pixels) * (int(r.period) if mode == "periodic" else 1)
            return n if lam.sum() > 0 else 0
        return int(min(int(r.max_pixels), math.ceil(float(r.pixels_per_lam) * float(lam.sum()))))

    def _replay_batch(self, lam: torch.Tensor, n: int):
        """Groups of (u [k, Du], y [k], stats_key, gn_stats) drawn with class probability ~ lam."""
        cls = torch.multinomial(lam / lam.sum(), n, replacement=True, generator=self.gen)
        groups = []
        for c in cls.unique().tolist():
            k = int((cls == c).sum())
            from_queue = c in self.queues and len(self.queues[c]) > 0 and self.dc.get("intra", True)
            from_sketch = self.store.has(c) and self.dc.get("inter", True)
            if from_queue and from_sketch:
                kq = int(torch.binomial(torch.tensor(float(k)), torch.tensor(0.5), generator=self.gen))
            else:
                kq = k if from_queue else 0
            if kq:
                u = self.queues[c].sample(kq, self.gen)
                groups.append((u, c, self.last_stats_key, self.gn.stats()))
            if k - kq:
                e = self.store.pick(c, self.gen)
                groups.append((SketchStore.sample(e, k - kq, self.gen), c, e.get("stats_key"),
                               (e["gn_mean"], e["gn_var"]) if e.get("gn_mean") is not None else None))
        return groups

    def before_step(self, opt, scaler):
        """Conflict-gated replay gradient, added to the current-task gradient in place (lam = 0: no-op)."""
        lam = self.current_lam()
        n = self._budget(lam) if lam.sum() > 0 else 0
        self.lam_bar = float(lam[lam > 0].mean()) if (lam > 0).any() else 0.0
        # the LoRA brake protects old tasks from encoder drift: only old-task (inter) classes drive it,
        # replaying the current task's rare classes must not slow the current task's encoder adaptation
        inter = lam * self.store.class_mask(self.C).float()
        self.lam_bar_inter = float(inter[inter > 0].mean()) if (inter > 0).any() else 0.0
        if n <= 0:
            return
        t0 = time.time()
        groups = self._replay_batch(lam, n)
        allowed = self._seen_mask().to(self.device)
        logits, ys = [], []
        for u, c, sk, gn in groups:
            z = self.model.decoder.forward_pixels(u.to(self.device), gn)
            logits.append(self._logits(z, self._s_from_stats(sk)))
            ys.append(torch.full((len(u),), c, device=self.device, dtype=torch.long))
        lg = torch.cat(logits).float().masked_fill(~allowed.view(1, -1), -1e4)
        l_mem = F.cross_entropy(lg, torch.cat(ys))
        params = self.protect_params
        scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        g_mem = torch.autograd.grad(l_mem * scale, params, allow_unused=True)
        g_mem = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(g_mem, params)])
        g_cur = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
        g, dot, proj = project_conflict(g_cur.float(), g_mem.float())
        g = g + self.lam_bar * g_mem.float()
        off = 0
        for p in params:
            k = p.numel()
            p.grad = g[off:off + k].view_as(p).to(p.dtype)
            off += k
        self.replay_steps += 1
        self.replay_steps_total += 1
        self.replay_px += n
        self.conflicts += int(proj)
        self.last_l_mem = float(l_mem)
        self.replay_time += time.time() - t0

    def after_step(self, opt, step_time: float):
        """LoRA brake and, every K steps (and at step 1), the drift monitor + controller."""
        self.task_step += 1
        self.total_steps += 1
        self.step_time += step_time
        if self.dc.replay.get("lora_brake", True) and self.lam_bar_inter > 0:
            for g in opt.param_groups:
                if g["name"].startswith("lora"):
                    g["lr"] = g["lr"] / (1.0 + self.lam_bar_inter)
        if self.task_step == 1 or self.task_step % int(self.dc.interval) == 0:
            self.monitor()

    # ---------------------------------------------------------------- monitor / estimator / controller
    @torch.no_grad()
    def _pseudo_error(self, c: int, entry: dict | None, n: int, protos=None, gn=None) -> tuple[float, float]:
        """(fraction with margin < 0, fraction classified as c) for n pseudo-features of class c."""
        was = self.model.training
        self.model.eval()
        try:
            if entry is not None:
                u = SketchStore.sample(entry, n, self.gen)
                gn = (entry["gn_mean"], entry["gn_var"]) if entry.get("gn_mean") is not None else None
                sk = entry.get("stats_key")
            else:
                u = self.queues[c].sample(n, self.gen)
                gn, sk = self.gn.stats(), self.last_stats_key
            z = self.model.decoder.forward_pixels(u.to(self.device), gn)
            lg = self._logits(z, self._s_from_stats(sk), protos)
            allowed = self._seen_mask().to(self.device)
            y = torch.full((n,), c, device=self.device, dtype=torch.long)
            m = margins_of(lg, y, allowed)
            pred = lg.masked_fill(~allowed.view(1, -1), -1e4).argmax(1)
            return float((m < 0).float().mean()), float((pred == c).float().mean())
        finally:
            self.model.train(was)

    def monitor(self):
        if self._embed is not probe_embed:
            raise DTRRError("probe embeddings must come from the single probe_embed code path")
        zt, pt, settings = probe_embed(self.model, self.probe, self.gn)
        if settings != self.settings0:
            raise DTRRError(f"probe settings changed: {settings} vs {self.settings0}")
        d = (zt - self.z0).norm(dim=1)
        eps_hat = float(d.mean())
        labels = self.probe["labels"].to(d.device)
        eps_c = {c: float(d[labels == c].mean()) for c in self.probe["scanned_classes"]}
        seen = self._seen_mask().to(pt.device)
        delta_hat = float((pt - self.p0).norm(dim=1)[seen].max()) if seen.any() else 0.0
        self.last_eps = eps_hat
        worst = 2 * (eps_hat + delta_hat) / self.tau             # Lipschitz bound on a margin change
        if self.dc.estimator.get("bound", "calibrated") == "lipschitz":
            thr = worst
        else:   # mean |margin change| actually realised on the probe pixels (<= worst by construction)
            allowed = seen.to(zt.device)
            y = labels.to(zt.device)
            g0 = margins_of(self.z0 @ self.p0.t() / self.tau, y, allowed)
            gt = margins_of(zt @ pt.t() / self.tau, y, allowed)
            thr = float((gt - g0).abs().mean())
        M = int(self.dc.estimator.samples)
        F_meas, F_bound = torch.zeros(self.C), torch.zeros(self.C)
        elig = self.eligible()
        for c in torch.nonzero(elig).flatten().tolist():
            fm, fb = 0.0, 0.0
            if self.store.has(c) and self.dc.get("inter", True):
                e = self.store.pick(c, self.gen)
                err, _ = self._pseudo_error(c, e, M)
                fm = max(fm, err - float(e["err0"]))
                hist = sum(x["hist"] for x in self.store.entries[c])
                fb = max(0.0, hist_cdf(hist, self.store.edges, thr) - hist_cdf(hist, self.store.edges, 0.0))
            if c in self.queues and len(self.queues[c]) and self.dc.get("intra", True):
                err, _ = self._pseudo_error(c, None, M)
                best = min(self.best_err.get(c, err), err)     # intra forgetting: rise over the best so far
                self.best_err[c] = best
                fm = max(fm, err - best)
            F_meas[c], F_bound[c] = max(fm, 0.0), fb
        F_hat = torch.maximum(F_meas, F_bound)
        stale = self.tr.tracker.normalised()
        lam = self.ctrl.update(F_hat, elig, stale, used_eps=True) if self.dc.get("mode", "adaptive") == "adaptive" \
            else self.current_lam()
        self.lam_max_seen = torch.maximum(self.lam_max_seen, lam)
        row = {"step": self.task_step, "eps_hat": eps_hat, "delta_hat": delta_hat, "bound_arg": thr,
               "bound_worst": worst,
               "lam_sum": float(lam.sum()), "replay_px": self.replay_px, "replay_steps": self.replay_steps,
               "replay_frac": self.replay_steps / max(self.task_step, 1),
               **{f"eps_{self.classes[c]}": v for c, v in eps_c.items()}}
        if self.prev_val0 is not None and int(self.dc.get("corr_val_batches", 0)) > 0:
            drop = self.prev_val0 - self._prev_val_miou()
            row["prev_val_drop"] = drop
            self.corr_pairs.append((eps_hat, drop))
        self.curve_rows.append(row)
        for c in torch.nonzero(elig | (lam > 0)).flatten().tolist():
            self.ctrl_rows.append({"step": self.task_step, "class": self.classes[c], "F_meas": float(F_meas[c]),
                                   "F_bound": float(F_bound[c]), "F_hat": float(F_hat[c]), "lam": float(lam[c]),
                                   "rho_c": float(self.ctrl.rho_c[c]), "staleness": float(self.tr.tracker.staleness[c]),
                                   "inter": self.store.has(c), "intra": c in self.queues})
        act = [f"{self.classes[c]} {float(lam[c]):.2f}" for c in torch.nonzero(lam > 0).flatten().tolist()]
        print(f"    [dtrr] step {self.task_step:5d}  eps_hat {eps_hat:.4f}  delta_hat {delta_hat:.4f}  "
              f"margin-risk {thr:.2f} (worst {worst:.1f}) logits  "
              f"F_hat max {float(F_hat.max()):.3f}  lam>0: {', '.join(act) or '-'}  "
              f"replay {self.replay_steps}/{self.task_step} steps, {self.replay_px} px", flush=True)

    def _prev_val_miou(self) -> float:
        from .engine import evaluate

        vl = self.tr._val_loader(self.cfg, self.t - 1)
        if vl is None:
            return float("nan")
        was = self.model.training
        r = evaluate(self.model, vl, self.cfg, self.device, with_loss=False,
                     max_batches=int(self.dc.get("corr_val_batches", 0)))
        self.model.train(was)
        return float(r["mIoU"])

    # ---------------------------------------------------------------- task end
    def end_training(self):
        """Last drift measurement of the task, before the memory is rewritten."""
        self.monitor()

    @torch.no_grad()
    def _sketch_pass(self, cfg, t, samples):
        """Per-class decoder-input moments, margins and counts on the task's training images (val transform)."""
        from .engine import make_loader

        loader = make_loader(cfg, samples, "val", False, cfg.eval.batch_size, cfg.eval.num_workers,
                             drop_classes=self.tr.task_drop[t], generator=torch.Generator())
        P = int(self.dc.sketch.pixels_per_class_image)
        acc = {}
        stats_sum, s_n = None, 0
        tracker = GroupStatsTracker(self.model.decoder.norm_module(), momentum=0.0)
        tracker.cumulative = True
        self.gn.paused = True
        self.model.eval()
        allowed = self._seen_mask().to(self.device)
        try:
            for b in loader:
                x, y = b["image"].to(self.device), b["label"].to(self.device)
                out = self.model(x)
                if out.get("stats") is not None:
                    st = out["stats"].float().sum(0)
                    stats_sum = st if stats_sum is None else stats_sum + st
                    s_n += x.shape[0]
                h, w = out["z"].shape[-2:]
                cells, _ = class_cells(y, h, w, self.C, self.ign)
                feats = [f.float() for f in out["feats"]]
                logits = out["logits"].float()
                for c, idx in cells.items():
                    a = acc.setdefault(c, {"sum": 0.0, "sq": 0.0, "n": 0, "count": 0,
                                           "hist": torch.zeros(len(self.store.edges) - 1)})
                    a["count"] += len(idx)
                    for bi in idx[:, 0].unique().tolist():
                        ib = idx[idx[:, 0] == bi]
                        ib = ib[torch.randperm(len(ib), generator=self.gen)[:P].to(ib.device)]
                        u = FusionDecoder.gather_inputs(feats, ib[:, 0], ib[:, 1:]).double()
                        a["sum"] = a["sum"] + u.sum(0)
                        a["sq"] = a["sq"] + (u * u).sum(0)
                        a["n"] += len(ib)
                        lg = logits[ib[:, 0], :, ib[:, 1], ib[:, 2]]
                        m = margins_of(lg, torch.full((len(ib),), c, device=lg.device, dtype=torch.long), allowed)
                        a["hist"] += margin_hist(m, self.store.edges)
        finally:
            self.gn.paused = False
            self.model.train()
        gn = tracker.stats()
        tracker.remove()
        stats_key = (stats_sum / s_n).cpu() if stats_sum is not None else None
        return acc, stats_key, gn

    def end_task(self, cfg, t: int, train_samples, mem_counts: torch.Tensor | None):
        dc = self.dc
        acc, stats_key, gn = self._sketch_pass(cfg, t, train_samples)
        sk = self.dc.sketch
        pg = self.model.prototypes
        with torch.no_grad():
            protos = self.model.prototypes(self._s_from_stats(stats_key), 1)[0].detach().cpu() \
                if pg is not None else None
        if protos is not None:
            self.store.task_protos[t] = protos
        seen_now = sorted(acc)
        for c in seen_now:
            a = acc[c]
            mean = (a["sum"] / a["n"]).float().cpu()
            var = (a["sq"] / a["n"] - (a["sum"] / a["n"]) ** 2).clamp(min=0).float().cpu()
            entry = {"task": t, "mean": mean, "var": var, "hist": a["hist"].cpu(), "count": int(a["count"]),
                     "low_confidence": a["count"] < int(sk.low_confidence_count),
                     "slot": pg.memory[c].detach().cpu().clone() if (pg is not None and pg.use_memory) else None,
                     "slot_count": pg.mem_count[c].detach().cpu().clone() if (pg is not None and pg.use_memory) else None,
                     "slot_src_pixels": int(mem_counts[c]) if mem_counts is not None else None,
                     "stats_key": stats_key, "gn_mean": gn[0].cpu() if gn else None, "gn_var": gn[1].cpu() if gn else None,
                     "share": float(self.seen_share[c])}
            err0, _ = self._pseudo_error(c, entry, int(dc.estimator.samples) * 4)
            _, cons = self._pseudo_error(c, entry, int(sk.consistency_samples),
                                         protos=protos.to(self.device) if protos is not None else None)
            entry["err0"], entry["self_consistency"] = err0, cons
            self.store.add(c, entry)

        # save, reload into a fresh object, audit
        self.mem_path = os.path.join(self.out_dir, f"memory_after_T{t}.pt")
        self.store.save(self.mem_path)
        verify_reload(self.store, self.mem_path, seen_now)
        audit = self.store.audit()
        self.audit_path = os.path.join(self.out_dir, "memory_audit.json")
        dump_json({"after_task": t, "file": self.mem_path, "bytes": self.store.nbytes(), "classes": audit},
                  self.audit_path)
        low = [self.classes[c] for c in seen_now if self.store.entries[c][-1]["low_confidence"]]
        print(f"[dtrr] T{t} memory: wrote sketches for {len(seen_now)} classes ({', '.join(self.classes[c] for c in seen_now)})"
              f"{'; low confidence: ' + ', '.join(low) if low else ''}; reload check PASS; "
              f"total memory {self.store.nbytes() / 1024:.1f} KiB ({sum(len(v) for v in self.store.entries.values())} "
              f"sketches, {len(self.store.entries)} classes) -> {self.audit_path}")
        weak = [(self.classes[c], self.store.entries[c][-1]["self_consistency"]) for c in seen_now
                if self.store.entries[c][-1]["self_consistency"] < float(sk.consistency_warn)]
        print("[dtrr] sketch self-consistency (64 pseudo-features -> stored prototypes, fraction kept as the class): "
              + ", ".join(f"{self.classes[c]} {self.store.entries[c][-1]['self_consistency']:.2f}" for c in seen_now))
        if weak:
            print(f"[dtrr] WARNING: sketch self-consistency below {float(sk.consistency_warn):.2f} for: "
                  + ", ".join(f"{n} ({v:.2f})" for n, v in weak))

        # minor-class report
        tk = self.tr.tracker
        lines = [f"    {'class':16s} {'minor':>5s} {'seen(steps)':>11s} {'mean stale':>10s} {'max stale':>9s} {'queue pushes':>12s} {'queue':>6s}"]
        for c in range(self.C):
            if not (self.valid[c] or self.minor[c]):
                continue
            lines.append(f"    {self.classes[c]:16s} {'yes' if self.minor[c] else '-':>5s} {int(tk.times_seen[c]):11d} "
                         f"{float(tk.sum_stale[c]) / max(tk.steps, 1):10.1f} {int(tk.max_stale[c]):9d} "
                         f"{int(self.pushes[c]):12d} {len(self.queues.get(c, ())):6d}")
        print(f"[dtrr] T{t} end: per-class activity\n" + "\n".join(lines))
        for c in torch.nonzero(self.minor).flatten().tolist():
            if self.pushes[c] == 0:
                why = ("it never appeared in a training batch (times seen 0)" if tk.times_seen[c] == 0 else
                       f"it appeared in {int(tk.times_seen[c])} steps but was never queued (bug)")
                raise DTRRError(f"minor class '{self.classes[c]}' present in T{t} has 0 queue pushes: {why}")

        # logs + plots
        self._write_curves(t)
        corr = None
        if len(self.corr_pairs) >= 3:
            e, f = np.array(self.corr_pairs).T
            if e.std() > 0 and f.std() > 0:
                corr = float(np.corrcoef(e, f)[0, 1])
        replay_frac = self.replay_steps / max(self.task_step, 1)
        compute_frac = self.replay_time / max(self.step_time, 1e-9)
        eps_vals = [r["eps_hat"] for r in self.curve_rows]
        print(f"[dtrr] T{t} summary: eps_hat {eps_vals[0]:.4f} -> {eps_vals[-1]:.4f} over {len(eps_vals)} probes; "
              f"replay on {self.replay_steps}/{self.task_step} steps ({100 * replay_frac:.1f}%), {self.replay_px} pseudo px, "
              f"{self.conflicts} conflicts projected, replay time {100 * compute_frac:.1f}% of step time; "
              f"lam max " + (", ".join(f"{self.classes[c]} {float(self.lam_max_seen[c]):.2f}"
                                      for c in torch.nonzero(self.lam_max_seen > 0).flatten().tolist()) or "0 (all classes)")
              + (f"; corr(eps_hat, prev-task val drop) {corr:+.2f} over {len(self.corr_pairs)} probes" if corr is not None else ""))
        self.task_report.update(
            eps_curve=eps_vals, lam_max={self.classes[c]: float(v) for c, v in enumerate(self.lam_max_seen) if v > 0},
            replay_steps=self.replay_steps, replay_px=self.replay_px, replay_frac=replay_frac,
            replay_compute_frac=compute_frac, conflicts=self.conflicts, corr_eps_forgetting=corr,
            pushes={self.classes[c]: int(self.pushes[c]) for c in torch.nonzero(self.minor).flatten().tolist()},
            memory_file=self.mem_path, audit_file=self.audit_path, reload_ok=True, sketched=[self.classes[c] for c in seen_now],
            controller_updates=self.ctrl.updates, eps_updates=self.ctrl.eps_updates,
            last_F_hat_max=max((r["F_hat"] for r in self.ctrl_rows), default=0.0),
            rho=float(self.ctrl.rho), steps=self.task_step)
        self.report["tasks"].append(self.task_report)
        dump_json(self.report, os.path.join(self.out_dir, "dtrr_report.json"))
        self.probe = None                                # current-task probe images are discarded

    def _write_curves(self, t):
        keys = sorted({k for r in self.curve_rows for k in r}, key=lambda k: (not k in ("step", "eps_hat", "delta_hat"), k))
        path = os.path.join(self.out_dir, f"drift_curve_T{t}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.curve_rows)
        if self.ctrl_rows:
            with open(os.path.join(self.out_dir, f"controller_T{t}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.ctrl_rows[0]))
                w.writeheader()
                w.writerows(self.ctrl_rows)
        try:
            from .utils import setup_matplotlib
            plt = setup_matplotlib()
            st = [r["step"] for r in self.curve_rows]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(st, [r["eps_hat"] for r in self.curve_rows], "o-", label="eps_hat (probe z drift)")
            ax.plot(st, [r["delta_hat"] for r in self.curve_rows], "s--", label="delta_hat (max prototype shift)")
            ax.set_xlabel("step in task")
            ax.set_ylabel("L2 distance (normalised)")
            ax2 = ax.twinx()
            ax2.plot(st, [r["lam_sum"] for r in self.curve_rows], color="tab:red", alpha=0.6, label="sum lam")
            ax2.set_ylabel("sum lam", color="tab:red")
            ax.legend(loc="upper left", fontsize=8)
            ax.set_title(f"DTRR drift monitor, task {t}")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, f"drift_curve_T{t}.png"), dpi=110)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001  - a plotting failure must not kill training
            print(f"[dtrr] drift plot skipped: {e}")

    def final_checks(self):
        if self.ctrl.eps_updates == 0 and self.dc.get("mode", "adaptive") == "adaptive":
            raise DTRRError("eps_hat was never consumed by the controller (0 controller updates with eps_hat)")
        self.report["eps_consumed_updates"] = self.ctrl.eps_updates
        dump_json(self.report, os.path.join(self.out_dir, "dtrr_report.json"))
