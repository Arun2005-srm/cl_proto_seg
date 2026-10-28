"""Segmentation and continual-learning metrics.

SegMetrics accumulates a confusion matrix (pixel acc, per-class IoU, mIoU,
frequency-weighted IoU, per-class F1 / precision / recall, mean Dice) and
Boundary IoU (Cheng et al., 2021) per class on GPU.

Note: pixel-level Dice and F1 are the same formula, 2TP / (2TP + FP + FN).
"Dice" is reported as the class mean, "F1" per class.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _boundary(mask: torch.Tensor, d: int) -> torch.Tensor:
    """Inner boundary band of width d of a [B,H,W] bool mask."""
    m = mask.float().unsqueeze(1)
    eroded = 1 - F.max_pool2d(1 - m, kernel_size=2 * d + 1, stride=1, padding=d)
    return (m - eroded).squeeze(1) > 0.5


class SegMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255, boundary_dilation: float = 0.02,
                 device="cpu", compute_boundary: bool = True, keep_per_image: bool = False):
        self.C, self.ign = num_classes, ignore_index
        self.keep_per_image = keep_per_image
        self.dil = boundary_dilation
        self.device = device
        self.compute_boundary = compute_boundary
        self.reset()

    def reset(self):
        self.conf = torch.zeros(self.C, self.C, dtype=torch.float64, device=self.device)
        self.b_inter = torch.zeros(self.C, dtype=torch.float64, device=self.device)
        self.b_union = torch.zeros(self.C, dtype=torch.float64, device=self.device)
        self.unknown_fn = torch.zeros(self.C, dtype=torch.float64, device=self.device)
        self.per_image: list[np.ndarray] = []      # [C, C+1] per image (last column: "unknown" misses)

    def _d(self, h, w) -> int:
        if isinstance(self.dil, int) or self.dil >= 1:
            return max(1, int(self.dil))
        return max(1, int(round(self.dil * (h ** 2 + w ** 2) ** 0.5)))

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: [B,H,W] long on the same device."""
        pred, target = pred.to(self.device), target.to(self.device)
        valid = target != self.ign
        t, p = target[valid], pred[valid]
        known = (p >= 0) & (p < self.C)        # "unknown" predictions (ignore_index) count as misses
        idx = t[known] * self.C + p[known]
        self.conf += torch.bincount(idx, minlength=self.C ** 2).reshape(self.C, self.C).double()
        if (~known).any():
            self.unknown_fn += torch.bincount(t[~known], minlength=self.C).double()
        if self.keep_per_image:
            C = self.C
            for b in range(target.shape[0]):
                tb, pb_ = target[b][valid[b]], pred[b][valid[b]]
                kb = (pb_ >= 0) & (pb_ < C)
                m = torch.zeros(C, C + 1, dtype=torch.int64, device=self.device)
                m[:, :C] = torch.bincount(tb[kb] * C + pb_[kb], minlength=C * C).reshape(C, C)
                m[:, C] = torch.bincount(tb[~kb], minlength=C)
                self.per_image.append(m.cpu().numpy())
        if self.compute_boundary:
            d = self._d(*target.shape[-2:])
            for c in torch.unique(torch.cat([t, p[known]])).tolist():
                gt_c = (target == c) & valid
                pr_c = (pred == c) & valid
                gb, pb = _boundary(gt_c, d) & gt_c, _boundary(pr_c, d) & pr_c
                self.b_inter[c] += (gb & pb).sum()
                self.b_union[c] += (gb | pb).sum()

    def compute(self) -> dict:
        conf = self.conf.cpu().numpy()
        fn_extra = self.unknown_fn.cpu().numpy()
        tp = np.diag(conf)
        gt = conf.sum(1) + fn_extra
        pr = conf.sum(0)
        fp, fn = pr - tp, gt - tp
        present = gt > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (tp + fp + fn)
            f1 = 2 * tp / (2 * tp + fp + fn)
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            biou = (self.b_inter / self.b_union).cpu().numpy()
        nanmean = lambda a: float(np.nanmean(np.where(present, a, np.nan))) if present.any() else 0.0  # noqa: E731
        freq = gt / max(gt.sum(), 1)
        return {
            "pixel_acc": float(tp.sum() / max(gt.sum(), 1)),
            "mean_acc": nanmean(rec),
            "mIoU": nanmean(iou),
            "fwIoU": float(np.nansum(freq * np.nan_to_num(iou))),
            "BIoU": nanmean(biou) if self.compute_boundary else float("nan"),
            "Dice": nanmean(f1),
            "IoU_per_class": np.where(present, iou, np.nan).tolist(),
            "F1_per_class": np.where(present, f1, np.nan).tolist(),
            "BIoU_per_class": np.where(present, biou, np.nan).tolist(),
            "precision_per_class": np.where(present, prec, np.nan).tolist(),
            "recall_per_class": np.where(present, rec, np.nan).tolist(),
            "support": gt.astype(np.int64).tolist(),
            "confusion": conf,
            "unknown_fn": fn_extra.tolist(),
        }


def format_class_table(res: dict, classes, cols=("IoU", "F1", "BIoU")) -> str:
    key = {"IoU": "IoU_per_class", "F1": "F1_per_class", "BIoU": "BIoU_per_class",
           "Prec": "precision_per_class", "Rec": "recall_per_class"}
    w = max(12, max(len(c) for c in classes) + 1)
    head = f"    {'class':{w}s}" + "".join(f"{c:>8s}" for c in cols)
    lines = [head]
    for i, name in enumerate(classes):
        vals = []
        for c in cols:
            v = res[key[c]][i]
            vals.append(f"{'   -   ':>8s}" if v is None or np.isnan(v) else f"{v * 100:8.2f}")
        lines.append(f"    {name:{w}s}" + "".join(vals))
    return "\n".join(lines)


def miou_from_conf(conf: np.ndarray, unknown_fn=None, classes: np.ndarray | None = None) -> float:
    """mIoU from a [C, C] confusion matrix, over classes present in GT (optionally a fixed subset)."""
    conf = np.asarray(conf, float)
    tp = np.diag(conf)
    gt = conf.sum(1) + (0 if unknown_fn is None else np.asarray(unknown_fn, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (gt + conf.sum(0) - tp)
    sel = gt > 0
    if classes is not None:
        sel &= np.asarray(classes, bool)
    return float(np.nanmean(iou[sel])) if sel.any() else float("nan")


def bootstrap_miou(per_image: list[np.ndarray], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Image-level bootstrap of mIoU: resample test images with replacement."""
    if not per_image:
        return {}
    P = np.stack(per_image)                               # [N, C, C+1]
    rng = np.random.default_rng(seed)
    C = P.shape[1]
    vals = []
    for _ in range(n):
        m = P[rng.integers(0, len(P), len(P))].sum(0)
        vals.append(miou_from_conf(m[:, :C], m[:, C]))
    vals = np.array(vals)
    return {"mean": float(vals.mean()), "std": float(vals.std()), "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)), "n_boot": n, "n_images": len(P)}


# ----------------------------------------------------------------------- continual
def class_forgetting(Rc: np.ndarray) -> dict:
    """Rc[i, c] = IoU of class c on the pooled val of all tasks seen so far, after task i.
    Forgetting per class uses one fixed class set, so it is not confounded by which
    classes each task happens to contain (unlike per-task mIoU)."""
    T = Rc.shape[0]
    if T < 2:
        return {"avg_class_forgetting": 0.0, "class_forgetting": [0.0] * Rc.shape[1]}
    best = np.nanmax(Rc[:-1], axis=0)
    f = best - Rc[-1]
    return {"avg_class_forgetting": float(np.nanmean(f)) if np.isfinite(f).any() else 0.0,
            "class_forgetting": [None if np.isnan(x) else float(x) for x in f]}


def continual_metrics(R: np.ndarray) -> dict:
    """R[i, j] = metric on task j after training task i (NaN if not evaluated)."""
    T = R.shape[0]
    final = R[T - 1]
    out = {"avg_final": float(np.nanmean(final)), "avg_incremental": float(np.nanmean(
        [np.nanmean(R[i, : i + 1]) for i in range(T)]))}
    if T > 1:
        bwt = [R[T - 1, j] - R[j, j] for j in range(T - 1)]
        forget = [np.nanmax(R[j:T - 1, j]) - R[T - 1, j] for j in range(T - 1)]
        out["BWT"] = float(np.nanmean(bwt))
        out["avg_forgetting"] = float(np.nanmean(forget))
        out["forgetting_per_task"] = [float(f) for f in forget]
        if not np.all(np.isnan(np.diag(R, 1))):
            out["FWT_zero_shot"] = float(np.nanmean(np.diag(R, 1)))   # next task before training on it
    else:
        out.update(BWT=0.0, avg_forgetting=0.0, forgetting_per_task=[])
    return out
