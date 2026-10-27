"""Small shared helpers: seeding, palettes, colorizing, plot style, IO."""
from __future__ import annotations

import colorsys
import json
import os
import random
import time

import numpy as np
import torch

# Fixed categorical order for series that are not dataset classes (train/val, tasks, ...)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_palette(dcfg) -> np.ndarray:
    """[C, 3] uint8 class colors; generated when the config gives none."""
    n = len(dcfg.classes)
    if dcfg.get("palette"):
        pal = np.array(dcfg.palette, dtype=np.uint8)
        if len(pal) >= n:
            return pal[:n]
    cols = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95 if i % 2 == 0 else 0.7)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(cols, dtype=np.uint8)


def colorize(mask: np.ndarray, palette: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask != ignore_index) & (mask < len(palette)) & (mask >= 0)
    out[valid] = palette[mask[valid]]
    return out


def denormalize(img: torch.Tensor, mean, std) -> np.ndarray:
    m = torch.tensor(mean, device=img.device).view(3, 1, 1)
    s = torch.tensor(std, device=img.device).view(3, 1, 1)
    x = (img * s + m).clamp(0, 1)
    return (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titleweight": "semibold", "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
        "font.size": 9,
    })
    return plt


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def make_run_dir(cfg) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(cfg.experiment.output_dir, cfg.experiment.name, stamp)
    os.makedirs(d, exist_ok=True)
    return d


class Tee:
    """Mirror stdout into a log file."""

    def __init__(self, path: str):
        import sys

        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
