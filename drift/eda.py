"""Exploratory data analysis for any dataset config.

Writes to <out_dir>:
  metadata.json, class_weights.json, eda_report.md and the plots listed in
  the report (class distribution, presence, per-domain / per-task heatmaps,
  co-occurrence, spatial priors, resolution, RGB / brightness, samples).
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data.datasets import SegDataset, build_splits, build_tasks
from .utils import SERIES, INK2, colorize, dump_json, get_palette, setup_matplotlib


class _LabelStats(Dataset):
    """Per-sample label + (optionally) image statistics, computed in workers."""

    def __init__(self, ds: SegDataset, spatial_res, image_idx: set[int]):
        self.ds, self.spatial_res, self.image_idx = ds, tuple(spatial_res), image_idx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        C, ign = self.ds.num_classes, self.ds.ignore_index
        lbl = self.ds.read_label(i)
        h, w = lbl.shape
        valid = lbl != ign
        counts = np.bincount(lbl[valid].ravel(), minlength=C)[:C].astype(np.int64)
        sh, sw = self.spatial_res
        small = np.array(Image.fromarray(lbl.astype(np.int32)).resize((sw, sh), Image.NEAREST))
        spatial = np.zeros((C, sh, sw), dtype=np.float32)
        for c in np.nonzero(counts)[0]:
            spatial[c] = small == c
        out = {"counts": counts, "h": h, "w": w, "ignore": int((~valid).sum()), "spatial": spatial,
               "domain": self.ds.samples[i].domain, "img": None}
        if i in self.image_idx:
            im = np.asarray(self.ds.read_image(i), dtype=np.float32) / 255.0
            gray = im.mean(axis=2)
            out["img"] = {
                "mean": im.reshape(-1, 3).mean(0), "sq": (im.reshape(-1, 3) ** 2).mean(0),
                "hist": np.stack([np.histogram(im[..., k], bins=64, range=(0, 1))[0] for k in range(3)]),
                "brightness": float(gray.mean()), "contrast": float(gray.std()),
                "iw": im.shape[1], "ih": im.shape[0],
            }
        return out


def _collate(batch):
    return batch


def _scan(ds: SegDataset, ecfg, num_workers: int, seed: int):
    rng = np.random.RandomState(seed)
    n = len(ds)
    k = min(n, int(ecfg.image_stats_max or n))
    img_idx = set(rng.choice(n, k, replace=False).tolist())
    loader = DataLoader(_LabelStats(ds, ecfg.spatial_res, img_idx), batch_size=8,
                        num_workers=num_workers, collate_fn=_collate)
    rows = []
    try:
        from tqdm import tqdm
        loader = tqdm(loader, desc="  scanning", leave=False)
    except ImportError:
        pass
    for batch in loader:
        rows.extend(batch)
    return rows


def run_eda(cfg, out_dir: str, splits_to_scan=("train", "val", "test"), num_workers: int = 4) -> dict:
    plt = setup_matplotlib()
    os.makedirs(out_dir, exist_ok=True)
    dcfg, ecfg = cfg.dataset, cfg.eda
    classes = list(dcfg.classes)
    C = len(classes)
    pal = get_palette(dcfg)
    pal_f = pal / 255.0
    splits = build_splits(cfg)
    tasks = build_tasks(cfg, splits["train"]) if "train" in splits else []

    meta: dict = {"dataset": dcfg.name, "root": dcfg.root, "num_classes": C, "classes": classes,
                  "ignore_index": dcfg.ignore_index, "splits": {}, "tasks": tasks}
    per_split = {}
    for name in splits_to_scan:
        if name not in splits:
            continue
        samples = splits[name]
        if ecfg.get("max_images"):
            samples = samples[: int(ecfg.max_images)]
        print(f"[eda] scanning {name}: {len(samples)} samples")
        ds = SegDataset(samples, dcfg, transform=None)
        per_split[name] = _scan(ds, ecfg, num_workers, cfg.experiment.seed)

    figs: list[tuple[str, str]] = []

    def save(fig, fname, title):
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=cfg.viz.dpi, bbox_inches="tight")
        plt.close(fig)
        figs.append((title, fname))

    # ---------------------------------------------------------------- per split numbers
    stats = {}
    for name, rows in per_split.items():
        counts = np.stack([r["counts"] for r in rows])                      # [N, C]
        total_px = np.array([r["h"] * r["w"] for r in rows])
        present = counts > 0
        area_frac = counts / np.maximum(total_px[:, None], 1)
        dom = np.array([r["domain"] for r in rows])
        img_rows = [r["img"] for r in rows if r["img"] is not None]
        s = {
            "num_images": len(rows),
            "domains": {d: int((dom == d).sum()) for d in sorted(set(dom))},
            "class_pixels": counts.sum(0),
            "class_pixel_fraction": counts.sum(0) / max(counts.sum(), 1),
            "class_image_presence": present.mean(0),
            "classes_per_image": present.sum(1),
            "ignore_fraction": np.array([r["ignore"] for r in rows]) / np.maximum(total_px, 1),
            "label_sizes": sorted({(int(r["h"]), int(r["w"])) for r in rows}),
            "area_frac": area_frac, "present": present, "dom": dom, "counts": counts,
            "spatial": np.sum([r["spatial"] for r in rows], axis=0) / len(rows),
        }
        if img_rows:
            mean = np.mean([r["mean"] for r in img_rows], 0)
            sq = np.mean([r["sq"] for r in img_rows], 0)
            s["rgb_mean"] = mean
            s["rgb_std"] = np.sqrt(np.maximum(sq - mean ** 2, 0))
            s["rgb_hist"] = np.sum([r["hist"] for r in img_rows], 0)
            s["brightness"] = [(r["brightness"]) for r in img_rows]
            s["brightness_dom"] = [d for r, d in zip(rows, dom) if r["img"] is not None]
            s["contrast"] = [r["contrast"] for r in img_rows]
            s["image_sizes"] = [(r["iw"], r["ih"]) for r in img_rows]
        stats[name] = s
        never = [classes[c] for c in range(C) if counts[:, c].sum() == 0]
        nz = s["class_pixels"][s["class_pixels"] > 0]
        meta["splits"][name] = {
            "num_images": s["num_images"], "num_domains": len(s["domains"]), "domains": s["domains"],
            "label_resolutions": s["label_sizes"][:20],
            "class_pixel_fraction": dict(zip(classes, s["class_pixel_fraction"].round(6).tolist())),
            "class_image_presence": dict(zip(classes, s["class_image_presence"].round(4).tolist())),
            "classes_never_seen": never,
            "mean_classes_per_image": float(s["classes_per_image"].mean()),
            "mean_ignore_fraction": float(s["ignore_fraction"].mean()),
            "imbalance_ratio_max_over_min": float(nz.max() / nz.min()) if len(nz) else None,
        }
        if "rgb_mean" in s:
            meta["splits"][name]["rgb_mean"] = s["rgb_mean"].round(4).tolist()
            meta["splits"][name]["rgb_std"] = s["rgb_std"].round(4).tolist()
            meta["splits"][name]["mean_brightness"] = float(np.mean(s["brightness"]))

    ref = "train" if "train" in stats else next(iter(stats))
    rs = stats[ref]

    # ---------------------------------------------------------------- class weights
    freq = rs["class_pixels"] / max(rs["class_pixels"].sum(), 1)
    med = np.median(freq[freq > 0]) if (freq > 0).any() else 1.0
    w_median = np.where(freq > 0, med / np.maximum(freq, 1e-12), 0.0)
    w_log = 1.0 / np.log(1.02 + freq)
    weights = {"classes": classes, "median_frequency": w_median.round(4).tolist(),
               "inverse_log": (w_log / w_log.mean()).round(4).tolist(), "source_split": ref}
    dump_json(weights, os.path.join(out_dir, "class_weights.json"))

    # ---------------------------------------------------------------- split / domain sizes
    fig, ax = plt.subplots(figsize=(5, 2.6))
    names = list(stats)
    vals = [stats[n]["num_images"] for n in names]
    ax.bar(names, vals, color=SERIES[0], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha="center", va="bottom", color=INK2, fontsize=8)
    ax.set_ylabel("images"); ax.set_title("Images per split"); ax.grid(axis="x", visible=False)
    save(fig, "split_sizes.png", "Images per split")

    all_doms = sorted({d for s in stats.values() for d in s["domains"]})
    if len(all_doms) > 1:
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(all_doms))))
        y = np.arange(len(all_doms))
        left = np.zeros(len(all_doms))
        for k, n in enumerate(names):
            v = np.array([stats[n]["domains"].get(d, 0) for d in all_doms])
            ax.barh(y, v, left=left, color=SERIES[k % len(SERIES)], label=n, height=0.7,
                    edgecolor="#fcfcfb", linewidth=1)
            left += v
        ax.set_yticks(y, all_doms); ax.invert_yaxis(); ax.set_xlabel("images")
        ax.set_title("Images per domain"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
        save(fig, "domain_counts.png", "Images per domain and split")

    # ---------------------------------------------------------------- class distribution
    def class_bar(values, title, xlabel, fname, log=False, fmt="{:.2%}"):
        order = np.argsort(-values)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.26 * C)))
        ax.barh(np.arange(C), values[order], color=pal_f[order], edgecolor="#52514e", linewidth=0.3, height=0.75)
        ax.set_yticks(np.arange(C), [classes[i] for i in order]); ax.invert_yaxis()
        if log:
            ax.set_xscale("log")
        for i, v in enumerate(values[order]):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=7, color=INK2)
        ax.set_xlabel(xlabel); ax.set_title(title); ax.grid(axis="y", visible=False)
        save(fig, fname, title)

    class_bar(np.maximum(rs["class_pixel_fraction"], 1e-9), f"Class pixel share ({ref}, log scale)",
              "fraction of labelled pixels", "class_pixel_distribution.png", log=True)
    class_bar(rs["class_image_presence"], f"Images containing each class ({ref})",
              "fraction of images", "class_image_presence.png")

    # train vs other splits, side by side
    if len(stats) > 1:
        fig, ax = plt.subplots(figsize=(max(7, 0.45 * C), 3.4))
        x = np.arange(C)
        wbar = 0.8 / len(stats)
        for k, n in enumerate(names):
            ax.bar(x + (k - (len(stats) - 1) / 2) * wbar, np.maximum(stats[n]["class_pixel_fraction"], 1e-9),
                   width=wbar, color=SERIES[k], label=n)
        ax.set_yscale("log"); ax.set_xticks(x, classes, rotation=60, ha="right")
        ax.set_ylabel("pixel share (log)"); ax.set_title("Class pixel share by split"); ax.legend()
        ax.grid(axis="x", visible=False)
        save(fig, "class_distribution_by_split.png", "Class pixel share by split")

    # pixel-count histogram per class index (bar over class id, linear)
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * C), 3))
    ax.bar(np.arange(C), rs["class_pixels"] / 1e6, color=pal_f, edgecolor="#52514e", linewidth=0.3)
    ax.set_xticks(np.arange(C), [f"{i}" for i in range(C)])
    ax.set_xlabel("class index"); ax.set_ylabel("pixels (millions)")
    ax.set_title(f"Pixel count per class index ({ref})"); ax.grid(axis="x", visible=False)
    save(fig, "class_index_histogram.png", "Pixel count per class index")

    # area fraction when present
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * C), 3.4))
    data = [rs["area_frac"][rs["present"][:, c], c] for c in range(C)]
    data = [d if len(d) else np.array([0.0]) for d in data]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, col in zip(bp["boxes"], pal_f):
        patch.set_facecolor(col); patch.set_edgecolor("#52514e"); patch.set_linewidth(0.6)
    for med_line in bp["medians"]:
        med_line.set_color("#0b0b0b")
    ax.set_yscale("log"); ax.set_xticks(np.arange(1, C + 1), classes, rotation=60, ha="right")
    ax.set_ylabel("share of image (log)"); ax.set_title("Class area per image, when present")
    ax.grid(axis="x", visible=False)
    save(fig, "class_area_boxplot.png", "Class area per image when present")

    # classes per image + ignore fraction
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    cpi = rs["classes_per_image"]
    axes[0].hist(cpi, bins=np.arange(cpi.min(), cpi.max() + 2) - 0.5, color=SERIES[0], edgecolor="#fcfcfb")
    axes[0].set_xlabel("classes in image"); axes[0].set_ylabel("images"); axes[0].set_title("Classes per image")
    axes[1].hist(rs["ignore_fraction"], bins=30, color=SERIES[1], edgecolor="#fcfcfb")
    axes[1].set_xlabel("ignored pixel fraction"); axes[1].set_title("Ignore-label share per image")
    save(fig, "classes_per_image_and_ignore.png", "Classes per image and ignore-label share")

    # ---------------------------------------------------------------- domain x class
    def heat(matrix, rows, title, fname, mark_zero=True):
        fig, ax = plt.subplots(figsize=(max(7, 0.42 * C), max(2.2, 0.3 * len(rows) + 1.2)))
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks(np.arange(C), classes, rotation=60, ha="right"); ax.set_yticks(np.arange(len(rows)), rows)
        ax.grid(False)
        if mark_zero:
            zr, zc = np.nonzero(matrix == 0)
            ax.scatter(zc, zr, marker="x", s=14, color="#d03b3b", linewidths=1, label="absent")
            if len(zr):
                ax.legend(loc="upper right", bbox_to_anchor=(1, 1.18))
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        ax.set_title(title)
        save(fig, fname, title)

    doms = sorted(set(rs["dom"]))
    if len(doms) > 1:
        m = np.stack([rs["counts"][rs["dom"] == d].sum(0) for d in doms]).astype(np.float64)
        heat(m / np.maximum(m.sum(1, keepdims=True), 1), doms, f"Class pixel share per domain ({ref})",
             "domain_class_heatmap.png")
        pres = np.stack([rs["present"][rs["dom"] == d].mean(0) for d in doms])
        heat(pres, doms, f"Class presence per domain ({ref})", "domain_presence_heatmap.png")
        meta["absent_classes_per_domain"] = {d: [classes[c] for c in range(C) if m[i, c] == 0]
                                             for i, d in enumerate(doms)}
    if len(tasks) > 1 and ref == "train":
        tm = []
        for t in tasks:
            sel = np.isin(rs["dom"], t)
            tm.append(rs["counts"][sel].sum(0))
        tm = np.stack(tm).astype(np.float64)
        labels = [f"T{i}: " + ",".join(t)[:28] for i, t in enumerate(tasks)]
        heat(tm / np.maximum(tm.sum(1, keepdims=True), 1), labels, "Class pixel share per continual task",
             "task_class_heatmap.png")
        meta["task_class_sets"] = {f"T{i}": [classes[c] for c in range(C) if tm[i, c] > 0] for i in range(len(tasks))}

    # ---------------------------------------------------------------- co-occurrence
    P = rs["present"].astype(np.float64)
    co = (P.T @ P) / np.maximum(P.sum(0)[:, None], 1)                       # P(j | i)
    fig, ax = plt.subplots(figsize=(max(6, 0.38 * C), max(5, 0.36 * C)))
    im = ax.imshow(co, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(C), classes, rotation=70, ha="right"); ax.set_yticks(np.arange(C), classes)
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Co-occurrence  P(column present | row present)")
    save(fig, "class_cooccurrence.png", "Class co-occurrence")

    # ---------------------------------------------------------------- spatial priors
    cols = min(5, C)
    nrows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.4, nrows * 1.5))
    for c, ax in enumerate(np.array(axes).reshape(-1)):
        ax.axis("off")
        if c < C:
            ax.imshow(rs["spatial"][c], cmap="Blues", vmin=0, vmax=max(rs["spatial"][c].max(), 1e-6))
            ax.set_title(classes[c], fontsize=8)
    fig.suptitle(f"Where each class appears ({ref})", fontsize=11)
    save(fig, "class_spatial_priors.png", "Spatial class priors")

    # ---------------------------------------------------------------- image statistics
    if "rgb_hist" in rs:
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
        bins = np.linspace(0, 1, 65)
        centers = (bins[:-1] + bins[1:]) / 2
        for k, (nm, col) in enumerate(zip("RGB", ["#d03b3b", "#0ca30c", "#2a78d6"])):
            h = rs["rgb_hist"][k] / rs["rgb_hist"][k].sum()
            axes[0].plot(centers, h, color=col, label=nm)
        axes[0].set_xlabel("intensity"); axes[0].set_ylabel("density"); axes[0].legend()
        axes[0].set_title("RGB intensity distribution")
        sizes = np.array(rs["image_sizes"])
        uniq, cnt = np.unique(sizes, axis=0, return_counts=True)
        axes[1].scatter(uniq[:, 0], uniq[:, 1], s=np.clip(cnt * 8, 10, 300), color=SERIES[0], alpha=0.8,
                        edgecolor="#fcfcfb", linewidth=1)
        for (wv, hv), n in zip(uniq[:8], cnt[:8]):
            axes[1].annotate(f"{wv}x{hv} ({n})", (wv, hv), fontsize=7, color=INK2,
                             xytext=(4, 4), textcoords="offset points")
        axes[1].set_xlabel("width"); axes[1].set_ylabel("height"); axes[1].set_title("Image resolutions")
        save(fig, "image_rgb_and_resolution.png", "RGB intensity and image resolution")

        bdoms = sorted(set(rs["brightness_dom"]))
        if len(bdoms) > 1:
            fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(bdoms)), 3))
            grouped = defaultdict(list)
            for b, d in zip(rs["brightness"], rs["brightness_dom"]):
                grouped[d].append(b)
            bp = ax.boxplot([grouped[d] for d in bdoms], patch_artist=True, showfliers=False, widths=0.6)
            for patch in bp["boxes"]:
                patch.set_facecolor("#86b6ef"); patch.set_edgecolor("#184f95")
            for med_line in bp["medians"]:
                med_line.set_color("#0b0b0b")
            ax.set_xticks(np.arange(1, len(bdoms) + 1), bdoms, rotation=60, ha="right")
            ax.set_ylabel("mean brightness"); ax.set_title("Brightness per domain (appearance shift)")
            ax.grid(axis="x", visible=False)
            save(fig, "brightness_by_domain.png", "Brightness per domain")

    # ---------------------------------------------------------------- sample grid
    n_show = int(ecfg.num_sample_grid)
    if n_show > 0 and "train" in splits:
        ds = SegDataset(splits["train"], dcfg, transform=None)
        idx = np.random.RandomState(0).choice(len(ds), min(n_show, len(ds)), replace=False)
        fig, axes = plt.subplots(len(idx), 2, figsize=(8, 2.1 * len(idx)))
        axes = np.array(axes).reshape(len(idx), 2)
        for r, i in enumerate(idx):
            im = np.asarray(ds.read_image(int(i)))
            lb = colorize(ds.read_label(int(i)), pal, dcfg.ignore_index)
            axes[r, 0].imshow(im); axes[r, 1].imshow((0.45 * im + 0.55 * lb).astype(np.uint8))
            axes[r, 0].set_title(ds.samples[int(i)].domain, fontsize=8)
            for a in axes[r]:
                a.axis("off")
        save(fig, "sample_grid.png", "Sample images with label overlay")

    # legend for palette
    fig, ax = plt.subplots(figsize=(7, 0.32 * C + 0.4))
    for c in range(C):
        ax.add_patch(__import__("matplotlib").patches.Rectangle((0, C - c - 1), 0.6, 0.8, color=pal_f[c]))
        ax.text(0.8, C - c - 0.6, f"{c:2d}  {classes[c]}", va="center", fontsize=8)
    ax.set_xlim(0, 6); ax.set_ylim(0, C); ax.axis("off"); ax.set_title("Class index and color")
    save(fig, "class_legend.png", "Class index and color legend")

    dump_json(meta, os.path.join(out_dir, "metadata.json"))
    _write_report(meta, figs, out_dir, weights)
    print(f"[eda] wrote {len(figs)} figures + metadata.json + eda_report.md to {out_dir}")
    return meta


def _write_report(meta, figs, out_dir, weights):
    lines = [f"# EDA - {meta['dataset']}", "", f"- root: `{meta['root']}`",
             f"- classes: {meta['num_classes']}  (ignore index {meta['ignore_index']})", ""]
    lines += ["## Splits", "", "| split | images | domains | mean classes/img | mean ignore | imbalance max/min | never seen |",
              "|---|---|---|---|---|---|---|"]
    for n, s in meta["splits"].items():
        ir = s["imbalance_ratio_max_over_min"]
        lines.append(f"| {n} | {s['num_images']} | {s['num_domains']} | {s['mean_classes_per_image']:.2f} | "
                     f"{s['mean_ignore_fraction']:.3f} | {ir:.1f} | {', '.join(s['classes_never_seen']) or '-'} |"
                     if ir else f"| {n} | {s['num_images']} | {s['num_domains']} | - | - | - | - |")
    for n, s in meta["splits"].items():
        if "rgb_mean" in s:
            lines += ["", f"Normalization suggestion from `{n}`: mean={s['rgb_mean']}, std={s['rgb_std']}"]
            break
    lines += ["", "## Class table (train)", "", "| idx | class | pixel share | image presence | median-freq weight |",
              "|---|---|---|---|---|"]
    ref = "train" if "train" in meta["splits"] else next(iter(meta["splits"]))
    s = meta["splits"][ref]
    for i, c in enumerate(meta["classes"]):
        lines.append(f"| {i} | {c} | {s['class_pixel_fraction'][c]:.4%} | {s['class_image_presence'][c]:.1%} | "
                     f"{weights['median_frequency'][i]:.2f} |")
    if meta.get("tasks"):
        lines += ["", "## Continual tasks", ""]
        for i, t in enumerate(meta["tasks"]):
            lines.append(f"- T{i}: {', '.join(t)}")
    if meta.get("absent_classes_per_domain"):
        lines += ["", "## Classes absent per domain", ""]
        for d, a in meta["absent_classes_per_domain"].items():
            lines.append(f"- {d}: {', '.join(a) or '-'}")
    lines += ["", "## Figures", ""]
    for title, f in figs:
        lines += [f"### {title}", f"![{title}]({f})", ""]
    with open(os.path.join(out_dir, "eda_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
