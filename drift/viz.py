"""Plots: train/val curves, metric curves, forgetting curves, accuracy matrix,
t-SNE of pixel embeddings, per-class test bars, confusion matrix, and
image / ground truth / prediction panels."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from .utils import INK2, SERIES, colorize, denormalize, get_palette, setup_matplotlib


def _task_colors(n):
    if n <= len(SERIES):
        return SERIES[:n]
    import matplotlib.cm as cm
    return [cm.Blues(0.35 + 0.6 * i / max(n - 1, 1)) for i in range(n)]


def _task_bounds(ax, history, label=True):
    prev = None
    for r in history:
        if prev is not None and r["task"] != prev:
            ax.axvline(r["global_epoch"] - 0.5, color="#b8b7b1", linestyle="--", linewidth=1)
            if label:
                ax.text(r["global_epoch"] - 0.4, 1.0, f"T{r['task']}", transform=ax.get_xaxis_transform(),
                        fontsize=7, color=INK2, va="top")
        prev = r["task"]


def _line(ax, x, y, color, label, marker=True):
    ax.plot(x, y, color=color, label=label, marker="o" if marker and len(x) < 40 else None, markersize=4,
            markeredgecolor="#fcfcfb", markeredgewidth=1)


def plot_training_curves(history, out_dir, dpi=120):
    plt = setup_matplotlib()
    os.makedirs(out_dir, exist_ok=True)
    if not history:
        return []
    x = [r["global_epoch"] for r in history]
    files = []

    for key, title, fname in (("loss", "Loss", "curve_loss.png"), ("acc", "Pixel accuracy", "curve_accuracy.png")):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        tr = [r.get(f"train_{key}") for r in history]
        va = [r.get(f"val_{key}", np.nan) for r in history]
        scale = 100 if key == "acc" else 1
        _line(ax, x, np.array(tr, float) * scale, SERIES[0], "train")
        _line(ax, x, np.array(va, float) * scale, SERIES[1], "val")
        _task_bounds(ax, history)
        ax.set_xlabel("epoch (across tasks)"); ax.set_ylabel(title + (" (%)" if key == "acc" else ""))
        ax.set_title(f"Train / val {title.lower()}"); ax.legend(loc="best")
        fig.savefig(os.path.join(out_dir, fname), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append(fname)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, (key, lab) in enumerate((("mIoU", "mIoU"), ("IoU", "IoU (freq-weighted)"), ("BIoU", "Boundary IoU"),
                                    ("Dice", "Dice"))):
        _line(ax, x, 100 * np.array([r.get(key, np.nan) for r in history], float), SERIES[k], lab)
    _task_bounds(ax, history)
    ax.set_xlabel("epoch (across tasks)"); ax.set_ylabel("%"); ax.set_title("Validation IoU / Dice / mIoU / BIoU")
    ax.legend(loc="best", ncol=2)
    fig.savefig(os.path.join(out_dir, "curve_metrics.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    files.append("curve_metrics.png")

    rows = [r for r in history if "IoU_per_class" in r]
    if rows:
        M = 100 * np.array([[np.nan if v is None else v for v in r["IoU_per_class"]] for r in rows], float)
        fig, ax = plt.subplots(figsize=(max(7, 0.3 * len(rows) + 3), max(3, 0.25 * M.shape[1] + 1)))
        im = ax.imshow(M.T, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_xlabel("epoch (across tasks)"); ax.set_ylabel("class index"); ax.grid(False)
        ax.set_xticks(np.arange(len(rows)), [r["global_epoch"] for r in rows], fontsize=6)
        ax.set_title("Per-class validation IoU over training (blank = class absent from val)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="IoU %")
        fig.savefig(os.path.join(out_dir, "curve_per_class_iou.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append("curve_per_class_iou.png")
    return files


def plot_forgetting(R, tasks, seen_curve, test_curve, history, out_dir, dpi=120):
    plt = setup_matplotlib()
    os.makedirs(out_dir, exist_ok=True)
    T = len(tasks)
    cols = _task_colors(T)
    files = []
    names = [f"T{j}" for j in range(T)]

    # fine-grained: every seen task, every epoch
    if seen_curve:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for j in range(T):
            pts = [(r["global_epoch"], r[str(j)]) for r in seen_curve if str(j) in r]
            if pts:
                xs, ys = zip(*pts)
                _line(ax, xs, 100 * np.array(ys), cols[j], f"T{j}: {','.join(tasks[j])[:22]}")
        _task_bounds(ax, history)
        ax.set_xlabel("epoch (across tasks)"); ax.set_ylabel("val mIoU (%)")
        ax.set_title("Forgetting curve - val mIoU of every seen task")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.savefig(os.path.join(out_dir, "forgetting_curve_epochs.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append("forgetting_curve_epochs.png")

    done = int((~np.isnan(np.diag(R))).sum())
    if done:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
        ax = axes[0]
        for j in range(done):
            xs = np.arange(j, done)
            _line(ax, xs, 100 * R[j:done, j], cols[j], names[j])
        ax.set_xticks(np.arange(done), [f"after T{i}" for i in range(done)], rotation=0)
        ax.set_ylabel("val mIoU (%)"); ax.set_title("Per-task mIoU after each training stage")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1 if done <= 10 else 2)
        ax = axes[1]
        if done > 1:
            avg_f = [0.0] + [100 * np.mean([np.nanmax(R[j:i, j]) - R[i, j] for j in range(i)]) for i in range(1, done)]
            avg_acc = [100 * np.nanmean(R[i, : i + 1]) for i in range(done)]
            _line(ax, np.arange(done), avg_acc, SERIES[0], "avg mIoU over seen tasks")
            _line(ax, np.arange(done), avg_f, SERIES[7], "avg forgetting")
            ax.set_xticks(np.arange(done), [f"T{i}" for i in range(done)])
            ax.legend(loc="best")
        ax.set_ylabel("%"); ax.set_title("Average accuracy and forgetting")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "forgetting_curve_tasks.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append("forgetting_curve_tasks.png")

        fig, ax = plt.subplots(figsize=(max(4, 0.6 * T + 2), max(3, 0.5 * done + 1.5)))
        M = 100 * R[:done]
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(np.nanmax(M), 1))
        for i in range(done):
            for j in range(T):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                            color="#fcfcfb" if M[i, j] > 0.6 * np.nanmax(M) else "#0b0b0b")
        ax.set_xticks(np.arange(T), [f"eval T{j}" for j in range(T)], rotation=45, ha="right")
        ax.set_yticks(np.arange(done), [f"after T{i}" for i in range(done)]); ax.grid(False)
        ax.set_title("Accuracy matrix R[i, j] (val mIoU %); right of diagonal = unseen")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.savefig(os.path.join(out_dir, "accuracy_matrix.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append("accuracy_matrix.png")

    tc = [r for r in test_curve if r.get("split", "test") == "test"]
    if tc:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        xs = [r["task"] for r in tc]
        for k, key in enumerate(("mIoU", "IoU", "BIoU", "Dice")):
            _line(ax, xs, [100 * r[key] for r in tc], SERIES[k], key)
        for r_split in sorted({r["split"] for r in test_curve if r.get("split", "test") != "test"}):
            rr = [r for r in test_curve if r.get("split") == r_split]
            ax.plot([r["task"] for r in rr], [100 * r["mIoU"] for r in rr], color=SERIES[4], linestyle="--",
                    label=f"{r_split} mIoU")
        ax.set_xticks(xs, [f"after T{i}" for i in xs]); ax.set_ylabel("%")
        ax.set_title("Held-out metrics after each task"); ax.legend(loc="best", ncol=2)
        fig.savefig(os.path.join(out_dir, "test_metrics_per_task.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append("test_metrics_per_task.png")
    return files


def plot_class_forgetting(Rc, classes, out_dir, dpi=120):
    """Per-class IoU on the pooled val of all seen tasks after each stage (fixed class set)."""
    plt = setup_matplotlib()
    done = int((~np.all(np.isnan(Rc), axis=1)).sum())
    if done < 1:
        return []
    M = 100 * Rc[:done]
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(classes)), max(2.4, 0.4 * done + 1.4)))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(np.arange(done), [f"after T{i}" for i in range(done)]); ax.grid(False)
    if done > 1:
        f = np.nanmax(M[:-1], axis=0) - M[-1]
        ax.set_title(f"Class IoU on pooled seen-task val (avg class forgetting {np.nanmean(f):.2f} pts)")
    else:
        ax.set_title("Class IoU on pooled seen-task val")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="IoU %")
    fig.savefig(os.path.join(out_dir, "class_forgetting.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return ["class_forgetting.png"]


def plot_test_results(res, classes, palette, out_dir, dpi=120, prefix="test"):
    plt = setup_matplotlib()
    C = len(classes)
    pal = palette / 255.0
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.27 * C)), sharey=True)
    for ax, key, title in ((axes[0], "IoU_per_class", "IoU"), (axes[1], "F1_per_class", "F1")):
        v = 100 * np.array([np.nan if x is None else x for x in res[key]], float)
        ax.barh(np.arange(C), np.nan_to_num(v), color=pal, edgecolor="#52514e", linewidth=0.3, height=0.75)
        for i, val in enumerate(v):
            ax.text(0 if np.isnan(val) else val, i, "  n/a" if np.isnan(val) else f" {val:.1f}",
                    va="center", fontsize=7, color=INK2)
        ax.set_xlim(0, 105); ax.set_xlabel(f"{title} (%)"); ax.grid(axis="y", visible=False)
        mean = res["mIoU"] if title == "IoU" else res["Dice"]
        ax.set_title(f"Class-wise {title}   (mean {100 * mean:.2f})")
    axes[0].set_yticks(np.arange(C), classes); axes[0].invert_yaxis()
    fig.savefig(os.path.join(out_dir, f"{prefix}_classwise_iou_f1.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    files.append(f"{prefix}_classwise_iou_f1.png")

    conf = np.array(res["confusion"], float)
    norm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * C), max(5, 0.38 * C)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(C), classes, rotation=70, ha="right"); ax.set_yticks(np.arange(C), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.grid(False)
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    files.append(f"{prefix}_confusion_matrix.png")

    if res.get("per_domain_mIoU") and len(res["per_domain_mIoU"]) > 1:
        d = res["per_domain_mIoU"]
        fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(d) + 2), 3))
        ax.bar(list(d), [100 * v for v in d.values()], color=SERIES[0], width=0.6)
        for i, v in enumerate(d.values()):
            ax.text(i, 100 * v, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_ylabel("mIoU (%)"); ax.set_title("Test mIoU per domain"); ax.grid(axis="x", visible=False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.savefig(os.path.join(out_dir, f"{prefix}_per_domain_miou.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append(f"{prefix}_per_domain_miou.png")
    return files


@torch.no_grad()
def plot_samples(model, loader, cfg, device, out_dir, n=8, dpi=120, prefix="test"):
    plt = setup_matplotlib()
    from matplotlib.patches import Patch

    model.eval()
    classes, ign = list(cfg.dataset.classes), cfg.dataset.ignore_index
    pal = get_palette(cfg.dataset)
    mean, std = loader.dataset.transform.normalization()
    thr = cfg.model.inference.get("unknown_threshold")
    ds = loader.dataset
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    rows = []
    for i in idx:
        b = ds[int(i)]
        x = b["image"][None].to(device)
        logits = model(x, out_size=b["label"].shape[-2:])["logits"].float()
        conf, pred = logits.softmax(1).max(1)
        if thr is not None:
            pred[conf < thr] = ign
        rows.append((denormalize(b["image"], mean, std), b["label"].numpy(), pred[0].cpu().numpy(),
                     ds.samples[int(i)].domain))
    if not rows:
        return []
    h, w = rows[0][1].shape
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 12 * h / w / 3 * len(rows) + 0.8))
    axes = np.array(axes).reshape(len(rows), 3)
    for r, (im, gt, pr, dom) in enumerate(rows):
        valid = gt != ign
        acc = (pr[valid] == gt[valid]).mean() * 100 if valid.any() else 0
        for c, (img, title) in enumerate(((im, f"image ({dom})"), (colorize(gt, pal, ign), "ground truth"),
                                          (colorize(pr, pal, ign), f"prediction (pixel acc {acc:.1f}%)"))):
            axes[r, c].imshow(img); axes[r, c].axis("off")
            axes[r, c].set_title(title, fontsize=8)
    present = sorted({int(c) for _, g, p, _ in rows for c in np.unique(np.concatenate([g.ravel(), p.ravel()])) if c < len(classes)})
    fig.legend(handles=[Patch(color=pal[c] / 255.0, label=classes[c]) for c in present], loc="lower center",
               ncol=min(10, len(present)), fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fname = f"{prefix}_samples.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return [fname]


@torch.no_grad()
def plot_tsne(model, loader, cfg, device, out_dir, dpi=120, prefix="test"):
    from sklearn.manifold import TSNE

    plt = setup_matplotlib()
    tc = cfg.viz.tsne
    model.eval()
    classes, ign = list(cfg.dataset.classes), cfg.dataset.ignore_index
    C = len(classes)
    pal = get_palette(cfg.dataset) / 255.0
    rng = np.random.RandomState(0)
    feats, labs, doms, proto_sum, proto_n = [], [], [], None, 0
    base = loader.dataset
    # images spread evenly over the (domain-sorted) split so every domain is represented
    pick_idx = np.linspace(0, len(base) - 1, min(int(tc.max_images), len(base))).astype(int).tolist()
    sub = torch.utils.data.DataLoader(torch.utils.data.Subset(base, pick_idx), batch_size=loader.batch_size,
                                      num_workers=0, collate_fn=loader.collate_fn)
    for b in sub:
        x, y = b["image"].to(device), b["label"].to(device)
        out = model(x)
        z = F.normalize(out["z"].float(), dim=1)
        ys = F.interpolate(y[:, None].float(), size=z.shape[-2:], mode="nearest")[:, 0].long()
        if out["protos"] is not None:
            ps = out["protos"].float().sum(0)
            proto_sum = ps if proto_sum is None else proto_sum + ps
            proto_n += x.shape[0]
        for k in range(x.shape[0]):
            v = torch.nonzero((ys[k] != ign) & (ys[k] < C))
            if len(v) == 0:
                continue
            # class-balanced pixel sampling inside the image
            lab_all = ys[k][v[:, 0], v[:, 1]].cpu().numpy()
            per = max(1, tc.pixels_per_image // max(1, len(np.unique(lab_all))))
            pick = np.concatenate([rng.permutation(np.nonzero(lab_all == c)[0])[:per] for c in np.unique(lab_all)])
            vv = v[torch.from_numpy(pick).to(v.device)]
            feats.append(z[k][:, vv[:, 0], vv[:, 1]].t().cpu().numpy())
            labs.append(lab_all[pick])
            doms.append(np.full(len(pick), int(b["domain"][k])))
    if not feats:
        return []
    X, L, Dm = np.concatenate(feats), np.concatenate(labs), np.concatenate(doms)
    if len(X) > tc.max_points:
        sel = rng.choice(len(X), tc.max_points, replace=False)
        X, L, Dm = X[sel], L[sel], Dm[sel]
    P = None
    if proto_sum is not None:
        P = F.normalize(proto_sum / proto_n, dim=-1).cpu().numpy()
        X = np.concatenate([X, P])
    emb = TSNE(n_components=2, perplexity=min(tc.perplexity, max(5, len(X) // 4)), init="pca",
               random_state=0).fit_transform(X)
    E, EP = (emb[: -C], emb[-C:]) if P is not None else (emb, None)

    files = []
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in np.unique(L):
        m = L == c
        ax.scatter(E[m, 0], E[m, 1], s=4, color=pal[c], label=classes[c], alpha=0.8, linewidths=0)
    if EP is not None:
        ax.scatter(EP[:, 0], EP[:, 1], s=110, marker="*", color=pal, edgecolor="#0b0b0b", linewidths=0.8,
                   label="prototypes p_c", zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("t-SNE of pixel embeddings z(p), coloured by class (stars = mean prototypes)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=2.5, fontsize=7)
    fig.savefig(os.path.join(out_dir, f"{prefix}_tsne_classes.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    files.append(f"{prefix}_tsne_classes.png")

    idx2dom = {v: k for k, v in base.domain_to_idx.items()}
    if len(np.unique(Dm)) > 1:
        cols = _task_colors(len(idx2dom))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for d in np.unique(Dm):
            m = Dm == d
            ax.scatter(E[m, 0], E[m, 1], s=4, color=cols[d], label=idx2dom[d], alpha=0.7, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title("t-SNE of pixel embeddings z(p), coloured by domain")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=2.5)
        fig.savefig(os.path.join(out_dir, f"{prefix}_tsne_domains.png"), dpi=dpi, bbox_inches="tight"); plt.close(fig)
        files.append(f"{prefix}_tsne_domains.png")
    return files
