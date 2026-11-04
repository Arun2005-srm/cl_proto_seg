"""Prediction images from a checkpoint, including one written mid-training (last.pt is saved
after every epoch): image | ground truth | prediction | prediction blended over the image | error map.

    python tools/predict_overlay.py --run outputs/bdd_idd_dtrr/<timestamp>
    python tools/predict_overlay.py --run outputs/bdd_idd_dtrr/<timestamp> --domain idd --split test --num 16
    python tools/predict_overlay.py --run ... --ckpt task0.pt --device cpu     # leave the GPU to training

Writes <run>/predictions_<ckpt>/<domain>_<split>_<i>.png (4-panel figure) and
<domain>_<split>_<i>_overlay.png (full-resolution blend only).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from drift.config import load_config  # noqa: E402
from drift.data.datasets import SegDataset, build_splits, filter_domains  # noqa: E402
from drift.data.transforms import build_transforms  # noqa: E402
from drift.engine import amp_dtype, load_model  # noqa: E402
from drift.utils import colorize, denormalize, get_palette, setup_matplotlib  # noqa: E402


def norm_stats(tf_cfg):
    for op in tf_cfg:
        if op.get("name") == "Normalize":
            return list(op.get("mean", (0.485, 0.456, 0.406))), list(op.get("std", (0.229, 0.224, 0.225)))
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory (has config.yaml and the checkpoints)")
    ap.add_argument("--ckpt", default="last.pt", help="checkpoint file name inside --run (or a path)")
    ap.add_argument("--domain", default=None, help="dataset / domain to show (default: every domain)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num", type=int, default=8, help="images per domain")
    ap.add_argument("--alpha", type=float, default=0.5, help="prediction opacity in the overlay")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="which images: a fixed random pick per seed")
    ap.add_argument("--mask-ignore", action="store_true",
                    help="grey out pixels whose ground truth is ignore (e.g. the ego bonnet in BDD100K) in the overlay")
    ap.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. dataset roots")
    args = ap.parse_args()

    cfg = load_config(os.path.join(args.run, "config.yaml"), args.opts)
    ckpt = args.ckpt if os.path.isfile(args.ckpt) else os.path.join(args.run, args.ckpt)
    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, sd = load_model(cfg, ckpt, dev)
    model.eval()
    print(f"[predict] {ckpt}: saved during task {sd.get('task')} -> {dev}")

    samples = build_splits(cfg, verbose=False)[args.split]
    domains = [args.domain] if args.domain else sorted({s.domain for s in samples})
    tf = build_transforms(cfg.transforms["test" if args.split == "test" else "val"], cfg.dataset.ignore_index)
    mean, std = norm_stats(cfg.transforms["val"])
    pal = get_palette(cfg.dataset)
    classes = list(cfg.dataset.classes)
    ign = cfg.dataset.ignore_index
    amp = bool(cfg.experiment.amp) and dev.type == "cuda"
    out_dir = os.path.join(args.run, f"predictions_{os.path.splitext(os.path.basename(ckpt))[0]}")
    os.makedirs(out_dir, exist_ok=True)
    plt = setup_matplotlib()

    for dom in domains:
        sub = filter_domains(samples, [dom])
        if not sub:
            print(f"[predict] no {args.split} images for domain '{dom}' (have: {sorted({s.domain for s in samples})})")
            continue
        g = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(len(sub), generator=g)[: args.num].tolist()
        ds = SegDataset([sub[i] for i in pick], cfg.dataset, transform=tf)
        for i in range(len(ds)):
            item = ds[i]
            x, y = item["image"][None].to(dev), item["label"].numpy()
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dtype(cfg), enabled=amp):
                logits = model(x)["logits"]
            pred = F.interpolate(logits.float(), size=y.shape, mode="bilinear", align_corners=False).argmax(1)[0]
            pred = pred.cpu().numpy()
            img = denormalize(item["image"], mean, std)
            pc = colorize(pred, pal, ign)
            blend = (img * (1 - args.alpha) + pc * args.alpha).astype(np.uint8)
            if args.mask_ignore:
                blend[y == ign] = (img[y == ign] * 0.35 + 90).astype(np.uint8)
            # error map: correct green, wrong red, ignore (not scored) dark grey
            err = np.full((*y.shape, 3), 40, dtype=np.uint8)
            err[(y != ign) & (pred == y)] = (60, 170, 90)
            err[(y != ign) & (pred != y)] = (220, 50, 50)
            name = f"{dom}_{args.split}_{i:02d}"
            Image.fromarray(blend).save(os.path.join(out_dir, f"{name}_overlay.png"))

            valid = y != ign
            acc = float((pred[valid] == y[valid]).mean()) if valid.any() else float("nan")
            present = [c for c in np.unique(np.concatenate([y[valid], pred.ravel()])) if c < len(classes)]
            fig, ax = plt.subplots(1, 5, figsize=(27, 4.2))
            for a, im, title in zip(ax, (img, colorize(y, pal, ign), pc, blend, err),
                                    ("image", "ground truth (black = ignore, not scored)", "prediction",
                                     f"overlay (alpha {args.alpha})", "errors: green ok / red wrong / grey ignore")):
                a.imshow(im)
                a.set_title(title)
                a.axis("off")
            handles = [plt.Rectangle((0, 0), 1, 1, color=pal[c] / 255) for c in present]
            fig.legend(handles, [classes[c] for c in present], loc="lower center", ncol=min(len(present), 10),
                       fontsize=8, frameon=False)
            fig.suptitle(f"{dom} {args.split} | {os.path.basename(sub[pick[i]].image)} | pixel acc {100 * acc:.1f}%",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))
            fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=110)
            plt.close(fig)
        print(f"[predict] {dom}: {len(ds)} images -> {out_dir}")


if __name__ == "__main__":
    main()
