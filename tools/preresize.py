"""Resize a dataset once, offline, so training does not decode and resize full-size images
every step (on IDD that was the main bottleneck: 1080p PNG decode + resize per sample).

Images are resized with bilinear filtering, labels with nearest neighbour. The folder
layout under the new root is identical, so only dataset.root changes.

    python tools/preresize.py --config configs/idd.yaml --dst /raid/workspace/idd_720 --size 720 1280
    python tools/preresize.py --config configs/idd.yaml --dst /raid/workspace/idd_720 --size 720 1280 --jpg

With --jpg the images are stored as JPEG (quality 95, ~3x faster to decode than PNG); the
script prints the --opts to point the config at them. Labels always stay PNG (lossless).
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from drift.config import load_config  # noqa: E402
from drift.data.datasets import discover_samples, is_multi, sub_dataset_cfg  # noqa: E402


def _job(args):
    img_src, img_dst, lbl_src, lbl_dst, h, w, jpg = args
    if os.path.isfile(img_dst) and os.path.isfile(lbl_dst):
        return 0
    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
    im = Image.open(img_src).convert("RGB")
    if w <= 0:                       # short-side mode: h is the short side, aspect ratio kept
        s = h / min(im.size)
        w, h = round(im.size[0] * s), round(im.size[1] * s)
    if im.size != (w, h):
        im = im.resize((w, h), Image.BILINEAR)
    if jpg:
        im.save(img_dst, quality=95, subsampling=0)
    else:
        im.save(img_dst, compress_level=1)
    lb = Image.open(lbl_src)
    if lb.size != (w, h):
        lb = lb.resize((w, h), Image.NEAREST)
    lb.save(lbl_dst, compress_level=1)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dst", required=True, help="new dataset root")
    ap.add_argument("--size", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="fixed output size (may change the aspect ratio)")
    ap.add_argument("--short-side", type=int, default=None,
                    help="resize so the short side has this length, keeping the aspect ratio")
    ap.add_argument("--splits", nargs="*", default=["train", "val"], help="physical split folders")
    ap.add_argument("--jpg", action="store_true", help="store images as JPEG instead of PNG")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if is_multi(cfg.dataset):          # one sub-folder per dataset under --dst
        for name in cfg.dataset.domains:
            sub = sub_dataset_cfg(cfg.dataset, name)
            splits = sorted({v["split"] for v in cfg.dataset.domains[name].splits.values()
                             if isinstance(v, dict) and "split" in v and "from" not in v})
            print(f"== {name}: splits {splits}")
            run_one(sub, os.path.join(args.dst, name), splits, args)
        print(f"use with:\n  --opts " + " ".join(
            f'"dataset.domains.{n}.root={os.path.join(args.dst, n)}"' for n in cfg.dataset.domains))
        if args.jpg:
            print("  plus each dataset's image_glob / label_from_image changed from .png to .jpg")
        return
    run_one(cfg.dataset, args.dst, args.splits, args, print_opts=True)


def run_one(dcfg, dst, splits, args, print_opts=False):
    if args.short_side:
        h, w = args.short_side, 0
    elif args.size:
        h, w = args.size
    else:
        raise SystemExit("give --short-side N or --size H W")
    root = os.path.abspath(dcfg.root)
    jobs, lists = [], {}
    for split in splits:
        for s in discover_samples(dcfg, split):
            rel_img = s.rel
            rel_lbl = os.path.relpath(s.label, root).replace("\\", "/")
            if args.jpg:
                rel_img = os.path.splitext(rel_img)[0] + ".jpg"
            jobs.append((s.image, os.path.join(dst, rel_img), s.label, os.path.join(dst, rel_lbl),
                         h, w, args.jpg))
            lists.setdefault(split, []).append(f"{rel_img} {rel_lbl}")
    print(f"resizing {len(jobs)} image/label pairs to {f'short side {h}' if w <= 0 else f'{h}x{w}'} into {dst}")
    done = 0
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            done += r
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    print(f"done: {done} written, {len(jobs) - done} already present")
    opts = [f"dataset.root={dst}"]
    src = dcfg.source
    if src.mode == "list":       # write the same list file(s) under the new root, with the new image names
        for split, lines in lists.items():
            lf = os.path.join(dst, src.list_file.format(split=split))
            os.makedirs(os.path.dirname(lf) or ".", exist_ok=True)
            with open(lf, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
    elif args.jpg:
        glob_ = src.image_glob.replace(".png", ".jpg")
        repl = [[a, b] for a, b in (src.label_from_image or [])]
        repl = [[a.replace(".png", ".jpg") if a.endswith(".png") and "leftImg8bit" in a else a, b] for a, b in repl]
        opts += [f"dataset.source.image_glob={glob_}", f"dataset.source.label_from_image={repl}"]
    if print_opts:
        print("use with:\n  --opts " + " ".join(f'"{o}"' for o in opts))


if __name__ == "__main__":
    main()
