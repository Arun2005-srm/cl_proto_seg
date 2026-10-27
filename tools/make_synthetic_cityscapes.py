"""Write a tiny fake dataset in the exact Cityscapes folder layout (labelIds PNGs),
so the whole pipeline (configs, EDA, training, test, plots) can be smoke-tested
without the real data. Each city has its own colour shift and some absent classes.

    python tools/make_synthetic_cityscapes.py --out data/synthetic_cityscapes
"""
import argparse
import os

import numpy as np
from PIL import Image

# labelId: base colour
CLS = {7: (128, 64, 128), 8: (232, 35, 244), 11: (70, 70, 70), 17: (153, 153, 153), 20: (220, 220, 0),
       21: (107, 142, 35), 23: (70, 130, 180), 24: (220, 20, 60), 26: (0, 0, 142), 33: (119, 11, 32)}
TRAIN = {"aachen": [], "bochum": [33], "bremen": [24], "cologne": [20, 33], "darmstadt": [], "dusseldorf": [24]}
VAL = {"frankfurt": [], "lindau": [33]}


def scene(rng, h, w, absent, tint):
    lbl = np.full((h, w), 23, np.uint8)                       # sky
    hz = int(h * rng.uniform(0.35, 0.5))
    lbl[int(h * 0.15):hz, :] = 11                              # buildings
    for _ in range(rng.randint(1, 4)):                         # trees
        x = rng.randint(0, w - 30)
        lbl[int(h * 0.2):hz, x:x + rng.randint(15, 40)] = 21
    lbl[hz:, :] = 7                                            # road
    sw = int(w * rng.uniform(0.1, 0.2))
    lbl[hz:, :sw] = 8
    lbl[hz:, w - sw:] = 8
    objs = [(26, 0.12, 0.22, 2), (24, 0.04, 0.14, 3), (33, 0.05, 0.09, 1), (17, 0.015, 0.3, 2), (20, 0.05, 0.05, 1)]
    for cid, ow, oh, n in objs:
        if cid in absent:
            continue
        for _ in range(rng.randint(0, n + 1)):
            ww, hh = max(2, int(w * ow * rng.uniform(0.7, 1.3))), max(2, int(h * oh * rng.uniform(0.7, 1.3)))
            x = rng.randint(0, w - ww)
            y = rng.randint(max(0, hz - hh // 2), max(hz - hh // 2 + 1, h - hh)) if cid != 20 else rng.randint(0, hz)
            lbl[y:y + hh, x:x + ww] = cid
    img = np.zeros((h, w, 3), np.float32)
    for cid, col in CLS.items():
        img[lbl == cid] = col
    img = img * tint[None, None] + rng.normal(0, 12, img.shape)
    lbl[rng.rand(h, w) < 0.01] = 0                             # a few "unlabeled" pixels -> ignore
    return np.clip(img, 0, 255).astype(np.uint8), lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_cityscapes")
    ap.add_argument("--per-city", type=int, default=12)
    ap.add_argument("--size", type=int, nargs=2, default=[128, 256])
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    h, w = args.size
    for split, cities in (("train", TRAIN), ("val", VAL)):
        for ci, (city, absent) in enumerate(cities.items()):
            tint = rng.uniform(0.6, 1.3, 3)
            di, dl = (os.path.join(args.out, "leftImg8bit", split, city), os.path.join(args.out, "gtFine", split, city))
            os.makedirs(di, exist_ok=True)
            os.makedirs(dl, exist_ok=True)
            for k in range(args.per_city):
                img, lbl = scene(rng, h, w, absent, tint)
                stem = f"{city}_{k:06d}_000019"
                Image.fromarray(img).save(os.path.join(di, f"{stem}_leftImg8bit.png"))
                Image.fromarray(lbl).save(os.path.join(dl, f"{stem}_gtFine_labelIds.png"))
    print(f"wrote synthetic Cityscapes-layout data to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
