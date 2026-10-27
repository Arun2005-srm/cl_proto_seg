"""Joint image / label transforms built from config lists.

Each transform takes and returns (image: PIL.Image RGB, label: PIL.Image 'L' or 'I')
until Normalize, which converts to (float tensor [3,H,W], long tensor [H,W]).
Add a new op with @register_transform and use its class name in the config.
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF

TRANSFORMS: dict[str, type] = {}


def register_transform(cls):
    TRANSFORMS[cls.__name__] = cls
    return cls


def _hw(size):
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


@register_transform
class Resize:
    """size: [H, W] or int (short side when keep_ratio)."""

    def __init__(self, size, keep_ratio: bool = False):
        self.size, self.keep_ratio = size, keep_ratio

    def __call__(self, img, lbl):
        if self.keep_ratio and isinstance(self.size, int):
            w, h = img.size
            s = self.size / min(h, w)
            h2, w2 = round(h * s), round(w * s)
        else:
            h2, w2 = _hw(self.size)
        return img.resize((w2, h2), Image.BILINEAR), lbl.resize((w2, h2), Image.NEAREST)


@register_transform
class RandomScale:
    def __init__(self, scale=(0.5, 2.0), base_size=None):
        self.scale, self.base_size = scale, base_size

    def __call__(self, img, lbl):
        s = random.uniform(*self.scale)
        w, h = img.size
        if self.base_size is not None:
            bh, bw = _hw(self.base_size)
            h, w = bh, bw
        h2, w2 = max(1, int(h * s)), max(1, int(w * s))
        return img.resize((w2, h2), Image.BILINEAR), lbl.resize((w2, h2), Image.NEAREST)


@register_transform
class RandomCrop:
    """Pads with 0 / ignore_index when the image is smaller than the crop.
    cat_max_ratio < 1 retries crops dominated by a single class (mmseg style)."""

    def __init__(self, size, ignore_index: int = 255, cat_max_ratio: float = 1.0, tries: int = 10):
        self.size = _hw(size)
        self.ignore_index, self.cat_max_ratio, self.tries = ignore_index, cat_max_ratio, tries

    def __call__(self, img, lbl):
        ch, cw = self.size
        w, h = img.size
        if h < ch or w < cw:
            ph, pw = max(ch - h, 0), max(cw - w, 0)
            new_img = Image.new("RGB", (w + pw, h + ph), (0, 0, 0))
            new_img.paste(img, (0, 0))
            new_lbl = Image.new(lbl.mode, (w + pw, h + ph), self.ignore_index)
            new_lbl.paste(lbl, (0, 0))
            img, lbl = new_img, new_lbl
            w, h = img.size
        for _ in range(self.tries):
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            box = (x, y, x + cw, y + ch)
            if self.cat_max_ratio >= 1.0:
                break
            arr = np.asarray(lbl.crop(box))
            vals, cnt = np.unique(arr[arr != self.ignore_index], return_counts=True)
            if len(cnt) > 1 and cnt.max() / cnt.sum() < self.cat_max_ratio:
                break
        return img.crop(box), lbl.crop(box)


@register_transform
class CenterCrop:
    def __init__(self, size):
        self.size = _hw(size)

    def __call__(self, img, lbl):
        ch, cw = self.size
        w, h = img.size
        x, y = max((w - cw) // 2, 0), max((h - ch) // 2, 0)
        box = (x, y, x + min(cw, w), y + min(ch, h))
        return img.crop(box), lbl.crop(box)


@register_transform
class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT), lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl


@register_transform
class VerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM), lbl.transpose(Image.FLIP_TOP_BOTTOM)
        return img, lbl


@register_transform
class RandomRotate:
    def __init__(self, degrees: float = 10, p: float = 0.5, ignore_index: int = 255):
        self.degrees, self.p, self.ignore_index = degrees, p, ignore_index

    def __call__(self, img, lbl):
        if random.random() < self.p:
            a = random.uniform(-self.degrees, self.degrees)
            img = img.rotate(a, Image.BILINEAR, fillcolor=(0, 0, 0))
            lbl = lbl.rotate(a, Image.NEAREST, fillcolor=self.ignore_index)
        return img, lbl


@register_transform
class ColorJitter:
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p: float = 1.0):
        from torchvision.transforms import ColorJitter as CJ

        self.cj, self.p = CJ(brightness, contrast, saturation, hue), p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = self.cj(img)
        return img, lbl


@register_transform
class GaussianBlur:
    def __init__(self, kernel: int = 5, sigma=(0.1, 2.0), p: float = 0.5):
        self.sigma, self.p = sigma, p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))
        return img, lbl


@register_transform
class Grayscale:
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = img.convert("L").convert("RGB")
        return img, lbl


@register_transform
class Normalize:
    """Final op: PIL -> tensors."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean, self.std = list(mean), list(std)

    def __call__(self, img, lbl):
        t = TF.normalize(TF.to_tensor(img), self.mean, self.std)
        return t, torch.from_numpy(np.array(lbl, dtype=np.int64))


class Compose:
    def __init__(self, ops: list[Callable]):
        self.ops = ops
        if not ops or not isinstance(ops[-1], Normalize):
            self.ops = list(ops) + [Normalize()]

    def __call__(self, img, lbl):
        for op in self.ops:
            img, lbl = op(img, lbl)
        return img, lbl

    def normalization(self):
        n = self.ops[-1]
        return n.mean, n.std


def build_transforms(spec_list, ignore_index: int = 255) -> Compose:
    ops = []
    for spec in spec_list or []:
        spec = dict(spec)
        name = spec.pop("name")
        if name not in TRANSFORMS:
            raise KeyError(f"unknown transform '{name}'. Available: {sorted(TRANSFORMS)}")
        cls = TRANSFORMS[name]
        if "ignore_index" in cls.__init__.__code__.co_varnames and "ignore_index" not in spec:
            spec["ignore_index"] = ignore_index
        ops.append(cls(**spec))
    return Compose(ops)
