"""Dataset-agnostic sample discovery and the segmentation Dataset.

Nothing here knows about Cityscapes. Everything comes from `cfg.dataset`:
where images are, how to find each image's label, how to read the domain
(city) from a path, and how raw label ids map to train ids.
"""
from __future__ import annotations

import glob
import os
import random
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    image: str          # absolute path
    label: str          # absolute path
    domain: str         # city / site / "all"
    rel: str            # image path relative to root (forward slashes)


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def discover_samples(dcfg, split: str) -> list[Sample]:
    """Find all (image, label, domain) triples of one physical split."""
    root = dcfg.root
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"dataset.root '{root}' does not exist - set it in the config")
    src = dcfg.source
    if src.mode == "list":
        list_file = src.list_file.format(split=split)
        list_file = list_file if os.path.isabs(list_file) else os.path.join(root, list_file)
        pairs = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((os.path.join(root, parts[0]), os.path.join(root, parts[1])))
    elif src.mode == "glob":
        images = sorted(glob.glob(os.path.join(root, src.image_glob.format(split=split)), recursive=True))
        if src.get("label_glob"):
            labels = sorted(glob.glob(os.path.join(root, src.label_glob.format(split=split)), recursive=True))
            if len(labels) != len(images):
                raise ValueError(f"{len(images)} images but {len(labels)} labels for split '{split}'")
            pairs = list(zip(images, labels))
        else:
            pairs = []
            for im in images:
                rel = _rel(im, root)
                for old, new in src.label_from_image or []:
                    rel = rel.replace(old, new)
                pairs.append((im, os.path.join(root, rel)))
    else:
        raise ValueError(f"unknown dataset.source.mode '{src.mode}'")

    rx = re.compile(src.domain_regex) if src.get("domain_regex") else None
    out, missing = [], 0
    for im, lb in pairs:
        if not os.path.isfile(lb):
            missing += 1
            continue
        rel = _rel(im, root)
        dom = "all"
        if rx is not None:
            m = rx.search(rel)
            dom = m.group(1) if m else "unknown"
        out.append(Sample(os.path.abspath(im), os.path.abspath(lb), dom, rel))
    if missing:
        print(f"[data] split '{split}': skipped {missing} images without a label file")
    if not out:
        raise FileNotFoundError(f"no samples found for split '{split}' under {root} "
                                f"(image_glob={src.get('image_glob')}, list_file={src.get('list_file')})")
    return out


def build_lut(dcfg) -> np.ndarray | None:
    """Raw id -> train id lookup table (65536 entries) or None for identity."""
    ign = dcfg.ignore_index
    if dcfg.get("label_map"):
        lut = np.full(65536, ign, dtype=np.int64)
        for k, v in dcfg.label_map.items():
            lut[int(k)] = int(v)
        return lut
    if dcfg.get("reduce_zero_label"):
        lut = np.arange(65536, dtype=np.int64) - 1
        lut[0] = ign
        lut[ign] = ign
        return lut
    return None


def load_label(path: str, lut: np.ndarray | None, num_classes: int, ignore_index: int) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:            # some datasets store labels as RGB with id in channel 0
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    if lut is not None:
        arr = lut[arr]
    arr[(arr >= num_classes) & (arr != ignore_index)] = ignore_index
    return arr


class SegDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], dcfg, transform=None, return_meta: bool = True):
        self.samples = list(samples)
        self.dcfg = dcfg
        self.num_classes = len(dcfg.classes)
        self.ignore_index = dcfg.ignore_index
        self.lut = build_lut(dcfg)
        self.transform = transform
        self.return_meta = return_meta
        domains = sorted({s.domain for s in self.samples})
        self.domain_to_idx = {d: i for i, d in enumerate(domains)}

    def __len__(self):
        return len(self.samples)

    def read_label(self, idx: int) -> np.ndarray:
        s = self.samples[idx]
        return load_label(s.label, self.lut, self.num_classes, self.ignore_index)

    def read_image(self, idx: int) -> Image.Image:
        return Image.open(self.samples[idx].image).convert("RGB")

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self.read_image(idx)
        lbl = self.read_label(idx)
        mode = "L" if self.num_classes < 255 and self.ignore_index <= 255 else "I"
        lbl_img = Image.fromarray(lbl.astype(np.uint8 if mode == "L" else np.int32), mode=mode)
        if self.transform is not None:
            img_t, lbl_t = self.transform(img, lbl_img)
        else:
            import torchvision.transforms.functional as TF
            img_t, lbl_t = TF.to_tensor(img), torch.from_numpy(lbl)
        out = {"image": img_t, "label": lbl_t.long()}
        if self.return_meta:
            out["index"] = idx
            out["domain"] = self.domain_to_idx[s.domain]
        return out


# --------------------------------------------------------------------------- splits
def _select_domains(samples: list[Sample], domains) -> list[Sample]:
    if domains in (None, "all"):
        return samples
    keep = set(domains)
    return [s for s in samples if s.domain in keep]


def build_splits(cfg, verbose: bool = True) -> dict[str, list[Sample]]:
    """Resolve cfg.splits into {'train': [...], 'val': [...], 'test': [...]}.

    A split is either `{split: <folder>, domains: all|[...]}` or
    `{from: <other split>, fraction: f, per_domain: bool}` which moves a
    deterministic random fraction out of that split.
    """
    scfg = cfg.splits
    cache: dict[str, list[Sample]] = {}
    out: dict[str, list[Sample]] = {}
    rng = random.Random(cfg.experiment.seed)

    def physical(name):
        if name not in cache:
            cache[name] = discover_samples(cfg.dataset, name)
        return cache[name]

    for name in ("train", "val", "test"):
        spec = scfg.get(name)
        if not spec or "split" not in spec:
            continue
        out[name] = _select_domains(physical(spec.split), spec.get("domains", "all"))

    for name in ("train", "val", "test"):
        spec = scfg.get(name)
        if not spec or "from" not in spec:
            continue
        parent = out[spec["from"]]
        frac = float(spec.get("fraction", 0.1))
        groups: dict[str, list[Sample]] = {}
        for s in parent:
            groups.setdefault(s.domain if spec.get("per_domain", True) else "_", []).append(s)
        taken, kept = [], []
        for _, g in sorted(groups.items()):
            g = list(g)
            rng.shuffle(g)
            k = max(1, int(round(len(g) * frac))) if len(g) > 1 else 0
            taken += g[:k]
            kept += g[k:]
        out[name] = sorted(taken, key=lambda s: s.rel)
        out[spec["from"]] = sorted(kept, key=lambda s: s.rel)

    caps = scfg.get("max_samples", {}) or {}
    for name, lst in out.items():
        cap = caps.get(name)
        if cap:
            by_dom: dict[str, list[Sample]] = {}
            for s in lst:
                by_dom.setdefault(s.domain, []).append(s)
            per = max(1, int(cap) // max(1, len(by_dom)))
            out[name] = [s for g in by_dom.values() for s in g[:per]]
    if verbose:
        for name, lst in out.items():
            doms = sorted({s.domain for s in lst})
            print(f"[data] {name:5s}: {len(lst):6d} samples, {len(doms)} domains")
    return out


def build_tasks(cfg, train_samples: list[Sample]) -> list[list[str]]:
    """Continual tasks as lists of domains."""
    ccfg = cfg.continual
    domains = sorted({s.domain for s in train_samples})
    if not ccfg.enabled:
        return [domains]
    if ccfg.tasks == "auto":
        if ccfg.order == "random":
            random.Random(cfg.experiment.seed).shuffle(domains)
        elif ccfg.order == "as_found":
            seen, domains = set(), []
            for s in train_samples:
                if s.domain not in seen:
                    seen.add(s.domain)
                    domains.append(s.domain)
        g = int(ccfg.group_size)
        tasks = [domains[i:i + g] for i in range(0, len(domains), g)]
    else:
        tasks = [list(t) if isinstance(t, (list, tuple)) else [t] for t in ccfg.tasks]
        unknown = {d for t in tasks for d in t} - set(domains)
        if unknown:
            raise ValueError(f"continual.tasks names unknown domains {sorted(unknown)}; found {domains}")
    if ccfg.get("num_tasks"):
        tasks = tasks[: int(ccfg.num_tasks)]
    return tasks


def filter_domains(samples: list[Sample], domains: Sequence[str]) -> list[Sample]:
    keep = set(domains)
    return [s for s in samples if s.domain in keep]
