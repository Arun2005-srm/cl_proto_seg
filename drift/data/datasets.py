"""Dataset-agnostic sample discovery and the segmentation Dataset.

Nothing here knows about Cityscapes. Everything comes from `cfg.dataset`:
where images are, how to find each image's label, how to read the domain
(city) from a path, and how raw label ids map to train ids.

Several datasets can be combined as domains (cross-dataset continual learning):
`dataset.domains: {name: {root, source, label_map, splits}}`. Each sample then
carries `source` (its dataset, which selects its label map) and its domain is the
dataset name; the original city/sequence is kept in `subdomain`.
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
    domain: str         # city / site / dataset name / "all"
    rel: str            # image path relative to root (forward slashes)
    source: str = ""    # dataset name in a multi-dataset config ("" for a single dataset)
    subdomain: str = "" # original domain (city / sequence) when `domain` is the dataset name

    @property
    def key(self) -> str:
        """Unique across datasets (rel paths of two datasets can coincide)."""
        return f"{self.source}|{self.rel}" if self.source else self.rel


def is_multi(dcfg) -> bool:
    return bool(dcfg.get("domains"))


def sub_dataset_cfg(dcfg, name: str):
    """Config of one dataset inside a multi-dataset config: shared keys (classes, palette,
    ignore_index, ...) plus that dataset's own root / source / label_map."""
    from ..config import Config

    base = {k: v for k, v in dcfg.to_dict().items() if k != "domains"}
    own = {k: v for k, v in dcfg.domains[name].to_dict().items() if k != "splits"}
    base.update(own)
    base["name"] = name
    return Config(base)


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


def natural_key(s: str):
    """Sort '2' before '10' (IDD sequence folders are numbers)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


class SegDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], dcfg, transform=None, return_meta: bool = True,
                 drop_classes: Sequence[int] = ()):
        self.samples = list(samples)
        self.dcfg = dcfg
        self.num_classes = len(dcfg.classes)
        self.ignore_index = dcfg.ignore_index
        self.lut = build_lut(dcfg) if not is_multi(dcfg) else None
        self.luts = {n: build_lut(sub_dataset_cfg(dcfg, n)) for n in dcfg.domains} if is_multi(dcfg) else {}
        self.drop_classes = list(drop_classes)   # controlled absence: these train ids become ignore
        self.transform = transform
        self.return_meta = return_meta
        domains = sorted({s.domain for s in self.samples}, key=natural_key)
        self.domain_to_idx = {d: i for i, d in enumerate(domains)}

    def __len__(self):
        return len(self.samples)

    def read_label(self, idx: int) -> np.ndarray:
        s = self.samples[idx]
        lut = self.luts[s.source] if s.source else self.lut
        lbl = load_label(s.label, lut, self.num_classes, self.ignore_index)
        if self.drop_classes:
            lbl[np.isin(lbl, self.drop_classes)] = self.ignore_index
        return lbl

    def read_image(self, idx: int) -> Image.Image:
        return Image.open(self.samples[idx].image).convert("RGB")

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self.read_image(idx)
        lbl = self.read_label(idx)
        mode = "L" if self.num_classes < 255 and self.ignore_index <= 255 else "I"
        # dtype picks the mode (uint8 -> "L", int32 -> "I"); the `mode=` argument is removed in Pillow 13
        lbl_img = Image.fromarray(lbl.astype(np.uint8 if mode == "L" else np.int32))
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
    """Resolve the splits into {name: [samples]} for every split named in the config
    (train / val / test, plus extras such as val_unseen).

    A split is one of
      {split: <folder>, domains: all|[...]}                 a physical folder split
      {from: <other>, fraction: f, per_domain: bool}         a random fraction moved out of another split
      {from: <other>, holdout_domains: k | [names]}          whole domains moved out (unseen-domain val)

    Multi-dataset configs (`dataset.domains`) give each dataset its own `splits` block; the
    splits are built per dataset (stratified by its cities / sequences) and then merged, with
    every sample's domain set to its dataset name.
    """
    seed = cfg.experiment.seed
    if is_multi(cfg.dataset):
        out: dict[str, list[Sample]] = {}
        for name in cfg.dataset.domains:
            sub = sub_dataset_cfg(cfg.dataset, name)
            part = _splits_for(sub, cfg.dataset.domains[name].splits, seed, source=name)
            for split, lst in part.items():
                for s in lst:
                    s.source, s.subdomain, s.domain = name, s.domain, name
                out.setdefault(split, []).extend(lst)
            if verbose:
                print(f"[data] {name}: " + ", ".join(f"{k} {len(v)}" for k, v in part.items()))
    else:
        out = _splits_for(cfg.dataset, cfg.splits, seed)
    caps = (cfg.get("splits", {}) or {}).get("max_samples", {}) or {}
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
            doms = sorted({s.domain for s in lst}, key=natural_key)
            print(f"[data] {name:10s}: {len(lst):6d} samples, {len(doms)} domains")
    return out


def _splits_for(dcfg, scfg, seed: int, source: str = "") -> dict[str, list[Sample]]:
    names = [k for k in scfg.keys() if k != "max_samples" and scfg[k]]
    cache: dict[str, list[Sample]] = {}
    out: dict[str, list[Sample]] = {}
    rng = random.Random(seed)

    def physical(name):
        if name not in cache:
            cache[name] = discover_samples(dcfg, name)
        return cache[name]

    for name in names:
        spec = scfg[name]
        if "split" in spec and "from" not in spec:   # `from` wins when a merge left both keys
            out[name] = _select_domains(physical(spec.split), spec.get("domains", "all"))

    # domain hold-outs first, so fractions are carved from what is left
    derived = [n for n in names if "from" in scfg[n]]
    derived.sort(key=lambda n: 0 if "holdout_domains" in scfg[n] else 1)
    for name in derived:
        spec = scfg[name]
        parent = out[spec["from"]]
        if "holdout_domains" in spec:
            hd = spec["holdout_domains"]
            doms = sorted({s.domain for s in parent}, key=natural_key)
            if isinstance(hd, int):
                hd = sorted(random.Random(seed + 1).sample(doms, min(hd, len(doms) - 1)), key=natural_key)
            hd = set(hd)
            out[name] = [s for s in parent if s.domain in hd]
            out[spec["from"]] = [s for s in parent if s.domain not in hd]
            continue
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
    return out


def _balanced_groups(domains: list[str], counts: dict[str, int], n: int) -> list[list[str]]:
    """Split an ordered domain list into n contiguous groups with similar image counts."""
    n = max(1, min(n, len(domains)))
    total = sum(counts[d] for d in domains)
    groups, cur, acc = [], [], 0
    for i, d in enumerate(domains):
        cur.append(d)
        acc += counts[d]
        left_groups = n - len(groups) - 1
        left_domains = len(domains) - i - 1
        if left_groups > 0 and (acc >= total * (len(groups) + 1) / n or left_domains == left_groups):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_tasks(cfg, train_samples: list[Sample]) -> list[list[str]]:
    """Continual tasks as lists of domains.

    tasks: auto groups domains by `group_size` (domains per task) or `num_groups`
    (that many tasks with balanced image counts); `min_images_per_task` then merges
    any task that is still too small into its neighbour."""
    ccfg = cfg.continual
    counts: dict[str, int] = {}
    for s in train_samples:
        counts[s.domain] = counts.get(s.domain, 0) + 1
    domains = sorted(counts, key=natural_key)
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
        if ccfg.get("num_groups"):
            tasks = _balanced_groups(domains, counts, int(ccfg.num_groups))
        else:
            g = int(ccfg.group_size)
            tasks = [domains[i:i + g] for i in range(0, len(domains), g)]
    else:
        tasks = [list(map(str, t)) if isinstance(t, (list, tuple)) else [str(t)] for t in ccfg.tasks]
        unknown = {d for t in tasks for d in t} - set(domains)
        if unknown:
            raise ValueError(f"continual.tasks names unknown domains {sorted(unknown)}; found {domains}")
    min_imgs = int(ccfg.get("min_images_per_task") or 0)
    if min_imgs:
        merged: list[list[str]] = []
        for t in tasks:
            if merged and sum(counts[d] for d in merged[-1]) < min_imgs:
                merged[-1] = merged[-1] + t
            else:
                merged.append(list(t))
        if len(merged) > 1 and sum(counts[d] for d in merged[-1]) < min_imgs:
            merged[-2] += merged.pop()
        tasks = merged
    if ccfg.get("num_tasks"):
        tasks = tasks[: int(ccfg.num_tasks)]
    return tasks


def filter_domains(samples: list[Sample], domains: Sequence[str]) -> list[Sample]:
    keep = set(domains)
    return [s for s in samples if s.domain in keep]
