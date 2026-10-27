"""YAML config loading with `_base_` inheritance, deep merge and CLI overrides.

A dataset config only has to state what differs from `configs/defaults.yaml`:

    _base_: defaults.yaml
    dataset: {root: D:/data/cityscapes, ...}

CLI overrides use dotted keys:  --opts train.lr=3e-4 model.modules.lora=false
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


class Config(dict):
    """dict with attribute access (cfg.train.lr). Nested dicts become Config."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e

    def __setattr__(self, key, value):
        self[key] = _wrap(value)

    def get_path(self, dotted: str, default=None):
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(v):
    if isinstance(v, Config):
        return v
    if isinstance(v, dict):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _unwrap(v):
    if isinstance(v, dict):
        return {k: _unwrap(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unwrap(x) for x in v]
    return v


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base. `_replace_: true` inside a
    dict in override replaces the base dict instead of merging into it."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and v.get("_replace_"):
            v = {kk: vv for kk, vv in v.items() if kk != "_replace_"}
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_tree(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        bpath = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        merged = deep_merge(merged, _load_yaml_tree(bpath))
    return deep_merge(merged, data)


def _parse_value(s: str):
    # yaml gives us ints, floats, bools, null, lists ([a,b]) and dicts for free
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def apply_overrides(cfg: dict, opts: Iterable[str]) -> dict:
    for item in opts or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def load_config(path: str, opts: Iterable[str] | None = None) -> Config:
    data = _load_yaml_tree(path)
    data = apply_overrides(data, opts)
    cfg = Config(data)
    cfg.setdefault("config_path", os.path.abspath(path))
    return cfg


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
