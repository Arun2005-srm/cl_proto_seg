"""Minimal LoRA for nn.Linear layers, with merge for inference.

Targets are regular expressions matched against each Linear's qualified name
(e.g. "encoder.block.0.0.attention.self.query"), so they survive renames across
transformers versions (query/key/value vs q_proj/k_proj/v_proj).
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        return out + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale
            self.merged = False


def linear_names(module: nn.Module) -> list[str]:
    return [n for n, m in module.named_modules() if isinstance(m, nn.Linear)]


def inject_lora(module: nn.Module, targets, rank: int, alpha: float, dropout: float) -> list[str]:
    """Wrap every nn.Linear whose qualified name matches one of `targets` (regexes).
    A bare word such as "query" matches the last name component exactly. Returns wrapped names."""
    pats = []
    for t in targets:
        pats.append(re.compile(rf"(^|\.){t}$") if re.fullmatch(r"\w+", t) else re.compile(t))
    wrapped = []
    for name in linear_names(module):
        if not any(p.search(name) for p in pats):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha, dropout))
        wrapped.append(name)
    return wrapped


def set_lora_merged(module: nn.Module, merged: bool) -> None:
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.merge() if merged else m.unmerge()
