"""Minimal LoRA for nn.Linear layers, with merge for inference."""
from __future__ import annotations

import math

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


def inject_lora(module: nn.Module, targets, rank: int, alpha: float, dropout: float) -> int:
    """Replace every nn.Linear whose attribute name is in `targets`. Returns count."""
    targets = set(targets)
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            n += 1
        else:
            n += inject_lora(child, targets, rank, alpha, dropout)
    return n


def set_lora_merged(module: nn.Module, merged: bool) -> None:
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.merge() if merged else m.unmerge()
