"""DRIFT building blocks: fusion decoder, shift estimator, prototype generator
(anchors + memory + shift library), cosine head and linear head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionDecoder(nn.Module):
    """SegFormer all-MLP decoder: project each stage, upsample to stride 4, fuse -> z(p)."""

    def __init__(self, in_channels, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, feats):
        size = feats[0].shape[-2:]
        ups = [F.interpolate(p(f), size=size, mode="bilinear", align_corners=False) if i else p(f)
               for i, (p, f) in enumerate(zip(self.proj, feats))]
        return self.fuse(torch.cat(ups, dim=1))


class ShiftEstimator(nn.Module):
    """g_phi: per-stage channel mean / std of encoder features -> s in R^shift_dim.
    One forward pass per image, no domain id."""

    def __init__(self, in_channels, shift_dim: int = 64, hidden: int = 256, use_std: bool = True,
                 detach_input: bool = False):
        super().__init__()
        self.use_std, self.detach_input = use_std, detach_input
        d_in = sum(in_channels) * (2 if use_std else 1)
        self.mlp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, shift_dim))

    def forward(self, feats):
        stats = []
        for f in feats:
            f = f.detach() if self.detach_input else f
            f = f.float()
            stats.append(f.mean(dim=(2, 3)))
            if self.use_std:
                stats.append(f.std(dim=(2, 3)))
        return self.mlp(torch.cat(stats, dim=1))


class PrototypeGenerator(nn.Module):
    """p_c = norm(a_c + m_c + U_c B s [+ w_c]); each term exists only if enabled.

    anchors a_c   : frozen buffer [C, D]
    memory  m_c   : buffer [C, K, D], written at the end of a task (EMA of z over
                    observed classes only), read as the mean of filled slots
    shift   U_c B s: W_u [C, Kb] mixes Kb basis shifts B_k s, B: [Kb, D, S]
    base    w_c   : optional learnable [C, D]
    """

    def __init__(self, num_classes: int, embed_dim: int, shift_dim: int, use_anchors: bool, use_memory: bool,
                 use_shift: bool, use_base: bool, anchors: torch.Tensor | None = None, num_slots: int = 1,
                 momentum: float = 0.0, num_bases: int = 8):
        super().__init__()
        C, D = num_classes, embed_dim
        self.C, self.D = C, D
        self.use_anchors, self.use_memory, self.use_shift, self.use_base = use_anchors, use_memory, use_shift, use_base
        self.momentum = momentum
        if not (use_anchors or use_memory or use_shift or use_base):
            raise ValueError("prototype generator needs at least one of anchors/memory/shift_library/learnable_base")
        if use_anchors:
            self.register_buffer("anchors", F.normalize(anchors.float(), dim=-1))
        if use_memory:
            self.register_buffer("memory", torch.zeros(C, num_slots, D))
            self.register_buffer("slot_filled", torch.zeros(C, num_slots, dtype=torch.bool))
            self.register_buffer("write_count", torch.zeros(C, dtype=torch.long))
        if use_shift:
            self.W_u = nn.Parameter(torch.randn(C, num_bases) * 0.1)
            self.B = nn.Parameter(torch.randn(num_bases, D, shift_dim) * (0.02 / shift_dim ** 0.5))
        if use_base:
            self.base = nn.Parameter(torch.randn(C, D) * 0.02)

    # -------------------------------------------------------------- memory
    def memory_read(self) -> torch.Tensor:
        filled = self.slot_filled.float().unsqueeze(-1)                     # [C,K,1]
        return (self.memory * filled).sum(1) / filled.sum(1).clamp(min=1)

    def has_memory(self) -> torch.Tensor:
        if not self.use_memory:
            return torch.zeros(self.C, dtype=torch.bool)
        return self.slot_filled.any(dim=1)

    @torch.no_grad()
    def memory_write(self, class_means: torch.Tensor, observed: torch.Tensor):
        """class_means [C, D] (any norm), observed [C] bool. Absent classes untouched."""
        K = self.memory.shape[1]
        for c in torch.nonzero(observed).flatten().tolist():
            k = int(self.write_count[c]) % K
            new = F.normalize(class_means[c].float(), dim=-1)
            if self.slot_filled[c, k] and self.momentum > 0:
                new = F.normalize(self.momentum * self.memory[c, k] + (1 - self.momentum) * new, dim=-1)
            self.memory[c, k] = new
            self.slot_filled[c, k] = True
            self.write_count[c] += 1

    # -------------------------------------------------------------- forward
    def forward(self, s: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        p = torch.zeros(self.C, self.D, device=self._device())
        if self.use_anchors:
            p = p + self.anchors
        if self.use_memory:
            p = p + self.memory_read()
        if self.use_base:
            p = p + self.base
        p = p.unsqueeze(0).expand(batch_size, -1, -1)
        if self.use_shift and s is not None:
            Bs = torch.einsum("kds,bs->bkd", self.B, s.float())              # [B,Kb,D]
            p = p + torch.einsum("ck,bkd->bcd", self.W_u, Bs)
        return F.normalize(p, dim=-1)                                       # [B,C,D]

    def _device(self):
        for t in list(self.buffers()) + list(self.parameters()):
            return t.device
        return torch.device("cpu")


class CosineHead(nn.Module):
    """l_c = cos(z, p_c) / tau."""

    def __init__(self, tau: float = 0.07, learnable: bool = False):
        super().__init__()
        t = torch.tensor(float(tau)).log()
        if learnable:
            self.log_tau = nn.Parameter(t)
        else:
            self.register_buffer("log_tau", t)

    def forward(self, z, protos):
        z = F.normalize(z.float(), dim=1)
        return torch.einsum("bdhw,bcd->bchw", z, protos.float()) / self.log_tau.exp()


class LinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.cls = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, z, protos=None):
        return self.cls(z)
