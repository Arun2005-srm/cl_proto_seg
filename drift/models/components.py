"""DRIFT building blocks: fusion decoder, shift estimator, prototype generator
(anchors + domain-keyed memory + shift library), cosine head and linear head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(kind: str, c: int) -> nn.Module:
    """GroupNorm by default: its statistics do not drift with the task or depend on batch size."""
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        return nn.GroupNorm(min(32, c), c)
    if kind == "ln":
        return nn.GroupNorm(1, c)
    if kind in (None, "none"):
        return nn.Identity()
    raise KeyError(f"unknown norm '{kind}' (bn | gn | ln | none)")


class FusionDecoder(nn.Module):
    """SegFormer all-MLP decoder: project each stage, upsample to stride 4, fuse -> z(p)."""

    def __init__(self, in_channels, embed_dim: int = 256, dropout: float = 0.1, norm: str = "gn"):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            make_norm(norm, embed_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, feats):
        size = feats[0].shape[-2:]
        ups = [F.interpolate(p(f), size=size, mode="bilinear", align_corners=False) if i else p(f)
               for i, (p, f) in enumerate(zip(self.proj, feats))]
        return self.fuse(torch.cat(ups, dim=1))


class ShiftEstimator(nn.Module):
    """g_phi: per-stage channel mean / std of encoder features -> s in R^shift_dim.
    One forward pass per image, no domain id. `stats` and `head` are split so the
    hidden-class loss can run the head on detached statistics."""

    def __init__(self, in_channels, shift_dim: int = 64, hidden: int = 256, use_std: bool = True,
                 detach_input: bool = True):
        super().__init__()
        self.use_std, self.detach_input = use_std, detach_input
        d_in = sum(in_channels) * (2 if use_std else 1)
        self.mlp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, shift_dim))

    def stats(self, feats):
        out = []
        for f in feats:
            f = f.detach() if self.detach_input else f
            f = f.float()
            out.append(f.mean(dim=(2, 3)))
            if self.use_std:   # population std with eps: finite value and gradient even for 1x1 maps
                out.append((f.var(dim=(2, 3), unbiased=False) + 1e-6).sqrt())
        return torch.cat(out, dim=1)

    def head(self, stats):
        return self.mlp(stats)

    def forward(self, feats):
        st = self.stats(feats)
        return self.head(st), st


class PrototypeGenerator(nn.Module):
    """p_c(x) = norm(g_a a_c + g_m m_c(x) + g_s U_c B s(x) [+ g_b w_c]).

    anchors a_c : frozen buffer [C, D] (CLIP text or fixed)
    memory m_c  : K slots per class, each with a domain key (mean s of the images that
                  wrote it) and a pixel count. Writes are count-weighted running means,
                  so a slot is the average over every earlier city routed to it, never
                  just the last one. Reads attend over slots with the image's s, so a
                  test image picks the past cities it resembles.
    shift       : W_u [C, Kb] mixes Kb basis shifts B_k s, B [Kb, D, S]. W_u starts at 0,
                  so the shift is a residual that grows from nothing. Classes never
                  observed in training get no shift (their W_u row is untrained).
    g_*         : learnable per-term gains (log-space), so the terms are not summed blindly.
    """

    def __init__(self, num_classes: int, embed_dim: int, shift_dim: int, use_anchors: bool, use_memory: bool,
                 use_shift: bool, use_base: bool, anchors: torch.Tensor | None = None, num_slots: int = 4,
                 momentum: float = 0.0, num_bases: int = 8, read_tau: float = 0.1, term_gains: bool = True):
        super().__init__()
        C, D, S = num_classes, embed_dim, shift_dim
        self.C, self.D, self.S, self.K = C, D, S, num_slots
        self.use_anchors, self.use_memory, self.use_shift, self.use_base = use_anchors, use_memory, use_shift, use_base
        self.momentum, self.read_tau = momentum, read_tau
        if not (use_anchors or use_memory or use_shift or use_base):
            raise ValueError("prototype generator needs at least one of anchors/memory/shift_library/learnable_base")
        if use_anchors:
            self.register_buffer("anchors", F.normalize(anchors.float(), dim=-1))
        if use_memory:
            self.register_buffer("memory", torch.zeros(C, num_slots, D))
            self.register_buffer("mem_keys", torch.zeros(C, num_slots, S))
            self.register_buffer("mem_count", torch.zeros(C, num_slots))
        self.register_buffer("class_seen", torch.zeros(C, dtype=torch.bool))
        if use_shift:
            self.W_u = nn.Parameter(torch.zeros(C, num_bases))
            self.B = nn.Parameter(torch.randn(num_bases, D, S) * (1.0 / S ** 0.5))
        if use_base:
            self.base = nn.Parameter(torch.randn(C, D) * 0.02)
        names = ["anchors", "memory", "shift", "base"]
        self.log_gain = nn.ParameterDict({n: nn.Parameter(torch.zeros(()), requires_grad=term_gains) for n in names})

    # -------------------------------------------------------------- memory
    @property
    def slot_filled(self) -> torch.Tensor:
        return self.mem_count > 0

    def has_memory(self) -> torch.Tensor:
        if not self.use_memory:
            return torch.zeros(self.C, dtype=torch.bool)
        return self.slot_filled.any(dim=1)

    def memory_read(self, s: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """[B, C, D] memory term. Attention over filled slots keyed by s, else a count-weighted mean."""
        filled = self.slot_filled                                          # [C,K]
        if s is not None and self.K > 1:
            sk = F.normalize(self.mem_keys, dim=-1)                        # [C,K,S]
            sim = torch.einsum("bs,cks->bck", F.normalize(s.float(), dim=-1), sk) / self.read_tau
            sim = sim.masked_fill(~filled.unsqueeze(0), -1e4)
            w = sim.softmax(-1) * filled.unsqueeze(0).float()              # rows of empty classes -> 0
            return torch.einsum("bck,ckd->bcd", w, self.memory)
        w = self.mem_count / self.mem_count.sum(1, keepdim=True).clamp(min=1)
        return (w.unsqueeze(-1) * self.memory).sum(1).unsqueeze(0).expand(batch_size, -1, -1)

    @torch.no_grad()
    def memory_write(self, class_means: torch.Tensor, class_pixels: torch.Tensor, key: torch.Tensor | None):
        """class_means [C, D], class_pixels [C] (0 = absent, untouched), key [S] = mean s of the writer."""
        key = torch.zeros(self.S, device=self.memory.device) if key is None else key.float()
        for c in torch.nonzero(class_pixels > 0).flatten().tolist():
            new, n_new = F.normalize(class_means[c].float(), dim=-1), float(class_pixels[c])
            empty = torch.nonzero(~self.slot_filled[c]).flatten()
            if len(empty):
                k = int(empty[0])
                self.memory[c, k], self.mem_keys[c, k], self.mem_count[c, k] = new, key, n_new
                continue
            # all slots used: merge into the slot whose domain key is closest
            sims = F.cosine_similarity(self.mem_keys[c], key.unsqueeze(0), dim=-1)
            if key.abs().sum() == 0:
                sims = self.memory[c] @ new
            k = int(sims.argmax())
            n_old = float(self.mem_count[c, k])
            a = max(n_new / (n_old + n_new), self.momentum)             # momentum = floor on the new weight
            self.memory[c, k] = F.normalize((1 - a) * self.memory[c, k] + a * new, dim=-1)
            self.mem_keys[c, k] = (1 - a) * self.mem_keys[c, k] + a * key
            self.mem_count[c, k] = n_old + n_new

    @torch.no_grad()
    def apply_drift(self, drift: torch.Tensor):
        """Semantic drift compensation: drift [C, K, D] added to filled slots."""
        filled = self.slot_filled.unsqueeze(-1).float()
        self.memory.copy_(F.normalize(self.memory + drift * filled, dim=-1) * filled)

    # -------------------------------------------------------------- forward
    def gain(self, name):
        return self.log_gain[name].exp()

    def forward(self, s: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        dev = self.log_gain["anchors"].device
        p = torch.zeros(batch_size, self.C, self.D, device=dev)
        if self.use_anchors:
            p = p + self.gain("anchors") * self.anchors.unsqueeze(0)
        if self.use_memory:
            p = p + self.gain("memory") * self.memory_read(s if self.use_shift else None, batch_size)
        if self.use_base:
            p = p + self.gain("base") * self.base.unsqueeze(0)
        if self.use_shift and s is not None:
            # (B s) as a plain matmul: [b, S] @ [S, Kb*D] -> [b, Kb, D]
            Kb = self.B.shape[0]
            Bs = (s.float() @ self.B.reshape(Kb * self.D, self.S).t()).reshape(-1, Kb, self.D)
            shift = torch.matmul(self.W_u.unsqueeze(0), Bs)              # [b, C, D]
            shift = shift * self.class_seen.view(1, -1, 1).float()       # never-seen classes: no shift
            p = p + self.gain("shift") * shift
        return F.normalize(p, dim=-1)                                   # [B,C,D]


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
        b, d, h, w = z.shape
        logits = torch.bmm(protos.float(), z.reshape(b, d, h * w))       # [B, C, HW]
        return logits.reshape(b, -1, h, w) / self.log_tau.exp().clamp(min=1e-3)


class LinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.cls = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, z, protos=None):
        return self.cls(z)
