"""DRIFT training loss:  validCE + lambda_D validDice + lambda_h hidden-CE.

- validCE / validDice: masked softmax over the valid set (classes labelled in the
  current task), so absent classes are never negatives.
- simulated absence: each iteration a random set H of valid classes is hidden.
  Their pixels leave the main loss; a separate CE on those pixels is computed
  from detached features, so only the prototype path (memory read + shift
  U_c B s) learns to carry a class that the current city "does not have".
"""
from __future__ import annotations

import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


def _load_class_weights(spec, C):
    if spec is None:
        return None
    if isinstance(spec, str):
        key = "median_frequency"
        path = spec
        if "::" in spec:
            path, key = spec.split("::", 1)
        with open(path, "r", encoding="utf-8") as f:
            w = json.load(f)[key]
    else:
        w = list(spec)
    if len(w) != C:
        raise ValueError(f"class_weights has {len(w)} entries, expected {C}")
    return torch.tensor(w, dtype=torch.float32)


def mask_logits(logits, allowed: torch.Tensor):
    """allowed: [C] bool. Disallowed classes get a large negative logit."""
    if allowed.all():
        return logits
    return logits.masked_fill(~allowed.view(1, -1, 1, 1).to(logits.device), NEG)


def dice_loss(logits, target, allowed, ignore_index, eps: float = 1.0):
    C = logits.shape[1]
    prob = logits.float().softmax(1)
    valid = (target != ignore_index).unsqueeze(1).float()
    onehot = F.one_hot(target.clamp(0, C - 1), C).permute(0, 3, 1, 2).float() * valid
    prob = prob * valid
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    sel = allowed.to(logits.device) & (onehot.sum((0, 2, 3)) > 0)
    return dice[sel].mean() if sel.any() else logits.sum() * 0


class DRIFTLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.C = len(cfg.dataset.classes)
        self.ign = cfg.dataset.ignore_index
        self.set_cfg(cfg.loss)

    def set_cfg(self, lcfg):
        self.lcfg = lcfg
        w = _load_class_weights(lcfg.ce.get("class_weights"), self.C)
        self.register_buffer("class_weights", w if w is not None else torch.ones(self.C), persistent=False)

    def _sample_hidden(self, allowed, target, memory_mask):
        h = self.lcfg.hidden
        hidden = torch.zeros(self.C, dtype=torch.bool)
        if not h.enabled or random.random() >= h.prob:
            return hidden
        present = torch.zeros(self.C, dtype=torch.bool)
        u = torch.unique(target)
        u = u[(u != self.ign) & (u < self.C)].cpu()
        present[u] = True
        cand = present & allowed.cpu()
        if h.require_memory:
            cand &= memory_mask.cpu()
        idx = torch.nonzero(cand).flatten().tolist()
        if not idx:
            return hidden
        k = min(int(h.num_hidden), len(idx), max(1, int(cand.sum()) - 1))
        for c in random.sample(idx, k):
            hidden[c] = True
        return hidden

    def forward(self, model, out, target, allowed, memory_mask):
        """out: model forward dict. allowed: [C] bool valid set. memory_mask: [C] bool."""
        lc = self.lcfg
        size = target.shape[-2:]
        logits = F.interpolate(out["logits"], size=size, mode="bilinear", align_corners=False).float()
        hidden = self._sample_hidden(allowed, target, memory_mask)
        main_allowed = (allowed & ~hidden) if lc.masked_softmax else ~hidden
        tgt_main = target.clone()
        if hidden.any():
            tgt_main[hidden.to(target.device)[target.clamp(0, self.C - 1)] & (target != self.ign)] = self.ign
        if lc.masked_softmax:   # labelled classes outside the valid set (valid_set=batch/task) are ignored
            outside = ~allowed.to(target.device)
            tgt_main[outside[target.clamp(0, self.C - 1)] & (target != self.ign)] = self.ign
        ml = mask_logits(logits, main_allowed)
        w = self.class_weights.to(logits.device)
        losses = {}
        ce = F.cross_entropy(ml, tgt_main, weight=w, ignore_index=self.ign, label_smoothing=lc.ce.label_smoothing)
        losses["ce"] = lc.ce.weight * torch.nan_to_num(ce)
        if lc.dice.enabled:
            losses["dice"] = lc.dice.weight * dice_loss(ml, tgt_main, main_allowed, self.ign)
        if hidden.any() and out.get("protos") is not None:
            z = out["z"].detach() if lc.hidden.detach_features else out["z"]
            hl = model.logits_from(z, out["protos"], out_size=size).float()
            hl = mask_logits(hl, allowed | hidden) if lc.masked_softmax else hl
            tgt_h = torch.full_like(target, self.ign)
            hm = hidden.to(target.device)[target.clamp(0, self.C - 1)] & (target != self.ign)
            tgt_h[hm] = target[hm]
            if hm.any():
                losses["hidden"] = lc.hidden.weight * F.cross_entropy(hl, tgt_h, ignore_index=self.ign)
        total = sum(losses.values())
        with torch.no_grad():
            pred = ml.argmax(1)
            v = tgt_main != self.ign
            correct, count = (pred[v] == tgt_main[v]).sum().item(), v.sum().item()
        return total, {k: float(v.detach()) for k, v in losses.items()}, (correct, count), hidden
