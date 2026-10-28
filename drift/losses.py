"""DRIFT training loss:  CE + lambda_D Dice + lambda_h hidden-CE (+ optional shift variance term).

- Valid set: classes labelled in the current task. How the remaining ("absent")
  classes enter the softmax is `loss.absent_negatives`:
    masked   - excluded from the softmax (never negatives); train/test softmax differ
    detached - included as negatives, but through detached prototypes: pixels learn
               to stay away from them (calibrated all-class softmax at test time)
               while the absent prototypes themselves are never pushed (default)
    full     - plain all-class softmax
- Simulated absence: a set H of valid classes that already have memory is hidden,
  per iteration or for a whole task (`loss.hidden.scope`). Hidden pixels leave the
  main loss; a separate CE on them uses detached z and detached encoder statistics,
  so only the prototype path (g_phi head, W_u, B, gains) learns to carry a class
  the current city "does not have". Non-hidden prototypes are detached in that term.
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


def ce_loss(logits, target, weight, ignore_index, label_smoothing=0.0, ohem=None):
    """Cross-entropy that returns a graph-connected 0 when no pixel is labelled (no NaN),
    optionally with online hard example mining (keep pixels with p_true < thresh, at least min_kept)."""
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0
    if not ohem:
        return F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index,
                               label_smoothing=label_smoothing)
    pix = F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index,
                          label_smoothing=label_smoothing, reduction="none")[valid]
    with torch.no_grad():
        # clamp both ends: ignore pixels (255) must not index past the class dim; they are dropped by [valid]
        idx = target.clamp(0, logits.shape[1] - 1).unsqueeze(1)
        p_true = logits.float().softmax(1).gather(1, idx).squeeze(1)[valid]
        keep = p_true < ohem.get("thresh", 0.7)
        min_kept = min(int(ohem.get("min_kept", 100000)), pix.numel())
        if keep.sum() < min_kept:
            keep = torch.zeros_like(keep)
            keep[torch.topk(pix.detach(), min_kept).indices] = True
    return pix[keep].mean()


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

    def absent_mode(self):
        lc = self.lcfg
        if "absent_negatives" in lc:
            return lc.absent_negatives
        return "masked" if lc.get("masked_softmax", True) else "full"   # older configs

    # ------------------------------------------------------------------ hidden set
    def sample_hidden(self, candidates: torch.Tensor, memory_mask: torch.Tensor, present=None) -> torch.Tensor:
        """Draw H from candidate classes (valid, and with memory if require_memory)."""
        h = self.lcfg.hidden
        hidden = torch.zeros(self.C, dtype=torch.bool)
        if not h.enabled or random.random() >= h.prob:
            return hidden
        cand = candidates.cpu().clone()
        if present is not None:
            cand &= present.cpu()
        if h.require_memory:
            cand &= memory_mask.cpu()
        idx = torch.nonzero(cand).flatten().tolist()
        if not idx:
            return hidden
        k = min(int(h.num_hidden), len(idx), max(1, int(cand.sum()) - 1))
        for c in random.sample(idx, k):
            hidden[c] = True
        return hidden

    def _present(self, target):
        present = torch.zeros(self.C, dtype=torch.bool)
        u = torch.unique(target)
        present[u[(u != self.ign) & (u < self.C)].cpu()] = True
        return present

    # ------------------------------------------------------------------ forward
    def forward(self, model, out, target, allowed, memory_mask, task_hidden=None, candidates=None):
        """out: model forward dict. allowed: [C] bool valid set. memory_mask: [C] bool.
        task_hidden: [C] bool fixed hidden set when hidden.scope == task (all-False = not this iteration).
        candidates: [C] bool classes that may be hidden (default: the valid set)."""
        lc = self.lcfg
        size = target.shape[-2:]
        dev = target.device
        if lc.hidden.get("scope", "iteration") == "task" and task_hidden is not None:
            hidden = task_hidden.clone()
        else:
            cand = allowed if candidates is None else (allowed & candidates.to(allowed.device))
            hidden = self.sample_hidden(cand, memory_mask, self._present(target))
        mode = self.absent_mode()
        main_allowed = allowed & ~hidden
        hid_px = hidden.to(dev)[target.clamp(0, self.C - 1)] & (target != self.ign)
        out_px = (~allowed).to(dev)[target.clamp(0, self.C - 1)] & (target != self.ign)
        tgt_main = target.clone()
        tgt_main[hid_px | out_px] = self.ign

        protos = out.get("protos")
        if mode == "detached" and protos is not None and not main_allowed.all():
            neg = (~main_allowed).to(protos.device).view(1, -1, 1)
            mixed = torch.where(neg, protos.detach(), protos)
            logits = model.logits_from(out["z"], mixed)
        else:
            logits = out["logits"]
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False).float()
        if mode == "masked" or (mode == "detached" and protos is None):
            logits = mask_logits(logits, main_allowed)

        w = self.class_weights.to(logits.device)
        losses = {"ce": lc.ce.weight * ce_loss(logits, tgt_main, w, self.ign, lc.ce.label_smoothing,
                                                lc.ce.get("ohem"))}
        if lc.dice.enabled:
            losses["dice"] = lc.dice.weight * dice_loss(logits, tgt_main, main_allowed, self.ign)

        if hidden.any() and protos is not None and hid_px.any():
            z = out["z"].detach() if lc.hidden.detach_features else out["z"]
            stats = out.get("stats")
            ph = model.prototypes_from_stats(stats.detach() if stats is not None else None, z.shape[0])
            keep = hidden.to(ph.device).view(1, -1, 1)
            ph = torch.where(keep, ph, ph.detach())             # only hidden-class prototypes learn here
            hl = model.logits_from(z, ph, out_size=size).float()
            hl = mask_logits(hl, allowed | hidden) if mode == "masked" else hl
            tgt_h = torch.full_like(target, self.ign)
            tgt_h[hid_px] = target[hid_px]
            losses["hidden"] = lc.hidden.weight * ce_loss(hl, tgt_h, None, self.ign)

        sv = lc.get("shift_var", {}) or {}
        if sv.get("weight", 0) > 0 and out.get("s") is not None and out["s"].shape[0] > 1:
            std = out["s"].float().std(0)
            losses["shift_var"] = sv["weight"] * F.relu(sv.get("target_std", 1.0) - std).mean()

        total = sum(losses.values())
        with torch.no_grad():
            pred = logits.argmax(1)
            v = tgt_main != self.ign
            correct, count = (pred[v] == tgt_main[v]).sum().item(), v.sum().item()
        return total, {k: float(v.detach()) for k, v in losses.items()}, (correct, count), hidden
