"""LoRA training audit, run for every task: PASS/FAIL checklist at task start, then proofs that
gradient reaches the adapters (first backward), that they move (||W_lora_t - W_lora_start|| > 0
after `check_step` steps) and that they change the output (logits with vs without adapters on
one fixed batch). Any failure raises LoRAAuditError. Read-only: no RNG, no parameter change.
"""
from __future__ import annotations

import math

import torch

from .models.lora import lora_disabled, lora_layers


class LoRAAuditError(RuntimeError):
    pass


def trainable_split(model) -> dict[str, int]:
    out = {"lora": 0, "decoder": 0, "head": 0, "shift": 0, "other": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            k = "lora"
        elif n.startswith("decoder."):
            k = "decoder"
        elif n.startswith("head."):
            k = "head"
        elif n.startswith("shift_estimator.") or n.startswith("prototypes."):
            k = "shift"              # g_phi, W_u, B, term gains (the shift / prototype path)
        else:
            k = "other"
        out[k] += p.numel()
    return out


class LoRAAudit:
    def __init__(self, model, cfg):
        self.model = model
        self.enabled = bool(model.flags.get("lora")) and bool(cfg.get("audit", {}).get("lora", True))
        self.check_step = int(cfg.get("audit", {}).get("check_step", 50))

    def _params(self):
        return [p for m in lora_layers(self.model.encoder) for p in (m.lora_A, m.lora_B)]

    def _delta(self):
        return torch.cat([m.delta().detach().float().flatten() for m in lora_layers(self.model.encoder)])

    def _rows(self, rows, title):
        print(title)
        for ok, msg in rows:
            print(f"    [{'PASS' if ok is True else 'FAIL' if ok is False else 'info'}] {msg}")
        bad = [m for ok, m in rows if ok is False]
        if bad:
            raise LoRAAuditError("LoRA audit failed: " + " | ".join(bad))

    # ---------------------------------------------------------------- task start
    def task_start(self, t: int, opt, scaler, reset_per_task: bool):
        self.t, self.first_backward_done, self.step_checked = t, False, False
        self.batch = None
        split = trainable_split(self.model)
        tot = sum(split.values())
        print(f"[audit] T{t} trainable params: {tot:,d} = " + ", ".join(f"{k} {v:,d}" for k, v in split.items()))
        if not self.enabled:
            print(f"[audit] T{t} LoRA disabled in this config (model.modules.lora=false or audit.lora=false)")
            return
        layers = lora_layers(self.model.encoder)
        params = self._params()
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        missing = [p for p in params if id(p) not in in_opt]
        frozen = [p for p in params if not p.requires_grad]
        lora_groups = [g for g in opt.param_groups if g.get("name", "").startswith("lora")]
        lrs = sorted({g.get("initial_lr", g["lr"]) for g in lora_groups})
        now = sorted({g["lr"] for g in lora_groups})
        n_train = sum(p.numel() for p in params if p.requires_grad)
        B_norm = float(torch.cat([m.lora_B.detach().flatten() for m in layers]).norm()) if layers else 0.0
        merged = [m for m in layers if m.merged]
        gc = [n for n, m in self.model.encoder.named_modules()
              if getattr(m, "gradient_checkpointing", False) or getattr(m, "is_gradient_checkpointing", False)]
        rows = [
            (len(layers) > 0, f"LoRA tensors: {len(params)} in {len(layers)} layers"),
            (n_train > 0 and not frozen, f"trainable LoRA params: {n_train:,d} ({len(frozen)} tensors frozen)"),
            (not missing, f"LoRA params missing from optimizer.param_groups: {len(missing)}"),
            (bool(lora_groups) and all(lr > 0 for lr in lrs),
             f"LoRA param group lr: {', '.join(f'{lr:.2e}' for lr in lrs) or 'none'} "
             f"(now {', '.join(f'{lr:.2e}' for lr in now)} after warm-up/schedule; "
             f"groups {', '.join(g['name'] for g in lora_groups) or '-'})"),
            (not merged, f"adapters unmerged during training: {len(layers) - len(merged)}/{len(layers)}"),
            (torch.is_grad_enabled(), "autograd enabled at task start (no surrounding torch.no_grad)"),
            (not gc, f"gradient checkpointing in encoder: {', '.join(gc) or 'off'}"),
            (None, f"AMP GradScaler: {'on' if scaler.is_enabled() else 'off'}"
                   f"{f' (scale {scaler.get_scale():.0f})' if scaler.is_enabled() else ''}"),
        ]
        if t == 0 or reset_per_task:
            rows.append((B_norm == 0.0, f"adapter state: fresh for T{t} (||B|| = {B_norm:.3e}, must be 0; "
                                        f"{self.model.lora_merges} earlier adapter(s) merged into the base weights)"))
        else:
            rows.append((None, f"adapter state: continued from T{t - 1} (not merged; ||scale B A|| = "
                               f"{float(self._delta().norm()):.3e}); model.lora.reset_per_task=true would merge it "
                               f"and attach a fresh one"))
        self._rows(rows, f"[audit] T{t} LoRA checklist (start of task)")
        self.delta0 = self._delta().clone()

    # ---------------------------------------------------------------- during the task
    def capture_batch(self, x):
        if self.enabled and self.batch is None:
            self.batch = x[: min(2, x.shape[0])].detach().cpu()

    def check_forward(self, out):
        """First forward: the graph reaches the encoder output (it is not detached / under no_grad)."""
        if not self.enabled or self.first_backward_done:
            return
        enc_out = out["feats"][-1]
        if not enc_out.requires_grad:
            raise LoRAAuditError(f"T{self.t}: encoder output does not require grad (encoder detached or run "
                                 f"under no_grad) - LoRA cannot learn")

    def after_backward(self, scaler):
        if not self.enabled or self.first_backward_done:
            return
        self.first_backward_done = True
        grads = [p.grad for p in self._params()]
        none = sum(g is None for g in grads)
        scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        sq = sum(float(g.float().pow(2).sum()) for g in grads if g is not None)
        gn = math.sqrt(sq) / scale
        ok = none == 0 and math.isfinite(gn) and gn > 0
        self._rows([(ok, f"first backward: LoRA grad norm {gn:.3e} (finite and > 0; {none} tensors without grad)")],
                   f"[audit] T{self.t} gradient flow")

    @torch.no_grad()
    def adapter_effect(self) -> float:
        m = self.model
        was = m.training
        dev = next(m.parameters()).device
        m.eval()
        try:
            x = self.batch.to(dev)
            on = m(x)["logits"].float()
            with lora_disabled(m.encoder):
                off = m(x)["logits"].float()
        finally:
            m.train(was)
        return float((on - off).abs().mean())

    def on_step(self, step: int):
        if self.enabled and not self.step_checked and step >= self.check_step:
            self._step_check(step)

    def _step_check(self, step):
        self.step_checked = True
        moved = float((self._delta() - self.delta0).norm())
        eff = self.adapter_effect()
        self._rows([(moved > 0 and math.isfinite(moved), f"step {step}: ||W_lora_t - W_lora_start|| = {moved:.3e} (> 0)"),
                    (eff > 0 and math.isfinite(eff),
                     f"step {step}: mean |logits(adapter on) - logits(adapter off)| on a fixed batch = {eff:.3e} (> 0)")],
                   f"[audit] T{self.t} LoRA is learning")

    def task_end(self, steps: int):
        if not self.enabled:
            return
        if not self.first_backward_done:
            raise LoRAAuditError(f"T{self.t}: no backward pass reached the audit")
        if not self.step_checked:      # task shorter than check_step
            self._step_check(steps)
