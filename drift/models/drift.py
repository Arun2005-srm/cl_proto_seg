"""DRIFT model assembled from switchable modules.

Every entry of cfg.model.modules is a boolean. A disabled module is never
constructed, so it has no parameters, nothing flows through it in the forward
pass and nothing reaches it in the backward pass - the ablation is exact.
Dependencies are resolved up front and reported (e.g. shift_library without a
shift_estimator has no input, so it is switched off with a warning).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import CosineHead, FusionDecoder, LinearHead, PrototypeGenerator, ShiftEstimator
from .encoders import build_encoder
from .lora import LoRALinear, inject_lora, linear_names, set_lora_merged


def resolve_modules(mcfg) -> dict[str, bool]:
    m = {k: bool(v) for k, v in mcfg.modules.items()}
    notes = []
    if not m.get("cosine_head", True):
        for k in ("anchors", "memory", "shift_estimator", "shift_library", "learnable_base"):
            if m.get(k):
                notes.append(f"{k} -> off (only used by the cosine/prototype head)")
                m[k] = False
    if m.get("shift_library") and not m.get("shift_estimator"):
        notes.append("shift_library -> off (needs shift_estimator for s)")
        m["shift_library"] = False
    if m.get("shift_estimator") and not m.get("shift_library"):
        notes.append("shift_estimator -> off (its output s is only consumed by shift_library)")
        m["shift_estimator"] = False
    if m.get("cosine_head") and not (m.get("anchors") or m.get("shift_library") or m.get("learnable_base")):
        notes.append("learnable_base -> on (memory alone gives all-zero prototypes on the first task)")
        m["learnable_base"] = True
    for n in notes:
        print(f"[model] module dependency: {n}")
    return m


def _load_clip(name: str):
    from transformers import CLIPModel

    try:
        return CLIPModel.from_pretrained(name, use_safetensors=True).eval()
    except Exception:  # noqa: BLE001  - repo has no safetensors: load the .bin with weights_only
        from huggingface_hub import hf_hub_download
        from transformers import CLIPConfig

        clip = CLIPModel(CLIPConfig.from_pretrained(name)).eval()
        sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        missing, unexpected = clip.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing:
            raise RuntimeError(f"CLIP checkpoint {name} is missing {len(missing)} weights, e.g. {missing[:3]}")
        return clip


def build_anchors(pcfg, classes, embed_dim: int, seed: int = 0) -> torch.Tensor:
    """a_c: frozen class anchors [C, D] from a file, CLIP text embeddings or a fixed random draw.
    A failed file/CLIP load is an error unless anchors.allow_fallback is true."""
    acfg = pcfg.anchors
    C = len(classes)
    g = torch.Generator().manual_seed(seed)
    src = acfg.source
    emb = None
    try:
        if src == "file":
            emb = torch.load(acfg.file, map_location="cpu", weights_only=True).float()
        elif src == "clip":
            from transformers import CLIPTokenizer

            tok = CLIPTokenizer.from_pretrained(acfg.clip_model)
            clip = _load_clip(acfg.clip_model)
            with torch.no_grad():
                t = tok([acfg.template.format(c) for c in classes], padding=True, return_tensors="pt")
                emb = clip.get_text_features(**t)
                if not torch.is_tensor(emb):        # transformers >= 5 returns a model output
                    emb = emb.pooler_output
                emb = emb.float()
            print(f"[model] anchors: CLIP text embeddings from {acfg.clip_model}")
        elif src != "random":
            raise KeyError(f"unknown anchors.source '{src}' (clip | file | random)")
    except Exception as e:  # noqa: BLE001
        if not acfg.get("allow_fallback", False):
            raise RuntimeError(f"could not build '{src}' anchors: {e}. Fix the source, or set "
                               f"model.prototypes.anchors.allow_fallback=true to use random anchors.") from e
        print(f"[model] WARNING: '{src}' anchors failed ({e}); using fixed random anchors (allow_fallback=true)")
        emb = None
    if emb is None:
        emb = torch.randn(C, embed_dim, generator=g)
    if emb.shape[0] != C:
        raise ValueError(f"anchors have {emb.shape[0]} rows, dataset has {C} classes")
    if emb.shape[1] != embed_dim:   # fixed (frozen) random projection to D
        proj = torch.randn(emb.shape[1], embed_dim, generator=g) / emb.shape[1] ** 0.5
        emb = F.normalize(emb, dim=-1) @ proj
        # CLIP text embeddings are highly correlated; remove the shared mean so classes are separable
        emb = emb - emb.mean(0, keepdim=True)
    return F.normalize(emb, dim=-1)


class DRIFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg.model
        self.cfg = cfg
        self.num_classes = len(cfg.dataset.classes)
        self.flags = resolve_modules(mcfg)
        D, S = int(mcfg.embed_dim), int(mcfg.shift_dim)

        self.encoder = build_encoder(mcfg.encoder)
        self.encoder_frozen = bool(mcfg.encoder.get("freeze", True))
        if self.encoder_frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.lora_layers: list[str] = []
        if self.flags.get("lora"):
            lc = mcfg.lora
            self.lora_layers = inject_lora(self.encoder, lc.targets, lc.rank, lc.alpha, lc.dropout)
            if not self.lora_layers:
                names = linear_names(self.encoder)
                raise RuntimeError(f"LoRA is enabled but no encoder Linear matches {list(lc.targets)}. "
                                   f"{len(names)} Linear layers exist, e.g. {names[:6]}. Adjust model.lora.targets "
                                   f"or set model.modules.lora=false.")
        ch = self.encoder.out_channels

        self.decoder = FusionDecoder(ch, D, mcfg.decoder.dropout, mcfg.decoder.get("norm", "gn"))

        self.shift_estimator = None
        if self.flags.get("shift_estimator"):
            sc = mcfg.shift_estimator
            self.shift_estimator = ShiftEstimator(ch, S, sc.hidden, sc.use_std, sc.get("detach_input", True))

        self.prototypes = None
        if self.flags.get("cosine_head"):
            pc = mcfg.prototypes
            anchors = build_anchors(pc, cfg.dataset.classes, D, cfg.experiment.seed) if self.flags.get("anchors") else None
            mc = pc.memory
            self.prototypes = PrototypeGenerator(
                self.num_classes, D, S, use_anchors=self.flags.get("anchors", False),
                use_memory=self.flags.get("memory", False), use_shift=self.flags.get("shift_library", False),
                use_base=self.flags.get("learnable_base", False), anchors=anchors,
                num_slots=mc.num_slots, momentum=mc.momentum, num_bases=pc.shift_library.num_bases,
                read_tau=mc.get("read_tau", 0.1), term_gains=pc.get("term_gains", True))
            self.head = CosineHead(mcfg.head.tau, mcfg.head.learnable_tau)
        else:
            self.head = LinearHead(D, self.num_classes)

    # ------------------------------------------------------------------ forward
    def forward(self, x, out_size=None):
        feats = self.encoder(x)
        z = self.decoder(feats)
        s, stats = self.shift_estimator(feats) if self.shift_estimator is not None else (None, None)
        protos = self.prototypes(s, x.shape[0]) if self.prototypes is not None else None
        logits = self.head(z, protos)
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return {"logits": logits, "z": z, "s": s, "stats": stats, "protos": protos}

    def logits_from(self, z, protos, out_size=None):
        logits = self.head(z, protos)
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits

    def prototypes_from_stats(self, stats, batch_size):
        """Prototypes from (detached) encoder statistics: gradients reach g_phi's head,
        W_u, B and the gains, never the encoder or LoRA."""
        s = self.shift_estimator.head(stats) if (self.shift_estimator is not None and stats is not None) else None
        return self.prototypes(s, batch_size)

    # ------------------------------------------------------------------ helpers
    @property
    def uses_memory(self) -> bool:
        return self.prototypes is not None and self.prototypes.use_memory

    def memory_mask(self) -> torch.Tensor:
        if not self.uses_memory:
            return torch.zeros(self.num_classes, dtype=torch.bool)
        return self.prototypes.has_memory().cpu()

    def mark_seen(self, classes_mask: torch.Tensor):
        if self.prototypes is not None:
            self.prototypes.class_seen |= classes_mask.to(self.prototypes.class_seen.device)

    def merge_lora(self, merged: bool = True):
        set_lora_merged(self.encoder, merged)

    def train(self, mode: bool = True):
        super().train(mode)
        # a frozen encoder runs in eval mode: no stochastic depth / dropout, fixed norm statistics.
        # LoRA dropout lives inside the LoRALinear wrappers and follows `mode`.
        if mode and self.encoder_frozen:
            self.encoder.eval()
            for m in self.encoder.modules():
                if isinstance(m, LoRALinear):
                    m.train(mode)
        return self

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named groups; 1-d params (norms, biases, gains, tau) and the prototype path go to
        '<name>_no_wd' so weight decay does not shrink them."""
        groups: dict[str, list[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            g = "lora" if "lora_" in name else name.split(".")[0]
            if p.ndim <= 1 or g == "prototypes":
                g += "_no_wd"
            groups.setdefault(g, []).append(p)
        return groups

    def checkpoint_state(self) -> dict:
        """State dict without frozen encoder weights (they come from `pretrained`)."""
        frozen = {f"encoder.{n}" for n, p in self.encoder.named_parameters()
                  if not p.requires_grad and "lora_" not in n} if self.encoder_frozen else set()
        return {k: v for k, v in self.state_dict().items() if k not in frozen}

    def module_report(self) -> str:
        rows = [("encoder", self.encoder, True), ("lora", None, self.flags.get("lora")),
                ("decoder", self.decoder, True), ("shift_estimator", self.shift_estimator, self.flags.get("shift_estimator")),
                ("prototypes", self.prototypes, self.flags.get("cosine_head")), ("head", self.head, True)]
        lora_params = [p for m in self.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)]
        lines = [f"{'module':18s} {'on':>4s} {'params':>12s} {'trainable':>12s}", "-" * 50]
        for name, mod, on in rows:
            if name == "lora":
                tot = sum(p.numel() for p in lora_params)
                tr = sum(p.numel() for p in lora_params if p.requires_grad)
            elif mod is None:
                tot = tr = 0
            else:
                ps = [p for n, p in mod.named_parameters() if "lora_" not in n]
                tot = sum(p.numel() for p in ps)
                tr = sum(p.numel() for p in ps if p.requires_grad)
            lines.append(f"{name:18s} {'yes' if on else 'no':>4s} {tot:12,d} {tr:12,d}")
        if self.prototypes is not None:
            p = self.prototypes
            parts = [n for n, f in (("anchors", p.use_anchors), ("memory", p.use_memory),
                                    ("shift_library", p.use_shift), ("learnable_base", p.use_base)) if f]
            lines.append(f"prototype terms: {' + '.join(parts)}   memory slots K={p.K}   "
                         f"head: cosine (tau={self.head.log_tau.exp().item():.3f})")
        else:
            lines.append("head: linear 1x1 classifier")
        lines.append(f"LoRA layers: {len(self.lora_layers)}   encoder: {'frozen' if self.encoder_frozen else 'trainable'}")
        return "\n".join(lines)
