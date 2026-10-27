"""Multi-scale encoders. Each returns a list of 4 feature maps (strides 4, 8, 16, 32)
and exposes `out_channels`. Register new backbones in ENCODERS."""
from __future__ import annotations

import torch
import torch.nn as nn


class SegFormerEncoder(nn.Module):
    """Mix Transformer (MiT) from HF transformers, e.g. nvidia/mit-b2."""

    def __init__(self, pretrained: str = "nvidia/mit-b2", **_):
        super().__init__()
        from transformers import SegformerModel

        try:
            self.net = SegformerModel.from_pretrained(pretrained)
        except ValueError as e:
            if "torch.load" not in str(e):
                raise
            # transformers refuses .bin checkpoints on torch < 2.6; load them with weights_only=True instead
            self.net = _segformer_from_bin(pretrained)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"could not load SegFormer weights '{pretrained}'. Download them once with internet "
                f"access or point model.encoder.pretrained at a local folder. ({e})") from e
        self.out_channels = list(self.net.config.hidden_sizes)

    def forward(self, x):
        out = self.net(pixel_values=x, output_hidden_states=True, return_dict=True)
        return list(out.hidden_states)


def _segformer_from_bin(pretrained: str):
    import os

    from transformers import SegformerConfig, SegformerModel

    if os.path.isdir(pretrained):
        path = os.path.join(pretrained, "pytorch_model.bin")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(pretrained, "pytorch_model.bin")
    net = SegformerModel(SegformerConfig.from_pretrained(pretrained))
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k[len("segformer."):]: v for k, v in sd.items() if k.startswith("segformer.")} or sd
    missing, _ = net.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"{len(missing)} SegFormer weights missing from {path}, e.g. {missing[:3]}")
    print(f"[model] encoder: loaded {pretrained} with torch.load(weights_only=True)")
    return net


class _TinyAttention(nn.Module):
    """Single-head self attention with query/key/value Linears so LoRA can target it."""

    def __init__(self, c: int):
        super().__init__()
        self.query, self.key, self.value = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        self.proj = nn.Linear(c, c)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x.flatten(2).transpose(1, 2))
        q, k, v = self.query(t), self.key(t), self.value(t)
        a = torch.softmax(q @ k.transpose(1, 2) / c ** 0.5, dim=-1)
        return x + self.proj(a @ v).transpose(1, 2).reshape(b, c, h, w)


class SimpleCNNEncoder(nn.Module):
    """Small randomly initialised backbone for smoke tests and CPU debugging."""

    def __init__(self, channels=(32, 64, 128, 192), **_):
        super().__init__()
        self.out_channels = list(channels)
        strides = [4, 2, 2, 2]
        layers, cin = [], 3
        for i, (c, s) in enumerate(zip(channels, strides)):
            block = [nn.Conv2d(cin, c, 3 if s == 2 else 7, s, 1 if s == 2 else 3, bias=False),
                     nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                     nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            if i >= 2:
                block.append(_TinyAttention(c))
            layers.append(nn.Sequential(*block))
            cin = c
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        feats = []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        return feats


ENCODERS = {"segformer": SegFormerEncoder, "simple_cnn": SimpleCNNEncoder}


def build_encoder(ecfg) -> nn.Module:
    if ecfg.name not in ENCODERS:
        raise KeyError(f"unknown encoder '{ecfg.name}'. Available: {sorted(ENCODERS)}")
    kwargs = {k: v for k, v in ecfg.items() if k not in ("name", "freeze")}
    return ENCODERS[ecfg.name](**kwargs)
