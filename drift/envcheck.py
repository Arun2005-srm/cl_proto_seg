"""Start-up environment checks.

Newer PyTorch builds route some eager ops (e.g. bmm with an outer-product shape)
through Triton kernels in `torch._native`. Triton compiles a small C helper on
first use and needs the Python development headers (Python.h). Without them the
run dies mid-training with "fatal error: Python.h: No such file or directory".

check_environment() detects this before training and, depending on
experiment.triton_native_ops (auto | on | off), disables those Triton ops so
PyTorch falls back to its regular CUDA kernels.
"""
from __future__ import annotations

import os
import sysconfig


def python_headers_present() -> bool:
    inc = sysconfig.get_paths().get("include") or ""
    return os.path.isfile(os.path.join(inc, "Python.h"))


def _python_native():
    try:
        import torch.backends.python_native as pn  # torch >= 2.12
        return pn
    except Exception:  # noqa: BLE001
        return None


def disable_triton_native_ops() -> bool:
    pn = _python_native()
    if pn is None or not hasattr(pn, "triton"):
        return False
    try:
        pn.triton.enabled = False
        return True
    except Exception:  # noqa: BLE001
        return False


def check_environment(cfg) -> dict:
    mode = str(cfg.experiment.get("triton_native_ops", "auto")).lower()
    headers = python_headers_present()
    info = {"python_headers": headers, "triton_native_ops": "unchanged"}
    pn = _python_native()
    if pn is None:
        return info
    if mode == "off" or (mode == "auto" and not headers):
        if disable_triton_native_ops():
            info["triton_native_ops"] = "disabled"
            why = "requested" if mode == "off" else "Python.h not found"
            print(f"[env] Triton-backed torch native ops disabled ({why}); PyTorch uses its CUDA kernels instead.")
            if not headers:
                print("[env] To re-enable them install the Python headers, e.g. `sudo apt-get install python3-dev` "
                      "(python3.12-dev for 3.12), or use a conda/uv-managed Python, which ships them.")
    return info
