"""Phase 2 evaluation of a checkpoint (everything frozen, no domain id, no memory write).

    python tools/test.py --config configs/cityscapes.yaml --ckpt outputs/.../task5.pt
    python tools/test.py --config configs/cityscapes.yaml --ckpt ... --split val     # carved val split
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from drift.config import load_config  # noqa: E402
from drift.data.datasets import build_splits  # noqa: E402
from drift.engine import load_model  # noqa: E402
from drift.testing import run_test  # noqa: E402
from drift.envcheck import check_environment  # noqa: E402
from drift.utils import Tee, seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None, help="output dir (default: next to the checkpoint)")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    check_environment(cfg)
    seed_everything(cfg.experiment.seed)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"eval_{args.split}")
    os.makedirs(out, exist_ok=True)
    Tee(os.path.join(out, "test.log"))
    dev = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    model, sd = load_model(cfg, args.ckpt, dev)
    print(model.module_report())
    splits = build_splits(cfg)
    run_test(model, cfg, splits[args.split], dev, out, R=sd.get("R"), prefix=args.split)


if __name__ == "__main__":
    main()
