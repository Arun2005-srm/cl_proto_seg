"""Run EDA for a dataset config.

    python tools/eda.py --config configs/cityscapes.yaml            # -> outputs/eda/cityscapes/
    python tools/eda.py --config configs/cityscapes.yaml --opts eda.max_images=500
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift.config import load_config  # noqa: E402
from drift.eda import run_eda  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    out = args.out or os.path.join(cfg.experiment.output_dir, "eda", cfg.dataset.name)
    run_eda(cfg, out, args.splits, args.workers)


if __name__ == "__main__":
    main()
