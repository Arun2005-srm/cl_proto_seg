"""Train DRIFT over the configured task sequence, then test and plot everything.

    python tools/train.py --config configs/cityscapes.yaml
    python tools/train.py --config configs/cityscapes.yaml --opts model.modules.memory=false experiment.name=no_memory
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift.config import load_config, save_config  # noqa: E402
from drift.engine import Trainer  # noqa: E402
from drift.testing import run_test  # noqa: E402
from drift.utils import Tee, dump_json, make_run_dir, seed_everything  # noqa: E402
from drift.envcheck import check_environment  # noqa: E402
from drift.viz import plot_class_forgetting, plot_forgetting, plot_training_curves  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4")
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    check_environment(cfg)
    run_dir = make_run_dir(cfg)
    Tee(os.path.join(run_dir, "train.log"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"[run] {run_dir}")
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)

    trainer = Trainer(cfg, run_dir)
    trainer.fit()

    fig_dir = os.path.join(run_dir, "figures")
    files = plot_training_curves(trainer.history, fig_dir, cfg.viz.dpi)
    files += plot_forgetting(trainer.R, trainer.tasks, trainer.seen_curve, trainer.test_curve,
                             trainer.history, fig_dir, cfg.viz.dpi)
    files += plot_class_forgetting(trainer.Rc, trainer.classes, fig_dir, cfg.viz.dpi)
    print(f"[viz] training figures: {', '.join(files)}")
    if not args.no_test and "test" in trainer.splits:
        summary = run_test(trainer.model, cfg, trainer.splits["test"], trainer.device, fig_dir, R=trainer.R,
                           continual=trainer.continual_summary())
        dump_json({"test": {k: summary[k] for k in ("mIoU", "fwIoU", "BIoU", "Dice", "pixel_acc", "mIoU_bootstrap")},
                   "continual": trainer.continual_summary(), "shift_diagnostics": trainer.diag,
                   "test_curve": trainer.test_curve}, os.path.join(run_dir, "results.json"))
    print(f"[run] done -> {run_dir}")


if __name__ == "__main__":
    main()
