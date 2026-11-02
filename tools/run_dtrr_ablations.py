"""Run every DTRR ablation in configs/dtrr/ with the same replay compute budget, then write a
results table (markdown + JSON).

The adaptive variants (full / intra / inter / threshold) run first; their mean pseudo-pixels per
step sets dtrr.replay.budget_pixels of the fixed-schedule variants (always / periodic), so all
replay variants spend the same replay compute on average.

    python tools/run_dtrr_ablations.py                                   # smoke data, CPU
    python tools/run_dtrr_ablations.py --base configs/bdd_idd.yaml --opts dataset.domains.bdd100k.root=... \
        dataset.domains.idd.root=...                                     # the real protocol
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from drift.config import load_config, save_config  # noqa: E402
from drift.engine import Trainer  # noqa: E402
from drift.utils import Tee, dump_json, make_run_dir, seed_everything  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["no_replay", "full", "intra_only", "inter_only", "threshold", "always", "periodic"]
LABEL = {"no_replay": "no replay", "always": "always-on replay", "periodic": "periodic replay",
         "intra_only": "DTRR intra only", "inter_only": "DTRR inter only", "full": "DTRR full (self-adjusting lam)",
         "threshold": "DTRR fixed thresholds (s*, eps*)"}


def flatten(d, prefix=""):
    out = []
    for k, v in d.items():
        if k == "_base_":
            continue
        if isinstance(v, dict):
            out += flatten(v, f"{prefix}{k}.")
        else:
            out.append(f"{prefix}{k}={json.dumps(v)}")
    return out


def overlay(name):
    """dtrr/model/experiment keys of configs/dtrr/<name>.yaml on top of configs/dtrr/base.yaml, as --opts."""
    keys = []
    for f in ("base", name):
        with open(os.path.join(ROOT, "configs", "dtrr", f"{f}.yaml"), encoding="utf-8") as fh:
            keys += flatten(yaml.safe_load(fh))
    return keys


def run(base, opts, name):
    cfg = load_config(base, overlay(name) + list(opts) + [f"experiment.name=ablation_{name}"])
    run_dir = make_run_dir(cfg)
    tee = Tee(os.path.join(run_dir, "train.log"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    try:
        seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
        tr = Trainer(cfg, run_dir)
        tr.fit()
    finally:
        sys.stdout = tee.stdout
    cs = tr.continual_summary()
    rep = tr.dtrr.report["tasks"] if tr.dtrr is not None else []
    steps = sum(r["steps"] for r in rep) or 1
    return {"name": name, "run_dir": run_dir, "R": np.asarray(tr.R).tolist(),
            "final_mIoU": [float(v) for v in tr.R[-1]], "avg_final": cs["avg_final"], "BWT": cs["BWT"],
            "avg_forgetting": cs["avg_forgetting"], "rare_IoU": cs.get("rare_class_IoU"),
            "replay_step_frac": sum(r["replay_steps"] for r in rep) / steps,
            "replay_px_per_step": sum(r["replay_px"] for r in rep) / steps,
            "replay_time_frac": float(np.mean([r["replay_compute_frac"] for r in rep])) if rep else 0.0}


def table(rows, tasks):
    head = ("| variant | " + " | ".join(f"T{j} mIoU" for j in range(tasks)) +
            " | avg mIoU | BWT | avg forgetting | rare-class IoU | replay steps | replay px/step | replay time |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        f = lambda v: "-" if v is None else f"{100 * v:.2f}"  # noqa: E731
        lines.append(f"| {LABEL[r['name']]} | " + " | ".join(f(v) for v in r["final_mIoU"]) +
                     f" | {f(r['avg_final'])} | {f(r['BWT'])} | {f(r['avg_forgetting'])} | {f(r['rare_IoU'])} | "
                     f"{100 * r['replay_step_frac']:.1f}% | {r['replay_px_per_step']:.0f} | "
                     f"{100 * r['replay_time_frac']:.1f}% |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "configs", "bdd_idd_smoke.yaml"))
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {ORDER}")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "dtrr_ablations"))
    args = ap.parse_args()

    rows = {}
    for name in [n for n in ORDER if not args.only or n in args.only]:
        extra = list(args.opts)
        if name in ("always", "periodic"):
            adaptive = [rows[k]["replay_px_per_step"] for k in ("full", "intra_only", "inter_only", "threshold") if k in rows]
            if adaptive:   # same average replay pixels per step as the adaptive variants
                # (periodic replays budget_pixels x period every period steps: the same average)
                extra.append(f"dtrr.replay.budget_pixels={max(1, round(float(np.mean(adaptive))))}")
        print(f"\n######## ablation {name}")
        rows[name] = run(args.base, extra, name)
    os.makedirs(args.out, exist_ok=True)
    T = len(next(iter(rows.values()))["final_mIoU"])
    md = table(list(rows.values()), T)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(f"Base config: `{os.path.relpath(args.base, ROOT)}` {' '.join(args.opts)}\n\n{md}\n")
    dump_json(rows, os.path.join(args.out, "results.json"))
    print("\n" + md + f"\n\n-> {os.path.join(args.out, 'results.md')}")


if __name__ == "__main__":
    main()
