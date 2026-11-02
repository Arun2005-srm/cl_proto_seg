"""DTRR + LoRA proof of life: task 1 then task 2 on the tiny local BDD100K -> IDD subset (CPU),
then a PASS/FAIL checklist. Exits non-zero if any item fails.

Two runs:
  normal      configs/bdd_idd_smoke.yaml with DTRR on
  aggressive  the same with a 50x LoRA learning rate: drift must make the controller fire

    python tools/run_dtrr_smoke.py
    python tools/run_dtrr_smoke.py --config configs/bdd_idd_smoke.yaml --opts dtrr.interval=20
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from drift.config import load_config, save_config  # noqa: E402
from drift.engine import Trainer  # noqa: E402
from drift.utils import Tee, make_run_dir, seed_everything  # noqa: E402

COMMON = ["dtrr.enabled=true", "dtrr.interval=10"]
AGGRESSIVE = ["train.optimizer.lr_mult.lora=50.0"]


def train(config, opts, name):
    cfg = load_config(config, COMMON + list(opts) + [f"experiment.name={name}"])
    run_dir = make_run_dir(cfg)
    tee = Tee(os.path.join(run_dir, "train.log"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"\n######## {name}: {run_dir}")
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    trainer, err = None, None
    try:
        trainer = Trainer(cfg, run_dir)
        trainer.fit()
    except Exception as e:  # noqa: BLE001  - reported as a failed checklist item
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        sys.stdout = tee.stdout
    return trainer, run_dir, err


def checklist(trainer, run_dir, err, require_replay: bool):
    items = []

    def item(ok, msg, why=""):
        items.append((bool(ok), msg + ("\n          -> " + why if why else "")))

    if err is not None:
        item(False, "training ran to the end", err)
    rep = trainer.dtrr.report["tasks"] if (trainer is not None and trainer.dtrr is not None) else []
    t0 = rep[0] if rep else {}
    t1 = rep[1] if len(rep) > 1 else {}
    curves = [r.get("eps_curve", []) for r in rep]
    item(len(rep) == 2 and all(r.get("probe_pixels", 0) > 0 for r in rep) and
         all(len(c) > 1 and min(c) > 0 and np.std(c) > 0 for c in curves),
         "probe set built, eps_hat > 0 and changing over steps",
         "; ".join(f"T{i}: {r.get('probe_pixels', 0)} px, eps {', '.join(f'{v:.3f}' for v in c)}"
                   for i, (r, c) in enumerate(zip(rep, curves))))
    item(len(rep) == 2 and all(r.get("self_test") for r in rep), "drift self-test passed (perturb > 0, restore == 0)")
    stats = [os.path.join(run_dir, f"class_stats_T{i}.json") for i in range(2)]
    item(all(os.path.isfile(p) for p in stats) and all(r.get("minor") for r in rep),
         "class_stats.json written, minor classes identified",
         "; ".join(f"T{i} minor: {', '.join(r.get('minor', [])) or 'none'}" for i, r in enumerate(rep)))
    item(len(rep) == 2 and all(all(v > 0 for v in r.get("pushes", {}).values()) and r.get("pushes") for r in rep),
         "every minor class present had queue pushes > 0",
         "; ".join(f"T{i}: " + ", ".join(f"{k} {v}" for k, v in r.get("pushes", {}).items()) for i, r in enumerate(rep)))
    item(len(rep) == 2 and all(r.get("reload_ok") and os.path.isfile(r.get("audit_file", "")) for r in rep),
         "memory_audit.json written and reload check passed", t1.get("audit_file", t0.get("audit_file", "")))
    prev = set(t1.get("prev_classes_present") or [])
    item(bool(prev) and set(t0.get("sketched", [])) <= prev,
         "previous-task classes present in memory at task 2 start (hash-checked)", ", ".join(sorted(prev)))
    lam = t1.get("lam_max", {})
    why = (", ".join(f"{k} {v:.2f}" for k, v in lam.items()) if lam else
           (f"never: max F_hat {t1.get('last_F_hat_max', 0):.4f} stayed below the budget rho {t1.get('rho', 0):.3f} "
            f"(eps below budget)" if t1.get("controller_updates") else "never: controller not updated (bug)"))
    item(bool(lam), "controller dual variables lam moved off 0 for >= 1 class during task 2", why)
    item(t1.get("replay_steps", 0) > 0 if require_replay else True,
         "replay batches were actually drawn (count > 0)" + ("" if require_replay else " [not required]"),
         f"{t1.get('replay_steps', 0)} replay steps, {t1.get('replay_px', 0)} pseudo pixels in task 2")
    item(trainer is not None and trainer.dtrr is not None and trainer.dtrr.ctrl.eps_updates > 0,
         "eps_hat consumed by the controller", f"{trainer.dtrr.ctrl.eps_updates if trainer and trainer.dtrr else 0} updates")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bdd_idd_smoke.yaml")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()

    results = {}
    for name, extra, need in (("dtrr_smoke_normal", [], True), ("dtrr_smoke_aggressive", AGGRESSIVE, True)):
        tr, run_dir, err = train(args.config, args.opts + extra, name)
        results[name] = (checklist(tr, run_dir, err, need), tr)

    failed = False
    for name, (items, tr) in results.items():
        print(f"\n================ DTRR proof-of-life checklist: {name} ================")
        for ok, msg in items:
            print(f"  [{'x' if ok else ' '}] {'PASS' if ok else 'FAIL'}  {msg}")
            failed |= not ok
    n, a = (results[k][1] for k in ("dtrr_smoke_normal", "dtrr_smoke_aggressive"))
    if n is not None and a is not None and n.dtrr is not None and a.dtrr is not None:
        rn, ra = n.dtrr.report["tasks"][-1], a.dtrr.report["tasks"][-1]
        print(f"\nnormal vs aggressive LoRA lr (task 2): eps_hat end {rn['eps_curve'][-1]:.3f} vs {ra['eps_curve'][-1]:.3f}, "
              f"replay steps {rn['replay_steps']} vs {ra['replay_steps']}, pseudo px {rn['replay_px']} vs {ra['replay_px']}, "
              f"sum of max lam {sum(rn['lam_max'].values()):.2f} vs {sum(ra['lam_max'].values()):.2f}")
    print("\nRESULT:", "FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
