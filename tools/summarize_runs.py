"""Aggregate runs into a markdown table (mean +- std over seeds).

    python tools/summarize_runs.py --root outputs/bench --out docs/benchmark_camvid.md

A run directory is <root>/<name>/<timestamp>/. Its seed comes from config.yaml; metrics come
from results.json (new code) or figures/test_metrics.json + history.json (older code).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import yaml


def _continual_from_R(R):
    R = np.array(R, dtype=float)
    d = int((~np.isnan(np.diag(R))).sum())
    if not d:
        return {}
    R = R[:d, :d]
    out = {"avg_final": float(np.nanmean(R[-1])),
           "avg_incremental": float(np.nanmean([np.nanmean(R[i, : i + 1]) for i in range(d)]))}
    if d > 1:
        out["BWT"] = float(np.nanmean([R[-1, j] - R[j, j] for j in range(d - 1)]))
        out["avg_forgetting"] = float(np.nanmean([np.nanmax(R[j:d - 1, j]) - R[-1, j] for j in range(d - 1)]))
    return out


def load_run(d):
    cfg = yaml.safe_load(open(os.path.join(d, "config.yaml"), encoding="utf-8"))
    seed = cfg["experiment"]["seed"]
    hist = json.load(open(os.path.join(d, "history.json"), encoding="utf-8"))
    tm_path = os.path.join(d, "figures", "test_metrics.json")
    if not os.path.isfile(tm_path):
        return None
    tm = json.load(open(tm_path, encoding="utf-8"))
    cont = hist.get("continual") or _continual_from_R(hist["R"])
    row = {"seed": seed, "test_mIoU": tm["mIoU"], "test_BIoU": tm["BIoU"], "test_Dice": tm["Dice"],
           "avg_seen_val_mIoU": cont.get("avg_final", np.nan), "avg_incremental": cont.get("avg_incremental", np.nan),
           "forgetting": cont.get("avg_forgetting", 0.0), "BWT": cont.get("BWT", 0.0),
           "class_forgetting": cont.get("avg_class_forgetting", np.nan)}
    ci = tm.get("mIoU_bootstrap") or {}
    row["ci"] = (ci.get("lo"), ci.get("hi")) if ci else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/bench")
    ap.add_argument("--out", default=None)
    ap.add_argument("--order", nargs="*", default=None)
    args = ap.parse_args()
    runs = defaultdict(list)
    for name in sorted(os.listdir(args.root)):
        for d in sorted(glob.glob(os.path.join(args.root, name, "*"))):
            if os.path.isfile(os.path.join(d, "config.yaml")):
                r = load_run(d)
                if r:
                    runs[name].append(r)
    keys = [("test_mIoU", "test mIoU"), ("test_BIoU", "test BIoU"), ("test_Dice", "test Dice"),
            ("avg_seen_val_mIoU", "avg seen-task val mIoU"), ("forgetting", "forgetting"), ("BWT", "BWT"),
            ("class_forgetting", "class-level forgetting")]
    names = args.order or list(runs)
    lines = ["| run | seeds | " + " | ".join(k[1] for k in keys) + " |",
             "|---|---|" + "---|" * len(keys)]
    for n in names:
        rs = runs.get(n)
        if not rs:
            continue
        cells = []
        for k, _ in keys:
            v = np.array([r[k] for r in rs], dtype=float) * 100
            if np.all(np.isnan(v)):
                cells.append("-")
            elif len(v) > 1:
                cells.append(f"{np.nanmean(v):.2f} ± {np.nanstd(v, ddof=1):.2f}")
            else:
                cells.append(f"{v[0]:.2f}")
        lines.append(f"| {n} | {len(rs)} | " + " | ".join(cells) + " |")
    ci_lines = []
    for n in names:
        for r in runs.get(n, []):
            if r["ci"] and r["ci"][0] is not None:
                ci_lines.append(f"- {n} seed {r['seed']}: test mIoU {100 * r['test_mIoU']:.2f}, "
                                f"95% image-bootstrap CI [{100 * r['ci'][0]:.2f}, {100 * r['ci'][1]:.2f}]")
    text = "\n".join(lines) + ("\n\n" + "\n".join(ci_lines) if ci_lines else "") + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
