"""Phase 2 test report: metrics tables, JSON / CSV, and all test-time figures."""
from __future__ import annotations

import csv
import os

import numpy as np

from .engine import evaluate, make_loader
from .metrics import continual_metrics, format_class_table
from .utils import dump_json, get_palette
from .viz import plot_samples, plot_test_results, plot_tsne


def run_test(model, cfg, samples, device, out_dir, R=None, prefix="test") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    classes = list(cfg.dataset.classes)
    if cfg.model.inference.get("merge_lora", True) and hasattr(model, "merge_lora"):
        model.merge_lora(True)
    model.eval()
    loader = make_loader(cfg, samples, "test", False, cfg.eval.batch_size, cfg.eval.num_workers)
    res = evaluate(model, loader, cfg, device, with_loss=True)

    print(f"\n================ {prefix.upper()} ({len(samples)} images) ================")
    print(f"mIoU {100 * res['mIoU']:.2f} | IoU(fw) {100 * res['fwIoU']:.2f} | BIoU {100 * res['BIoU']:.2f} | "
          f"Dice {100 * res['Dice']:.2f} | pixel acc {100 * res['pixel_acc']:.2f} | mean acc {100 * res['mean_acc']:.2f}")
    print(format_class_table(res, classes, cols=("IoU", "F1", "BIoU", "Prec", "Rec")))
    if res["per_domain_mIoU"]:
        print("per-domain mIoU: " + ", ".join(f"{d} {100 * v:.2f}" for d, v in res["per_domain_mIoU"].items()))
    summary = {k: v for k, v in res.items() if k != "confusion"}
    summary["confusion"] = np.asarray(res["confusion"]).astype(np.int64)
    summary["classes"] = classes
    if R is not None and (~np.isnan(np.diag(R))).any():
        done = int((~np.isnan(np.diag(R))).sum())
        cl = continual_metrics(R[:done, :done])
        summary["continual"] = cl
        print("continual: " + ", ".join(f"{k} {100 * v:.2f}" for k, v in cl.items() if isinstance(v, float)))
    dump_json(summary, os.path.join(out_dir, f"{prefix}_metrics.json"))
    with open(os.path.join(out_dir, f"{prefix}_classwise.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "IoU", "F1", "BIoU", "precision", "recall", "support_pixels"])
        for i, c in enumerate(classes):
            w.writerow([c, res["IoU_per_class"][i], res["F1_per_class"][i], res["BIoU_per_class"][i],
                        res["precision_per_class"][i], res["recall_per_class"][i], res["support"][i]])
        w.writerow(["mean", res["mIoU"], res["Dice"], res["BIoU"], "", "", ""])

    dpi = cfg.viz.dpi
    files = plot_test_results(res, classes, get_palette(cfg.dataset), out_dir, dpi, prefix)
    files += plot_samples(model, loader, cfg, device, out_dir, cfg.viz.num_samples, dpi, prefix)
    if cfg.viz.tsne.enabled:
        files += plot_tsne(model, loader, cfg, device, out_dir, dpi, prefix)
    print(f"[{prefix}] figures: {', '.join(files)}")
    return summary
