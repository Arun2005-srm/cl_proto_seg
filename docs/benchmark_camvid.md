# CamVid benchmark: status and results so far

**Status: incomplete.** The benchmark was stopped early at the user's request. Only the
fixed code produced results. The original code was not run to completion on this
benchmark, so **there is no before/after comparison yet**. Nothing here is a comparison
with published state of the art.

## Setup

- Data: CamVid (SegNet-Tutorial version, 360x480, 11 classes, 701 frames, void = ignore).
- Domains: the 4 drive sequences. Tasks in order: 0016E5 → 0006R0 → 0001TP → Seq05VD.
- Splits: each sequence split 70 / 10 / 20 into train / val / test (490 / 88 / 123 images).
  The split is seeded and identical for the original and fixed code (verified file by file).
- Model: SegFormer MiT-B2, frozen, with LoRA (96 layers, rank 16), CLIP anchors.
  RTX 3050 Laptop GPU (4 GB), batch 4.

## Results so far (fixed code)

### Short run: all 4 tasks, 220 steps in total

| | value |
|---|---|
| Steps per task | 100 / 40 / 40 / 40 |
| Test mIoU after the last task (no TTA) | 67.56 |
| Test mIoU with TTA (scales 0.75 / 1 / 1.25 + flip) | **67.91**, 95% image-bootstrap CI [66.01, 69.37] |
| Test BIoU / Dice / pixel acc (TTA) | 53.88 / 78.44 / 92.19 |
| Avg seen-task val mIoU after T3 | 59.45 |
| Forgetting (per task / classes common to all tasks / class-level) | 1.50 / 1.50 / 4.55 |
| BWT | -1.33 |
| Shift estimator: task identified from `s` | 56.8% (chance 25%); std 1.49, so `s` has not collapsed |

Val mIoU per task after each stage (the accuracy matrix R):

| after | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| T0 | 69.0 | 42.0 | 51.9 | 54.4 |
| T1 | 69.5 | 52.5 | 50.0 | 56.0 |
| T2 | 69.4 | 50.2 | 59.9 | 56.4 |
| T3 | 69.7 | 48.9 | 58.9 | 60.4 |

T0 does not drop while T1 to T3 are learned (69.0 → 69.7).

### Full-budget run, seed 0: stopped during task 2

| | value |
|---|---|
| Task 1 (0016E5, 1007 steps): val mIoU, last epoch | 80.62 (BIoU 71.86, Dice 87.96) |
| Held-out test mIoU after task 1 (all 4 sequences) | 71.26 (BIoU 59.19) |
| Zero-shot val mIoU on not-yet-seen sequences | T1 51.9, T2 60.5, T3 65.2 |

## Not done yet

The full script is `tools/run_camvid_benchmark.sh`; `tools/summarize_runs.py` turns its
output into a mean ± std table.

- Original code vs fixed code, 3 seeds each
- Naive sequential fine-tuning (lower bound) and joint training (upper bound)
- Full encoder fine-tuning
- Ablations: hidden loss off, masked softmax, no drift compensation
