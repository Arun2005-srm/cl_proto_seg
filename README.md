# DRIFT segmentation pipeline

Dataset-agnostic, config-driven pipeline for domain-incremental semantic
segmentation (train city by city, test on any city with no domain ID).
It implements the DRIFT two-phase design: a frozen SegFormer with LoRA, a fusion
decoder, a cosine head over prototypes `p_c = norm(a_c + m_c + U_c B s)`, a shift
estimator `g_phi`, class memory written at the end of each task, and simulated
class absence during training.

The risk-register review and the IDD run problems are answered item by item in
[docs/REVIEW_RESPONSE.md](docs/REVIEW_RESPONSE.md). Measured results are in
[docs/benchmark_camvid.md](docs/benchmark_camvid.md).

## Layout

```
configs/defaults.yaml        every option, with comments
configs/idd.yaml             IDD level-3 (26 classes), 120 sequences grouped into 6 tasks
configs/idd_full_ft.yaml     IDD with the encoder fine-tuned + test-time augmentation
configs/cityscapes.yaml      Cityscapes: paths, label map, splits, transforms, tasks
configs/camvid.yaml          CamVid, 4 drive sequences (local real-data benchmark)
configs/smoke_synthetic.yaml tiny synthetic run for checking the pipeline
drift/config.py              YAML + _base_ inheritance + --opts overrides
drift/envcheck.py            start-up check for the Triton / Python.h problem
drift/data/                  sample discovery, splits, CL tasks, joint transforms
drift/eda.py                 dataset EDA (plots + metadata.json + eda_report.md)
drift/models/                encoders, LoRA, components, DRIFT assembly
drift/losses.py              CE + Dice + hidden-class loss (+ shift anti-collapse)
drift/metrics.py             IoU / mIoU / fwIoU / BIoU / Dice / F1 / bootstrap CI / CL metrics
drift/engine.py              trainer (phase 1) and frozen evaluation (phase 2)
drift/viz.py, testing.py     curves, forgetting, t-SNE, samples, test report
drift/dtrr.py                DTRR: class stats, sketch memory, probe drift monitor, dual controller, replay
drift/lora_audit.py          per-task LoRA checklist and gradient / weight / output proofs
configs/dtrr/                DTRR ablations (no replay, always, periodic, intra, inter, full, fixed thresholds)
configs/bdd_idd_smoke.yaml   BDD100K -> IDD on the tiny local subset, CPU
tools/                       eda, train, test, summarize_runs, make_camvid_lists, make_synthetic_cityscapes,
                             run_dtrr_smoke (proof-of-life checklist), run_dtrr_ablations
tests/                       unit tests, one or more per risk-register item; test_dtrr.py for DTRR / LoRA
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests
python tools/eda.py   --config configs/idd.yaml
python tools/train.py --config configs/idd.yaml
python tools/test.py  --config configs/idd.yaml --ckpt outputs/idd_drift/<run>/task5.pt
```

### Main experiment: Cityscapes → IDD (cross-dataset, domain-incremental)

`configs/cs_idd.yaml` (DRIFT as designed: frozen encoder + LoRA) and `configs/cs_idd_ft.yaml`
(encoder fine-tuned + TTA):
- **Tasks:** one per dataset, in order. BDD100K can be added as a third task; the config has a commented block.
- **Labels:** the 19 Cityscapes classes. IDD is mapped to them, and IDD classes without a counterpart become ignore. `train` is genuinely absent from IDD.
- **Schedule:** 30 epochs per task, identical for every task.
- **Splits:** each dataset's official val is its test set, and 10% of its train is val.
- **Results:** test mIoU per dataset plus their mean, and forgetting on Cityscapes after learning IDD.

```bash
python tools/preresize.py --config configs/cs_idd.yaml --dst /raid/workspace/cs_idd_720 --short-side 720
python tools/train.py --config configs/cs_idd.yaml --opts \
  dataset.domains.cityscapes.root=/raid/workspace/cs_idd_720/cityscapes \
  dataset.domains.idd.root=/raid/workspace/cs_idd_720/idd
```

(`preresize.py` reads the original locations from `dataset.domains.*.root` in the config.)

Without Cityscapes, use `configs/bdd_idd.yaml` / `bdd_idd_ft.yaml`. It is the same protocol with
**BDD100K → IDD**: BDD's 10k segmentation labels already use the 19 Cityscapes train ids.

### Fast IDD training (IDD-only protocol)

`configs/idd_fast.yaml` (LoRA) and `configs/idd_fast_ft.yaml` (encoder fine-tuned + TTA, the
highest-scoring setting) train each task for a fixed number of steps instead of epochs:
6000 steps on task 1, 1500 on each later task, at batch 16. They use bf16 and TF32, 16 loader
workers, and validation every ~1500 steps. Resize the dataset to 720p once, so the loaders
don't decode 1080p PNGs on every step:

```bash
python tools/preresize.py --config configs/idd.yaml --dst /raid/workspace/idd_720 --size 720 1280
python tools/train.py --config configs/idd_fast_ft.yaml --opts dataset.root=/raid/workspace/idd_720
```

Set `dataset.root` in the config first. To check everything on fake data in about a minute:

```bash
python tools/make_synthetic_cityscapes.py --out data/synthetic_cityscapes
python tools/train.py --config configs/smoke_synthetic.yaml
```

### On the DGX: the `Python.h` crash

Newer PyTorch builds run some eager ops through Triton, and Triton needs the
Python development headers. `drift/envcheck.py` detects missing headers at
start-up and switches those Triton ops off (`experiment.triton_native_ops: auto`),
so training continues on the normal CUDA kernels. To fix it permanently, install
the headers (`sudo apt-get install python3.12-dev`) or use a conda/uv Python.

## Adding a dataset

Create `configs/<name>.yaml` with `_base_: defaults.yaml` and set:

| key | what |
|---|---|
| `dataset.root`, `classes`, `palette`, `ignore_index` | where the data is and what the train ids are |
| `dataset.source` | `glob` (image glob + label path replacements, or a label glob) or `list` (txt of pairs); `domain_regex` reads the domain from the path |
| `dataset.label_map` / `reduce_zero_label` | raw id to train id |
| `splits.<name>` | a folder split (`split:` + `domains:`), a fraction carved from another split (`from:` + `fraction:`), or whole domains held out (`from:` + `holdout_domains:`). Any extra `val_*` split is evaluated after every task. |
| `transforms.train/val/test` | ordered op lists (`Resize`, `RandomScale`, `RandomCrop`, `HorizontalFlip`, `ColorJitter`, `GaussianBlur`, `RandomRotate`, `Grayscale`, `Normalize`, ...) |
| `continual.tasks` | explicit lists of domains, or `auto` with `group_size` or `num_groups` (balanced by image count); `min_images_per_task` merges tiny tasks; `enabled: false` gives joint training |
| `continual.absent_classes` | `{task: [class names]}` removed from that task's labels, for controlled-absence experiments |
| `continual.task_overrides` | per-task changes to any key, e.g. `{0: {train: {iters_per_task: 20000}}}` |
| `train.*` | `epochs` or `iters_per_task` (int or per-task list), batch size, accumulation, optimizer, per-module lr multipliers, `later_task_lr_mult`, scheduler |

Any key can be overridden from the command line:
`--opts train.optimizer.lr=3e-4 "train.iters_per_task=[16000,4000,4000]"`.

## Ablations

`model.modules.*` are booleans. A module that is `false` is never built, so it
has no parameters, nothing passes through it in the forward pass, and no
gradient reaches it.

| switch | off means |
|---|---|
| `lora` | no adapters (encoder frozen, or fully trained with `encoder.freeze: false`) |
| `cosine_head` | plain 1x1 linear classifier; prototype modules are switched off with it |
| `anchors` | no `a_c` term |
| `memory` | no `m_c` term and no memory writes |
| `shift_estimator` / `shift_library` | no `U_c B s` term (each needs the other) |
| `learnable_base` | adds a learnable `w_c` term (switched on automatically if only memory remains) |

The method's other switches:

| switch | values |
|---|---|
| `loss.absent_negatives` | `masked` (absent classes never negatives), `detached` (negatives through frozen prototypes, the default), `full` |
| `loss.hidden` | `enabled`, `scope: task \| iteration`, `num_hidden`, `iter_prob`, `max_presence` |
| `loss.dice.enabled`, `loss.ce.ohem`, `loss.shift_var.weight` | Dice, hard-pixel mining, anti-collapse term for `s` |
| `model.prototypes.memory` | `num_slots`, `read_tau`, `write_per: domain \| task`, `drift_compensation.enabled` |
| `model.inference.tta` | `{scales: [...], flip: true}` |

The model prints a table of its modules at startup, with on/off state and
parameter counts.

## DTRR: drift-triggered reactive replay

Optional (`dtrr.enabled: true`, see `configs/dtrr/`). It replays old-task classes from
feature-level sketches only when the forgetting estimate exceeds its budget. No image of a
finished task is stored. During each task:

1. A probe set of 256 current-task pixels gives `eps_hat = mean ||norm(z_t) - norm(z_0)||` and
   `delta_hat = max prototype shift` every K = 50 steps.
2. Per class, `F_hat = max(F_meas, F_bound)`. `F_meas` is the rise in error of pseudo-features
   sampled from the class sketch (mean and diagonal var of the decoder input) through the current
   decoder and head. `F_bound` is the share of stored margins that the measured drift can flip.
3. A dual variable per class, `lam <- clip(lam + alpha (F_hat - rho) + beta * staleness, 0, lam_max)`,
   sets which classes are replayed and how much. There is no hand-tuned trigger, and `lam = 0`
   means zero overhead.
4. The replay gradient goes through a conflict gate (projection when it opposes the current
   gradient), and the LoRA lr is braked by `1 / (1 + mean lam)`.

Disabled, training is bit-identical to the code without it. Every stage asserts that it ran
(probe self-test, memory save / reload / hash check, minor-class queues, controller use of
`eps_hat`). `python tools/run_dtrr_smoke.py` prints a PASS/FAIL checklist.
[docs/DTRR.md](docs/DTRR.md) has the full algorithm, the files it writes, the results template,
the assumptions, and the LoRA audit, including what happens to LoRA between tasks
(`model.lora.reset_per_task`).

## What gets reported

**Every epoch** (validation on the current task):
`ep[e/n] T[t/T] | train loss/acc | val loss/acc | mIoU IoU(fw) BIoU Dice | seen-avg mIoU`,
then a per-class table of IoU, F1 and BIoU.

**After each task**:
- val mIoU on every task (the accuracy matrix R; right of the diagonal is zero-shot transfer)
- average seen mIoU and BWT
- forgetting three ways: per task, on the classes common to all tasks, and per class on the pooled val of all seen tasks (one fixed class set)
- held-out test and `val_unseen` metrics
- shift-estimator diagnostics: the std of `s`, and how well `s` identifies the task compared with chance

**Test**:
- class-wise IoU, F1, BIoU, precision and recall
- mIoU with a 95% image-level bootstrap confidence interval
- fwIoU, BIoU, Dice, pixel accuracy, per-domain mIoU
- continual-learning metrics

Saved as `test_metrics.json`, `test_classwise.csv` and `results.json`.
`tools/summarize_runs.py` tabulates several runs as mean ± std over seeds.

**Figures** (`<run>/figures/`):
- train/val loss and accuracy curves
- mIoU, IoU, BIoU and Dice curves, plus a per-class IoU heatmap
- forgetting curves per epoch and per task, the accuracy matrix, and a class-forgetting heatmap
- test metrics after each task
- class-wise IoU/F1 bars, confusion matrix, per-domain mIoU
- t-SNE of pixel embeddings, coloured by class (with prototypes) and by domain
- N rows of image / ground truth / prediction

**EDA** (`outputs/eda/<dataset>/`): class distributions, presence and co-occurrence;
per-domain and per-task heatmaps with absent classes marked; spatial priors; image
statistics; `metadata.json`, `class_weights.json` and `eda_report.md`.

## How the design maps to code

- **Valid set**: the classes labelled in the current task. Under the default
  `absent_negatives: detached`, absent classes still appear in the softmax, but
  their prototypes are detached. Pixels learn to stay away from them, and the
  absent prototypes themselves never move.
- **Hidden set H**: up to `num_hidden` rarer classes that already have memory,
  fixed for the whole task (`scope: task`). Their pixels leave the main loss. A
  separate CE on them uses detached `z` and detached encoder statistics, so only
  the prototype path (`g_phi` head, `W_u`, `B`, gains) learns to carry a class
  the city "doesn't have".
- **Memory**: K slots per class, each with a domain key (the mean `s` of the
  city that wrote it) and a pixel count. At the end of each city, every observed
  class merges its mean `z` into the slot with the closest key, as a running
  mean. Before that write, all stored slots get semantic drift compensation for
  the change in the feature space during the task. A read attends over the slots
  with the image's `s`.
- **Shift**: `U_c B s` with `W_u` starting at zero (a residual), applied only to
  classes seen in training, with learnable gains on every prototype term.
- **Phase 2**: LoRA is merged, the softmax covers all classes, TTA is optional,
  and there are no memory writes, no gradient steps and no domain ID.

## Notes

- Pixel-level Dice and F1 use the same formula (2TP / (2TP + FP + FN)). Dice is
  reported as the class mean, F1 per class.
- Checkpoints leave out the frozen encoder weights. They are rebuilt from
  `model.encoder.pretrained`, and loading is strict about everything else.
- With torch < 2.6, transformers will not open `.bin` checkpoints. The encoder
  and CLIP loaders then use `torch.load(weights_only=True)`.
