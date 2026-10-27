# DRIFT segmentation pipeline

Dataset-agnostic, config-driven pipeline for domain-incremental semantic
segmentation (train city by city, test on any city with no domain ID).
It implements the DRIFT two-phase design: a frozen SegFormer with LoRA, a fusion
decoder, a cosine head over prototypes `p_c = norm(a_c + m_c + U_c B s)`, a shift
estimator `g_phi`, class memory written at the end of each task, and simulated
class absence during training.

## Layout

```
configs/defaults.yaml        every option, with comments
configs/cityscapes.yaml      Cityscapes: paths, label map, splits, transforms, tasks
configs/smoke_synthetic.yaml tiny synthetic run for checking the pipeline
drift/config.py              YAML + _base_ inheritance + --opts overrides
drift/data/                  sample discovery, splits, CL tasks, joint transforms
drift/eda.py                 dataset EDA (plots + metadata.json + eda_report.md)
drift/models/                encoders, LoRA, components, DRIFT assembly
drift/losses.py              validCE + Dice + hidden-class loss
drift/metrics.py             IoU / mIoU / fwIoU / BIoU / Dice / F1 / CL metrics
drift/engine.py              trainer (phase 1) and frozen evaluation (phase 2)
drift/viz.py, testing.py     curves, forgetting, t-SNE, samples, test report
tools/                       eda.py, train.py, test.py, make_synthetic_cityscapes.py
```

## Quick start

```bash
pip install -r requirements.txt
python tools/eda.py   --config configs/cityscapes.yaml
python tools/train.py --config configs/cityscapes.yaml
python tools/test.py  --config configs/cityscapes.yaml --ckpt outputs/cityscapes_drift/<run>/task5.pt
```

Set `dataset.root` in `configs/cityscapes.yaml` first. To check everything on
fake data in about a minute:

```bash
python tools/make_synthetic_cityscapes.py --out data/synthetic_cityscapes
python tools/eda.py   --config configs/smoke_synthetic.yaml --workers 0
python tools/train.py --config configs/smoke_synthetic.yaml
```

## Adding a dataset

Create `configs/<name>.yaml` with `_base_: defaults.yaml` and set:

| key | what |
|---|---|
| `dataset.root`, `classes`, `palette`, `ignore_index` | where the data is and what the train ids are |
| `dataset.source` | `glob` (image glob + label path replacements, or a label glob) or `list` (txt of pairs); `domain_regex` reads the domain from the path |
| `dataset.label_map` / `reduce_zero_label` | raw id to train id |
| `splits.train/val/test` | a physical folder split (`split:` + `domains:`) or a fraction carved from another split (`from:` + `fraction:`) |
| `transforms.train/val/test` | ordered op lists (`Resize`, `RandomScale`, `RandomCrop`, `HorizontalFlip`, `ColorJitter`, `GaussianBlur`, `RandomRotate`, `Grayscale`, `Normalize`, ...) |
| `continual.tasks` | explicit lists of domains, or `auto` + `group_size`; `enabled: false` gives plain single-task training |
| `continual.task_overrides` | per-task changes to any key, e.g. `{0: {train: {epochs: 20, optimizer: {lr: 1e-3}}}}` |
| `train.*` | epochs (int, or a list with one per task), batch size, accumulation, optimizer, per-module lr multipliers, scheduler |

Any key can be overridden from the command line:
`--opts train.optimizer.lr=3e-4 train.epochs=[20,10,10]`.

## Ablations

`model.modules.*` are booleans. A module that is `false` is never built, so it
has no parameters, nothing passes through it in the forward pass, and no
gradient reaches it.

| switch | off means |
|---|---|
| `lora` | encoder stays frozen with no adapters |
| `cosine_head` | plain 1x1 linear classifier; prototype modules are switched off with it |
| `anchors` | no `a_c` term |
| `memory` | no `m_c` term and no memory writes |
| `shift_estimator` / `shift_library` | no `U_c B s` term (each needs the other) |
| `learnable_base` | adds a learnable `w_c` term (switched on automatically if only memory remains) |

The loss has its own switches: `loss.masked_softmax`, `loss.dice.enabled` and
`loss.hidden.enabled` (simulated absence). For example:
`--opts model.modules.memory=false experiment.name=abl_no_memory`.
The model prints a table of modules with their on/off state and parameter counts
at startup.

## What gets reported

**Every epoch** (validation on the current task):
`ep[e/n] T[t/T] | train loss/acc | val loss/acc | mIoU IoU(fw) BIoU Dice | seen-avg mIoU`,
then a per-class table of IoU, F1 and BIoU. With `eval.track_seen_every_epoch`,
every task seen so far is also evaluated each epoch for the forgetting curve.

**After each task**: val mIoU on every task (the accuracy matrix R, where the
right of the diagonal is zero-shot transfer), average seen mIoU, forgetting,
BWT, and held-out test metrics.

**Test**: class-wise IoU, F1, BIoU, precision and recall, plus mIoU, fwIoU, BIoU,
Dice, pixel accuracy, per-domain mIoU, and CL metrics (final average, average
incremental, BWT, average forgetting, FWT). Saved as `test_metrics.json` and
`test_classwise.csv`.

**Figures** (`<run>/figures/`): train/val loss and accuracy curves; mIoU, IoU,
BIoU and Dice curves; per-class IoU heatmap; forgetting curve per epoch and per
task; accuracy matrix; test metrics after each task; class-wise IoU/F1 bars;
confusion matrix; per-domain mIoU; t-SNE of pixel embeddings coloured by class
(with prototypes) and by domain; N rows of image / ground truth / prediction.

**EDA** (`outputs/eda/<dataset>/`): split and domain sizes; class pixel share
(log scale) and pixel count per class index; class presence; class area
boxplots; classes per image; ignore share; per-domain and per-task class
heatmaps with absent classes marked; co-occurrence; spatial priors; RGB
histograms and resolutions; brightness per domain; sample overlays; class legend;
`metadata.json`, `class_weights.json` and `eda_report.md`. To weight the loss
with these, set
`loss.ce.class_weights: outputs/eda/cityscapes/class_weights.json::median_frequency`.

## How the diagram maps to code

- **Valid set**: the classes labelled anywhere in the current task's training
  cities (`continual.valid_set: task`). Classes outside it are masked out of the
  softmax, so absent classes are never negatives.
- **Hidden set H**: on each iteration, with probability `loss.hidden.prob`,
  `num_hidden` valid classes that already have memory are hidden. Their pixels
  leave the main CE and Dice. A separate CE on those pixels uses detached `z`,
  so only the prototype and shift path (`g_phi`, `W_u`, `B`) learns to carry a
  class the city "doesn't have".
- **Memory**: `K = num_slots` slots per class. At the end of a task (or of each
  city, with `memory.write_per: domain`), the mean normalised `z` of every
  observed class goes into the next slot. Absent classes are untouched.
- **Phase 2**: LoRA is merged, the softmax covers all classes,
  `inference.unknown_threshold` is optional, and there are no memory writes,
  no gradient steps and no domain ID.

Interpretation choices, since the diagram leaves them open: `U_c` is row `c` of
`W_u [C, Kb]` mixing `Kb` basis shifts `B [Kb, D, S]`. Memory read is the mean
of filled slots. CLIP anchors are mapped to D with a fixed random projection and
mean-centred.

## Notes

- Pixel-level Dice and F1 use the same formula (2TP / (2TP + FP + FN)). Dice is
  reported as the class mean, F1 per class.
- Cityscapes test labels are not public. The config uses the official val
  cities (frankfurt, lindau, munster) as held-out test, and carves 10% of each
  training city out as val.
- With torch < 2.6, transformers will not open `.bin` checkpoints. The encoder
  and CLIP loaders fall back to `torch.load(weights_only=True)`.
- On a 4 GB GPU, batch 2 at a 512x512 crop uses about 1.4 GB in training. The
  Cityscapes config uses `accum_steps: 4`.
