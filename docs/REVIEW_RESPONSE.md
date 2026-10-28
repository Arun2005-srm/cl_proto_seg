# Response to the risk register

Each row of `drift_risk_register.xlsx` with what changed, where, and how it is checked.
Tests are in `tests/test_review_fixes.py` (`python -m pytest -q tests`, about 10 s on CPU).

Status meanings: **Fixed** = the failure mode is removed and tested. **Mitigated** = reduced and
measurable, but still a modelling assumption. **Addressed by protocol** = an evaluation change
rather than a code fix.

## Problems found in the IDD run (idd.md)

| problem | cause | fix |
|---|---|---|
| Crash: `fatal error: Python.h: No such file or directory` | The DGX's PyTorch routes some eager ops (here a batch-size-1 `bmm` in the prototype einsum's backward) through Triton kernels in `torch._native`. Triton compiles a C helper on first use, and that needs the Python headers, which the venv's Python lacks. | `drift/envcheck.py` runs at start-up. If `Python.h` is missing it turns those Triton ops off through `torch.backends.python_native`, so PyTorch uses its normal CUDA kernels (`experiment.triton_native_ops: auto`). The prototype einsum is now a plain matmul, and a task smaller than one batch is sampled with replacement, so the batch-1 shape that triggered the crash no longer happens. Permanent fix on the server: `sudo apt-get install python3.12-dev`, or a conda/uv Python. |
| `LoRA enabled but no Linear named ['query','key','value']`, so 0 LoRA layers | Newer transformers renamed SegFormer's attention projections. The frozen encoder trained nothing at all. | LoRA targets are regexes over the full layer name and cover both naming schemes, plus attention-output and MLP layers (96 layers on MiT-B2). LoRA enabled with no matching layer is now an error. |
| 120 tasks, some with 1 image; very low scores | IDD's numbered folders are capture sequences, not cities. One task per sequence gave 1 to 200 images per task. Lexicographic sorting ("0, 1, 10, 100") scattered the order. | `continual.num_groups` builds balanced tasks (IDD: 6), `min_images_per_task` merges small ones, and domains sort naturally. |
| LR never got past warm-up | A 100-iteration warm-up restarted on every task, and many tasks only had 20 to 100 iterations. | Warm-up runs on the first task only and is capped at 10% of its steps. `train.iters_per_task` gives each task a fixed step budget whatever its size. |

## Register

| ID | risk | change | where | test | status |
|---|---|---|---|---|---|
| M1 | Memory is last-writer-wins | A write merges into a slot as a count-weighted running mean over every city routed there (`momentum` is now only a floor). Absent classes are untouched. "Earlier cities only" holds because writes happen at task end. | `PrototypeGenerator.memory_write` | `test_M1_...` | Fixed |
| M2 | Stored m_c goes stale as LoRA/decoder train | Semantic drift compensation (Yu et al., CVPR 2020): at task end, every stored slot moves by the kernel-weighted mean of `z_new - z_old` over current-task pixels. No images are stored. Later tasks also train at `later_task_lr_mult` (0.3x). | `Trainer.compensate_drift`, `apply_drift` | ablation `abl_no_sdc` | Mitigated |
| M3 | Mean read collapses multimodality | K=4 slots per class, each keyed by the mean `s` of the cities that wrote it. A read is attention over the slots using the image's `s`, so a test image uses the past cities it resembles. | `memory_read`, `memory_write` | `test_M3_...` | Fixed |
| M4 | Never-seen classes get a noisy shift | `class_seen` mask: no shift term for classes never observed in training. `W_u` starts at 0, so the shift is a residual. | `PrototypeGenerator.forward` | `test_M4_...`, `test_shift_term_starts_at_zero` | Fixed |
| T1 | Hidden loss leaks into LoRA/encoder through s | `g_phi` reads detached encoder statistics (`detach_input: true`). The hidden term is built from detached `z` and detached stats, and detaches every non-hidden prototype. | `ShiftEstimator`, `DRIFTLoss`, `prototypes_from_stats` | `test_T1_...` (x2) | Fixed |
| T2 | Per-iteration hiding is too easy | `hidden.scope: task`: one hidden set H per task, so the class is consistently absent for that task. Within the task it is hidden in `iter_prob` of the iterations, and only for classes present in at most `max_presence` of its images. | `Trainer.fit`, `DRIFTLoss` | `test_T2_...` | Mitigated (the z of other iterations still sees H; full absence would discard data) |
| T3 | Masked train vs all-class test softmax | `absent_negatives: detached`: absent classes are negatives in the softmax, but through detached prototypes. Pixels learn to stay away from them, so the all-class test softmax is calibrated, while the absent prototypes are never pushed. | `DRIFTLoss.forward` | `test_T3_...`, ablation `abl_masked` | Fixed |
| T4 | Prototype terms summed with no scale | Learnable per-term gains (log-space) for anchors, memory, shift and base. | `PrototypeGenerator.log_gain` | - | Fixed |
| T5 | Weight decay shrinks B / W_u | The prototype path, gains, norms, biases and tau go in `*_no_wd` groups with weight decay 0. | `DRIFT.param_groups`, `build_optimizer` | `test_T5_...` | Fixed |
| A1 | BatchNorm in the decoder | GroupNorm (`decoder.norm: gn`): no running statistics to drift, any batch size. | `FusionDecoder`, `make_norm` | `test_A1_...` | Fixed |
| A2 | Same LR on every task, no drift constraint | `later_task_lr_mult` (0.3), optional `lr_task_decay`, plus M2's drift compensation. | `Trainer._lr_scale` | - | Mitigated |
| A3 | Stochastic depth active in the frozen encoder | A frozen encoder is kept in eval mode during training (no drop-path or dropout); LoRA layers follow the train flag. | `DRIFT.train` | `test_A3_...` | Fixed |
| S1 | g_phi may collapse | After every task: per-dimension std of `s`, and a nearest-centroid probe of task identity from `s` on held-out images vs chance (logged, and saved in `history.json` / `results.json`). Optional anti-collapse hinge `loss.shift_var`. | `shift_diagnostics`, `DRIFTLoss` | `test_S1_...` | Fixed (measured every run) |
| S2 | One image-level s assumes a correlated shift | Each class mixes the shift bases with its own `W_u` row and reads its own domain-keyed slots, so the effective shift is class-specific. The input is still one image-level `s`. | `PrototypeGenerator` | - | Mitigated |
| E1 | Cityscapes shift is mild | IDD config with balanced sequence tasks. `continual.absent_classes` removes chosen classes from chosen tasks to create genuine absences. CamVid benchmark (4 real drive sequences). | `configs/idd.yaml`, `Trainer` | `docs/benchmark_camvid.md` | Addressed by protocol |
| E2 | mIoU over different class sets per task | Also reported: `R_common` (classes present in every task's val) and class-level forgetting on the pooled val of all seen tasks (one fixed class set). | `Trainer.evaluate_stage`, `class_forgetting` | `test_E2_...` | Fixed |
| E3 | Few test cities, one seed | 95% image-level bootstrap CI on test mIoU in every report. `tools/summarize_runs.py` gives mean ± std over seeds; the CamVid benchmark uses 3 seeds. | `bootstrap_miou`, `run_test` | `test_metrics_on_known_confusion` | Fixed |
| E4 | Val carved from training cities only | `{from: train, holdout_domains: k}` holds out whole domains as an unseen-domain val (`val_unseen` in `idd.yaml`), evaluated after every task. | `build_splits` | - | Fixed |
| R1 | Silent fallback to random anchors | A failed CLIP/file load is an error unless `anchors.allow_fallback: true`; the default source is now `clip`. | `build_anchors` | `test_R1_...` | Fixed |
| R2 | CLIP .bin load with strict=False | Missing keys other than `position_ids` raise an error. | `_load_clip` | - | Fixed |
| R3 | load_model strict=False | Strict: only frozen encoder weights (deliberately omitted) may be missing; anything else raises. | `engine.load_model` | `test_R3_H3_...` | Fixed |
| N1 | nan_to_num on CE hides NaNs | CE with no labelled pixel returns a graph-connected 0. Non-finite losses skip the step and are counted; more than 20 per epoch aborts. Also fixed: the shift estimator's `std` was NaN for 1x1 feature maps. | `ce_loss`, `Trainer.train_epoch`, `ShiftEstimator.stats` | `test_N1_...` | Fixed |
| N2 | No merged-vs-unmerged LoRA test | Equivalence test with non-zero LoRA weights. | `lora.py` | `test_N2_...` | Fixed |
| H1 | No tests, no LICENSE, unpinned requirements | 23 unit tests, MIT LICENSE, `requirements.txt` (ranges) + `requirements-lock.txt` (exact tested versions). | repo root | - | Fixed |
| H2 | outputs/ and data committed, no .gitignore | `.gitignore` added; `outputs/` and `data/` removed from the index. Old history still holds them until it is rewritten. | repo root | - | Fixed going forward |
| H3 | Checkpoints store the full frozen encoder | Checkpoints omit frozen encoder weights; they are rebuilt from `model.encoder.pretrained` and checked on load. | `DRIFT.checkpoint_state` | `test_R3_H3_...` | Fixed |
