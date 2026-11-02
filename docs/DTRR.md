# DTRR: Drift-Triggered Reactive Replay, and the LoRA training audit

Code: [drift/dtrr.py](../drift/dtrr.py), [drift/lora_audit.py](../drift/lora_audit.py), hooks in
[drift/engine.py](../drift/engine.py). Config: the `dtrr:` and `audit:` blocks of
[configs/defaults.yaml](../configs/defaults.yaml). Ablations: [configs/dtrr/](../configs/dtrr).

With `dtrr.enabled: false` (the default), training is bit-identical to the code before DTRR.
`tests/test_dtrr.py::test_regression_disabled_matches_baseline` re-trains the 2-task smoke run and
compares R, every epoch's train loss and every epoch's val mIoU against
`tests/fixtures/baseline_bdd_idd_smoke.json`, recorded from commit 485ef49.

## 1. How memory worked before DTRR

- **Write**: only at the end of a task (`Trainer.write_memory`). For each domain of the task, one
  pass over its training images (val transform) gives the mean normalised `z(p)` of every class
  with pixels. That mean goes into one of the K slots `m_c` (`PrototypeGenerator.memory_write`):
  an empty slot if there is one, otherwise a count-weighted merge into the slot whose key is closest.
  The key is the mean `s` of the writer. Before writing, `compensate_drift` moves the stored slots
  by the kernel-weighted mean `z_new - z_old` of current-task pixels (SDC).
- **Read**: in every forward pass (`memory_read`), `s(x)` attends over the filled slots of each
  class (`read_tau`). The result is added to `p_c = norm(g_a a_c + g_m m_c + g_s U_c B s)`.
- **What was missing**: nothing is written during a task. Nothing measures how much the features of
  old classes have drifted, and nothing replays them. The slots are prototypes, not samples of
  the feature distribution.

## 2. Algorithm

**Injection level.** The decoder is pixel-wise up to its GroupNorm: per-scale 1x1 projections,
bilinear upsampling to stride 4 (a 1x1 conv commutes with it), concat, 1x1 conv, GN, ReLU,
1x1 conv. A pixel's decoder input `u(p) = concat_i up(f_i)(p)` (1024-d for MiT-B2, the per-scale
encoder features) therefore determines `z(p)` once the GN statistics are fixed.
`FusionDecoder.gather_inputs` and `forward_pixels` implement this, and with the image's own GN
statistics they match the dense decoder to 1e-6 (`test_forward_pixels_reproduces_dense_decoder`).
Pseudo-features are injected at `u`, so replay trains the whole decoder, the head and the
prototype path (g_phi head, W_u, B, gains).

**Per task t:**

1. **Class statistics** (labels only, before training): pixel share `q_c` and image share of every
   union-vocabulary class. Minor: `q_c < p25(q over present classes)` or `q_c < 0.5%`.
   Absent: no pixel. Written to `class_stats_T<t>.json`. The label mapping used is hashed and
   asserted equal to the training loader's mapping (single source of truth: `SegDataset`).
2. **Memory check** (t > 0): load `memory_after_T<t-1>.pt` from disk. Every class in
   `memory_audit.json` must be present with an identical hash, otherwise DTRR raises.
3. **Probe set**: from the first 50 batches of a seeded random order of the current task (val
   transform, so no randomness), 256 pixels on the z grid. Every class present gets at least 4
   pixels (images are chosen greedily, rarest class first, at most 16 images). The images are kept on
   the CPU in fp16 and dropped at task end. `z0, p0 = probe_embed(model, probe)`: eval mode,
   `no_grad`, fp32, adapters unmerged. The same function is used for every later `z_t`; settings and
   function identity are asserted.
4. **Drift self-test**: add noise to the LoRA weights, then check `eps_hat > 0`; restore the weights,
   then check `eps_hat <= 1e-6`. Raises on failure.
5. **Every step**: staleness of every class (steps since it last received gradient). For classes
   in the batch, 32 sampled pixels update an online EMA sketch (mean / diagonal var of `u`). For
   minor classes, up to 64 pixels are pushed into a FIFO queue of 2048 features (current task
   only, dropped at task end).
6. **Every K = 50 steps, at step 1 and at task end** (the drift monitor):
   - `eps_hat = mean ||z_t - z0||` over probe pixels, `eps_c` per class, and
     `delta_hat = max_c ||p_t,c - p_0,c||`.
   - `F_meas[c]`: M = 64 pseudo-features from the sketch (`mean + sqrt(var) * noise`) go through
     the current decoder tail and cosine head. F_meas is the fraction with margin < 0 minus
     `err0` (the same quantity measured when the sketch was written). For intra classes it is
     the queue error minus the best error seen so far in the task.
   - `F_bound[c] = P(0 <= gamma_c < m)` from the stored margin histogram, where m is the margin
     change the drift can cause (see the assumptions).
   - `F_hat = max(F_meas, F_bound)`.
   - Dual update: `lam[c] = clip(lam[c] + alpha (F_hat[c] - rho_c) + beta clip(stale_norm[c] - 1, 0, 1) 1[F_hat[c] > 0], 0, lam_max)`.
     It needs no trigger threshold: lam falls whenever `F_hat < rho_c`.
     Defaults: `alpha 0.5, beta 0.05, lam_max 5, rho 0.02` (`rho_by_share: true` gives
     `rho_c = rho (q_c / mean q)^0.5`).
7. **Replay** at every optimizer step where `sum(lam) > 0`:
   `B_mem = min(2048, 256 sum(lam))` pseudo-pixels. Classes are drawn with p ∝ lam. Old classes are
   sampled from their sketch; current minor classes from their queue (half each for a class that
   has both). `L_mem` is the CE over seen classes (masked) through `forward_pixels` and the head,
   with prototypes built from the sketch's stored encoder statistics (so `g_phi` gets gradient).
   Conflict gate on decoder, head, g_phi, W_u and B: if `<g_cur, g_mem> < 0`, the conflicting
   component is removed (`g_cur -= <g_cur,g_mem>/||g_mem||^2 g_mem`). The final gradient is
   `g_cur + mean(lam) g_mem`. The LoRA brake scales the LoRA lr by `1 / (1 + mean(lam))`.
   With `lam = 0` there is no extra forward or backward pass.
8. **Task end**: one more monitor, then a sketch pass over the task's training images. For every
   class with at least one pixel it stores the mean and diagonal var of `u`, a 64-bin margin
   histogram over `[-2/tau, 2/tau]`, the pixel count, the memory slot `m_c` with its count and
   source pixels, the mean encoder statistics, the GN statistics, `err0`, and
   `low_confidence = count < 20`. The memory is then saved, reloaded into a fresh object and
   checked (same keys and shapes, finite, count > 0 for every class seen), and
   `memory_audit.json` is written. Also at task end: the sketch self-consistency check (warns below
   0.7) and the per-class activity table, which raises if a minor class present in the task has
   0 queue pushes.
9. **End of training**: raises if the controller never consumed `eps_hat`.

Inference is unchanged: no domain or task ID, no test-time update, and memory reads as before.

## 3. Files written per run

| file | content |
|---|---|
| `class_stats_T<t>.json` | q_c, image share, minor / absent per class, p25 threshold, mapping hash |
| `dtrr/drift_curve_T<t>.csv` / `.png` | step, eps_hat, delta_hat, margin-risk m, sum lam, replay counters, eps per probe class, previous-task val drop |
| `dtrr/controller_T<t>.csv` | step, class, F_meas, F_bound, F_hat, lam, rho_c, staleness |
| `dtrr/memory_after_T<t>.pt` | the sketch store (feature-level only) |
| `dtrr/memory_audit.json` | per class: tasks, counts, low_confidence, mean norm, mean var, err0, self-consistency, hash |
| `dtrr/dtrr_report.json` | per-task summary used by `tools/run_dtrr_smoke.py` |

Logged after each task: R (all earlier tasks), avg mIoU, BWT, forgetting, rare-class IoU
(IoU over classes that were minor in any task so far), replay step fraction, replay time fraction,
and `corr(eps_hat, previous-task val drop)` over the probes of the task.

## 4. Proof of life

```bash
python tools/run_dtrr_smoke.py                       # normal + aggressive (50x LoRA lr) runs, checklist, exit code
DTRR_SLOW=1 python -m pytest -q tests/test_dtrr.py   # unit tests + regression + smoke + alpha sweep
python tools/run_dtrr_ablations.py --opts dtrr.interval=10   # all ablations, budget-matched, results.md
```

## 5. Results table (fill in from `outputs/dtrr_ablations/results.md`)

Protocol: BDD100K → IDD, 19 Cityscapes classes, frozen MiT-B2 with per-task LoRA, 30 epochs per
task. The replay variants are budget-matched: `always` and `periodic` get the mean pseudo-pixels
per step of the adaptive variants.

| variant | T0 (BDD) mIoU | T1 (IDD) mIoU | avg mIoU | BWT | avg forgetting | rare-class IoU | replay steps | replay px/step | replay time |
|---|---|---|---|---|---|---|---|---|---|
| no replay | | | | | | | 0% | 0 | 0% |
| always-on replay | | | | | | | 100% | | |
| periodic replay (every 10) | | | | | | | 10% | | |
| DTRR intra only | | | | | | | | | |
| DTRR inter only | | | | | | | | | |
| DTRR full (self-adjusting lam) | | | | | | | | | |
| DTRR fixed thresholds (s*, eps*) | | | | | | | | | |

Report alongside: corr(eps_hat, previous-task forgetting), and the alpha sweep
(alpha ∈ {0.1, 0.5, 1.0}, final avg mIoU).

## 6. Assumptions and choices

1. **Decoder pixel-wise?** Up to GroupNorm, yes. GN statistics are per image, so each sketch
   stores the mean GN statistics of its task's images, and pseudo-features are normalised with
   them. Queue features use an EMA of the current task's statistics. This is exact when an image's
   statistics equal the stored ones and an approximation otherwise. With `decoder.norm: bn` or
   `none` it is exact.
2. **Forgetting is relative.** `F_meas` is the rise of the pseudo-feature error over its
   value when the sketch was written (inter), or over the best value so far in the task (intra).
   With the raw error, a class that is simply not learned yet reads as "forgotten" and lam saturates
   on every class from step 1 (this was measured).
3. **`F_bound`'s margin m.** The spec's worst case `m = 2 (eps_hat + delta_hat) / tau` is kept as
   `dtrr.estimator.bound: lipschitz`. With tau = 0.07 it is vacuous: eps_hat ≈ 0.26 after one step
   on the smoke model gives m ≈ 8 logits and F_bound ≈ 0.8–1.0 for every class, so replay is
   always on. The default `calibrated` sets m to the mean |margin change| actually measured on the
   probe pixels since task start. That is the realised fraction of the worst case on current-task
   pixels, with no tuned constant. The controller itself is the spec's, unchanged.
4. **Staleness term**: `beta clip(stale/running_mean - 1, 0, 1)`, applied only when `F_hat > 0`,
   so a stale class reacts faster but never triggers replay on its own. Staleness counts
   optimizer steps over the whole run (all union classes, every step); its running mean is an EMA
   over steps of the class mean.
5. **mean(lam)** in the final gradient and in the LoRA brake is the mean over classes with lam > 0.
6. **lam is reset to 0 at every task start**, because eps_hat and delta_hat are measured from the task
   start.
7. **Prototypes for pseudo-features** come from the current prototype generator at the stored mean
   encoder statistics of that task (`stats -> g_phi head -> s`). Replay therefore trains g_phi,
   W_u, B and the gains, not only the decoder and head.
8. **Probe preprocessing** is the val transform (deterministic), not the random train augmentation,
   so z0 and z_t see identical inputs. The probe forward is fp32 even when training uses AMP.
9. **Queues hold current-task decoder-input features only** and are cleared at task end, so no
   pixel of a finished task is kept. A minor class that never appears in a training batch counts
   as a failure (the run raises and says that its times-seen count is 0).
10. **Class statistics** are counted at full label resolution on the training labels after
    controlled-absence drops. Sketch counts are pixels on the stride-4 grid.
11. **`corr_val_batches: 2`**: the previous task's val mIoU is measured on 2 batches at every probe.
    This is cheap, so the correlation is noisy.
12. **The ablation budget** matches mean pseudo-pixels per step, not wall time; replay time is
    reported separately.

## 7. LoRA training audit (every task)

At each task start a checklist is printed and enforced: the number of LoRA tensors, trainable
LoRA params, LoRA params missing from `optimizer.param_groups`, the LoRA group lr, adapters
unmerged, autograd on, gradient checkpointing off, GradScaler state, and the adapter state (fresh
with `B = 0`, or continued). Then the following are checked, and any failure raises `LoRAAuditError`:

- the first forward's encoder output requires grad;
- after the first backward, the LoRA grad norm is finite and > 0 (unscaled by the GradScaler);
- at step 50 (or at task end for shorter tasks), `||W_lora_t - W_lora_start|| > 0` and the mean
  |logits with adapters − logits with adapters bypassed| on the task's first batch is > 0.

Trainable params are printed split into lora / decoder / head / shift (g_phi, W_u, B, gains) / other.

**What happened to LoRA at the end of task 1.** Nothing. The adapter was never merged during
training: `merge_lora(True)` is only called in `drift/testing.py` for the final test, after `fit`.
The same adapter therefore kept training in task 2, with its Adam state reset
(`reset_optimizer_per_task: true`). The audit proves that it gets gradient in task 2 and that it moves.
`model.lora.reset_per_task: true` (on in `configs/dtrr/base.yaml`, off by default so the baseline
is unchanged) switches to per-task LoRA: at task end, `W <- W + scale B A`, A is re-drawn,
`B = 0`, and the next task trains a fresh delta from exactly the merged function. Test time still
uses a single merged weight, with no task ID. Each task then has its own measurable
`||ΔW_t||`, and the adapter's rank budget is not shared across tasks. Checkpoints written after a merge include the merged base
weights of the wrapped layers, which no longer equal `pretrained`.

**No hidden detach.** `torch.no_grad` is used only in eval, memory and diagnostic code. Gradient
checkpointing is off. The GradScaler is used only with fp16 (the BDD→IDD configs use bf16, so it is
off). `DRIFT.train()` puts the frozen encoder in eval mode (fixed norms, no stochastic depth) but
keeps the LoRA wrappers in train mode; eval mode does not detach anything. The only intentional
detach is `shift_estimator.detach_input: true` on the input of g_phi.

## 8. Findings on the reported symptoms

- **0.46 s/it (task 1) vs 0.19 s/it (task 2).** In the code, both tasks run the same batch (8), crop
  (512×1024) and encoder backward (LoRA; the audit proves it in each task). Task 2 does more work
  per step (the hidden-class loss and, with DTRR, replay), so the code alone does not explain a
  faster task 2. The `it ... s/it` line divides wall time since the epoch start, data waiting
  included, by the iterations so far. So the gap is most likely data loading or start-up in task 1:
  worker start, first-epoch decoding of the source files, cuDNN autotuning. Each task now prints
  `[speed] T<t>: batch, crop, encoder backward` at the start and
  `[speed] T<t> epoch 1: input shape, s/it = data wait + compute; encoder backward ran` after
  epoch 1. Every `it` line also splits data from compute. One real run settles which it is.
- **`shift_var 0.000` in both tasks.** The term is enabled (`weight 0.05, target_std 1.0`). It is
  the hinge `0.05 * mean_d relu(1 - std_batch(s_d))`, so it is exactly 0 when every dimension of
  `s` already varies with std ≥ 1 across the batch, and the 3-decimal print hides values below 5e-4.
  It does not mean `s` is constant. On the smoke run the new `[shift]` line shows both regimes:
  T0 has a per-dim std of 0.152 (100% of dims below target, term 0.042); T1 has 0.999 (49% below,
  term 0.0098). In the real run, that line shows the mean of s, the per-dim std across the batch, the
  fraction of dims below target and the term to 5 decimals.
- **`train` at IoU 0.00.** BDD100K is the only task with `train` pixels (IDD maps nothing to it),
  and they are rare there. In task 2 it is absent: with `absent_negatives: detached`, IDD pixels
  are pushed away from its frozen prototype, and it can never be hidden (only valid classes can).
  The new end-of-task class table prints, per class, the training pixels, the steps with gradient,
  the max staleness, the memory-slot source pixels, and the val GT and predicted pixel counts and
  IoU. It lists classes that are present in val but never predicted. The local subset has no
  `train` pixel, so it cannot reproduce this. On the smoke run the same table shows `pole` and
  `traffic sign` with 0 predicted pixels after task 2 without DTRR. With DTRR, rare classes are
  queued in task 1 (intra replay when their error rises) and replayed from their sketch in task 2.
