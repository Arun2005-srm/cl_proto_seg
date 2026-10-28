#!/usr/bin/env bash
# CamVid benchmark: original code (commit 9f75e0f, in a git worktree) vs the fixed code,
# identical splits and step budgets, 3 seeds, plus lower/upper bounds and ablations.
#
#   python tools/make_camvid_lists.py --root data/camvid
#   bash tools/run_camvid_benchmark.sh
#   python tools/summarize_runs.py --root outputs/bench --out docs/benchmark_camvid.md
#
# The baseline config (configs/camvid_old.yaml in the worktree) is generated below: the
# original code needs `_replace_` on the test split and epochs instead of iters_per_task.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BASELINE_DIR:-$REPO/../baseline_head}
OUT=$REPO/outputs/bench
EP='train.epochs=[19,24,19,13]'
COMMON="viz.tsne.enabled=false experiment.output_dir=$OUT"
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT/logs"

if [ ! -d "$BASE" ]; then
  git -C "$REPO" worktree add "$BASE" 9f75e0f
fi
sed -e "s#root: data/camvid#root: $REPO/data/camvid#" \
    -e 's#  test:  {from: train, fraction: 0.2, per_domain: true}#  test:  {_replace_: true, from: train, fraction: 0.2, per_domain: true}#' \
    -e 's#  iters_per_task: \[1500, 600, 600, 600\]#  epochs: [19, 24, 19, 13]#' \
    "$REPO/configs/camvid.yaml" > "$BASE/configs/camvid_old.yaml"

new() {  # name seed extra-overrides...
  local name=$1 seed=$2; shift 2
  echo "=== $(date +%H:%M) new $name seed $seed"
  (cd "$REPO" && python tools/train.py --config configs/camvid.yaml --opts train.iters_per_task=null "$EP" $COMMON \
     experiment.name=$name experiment.seed=$seed "$@" > "$OUT/logs/${name}_s${seed}.log" 2>&1) || echo "FAILED $name $seed"
}
old() {
  local seed=$1
  echo "=== $(date +%H:%M) old seed $seed"
  (cd "$BASE" && python tools/train.py --config configs/camvid_old.yaml --opts $COMMON \
     experiment.name=old experiment.seed=$seed > "$OUT/logs/old_s${seed}.log" 2>&1) || echo "FAILED old $seed"
}

for s in 0 1 2; do new new $s; old $s; done
new naive_ft 0 model.modules.cosine_head=false train.optimizer.later_task_lr_mult=1.0 loss.hidden.enabled=false
new joint 0 continual.enabled=false 'train.epochs=[18]'
new full_ft 0 model.modules.lora=false model.encoder.freeze=false
new abl_hidden_off 0 loss.hidden.enabled=false
new abl_masked 0 loss.absent_negatives=masked
new abl_no_sdc 0 model.prototypes.memory.drift_compensation.enabled=false
echo "=== $(date +%H:%M) all done"
