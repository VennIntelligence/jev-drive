#!/usr/bin/env bash
# Fast-perception runs on GPU 0 (todos/2026-09-26-fast-perception.md). Run inside tmux via scripts/tmux_run.sh.
#   scripts/fastperc.sh latency <env> <keep> <backend spec>...    Q4d-protocol latency, one run per spec
#   scripts/fastperc.sh detect  <env> <keep> <tag> <backend spec> the subset S (P5 + nuScenes), resumable
# Pinned to GPU 0 and 20 cores (FASTPERC_CPUS), the budget the pre-registration was given.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
step=$1 env=$2 keep=$3; shift 3
PY=$DATA_DIR/envs/$env/bin/python
export CUDA_VISIBLE_DEVICES=${FASTPERC_GPU:-0} OMP_NUM_THREADS=4 PYTHONPATH=$PWD
CPUS=${FASTPERC_CPUS:-180-199}
L=$DATA_DIR/processed/fusion_diag/lists
run() { taskset -c "$CPUS" "$PY" -m jevdrive.fastperc "$@"; }
if [[ $step == latency ]]; then
  for spec in "$@"; do
    run latency --backend "$spec" --keep "$keep" --list "$L/p5_prof200.parquet" --tag "lat-${spec//[:\/]/_}" \
      || echo "latency $spec failed"
  done
else
  tag=$1 spec=$2
  O=$DATA_DIR/processed/fastperc/dets/$tag
  run detect --backend "$spec" --keep "$keep" --list "$L/p5.parquet" --dataset p5 --out "$O/p5" --tag "det-$tag-p5" --workers 12
  run detect --backend "$spec" --keep "$keep" --list "$L/nusc.parquet" --dataset nusc --out "$O/nusc" --tag "det-$tag-nusc" --workers 12
fi
