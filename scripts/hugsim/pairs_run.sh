#!/usr/bin/env bash
# I3 full render (todos/2026-09-25-reactivity-program/i3-hugsim-pairs.md): N shard workers of pairs_render.py share
# one GPU, each pinned to its own block of the core range, then the P5-shaped index is built.
# Usage (on the box, inside tmux via slot_run.sh): [GPU=2 N=4 CORES=170-189] scripts/hugsim/pairs_run.sh [extra pairs_render args]
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
GPU=${GPU:-2}; N=${N:-4}; CORES=${CORES:-170-189}
IFS=- read -r lo hi <<< "$CORES"
per=$(( (hi - lo + 1) / N ))
logs=$DATA_DIR/runs/i3-hugsim-pairs/full-$(date +%Y%m%d-%H%M%S)
mkdir -p "$logs"
cd "$DATA_DIR/third_party/HUGSIM" || exit 1
pids=()
for i in $(seq 0 $((N - 1))); do
  a=$((lo + i * per)); b=$((a + per - 1))
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=$per taskset -c "$a-$b" "$DATA_DIR/envs/hugsim/bin/python" -u \
    "$repo/scripts/hugsim/pairs_render.py" --validate --tag full --shard "$i" "$N" "$@" > "$logs/worker$i.log" 2>&1 &
  pids+=($!)
  echo "worker $i: pid ${pids[-1]}, cores $a-$b, log $logs/worker$i.log"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "INFO" "$logs"/worker*.log | cut -c1-160
(( rc == 0 )) || { echo "a render worker failed"; exit 1; }
cd "$repo" && taskset -c "$CORES" "$DATA_DIR/envs/jevdrive/bin/python" -m jevdrive.hugsim_pairs index
