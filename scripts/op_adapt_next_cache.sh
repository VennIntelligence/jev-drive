#!/usr/bin/env bash
# op-adapt, next B round (sim + real), real-data half: Cinque trunk caches of WOD train and NAVSIM navtrain.
# One-shot lane script (todos/2026-09-28-op-adapt.md, "下一轮 B 的真实数据"): waits until the seed replications on
# GPU 2 are done and the card has >= MIN_FREE_MB free, then caches wodtrain, then navtrain. Resumable (the cache script
# skips finished files). Writes $OUT/STATUS (last line = state), $OUT/DONE or $OUT/ERROR.
#   scripts/tmux_run.sh op-next-cache scripts/op_adapt_next_cache.sh
set -uo pipefail
OUT=$DATA_DIR/runs/op_adapt/next_cache
mkdir -p "$OUT"
PY=$DATA_DIR/envs/op-train/bin/python
GPU=${GPU:-2}
CPUS=${CPUS:-48-67}
WORKERS=${WORKERS:-20}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-450}      # the two caches need ~150 GB; stop instead of filling the shared disk
st() { echo "$(date '+%F %T') $*" | tee -a "$OUT/STATUS"; }
rm -f "$OUT/DONE" "$OUT/ERROR"

st "waiting for the seed runs (train-lam10-s1 / -s2 readout/summary.json) and $MIN_FREE_MB MB free on GPU $GPU"
until ls "$DATA_DIR"/runs/op_adapt/train-lam10-s1/*/readout/summary.json "$DATA_DIR"/runs/op_adapt/train-lam10-s2/*/readout/summary.json >/dev/null 2>&1; do sleep 120; done
until [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)" -gt "$MIN_FREE_MB" ]; do sleep 120; done

for ds in wodtrain navtrain; do
  free_gb=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt "$DISK_FLOOR_GB" ]; then st "ERROR disk floor: ${free_gb} GB free < $DISK_FLOOR_GB before $ds"; touch "$OUT/ERROR"; exit 1; fi
  st "caching $ds on GPU $GPU, cpus $CPUS, $WORKERS render workers (${free_gb} GB free)"
  if ! CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" "$PY" scripts/op_adapt_cache.py "$ds" --workers "$WORKERS"; then
    st "ERROR $ds failed (rerun this script to resume)"; touch "$OUT/ERROR"; exit 1
  fi
  n=$(ls "$DATA_DIR/processed/op_adapt/$ds" | grep -c '\.npz$')
  st "$ds done: $n files, $(du -sh "$DATA_DIR/processed/op_adapt/$ds" | cut -f1)"
done
st "DONE"
touch "$OUT/DONE"
