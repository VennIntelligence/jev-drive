#!/usr/bin/env bash
# DrivoR and WA-JEPA on one T2 request set (jevdrive/top10_t2.py request format): DrivoR batched, WA-JEPA batch 1 in
# NSHARD processes on the same card, then merged. Outputs $DATA_DIR/runs/top10_t2/preds/<set>_<model>.npz.
# Usage (on the box, in tmux): GPU=3 NSHARD=2 WORKERS=2 scripts/top10_t2/real_infer.sh <set> [drivor|wajepa|both]
set -euo pipefail
: "${DATA_DIR:?}"
set_=$1; which=${2:-both}
GPU=${GPU:-3}; NSHARD=${NSHARD:-2}; WORKERS=${WORKERS:-2}
repo=$(cd "$(dirname "$0")/../.." && pwd)
R=$DATA_DIR/runs/top10_t2; req=$R/requests/$set_.npz; out=$R/preds; mkdir -p "$out" "$R/logs"
t0=$SECONDS
if [[ $which == drivor || $which == both ]]; then
  (cd "$DATA_DIR/third_party/drivor" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 "$DATA_DIR/envs/drivor/bin/python" \
    "$repo/scripts/top10_t2/drivor_run.py" "$req" --out "$out/${set_}_drivor.npz" --workers $((WORKERS * 2)) \
    2>&1 | tr '\r' '\n' | grep -v "it/s\]$" > "$R/logs/${set_}_drivor.log")
  echo "drivor done at $((SECONDS - t0)) s"
fi
if [[ $which == wajepa || $which == both ]]; then
  pids=(); parts=()
  for i in $(seq 0 $((NSHARD - 1))); do
    p=$out/${set_}_wajepa.part$i.npz; parts+=("$p")
    (cd "$DATA_DIR/third_party/wajepa" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 "$DATA_DIR/envs/wajepa/bin/python" \
      "$repo/scripts/top10_t2/wajepa_run.py" "$req" --out "$p" --shard "$i" "$NSHARD" --workers "$WORKERS" \
      > "$R/logs/${set_}_wajepa.part$i.log" 2>&1) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  (cd "$DATA_DIR/third_party/wajepa" && "$DATA_DIR/envs/wajepa/bin/python" "$repo/scripts/top10_t2/wajepa_run.py" \
    --merge "${parts[@]}" --out "$out/${set_}_wajepa.npz")
  echo "wajepa done at $((SECONDS - t0)) s"
fi
