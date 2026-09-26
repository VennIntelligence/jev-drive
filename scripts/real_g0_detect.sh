#!/usr/bin/env bash
# Shared YOLO26x-seg detections of the real-data transfer round (todos/2026-09-26-real-data-transfer.md, [G0]).
# One detector process per slice of processed/real_transfer/yolo/slices (python -m jevdrive.real_g0 lists), slices
# spread round-robin over the GPUs; E5's detector config unchanged. Resumable: finished part files are skipped.
# Usage (on the box, inside tmux): scripts/real_g0_detect.sh [gpus, default 0,1,2,3,4]
set -uo pipefail
gpus=(${1:-0 1 2 3 4})
gpus=(${gpus[@]//,/ })
Y=$DATA_DIR/processed/real_transfer/yolo
source "$DATA_DIR/envs/ultralytics/bin/activate"
mkdir -p "$Y/dets" "$Y/logs"
pids=()
for f in "$Y"/slices/slice_*.parquet; do
  k=$(basename "$f" .parquet); k=${k#slice_}
  g=${gpus[$((10#$k % ${#gpus[@]}))]}
  CUDA_VISIBLE_DEVICES=$g python -m jevdrive.fastperc detect --backend yolo:yolo26x-seg.pt:640:half --list "$f" --full \
    --out "$Y/dets/s$k" --keep 0.25 --batch 16 --workers 4 --tag "g0-s$k" > "$Y/logs/s$k.txt" 2>&1 &
  pids+=($!)
done
echo "$(date +%H:%M:%S) started ${#pids[@]} detector processes on GPUs ${gpus[*]}"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date +%H:%M:%S) all detector processes finished (rc=$rc); done.json per slice: $(ls "$Y"/dets/s*/done.json | wc -l)"
exit $rc
