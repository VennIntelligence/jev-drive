#!/usr/bin/env bash
# Night queue 2, N4 [B]: YOLO26x-seg + per-detection RoI features on E5's P5 v1 BA image list, P processes on one card.
#   scripts/night2_n4_detect.sh [P=6]      card: N4_GPU (default 4)
set -uo pipefail
cd ~/data/jev-drive
P=${1:-6} GPU=${N4_GPU:-4}
echo "$(date '+%F %H:%M') [night2/B] N4 detection, $P processes on GPU $GPU" >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
for i in $(seq 0 $((P - 1))); do
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 nice -n 5 "$DATA_DIR/envs/ultralytics/bin/python" -m jevdrive.night2_n4 detect --part "$i/$P" \
    2>&1 | grep -v deprecated > "$DATA_DIR/processed/night2/n4/detect_$i.log" &
done
wait
n=$(ls "$DATA_DIR"/processed/night2/n4/dets/part-*.parquet | wc -l)
echo "$(date '+%F %H:%M') [night2/B] N4 detection done ($n parts), GPU $GPU released" >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
echo "done: $n parts"
