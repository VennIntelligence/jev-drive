#!/usr/bin/env bash
# G3b (3), todos/2026-09-26-real-data-transfer.md [G3]: YOLO26x-seg on the t0 images of every side of the navtrain
# edit pairs (python -m jevdrive.real_g3 edit-list), E5 / G0 detector config unchanged, one process on one GPU.
# Usage (on the box, inside tmux): scripts/real_g3_detect.sh <gpu>
set -euo pipefail
G=$DATA_DIR/processed/real_transfer/g3
source "$DATA_DIR/envs/ultralytics/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=4
CUDA_VISIBLE_DEVICES=${1:-0} python -m jevdrive.fastperc detect --backend yolo:yolo26x-seg.pt:640:half \
  --list "$G/edit_images.parquet" --full --out "$G/dets" --keep 0.25 --batch 16 --workers 6 --tag g3-edit
