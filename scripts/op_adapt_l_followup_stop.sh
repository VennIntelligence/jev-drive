#!/usr/bin/env bash
# op-adapt L follow-up, stop split: rows -> YOLO detections -> labels -> per-cause readout (GPU 0, cores 41-74).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=2 PYTHONPATH=$PWD
F=$DATA_DIR/runs/op_adapt_L/followup
TR="taskset -c 41-74"
$TR $DATA_DIR/envs/op-train/bin/python scripts/op_adapt_l_followup.py stoprows
$TR $DATA_DIR/envs/ultralytics/bin/python -m jevdrive.fastperc detect --backend yolo:yolo26x-seg.pt:1280:half --keep 0.05 \
  --full --list $F/stop_list.parquet --out $F/dets --tag opL-fu-stop --workers 12
$TR $DATA_DIR/envs/op-train/bin/python scripts/op_adapt_l_followup.py stoplabel
$TR $DATA_DIR/envs/op-train/bin/python scripts/op_adapt_l_followup.py stopread
