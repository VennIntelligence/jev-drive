#!/usr/bin/env bash
# Night queue 2, N5: metric depth at the stored detections' contact points (envs/depth), one model per call.
#   GPU=3 scripts/n5_depth.sh unidepth|da3
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-3} PYTHONPATH=$PWD OMP_NUM_THREADS=4
echo "$(date '+%F %T') N5 depth $1 on GPU $CUDA_VISIBLE_DEVICES"
nice -n 5 "$DATA_DIR/envs/depth/bin/python" -m jevdrive.n5_depth depth --model "$1" --workers ${WORKERS:-8}
echo "$(date '+%F %T') N5 depth $1 done"
