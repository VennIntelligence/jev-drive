#!/usr/bin/env bash
# Night queue 2, N6: ridge_late + pair-Delta fits for one route-fold seed (GPU for ridge, CPU threads for eigh).
#   GPU=3 [EIGH=cpu|cuda] scripts/n6_fit.sh <seed> <backbones comma list>
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-3} P5_SET=carla_p5v1_ba OMP_NUM_THREADS=${THREADS:-12} MKL_NUM_THREADS=${THREADS:-12} OPENBLAS_NUM_THREADS=${THREADS:-12}
nice -n 5 .venv/bin/python -m jevdrive.n6_backbones fit --seed "$1" --backbones "$2" --eigh ${EIGH:-cuda}
