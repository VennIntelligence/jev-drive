#!/usr/bin/env bash
# Night queue 2, N2 on P5 v1 (todos/2026-09-26-night-queue-2.md): labels, probes, desire targets, desire runner.
# GPU 2 only, pinned to half the container's cores at most. Usage (in tmux): scripts/night2_n2.sh [sets...]
set -euo pipefail
cd ~/data/jev-drive
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=.venv/bin/python OP=$DATA_DIR/envs/openpilot/bin/python
SETS=${*:-carla_p5v1_ba carla_p5v1_pdm}
for s in $SETS; do
    [[ -f $DATA_DIR/processed/$s/night2_labels.parquet ]] || $PY -m jevdrive.night2_n2 labels --set "$s" --workers 24
    $PY -m jevdrive.night2_n2 targets --set "$s"
done
for s in $SETS; do $OP scripts/night2_desire.py --set "$s" --workers 6; done
$PY -m jevdrive.night2_n2 desire
for s in $SETS; do
    dets=""; [[ $s == carla_p5v1_ba ]] && dets=$DATA_DIR/processed/elicit_e5/dets
    $PY -m jevdrive.night2_n2 probe --set "$s" ${dets:+--dets "$dets"}
done
