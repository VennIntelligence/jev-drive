#!/usr/bin/env bash
# E2 WOD chain ([E2] 01:00 / 01:22): wait for the SAM scan -> candidates -> edit pairs on the given cards
# (lock-claimed chunks) -> Qwen on x+ / x- / placebo clips -> WOD-only and navtrain+WOD fits and readouts.
#   scripts/elicit_e2_wod_chain.sh "<gpu> <gpu> ..."      (inside tmux; resumable)
set -euo pipefail
GPUS=${1:-2}
cd ~/data/jev-drive
PY=.venv/bin/python SAM=$DATA_DIR/envs/sam3/bin/python
W=$DATA_DIR/processed/elicit_e2/wod
for i in $(seq 720); do [[ -e $W/sam/done.json ]] && break; sleep 20; done
[[ -e $W/sam/done.json ]] || { echo "scan not done after 4 h"; exit 1; }
[[ -e $W/candidates.done ]] || { $PY -m jevdrive.elicit_e2 wod-candidates && touch $W/candidates.done; }
pids=()
for g in $GPUS; do
  env CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 nice -n 10 $SAM -m jevdrive.elicit_e2 wod-build --tag main &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
[[ -e $W/main/done.json ]] || { echo "wod build incomplete"; exit 1; }
$PY -m jevdrive.elicit_e2 wod-feat-index
G1=${GPUS%% *}
for sd in plus minus placebo; do
  env CUDA_VISIBLE_DEVICES=$G1 OMP_NUM_THREADS=2 nice -n 10 $PY -m jevdrive.navsim_qwen work e2wod_$sd --workers 4
  $PY -m jevdrive.navsim_qwen check e2wod_$sd
done
until [[ -n $(ls $DATA_DIR/runs/elicitation/e2-train/*/r1_magnitude.csv 2>/dev/null) ]]; do sleep 60; done   # navtrain features exist
env CUDA_VISIBLE_DEVICES=$G1 P5_SET=carla_p5v1_ba OMP_NUM_THREADS=8 nice -n 10 $PY -m jevdrive.elicit_e2_train wod
echo "E2 WOD chain done"
