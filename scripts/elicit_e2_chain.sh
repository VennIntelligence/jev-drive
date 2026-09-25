#!/usr/bin/env bash
# E2 navtrain chain after the edit-pair build (todos/2026-09-26-elicitation-program.md, [E2] 00:57):
# feature inputs -> Qwen L18_last on x+ / x- / placebo clips -> openpilot temporal on the same -> fit + readouts.
#   scripts/elicit_e2_chain.sh <gpu>      (run inside tmux; every step is resumable)
set -euo pipefail
GPU=${1:-1}
cd ~/data/jev-drive
PY=.venv/bin/python OP=$DATA_DIR/envs/openpilot/bin/python
D=$DATA_DIR/processed/elicit_e2/navtrain/main
for i in $(seq 360); do [[ -e $D/done.json ]] && break; sleep 20; done
[[ -e $D/done.json ]] || { echo "build not done after 2 h"; exit 1; }
$PY -m jevdrive.elicit_e2 feat-index
for sd in plus minus placebo; do
  env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 nice -n 10 $PY -m jevdrive.navsim_qwen work e2nav_$sd --workers 4
  $PY -m jevdrive.navsim_qwen check e2nav_$sd
done
for sd in plus minus placebo; do
  env CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $OP scripts/navsim_zs_openpilot.py feat --split e2nav_$sd --models cinque lebowski --workers 3 --chunk 500
  env CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $OP scripts/navsim_zs_openpilot.py feat --split e2nav_$sd --models cinque lebowski --merge
done
env CUDA_VISIBLE_DEVICES=$GPU P5_SET=carla_p5v1_ba OMP_NUM_THREADS=8 nice -n 10 $PY -m jevdrive.elicit_e2_train run
echo "E2 navtrain chain done"
