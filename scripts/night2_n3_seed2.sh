#!/usr/bin/env bash
# Night queue 2, N3 [B]: after seed 2's per-anchor scoring, fit Hydra seed 2 (GPU 4), write its navtest jobs, score them.
set -euo pipefail
cd ~/data/jev-drive
P=$(ls -d "$DATA_DIR"/runs/night2/n3-prep-s2/*)
until [[ -f $P/score.done ]]; do sleep 30; done
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
taskset -c 160-167 .venv/bin/python -m jevdrive.night2_n3 fit --seed 2
taskset -c 160-167 .venv/bin/python -m jevdrive.night2_n3 navjobs --seed 2
J=$(ls -td "$DATA_DIR"/runs/night2/n3-navjobs/* | head -1)
env NAVSIM_THREADS=4 taskset -c 160-187 scripts/real_g1_score.sh "$J/score_jobs.txt" 6
