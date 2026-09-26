#!/usr/bin/env bash
# Night queue 2, N3 [B]: the GPU-minute steps on one card (N3_GPU, default 4), in order; stops at the first failure.
#   scripts/night2_n3_gpu.sh <step> [<step> ...]    steps: "fit:<seeds>" "navridge" "gates" "p5cls:<seeds>"
set -euo pipefail
cd ~/data/jev-drive
GPU=${N3_GPU:-4}
for st in "$@"; do
  step=${st%%:*}; seeds=${st#*:}; [[ $seeds == "$st" ]] && seeds=0
  echo "$(date '+%F %H:%M') [night2/B] $step (seeds $seeds) on GPU $GPU" >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16 nice -n 5 .venv/bin/python -m jevdrive.night2_n3 "$step" --seed "$seeds"
  echo "$(date '+%F %H:%M') [night2/B] $step done, GPU $GPU released" >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
done
