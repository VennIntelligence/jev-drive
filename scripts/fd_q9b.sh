#!/usr/bin/env bash
# Fusion diagnostics Q9b (todos/2026-09-25-fusion-diagnostics.md): Qwen L18_grid on the P5 obs rows, then the
# attention-pooling reaction head. GPU from FD_GPU (default 3), P5 set from P5_SET (default carla_p5); 8 pinned cores (FD_CORES).
# Usage (in tmux): scripts/fd_q9b.sh profile | extract [--compile --batch N] | fit
set -uo pipefail
cd ~/data/jev-drive
CORES=${FD_CORES:-198,199,202-207}
export CUDA_VISIBLE_DEVICES=${FD_GPU:-3} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PIN=(taskset -c "$CORES" nice -n 10)
step=$1; shift
"${PIN[@]}" .venv/bin/python -m jevdrive.fusion_q9b "$step" --vram-gb ${FD_VRAM_GB:-12} "$@"
