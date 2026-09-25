#!/usr/bin/env bash
# Fusion diagnostics Q9b (todos/2026-09-25-fusion-diagnostics.md): Qwen L18_grid on the P5 obs rows, then the
# attention-pooling reaction head. GPU 3 only, <= 12 GB; 8 pinned cores before 19:00 CST (FD_CORES to change).
# Usage (in tmux): scripts/fd_q9b.sh profile | extract [--compile --batch N] | fit
set -uo pipefail
cd ~/data/jev-drive
CORES=${FD_CORES:-198,199,202-207}
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PIN=(taskset -c "$CORES" nice -n 10)
step=$1; shift
"${PIN[@]}" .venv/bin/python -m jevdrive.fusion_q9b "$step" --vram-gb 12 "$@"
