#!/usr/bin/env bash
# Zero-shot exam agent launcher in the shape closed_loop.py expects (`zsh <this> <cuda_id> <output_dir>`).
# The model runs in a resident server (scripts/hugsim_zs_server.py); this process only adapts I/O on the CPU.
# Configuration: HUGSIM_ZS_* environment variables, set by scripts/hugsim/zs_run.py (see scripts/hugsim/zs_agent.py).
repo=$(cd "$(dirname "$0")/../.." && pwd)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 exec "${DATA_DIR:-$HOME/data}/envs/hugsim/bin/python" -u \
  "$repo/scripts/hugsim/zs_agent.py" --output "$2"
