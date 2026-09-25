#!/usr/bin/env bash
# Preset (logged-trajectory) agent launcher in the shape closed_loop.py expects (`zsh <this> <cuda_id> <output_dir>`).
# Configuration: HUGSIM_* environment variables, set by scripts/hugsim/zs_run.py --agent preset.
repo=$(cd "$(dirname "$0")/../.." && pwd)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 exec "${DATA_DIR:-$HOME/data}/envs/hugsim/bin/python" -u \
  "$repo/scripts/hugsim/preset_agent.py" --output "$2"
