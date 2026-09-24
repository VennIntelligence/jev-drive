#!/usr/bin/env bash
# Agent launcher in the shape closed_loop.py expects: `zsh <this> <cuda_id> <output_dir>`.
# Policy and scene come from the environment (HUGSIM_POLICY, HUGSIM_SCENE_DIR), set by run_smoke.sh.
repo=$(cd "$(dirname "$0")/../.." && pwd)
CUDA_VISIBLE_DEVICES=$1 exec "${DATA_DIR:-$HOME/data}/envs/hugsim/bin/python" -u \
  "$repo/scripts/hugsim/agent_client.py" --output "$2" --policy "${HUGSIM_POLICY:-route}"
