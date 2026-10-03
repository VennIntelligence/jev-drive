#!/usr/bin/env bash
# One card's share of the real-vs-render run for a HUGSIM dataset (waymo | pandaset | kitti360): build real frames (when the raw files are there),
# then openpilot stream + probe, scene by scene; skips what exists, so it can be rerun as downloads finish.
# Usage (on the box, in tmux): rvr_ds_chain.sh <dataset> <card> <ncards> [workers]
set -u
ds=$1; card=$2; n=$3; w=${4:-16}
export DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
repo=$(cd "$(dirname "$0")/../../.." && pwd)
S=$repo/experiments/leaderboard_audit/scripts
i=0
for scene in $(ls "$DATA_DIR/datasets/hugsim/scenes/$ds" | grep -v zip | grep -v "^$ds\$"); do
  if (( i++ % n != card )); then continue; fi
  [[ -f $DATA_DIR/runs/real_vs_render/$ds/$scene/render.npy ]] || { echo "no render $scene"; continue; }
  [[ -f $DATA_DIR/runs/real_vs_render/$ds/$scene/real.npy ]] || $DATA_DIR/envs/waymo/bin/python $S/rvr_ds_real.py "$ds" build "$scene"
  [[ -f $DATA_DIR/runs/real_vs_render/$ds/$scene/real.npy ]] || continue
  for cmd in stream probe; do
    CUDA_VISIBLE_DEVICES=$card $DATA_DIR/envs/openpilot/bin/python $S/rvr_ds_op.py "$ds" $cmd "$scene" --workers "$w" 2>&1 | grep -a -E "stream|probe|Error|Traceback" 
  done
done
echo CHAIN_DONE "$ds" "$card"
