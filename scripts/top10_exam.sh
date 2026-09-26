#!/usr/bin/env bash
# Top-10 exam runner in a model's own env and repo (todos/2026-09-26-top10-intersection.md, [T1]).
# Usage (on the box, inside scripts/tmux_run.sh): scripts/top10_exam.sh <sparsedrivev2|ztrs> <runner args ...>
#   CUDA_VISIBLE_DEVICES=4 scripts/top10_exam.sh sparsedrivev2 --check 256
set -euo pipefail
model=$1; shift
repo=$(cd "$(dirname "$0")/.." && pwd)
case $model in
  sparsedrivev2) env=sparsedrivev2; code=sparsedrivev2 ;;
  ztrs) env=gtrs; code=ztrs ;;
  *) echo "unknown model $model" >&2; exit 1 ;;
esac
cd "$DATA_DIR/third_party/$code"
export OPENBLAS_CORETYPE=Haswell NAVSIM_DEVKIT_ROOT=$PWD NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim
exec "$DATA_DIR/envs/$env/bin/python" "$repo/scripts/top10_exam_infer.py" --model "$model" "$@"
