#!/usr/bin/env bash
# SLOW-NOVELTY (plans/2026-10-14-slow-novelty-prereg.md; rf_slownov.py docstring). Box, repo root, CPU only (outside the GPU pool; check
# `python -m jevdrive.cl top` for the CPU margin first).   rf_slownov.sh extract [workers] | analyse
set -euo pipefail; cd "$(dirname "$0")/../../.."
M=$DATA_DIR/runs/route_mem/read; O=$DATA_DIR/runs/route_follow/slownov/data; P=$DATA_DIR/third_party/alpasim/.venv/bin/python
X=${SLOWNOV_PY:-experiments/route_follow/scripts/rf_slownov.py}
case $1 in
extract)
  $P $X extract --out $O --workers ${2:-12} --runs "RSG=$(ls -d $M/RSG-L18-s0/c*/run | paste -sd,)" \
     "RSG+hold=$(ls -d $M/RSG-L18-s0-vhold/c*/run | paste -sd,)" "RSG+holdlead=$(ls -d $M/RSG-L18-s0-vholdl/c*/run | paste -sd,)" ;;
analyse)
  $P $X analyse --in $O --out $O/../report --figs $O/../figs ;;
esac
