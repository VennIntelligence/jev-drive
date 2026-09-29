#!/usr/bin/env bash
# Cosmos full run, staged-launch checklist (todos/2026-09-28-cosmos-pilot.md, "full run"): the v2 readouts on every pair a
# stage finished (YOLO recall / hallucination, pixels outside the region and the edge ring, openpilot), then
# `python -m jevdrive.cosmos_full checklist`. The stage must have run with --keep-npy.
#   scripts/cosmos_full_check.sh <stage dir under runs/cosmos_full, e.g. stage10> [gpu]
# Out: research/results/cosmos/full/<stage>/ (per_pair_G4b.csv, checklist.json, checklist_pairs.csv), one WebP in
# research/figs/cosmos/.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
stage=$1 gpu=${2:-1} E=$DATA_DIR/envs
R=$DATA_DIR/runs/cosmos_full/$stage
export COSMOS_ROOT=$R COSMOS_RESULTS=$PWD/research/results/cosmos/full/$stage CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PWD
mkdir -p "$COSMOS_RESULTS"
pairs=$(for f in "$R"/pairs/*/done.json; do basename "$(dirname "$f")"; done | paste -sd, -)
echo "pair" > "$COSMOS_RESULTS/pairs.csv"; tr ',' '\n' <<< "$pairs" >> "$COSMOS_RESULTS/pairs.csv"
echo "pairs: $pairs"
$E/ultralytics/bin/python -m jevdrive.cosmos_eval detect --variant G4b --only "$pairs"
$E/r3d2/bin/python -m jevdrive.cosmos_eval pixels --variant G4b --only "$pairs" --gpu 0
$E/openpilot/bin/python scripts/cosmos_openpilot.py --variant G4b --pairs "$pairs"
$E/jevdrive/bin/python -m jevdrive.cosmos_eval report --variant G4b --only "$pairs"
$E/jevdrive/bin/python -m jevdrive.cosmos_full checklist --stage "$stage"
$E/jevdrive/bin/python -m jevdrive.cosmos_eval webp --variant G4b --only "${pairs%%,*}"
