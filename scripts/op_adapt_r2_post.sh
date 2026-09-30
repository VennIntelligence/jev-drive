#!/usr/bin/env bash
# op-adapt r2: readouts of one finished training run (after `readout.py eval --model TAG` has written its feature pass):
#   scripts/op_adapt_r2_post.sh <tag> <cores>      (started by the full lane on its spare core slice; idempotent via POST_DONE)
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_CORETYPE=Haswell
TAG=$1; CORES=${2:?cores}
R=$DATA_DIR/runs/op_adapt_r2
TR=$DATA_DIR/envs/op-train/bin/python
mkdir -p "$R/readout/$TAG"
[[ -f $R/readout/$TAG/POST_DONE ]] && exit 0
"$TR" scripts/op_adapt_r2_readout.py read --model "$TAG" || exit 1
# navsim scores the port and the model poses in shared directories: one at a time
flock "$R/readout/navsim.lock" "$TR" scripts/op_adapt_r2_readout.py navsim --split navtest --models "$TAG" --cpus "$CORES" || exit 1
date +%F' '%T > "$R/readout/$TAG/POST_DONE"
