#!/usr/bin/env bash
# HUGSIM job of the guard line `hugsim` (cllib.hugsim_unit, one GPU-pool job): one resident Cinque server (the candidate's
# serving ONNX if $ONNX is set, shipped otherwise) and zs_run.py with interface preset `spec` on the scenarios of $SCEN, tag $TAG.
# Resumable (zs_run skips finished scenarios). Env: DATA_DIR GPU OUT SCEN TAG [ONNX] [WORKERS=5].
# Same server / zs_run calls as experiments/leaderboard_audit/scripts/unified_hugsim_chain.sh (arm d118 = spec) and
# experiments/op_adapt_h/scripts/h_hugsim.sh (--onnx); the server is killed above 40 GB RSS.
set -euo pipefail
: "${DATA_DIR:?}" "${OUT:?}" "${SCEN:?}" "${TAG:?}"
cd "$(dirname "$0")/../../.."
# cllib supplies the registered candidate name, so its ONNX and protocol resolve in the same registry as every bench run.
source scripts/bench_lane.sh
bench_hugsim "${BENCH_MODEL:-cinque}" spec "$TAG" "$SCEN" "${WORKERS:-5}"
