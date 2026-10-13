#!/usr/bin/env bash
# Off-screen closed-loop test run on card $MON_GPU (server index 2: rpc port 2100), preview to $PREVIEW_DIR (default $V/live-test).
# usage: run_test.sh <out dir> <comma route ids> [agent .py] [agent config]   env: MAX_TICKS caps a route
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export $B2D_RUN_ENV B2D_PREVIEW_DIR=${PREVIEW_DIR:-$V/live-test} CARLA_WINDOWED=0
mkdir -p "$1"
$PY scripts/b2d_run.py --routes $XML --route-ids "$2" --workers 1 --server-index 2 --gpu-rank $MON_GPU --tm-seed 0 --no-reap \
  --client-threads 8 --max-attempts 2 --stall-s 480 --route-timeout-s 2400 --out "$1" --python $PY \
  --agent "${3:-experiments/b2d_monitor/lib/op_arb_viz_agent.py}" --agent-config "${4:-experiments/b2d_monitor/cfg/op_alone_smooth.json}" \
  --fast-copy --cache-lights ${MAX_TICKS:+--max-ticks $MAX_TICKS} 2>&1 | tee -a "$1/runner.log"
