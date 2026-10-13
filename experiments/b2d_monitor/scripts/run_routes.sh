#!/usr/bin/env bash
# One-shot windowed Bench2Drive closed loop on card $MON_GPU (all listed routes in one b2d_run, CARLA stopped at the end, the map loaded
# per route as the leaderboard ships). The monitor loop does not use it (loop.sh -> viz_loop.py); kept for direct runs.
# usage: run_routes.sh <out dir> <comma route ids> [agent .py] [agent config]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export $B2D_RUN_ENV B2D_PREVIEW_DIR=$V/live
mkdir -p "$1"
$PY scripts/b2d_run.py --routes $XML --route-ids "$2" --workers 1 --server-index 0 --gpu-rank $MON_GPU --windowed --tm-seed 0 --no-reap \
  --client-threads 8 --max-attempts 2 --stall-s 480 --route-timeout-s 2400 --out "$1" --python $PY \
  --agent "${3:-experiments/b2d_monitor/lib/op_arb_viz_agent.py}" --agent-config "${4:-experiments/b2d_monitor/cfg/op_alone_smooth.json}" \
  --fast-copy --cache-lights --server-args "-WinX=40 -WinY=1300" 2>&1 | tee -a "$1/runner.log"
