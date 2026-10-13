#!/usr/bin/env bash
# Route loop for the monitor: one route per b2d_run Runner on one kept windowed CARLA server (viz_loop.py); the next route is the
# viewer picker's request ($V/live/request.json) or the next id of the list; one fresh output dir per pass ($V/runs/loop-<MMDD-HHMM>-p<n>).
# usage: loop.sh [comma route ids] [agent config]   defaults: the 11-route mix, cfg/op_alone_smooth.json (shipped Cinque, FIX1 longitudinal)
# env: VIZ_MODEL / VIZ_CONFIG (top-bar labels), VIZ_AGENT (agent .py), VIZ_REUSE_WORLD (1 = keep a loaded map; 0 for scored runs)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ids=${1:-$MON_IDS}
cfg=${2:-experiments/b2d_monitor/cfg/op_alone_smooth.json}
export $B2D_RUN_ENV B2D_PREVIEW_DIR=$V/live
export VIZ_CONFIG=${VIZ_CONFIG:-op_alone_smooth: spec lateral + FIX1 longitudinal} VIZ_MODEL=${VIZ_MODEL:-Cinque shipped}
exec $PY -u experiments/b2d_monitor/scripts/viz_loop.py "$ids" "$cfg" ${VIZ_AGENT:+--agent $VIZ_AGENT}
