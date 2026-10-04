#!/usr/bin/env bash
# WOD-E2E train + val route labels (CPU). Usage on the box: scripts/tmux_run.sh route_wod experiments/op_route_cmd/scripts/run_wod.sh [workers]
set -euo pipefail
exec $DATA_DIR/envs/jevdrive/bin/python experiments/op_route_cmd/scripts/route_wod.py --workers "${1:-32}"
