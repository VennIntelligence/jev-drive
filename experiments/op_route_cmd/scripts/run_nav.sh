#!/usr/bin/env bash
# navtrain route labels (CPU). Usage on the box: scripts/tmux_run.sh route_nav experiments/op_route_cmd/scripts/run_nav.sh [workers]
set -euo pipefail
export NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim
exec $DATA_DIR/envs/navsim2/bin/python experiments/op_route_cmd/scripts/route_nav.py --workers "${1:-32}"
