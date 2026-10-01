#!/usr/bin/env bash
# P0: decisions 3d and 20 again with the head fitted on the train split (fc65452:todos/2026-09-22-p0-train-split-recheck.md).
# Usage: experiments/prediag/archive/waymo_p0.sh [--steps mlp,cls] [...]   (args go to jevdrive.waymo_p0)
# Start it with: scripts/tmux_run.sh p0 experiments/prediag/archive/waymo_p0.sh [...]. Run outputs: docs/long-runs.md.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied so torch cannot see the card at all, because another experiment
# owns it. Pass DATA_DIR=<snapshot> (experiments/probe_planner_v0/archive/snapshot_processed.sh p0) to pin the index, past, future and
# rater files against a concurrent reindex.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$DATA_DIR/envs/jevdrive}"
export PYTHONWARNINGS=ignore::FutureWarning CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")/../../.."
uv sync -q
exec uv run python -m jevdrive.waymo_p0 "$@"
