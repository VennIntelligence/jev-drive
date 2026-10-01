#!/usr/bin/env bash
# Rerun planner v0 end to end on the box: targets -> vocabulary -> heads -> tables.
# Usage: experiments/probe_planner_v0/archive/planner_v0.sh [--version v1.0-trainval] [--steps plan,bench] [...]  (args go to experiments.probe_planner_v0.archive.planner_v0)
# Start it with: scripts/tmux_run.sh planner experiments/probe_planner_v0/archive/planner_v0.sh [...]. Run outputs: docs/long-runs.md.
# Needs the cached features of experiments/probe_planner_v0/archive/probe_v0.sh --steps features.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" PYTHONWARNINGS=ignore::FutureWarning
cd "$(dirname "$0")/../../.."
uv sync -q
exec uv run python -m experiments.probe_planner_v0.archive.planner_v0 "$@"
