#!/usr/bin/env bash
# Rerun probe v0 end to end on the box: index -> labels -> features -> probes -> table.
# Usage: scripts/probe_v0.sh [--version v1.0-trainval] [--force] [--steps ...]   (args go to jevdrive.probe_v0)
# Start it with: scripts/tmux_run.sh probe scripts/probe_v0.sh [...]. Run outputs: docs/long-runs.md.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" PYTHONWARNINGS=ignore::FutureWarning
cd "$(dirname "$0")/.."
uv sync -q
exec uv run python -m jevdrive.probe_v0 "$@"
