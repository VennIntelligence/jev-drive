#!/usr/bin/env bash
# Stage A on Waymo E2E, half-val dress rehearsal: split -> vocabulary table (-> heads, once features exist).
# Usage: scripts/waymo_stage_a.sh [--steps split,vocab] [--ks ...]   (args go to jevdrive.waymo_stage_a)
# Start it with: scripts/tmux_run.sh wsa scripts/waymo_stage_a.sh [...]. Run outputs: docs/long-runs.md.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" PYTHONWARNINGS=ignore::FutureWarning
cd "$(dirname "$0")/.."
uv sync -q
exec uv run python -m jevdrive.waymo_stage_a "$@"
