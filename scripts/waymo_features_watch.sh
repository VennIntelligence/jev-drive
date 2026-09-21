#!/usr/bin/env bash
# Build Waymo frozen features as shards land, and keep doing it while the download runs.
#
# The loop itself is `jevdrive.waymo features_inc --watch` (this script only sets the environment), and it
# re-runs the index BEFORE deciding what to extract in every pass. Without that it would be incremental only
# with respect to extraction, not with respect to arrival: it would ask the index what exists, the index
# would still describe the shards it was built from, and the loop would sleep happily while new shards piled
# up unseen. That cost us two idle hours once.
#
# The model is loaded and compiled once for the whole run. Once extraction has caught up with the download,
# a pass is a single shard (~3.5 min) and reloading the model every pass would be pure overhead.
#
# Usage (on the box): scripts/tmux_run.sh wfeat scripts/waymo_features_watch.sh [split]
# Env: INTERVAL (sleep seconds between passes, default 600), BATCH (batch size; 4 is the only value that
#      reproduces the val features bit for bit -- see docs/waymo-e2e.md "Frozen features"),
#      WORKERS (DataLoader workers; unset = the measured default), PASSES (pass limit).
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" PYTHONWARNINGS=ignore::FutureWarning
cd "$(dirname "$0")/.."
split=${1:-val}
exec uv run python -m jevdrive.waymo features_inc --splits "$split" --watch "${INTERVAL:-600}" \
  --batch-size "${BATCH:-4}" --passes "${PASSES:-2000}" ${WORKERS:+--workers "$WORKERS"}
