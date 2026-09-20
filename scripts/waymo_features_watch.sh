#!/usr/bin/env bash
# Build Waymo frozen features as shards land, and keep doing it while the download runs.
#
# Every pass re-runs the incremental index BEFORE deciding what to extract. Without that the loop is
# incremental only with respect to extraction, not with respect to arrival: it asks the index what exists,
# the index still describes the shards it was built from, and the loop sleeps happily while new shards pile
# up unseen. That cost us two idle hours once; the fix is the first line of the loop body.
#
# Usage (on the box): scripts/tmux_run.sh wfeat scripts/waymo_features_watch.sh [passes]
# Env: INTERVAL (sleep seconds, default 600), SET (feature set name), BATCH (batch size).
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" PYTHONWARNINGS=ignore::FutureWarning
cd "$(dirname "$0")/.."
passes=${1:-200}; interval=${INTERVAL:-600}; set_name=${SET:-qwen_front3}; batch=${BATCH:-4}
status() { uv run python -m jevdrive.waymo_stage_a --steps status --set "$set_name"; }

idle=0
for ((i = 1; i <= passes; i++)); do
  uv run python -m jevdrive.waymo index || true          # pick up whatever landed since the last pass
  read -r disk0 idx0 built0 frames0 < <(status)
  uv run python -m jevdrive.waymo features_inc --splits val --batch-size "$batch" || true
  read -r disk1 idx1 built1 frames1 < <(status)
  echo "==> pass $i at $(date +%H:%M:%S): shards on disk $disk1, indexed $idx1 (+$((idx1 - idx0))), " \
       "features built $built1 (+$((built1 - built0))), frames indexed $frames1"
  if (( built1 > built0 || idx1 > idx0 )); then
    idle=0
  else
    idle=$((idle + interval))
    (( disk1 > built1 )) && echo "==> WARNING: $((disk1 - built1)) shards on disk are not built and nothing" \
      "moved for $((idle / 60)) min"
    (( idle >= 3600 )) && echo "==> WARNING: no progress for $((idle / 60)) min; is the download still running?"
  fi
  (( built1 >= disk1 && disk1 >= 93 )) && { echo "==> all $disk1 val shards built; stopping"; break; }
  sleep "$interval"
done
