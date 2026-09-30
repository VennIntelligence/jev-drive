#!/usr/bin/env bash
# op-adapt r2 package S, CPU chain (cores 0-23): route points -> CARLA rasters -> nuScenes scene files -> nuScenes
# rasters -> V1 -> V2 / V5b. Writes $DATA_DIR/runs/op_adapt_r2/logs/prep/{log.txt,events.jsonl,STATUS,DONE|ERROR}.
# Usage (box): scripts/tmux_run.sh s-prep scripts/op_adapt_score_prep.sh [step ...]
set -uo pipefail
cd "$(dirname "$0")/.."
D=$DATA_DIR/runs/op_adapt_r2/logs/prep
mkdir -p "$D"
rm -f "$D/DONE" "$D/ERROR"
PY=$DATA_DIR/envs/jevdrive/bin/python
steps=("$@")
(( ${#steps[@]} )) || steps=(points maps-carla nus-scenes maps-nus v1 v2)
ev() { printf '{"t": %s, "kind": "%s", "step": "%s"%s}\n' "$(date +%s)" "$1" "$2" "${3:-}" >> "$D/events.jsonl"; }
ev start chain
for s in "${steps[@]}"; do
  echo "$(date +%F' '%T) step $s" | tee -a "$D/log.txt"; echo "$s" > "$D/STATUS"; ev step_start "$s"
  if ! taskset -c 0-23 "$PY" -m jevdrive.op_adapt_score_data "$s" --workers 24 2>&1 | tee -a "$D/log.txt"; then
    ev end "$s" ', "ok": false'; echo "$s" > "$D/ERROR"; exit 1
  fi
  ev step_end "$s"
done
ev end chain ', "ok": true'
date +%F' '%T > "$D/DONE"
