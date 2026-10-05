#!/usr/bin/env bash
# CARLA pair material at the open-loop-aligned rig, one self-advancing chain: render (one CARLA server + client per shard) -> pack -> stats.
# The plan is the existing one (same poses as the 1.22 m set): <root>/plan.path must exist (copy it from carla_pairs_<tag>/plan.path).
# Usage (box, in tmux via scripts/tmux_run.sh): run_carla_pairs_ol.sh <root tag> "<gpu> <idx> <cpus>" "<gpu> <idx> <cpus>" ...   (specs from cards held outside the pool, `python -m jevdrive.cl hold`)
#   e.g. run_carla_pairs_ol.sh s10000ol "0 160 0-3" "0 161 4-7" "1 184 52-55"
# Resumable: the renderer skips poses that already have frames. STATUS / DONE / ERROR in the run root.
set -uo pipefail
TAG=$1; shift
D=${DATA_DIR:?}; R=$D/runs/op_route_cmd/carla_pairs_$TAG
PY=$D/envs/carla/bin/python; PYJ=$D/envs/jevdrive/bin/python; S=experiments/op_route_cmd/scripts
fail() { echo "$1" > "$R/ERROR"; echo "ERROR: $1" > "$R/STATUS"; exit 1; }
rm -f "$R/ERROR" "$R/DONE"
P=$(cat "$R/plan.path"); K=$#; i=0; pids=()
echo "rendering $K shards" > "$R/STATUS"
for spec in "$@"; do
  read -r gpu idx cpus <<< "$spec"
  $PY $S/carla_pairs_render_ol.py --plan "$P" --out "$R/render" --gpu "$gpu" --idx "$idx" --cpus "$cpus" --shard "$i/$K" --hero > "$R/render_$i.log" 2>&1 &
  pids+=($!); i=$((i+1)); sleep 8
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] || fail "a render shard failed (see render_*.log, render/ERROR_*)"
echo "packing" > "$R/STATUS"
$PYJ $S/carla_pairs_pack.py --plan "$P" --render "$R/render" --root "$R/packed" > "$R/pack.log" 2>&1 || fail "pack failed"
$PY $S/carla_pairs_stats.py "$R/render" > "$R/render_stats.txt" 2>&1
echo "done: $(tail -1 $R/pack.log)" > "$R/STATUS"; touch "$R/DONE"
