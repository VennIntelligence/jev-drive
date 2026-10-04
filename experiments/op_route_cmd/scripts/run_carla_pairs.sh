#!/usr/bin/env bash
# CARLA pair material, one self-advancing chain: plan -> render (N shards, one CARLA server each) -> pack.
# Usage (box, in tmux via scripts/tmux_run.sh): run_carla_pairs.sh <tag> <n-poses> <gpu> "<idx0> <cpus0>" "<idx1> <cpus1>" ...
#   e.g. run_carla_pairs.sh s2000 2000 2 "240 102-103,206-207" "241 156-159"
# Resumable: the plan is reused if it exists, the renderer skips poses that already have frames. STATUS / DONE / ERROR in the run root.
set -uo pipefail
TAG=$1; N=$2; GPU=$3; shift 3
D=${DATA_DIR:?}; R=$D/runs/op_route_cmd/carla_pairs_$TAG; mkdir -p "$R"
PY=$D/envs/carla/bin/python; PYJ=$D/envs/jevdrive/bin/python; S=experiments/op_route_cmd/scripts
fail() { echo "$1" > "$R/ERROR"; echo "ERROR: $1" > "$R/STATUS"; exit 1; }
rm -f "$R/ERROR" "$R/DONE"
if [ ! -s "$R/plan.path" ]; then
  echo "planning" > "$R/STATUS"
  T=$(ls -d $D/runs/op_route_cmd/carla_topo/*/ | while read d; do [ -e $d/DONE ] && echo $d; done | tail -1)
  $PY $S/carla_pairs_plan.py --topo "$T" --n-poses "$N" --tag "$TAG" > "$R/plan.log" 2>&1 || fail "plan failed"
  ls -d $D/runs/op_route_cmd/carla_plan_$TAG/*/poses.pkl | tail -1 > "$R/plan.path"
fi
P=$(cat "$R/plan.path"); K=$#; i=0; pids=()
echo "rendering $K shards" > "$R/STATUS"
for spec in "$@"; do
  read -r idx cpus <<< "$spec"
  $PY $S/carla_pairs_render.py --plan "$P" --out "$R/render" --gpu "$GPU" --idx "$idx" --cpus "$cpus" --shard "$i/$K" --hero > "$R/render_$i.log" 2>&1 &
  pids+=($!); i=$((i+1)); sleep 20
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] || fail "a render shard failed (see render_*.log, render/ERROR_*)"
echo "packing" > "$R/STATUS"
$PYJ $S/carla_pairs_pack.py --plan "$P" --render "$R/render" --root "$R/packed" > "$R/pack.log" 2>&1 || fail "pack failed"
$PY $S/carla_pairs_stats.py "$R/render" > "$R/render_stats.txt" 2>&1
echo "done: $(tail -1 $R/pack.log)" > "$R/STATUS"; touch "$R/DONE"
