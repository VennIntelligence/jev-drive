#!/usr/bin/env bash
# openpilot in Bench2Drive after the adapter fixes (todos/2026-09-24-zeroshot-exam/openpilot-migration.md):
# one openpilot policy server + one CARLA worker per phase, on GPU $GPU, the 5 pre-registered smoke routes.
#   scripts/tmux_run.sh opm-carla scripts/zeroshot_b2d_opfix.sh [phase ...]
# phases: shadow fixed origin-only curvature cinque shadow-h122 shadow-h181   (default: all, in that order)
set -u
GPU=${GPU:-1}
D=$DATA_DIR/runs/zeroshot-exam/b2d-opfix
ROUTES=${ROUTES:-2390,24211,1711,2373,3564}
PY_OP=$DATA_DIR/envs/openpilot/bin/python
PY_CARLA=$DATA_DIR/envs/carla/bin/python
C=$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json
mkdir -p "$D"

config() {  # name model plan_every extra-json
  echo "{\"model\": \"$2\", \"socket\": \"$D/$2.sock\", \"controller_preset\": \"carla\", \"controller_config\": \"$C\",
 \"seed\": 0, \"plan_every\": $3, \"dump_every\": 5 $4}" > "$D/agent-$1.json"
}
FIX='"op_camera_tick": 0.05, "plan_origin": "rear", "warmup_s": 5.0'
config shadow lebowski 4 ", $FIX, \"drive\": \"oracle\""
config fixed lebowski 4 ", $FIX, \"drive\": \"model\""
config origin-only lebowski 1 ', "op_camera_tick": 0.2, "plan_origin": "rear", "warmup_s": 0.0, "drive": "model"'
config curvature lebowski 4 ", $FIX, \"drive\": \"model\", \"lateral\": \"curvature\""
config cinque cinque 1 ", $FIX, \"drive\": \"model\""
# camera height in the real 3D renderer (shadow mode): nominal comma 1.22 m placed at the front bumper line
# (x = 3.8 m, no hood in view) and a Waymo-like 1.81 m roof mount at the windshield-top x
config shadow-h122 lebowski 4 ", $FIX, \"drive\": \"oracle\", \"op_mount\": [3.8, 0.0, 1.22]"
config shadow-h181 lebowski 4 ", $FIX, \"drive\": \"oracle\", \"op_mount\": [1.779, 0.0, 1.81]"

server_pid=""
start_server() {  # model
  [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null && wait "$server_pid" 2>/dev/null
  rm -f "$D/$1.ready"
  CUDA_VISIBLE_DEVICES=$GPU $PY_OP scripts/zeroshot_policy_server.py "$1" --socket "$D/$1.sock" \
      --ready-file "$D/$1.ready" > "$D/server-$1.log" 2>&1 &
  server_pid=$!
  until [ -f "$D/$1.ready" ]; do sleep 2; kill -0 $server_pid || { echo "server $1 died"; exit 1; }; done
  echo "server $1 ready (pid $server_pid)"
}
trap '[ -n "$server_pid" ] && kill $server_pid' EXIT

phases=${*:-shadow fixed origin-only curvature cinque shadow-h122 shadow-h181}
current=""
for ph in $phases; do
  model=$(python3 -c "import json;print(json.load(open('$D/agent-$ph.json'))['model'])")
  [ "$model" != "$current" ] && start_server "$model" && current=$model
  echo "$(date '+%F %T') phase $ph"
  DATA_DIR=$DATA_DIR $PY_CARLA scripts/b2d_run.py --route-ids "$ROUTES" --workers 1 --server-index 300 \
      --gpu-rank "$GPU" --agent scripts/b2d_zeroshot_agent.py --agent-config "$D/agent-$ph.json" --decimate 4 \
      --no-spectator --max-attempts 2 --out "$D/$ph" || echo "phase $ph exited $?"
done
echo "$(date '+%F %T') all phases done"
