#!/usr/bin/env bash
# One CARLA server (GPU 3, rpc port 9950) -> scripts/op_adapt_carla_objects.py over every town with route points -> stop.
# Usage (box): scripts/tmux_run.sh s-objs scripts/op_adapt_carla_objects.sh [town ...]
set -uo pipefail
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/op_adapt_r2
D=$R/logs/objs
mkdir -p "$D"
rm -f "$D/DONE" "$D/ERROR"
PORT=9950
icd=/etc/vulkan/icd.d/nvidia_icd.json; [[ -e $icd ]] || icd=/usr/share/vulkan/icd.d/nvidia_icd.json
VK_ICD_FILENAMES=$icd setsid nohup taskset -c 0-23 "$DATA_DIR/third_party/carla/CARLA_0.9.15/CarlaUE4.sh" -RenderOffScreen -nosound \
  -carla-rpc-port=$PORT -graphicsadapter=3 -quality-level=Low >"$D/server.log" 2>&1 &
SPID=$!
echo "$SPID" > "$D/server.pid"
trap 'kill -- -$SPID 2>/dev/null' EXIT
for _ in $(seq 90); do (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break; sleep 2; done
towns=(); (( $# )) && towns=(--towns "$@")
if taskset -c 0-23 "$DATA_DIR/envs/carla/bin/python" scripts/op_adapt_carla_objects.py --port $PORT \
     --points-dir "$R/maps/carla/points" --out-dir "$R/maps/carla/objects" "${towns[@]}" 2>&1 | tee -a "$D/log.txt"; then
  date +%F' '%T > "$D/DONE"
else
  echo fail > "$D/ERROR"
fi
