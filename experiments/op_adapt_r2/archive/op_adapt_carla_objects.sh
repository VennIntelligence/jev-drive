#!/usr/bin/env bash
# CARLA static scene geometry near the routes (prereg v4): one fresh CARLA server per town (GPU 3, rpc port 9950;
# one load_world per server; Low quality segfaulted on Town02), experiments/op_adapt_r2/archive/op_adapt_carla_objects.py, stop; up to 2 attempts per town.
# Usage (box): scripts/tmux_run.sh s-objs experiments/op_adapt_r2/archive/op_adapt_carla_objects.sh [town ...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=$DATA_DIR/runs/op_adapt_r2
D=$R/logs/objs
mkdir -p "$D"
rm -f "$D/DONE" "$D/ERROR"
PORT=9950
icd=/etc/vulkan/icd.d/nvidia_icd.json; [[ -e $icd ]] || icd=/usr/share/vulkan/icd.d/nvidia_icd.json
towns=("$@")
(( ${#towns[@]} )) || towns=($(ls "$R/maps/carla/points" | sed 's/\.npy$//'))
SPID=
stop() { [[ -n $SPID ]] && kill -- -"$SPID" 2>/dev/null; SPID=; sleep 5; }
trap stop EXIT
for t in "${towns[@]}"; do
  for attempt in 1 2; do
    [[ -f $R/maps/carla/objects/$t.npz ]] && break
    VK_ICD_FILENAMES=$icd setsid nohup taskset -c 0-23 "$DATA_DIR/third_party/carla/CARLA_0.9.15/CarlaUE4.sh" -RenderOffScreen \
      -nosound -carla-rpc-port=$PORT -graphicsadapter=3 -quality-level=Epic >"$D/server-$t.log" 2>&1 &
    SPID=$!
    for _ in $(seq 90); do (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break; sleep 2; done
    sleep 10
    taskset -c 0-23 "$DATA_DIR/envs/carla/bin/python" experiments/op_adapt_r2/archive/op_adapt_carla_objects.py --port $PORT \
      --points-dir "$R/maps/carla/points" --out-dir "$R/maps/carla/objects" --towns "$t" 2>&1 | tee -a "$D/log.txt"
    stop
  done
  [[ -f $R/maps/carla/objects/$t.npz ]] || { echo "$t" >> "$D/ERROR"; }
done
[[ -f $D/ERROR ]] || date +%F' '%T > "$D/DONE"
