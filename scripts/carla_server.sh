#!/usr/bin/env bash
# Start / stop / check a headless CARLA 0.9.15 server on the box. See docs/carla.md.
# Usage: scripts/carla_server.sh start [index]   # index 0,1,2... -> rpc port 2000+4*index
#        scripts/carla_server.sh stop  [index]   # omit index to stop every server we started
#        scripts/carla_server.sh status
# One server per index: RPC port 2000+50i, traffic manager 8000+50i. The 50-port spacing is not
# cosmetic: a CARLA server claims several ports above its RPC port, and a traffic manager port
# OUTLIVES the server that owned it. See docs/carla.md.
set -euo pipefail

CARLA_ROOT=${CARLA_ROOT:-$DATA_DIR/third_party/carla/CARLA_0.9.15}
RUN_DIR=${CARLA_RUN_DIR:-$DATA_DIR/runs/carla}
QUALITY=${CARLA_QUALITY:-Epic}

port_of() { echo $((2000 + 50 * $1)); }

start() {
  local i=${1:-0} port; port=$(port_of "$i")
  mkdir -p "$RUN_DIR"
  local pidf="$RUN_DIR/carla-$i.pid" log="$RUN_DIR/carla-$i.log"
  if [[ -f $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "server $i already running (pid $(cat "$pidf"), port $port)"; return 0
  fi
  # Pin the NVIDIA ICD: mesa-vulkan-drivers also installs llvmpipe, which CARLA would happily
  # pick and then render at 1 FPS on the CPU.
  # Do NOT set SDL_VIDEODRIVER=offscreen: it makes CarlaUE4 exit 1 immediately (measured).
  # -RenderOffScreen already does the headless part; SDL is not involved.
  local nvidia_icd=/etc/vulkan/icd.d/nvidia_icd.json
  [[ -e $nvidia_icd ]] || nvidia_icd=/usr/share/vulkan/icd.d/nvidia_icd.json
  VK_ICD_FILENAMES="$nvidia_icd" \
  setsid nohup "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound \
    -carla-rpc-port="$port" -quality-level="$QUALITY" >"$log" 2>&1 &
  echo $! >"$pidf"   # setsid makes this pid the process-group leader, so stop() can kill the group
  echo "server $i starting (pid $(cat "$pidf"), rpc port $port, log $log)"
  for _ in $(seq 60); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && { echo "server $i up on $port"; return 0; }
    sleep 2
  done
  echo "server $i did not open port $port in 120 s; see $log" >&2; return 1
}

stop() {
  local pidf
  for pidf in "$RUN_DIR"/carla-${1:-*}.pid; do
    [[ -e $pidf ]] || continue
    local pid; pid=$(cat "$pidf")
    # Kill the whole process group: CarlaUE4.sh is a wrapper that forks the real binary.
    # Never pkill -f here, it would match other sessions' command lines (docs/long-runs.md).
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$pidf"
    echo "stopped $(basename "$pidf" .pid)"
  done
}

status() {
  pgrep -a -u "$(id -u)" CarlaUE4-Linux || echo "no CARLA server running"
}

case ${1:-} in
  start) shift; start "$@" ;;
  stop) shift; stop "$@" ;;
  status) status ;;
  *) sed -n '2,6p' "$0"; exit 1 ;;
esac
