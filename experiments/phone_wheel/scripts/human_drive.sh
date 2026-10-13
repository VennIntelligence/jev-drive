#!/usr/bin/env bash
# Drive Bench2Drive routes by hand with the phone wheel (lib/human_agent.py), Tokyo box. Starts tmux session `human`:
#   human:viewer  the "human drive (phone wheel)" window on DISPLAY=:0 (openpilot's wide model frame + route + wheel panel)
#   human:drive   scripts/b2d_run.py with its own off-screen CARLA server, then the results table
# usage: human_drive.sh [route ids ...]   default: the openpilot monitor loop's 11 routes, in its order
# env: HUMAN_GPU (0) card = -graphicsadapter; HUMAN_SERVER_INDEX (4: rpc 2200, tm 8200; the openpilot monitor runs use 0 and 2);
#      HUMAN_CONFIG agent config JSON (steer_gain, steer_expo, realtime, ...; see lib/human_agent.py); HUMAN_MIN_FREE_MB (8000)
# Results: /data/runs/human_drive/<stamp>/{results.md,results.json}, per attempt human.jsonl / human_summary.json.
set -uo pipefail
# $LIVE/launcher.json tells the viewer what the launcher is doing: running (run dir, route list) / finished / refused
status() { python3 -c 'import json,sys,time; print(json.dumps(dict(state=sys.argv[1], msg=sys.argv[2], run=sys.argv[3], routes=sys.argv[4].split(","), t=time.time())))' "$@" > $LIVE/launcher.tmp && mv $LIVE/launcher.tmp $LIVE/launcher.json; }
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
H=/data/runs/human_drive LIVE=/data/runs/human_drive/live
GPU=${HUMAN_GPU:-0} IDX=${HUMAN_SERVER_INDEX:-4}
export DISPLAY=:0 XAUTHORITY=${XAUTHORITY:-/run/user/1001/gdm/Xauthority} DATA_DIR=/data
export CARLA_ROOT=/data/third_party/carla/CARLA_0.9.15 BENCH2DRIVE_ROOT=/data/third_party/Bench2Drive

if [[ ${1:-} == --run ]]; then          # inside human:drive
  OUT=$2; IDS=$3
  export B2D_SENSOR_TICK=1 B2D_PREVIEW_DIR=$LIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  cd "$REPO"
  (  # the server must sit on card $GPU: CARLA ignores CUDA_VISIBLE_DEVICES and the two-card -graphicsadapter map is unverified
    for _ in $(seq 120); do
      sleep 5
      pid=$(cat $OUT/servers/carla-$IDX.pid 2>/dev/null) || continue
      kids=$(pgrep -g $pid -d'|' CarlaUE4) || continue
      card=$(nvidia-smi | grep -E "^\|\s+[0-9]+ .* ($kids) " | awk '{print $2}' | head -1)
      [[ -n $card ]] || continue
      echo "$(date '+%F %T') CARLA server ($kids) on card $card, wanted $GPU" | tee -a $OUT/gpu_check.txt
      [[ $card == "$GPU" ]] || { echo "WRONG CARD: stopping the run" | tee -a $OUT/gpu_check.txt; pkill -TERM -f "b2d_run.py.*$OUT"; }
      exit 0
    done; echo "server never seen in nvidia-smi" >> $OUT/gpu_check.txt ) &
  /data/envs/carla/bin/python scripts/b2d_run.py --routes $BENCH2DRIVE_ROOT/leaderboard/data/bench2drive_0.0.4_val.xml \
    --route-ids "$IDS" --workers 1 --server-index $IDX --gpu-rank $GPU --tm-seed 0 --no-reap --no-spectator --client-threads 8 \
    --max-attempts 2 --stall-s 600 --route-timeout-s 3600 --out "$OUT" --python /data/envs/carla/bin/python \
    --agent experiments/phone_wheel/lib/human_agent.py ${HUMAN_CONFIG:+--agent-config $HUMAN_CONFIG} --fast-copy --cache-lights \
    2>&1 | tee -a "$OUT/runner.log"
  /data/envs/carla/bin/python experiments/phone_wheel/scripts/human_results.py "$OUT" 2>&1 | tee "$OUT/results.txt"
  date > "$OUT/DONE"
  status finished "all routes done: $OUT/results.md" "$OUT" "$IDS"
  exit 0
fi

mkdir -p "$LIVE"
tmux has-session -t human 2>/dev/null || tmux new-session -d -s human -n main
if ! tmux list-windows -t human -F '#{window_name}' | grep -qx viewer; then
  tmux new-window -d -t human -n viewer "cd $REPO && SDL_VIDEO_WINDOW_POS=${HUMAN_WIN_POS:-1340,40} \
    /data/envs/carla/bin/python experiments/phone_wheel/scripts/human_viewer.py --dir $LIVE --window-size ${HUMAN_WIN_SIZE:-2480 1736}; exec bash"
fi
ids=$(echo "${*:-24721 24944 27005 24416 27870 24519 26023 334 24948 28147 27297}" | tr ' ' ',' | tr -s ,)
free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free < ${HUMAN_MIN_FREE_MB:-8000} )); then
  echo "card $GPU has $free MiB free (< ${HUMAN_MIN_FREE_MB:-8000}): not starting"; nvidia-smi
  status refused "card $GPU has only $free MiB free (needs ${HUMAN_MIN_FREE_MB:-8000})" "" "$ids"; exit 1
fi
port=$((2000 + 50 * IDX))
if ss -ltn | grep -q ":$port "; then echo "port $port is taken: set HUMAN_SERVER_INDEX"; status refused "port $port taken (another run?)" "" "$ids"; exit 1; fi
OUT=$H/$(date +%m%d-%H%M%S); mkdir -p "$OUT"
status running "starting CARLA" "$OUT" "$ids"
echo "routes $ids, card $GPU, server index $IDX (rpc $port) -> $OUT"
tmux kill-window -t human:drive 2>/dev/null
tmux new-window -d -t human -n drive "bash $REPO/experiments/phone_wheel/scripts/human_drive.sh --run $OUT $ids; exec bash"
echo "tmux: human:viewer, human:drive; results: $OUT/results.md"
