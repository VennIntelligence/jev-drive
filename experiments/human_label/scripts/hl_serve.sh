#!/usr/bin/env bash
# (Re)start hl_server.py in tmux session `label` (CPU only), log appended to <pack>/../server.log, password from <pack>/../.env.
#   hl_serve.sh <pack dir> <port> <python with aiohttp>
#   box:   hl_serve.sh $DATA_DIR/runs/human_label/b1 6006 $DATA_DIR/envs/garage/bin/python
#   Tokyo: hl_serve.sh /data/runs/human_label/b1 8790 /data/envs/wheel/bin/python
set -euo pipefail
PACK=$(realpath "$1"); PORT=$2; PY=$3
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$PACK")
tmux kill-session -t label 2>/dev/null || true
tmux new-session -d -s label "cd '$HERE' && '$PY' hl_server.py --pack '$PACK' --port $PORT --env-file '$ROOT/.env' 2>&1 | tee -a '$ROOT/server.log'"
sleep 2
tail -n 3 "$ROOT/server.log"
