#!/usr/bin/env bash
# Wait for the GPU to be free of other Alpamayo work, then run the full 220-route
# Bench2Drive zero-shot exam for Alpamayo 1.5 (see todos/2026-09-24-zeroshot-exam/bench2drive.md).
# Box-side only; run inside tmux (see docs/long-runs.md). Polls every 60 s and starts nothing
# until both conditions hold:
#   (a) no navsim_zs_alpamayo.py and no wod_zeroshot_alpamayo.py process is running
#   (b) free VRAM >= 55 GB
#
# Usage: scripts/tmux_run.sh b2d-alp-wait scripts/b2d_wait_and_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
D=$DATA_DIR/runs/zeroshot-exam/b2d
OUT=$D/full220-alpamayo
SERVER_WIN=b2d-alp-server
RUN_WIN=b2d-alp-full
VRAM_MIN_MB=55000
POLL_S=60

mkdir -p "$D"

other_alpamayo_running() {
    pgrep -f 'navsim_zs_alpamayo\.py' >/dev/null 2>&1 && return 0
    pgrep -f 'wod_zeroshot_alpamayo\.py' >/dev/null 2>&1 && return 0
    return 1
}

free_vram_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

echo "waiting: need no navsim_zs_alpamayo.py / wod_zeroshot_alpamayo.py and >= ${VRAM_MIN_MB} MiB free VRAM"
while true; do
    if ! other_alpamayo_running; then
        free=$(free_vram_mb)
        if (( free >= VRAM_MIN_MB )); then
            echo "conditions met: free VRAM ${free} MiB"
            break
        fi
    fi
    sleep "$POLL_S"
done

echo "## $(date +%Y-%m-%d\ %H:%M) [B2D] start: Alpamayo full 220-route run (1 server + 4 CARLA workers, ~50 GB, ~16 cores; est. 4.3 h idle / 6.6 h shared)" >> "$PLAN"

cat > "$D/agent-alpamayo-full.json" <<EOF
{"model": "alpamayo", "socket": "$D/alpamayo-full.sock", "plan_every": 5, "controller_preset": "carla",
 "controller_config": "$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json",
 "seed": 0, "dump_every": 0}
EOF

scripts/tmux_run.sh "$SERVER_WIN" env HF_ENDPOINT=https://hf-mirror.com \
    ~/data/third_party/alpamayo1.5/.venv/bin/python scripts/zeroshot_policy_server.py alpamayo \
    --socket "$D/alpamayo-full.sock" --ready-file "$D/alpamayo-full.ready"

echo "waiting for policy server ready file..."
while [ ! -e "$D/alpamayo-full.ready" ]; do sleep 5; done
echo "server ready, starting full run"

scripts/tmux_run.sh "$RUN_WIN" env DATA_DIR="$DATA_DIR" ~/data/envs/carla/bin/python scripts/b2d_run.py \
    --workers 4 --server-index 240 --agent scripts/b2d_zeroshot_agent.py \
    --agent-config "$D/agent-alpamayo-full.json" --decimate 2 --no-spectator --max-attempts 2 --out "$OUT"

echo "run started in jev:$RUN_WIN, watching for completion (resumable: reruns skip done/<id>.json)"
while pane_pid=$(tmux list-panes -t "jev:$RUN_WIN" -F '#{pane_pid}' 2>/dev/null); do
    [ -n "$pane_pid" ] || break
    pgrep -P "$pane_pid" >/dev/null 2>&1 || break
    sleep "$POLL_S"
done

tmux kill-window -t "jev:$SERVER_WIN" 2>/dev/null || true

summary="$OUT/summary.json"
if [ -f "$summary" ]; then
    finished=$(python3 -c "import json;print(json.load(open('$summary')).get('routes_finished','?'))" 2>/dev/null || echo "?")
else
    finished="?"
fi

echo "## $(date +%Y-%m-%d\ %H:%M) [B2D] Alpamayo full run finished (routes_finished=${finished}); see $OUT/summary.json, results in $OUT" >> "$PLAN"
echo "done"
