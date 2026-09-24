#!/usr/bin/env bash
# Wait for the GPU to be free of other Alpamayo work, then run the full 220-route
# Bench2Drive zero-shot exam for Alpamayo 1.5 (see todos/2026-09-24-zeroshot-exam/bench2drive.md).
# Box-side only; run inside tmux (see docs/long-runs.md). Polls every 60 s and starts nothing
# until both conditions hold:
#   (a) the NAVSIM exam has written $NAVSIM_DONE (all its Alpamayo phases finished; a bare
#       process check is not enough because NAVSIM restarts processes between phases)
#   (b) no navsim_zs_alpamayo.py / wod_zeroshot_alpamayo.py process and no queued WOD waiter
#       (the WOD ADE-extra job waits as a bash loop before its python process exists)
#   (c) free VRAM >= 55 GB
#
# Env overrides:
#   GPU=<rank>    which card to use (CUDA_VISIBLE_DEVICES for the Alpamayo server,
#                 --gpu-rank / -graphicsadapter for every CARLA server). Default 0.
#   NO_WAIT=1     skip the wait loop entirely and start immediately (e.g. once a second
#                 card makes GPU contention moot). Default 0 (wait as below).
#
# Usage: scripts/tmux_run.sh b2d-alp-wait scripts/b2d_wait_and_run.sh
#        GPU=1 NO_WAIT=1 scripts/tmux_run.sh b2d-alp-wait scripts/b2d_wait_and_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
NO_WAIT=${NO_WAIT:-0}

PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
D=$DATA_DIR/runs/zeroshot-exam/b2d
OUT=$D/full220-alpamayo
SERVER_WIN=b2d-alp-server
RUN_WIN=b2d-alp-full
VRAM_MIN_MB=55000
NAVSIM_DONE=$DATA_DIR/runs/zeroshot-exam/navsim-alpamayo.done
POLL_S=60

mkdir -p "$D"

other_alpamayo_running() {
    [[ -f $NAVSIM_DONE ]] || return 0
    pgrep -f 'ADE-extra' >/dev/null 2>&1 && return 0
    pgrep -f 'navsim_zs_alpamayo\.py' >/dev/null 2>&1 && return 0
    pgrep -f 'wod_zeroshot_alpamayo\.py' >/dev/null 2>&1 && return 0
    return 1
}

free_vram_mb() {
    nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

if [[ $NO_WAIT == 1 ]]; then
    echo "NO_WAIT=1: starting immediately on GPU $GPU, no polling"
else
    echo "waiting: need $NAVSIM_DONE, no NAVSIM/WOD Alpamayo process or WOD waiter, and >= ${VRAM_MIN_MB} MiB free VRAM on GPU $GPU"
    while true; do
        if ! other_alpamayo_running; then
            free=$(free_vram_mb)
            if (( free >= VRAM_MIN_MB )); then
                echo "conditions met: free VRAM ${free} MiB on GPU $GPU"
                break
            fi
        fi
        sleep "$POLL_S"
    done
fi

# Stale sock/ready files from a previous (e.g. crashed) run would make the ready-file
# check below pass instantly against a server that is not actually up.
rm -f "$D/alpamayo-full.sock" "$D/alpamayo-full.ready"

echo "## $(date +%Y-%m-%d\ %H:%M) [B2D] start: Alpamayo full 220-route run on GPU $GPU (1 server + 4 CARLA workers, ~50 GB, ~16 cores; est. ~4.3 h idle)" >> "$PLAN"

cat > "$D/agent-alpamayo-full.json" <<EOF
{"model": "alpamayo", "socket": "$D/alpamayo-full.sock", "plan_every": 5, "controller_preset": "carla",
 "controller_config": "$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json",
 "seed": 0, "dump_every": 0}
EOF

scripts/tmux_run.sh "$SERVER_WIN" env HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU \
    ~/data/third_party/alpamayo1.5/.venv/bin/python scripts/zeroshot_policy_server.py alpamayo \
    --socket "$D/alpamayo-full.sock" --ready-file "$D/alpamayo-full.ready"

echo "waiting for policy server ready file..."
while [ ! -e "$D/alpamayo-full.ready" ]; do sleep 5; done
echo "server ready, starting full run"

scripts/tmux_run.sh "$RUN_WIN" env DATA_DIR="$DATA_DIR" ~/data/envs/carla/bin/python scripts/b2d_run.py \
    --towns all --workers 4 --server-index 240 --gpu-rank "$GPU" --agent scripts/b2d_zeroshot_agent.py \
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
