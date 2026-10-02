#!/usr/bin/env bash
# Selector arm of the history de-rotation test (plans/2026-10-03-history-derotate-plan.md, addendum 2): own Cinque server
# (with lat_std4), sel3 on the 10 spin scenarios, then the non-spin scenarios in derot_rest54.txt order until STOP_AT (box clock).
# Usage (box, tmux):  GPU=<card> [WORKERS=2] [STOP_AT=07:25] experiments/hugsim/scripts/derot_sel.sh [out_dir]
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-derot}
S=experiments/hugsim/scripts
HPY=$DATA_DIR/envs/hugsim/bin/python
SOCK=$OUT/servers/cinque-sel.sock
rm -f "$OUT/servers/cinque-sel.ready" "$OUT/SEL_DONE" "$OUT/SEL_ERROR"
CUDA_VISIBLE_DEVICES=$GPU setsid "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/archive/hugsim_zs_server.py cinque \
    --socket "$SOCK" --ready-file "$OUT/servers/cinque-sel.ready" > "$OUT/servers/cinque-sel.log" 2>&1 &
srv=$!
trap 'kill -- -$srv 2>/dev/null' EXIT
until [[ -f $OUT/servers/cinque-sel.ready ]]; do sleep 5; kill -0 $srv 2>/dev/null || { echo server died > "$OUT/SEL_ERROR"; exit 3; }; done
OPTS='{"derot_below": 3.0, "derot_sel": 0.6}'
run() { timeout $(( $(date -d "${STOP_AT:-07:25}" +%s) - $(date +%s) )) $HPY experiments/hugsim/archive/zs_run.py run --out "$OUT" \
        --agent cinque --controller fixed --gpu "$GPU" --workers "${WORKERS:-2}" --scenarios "$1" --socket "$SOCK" --opts "$OPTS" --tag cinque-fixed-sel3; }
echo "$(date +%T) sel3 stage 1" > "$OUT/SEL_STATUS"
run $S/derot_spin10.txt
echo "$(date +%T) sel3 stage 2" > "$OUT/SEL_STATUS"
run $S/derot_rest54.txt
echo "$(date +%T) sel3 stopped" > "$OUT/SEL_STATUS"
touch "$OUT/SEL_DONE"
