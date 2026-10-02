#!/usr/bin/env bash
# History de-rotation at low speed in HUGSIM closed loop (plans/2026-10-03-history-derotate-plan.md), Cinque, PR #57 controller.
# One resident Cinque server on one card; stages run in order and are resumable (zs_run skips finished jobs):
#   1 derot3 + replay3 on the 10 PR #57 spin scenarios (in parallel)   2 derot3 on the other 54   3 base rerun on the 10
# Usage (box, tmux):  GPU=<card> [WORKERS=3] experiments/hugsim/scripts/derot_chain.sh [out_dir]
# Files: <out>/STATUS (current stage), DONE or ERROR at the end; results in <out>/results.csv.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-derot}
S=experiments/hugsim/scripts
HPY=$DATA_DIR/envs/hugsim/bin/python
W=${WORKERS:-3}
mkdir -p "$OUT/servers"
rm -f "$OUT/servers/cinque.ready" "$OUT/DONE" "$OUT/ERROR"
fail() { echo "$(date +%T) $*" | tee "$OUT/ERROR"; exit 1; }
st() { echo "$(date +%T) $*" | tee "$OUT/STATUS"; }
CUDA_VISIBLE_DEVICES=$GPU setsid "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/archive/hugsim_zs_server.py cinque \
    --socket "$OUT/servers/cinque.sock" --ready-file "$OUT/servers/cinque.ready" > "$OUT/servers/cinque.log" 2>&1 &
srv=$!
trap 'kill -- -$srv 2>/dev/null' EXIT
until [[ -f $OUT/servers/cinque.ready ]]; do sleep 5; kill -0 $srv 2>/dev/null || fail "server died"; done
st "server ready (pid $srv)"
$HPY experiments/hugsim/archive/zs_run.py setup-trees official fixed || fail "setup-trees"
run() {  # tag opts list workers
    $HPY experiments/hugsim/archive/zs_run.py run --out "$OUT" --agent cinque --controller fixed --gpu "$GPU" --workers "$4" \
        --scenarios "$3" --socket "$OUT/servers/cinque.sock" --opts "$2" --tag "$1"
}
st "stage 1: derot3 + replay3 on 10 spin scenarios"
run cinque-fixed-derot3 '{"derot_below": 3.0}' $S/derot_spin10.txt 2 & p1=$!
run cinque-fixed-replay3 '{"derot_below": 3.0, "derot_rotate": false}' $S/derot_spin10.txt 2 & p2=$!
wait $p1 || fail "stage 1 derot3"
wait $p2 || fail "stage 1 replay3"
st "stage 2: derot3 on the other 54"
run cinque-fixed-derot3 '{"derot_below": 3.0}' $S/derot_rest54.txt "$W" || fail "stage 2"
st "stage 3: base rerun on 10 spin scenarios"
run cinque-fixed-base '{}' $S/derot_spin10.txt "$W" || fail "stage 3"
st "done"
touch "$OUT/DONE"
