#!/usr/bin/env bash
# History de-rotation at low speed in HUGSIM closed loop (plans/2026-10-03-history-derotate-plan.md), Cinque, PR #57 controller.
# Shared bench workers on the GPU pool; legacy tags/outputs retained. Previously one resident server on one card; stages run in order and are resumable (zs_run skips finished jobs):
#   1 derot3 + replay3 on the 10 PR #57 spin scenarios (in parallel)   2 derot3 on the other 54   3 base rerun on the 10
# Usage (box, tmux):  GPU=<card> [WORKERS=3] experiments/hugsim/scripts/derot_chain.sh [out_dir]
# Files: <out>/STATUS (current stage), DONE or ERROR at the end; results in <out>/results.csv.
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-derot}
S=experiments/hugsim/scripts
HPY=$DATA_DIR/envs/hugsim/bin/python
W=${WORKERS:-3}
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/ERROR"
source scripts/bench_lane.sh
fail() { echo "$(date +%T) $*" | tee "$OUT/ERROR"; exit 1; }
st() { echo "$(date +%T) $*" | tee "$OUT/STATUS"; }
run() {
    bench_hugsim cinque exam "$1" "$3" "$4" fixed "$2"
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
