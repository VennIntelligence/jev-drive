#!/usr/bin/env bash
# Selector arm of the history de-rotation test (plans/2026-10-03-history-derotate-plan.md, addendum 2): own Cinque server
# (with lat_std4), sel3 on the 10 spin scenarios, then the non-spin scenarios in derot_rest54.txt order until STOP_AT (box clock).
# Usage (box, tmux):  [WORKERS=2] [STOP_AT=07:25] experiments/hugsim/scripts/derot_sel.sh [out_dir]
# Run as an orchestrator outside a GPU lease; bench places and cancels its jobs.
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-derot}
S=experiments/hugsim/scripts
source scripts/bench_lane.sh
mkdir -p "$OUT"
rm -f "$OUT/SEL_DONE" "$OUT/SEL_ERROR"
OPTS='{"derot_below": 3.0, "derot_sel": 0.6}'
run() {
    local remain=$(( $(date -d "${STOP_AT:-07:25}" +%s) - $(date +%s) ))
    (( remain > 0 )) || return 124
    BENCH_WAIT_TIMEOUT_S=$remain bench_hugsim cinque exam cinque-fixed-sel3 "$1" "${WORKERS:-2}" fixed "$OPTS"
}
for stage in spin10 rest54; do
    echo "$(date +%T) sel3 $stage" > "$OUT/SEL_STATUS"
    run "$S/derot_${stage}.txt"; rc=$?
    (( rc == 124 )) && break
    (( rc == 0 )) || { echo "bench failed ($rc)" > "$OUT/SEL_ERROR"; exit "$rc"; }
done
echo "$(date +%T) sel3 stopped (deadline or all requested scenes finished)" > "$OUT/SEL_STATUS"
touch "$OUT/SEL_DONE"
