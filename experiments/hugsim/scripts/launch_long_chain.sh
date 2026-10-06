#!/usr/bin/env bash
# Non-privileged longitudinal launch assist in HUGSIM closed loop (plans/2026-10-04-launch-long-prereg.md), Cinque, PR #57 controller (tree `fixed`).
# Shared bench workers on the GPU pool; legacy tags/outputs retained. Previously one resident server on one leased card; stages resumable (zs_run skips finished jobs):
#   1 smoke (SMOKE list, tag cinque-launchlong-smoke)    2 all 64 (tag cinque-launchlong)
# Usage (box, tmux):  GPU=<card> [WORKERS=5] [STAGES="1 2"] experiments/hugsim/scripts/launch_long_chain.sh [out_dir]
# Files: <out>/STATUS, DONE or ERROR; results in <out>/results.csv.
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-launchlong/closed}
HPY=$DATA_DIR/envs/hugsim/bin/python
W=${WORKERS:-5}
OPTS=${OPTS:-'{"launch_long": {}}'}
SMOKE=${SMOKE:-experiments/hugsim/scripts/launch_long_smoke.txt}
L=${SCEN:-experiments/hugsim/scripts/derot_all64.txt}
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/ERROR"
source scripts/bench_lane.sh
fail() { echo "$(date +%T) $*" | tee "$OUT/ERROR"; exit 1; }
st() { echo "$(date +%T) $*" | tee "$OUT/STATUS"; }
run() {
    bench_hugsim cinque exam "$1" "$2" "$3" fixed "$OPTS"
}
for s in ${STAGES:-1 2}; do
    case $s in
    1) st "stage 1: smoke ($OPTS)"; run cinque-launchlong-smoke "$SMOKE" 2 || fail "stage 1";;
    2) st "stage 2: launch_long on 64 ($OPTS)"; run cinque-launchlong "$L" "$W" || fail "stage 2";;
    esac
done
st "done"
touch "$OUT/DONE"
