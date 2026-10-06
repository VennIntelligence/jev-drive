#!/usr/bin/env bash
# openpilot's lateral control path in HUGSIM closed loop (plans/2026-10-05-op-control-stack-prereg.md), Cinque, tree `opctrl`
# (fixed + patches/hugsim/optional/op-ctrl.patch, lib/op_ctrl.py). One resident Cinque server on one leased card; stages resumable:
#   1 smoke (SMOKE list, tag cinque-opctrl-smoke)    2 all 64 (tag cinque-opctrl)    3 same-day base rerun on 64 (tree fixed, tag cinque-fixed-base3)
# Usage (box, tmux):  GPU=<card> [WORKERS=5] [STAGES="1 2 3"] experiments/hugsim/scripts/opctrl_chain.sh [out_dir]
# Files: <out>/STATUS, DONE or ERROR; results in <out>/results.csv.
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/opctrl/closed}
HPY=$DATA_DIR/envs/hugsim/bin/python
W=${WORKERS:-5}
RULE=${RULE:-'{}'}
SMOKE=${SMOKE:-experiments/hugsim/scripts/lowsel_smoke.txt}
L=${SCEN:-experiments/hugsim/scripts/derot_all64.txt}
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/ERROR"
source scripts/bench_lane.sh
fail() { echo "$(date +%T) $*" | tee "$OUT/ERROR"; exit 1; }
st() { echo "$(date +%T) $*" | tee "$OUT/STATUS"; }
run() {
    bench_hugsim cinque exam "$1" "$3" "$4" "$2" "$5"
}
for s in ${STAGES:-1 2 3}; do
    case $s in
    1) st "stage 1: smoke ($RULE)"
       OP_CTRL="$RULE" OP_CTRL_LIB="$PWD/lib" run cinque-opctrl-smoke opctrl "$SMOKE" 2 '{"op_ctrl": true}' || fail "stage 1";;
    2) st "stage 2: opctrl on 64 ($RULE)"
       OP_CTRL="$RULE" OP_CTRL_LIB="$PWD/lib" run cinque-opctrl opctrl "$L" "$W" '{"op_ctrl": true}' || fail "stage 2";;
    3) st "stage 3: base rerun 3 on 64"
       run cinque-fixed-base3 fixed "$L" "$W" '{}' || fail "stage 3";;
    esac
done
st "done"
touch "$OUT/DONE"
