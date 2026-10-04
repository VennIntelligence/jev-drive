#!/usr/bin/env bash
# openpilot's lateral control path in HUGSIM closed loop (plans/2026-10-05-op-control-stack-prereg.md), Cinque, tree `opctrl`
# (fixed + patches/hugsim/optional/op-ctrl.patch, lib/op_ctrl.py). One resident Cinque server on one leased card; stages resumable:
#   1 smoke (SMOKE list, tag cinque-opctrl-smoke)    2 all 64 (tag cinque-opctrl)    3 same-day base rerun on 64 (tree fixed, tag cinque-fixed-base3)
# Usage (box, tmux):  GPU=<card> [WORKERS=5] [STAGES="1 2 3"] experiments/hugsim/scripts/opctrl_chain.sh [out_dir]
# Files: <out>/STATUS, DONE or ERROR; results in <out>/results.csv.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/opctrl/closed}
HPY=$DATA_DIR/envs/hugsim/bin/python
W=${WORKERS:-5}
RULE=${RULE:-'{}'}
SMOKE=${SMOKE:-experiments/hugsim/scripts/lowsel_smoke.txt}
L=${SCEN:-experiments/hugsim/scripts/derot_all64.txt}
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
$HPY experiments/hugsim/archive/zs_run.py setup-trees official fixed opctrl || fail "setup-trees"
run() {  # tag controller list workers opts
    $HPY experiments/hugsim/archive/zs_run.py run --out "$OUT" --agent cinque --controller "$2" --gpu "$GPU" --workers "$4" \
        --scenarios "$3" --socket "$OUT/servers/cinque.sock" --opts "$5" --tag "$1"
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
