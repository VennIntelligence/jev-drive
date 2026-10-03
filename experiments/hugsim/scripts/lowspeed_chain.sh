#!/usr/bin/env bash
# Low-speed lateral transfer limit in HUGSIM closed loop (plans/2026-10-04-lowspeed-ctrl-prereg.md), Cinque, PR #57 controller + lowspeed patch.
# One resident Cinque server on one leased card; stages resumable (zs_run skips finished jobs):
#   1 lowspeed on all 64    2 base rerun on all 64 (same-day noise control; tree `fixed`, no rule)
#   3 lowsel (selective rule, plans/2026-10-04-lowspeed-ctrl-selective-prereg.md) on all 64    4 lowsel smoke (SMOKE list)    5 base rerun #2 (tag cinque-fixed-base2)
# Usage (box, tmux):  GPU=<card> [WORKERS=5] [STAGES="1 2"] experiments/hugsim/scripts/lowspeed_chain.sh [out_dir]
# Files: <out>/STATUS, DONE or ERROR; results in <out>/results.csv.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-lowspeed/closed}
HPY=$DATA_DIR/envs/hugsim/bin/python
W=${WORKERS:-5}
RULE=${RULE:-'{"jerk": 5.0, "tau0": 3.0, "v0": 2.5, "v1": 3.5}'}
SEL=${SEL:-'{"d0": 2.0, "d1": 4.0, "v0": 2.5, "v1": 3.5, "tau0": 0, "jerk": 0}'}
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
$HPY experiments/hugsim/archive/zs_run.py setup-trees official fixed lowspeed lowsel || fail "setup-trees"
run() {  # tag controller list workers
    $HPY experiments/hugsim/archive/zs_run.py run --out "$OUT" --agent cinque --controller "$2" --gpu "$GPU" --workers "$4" \
        --scenarios "$3" --socket "$OUT/servers/cinque.sock" --opts '{}' --tag "$1"
}
for s in ${STAGES:-1 2}; do
    case $s in
    1) st "stage 1: lowspeed on 64 ($RULE)"
       LOWSPEED_CTRL="$RULE" LOWSPEED_LIB="$PWD/lib" run cinque-lowspeed lowspeed $L "$W" || fail "stage 1";;
    2) st "stage 2: base rerun on 64"
       run cinque-fixed-base fixed $L "$W" || fail "stage 2";;
    3) st "stage 3: lowsel on 64 ($SEL)"
       LOWSPEED_SEL="$SEL" LOWSPEED_LIB="$PWD/lib" run cinque-lowsel lowsel $L "$W" || fail "stage 3";;
    4) st "stage 4: lowsel smoke"
       LOWSPEED_SEL="$SEL" LOWSPEED_LIB="$PWD/lib" run cinque-lowsel-smoke lowsel $SMOKE 2 || fail "stage 4";;
    5) st "stage 5: base rerun 2 on 64"
       run cinque-fixed-base2 fixed $L "$W" || fail "stage 5";;
    esac
done
st "done"
touch "$OUT/DONE"
