#!/usr/bin/env bash
# HUGSIM re-baseline of shipped Cinque under the unified openpilot interface (docs/openpilot-interface.md,
# results/unified_interface.md). One resident Cinque server on one leased card; stages resumable (zs_run skips finished jobs):
#   smoke  preset spec_cold on 2 scenes (tag uni-spec-smoke)
#   small  11 representative scenes (unified_hugsim_small.txt: 6 base spinners incl. one per dataset, d118's new-stuck
#          3000_3200-medium / 0418-hard, completers 0930-hard / 0051-easy, crash 034-hard-01), four arms:
#            uni-exam   preset exam, tree fixed, opts {}           (every HUGSIM number before 2026-10-05: iLQR, 5 s static warm-up)
#            uni-d118   preset opctrl_d118                         (decision 118: lateral path, 5 s static warm-up)
#            uni-spec   preset spec_cold (was "spec" when run)  (lateral path, no static warm-up, dilate clock)
#            uni-hold   preset spec_hold                           (lateral path, no static warm-up, hold clock, delay 0.2 s)
#   full   preset spec (+ spec_hold if CLOCK=both) on all 64 scenes, REPEATS runs each (tags uni-spec-r<k>)
# Usage (box, tmux):  GPU=<card> [WORKERS=5] [STAGES="smoke small"] experiments/leaderboard_audit/scripts/unified_hugsim_chain.sh [out]
# Files: <out>/STATUS, DONE or ERROR; rows in <out>/results.csv; interface.json in every run dir.
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/unified/hugsim}
HPY=$DATA_DIR/envs/hugsim/bin/python
W=${WORKERS:-5}
SMALL=experiments/leaderboard_audit/scripts/unified_hugsim_small.txt
ALL=experiments/hugsim/scripts/derot_all64.txt
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/ERROR"
source scripts/bench_lane.sh
fail() { echo "$(date +%T) $*" | tee "$OUT/ERROR"; exit 1; }
st() { echo "$(date +%T) $*" | tee "$OUT/STATUS"; }
run() {  # tag preset list workers [controller]; each repeat tag has an independent bench identity
    bench_hugsim cinque "$2" "$1" "$3" "$4" "${5:-}"
}
for s in ${STAGES:-smoke small}; do
    case $s in
    smoke) st "smoke: spec on 2 scenes"
       head -2 $SMALL > "$OUT/smoke.txt"
       run uni-spec-smoke spec_cold "$OUT/smoke.txt" 2 || fail "smoke";;
    small) st "small: 11 scenes x exam / d118 / spec / hold"
       run uni-exam exam $SMALL "$W" fixed || fail "small exam"
       run uni-d118 opctrl_d118 $SMALL "$W" || fail "small d118"
       run uni-spec spec_cold $SMALL "$W" || fail "small spec"
       run uni-hold spec_hold $SMALL "$W" || fail "small hold";;
    full) for k in $(seq 1 "${REPEATS:-2}"); do
           st "full: spec on 64, repeat $k"
           run uni-spec-r$k spec $ALL "$W" || fail "full spec r$k"
           [[ ${CLOCK:-spec} == both ]] && { run uni-hold-r$k spec_hold $ALL "$W" || fail "full hold r$k"; }
       done;;
    esac
done
st "done"
touch "$OUT/DONE"
