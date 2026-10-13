#!/usr/bin/env bash
# Turning school P3, branch ceiling from the policy's own plan (prereg experiments/route_follow/plans/2026-10-14-p3-branch-prereg.md), Tokyo,
# one card. P2's plan dir (same 12 scenes, same k); per arm one rf_branch.sh run of 12 scenes, arms `<handover>@<transform>`.
#   rf_p3.sh <card> <out root> <P2 plan dir> <P2 pilot root>          e.g. rf_p3.sh 1 /data/runs/route_night/p3 /data/runs/route_night/p2/plan /data/runs/route_night/p2/pilot
# Stages: S1 identity arms d0@g1s1, d10@g1s1, compared with P2's stored d0 / d10 (rf_branch.py cmp; gate: every plan bit-identical);
# S2 one scene (f9e361aa) at d0@g1.5s1 vs the identity run (rf_branch.py cmp --branch; gate: 30 decisions transformed, raw plan after);
# then the 14 transform arms in the prereg's order (the drop-first ones last), then rf_branch.py read + p3.
# State: <out>/{log.txt, s1.txt, s2.txt, GATE_FAIL, read.json, table.md, DONE}.
# Env: TAG, CODE, CONC as rf_branch.sh; RUNPFX p3-; ARMS (override the main arm list).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
g=${1:?card} O=$(mkdir -p "$2" && cd "$2" && pwd) PLAN=$(cd "$3" && pwd) P2=$(cd "$4" && pwd)
export CARDS=$g RUNPFX=${RUNPFX-p3-} CODE=${CODE:-$(cd "$here/../../.." && pwd)}
py() { python3 "$here/rf_branch.py" "$@"; }             # cmp / p3: numpy only
rd() {  # rf_branch.py read inside the AlpaSim image (scene loader, route generator), as rf_pai_follow.sh
  docker run --rm --entrypoint bash -v /data/datasets/nurec:/mnt/nre-data:ro -v /data/third_party/alpasim/src:/repo/src \
    -v /data/third_party/alpasim/plugins:/repo/plugins -v /data/runs:/data/runs -v "$here":/s:ro alpasim-base:0.89.0 \
    -c "umask 0000 && cd /repo && uv run --no-sync python /s/rf_branch.py read $*"; }
log() { echo "$(date '+%F %T') $*" | tee -a "$O/log.txt"; }
rm -f "$O"/{DONE,GATE_FAIL}
log "P3 on card $g, code $(git -C "$CODE" rev-parse --short HEAD), plan $PLAN"
# S1
bash "$here/rf_branch.sh" "$O/arms" "$PLAN" d0@g1s1 d10@g1s1 >> "$O/log.txt" 2>&1 || { log "S1 run failed"; echo S1 run > "$O/GATE_FAIL"; exit 1; }
{ py cmp "$O/arms/d0@g1s1" "$P2/d0"; py cmp "$O/arms/d10@g1s1" "$P2/d10"; } > "$O/s1.txt" 2>&1
log "S1: $(grep -c 'plan max|diff| 0.000000' "$O/s1.txt") / 24 scenes bit-identical plans"
(( $(grep -c 'plan max|diff| 0.000000' "$O/s1.txt") == 24 )) || { echo S1 > "$O/GATE_FAIL"; log "S1 gate failed: see s1.txt"; exit 1; }
# S2
mkdir -p "$O/s2plan"; grep f9e361aa "$PLAN/scenes.txt" > "$O/s2plan/scenes.txt"; grep f9e361aa "$PLAN/sw_d0.txt" > "$O/s2plan/sw_d0.txt"
bash "$here/rf_branch.sh" "$O/s2" "$O/s2plan" d0@g1.5s1 >> "$O/log.txt" 2>&1 || { log "S2 run failed"; echo S2 run > "$O/GATE_FAIL"; exit 1; }
py cmp "$O/s2/d0@g1.5s1" "$O/arms/d0@g1s1" --branch > "$O/s2.txt" 2>&1
grep -q '(30 decisions), poses_branch on all of them: True, off records with poses_branch: 0' "$O/s2.txt" && grep -q 'raw plan == served plan: True' "$O/s2.txt" \
  && grep -q "first differing k $(awk '{print ($2 > 17 ? $2 : 17)}' "$O/s2plan/sw_d0.txt");" "$O/s2.txt" \
  || { echo S2 > "$O/GATE_FAIL"; log "S2 gate failed: see s2.txt"; exit 1; }
log "S2 passed"
# main arms
arms=${ARMS:-d0@g1.5s1 d10@g1.5s1 d0@g1.25s1 d10@g1.25s1 d0@g1s0.8 d10@g1s0.8 d0@g1s0.6 d10@g1s0.6 d0@g1.5s0.8 d10@g1.5s0.8 d0@g2s1 d10@g2s1 d0@g1.5s0.6 d10@g1.5s0.6}
bash "$here/rf_branch.sh" "$O/arms" "$PLAN" $arms >> "$O/log.txt" 2>&1; rc=$?
log "arms rc $rc"
spec=$(for d in "$O"/arms/*/; do [[ -f $d/DONE ]] && echo "$(basename "$d")=$d"; done)
rd --plan "$PLAN/plan.json" --win 30 --out "$O/read.json" --arm $spec > "$O/read.log" 2>&1; log "read rc $?"
py p3 --read "$O/read.json" --md "$O/table.md" > /dev/null; date > "$O/DONE"
