#!/usr/bin/env bash
# SCH: ~10-route pilots of lane B's queued CL arms on the debug card, ahead of the chain (user rule 2026-09-26: 1 unit ->
# ~10 units + checklist -> full batch). Each pilot is lane B's own step, `scripts/nq3_b.sh arm`, run unchanged in its own
# lane directory (B_DIR = runs/sched/pilot/b, B_REPORT=0), so its routes never enter lane B's arms or tables.
# Then scripts/sch_cl_checklist.py writes <pilot>/verdict.json:
#   PASS -> nothing to do;  FAIL -> "<arm> <seed>" for seeds 0-2 appended to runs/nq3/b/SKIP (the chain skips them)
#   and a line in runs/sched/ESCALATE.md;  FLAG -> a line in ESCALATE.md only (main decides).
# An arm the chain has already started (runs/nq3/b/arms/<arm>/s0/requested.json) is checked in place, read-only, once
# its first 10 routes are settled (no SKIP: the chain is inside it; FAIL / FLAG are escalated).
#
#   scripts/tmux_run.sh sch-pilots scripts/sch_cl_pilots.sh [arm ...]      (default: the chain's order)
# Resources (runs/sched/table.tsv, row sch-pilot): GPU $PILOT_GPU (1, the debug card), $PILOT_WORKERS (3) CARLA servers,
# index block 390-419 (i +- 120 clear of every other block), cores $PILOT_CPUS (110-117).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
MAIN=$DATA_DIR/runs/nq3/b
P=$DATA_DIR/runs/sched/pilot/b
ESC=$DATA_DIR/runs/sched/ESCALATE.md
mkdir -p "$P"
echo $$ > "$P/pilots.pid"
ARMS=(${@:-cl3 cl4 cl2 cl7 cl6 cl8 tfv6 bridgedrive blue simlingo})
log() { echo "$(date '+%F %T') $*" | tee -a "$P/pilots.log" >&2; }
routes_for() {  # 10 routes spread over the 220 (every 22nd in file order); obstacle arms: the first 10 obstacle routes
    python3 - "$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml" "$1" <<'PY'
import sys, xml.etree.ElementTree as ET
keep = {"Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays", "ParkedObstacle",
        "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays"}
rs = list(ET.parse(sys.argv[1]).getroot().iter("route"))
if sys.argv[2] in ("cl5", "cl5d"):
    ids = [r.get("id") for r in rs if any(s.get("type") in keep for s in r.iter("scenario"))][:10]
else:
    ids = [r.get("id") for r in rs][::22][:10]
print(",".join(ids))
PY
}
escalate() { echo "- $(date '+%F %T') [SCH pilot] $*" >> "$ESC"; log "ESCALATED: $*"; }
verdict_of() { python3 -c "import json;v=json.load(open('$1'));print(v['verdict'], '; '.join(v.get('fail', []) + v.get('flag', [])))"; }

for arm in "${ARMS[@]}"; do
    d=$P/arms/$arm/s0
    [[ -f $d/verdict.json ]] && { log "$arm: verdict exists ($(verdict_of "$d/verdict.json"))"; continue; }
    if [[ -f $MAIN/arms/$arm/s0/requested.json ]]; then          # the chain got there first: check it in place
        log "$arm: the chain has started it; checking its first routes in place"
        mkdir -p "$d"
        until .venv/bin/python scripts/sch_cl_checklist.py "$MAIN/arms/$arm/s0" --out "$d/verdict.json" > /dev/null; (( $? != 4 )); do sleep 120; done
        read -r v why <<< "$(verdict_of "$d/verdict.json")"
        log "$arm (in place): $v $why"
        [[ $v == PASS ]] || escalate "$arm s0 in place (chain running it): $v - $why ($d/verdict.json)"
        continue
    fi
    [[ $arm == cl5 || $arm == cl5d ]] && [[ ! -e $DATA_DIR/runs/nq3/q2/closed_loop_head/READY ]] && { log "$arm: head not READY, skipped"; continue; }
    ids=$(routes_for "$arm")
    log "$arm: pilot on GPU ${PILOT_GPU:-1}, ${PILOT_WORKERS:-3} workers, routes $ids"
    B_DIR=$P B_REPORT=0 GPUS=${PILOT_GPU:-1} WORKERS=${PILOT_WORKERS:-3} B_CPUS=${PILOT_CPUS:-110-117} \
        B_IDX="${PILOT_GPU:-1}:${PILOT_IDX:-390}" scripts/nq3_b.sh arm "$arm" 0 "$ids" "${PILOT_EST_H:-3.0}" >> "$P/pilot-$arm.log" 2>&1
    log "$arm: pilot step exited rc=$?"
    .venv/bin/python scripts/sch_cl_checklist.py "$d" --out "$d/verdict.json" > /dev/null
    read -r v why <<< "$(verdict_of "$d/verdict.json")"
    log "$arm: $v $why"
    case $v in
        PASS) ;;
        FAIL) if [[ -f $MAIN/arms/$arm/s0/requested.json ]]; then
                  escalate "$arm: pilot FAIL ($why) but the chain already started it; not skipped ($d/verdict.json)"
              else
                  printf '%s 0\n%s 1\n%s 2\n' "$arm" "$arm" "$arm" >> "$MAIN/SKIP"
                  escalate "$arm: pilot FAIL ($why); seeds 0-2 appended to runs/nq3/b/SKIP ($d/verdict.json)"
              fi ;;
        *) escalate "$arm: pilot $v ($why); not skipped, main decides ($d/verdict.json)" ;;
    esac
done
log "pilots done"
