#!/usr/bin/env bash
# openpilot closed-loop integration study: base route follower + openpilot modifier (lib/op_arb_agent.py) on
# Bench2Drive 0.0.4 val routes (not the 220 exam routes). Plan: fc65452:todos/2026-09-28-op-closedloop.md.
#
#   experiments/op_closed_loop/archive/op_arb.sh routes                         print the registered route lists (phase 1 diagnosis, phase 2 eval)
#   experiments/op_closed_loop/archive/op_arb.sh phase <1|2>                    every arm of the phase, one after another, on the test card
#   experiments/op_closed_loop/archive/op_arb.sh arm <arm> <ids> <out>          one arm over a comma list of route ids
#   experiments/op_closed_loop/archive/op_arb.sh set <1|2|h> <tag>              op-drive (fc65452:todos/2026-09-29-op-drive.md): every arm in $ARMS x seed in
#                                                    $SEEDS over route set 1 (tuning), 2 (dev) or h (held-out), out
#                                                    arms/<tag>-<arm>-s<seed>; arm "dbaseslow" is speed-matched to the
#                                                    same tag's "drive" (or $MATCH_ARM) run of the same seed
#
# Resources (SCH row op-arb): GPU $GPU, $WORKERS CARLA servers at indices $IDX0.., cores $CPUS. One openpilot server
# (Cinque, 2 sessions per worker: the route-desire session and its desire-free twin) serves every arm. Only PIDs this
# script recorded are ever stopped. Hand-offs: $O/{log.txt, events.jsonl, STATUS, DONE-phase<k>, ERROR}.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/../../.."
REPO=$(pwd)
O=${OP_ARB_DIR:-$DATA_DIR/runs/op_arb}
GPU=${GPU:-6} WORKERS=${WORKERS:-2} IDX0=${IDX0:-160} CPUS=${CPUS:-144-167} SEED=${SEED:-0}
AD=${OP_ARB_ARMS:-$O/arms}      # arm dirs; two cards can run with their own $O (server, config) and one shared $AD
mkdir -p "$O/srv" "$O/cfg" "$AD"
export B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000} B2D_SENSOR_TICK=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
XML=${B2D_XML:-$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml}
P7=$REPO/experiments/b2d_tfv6/results/tfv6-controller/controller-eval/P7.json
PY_CARLA=$DATA_DIR/envs/carla/bin/python PY_OP=$DATA_DIR/envs/openpilot/bin/python
SOCK=$O/srv/op.sock

ev() { printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N | cut -c1-14)" "$1" "${2:+, $2}" >> "$O/events.jsonl"; }
log() { echo "$(date '+%F %T') $*" | tee -a "$O/log.txt" >&2; }
error() { { echo "# op_arb ERROR $(date '+%F %T %Z')"; echo; echo "$1"; } > "$O/ERROR"; ev error "\"reason\": \"$1\""; log "ERROR: $1"; exit 1; }

routes() {  # the registered selection: per scenario type, the first route in XML order outside Town12/13 if any, else the first
    python3 - "$XML" "$1" <<'EOF'
import sys, xml.etree.ElementTree as ET
P1 = ["SignalizedJunctionLeftTurn", "VanillaSignalizedTurnEncounterRedLight", "HardBreakRoute",
      "NonSignalizedJunctionLeftTurn", "ParkingExit", "DynamicObjectCrossing"]
P2 = ["SignalizedJunctionRightTurn", "VanillaSignalizedTurnEncounterGreenLight", "T_Junction", "VanillaNonSignalizedTurn",
      "StaticCutIn", "MergerIntoSlowTrafficV2", "ConstructionObstacle", "VehicleTurningRoutePedestrian",
      "OppositeVehicleTakingPriority", "SignalizedJunctionLeftTurnEnterFlow"]
rs = [(r.get("id"), r.get("town"), [s.get("type") for s in r.iter("scenario")][0]) for r in ET.parse(sys.argv[1]).getroot().iter("route")]
def pick(t):
    c = [r for r in rs if r[2] == t]
    small = [r for r in c if r[1] not in ("Town12", "Town13")]
    return (small or c)[0][0]
if sys.argv[2] == "h":   # held-out (op-drive): per scenario type, the first route outside Town12/13 not in P1 / P2
    used = {pick(t) for t in P1 + P2}
    seen, out = set(), []
    for rid, town, t in rs:
        if t not in seen and rid not in used and town not in ("Town12", "Town13"):
            seen.add(t)
            out.append(rid)
    print(",".join(out))
else:
    print(",".join(pick(t) for t in (P1 if sys.argv[2] == "1" else P2)))
EOF
}

srv_alive() { local p; p=$(cat "$O/srv/op.pid" 2>/dev/null) && [[ -n $p ]] && kill -0 "$p" 2>/dev/null; }
srv_start() {
    local mid="${SRV_ONNX:-base}:$WORKERS:${SRV_NO_TWIN:-0}:${SRV_PY:-}"      # KEEP_SRV: a live server with the same model and pool is reused across set calls
    if srv_alive; then
        [[ $(cat "$O/srv/model_id" 2>/dev/null) == "$mid" ]] && return 0
        log "openpilot server model / pool changed ($(cat "$O/srv/model_id" 2>/dev/null) -> $mid): restarting"
        srv_stop; sleep 8
    fi
    echo "$mid" > "$O/srv/model_id"
    rm -f "$O/srv/op.ready" "$SOCK"
    (
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 setsid taskset -c "$CPUS" "$PY_OP" "${SRV_PY:-experiments/op_closed_loop/archive/op_arb_server.py}" cinque \
            --pool "$WORKERS" --backend cuda-iob --socket "$SOCK" --ready-file "$O/srv/op.ready" \
            ${SRV_ONNX:+--onnx "$SRV_ONNX"} ${SRV_NO_TWIN:+--no-twin} >> "$O/srv/op.log" 2>&1 &
        echo $! > "$O/srv/op.pid"
        wait $!
        echo "$(date '+%F %T') server exited rc=$?" >> "$O/srv/op.log"
    ) &
    until [[ -s $O/srv/op.pid ]]; do sleep 0.2; done
    local t0=$SECONDS
    until [[ -e $O/srv/op.ready ]]; do
        srv_alive || error "openpilot server died at start-up (see $O/srv/op.log)"
        (( SECONDS - t0 > 900 )) && error "openpilot server not ready after 15 min"
        sleep 5
    done
    ev server_ready "\"pid\": $(cat "$O/srv/op.pid")"
}
srv_stop() { local p; p=$(cat "$O/srv/op.pid" 2>/dev/null) && [[ -n $p ]] && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; }; rm -f "$O/srv/op.pid"; }

arm_cfg() {  # arm_cfg <arm>: the agent config (every arm: CL2's openpilot path and P7; only "arb" differs)
    local arm=$1 arb
    case $arm in
        native)  arb='{"mode": "native", "twin": true}' ;;
        oshadow) arb='{"mode": "oshadow", "twin": true}' ;;
        base)    arb='{"mode": "base"}' ;;
        acc)     arb='{"mode": "acc"}' ;;
        e2e)     arb="{\"mode\": \"e2e\"${E2E_ARGS:+, $E2E_ARGS}}" ;;
        switch)  arb="{\"mode\": \"switch\"${E2E_ARGS:+, $E2E_ARGS}}" ;;
        baseslow) arb="{\"mode\": \"base\", \"cruise_by_route\": ${CRUISE_BY_ROUTE:?}}" ;;
        e2enofb) arb="{\"mode\": \"e2e\"${E2E_ARGS:+, $E2E_ARGS}, \"latch_max_s\": 1e9}" ;;
        oplat)   arb="{\"mode\": \"switch\", \"zones\": false${E2E_ARGS:+, $E2E_ARGS}}" ;;
        # op-drive arms (fc65452:todos/2026-09-29-op-drive.md); every one coasts instead of light braking below 2.5 m/s
        dbase)   arb='{"mode": "base", "coast_v": 2.5}' ;;
        dbaseslow|dbaseslow[0-9]|dslow) arb="{\"mode\": \"drive\", \"lat\": \"op\", \"lat_exec\": \"${LAT_EXEC:-curv}\", \"lon\": \"op\", \"hold\": \"intent\", \"release\": \"planx\", \"release_th\": 2.0, \"release_s\": 1.0, \"latch_max_s\": ${RESUME_S:-5}, \"coast_v\": 2.5, \"cruise_by_route\": ${CRUISE_BY_ROUTE:-{}}${DRIVE_ARGS:+, $DRIVE_ARGS}}" ;;
        latp7|latk) arb="{\"mode\": \"drive\", \"lat\": \"op\", \"lat_exec\": \"$([[ $arm == latk ]] && echo curv || echo p7)\", \"lon\": \"base\", \"coast_v\": 2.5}" ;;
        # op-adapt L and vlm_arb arms: drive, jslow, vred, vbyp, vall
        # the `drive` arbitration is the named preset "drive" of lib/op_arb_agent.py PRESETS (moved there verbatim 2026-10-05)
        drive|pjunc|pbyp|pbypgap|pbyp2|pbyp2ng|pred|pall|jslow|vred|vred3|vmerge|vbyp|vall|dlon|dnod|dtz|lmain*|lnoint*|ldw10*|lkd*|ltz*) arb="{\"preset\": \"drive\"$([[ $arm == dlon ]] && echo ', "lat": "route"' || true)${LAT_EXEC:+, \"lat_exec\": \"$LAT_EXEC\"}${RESUME_S:+, \"latch_max_s\": $RESUME_S}${DRIVE_ARGS:+, $DRIVE_ARGS}}" ;;
        # openpilot as on the car (jevdrive/openpilot/interface.py, docs/openpilot-interface.md): preset "spec" (camera: open-loop-aligned) or spec_bumper122 / spec_windshield143 (earlier rigs); zones off: DRIVE_ARGS='"zones": false, "div_m": 1e9'
        spec|spec_*) arb="{\"preset\": \"$arm\"${LAT_EXEC:+, \"lat_exec\": \"$LAT_EXEC\"}${RESUME_S:+, \"latch_max_s\": $RESUME_S}${DRIVE_ARGS:+, $DRIVE_ARGS}}" ;;
        # R3a (fc65452:todos/2026-09-29-op-drive.md): drive + privileged traffic-light stop; a stop latch is released only by plan / lead away from a red light, or at green
        dtl) arb="{\"mode\": \"drive\", \"lat\": \"op\", \"lat_exec\": \"${LAT_EXEC:-p7}\", \"lon\": \"op\", \"hold\": \"intent\", \"release\": \"planx\",
 \"release_th\": 2.0, \"release_s\": 1.0, \"latch_max_s\": 1e9, \"coast_v\": 2.5, \"tl_stop\": true, \"tl_n\": ${TL_N:-50}}" ;;
        *) error "unknown arm $arm" ;;
    esac
    local pc=""   # TOP_ARGS: extra top-level agent keys, e.g. '"op_mount": [3.8, 0.0, 1.22]' 
    [[ ${PC_ENABLE:-0} == 1 ]] && pc=", \"pc\": {\"arm\": \"$arm\"}"
    echo "{\"model\": \"cinque\", \"socket\": \"$SOCK\", \"plan_every\": 1, \"ctl_every\": 4, \"op_camera_tick\": 0.05,
 \"plan_origin\": \"rear\", \"warmup_s\": 5.0, \"desire\": ${DESIRE:-true}, \"controller\": \"fixed\", \"controller_preset\": \"pursuit\",
 \"controller_config\": \"$P7\", \"seed\": 0, \"dump_every\": 0, \"arb\": $arb$pc${TOP_ARGS:+, $TOP_ARGS}}" > "$O/cfg/$arm.json"
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$O/cfg/$arm.json" || error "bad config for $arm"
    echo "$O/cfg/$arm.json"
}

match() {  # match <drive dir> <base dir>: {route: 8 x v_drive / v_base} clipped to [0.5, 8], mean speed over non-warm steps
    python3 - "$1" "$2" <<'EOF'
import glob, json, os, sys
def speeds(d):
    out = {}
    for f in glob.glob(os.path.join(d, "done", "*.json")):
        rid = os.path.basename(f)[:-5]
        p = os.path.join(d, "attempts", rid, str(json.load(open(f))["attempt"]), "plans.jsonl")
        v = [r["v"] for r in map(json.loads, open(p)) if not r["warm"]]
        out[rid] = sum(v) / max(len(v), 1)
    return out
a, b = speeds(sys.argv[1]), speeds(sys.argv[2])
print(json.dumps({r: round(min(max(8.0 * a[r] / max(b[r], 1e-3), 0.5), 8.0), 2) for r in a if r in b}))
EOF
}

match_iter() {  # match_iter <drive dir> <prev slow dir>: next per-route set speed = prev set speed x v_drive / v_prev_slow, clipped to [0.5, 8]
    python3 - "$1" "$2" <<'PYEOF'
import glob, json, os, sys
def speeds(d):
    out = {}
    for f in glob.glob(os.path.join(d, "done", "*.json")):
        rid = os.path.basename(f)[:-5]
        p = os.path.join(d, "attempts", rid, str(json.load(open(f))["attempt"]), "plans.jsonl")
        v = [r["v"] for r in map(json.loads, open(p)) if not r["warm"]]
        out[rid] = sum(v) / max(len(v), 1)
    return out
a, b = speeds(sys.argv[1]), speeds(sys.argv[2])
c = json.load(open(os.path.join(sys.argv[2], "cruise_by_route.json")))
print(json.dumps({r: round(min(max(c[r] * a[r] / max(b[r], 0.05), 0.5), 8.0), 2) for r in a if r in b and r in c}))
PYEOF
}

run_arm() {  # run_arm <arm> <ids> <out>
    local arm=$1 ids=$2 out=$3 cfg pid t0=$SECONDS
    mkdir -p "$out"
    cfg=$(arm_cfg "$arm") || exit 1
    srv_start
    echo "$arm $out" > "$O/CURRENT"
    ev arm_start "\"arm\": \"$arm\", \"routes\": \"$ids\""
    log "start $arm on $ids (GPU $GPU x $WORKERS, idx $IDX0, cores $CPUS)"
    taskset -c "$CPUS" "$PY_CARLA" scripts/b2d_run.py --routes "$XML" --route-ids "$ids" --workers "$WORKERS" \
        --server-index "$IDX0" --index-span "$WORKERS" --gpu-rank "$GPU" --tm-seed "$SEED" --no-spectator --no-reap \
        --client-threads 8 --max-attempts 3 --stall-s 480 --route-timeout-s 2400 --out "$out" --python "$PY_CARLA" \
        --agent "${OP_ARB_AGENT:-scripts/op_arb_agent.py}" --agent-config "$cfg" --fast-copy --cache-lights >> "$out/runner.log" 2>&1 &
    pid=$!
    echo "$pid" > "$out/runner.pid"
    while kill -0 "$pid" 2>/dev/null; do
        srv_alive || { log "openpilot server died during $arm; restarting"; ev server_died; srv_start; }
        sleep 20
    done
    wait "$pid"
    local n; n=$(ls "$out"/done/*.json 2>/dev/null | wc -l)
    ev arm_end "\"arm\": \"$arm\", \"done\": $n, \"wall_min\": $(( (SECONDS - t0) / 60 ))"
    log "done $arm: $n routes finished in $(( (SECONDS - t0) / 60 )) min"
}

case ${1:-} in
    routes) echo "phase1 $(routes 1)"; echo "phase2 $(routes 2)"; echo "heldout $(routes h)" ;;
    set)
        trap '[[ -n ${KEEP_SRV:-} ]] || srv_stop' EXIT
        ids=${OPL_IDS:-$(routes "$2")}; tag=$3
        for seed in ${SEEDS:-0}; do
            for a in ${ARMS:?}; do
                d=$AD/$tag-$a-s$seed
                [[ -e $d/DONE ]] && continue
                if [[ $a == dbaseslow* ]]; then       # dbaseslow: linear from dbase; dbaseslowN (N >= 2): iterate from the previous one
                    n=${a#dbaseslow}; prev=dbaseslow$([[ $n == 2 ]] && echo "" || echo $((n - 1)))
                    if [[ -z $n ]]; then CRUISE_BY_ROUTE=$(match "$AD/$tag-${MATCH_ARM:-drive}-s$seed" "$AD/$tag-dbase-s$seed")
                    else CRUISE_BY_ROUTE=$(match_iter "$AD/$tag-${MATCH_ARM:-drive}-s$seed" "$AD/$tag-$prev-s$seed"); fi || error "speed match failed"
                    [[ -n $CRUISE_BY_ROUTE ]] || error "speed match empty"
                    export CRUISE_BY_ROUTE; log "$a s$seed set speeds $CRUISE_BY_ROUTE"
                    mkdir -p "$d"; echo "$CRUISE_BY_ROUTE" > "$d/cruise_by_route.json"
                fi
                echo "set $2 $tag arm $a seed $seed $(date '+%F %T')" > "$O/STATUS"
                SEED=$seed run_arm "$a" "$ids" "$d"
                date > "$d/DONE"
            done
        done
        date > "$O/DONE-$tag"; echo "set $tag done $(date '+%F %T')" > "$O/STATUS" ;;
    arm) run_arm "$2" "$3" "$4" ;;
    phase)
        trap 'srv_stop' EXIT
        k=$2; ids=$(routes "$k")
        arms=${ARMS:-$([[ $k == 1 ]] && echo "native oshadow" || echo "base acc e2e switch oplat")}
        for a in $arms; do
            [[ -e $AD/p$k-$a/DONE ]] && continue
            echo "phase $k arm $a $(date '+%F %T')" > "$O/STATUS"
            run_arm "$a" "$ids" "$AD/p$k-$a"
            date > "$AD/p$k-$a/DONE"
        done
        date > "$O/DONE-phase$k"; echo "phase $k done $(date '+%F %T')" > "$O/STATUS" ;;
    *) sed -n 2,12p "$0"; exit 2 ;;
esac
