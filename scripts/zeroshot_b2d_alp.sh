#!/usr/bin/env bash
# Alpamayo 1.5 zero-shot Bench2Drive exam with the Bench2DriveZoo PID (todos/2026-09-24-zeroshot-exam/bench2drive.md).
# One job = one resident Alpamayo policy server + b2d_run.py workers, then the server is stopped.
# Meant to run under scripts/slot_run.sh (which sets CUDA_VISIBLE_DEVICES from --gpu); CARLA takes the same card via
# --gpu-rank (Vulkan index == CUDA index on this box, checked by UUID 2026-09-25).
#
#   scripts/zeroshot_b2d_alp.sh smoke <gpu>   5 pre-registered routes, 1 worker. Exits non-zero on INFRASTRUCTURE
#                                             failure only: runner error, server crash/restart, route never finished,
#                                             harness error in a route, policy server gone. Driving outcome never.
#   scripts/zeroshot_b2d_alp.sh smoke2 <gpu>  re-smoke after the stall diagnosis (todos/2026-09-24-zeroshot-exam/
#                                             alpamayo-closed-loop-diagnosis.md, section 6): 11 routes x 2 arms in parallel
#                                             on one server, f1 (forward-only plan, AD-MLP hold) and f1f2b (+ per-tick
#                                             control_pid); then scripts/zeroshot_b2d_alp_smoke2_check.py writes
#                                             smoke2-alpamayo-choice.json. Non-zero on infrastructure failure or no arm passing.
#   scripts/zeroshot_b2d_alp.sh lat <gpu>     lateral fix (diagnosis doc, section 8): pilot P1 (fixed-controller steer) and
#                                             P2 (Zoo turn PID on the 1.5 s plan point) on routes 1833, 1852 in parallel,
#                                             scripts/zeroshot_b2d_alp_lat_check.py picks one (lat-pilot-choice.json), then
#                                             the 16-route re-smoke with it and its acceptance (lat-resmoke-check.json).
#                                             Non-zero on infrastructure failure or when no pilot variant is eligible;
#                                             a failed re-smoke acceptance is reported in the json, not as a job failure.
#   scripts/zeroshot_b2d_alp.sh full <gpu>    220 routes, 4 workers, resumable (rerun skips done/<id>.json), with the arm
#                                             chosen by smoke2. Non-zero if the policy server dies, there is no summary, or
#                                             more than 15 routes never finished (11 is the known CARLA-crash baseline).
#                                             The first full run (no fixes, full220-alpamayo-zoopid/) is discarded.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
mode=$1 gpu=$2
D=$DATA_DIR/runs/zeroshot-exam/b2d
choice=$D/smoke2-alpamayo-choice.json
fixes=""                                      # extra agent-config keys
case $mode in
    smoke) out=$D/smoke-alpamayo-zoopid; workers=1; sidx=400; routes=(--route-ids 2390,24211,1711,2373,3564) ;;
    smoke2) rm -f "$choice"; routes=(--route-ids 2390,24211,1711,2373,3564,1833,1852,1956,2668,4183,11381) ;;
    lat) routes=() ;;
    full)
        arm=$(python3 -c "import json; print(json.load(open('$choice'))['choice'] or '')" 2>/dev/null)
        case $arm in
            f1) fixes=', "plan_forward_only": true, "zoo_cadence": "plan"' ;;
            f1f2b) fixes=', "plan_forward_only": true, "zoo_cadence": "tick"' ;;
            *) echo "no smoke2 choice in $choice; refusing to start the full run" >&2; exit 2 ;;
        esac
        out=$D/full220-alpamayo-zoopid-$arm; workers=4; sidx=440; routes=(--towns all) ;;
    *) echo "mode must be smoke, smoke2, lat or full" >&2; exit 2 ;;
esac
sock=$D/alpamayo-$mode-zoopid.sock ready=$D/alpamayo-$mode-zoopid.ready
rm -f "$sock" "$ready"
agent_cfg() {   # agent_cfg <file> <extra keys>
    cat > "$1" <<EOF
{"model": "alpamayo", "socket": "$sock", "plan_every": 5, "controller": "zoo_pid", "controller_preset": "carla",
 "controller_config": "$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json",
 "seed": 0, "dump_every": $([[ $mode == smoke ]] && echo 1 || echo 0)$2}
EOF
}
cfg=$D/agent-alpamayo-$mode-zoopid.json
agent_cfg "$cfg" "$fixes"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    "$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python" scripts/zeroshot_policy_server.py alpamayo \
    --socket "$sock" --ready-file "$ready" > "$D/alpamayo-$mode-zoopid.server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null' EXIT
until [[ -e $ready ]]; do
    kill -0 $server 2>/dev/null || { echo "policy server died during start-up, see $D/alpamayo-$mode-zoopid.server.log"; exit 3; }
    sleep 5
done
echo "$(date +%T) policy server ready (pid $server) on GPU $gpu"

run_arm() {   # run_arm <name> <extra keys> <route ids> <workers> <server index> <out>
    agent_cfg "$D/agent-alpamayo-$mode-$1.json" "$2"
    "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --route-ids "$3" --workers "$4" --server-index "$5" \
        --gpu-rank "$gpu" --agent scripts/b2d_zeroshot_agent.py --agent-config "$D/agent-alpamayo-$mode-$1.json" \
        --decimate 2 --no-spectator --no-reap --max-attempts 2 --out "$6"
}
if [[ $mode == lat ]]; then
    f1=', "plan_forward_only": true, "zoo_cadence": "plan"'
    declare -A lat=([p1]="$f1"', "zoo_lateral": "fixed"' [p2]="$f1"', "zoo_lateral": "time", "zoo_aim_s": 1.5')
    rm -f "$D/lat-pilot-choice.json" "$D/lat-resmoke-check.json"
    run_arm p1 "${lat[p1]}" 1833,1852 2 480 "$D/lat-pilot-alpamayo-p1" & a=$!
    run_arm p2 "${lat[p2]}" 1833,1852 2 482 "$D/lat-pilot-alpamayo-p2" & b=$!
    wait $a; ra=$?; wait $b; rb=$?
    kill -0 $server 2>/dev/null || { echo "policy server died during the pilot"; exit 3; }
    (( ra == 0 && rb == 0 )) || { echo "pilot runner exit $ra / $rb"; exit 1; }
    "$DATA_DIR/envs/carla/bin/python" scripts/zeroshot_b2d_alp_lat_check.py pilot p1="$D/lat-pilot-alpamayo-p1" \
        p2="$D/lat-pilot-alpamayo-p2" --out "$D/lat-pilot-choice.json" || { echo "no eligible pilot variant"; exit 4; }
    arm=$(python3 -c "import json; print(json.load(open('$D/lat-pilot-choice.json'))['choice'])")
    echo "$(date +%T) pilot choice: $arm; 16-route re-smoke"
    run_arm "$arm" "${lat[$arm]}" 2390,24211,1711,2373,3564,1833,1852,1956,2668,4183,11381,1825,2084,2086,2091,2115 \
        4 490 "$D/lat-resmoke-alpamayo-$arm"; rc=$?
    kill -0 $server 2>/dev/null || { echo "policy server died during the re-smoke"; exit 3; }
    (( rc == 0 )) || { echo "re-smoke runner exit $rc"; exit 1; }
    "$DATA_DIR/envs/carla/bin/python" scripts/zeroshot_b2d_alp_lat_check.py resmoke arm="$D/lat-resmoke-alpamayo-$arm" \
        --baseline "$D/smoke2-alpamayo-f1" "$D/full220-alpamayo-zoopid-f1:1825,2084,2086,2091,2115" \
        --out "$D/lat-resmoke-check.json"
    exit 0
fi

if [[ $mode == smoke2 ]]; then
    # Two arms at once on the one server (it keeps per-connection state); each has its own --out and server indices.
    pids=()
    for arm in f1 f1f2b; do
        if [[ $arm == f1 ]]; then keys=', "plan_forward_only": true, "zoo_cadence": "plan"' idx=420
        else keys=', "plan_forward_only": true, "zoo_cadence": "tick"' idx=430; fi
        agent_cfg "$D/agent-alpamayo-smoke2-$arm.json" "$keys"
        "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py "${routes[@]}" --workers 2 --server-index $idx \
            --gpu-rank "$gpu" --agent scripts/b2d_zeroshot_agent.py --agent-config "$D/agent-alpamayo-smoke2-$arm.json" \
            --decimate 2 --no-spectator --no-reap --max-attempts 2 --out "$D/smoke2-alpamayo-$arm" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "runner $p exit $?"; done
    kill -0 $server 2>/dev/null || { echo "policy server died during the run"; exit 3; }
    "$DATA_DIR/envs/carla/bin/python" scripts/zeroshot_b2d_alp_smoke2_check.py f1="$D/smoke2-alpamayo-f1" \
        f1f2b="$D/smoke2-alpamayo-f1f2b" --out "$choice"
    exit $?
fi

"$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py "${routes[@]}" --workers $workers --server-index $sidx \
    --gpu-rank "$gpu" --agent scripts/b2d_zeroshot_agent.py \
    --agent-config "$cfg" --decimate 2 --no-spectator --no-reap --max-attempts 2 --out "$out"
rc=$?
kill -0 $server 2>/dev/null || { echo "policy server died during the run"; exit 3; }
if [[ $mode == full ]]; then
    # b2d_run exits 1 when any route never finished; the previous full round lost 11 Town12/13 routes to CARLA
    # segfaults (docs/carla.md), which is reported with the score, not an infrastructure failure of this job.
    python3 -c "import json,sys; s=json.load(open('$out/summary.json')); n=len(s['routes_never_finished']); \
print('never finished:', s['routes_never_finished']); sys.exit(0 if n <= 15 else 1)" || { echo "runner exit $rc"; exit 1; }
    exit 0
fi
(( rc == 0 )) || { echo "runner exit $rc"; exit $rc; }

python3 - "$out" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
bad = []
if s.get("routes_never_finished"): bad.append("never finished: %s" % s["routes_never_finished"])
if s.get("restarts"): bad.append("server restarts: %s" % s["restarts"])
for rr in out.glob("attempts/*/*/route_result.json"):
    st = json.loads(rr.read_text()).get("status")
    if st != "finished": bad.append("%s: %s" % (rr.parent, st))
for d in sorted(out.glob("attempts/*/*")):
    if not (d / "plans.jsonl").exists() or not (d / "plans.jsonl").stat().st_size: bad.append("%s: no plans" % d)
print("smoke harness check:", "OK" if not bad else "; ".join(bad))
sys.exit(1 if bad else 0)
EOF
