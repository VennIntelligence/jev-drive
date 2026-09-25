#!/usr/bin/env bash
# openpilot zero-shot Bench2Drive exam with the fixed adapter (todos/2026-09-24-zeroshot-exam/openpilot-migration.md):
# rear-axle plan origin, every-tick synced cameras, 5 s warm-up. Controllers: "zoo_pid" = Bench2DriveZoo's official
# UniAD/VAD PID run as shipped (primary, user decision 2026-09-25), "native" = openpilot's desired curvature / accel
# (paired secondary), "fixed" = the pre-registered repo controller (for the before/after comparison with the smoke).
# Run under scripts/slot_run.sh (it sets CUDA_VISIBLE_DEVICES); CARLA takes the same card via --gpu-rank.
#
#   smoke  <gpu>  (done 2026-09-25 02:22) 5 pre-registered routes x zoo-lebowski, zoo-cinque, native-cinque,
#                 fixed-lebowski, shadow-cinque; 2 workers.
#   smoke2 <gpu>  engage-while-rolling + turn guidance (migration doc, section D2): 9 routes (the 5 smoke routes + the
#                 lowest route id of each of 4 junction-turn scenario types) x eng-implicit-lebowski,
#                 eng-desire-lebowski, eng-handover-lebowski, eng-implicit-cinque; 2 workers.
#   full   <gpu>  220 routes, 4 workers, resumable, with the model and guidance chosen from smoke2 by the rule in D2.
# Exit non-zero only on infrastructure failure (runner error, routes never finished, harness status, no plans).
# Driving outcome never. A policy server that dies is logged (sender of any catchable signal, a ps snapshot) and
# restarted by a watchdog; the phase is then resumed (b2d_run skips finished routes).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
mode=$1 gpu=$2
D=$DATA_DIR/runs/zeroshot-exam/b2d-op
C=$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json
PY_OP=$DATA_DIR/envs/openpilot/bin/python
mkdir -p "$D/ps"
declare -A spid
cleanup() { for m in "${!spid[@]}"; do kill -- -"${spid[$m]}" 2>/dev/null; done; kill ${watch:-} 2>/dev/null; }
trap cleanup EXIT

launch() {  # model: start its policy server in its own session / process group, wait for ready
    local m=$1 sock=$D/$1-$mode.sock ready=$D/$1-$mode.ready
    rm -f "$sock" "$ready"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 setsid bash -c "exec -a opb2d-policy-$m $PY_OP \
        scripts/zeroshot_policy_server.py $m --socket $sock --ready-file $ready" >> "$D/server-$m-$mode.log" 2>&1 &
    spid[$m]=$!
    until [[ -e $ready ]]; do
        kill -0 "${spid[$m]}" 2>/dev/null || { echo "policy server $m died at start-up, see $D/server-$m-$mode.log" >&2; exit 3; }
        sleep 5
    done
    echo "$(date '+%F %T') server $m ready (pid ${spid[$m]})" | tee -a "$D/server-$m-$mode.log" >&2
}

watchdog() {  # every 30 s: ps snapshot (kept 60 min); a dead server is logged and restarted
    while sleep 30; do
        ps -eo pid,ppid,pgid,user,etimes,args --sort=pid | cut -c1-240 > "$D/ps/$(date +%H%M%S).txt"
        find "$D/ps" -name '*.txt' -mmin +60 -delete
        for m in $(cat "$D/.servers-$mode" 2>/dev/null); do
            p=$(cat "$D/.pid-$m-$mode" 2>/dev/null)
            if [[ -n $p ]] && ! kill -0 "$p" 2>/dev/null; then
                echo "$(date '+%F %T') server $m (pid $p) is gone; last ps snapshot kept as $D/ps/death-$m-$(date +%H%M%S).txt" \
                    | tee -a "$D/server-$m-$mode.log" "$D/server-deaths.log" >&2
                cp "$(ls -t "$D"/ps/*.txt | sed -n 2p)" "$D/ps/death-$m-$(date +%H%M%S).txt" 2>/dev/null
                echo restart > "$D/.restart-$m-$mode"
            fi
        done
    done
}

start_servers() {
    : > "$D/.servers-$mode"
    for m in "$@"; do launch "$m"; echo "$m" >> "$D/.servers-$mode"; echo "${spid[$m]}" > "$D/.pid-$m-$mode"; done
    watchdog & watch=$!
}

revive() {  # restart any server the watchdog found dead; returns 0 if one was restarted
    local did=1
    for m in "${!spid[@]}"; do
        if [[ -e $D/.restart-$m-$mode ]] || ! kill -0 "${spid[$m]}" 2>/dev/null; then
            rm -f "$D/.restart-$m-$mode"
            launch "$m"; echo "${spid[$m]}" > "$D/.pid-$m-$mode"; did=0
        fi
    done
    return $did
}

config() {  # name model controller drive dump [extra json]
    local plan_every=1
    [[ $2 == lebowski ]] && plan_every=4
    cat > "$D/agent-$1.json" <<EOF
{"model": "$2", "socket": "$D/$2-$mode.sock", "plan_every": $plan_every, "controller": "$3", "drive": "$4",
 "controller_preset": "carla", "controller_config": "$C", "seed": 0, "dump_every": $5,
 "op_camera_tick": 0.05, "plan_origin": "rear", "warmup_s": 5.0 ${6:-}}
EOF
}

run() {  # phase workers server-index route-args... ; resumes once after a server death
    local ph=$1 w=$2 sidx=$3; shift 3
    for try in 1 2; do
        echo "$(date '+%F %T') phase $ph (try $try)" >&2
        "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py "$@" --workers "$w" --server-index "$sidx" \
            --gpu-rank "$gpu" --python "$DATA_DIR/envs/b2d-tcp/bin/python" --agent scripts/b2d_zeroshot_agent.py \
            --agent-config "$D/agent-$ph.json" --decimate 4 --no-spectator --max-attempts 2 --out "$D/$mode-$ph"
        local rc=$?
        revive || return $rc
        echo "$(date '+%F %T') a policy server died during $ph; restarted, resuming the phase" >&2
    done
}

harness_check() {  # out dir -> non-zero on infrastructure failure only
    python3 - "$1" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
bad = []
if s.get("routes_never_finished"): bad.append("never finished: %s" % s["routes_never_finished"])
for rr in out.glob("attempts/*/*/route_result.json"):
    st = json.loads(rr.read_text()).get("status")
    if st != "finished": bad.append("%s: %s" % (rr.parent.name, st))
for d in sorted(out.glob("attempts/*/*")):
    p = d / "plans.jsonl"
    if not p.exists() or not p.stat().st_size: bad.append("%s: no plans" % d)
print(out.name, "harness check:", "OK" if not bad else "; ".join(bad))
sys.exit(1 if bad else 0)
EOF
}

case $mode in
smoke)
    R=(--route-ids 2390,24211,1711,2373,3564)
    config zoo-lebowski lebowski zoo_pid model 5
    config zoo-cinque cinque zoo_pid model 20
    config native-cinque cinque native model 20
    config fixed-lebowski lebowski fixed model 5
    config shadow-cinque cinque fixed oracle 20
    start_servers lebowski cinque
    fail=0
    for ph in zoo-lebowski zoo-cinque native-cinque fixed-lebowski shadow-cinque; do
        run $ph 2 500 "${R[@]}" || fail=1
        harness_check "$D/smoke-$ph" || fail=1
    done
    exit $fail ;;
smoke2)
    # 5 pre-registered smoke routes + lowest id of NonSignalizedJunctionLeftTurn / NonSignalizedJunctionRightTurn /
    # SignalizedJunctionLeftTurn / SignalizedJunctionRightTurn in bench2drive220.xml
    R=(--route-ids 2390,24211,1711,2373,3564,2084,2115,3936,2050)
    E=', "engage_s": 5.0'
    config eng-implicit-lebowski lebowski zoo_pid model 5 "$E, \"desire\": false"
    config eng-desire-lebowski lebowski zoo_pid model 5 "$E, \"desire\": true"
    config eng-handover-lebowski lebowski zoo_pid model 5 "$E, \"desire\": false, \"junction_handover_m\": 15.0"
    config eng-implicit-cinque cinque zoo_pid model 20 "$E, \"desire\": false"
    start_servers lebowski cinque
    fail=0
    for ph in eng-implicit-lebowski eng-desire-lebowski eng-handover-lebowski eng-implicit-cinque; do
        run $ph 2 500 "${R[@]}" || fail=1
        harness_check "$D/smoke2-$ph" || fail=1
    done
    python3 scripts/zeroshot_b2d_junctions.py "$D"/smoke2-eng-* --csv "$D/smoke2-junctions.csv" | tee "$D/smoke2-summary.csv"
    exit $fail ;;
full)
    # Pre-registered choice (migration doc D2), from smoke2 only, before any full result.
    read -r model guide < <(python3 - "$D" <<'EOF'
import csv, sys
from pathlib import Path
D = Path(sys.argv[1])
rows = {r["phase"]: r for r in csv.DictReader(open(D / "smoke2-summary.csv"))}
def rate(ph):
    p, n = map(int, rows["smoke2-" + ph]["turns"].split("/"))
    return p / n if n else 0.0
ds = lambda ph: float(rows["smoke2-" + ph]["ds"])  # noqa: E731
model = "cinque" if ds("eng-implicit-cinque") >= ds("eng-implicit-lebowski") + 10 else "lebowski"
guide = "implicit"
for cand in ("desire", "handover"):   # less invasive first; a more invasive variant must earn its place
    if rate("eng-%s-lebowski" % cand) >= rate("eng-%s-lebowski" % guide) + 0.20 and \
            ds("eng-%s-lebowski" % cand) >= ds("eng-%s-lebowski" % guide) - 5:
        guide = cand
print(model, guide)
print("smoke2:", {k: (v["ds"], v["turns"]) for k, v in rows.items()}, "-> model", model, "guidance", guide, file=sys.stderr)
EOF
)
    echo "$(date '+%F %T') full: model=$model guidance=$guide" | tee "$D/full-choice.txt" >&2
    extra=', "engage_s": 5.0, "desire": false'
    [[ $guide == desire ]] && extra=', "engage_s": 5.0, "desire": true'
    [[ $guide == handover ]] && extra=', "engage_s": 5.0, "desire": false, "junction_handover_m": 15.0'
    config full-zoo "$model" zoo_pid model 0 "$extra"
    start_servers "$model"
    run full-zoo 4 520 --towns all
    python3 -c "import json,sys; s=json.load(open('$D/full-full-zoo/summary.json')); n=len(s['routes_never_finished']); \
print('never finished:', s['routes_never_finished']); sys.exit(0 if n <= 15 else 1)" || exit 1
    exit 0 ;;
*) echo "mode must be smoke, smoke2 or full" >&2; exit 2 ;;
esac
