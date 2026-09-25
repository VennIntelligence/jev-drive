#!/usr/bin/env bash
# openpilot zero-shot Bench2Drive exam with the fixed adapter (todos/2026-09-24-zeroshot-exam/openpilot-migration.md):
# rear-axle plan origin, every-tick synced cameras, 5 s warm-up. Controllers: "zoo_pid" = Bench2DriveZoo's official
# UniAD/VAD PID run as shipped (primary, user decision 2026-09-25), "native" = openpilot's desired curvature / accel
# (paired secondary), "fixed" = the pre-registered repo controller (for the before/after comparison with the smoke).
# Run under scripts/slot_run.sh (it sets CUDA_VISIBLE_DEVICES); CARLA takes the same card via --gpu-rank.
#
#   smoke  <gpu>  (done 2026-09-25 02:22) 5 pre-registered routes x zoo-lebowski, zoo-cinque, native-cinque,
#                 fixed-lebowski, shadow-cinque; 2 workers.
#   accept <gpu>  adapter acceptance (docs/zeroshot-adapters.md; migration doc D3): 5 routes not in any smoke x the
#                 TCP partnership with Lebowski under both Zoo PID switch sets the Alpamayo exam may freeze (f1, f1f2b);
#                 scripts/zeroshot_b2d_op_accept.py writes accept.json; exit 1 if neither arm passes.
#   smoke3 <gpu>  the 5 pre-registered routes + 4 pre-registered junction routes x partner-lebowski (primary),
#                 pure-lebowski, tcp-alone, partner-native-lebowski; needs the Alpamayo choice and an accepted arm.
#   full   <gpu>  220 routes, 4 workers, resumable: the configuration chosen from smoke3 by the rule in D3.
# Exit non-zero only on infrastructure failure (runner error, routes never finished, harness status, no plans) or a
# failed pre-registered gate. Driving outcome never. A policy server that dies is logged (its exit status, the
# sender of any catchable signal, a ps snapshot, the container cgroup's memory / pids events) and restarted by a
# watchdog; the phase is then resumed (b2d_run skips finished routes). CARLA server indices 600-699.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
mode=$1 gpu=$2
D=${OP_DIR:-$DATA_DIR/runs/zeroshot-exam/b2d-op}
C=$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json
PY_OP=$DATA_DIR/envs/openpilot/bin/python
ALP_CHOICE=$DATA_DIR/runs/zeroshot-exam/b2d/smoke2-alpamayo-choice.json
TCP_CKPT=$DATA_DIR/models/bench2drive/tcp/tcp_b2d.ckpt       # sha256 e6573ff1..., docs/b2d-tcp-controller.md
mkdir -p "$D/ps"
declare -A spid
cleanup() {  # our policy servers, the watchdog, and CARLA servers our runners started (their pid files, ports 600-699)
    for m in "${!spid[@]}"; do kill -- -"${spid[$m]}" 2>/dev/null; done; kill ${watch:-} 2>/dev/null
    local f p
    for f in "$D"/*/attempts/*/*/route.pid; do       # route processes of our runners
        p=$(cat "$f" 2>/dev/null) || continue
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qF "$D/" && kill "$p" 2>/dev/null
    done
    for f in "$D"/*/servers/carla-*.pid; do
        p=$(cat "$f" 2>/dev/null) || continue
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qE 'carla-rpc-port=3[2-6][0-9]{3}' && kill -- -"$p" "$p" 2>/dev/null
    done
}
trap cleanup EXIT
trap 'exit 129' HUP INT TERM    # run the EXIT trap on a killed window too

launch() {  # model: start its policy server in its own session / process group, wait for ready
    local m=$1 sock=$D/$1-$mode.sock ready=$D/$1-$mode.ready log=$D/server-$1-$mode.log
    rm -f "$sock" "$ready" "$D/.pid-$m-$mode"
    # own session / process group (setsid in a non-leader child execs in place, so pid = pgid = the server);
    # no `exec -a` renaming: a venv interpreter locates its prefix from argv[0]. The subshell records the exit status
    # (137 = SIGKILL) for the forensics.
    (
        if [[ $m == tcp ]]; then
            cmd=("$DATA_DIR/envs/jevdrive/bin/python" scripts/b2d_tcp_server.py --ckpt "$TCP_CKPT")
        else
            cmd=("$PY_OP" scripts/zeroshot_policy_server.py "$m" --pool "${POOL:-2}")
        fi
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 setsid "${cmd[@]}" --socket "$sock" --ready-file "$ready" >> "$log" 2>&1 &
        echo $! > "$D/.pid-$m-$mode"
        wait $!
        echo "$(date '+%F %T') server $m exited rc=$?" >> "$log"
    ) &
    until [[ -s $D/.pid-$m-$mode ]]; do sleep 0.2; done
    spid[$m]=$(cat "$D/.pid-$m-$mode")
    until [[ -e $ready ]]; do
        kill -0 "${spid[$m]}" 2>/dev/null || { echo "policy server $m died at start-up, see $log" >&2; exit 3; }
        sleep 5
    done
    echo "$(date '+%F %T') server $m ready (pid ${spid[$m]})" | tee -a "$log" >&2
}

cg_snapshot() {  # container cgroup counters (v2; the container is the root cgroup, no sub-cgroups) and pids in use
    local f
    for f in memory.events memory.events.local pids.events pids.current pids.max memory.current memory.max; do
        [[ -r /sys/fs/cgroup/$f ]] && echo "$f: $(tr '\n' ' ' < /sys/fs/cgroup/$f)"
    done
    find /sys/fs/cgroup -mindepth 2 -maxdepth 4 \( -name memory.events -o -name pids.events \) 2>/dev/null \
        | while read -r f; do echo "$f: $(tr '\n' ' ' < "$f")"; done
    echo "threads: $(ps -eLf --no-headers | wc -l)"
    dmesg 2>&1 | tail -5
}

watchdog() {  # every 30 s: ps + cgroup snapshot (kept 60 min); a dead server is logged and restarted
    while sleep 30; do
        { ps -eo pid,ppid,pgid,user,etimes,rss,nlwp,args --sort=pid | cut -c1-240; cg_snapshot; } > "$D/ps/$(date +%H%M%S).txt"
        find "$D/ps" -name '[0-9]*.txt' -mmin +60 -delete
        for m in $(cat "$D/.servers-$mode" 2>/dev/null); do
            p=$(cat "$D/.pid-$m-$mode" 2>/dev/null)
            if [[ -n $p ]] && ! kill -0 "$p" 2>/dev/null && [[ ! -e $D/.restart-$m-$mode ]]; then
                local tag; tag=$(date +%H%M%S)
                cp "$(ls -t "$D"/ps/[0-9]*.txt | sed -n 2p)" "$D/ps/death-$m-$tag.txt" 2>/dev/null
                cg_snapshot > "$D/ps/death-$m-$tag-after.txt"
                echo "$(date '+%F %T') server $m (pid $p) is gone ($(grep 'exited rc=' "$D/server-$m-$mode.log" | tail -1)); last snapshot kept as $D/ps/death-$m-$tag.txt" \
                    | tee -a "$D/server-$m-$mode.log" "$D/server-deaths.log" >&2
                echo restart > "$D/.restart-$m-$mode"
            fi
        done
    done
}

start_servers() {
    : > "$D/.servers-$mode"
    for m in "$@"; do launch "$m"; echo "$m" >> "$D/.servers-$mode"; done
    watchdog & watch=$!
}

revive() {  # restart any server the watchdog found dead; returns 0 if one was restarted
    local did=1
    for m in "${!spid[@]}"; do
        if [[ -e $D/.restart-$m-$mode ]] || ! kill -0 "${spid[$m]}" 2>/dev/null; then
            rm -f "$D/.restart-$m-$mode"
            launch "$m"; did=0
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
    for try in 1 2 3; do
        echo "$(date '+%F %T') phase $ph (try $try)" >&2
        "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py "$@" --workers "$w" --server-index "$sidx" \
            --gpu-rank "$gpu" --python "$DATA_DIR/envs/b2d-tcp/bin/python" --agent scripts/b2d_zeroshot_agent.py \
            --agent-config "$D/agent-$ph.json" --decimate 4 --no-spectator --max-attempts 2 --no-reap \
            --out "$D/$mode-$ph"
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

zoo_switches() {  # Alpamayo-exam arm -> the Zoo PID switches it freezes
    case $1 in
        f1) echo ', "plan_forward_only": true, "zoo_cadence": "plan"' ;;
        f1f2b) echo ', "plan_forward_only": true, "zoo_cadence": "tick"' ;;
        *) return 1 ;;
    esac
}
partner() {  # junctions [tcp_only] -> the "partner" config key
    echo "\"partner\": {\"ckpt\": \"$TCP_CKPT\", \"socket\": \"$D/tcp-$mode.sock\", \"junctions\": $1, \"tcp_only\": ${2:-false}}"
}
alp_arm() {  # the Zoo PID arm the Alpamayo smoke2 acceptance froze (the openpilot exam must use the same)
    local a
    a=$(python3 -c "import json; print(json.load(open('$ALP_CHOICE'))['choice'] or '')" 2>/dev/null)
    [[ $a == f1 || $a == f1f2b ]] || { echo "no Alpamayo choice in $ALP_CHOICE" >&2; return 1; }
    echo "$a"
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
        run $ph 2 600 "${R[@]}" || fail=1
        harness_check "$D/smoke-$ph" || fail=1
    done
    exit $fail ;;
accept)
    # Adapter acceptance before any scored run (docs/zeroshot-adapters.md, migration doc D3). Routes in no smoke:
    # 2086 NonSignalizedJunctionLeftTurn, 2903 NonSignalizedJunctionRightTurn, 3144 VanillaSignalizedTurnEncounter-
    # RedLight, 2416 VanillaNonSignalizedTurnEncounterStopsign, 3540 HardBreakRoute (lead brakes hard, resume).
    R=(--route-ids ${ACCEPT_ROUTES:-2086,2903,3144,2416,3540})
    for arm in f1 f1f2b; do
        config acc-$arm lebowski zoo_pid model 5 "$(zoo_switches $arm), \"desire\": false, $(partner true)"
    done
    start_servers lebowski tcp
    fail=0
    for arm in f1 f1f2b; do
        run acc-$arm "${ACCEPT_WORKERS:-4}" 600 "${R[@]}" || fail=1
        harness_check "$D/accept-acc-$arm" || fail=1
    done
    (( fail )) && exit 1
    python3 scripts/zeroshot_b2d_op_accept.py f1="$D/accept-acc-f1" f1f2b="$D/accept-acc-f1f2b" \
        --zoo "$DATA_DIR/third_party/Bench2DriveZoo" --out "$D/accept.json" ;;
smoke3)
    # Pre-registered in the migration doc, section D3, before any run of this mode.
    arm=$(alp_arm) || exit 2
    python3 -c "import json,sys; r=json.load(open('$D/accept.json')); sys.exit(0 if r['$arm']['pass'] else 1)" \
        || { echo "acceptance did not pass for the frozen arm $arm; no scored run" >&2; exit 2; }
    R=(--route-ids 2390,24211,1711,2373,3564,2084,2115,3936,2050)
    Z=$(zoo_switches "$arm")
    config partner-lebowski lebowski zoo_pid model 5 "$Z, \"desire\": false, $(partner true)"
    config pure-lebowski lebowski zoo_pid model 5 "$Z, \"desire\": false"
    config tcp-alone lebowski zoo_pid model 0 "$Z, \"desire\": false, $(partner true true)"
    config partner-native-lebowski lebowski native model 5 ", \"desire\": false, $(partner true)"
    start_servers lebowski tcp
    fail=0
    for ph in partner-lebowski pure-lebowski tcp-alone partner-native-lebowski; do
        run $ph 4 620 "${R[@]}" || fail=1
        harness_check "$D/smoke3-$ph" || fail=1
    done
    python3 scripts/zeroshot_b2d_junctions.py "$D"/smoke3-{partner-lebowski,pure-lebowski,tcp-alone,partner-native-lebowski} \
        --csv "$D/smoke3-routes.csv" | tee "$D/smoke3-summary.csv"
    (( fail )) && exit 1
    python3 scripts/zeroshot_b2d_op_choose.py "$D" --out "$D/full-choice.json" ;;
full)
    # Configuration chosen from smoke3 by the pre-registered rule (migration doc D3); Lebowski (pre-registered rule
    # of section D: it beat Cinque by 17.8 DS in the smoke).
    ctl=$(python3 -c "import json; print(json.load(open('$D/full-choice.json'))['controller'] or '')")
    [[ $ctl == zoo_pid || $ctl == native ]] || { echo "no full-run choice in $D/full-choice.json" >&2; exit 2; }
    arm=$(alp_arm) || exit 2
    echo "$(date '+%F %T') full: lebowski + TCP partner, controller=$ctl zoo_arm=$arm" | tee "$D/full-choice.txt" >&2
    config full-partner lebowski "$ctl" model 0 "$([[ $ctl == zoo_pid ]] && zoo_switches "$arm"), \"desire\": false, $(partner true)"
    POOL=4 start_servers lebowski tcp
    run full-partner 4 660 --towns all
    python3 scripts/zeroshot_b2d_summary.py "$D/full-full-partner" | tee "$D/full-summary.txt"
    python3 scripts/zeroshot_b2d_junctions.py "$D/full-full-partner" --csv "$D/full-routes.csv" | tee "$D/full-junctions.csv"
    python3 -c "import json,sys; s=json.load(open('$D/full-full-partner/summary.json')); n=len(s['routes_never_finished']); \
print('never finished:', s['routes_never_finished']); sys.exit(0 if n <= 15 else 1)" || exit 1
    exit 0 ;;
*) echo "mode must be smoke, accept, smoke3 or full" >&2; exit 2 ;;
esac
