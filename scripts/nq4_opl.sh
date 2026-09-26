#!/usr/bin/env bash
# Night queue 4 OPL: openpilot's native plan in Bench2Drive closed loop with the registered launch strategy, agent mode
# op_native_launch (the D3 TCP partner for launches only, "launch" in scripts/b2d_zeroshot_agent.py). Registration and
# every operative choice: todos/2026-09-26-night-queue-3.md, CL section, the [OPL] entries.
#
#   scripts/nq4_opl.sh smoke     acceptance on the debug card: CL2p and CL7p on the 3 smoke routes (first 200 requests
#                                dumped raw, rule-8 check scripts/nq4_opl_check.py) + 10 random 220 routes; gates by
#                                jevdrive.nq4_opl_report accept -> runs/nq4/opl/accept/{PASS,FAIL}
#   scripts/nq4_opl.sh chain     per arm: pilot on the debug card (1 route, sanity checklist; 10 routes, checklist), then
#                                the full route set on the batch cards of runs/nq4/opl/GO. Arms: CL2p (220, seed 0),
#                                CL7p (220, seed 0), CL5dp (obstacle routes, seeds 0 1 2; waits for
#                                runs/nq3/q2/closed_loop_head/READY, then its own 3-route smoke + rule-8 check first)
#   scripts/nq4_opl.sh arm <arm> <seed> <route ids | all | obstacle> <est_h> <raw_dump> <dump_every> <out>
#                                one run on the current resources (debugging)
#
# Resources. Debug card (smoke, pilots): $DEBUG_GPU, else the row of runs/sched/table.tsv (SCH) that names the
# infrastructure / debug card; 2 CARLA servers at indices 130-131, cores 200-203. Batch (full runs): runs/nq4/opl/GO,
# a shell fragment written by SCH / Codex / main, e.g.
#     GPUS="3 4"; WORKERS=5; IDX0=130; IDX_SPAN=5; OPL_CPUS=60-89
# (index i binds RPC 2000 + 50i and TM 8000 + 50i: i, i + 120 and i - 120 must be free). Without a GO file, after
# $OPL_GO_WAIT_MIN minutes the full runs fall back to the debug card if it has room (CarlaUE4 <= 6 - workers, >= 40 GB free
# VRAM, pids.current < 15000). The GO file is re-read before every arm. Estimates: worker-minutes per route from
# runs/nq4/opl/est.env (MIN_CL2P / MIN_CL7P / MIN_CL5DP, measured in the acceptance); a stage past twice its estimate
# stops the chain.
# Hand-offs: runs/nq4/opl/{STATUS.md, ERROR, DONE, events.jsonl, log.txt}; tables in runs/nq4/opl/results/. Only processes
# whose PIDs this script recorded are ever killed.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
REPO=$(pwd)
NQ3=$DATA_DIR/runs/nq3
O=$DATA_DIR/runs/nq4/opl
mkdir -p "$O/srv" "$O/cfg" "$O/arms" "$O/accept"
export B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000}
export B2D_SENSOR_TICK=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
XML=$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml
P7=$REPO/todos/2026-09-23-tfv6-controller/controller-eval/P7.json
TCP_CKPT=$DATA_DIR/models/bench2drive/tcp/tcp_b2d.ckpt
PY_CARLA=$DATA_DIR/envs/carla/bin/python PY_TCP=$DATA_DIR/envs/b2d-tcp/bin/python PY_SCOUT=$DATA_DIR/envs/scout-tfv6/bin/python
PY_OP=$DATA_DIR/envs/openpilot/bin/python PY_JD=$DATA_DIR/envs/jevdrive/bin/python PY_VENV=$REPO/.venv/bin/python
SMOKE_ROUTES=2084,24211,2668
ACC_ROUTES=28330,3890,26966,10857,2086,3378,23695,18356,4468,27529     # random.Random(0).sample, [OPL] 18:35 item 3
DBG_WORKERS=2 DBG_IDX0=130 DBG_CPUS=${DBG_CPUS:-200-203}
OPL_GO_WAIT_MIN=${OPL_GO_WAIT_MIN:-60}

ev() { printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N | cut -c1-14)" "$1" "${2:+, $2}" >> "$O/events.jsonl"; }
log() { echo "$(date '+%F %T') $*" | tee -a "$O/log.txt" >&2; }
error() {  # error <reason> [log file]: write ERROR and stop
    {
        echo "# OPL ERROR $(date '+%F %T %Z')"; echo; echo "reason: $1"; echo
        echo "current: $(cat "$O/CURRENT" 2>/dev/null)"; echo
        [[ -n ${2:-} && -f $2 ]] && { echo '```'; tail -50 "$2"; echo '```'; }
    } > "$O/ERROR"
    ev error "\"reason\": \"$1\""
    log "ERROR: $1"
    exit 1
}

debug_gpu() {  # the infrastructure / debug card: $DEBUG_GPU, else SCH's runs/sched/table.tsv
    [[ -n ${DEBUG_GPU:-} ]] && { echo "$DEBUG_GPU"; return; }
    python3 - "$DATA_DIR/runs/sched/table.tsv" <<'EOF'
import csv, re, sys
try:
    rows = list(csv.reader(open(sys.argv[1]), delimiter="\t"))
except OSError:
    sys.exit(1)
head = [h.strip().lower() for h in rows[0]] if rows else []
col = next((i for i, h in enumerate(head) if h in ("gpu", "card", "gpus", "cards")), None)
for r in rows[1:]:
    if any(re.search(r"debug|infra|repair|validation", c, re.I) for c in r):
        v = r[col] if col is not None and col < len(r) else next((c for c in r if re.fullmatch(r"\d", c.strip())), "")
        m = re.search(r"\d+", v)
        if m:
            print(m.group(0)); sys.exit(0)
sys.exit(1)
EOF
}
use_debug() {  # resources = the debug card
    local g; g=$(debug_gpu) || error "no debug card: set DEBUG_GPU or wait for SCH's runs/sched/table.tsv"
    GPUS=$g WORKERS=$DBG_WORKERS IDX0=$DBG_IDX0 IDX_SPAN=$DBG_WORKERS OPL_CPUS=$DBG_CPUS
}
use_go() {  # resources = the GO file (0 if there is none)
    [[ -f $O/GO ]] || return 1
    GPUS= WORKERS=4 IDX0=130 IDX_SPAN= OPL_CPUS=$DBG_CPUS
    # shellcheck disable=SC1091
    source "$O/GO"
    [[ -n $GPUS ]] || error "GO file without GPUS"
    IDX_SPAN=${IDX_SPAN:-$WORKERS}
}
debug_has_room() {
    local g c free p
    g=$(debug_gpu) || return 1
    c=$(nvidia-smi -i "$g" --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c CarlaUE4)
    free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    p=$(cat /sys/fs/cgroup/pids.current)
    (( c + DBG_WORKERS <= 6 && free >= 40000 && p < 15000 ))
}
ports_free() {  # ports_free <first index> <n>: RPC (+1, +2) and TM ports of the indices, and the i +- 120 partners, unbound
    python3 - "$1" "$2" <<'EOF'
import sys
i0, n = int(sys.argv[1]), int(sys.argv[2])
ports = set()
for f in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(f).readlines()[1:]:
        p = line.split()
        if p[3] == "0A":
            ports.add(int(p[1].split(":")[1], 16))
need = {q for i in range(i0, i0 + n) for q in (2000 + 50 * i, 2001 + 50 * i, 2002 + 50 * i, 8000 + 50 * i)}
sys.exit(1 if need & ports else 0)
EOF
}

# ---------------------------------------------------------------- model servers (own session; PID recorded)
srv_start() {  # srv_start <name> <gpu> <cmd ...>
    local name=$1 gpu=$2; shift 2
    srv_alive "$name" && return 0
    local ready=$O/srv/$name.ready
    rm -f "$ready" "$O/srv/$name.sock"
    (
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 setsid taskset -c "$OPL_CPUS" \
            "$@" --socket "$O/srv/$name.sock" --ready-file "$ready" >> "$O/srv/$name.log" 2>&1 &
        echo $! > "$O/srv/$name.pid"
        wait $!
        echo "$(date '+%F %T') server $name exited rc=$?" >> "$O/srv/$name.log"
    ) &
    until [[ -s $O/srv/$name.pid ]]; do sleep 0.2; done
    local t0=$SECONDS
    until [[ -e $ready ]]; do
        srv_alive "$name" || { log "server $name died at start-up"; return 1; }
        (( SECONDS - t0 > 900 )) && { log "server $name not ready after 15 min"; return 1; }
        sleep 5
    done
    ev server_ready "\"name\": \"$name\", \"pid\": $(cat "$O/srv/$name.pid")"
}
srv_alive() { local p; p=$(cat "$O/srv/$1.pid" 2>/dev/null) && [[ -n $p ]] && kill -0 "$p" 2>/dev/null; }
srv_stop() { local p; p=$(cat "$O/srv/$1.pid" 2>/dev/null) && [[ -n $p ]] && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; }; rm -f "$O/srv/$1.pid"; }
srv_stop_all() { local f; for f in "$O"/srv/*.pid; do [[ -e $f ]] && srv_stop "$(basename "$f" .pid)"; done; }
server_names() {
    case $1 in cl2p) echo "op-cinque-g$2 tcp-g$2" ;; cl7p) echo "op-lebowski-g$2 tcp-g$2" ;; cl5dp) echo "head-g$2 tcp-g$2" ;; esac
}
servers_for() {  # servers_for <arm> <gpu>
    local arm=$1 g=$2
    srv_start tcp-g$g "$g" "$PY_JD" scripts/b2d_tcp_server.py --ckpt "$TCP_CKPT" || return 1
    case $arm in
        cl2p) srv_start op-cinque-g$g "$g" "$PY_OP" scripts/zeroshot_policy_server.py cinque --pool "$WORKERS" ;;
        cl7p) srv_start op-lebowski-g$g "$g" "$PY_OP" scripts/zeroshot_policy_server.py lebowski ;;
        cl5dp) srv_start head-g$g "$g" "$PY_OP" scripts/nq3_cl_server.py --pool "$WORKERS" ;;
    esac
}

kill_runs() {  # the runners, route processes and CARLA servers recorded under one out dir
    local out=$1 p f
    for p in $(cat "$out/runner.pids" 2>/dev/null); do kill "$p" 2>/dev/null; done
    sleep 5
    for f in "$out"/attempts/*/*/route.pid; do
        p=$(cat "$f" 2>/dev/null) || continue
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qF "$out/" && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; }
    done
    for f in "$out"/servers/carla-*.pid; do
        p=$(cat "$f" 2>/dev/null) || continue
        kill -0 "$p" 2>/dev/null || continue
        pkill -P "$p" 2>/dev/null; kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null
    done
}

# ---------------------------------------------------------------- agent configs (CL2 / CL7 / CL5d of lane B + "launch")
arm_cfg() {  # arm_cfg <arm> <gpu> <seed> <raw_dump> <dump_every>
    local arm=$1 g=$2 seed=$3 raw=$4 dump=$5 f=$O/cfg/$1-g$2-s$3-r$4-d$5.json
    local ctl="\"controller\": \"fixed\", \"controller_preset\": \"pursuit\", \"controller_config\": \"$P7\""
    local op="\"op_camera_tick\": 0.05, \"plan_origin\": \"rear\", \"warmup_s\": 5.0, \"desire\": true"
    local launch="\"launch\": {\"ckpt\": \"$TCP_CKPT\", \"socket\": \"$O/srv/tcp-g$g.sock\"}"
    case $arm in
        cl2p) echo "{\"model\": \"cinque\", \"socket\": \"$O/srv/op-cinque-g$g.sock\", \"plan_every\": 1, \"ctl_every\": 4, $op, $ctl, $launch, \"seed\": $seed, \"dump_every\": 0, \"raw_dump\": $raw}" ;;
        cl7p) echo "{\"model\": \"lebowski\", \"socket\": \"$O/srv/op-lebowski-g$g.sock\", \"plan_every\": 4, $op, $ctl, $launch, \"seed\": $seed, \"dump_every\": 0, \"raw_dump\": $raw}" ;;
        cl5dp) echo "{\"model\": \"head\", \"warmup_s\": 5.0, \"desire\": true, \"head_cam_tick\": 0.0, \"arm\": \"q2d\", \"socket\": \"$O/srv/head-g$g.sock\", $ctl, $launch, \"seed\": $seed, \"dump_every\": $dump}" ;;
    esac > "$f"
    echo "$f"
}
route_python() { case $1 in cl5dp) echo "$PY_SCOUT" ;; *) echo "$PY_TCP" ;; esac; }
arm_routes() {  # arm_routes <all | obstacle | ids> -> comma list in XML order
    python3 - "$XML" "$1" <<'EOF'
import sys, xml.etree.ElementTree as ET
keep = {"Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays", "ParkedObstacle",
        "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays"}
rs = list(ET.parse(sys.argv[1]).getroot().iter("route"))
sel = sys.argv[2]
if sel == "all":
    ids = [r.get("id") for r in rs]
elif sel == "obstacle":
    ids = [r.get("id") for r in rs if any(s.get("type") in keep for s in r.iter("scenario"))]
else:
    want = set(sel.split(","))
    ids = [r.get("id") for r in rs if r.get("id") in want]
print(",".join(ids))
EOF
}
stages() {  # stages <comma list> -> three lines: the 1-route pilot, the 10-route pilot, the rest (random.Random(1) order)
    python3 -c "
import random, sys
ids = sys.argv[1].split(','); random.Random(1).shuffle(ids)
print(ids[0]); print(','.join(ids[1:11])); print(','.join(ids[11:]))" "$1"
}
est_h() {  # est_h <arm> <n routes>: worker-minutes per route (est.env) over the current workers, + 0.3 h start-up
    local arm=$1 n=$2 m
    [[ -f $O/est.env ]] && source "$O/est.env"
    case $arm in cl2p) m=${MIN_CL2P:-12} ;; cl7p) m=${MIN_CL7P:-10} ;; *) m=${MIN_CL5DP:-12} ;; esac
    python3 -c "print(round($n * $m / 60 / ($(wc -w <<< "$GPUS") * $WORKERS) + 0.3, 2))"
}

# ---------------------------------------------------------------- one run over a route list on the current resources
run_arm() {  # run_arm <arm> <seed> <ids> <est_h> <raw_dump> <dump_every> <out>
    local arm=$1 seed=$2 ids=$3 est=$4 raw=$5 dump=$6 out=$7 k=0 pids=() g t0=$SECONDS try nreq ndone dead
    mkdir -p "$out"
    rm -f "$out"/claims/*.lock
    echo "$arm $seed $out" > "$O/CURRENT"
    ev step_start "\"arm\": \"$arm\", \"seed\": $seed, \"routes\": $(tr ',' '\n' <<< "$ids" | wc -l), \"est_h\": $est, \"out\": \"$out\""
    log "start $arm seed $seed: $(tr ',' '\n' <<< "$ids" | wc -l) routes, estimate $est h, GPUs $GPUS x $WORKERS workers, cores $OPL_CPUS"
    for try in 1 2 3; do
        pids=(); k=0; dead=""
        for g in $GPUS; do servers_for "$arm" "$g" || error "server start failed for $arm on GPU $g" "$O/log.txt"; done
        for g in $GPUS; do
            local base=$((IDX0 + k * IDX_SPAN)) cfg
            ports_free "$base" "$WORKERS" || log "warning: some ports of indices $base.. are bound; b2d_run skips those slots"
            cfg=$(arm_cfg "$arm" "$g" "$seed" "$raw" "$dump")
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 taskset -c "$OPL_CPUS" "$PY_CARLA" scripts/b2d_run.py --routes "$XML" \
                --route-ids "$ids" --workers "$WORKERS" --server-index "$base" --index-span "$IDX_SPAN" --gpu-rank "$g" \
                --tm-seed "$seed" --no-spectator --no-reap --client-threads 8 --max-attempts 3 --stall-s 480 \
                --route-timeout-s 3600 --out "$out" --python "$(route_python "$arm")" --agent scripts/b2d_zeroshot_agent.py \
                --agent-config "$cfg" --fast-copy --cache-lights >> "$out/runner-g$g.log" 2>&1 &
            pids+=($!); echo "${pids[*]}" > "$out/runner.pids"
            sleep 20; k=$((k + 1))
        done
        while :; do
            local alive=0 p n
            for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
            (( alive )) || break
            if (( SECONDS - t0 > $(python3 -c "print(int(2 * $est * 3600))") )); then
                kill_runs "$out"; error "$arm seed $seed ($out) exceeded twice its estimate ($est h)" "$out/runner-g${GPUS%% *}.log"
            fi
            for g in $GPUS; do
                for n in $(server_names "$arm" "$g"); do
                    if ! srv_alive "$n"; then
                        log "server $n died during $arm seed $seed; restarting"; ev server_died "\"name\": \"$n\""
                        dead=1; servers_for "$arm" "$g" || error "server restart failed: $n" "$O/srv/$n.log"
                    fi
                done
            done
            sleep 30
        done
        for p in "${pids[@]}"; do wait "$p"; done
        nreq=$(tr ',' '\n' <<< "$ids" | wc -l)
        ndone=$(for r in ${ids//,/ }; do [[ -e $out/done/$r.json ]] && echo; done | wc -l)
        [[ -z $dead ]] && break
        (( ndone == nreq )) && break
        log "$arm seed $seed: a server died, re-running the unfinished routes (pass $try)"
    done
    ev step_end "\"arm\": \"$arm\", \"seed\": $seed, \"done\": $ndone, \"requested\": $nreq, \"wall_h\": $(python3 -c "print(round(($SECONDS - $t0) / 3600, 3))")"
    log "done $arm seed $seed: $ndone / $nreq routes in $(( (SECONDS - t0) / 60 )) min"
    "$PY_VENV" -m jevdrive.nq4_opl_report >> "$O/report.log" 2>&1 || log "report failed (see report.log)"
    python3 -c "import sys; sys.exit(1 if ($nreq - $ndone) / max($nreq, 1) > 0.10 else 0)" || \
        error "$arm seed $seed: $((nreq - ndone)) of $nreq routes never finished (> 10%)" "$out/runner-g${GPUS%% *}.log"
    return 0
}
stop_arm() { local g n; for g in $GPUS; do for n in $(server_names "$1" "$g"); do srv_stop "$n"; done; done; }

# ---------------------------------------------------------------- acceptance
smoke() {
    use_debug
    local arm
    trap '[[ -n ${arm:-} ]] && { stop_arm "$arm"; kill_runs "$O/accept/$arm-smoke3"; kill_runs "$O/accept/$arm-rand10"; }' EXIT
    for arm in cl2p cl7p; do
        local m=cinque; [[ $arm == cl7p ]] && m=lebowski
        [[ -e $O/accept/$arm-smoke3/DONE ]] || { run_arm "$arm" 0 "$SMOKE_ROUTES" 1.5 200 0 "$O/accept/$arm-smoke3"; date > "$O/accept/$arm-smoke3/DONE"; }
        if [[ ! -e $O/accept/$arm-check.json ]]; then
            local att=() r a
            for r in ${SMOKE_ROUTES//,/ }; do
                a=$(python3 -c "import json;print(json.load(open('$O/accept/$arm-smoke3/done/$r.json'))['attempt'])")
                att+=("$O/accept/$arm-smoke3/attempts/$r/$a")
            done
            CUDA_VISIBLE_DEVICES=$GPUS taskset -c "$OPL_CPUS" "$PY_OP" scripts/nq4_opl_check.py "$m" "${att[@]}" \
                --out "$O/accept/$arm-check.json" >> "$O/accept/$arm-check.log" 2>&1
            log "$arm rule-8 check rc=$? ($O/accept/$arm-check.json)"
        fi
        [[ -e $O/accept/$arm-rand10/DONE ]] || { run_arm "$arm" 0 "$ACC_ROUTES" 2.5 0 0 "$O/accept/$arm-rand10"; date > "$O/accept/$arm-rand10/DONE"; }
        stop_arm "$arm"
    done
    "$PY_VENV" -m jevdrive.nq4_opl_report accept "$O"/accept/cl2p-{smoke3,rand10} > "$O/accept/cl2p.json"
    "$PY_VENV" -m jevdrive.nq4_opl_report accept "$O"/accept/cl7p-{smoke3,rand10} > "$O/accept/cl7p.json"
    log "acceptance written to $O/accept/ (cl2p.json, cl7p.json, *-check.json)"
}

# ---------------------------------------------------------------- the chain
status_loop() {
    while sleep 600; do
        {
            echo "# OPL status $(date '+%F %T %Z')"; echo
            echo "- current: $(cat "$O/CURRENT" 2>/dev/null)"
            echo "- phase: $(cat "$O/PHASE" 2>/dev/null)"
            echo "- debug card: $(debug_gpu 2>/dev/null || echo '?'); GO: $([[ -f $O/GO ]] && tr '\n' ' ' < "$O/GO" || echo none)"
            echo "- pids.current $(cat /sys/fs/cgroup/pids.current), load $(cut -d' ' -f1-3 /proc/loadavg)"
            echo; echo "finished stages:"; ls "$O"/arms/*/s*/STAGE* 2>/dev/null | sed 's/^/- /'
        } > "$O/STATUS.md.tmp" && mv "$O/STATUS.md.tmp" "$O/STATUS.md"
    done
}
pilot() {  # pilot <arm> <seed> <selection>: 1 route, checklist, 10 routes, checklist - on the debug card
    local arm=$1 seed=$2 out=$O/arms/$1/s$2 st s1 s2
    mkdir -p "$out"
    arm_routes "$3" | tr ',' '\n' | python3 -c "import sys, json; print(json.dumps(sys.stdin.read().split()))" > "$out/requested.json"
    mapfile -t st < <(stages "$(arm_routes "$3")")
    s1=${st[0]} s2=${st[1]}
    echo "${st[2]}" > "$out/rest.txt"
    use_debug
    local n
    for n in 1 10; do
        local ids=$s1; [[ $n == 10 ]] && ids=$s2
        [[ -e $out/STAGE$n.ok ]] && continue
        echo "pilot $arm s$seed stage $n (debug card)" > "$O/PHASE"
        run_arm "$arm" "$seed" "$ids" "$(est_h "$arm" "$(tr ',' '\n' <<< "$ids" | wc -l)")" 0 0 "$out"
        if "$PY_VENV" -m jevdrive.nq4_opl_report sanity "$out" "$ids" "$NQ3/b/arms/cl1/s0" > "$out/STAGE$n.json" 2>> "$O/log.txt"; then
            mv "$out/STAGE$n.json" "$out/STAGE$n.ok"; ev stage_ok "\"arm\": \"$arm\", \"seed\": $seed, \"stage\": $n"
        else
            stop_arm "$arm"; error "$arm seed $seed: stage-$n pilot failed the sanity checklist ($out/STAGE$n.json)" "$out/STAGE$n.json"
        fi
    done
    stop_arm "$arm"
}
full() {  # full <arm> <seed>: the rest of the route set on the batch cards
    local arm=$1 seed=$2 out=$O/arms/$1/s$2 rest t0
    [[ -e $out/DONE ]] && return
    rest=$(cat "$out/rest.txt")
    t0=$SECONDS
    until use_go; do                      # wait for SCH's GO; fall back to the debug card after OPL_GO_WAIT_MIN
        echo "waiting for $O/GO ($arm s$seed full)" > "$O/PHASE"
        if (( SECONDS - t0 > 60 * OPL_GO_WAIT_MIN )) && debug_has_room; then
            use_debug; log "no GO after $OPL_GO_WAIT_MIN min: $arm s$seed full runs on the debug card"; break
        fi
        sleep 300
    done
    echo "full $arm s$seed on GPUs $GPUS" > "$O/PHASE"
    [[ -n $rest ]] && run_arm "$arm" "$seed" "$rest" "$(est_h "$arm" "$(tr ',' '\n' <<< "$rest" | wc -l)")" 0 0 "$out"
    stop_arm "$arm"
    local nreq ndone
    nreq=$(python3 -c "import json;print(len(json.load(open('$out/requested.json'))))"); ndone=$(ls "$out/done" | wc -l)
    printf '{"arm": "%s", "seed": %s, "done": %s, "requested": %s, "finished": "%s"}\n' "$arm" "$seed" "$ndone" "$nreq" \
        "$(date '+%F %T')" > "$out/DONE"
}
cl5dp_smoke() {  # 3 smoke routes, every request dumped, rule-8 check of the q2d path (debug card)
    local s=$O/cl5dp_smoke r a att=()
    [[ -e $s/check.ok ]] && return
    use_debug
    [[ -e $s/DONE ]] || { run_arm cl5dp 0 "$SMOKE_ROUTES" 1.5 0 1 "$s"; date > "$s/DONE"; }
    stop_arm cl5dp
    for r in ${SMOKE_ROUTES//,/ }; do
        a=$(python3 -c "import json;print(json.load(open('$s/done/$r.json'))['attempt'])"); att+=("$s/attempts/$r/$a")
    done
    CUDA_VISIBLE_DEVICES=$GPUS taskset -c "$OPL_CPUS" "$PY_OP" scripts/nq4_opl_check.py q2d "${att[@]}" --out "$s/check.json" \
        >> "$s/check.log" 2>&1 || error "CL5dp rule-8 check failed ($s/check.json)" "$s/check.log"
    cp "$s/check.json" "$s/check.ok"
}
chain() {
    status_loop & local st=$!
    trap 'kill $st 2>/dev/null; [[ -f $O/CURRENT ]] && kill_runs "$(cut -d" " -f3 "$O/CURRENT")"; srv_stop_all' EXIT
    trap 'exit 129' HUP INT TERM
    [[ -e $O/accept/PASS ]] || error "acceptance has not passed (runs/nq4/opl/accept/PASS is written by the executor)"
    pilot cl2p 0 all; full cl2p 0
    pilot cl7p 0 all; full cl7p 0
    until [[ -e $NQ3/q2/closed_loop_head/READY ]]; do echo "waiting for q2 READY (CL5dp)" > "$O/PHASE"; sleep 300; done
    cl5dp_smoke
    local s
    for s in 0 1 2; do pilot cl5dp "$s" obstacle; full cl5dp "$s"; done
    "$PY_VENV" -m jevdrive.nq4_opl_report >> "$O/report.log" 2>&1
    date '+%F %T' > "$O/DONE"; ev end '"what": "OPL chain done"'; log "OPL chain done"
}

case ${1:-} in
    smoke) trap 'exit 129' HUP INT TERM; smoke ;;
    chain) chain ;;
    arm) shift; use_go || use_debug; trap 'stop_arm "$1"; kill_runs "$7"' EXIT; trap 'exit 129' HUP INT TERM; run_arm "$@" ;;
    *) sed -n 2,30p "$0"; exit 1 ;;
esac
