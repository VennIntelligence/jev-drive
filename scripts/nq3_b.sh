#!/usr/bin/env bash
# Night queue 3, lane B: the closed-loop CL queue on Bench2Drive 220 (todos/2026-09-26-night-queue-3.md, section CL and
# "timetable / one-shot scripts"; operative choices in the CL section's [B] entries). One chain script, started once in
# tmux jev:nq3-b; nothing is advanced by hand.
#
#   scripts/nq3_b.sh expert [gpus]    CL1a: PDM-Lite expert logs on all 220 routes (SimLingo tree, as the acceptance)
#   scripts/nq3_b.sh smoke            CL0: 3 routes per agent, profiling, the rule-8 dumps of the head arms
#   scripts/nq3_b.sh chain            the queue: CL0 gate -> CL1 ... CL10 by priority; CL5 / CL5d inserted at the current
#                                     position when runs/nq3/q2/closed_loop_head/READY appears, mc_real0 appended on
#                                     runs/nq3/q4a/PASS
#   scripts/nq3_b.sh arm <arm> <seed> <routes> <est_h>   one queue step (resumable), for debugging
#
# Scheduler hooks (SCH, 2026-09-26 19:00; every default reproduces the behaviour before them):
#   $B/GO    grant file, sourced at start and again before every queue step (runs/sched/table.tsv -> GO, see
#            tmp/2026-09-26-codex-handoff.md "全局调度"): GPUS, WORKERS, B_CPUS, B_IDX ("g:index ..." server index base
#            per GPU, default 300 + 30 g), B_EXPAND_GPUS / B_EXPAND_WORKERS / B_CPUS_WIDE (the post-lane-A expansion).
#   $B/SKIP  "arm seed" lines: the chain skips those queue steps (re-read before every step).
#   B_DIR    lane directory (default runs/nq3/b); B_REPORT=0 skips the table refresh (pilots in their own B_DIR).
#
# Resources (fixed by main): GPUs 0,1,2, <= 6 CARLA servers each, 50 cores (taskset $B_CPUS), --client-threads 8; after
# runs/nq3/a/v1/DONE and once no CARLA server is left on GPUs 3-5: GPUs 0-5 and 90 cores. A new CARLA server waits while
# the cgroup's pids.current > 17000 (B2D_PIDS_WAIT, scripts/b2d_run.py). CARLA server indices 300-479 (30 per GPU).
# Hand-offs: runs/nq3/b/<step>/DONE, runs/nq3/b/ERROR, runs/nq3/b/STATUS.md (every 10 min), runs/nq3/b/events.jsonl.
# Only processes whose PIDs this script recorded are ever killed.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
REPO=$(pwd)
NQ=$DATA_DIR/runs/nq3
B=${B_DIR:-$NQ/b}
B_GO=${B_GO:-$B/GO}
load_go() { [[ -f $B_GO ]] && source "$B_GO"; return 0; }
load_go
mkdir -p "$B/srv" "$B/cfg" "$B/arms" "$B/steps"
B_CPUS=${B_CPUS:-60-109}
B_CPUS_WIDE=${B_CPUS_WIDE:-60-149}
GPUS=${GPUS:-0 1 2}
WORKERS=${WORKERS:-6}
export B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000}
export B2D_SENSOR_TICK=1        # cameras honour their spec's sensor_tick (b2d_hooks; the leaderboard drops it otherwise)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
SIM=$DATA_DIR/third_party/simlingo
XML=$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml
P7=$REPO/todos/2026-09-23-tfv6-controller/controller-eval/P7.json
PY_CARLA=$DATA_DIR/envs/carla/bin/python PY_TCP=$DATA_DIR/envs/b2d-tcp/bin/python
PY_SCOUT=$DATA_DIR/envs/scout-tfv6/bin/python PY_OP=$DATA_DIR/envs/openpilot/bin/python
PY_VENV=$REPO/.venv/bin/python PY_ULT=$DATA_DIR/envs/ultralytics/bin/python
PY_ALP=$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python
SMOKE_ROUTES=${SMOKE_ROUTES:-2084,24211,2668}
idx_of() {  # idx_of <gpu>: first CARLA server index of that GPU's block (B_IDX "g:i" entry, else 300 + 30 g)
    local e; for e in ${B_IDX:-}; do [[ ${e%%:*} == "$1" ]] && { echo "${e#*:}"; return; }; done
    echo $((300 + 30 * $1))
}

ev() {  # ev <kind> [json fields without braces]
    printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N | cut -c1-14)" "$1" "${2:+, $2}" >> "$B/events.jsonl"
}
log() { echo "$(date '+%F %T') $*" | tee -a "$B/log.txt" >&2; }
error() {  # error <reason> [log file]: write ERROR and stop the chain
    {
        echo "# lane B ERROR $(date '+%F %T %Z')"; echo; echo "reason: $1"; echo
        echo "current step: $(cat "$B/CURRENT" 2>/dev/null)"; echo "progress: $(progress_line)"; echo
        [[ -n ${2:-} && -f $2 ]] && { echo '```'; tail -50 "$2"; echo '```'; }
    } > "$B/ERROR"
    ev error "\"reason\": \"$1\""
    log "ERROR: $1"
    exit 1
}
progress_line() {
    local s=$(cat "$B/CURRENT" 2>/dev/null) d
    [[ -z $s ]] && return
    d=$B/arms/${s% *}/s${s#* }
    [[ -f $d/requested.json ]] || return
    echo "$(ls "$d/done" 2>/dev/null | wc -l) / $(python3 -c "import json;print(len(json.load(open('$d/requested.json'))))") routes"
}

# ---------------------------------------------------------------- model servers (own session; PID recorded)
srv_start() {  # srv_start <name> <gpu> <cmd ...>: start unless alive, wait for its ready file
    local name=$1 gpu=$2; shift 2
    srv_alive "$name" && return 0
    local ready=$B/srv/$name.ready
    rm -f "$ready" "$B/srv/$name.sock"
    (
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com setsid taskset -c "$B_CPUS" \
            "$@" --socket "$B/srv/$name.sock" --ready-file "$ready" >> "$B/srv/$name.log" 2>&1 &
        echo $! > "$B/srv/$name.pid"
        wait $!
        echo "$(date '+%F %T') server $name exited rc=$?" >> "$B/srv/$name.log"
    ) &
    until [[ -s $B/srv/$name.pid ]]; do sleep 0.2; done
    local t0=$SECONDS
    until [[ -e $ready ]]; do
        srv_alive "$name" || { log "server $name died at start-up"; return 1; }
        (( SECONDS - t0 > 900 )) && { log "server $name not ready after 15 min"; return 1; }
        sleep 5
    done
    ev server_ready "\"name\": \"$name\", \"pid\": $(cat "$B/srv/$name.pid")"
}
srv_alive() { local p; p=$(cat "$B/srv/$1.pid" 2>/dev/null) && [[ -n $p ]] && kill -0 "$p" 2>/dev/null; }
srv_stop() { local p; p=$(cat "$B/srv/$1.pid" 2>/dev/null) && [[ -n $p ]] && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; }; rm -f "$B/srv/$1.pid"; }
srv_stop_arm() { local g n; for g in $GPUS; do for n in $(server_names "$1" "$g"); do srv_stop "$n"; done; done; }
srv_stop_all() { local f; for f in "$B"/srv/*.pid; do [[ -e $f ]] && srv_stop "$(basename "$f" .pid)"; done; }

servers_for() {  # servers_for <arm> <gpu>: the model servers an arm needs on one GPU
    local arm=$1 g=$2
    case $arm in
        cl1|tfv6|bridgedrive|simlingo|blue) ;;
        cl2|cl9_cl2) srv_start op-cinque-g$g "$g" "$PY_OP" scripts/zeroshot_policy_server.py cinque --pool "$WORKERS" ;;
        cl7) srv_start op-lebowski-g$g "$g" "$PY_OP" scripts/zeroshot_policy_server.py lebowski ;;
        cl8) srv_start alpamayo-g$g "$g" "$PY_ALP" scripts/zeroshot_policy_server.py alpamayo ;;
        cl3|cl5|cl5d) srv_start head-g$g "$g" "$PY_OP" scripts/nq3_cl_server.py --pool "$WORKERS" ;;
        cl4|mc_real0)
            srv_start qwen-g$g "$g" "$PY_VENV" scripts/nq3_feat_server.py qwen || return 1
            srv_start headq-g$g "$g" "$PY_OP" scripts/nq3_cl_server.py --pool "$WORKERS" --qwen "$B/srv/qwen-g$g.sock" ;;
        cl6)
            srv_start yolo-g$g "$g" "$PY_ULT" scripts/nq3_feat_server.py yolo || return 1
            srv_start heady-g$g "$g" "$PY_OP" scripts/nq3_cl_server.py --pool "$WORKERS" --yolo "$B/srv/yolo-g$g.sock" ;;
        *) log "no server recipe for $arm"; return 1 ;;
    esac
}
server_names() {  # the server names an arm uses on one GPU
    case $1 in
        cl2) echo op-cinque-g$2 ;; cl7) echo op-lebowski-g$2 ;; cl8) echo alpamayo-g$2 ;; cl3|cl5|cl5d) echo head-g$2 ;;
        cl4|mc_real0) echo qwen-g$2 headq-g$2 ;; cl6) echo yolo-g$2 heady-g$2 ;; cl9_cl2) echo op-cinque-g$2 ;;
    esac
}

kill_runs() {  # kill_runs <out>: the runners, route processes and CARLA servers recorded under one arm's out dir
    local out=$1 p f
    for p in $(cat "$out/runner.pids" 2>/dev/null); do kill "$p" 2>/dev/null; done
    sleep 5
    for f in "$out"/attempts/*/*/route.pid; do
        p=$(cat "$f" 2>/dev/null) || continue
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qF "$out/" && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; }
    done
    for f in "$out"/servers/carla-*.pid; do       # the CarlaUE4.sh wrapper and its shipping child
        p=$(cat "$f" 2>/dev/null) || continue
        kill -0 "$p" 2>/dev/null || continue
        pkill -P "$p" 2>/dev/null; kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null
    done
}

# ---------------------------------------------------------------- agent configs
arm_cfg() {  # arm_cfg <arm> <gpu> <seed> <dump_every> -> path of the agent config
    local arm=$1 g=$2 seed=$3 dump=$4 f=$B/cfg/${CFG_TAG:-x}-$1-g$2-s$3-d$4.json
    local ctl="\"controller\": \"fixed\", \"controller_preset\": \"pursuit\", \"controller_config\": \"$P7\""
    local op="\"op_camera_tick\": 0.05, \"plan_origin\": \"rear\", \"warmup_s\": 5.0, \"desire\": true"
    local head="\"model\": \"head\", \"warmup_s\": 5.0, \"desire\": true, \"head_cam_tick\": ${HEAD_CAM_TICK:-0.0}"
    case $arm in
        cl1) echo "{\"model\": \"replay\", \"replay\": \"$B/cl1_expert/logs/{route}.jsonl\", \"replay_plan\": \"time\", \"cameras\": false, \"plan_ticks\": 4, $ctl, \"seed\": $seed, \"dump_every\": 0}" ;;
        cl2) echo "{\"model\": \"cinque\", \"socket\": \"$B/srv/op-cinque-g$g.sock\", \"plan_every\": 1, \"ctl_every\": 4, $op, $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        cl7) echo "{\"model\": \"lebowski\", \"socket\": \"$B/srv/op-lebowski-g$g.sock\", \"plan_every\": 4, $op, $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        cl8) echo "{\"model\": \"alpamayo\", \"socket\": \"$B/srv/alpamayo-g$g.sock\", \"plan_every\": 5, $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        cl3) echo "{$head, \"arm\": \"ridge_late\", \"socket\": \"$B/srv/head-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        cl4) echo "{$head, \"arm\": \"mc\", \"socket\": \"$B/srv/headq-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        cl5) echo "{$head, \"arm\": \"q2\", \"socket\": \"$B/srv/head-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        cl5d) echo "{$head, \"arm\": \"q2d\", \"socket\": \"$B/srv/head-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        mc_real0) echo "{$head, \"arm\": \"mc_real0\", \"socket\": \"$B/srv/headq-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
        cl6) echo "{$head, \"arm\": \"student_b\", \"socket\": \"$B/srv/heady-g$g.sock\", $ctl, \"seed\": $seed, \"dump_every\": $dump}" ;;
    esac > "$f"
    echo "$f"
}
route_python() {
    case $1 in cl1|cl8) echo "$PY_CARLA" ;; cl2|cl7) echo "$PY_TCP" ;; *) echo "$PY_SCOUT" ;; esac
}

# ---------------------------------------------------------------- one arm x seed over a route set
run_arm() {  # run_arm <arm> <seed> <routes: all | obstacle | id,id,...> <estimate h> [dump_every] [out]
    local arm=$1 seed=$2 routes=$3 est=$4 dump=${5:-0} out=${6:-$B/arms/$1/s$2}
    local sel k=0 pids=() g t0=$SECONDS try
    mkdir -p "$out"
    rm -f "$out"/claims/*.lock        # the chain runs one arm at a time: any claim left here is a dead runner's
    CFG_TAG=$(echo "$out" | md5sum | cut -c1-8)
    case $routes in
        all) sel=(--routes "$XML" --towns all) ;;
        obstacle) sel=(--routes "$XML" --route-ids "$(obstacle_routes)") ;;
        *) sel=(--routes "$XML" --route-ids "$routes") ;;
    esac
    [[ -f $out/requested.json ]] || route_ids "${sel[@]}" > "$out/requested.json"
    echo "$arm $seed" > "$B/CURRENT"
    ev step_start "\"arm\": \"$arm\", \"seed\": $seed, \"routes\": \"$routes\", \"est_h\": $est"
    log "start $arm seed $seed ($routes, estimate $est h) on GPUs $GPUS, $WORKERS workers each"
    for try in 1 2 3; do
        pids=(); k=0
        for g in $GPUS; do
            servers_for "$arm" "$g" || error "server start failed for $arm on GPU $g" "$B/log.txt"
        done
        for g in $GPUS; do
            if [[ $arm =~ ^(tfv6|bridgedrive|simlingo|blue)$ ]]; then     # CL10: author agents as shipped
                B_CPUS=$B_CPUS scripts/nq3_b_cl10.sh "$arm" "$g" "$WORKERS" "$(idx_of "$g")" "$seed" "$out" \
                    "$([[ $routes == all ]] && echo all || echo "${sel[3]}")" >> "$out/runner-g$g.log" 2>&1 &
                pids+=($!); echo "${pids[*]}" > "$out/runner.pids"; sleep 20; continue
            fi
            local cfg; cfg=$(arm_cfg "$arm" "$g" "$seed" "$dump")
            taskset -c "$B_CPUS" "$PY_CARLA" scripts/b2d_run.py "${sel[@]}" --workers "$WORKERS" \
                --server-index "$(idx_of "$g")" --index-span 30 --gpu-rank "$g" --tm-seed "$seed" --no-spectator \
                --no-reap --client-threads 8 --max-attempts 3 --stall-s 480 --route-timeout-s 3600 --out "$out" \
                --python "$(route_python "$arm")" --agent scripts/b2d_zeroshot_agent.py --agent-config "$cfg" \
                ${RUN_FLAGS---fast-copy --cache-lights} \
                >> "$out/runner-g$g.log" 2>&1 &
            pids+=($!)
            echo "${pids[*]}" > "$out/runner.pids"
            sleep 20; k=$((k + 1))
        done
        local dead=""
        while :; do
            local alive=0 p
            for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
            (( alive )) || break
            if (( SECONDS - t0 > $(python3 -c "print(int(2 * $est * 3600))") )); then
                kill_runs "$out"
                error "$arm seed $seed exceeded twice its estimate ($est h)" "$out/runner-g${GPUS%% *}.log"
            fi
            for g in $GPUS; do       # a model server that died is restarted at once; its routes retry
                for n in $(server_names "$arm" "$g"); do
                    if ! srv_alive "$n"; then
                        log "server $n died during $arm seed $seed; restarting"; ev server_died "\"name\": \"$n\""
                        dead=1; servers_for "$arm" "$g" || error "server restart failed: $n" "$B/srv/$n.log"
                    fi
                done
            done
            sleep 30
        done
        for p in "${pids[@]}"; do wait "$p"; done
        local nreq ndone
        nreq=$(python3 -c "import json;print(len(json.load(open('$out/requested.json'))))")
        ndone=$(ls "$out/done" 2>/dev/null | wc -l)
        [[ -z $dead ]] && break
        (( ndone == nreq )) && break
        log "$arm seed $seed: a server died, re-running the unfinished routes (pass $try)"
    done
    local fail
    fail=$(python3 -c "print(1 if ($nreq - $ndone) / max($nreq, 1) > 0.10 else 0)")
    [[ ${B_REPORT:-1} == 1 ]] && { "$PY_VENV" -m jevdrive.nq3_cl_report >> "$B/report.log" 2>&1 || log "report failed (see report.log)"; }
    ev step_end "\"arm\": \"$arm\", \"seed\": $seed, \"done\": $ndone, \"requested\": $nreq, \"wall_h\": $(python3 -c "print(round(($SECONDS - $t0) / 3600, 3))")"
    printf '{"arm": "%s", "seed": %s, "done": %s, "requested": %s, "wall_h": %s, "out": "%s", "finished": "%s"}\n' \
        "$arm" "$seed" "$ndone" "$nreq" "$(python3 -c "print(round(($SECONDS - $t0) / 3600, 3))")" "$out" "$(date '+%F %T')" \
        > "$out/DONE"
    log "done $arm seed $seed: $ndone / $nreq routes in $(( (SECONDS - t0) / 60 )) min"
    (( fail )) && error "$arm seed $seed: $((nreq - ndone)) of $nreq routes never finished (> 10%)" "$out/runner-g${GPUS%% *}.log"
    return 0
}
route_ids() {  # the route ids b2d_run will run for these selection args, as a JSON list
    python3 - "$@" <<'EOF'
import json, sys, xml.etree.ElementTree as ET
a = sys.argv[1:]
xml = a[a.index("--routes") + 1]
ids = [r.get("id") for r in ET.parse(xml).getroot().iter("route")]
if "--route-ids" in a:
    want = a[a.index("--route-ids") + 1].split(",")
    ids = [i for i in ids if i in want]
print(json.dumps(ids))
EOF
}
obstacle_routes() {  # CL5 / CL5d: Accident / Construction / ParkedObstacle / HazardAtSideLane and their TwoWays
    python3 - "$XML" <<'EOF'
import sys, xml.etree.ElementTree as ET
keep = {"Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays", "ParkedObstacle",
        "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays"}
ids = [r.get("id") for r in ET.parse(sys.argv[1]).getroot().iter("route")
       if any(s.get("type") in keep for s in r.iter("scenario"))]
print(",".join(ids))
EOF
}

# ---------------------------------------------------------------- CL1a: expert logs
expert() {  # expert logs on GPUs $1 (comma list), 6 workers per GPU; resumable
    local out=$B/cl1_expert gpus=${1:-0,1} pids=() g
    mkdir -p "$out"
    ev step_start '"step": "cl1_expert"'
    log "cl1_expert on GPUs $gpus"
    for g in ${gpus//,/ }; do
        BENCH2DRIVE_ROOT=$SIM/Bench2Drive WORK_DIR=$SIM taskset -c "$B_CPUS" "$PY_CARLA" \
            scripts/b2d_run.py --routes "$SIM/leaderboard/data/bench2drive220.xml" --towns all --workers 6 \
            --server-index "$(idx_of "$g")" --index-span 30 --gpu-rank "$g" --tm-seed 0 --no-spectator --no-reap \
            --client-threads 8 --max-attempts 3 --stall-s 480 --route-timeout-s 3600 --out "$out" \
            --python "$DATA_DIR/envs/simlingo/bin/python" --agent scripts/b2d_expert_agent.py \
            --agent-config "expert+nq3" >> "$out/runner-g$g.log" 2>&1 &
        pids+=($!)
        sleep 60
    done
    local bad=0 p d rid a
    for p in "${pids[@]}"; do wait "$p" || bad=1; done
    mkdir -p "$out/logs"
    for d in "$out"/done/*.json; do         # the expert log of each route's finished attempt -> logs/<id>.jsonl
        rid=$(basename "$d" .json)
        a=$(python3 -c "import json;print(json.load(open('$d'))['attempt'])")
        cp "$out/attempts/$rid/$a/expert.jsonl" "$out/logs/$rid.jsonl"
    done
    ev step_end "\"step\": \"cl1_expert\", \"logs\": $(ls "$out/logs" | wc -l), \"runner_bad\": $bad"
    log "cl1_expert: $(ls "$out/logs" | wc -l) route logs (runner status $bad)"
    date '+%F %T' > "$out/DONE"
}

# ---------------------------------------------------------------- status and hand-offs
status_loop() {  # STATUS.md every 10 min
    while sleep 600; do
        {
            echo "# lane B status $(date '+%F %T %Z')"; echo
            echo "- current: $(cat "$B/CURRENT" 2>/dev/null) ($(progress_line))"
            echo "- queue left: $(cat "$B/QUEUE" 2>/dev/null)"
            echo "- GPUs $GPUS, workers/GPU $WORKERS, cores $B_CPUS"
            echo "- pids.current $(cat /sys/fs/cgroup/pids.current), load $(cut -d' ' -f1-3 /proc/loadavg)"
            echo; echo '```'
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
            echo '```'; echo; echo "finished steps:"
            for f in "$B"/arms/*/s*/DONE; do [[ -e $f ]] && echo "- $(cat "$f")"; done; true
        } > "$B/STATUS.md.tmp" && mv "$B/STATUS.md.tmp" "$B/STATUS.md"
    done
}
maybe_expand() {  # after lane A's v1 is done and no CARLA server is left on the added GPUs: $B_EXPAND_GPUS (default 0-5)
    local target=${B_EXPAND_GPUS:-0 1 2 3 4 5} add="" g n
    [[ $GPUS == "$target" ]] && return
    [[ -e $NQ/a/v1/DONE ]] || return
    for g in $target; do [[ " $GPUS " == *" $g "* ]] || add+=" $g"; done
    n=$(for g in $add; do nvidia-smi -i $g --query-compute-apps=process_name --format=csv,noheader; done | grep -c CarlaUE4)
    (( n == 0 )) || { log "lane A done but $n CARLA servers remain on GPUs$add; not expanding yet"; return; }
    GPUS=$target; B_CPUS=$B_CPUS_WIDE; WORKERS=${B_EXPAND_WORKERS:-$WORKERS}
    ev expand "\"gpus\": \"$GPUS\", \"workers\": $WORKERS, \"cores\": \"$B_CPUS\""
    log "expanded to GPUs $GPUS, $WORKERS workers each, cores $B_CPUS"
}

# ---------------------------------------------------------------- the queue
chain() {
    status_loop & local st=$!
    trap 'kill $st 2>/dev/null; [[ -f $B/CURRENT ]] && { set -- $(cat "$B/CURRENT"); kill_runs "$B/arms/$1/s$2"; }; srv_stop_all' EXIT
    trap 'exit 129' HUP INT TERM
    [[ -e $B/cl0/DONE ]] || error "CL0 (smoke + rule-8 equivalence) has not passed; run the smoke first"
    [[ -e $B/cl1_expert/DONE ]] || expert "${GPUS// /,}"
    # step = "arm seed routes estimate_h"
    local -a Q=("cl1 0 all ${EST_CL1:-1.0}" "cl2 0 all ${EST_CL2:-1.5}" "cl3 0 all ${EST_CL3:-2.0}" "cl4 0 all ${EST_CL4:-4.5}"
                "cl6 0 all ${EST_CL6:-2.5}" "cl7 0 all ${EST_CL7:-1.5}" "cl8 0 all ${EST_CL8:-4.0}"
                "cl2 1 all ${EST_CL2:-1.5}" "cl3 1 all ${EST_CL3:-2.0}" "cl4 1 all ${EST_CL4:-4.5}"
                "cl2 2 all ${EST_CL2:-1.5}" "cl3 2 all ${EST_CL3:-2.0}" "cl4 2 all ${EST_CL4:-4.5}"
                "tfv6 0 all ${EST_CL10:-3.0}" "bridgedrive 0 all ${EST_CL10:-3.0}" "blue 0 all ${EST_CL10:-3.0}"
                "simlingo 0 all ${EST_SIMLINGO:-8.0}")      # SimLingo ~1.1 s / tick, single-core bound (CL10 smoke)
    local inserted=0 appended=0 s
    while (( ${#Q[@]} )); do
        load_go
        maybe_expand
        if (( ! inserted )) && [[ -e $NQ/q2/closed_loop_head/READY ]]; then
            local ob=${EST_CL5:-0.8}
            Q=("cl5 0 obstacle $ob" "cl5d 0 obstacle $ob" "cl5 1 obstacle $ob" "cl5d 1 obstacle $ob"
               "cl5 2 obstacle $ob" "cl5d 2 obstacle $ob" "${Q[@]}")
            inserted=1; ev insert '"what": "cl5, cl5d"'; log "Q2 head READY: CL5 / CL5d inserted"
        fi
        if (( ! appended )) && [[ -e $NQ/q4a/PASS ]]; then
            "$PY_VENV" -m jevdrive.nq3_cl convert-mc-real0 >> "$B/log.txt" 2>&1 || error "mc_real0 package conversion failed" "$B/log.txt"
            Q+=("mc_real0 0 all ${EST_CL4:-4.5}"); appended=1; ev append '"what": "mc_real0"'; log "Q4a PASS: mc_real0 appended"
        fi
        s=${Q[0]}; Q=("${Q[@]:1}")
        printf '%s\n' "${Q[@]}" > "$B/QUEUE"
        set -- $s
        [[ -e $B/arms/$1/s$2/DONE ]] && continue
        if grep -qxF "$1 $2" "$B/SKIP" 2>/dev/null; then
            ev skip "\"arm\": \"$1\", \"seed\": $2"; log "skip $1 seed $2 (listed in $B/SKIP)"; continue
        fi
        run_arm "$1" "$2" "$3" "$4"
        srv_stop_all             # free the cards between arms (the next arm starts its own servers)
    done
    date '+%F %T' > "$B/DONE"
    ev end '"what": "lane B queue done"'
    log "lane B queue done"
}

smoke() {  # CL0: every P7 agent on 3 routes (profiling); the head arms dump every request for the rule-8 check
    local arm
    trap '[[ -n ${arm:-} ]] && { srv_stop_arm "$arm"; kill_runs "$B/cl0/$arm"; }' EXIT
    for arm in ${SMOKE_ARMS:-cl3 cl4 cl6 cl2 cl7 cl8 cl1}; do
        local d=0
        [[ $arm == cl3 || $arm == cl4 || $arm == cl6 ]] && d=1
        [[ -e $B/cl0/$arm/DONE ]] || run_arm "$arm" 0 "$SMOKE_ROUTES" 1.0 "$d" "$B/cl0/$arm"
        srv_stop_arm "$arm"
    done
}

case ${1:-} in
    smoke) trap 'exit 129' HUP INT TERM; smoke ;;
    expert) expert "${2:-0,1}" ;;
    arm) shift; trap 'srv_stop_arm "$1"; [[ -n ${6:-} ]] && kill_runs "$6"' EXIT; trap 'exit 129' HUP INT TERM; run_arm "$@" ;;
    chain) chain ;;
    *) sed -n 2,19p "$0"; exit 1 ;;
esac
