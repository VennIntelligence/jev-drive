#!/usr/bin/env bash
# Night queue 4, K-prep chain (todos/2026-09-26-night-queue-4.md, section K and its [K] entries). One chain, started in
# tmux jev:nq4-k; every step writes runs/nq4/k/steps/<step>/DONE and is skipped when that exists (resumable).
#
#   scripts/nq4_k.sh [all]     lead_ba -> labels -> fit -> check_eigh -> lead_p6 -> export_p6 -> cl -> ready
#                              (cl borrows the card with the fewest CARLA servers, K_CL_GPU overrides; 2 servers)
#   scripts/nq4_k.sh <step>    one step
#
# Resources (fixed by the [K] 17:30 entry): GPU 6 (shared; each GPU step waits for room first), cores $K_CPUS,
# BLAS / OMP threads 4. Only PIDs this script started are ever killed. Hand-offs: runs/nq4/k/STATUS.md, ERROR, READY.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
K=$DATA_DIR/runs/nq4/k
mkdir -p "$K/steps"
K_CPUS=${K_CPUS:-146-149}
GPU=${K_GPU:-6}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=$(pwd)/.venv/bin/python PY_OP=$DATA_DIR/envs/openpilot/bin/python

log() { echo "$(date '+%F %T') $*" | tee -a "$K/log.txt" >&2; }
ev() { printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N | cut -c1-14)" "$1" "${2:+, $2}" >> "$K/events.jsonl"; }
status() { printf '# K-prep status %s\n\n- step: %s\n- %s\n' "$(date '+%F %T %Z')" "$1" "$2" > "$K/STATUS.md"; }
error() {
    { echo "# K-prep ERROR $(date '+%F %T %Z')"; echo; echo "reason: $1"; echo
      [[ -n ${2:-} && -f $2 ]] && { echo '```'; tail -50 "$2"; echo '```'; }; } > "$K/ERROR"
    ev error "\"reason\": \"$1\""; log "ERROR: $1"; exit 1
}
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' '; }
wait_gpu() {  # wait_gpu <MB>: GPU $GPU is shared with night-queue-3 C / D, so wait for room instead of OOMing
    local t0=$SECONDS
    until (( $(free_mb) >= $1 )); do
        (( SECONDS - t0 > 3 * 3600 )) && error "GPU $GPU had < $1 MB free for 3 h"
        status "$STEP" "waiting for $1 MB on GPU $GPU (free $(free_mb) MB)"; sleep 60
    done
}
step() {  # step <name> <estimate h> <function>: run once, time it, enforce 2x the estimate
    STEP=$1; local est=$2 fn=$3 d=$K/steps/$1
    [[ -e $d/DONE ]] && return 0
    mkdir -p "$d"; local t0=$SECONDS
    log "start $1 (estimate $est h)"; ev step_start "\"step\": \"$1\", \"est_h\": $est"; status "$1" "running since $(date '+%T')"
    "$fn" "$d" "$est" || error "step $1 failed" "$d/log.txt"
    local h; h=$(python3 -c "print(round(($SECONDS - $t0) / 3600, 3))")
    printf '{"step": "%s", "wall_h": %s, "finished": "%s"}\n' "$1" "$h" "$(date '+%F %T')" > "$d/DONE"
    ev step_end "\"step\": \"$1\", \"wall_h\": $h"; log "done $1 in $h h"
}
guard() {  # guard <pid> <estimate h> <log>: wait for a background job, kill it (by PID) past twice the estimate
    local p=$1 lim; lim=$(python3 -c "print(int(2 * $2 * 3600))"); local t0=$SECONDS
    while kill -0 "$p" 2>/dev/null; do
        (( SECONDS - t0 > lim )) && { kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; error "$STEP exceeded twice its estimate" "$3"; }
        sleep 20
    done
    wait "$p"
}

# ---------------------------------------------------------------- K2: Cinque lead outputs on P5 v1 BA and P6 v0
lead_set() {  # lead_set <dir> <estimate h> <set>: 4 single-model processes (GPU 6 is time-sliced among ~10 processes;
    # each openpilot process spin-waits at 100% of one core, so 4 processes = the 4 cores and 4 shares of the card)
    local d=$1 est=$2 set=$3 pids=() s nsh p t0=$SECONDS
    for p in $(cat "$d/pids" 2>/dev/null); do     # processes of an earlier chain on this step: let them finish first
        while kill -0 "$p" 2>/dev/null; do (( SECONDS - t0 > 2 * 3600 )) && return 1; status "$STEP" "waiting for earlier PID $p"; sleep 20; done
    done
    wait_gpu 3600
    nsh=$(( ($(free_mb) - 1000) / 2600 )); (( nsh > 4 )) && nsh=4; (( nsh < 1 )) && nsh=1   # ~2.6 GB per process
    log "lead $set: $nsh processes (GPU $GPU free $(free_mb) MB)"
    for (( s = 0; s < nsh; s++ )); do
        P5_SET=$set CUDA_VISIBLE_DEVICES=$GPU setsid taskset -c "$K_CPUS" "$PY_OP" scripts/p5_openpilot.py --models cinque \
            --arrays temporal lead lead_prob --out-sub op_streams_lead --shard $s/$nsh --workers 1 >> "$d/log-$set-$s.txt" 2>&1 &
        pids+=($!); echo "${pids[*]}" > "$d/pids"
    done
    for s in "${pids[@]}"; do guard "$s" "$est" "$d/log-$set-0.txt" || return 1; done
    P5_SET=$set taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k check-lead --set "$set" >> "$d/log.txt" 2>&1 || return 1
    P5_SET=$set taskset -c "$K_CPUS" "$PY" -m jevdrive.p5_openpilot finalize --arrays temporal,lead,lead_prob \
        --sub op_streams_lead --models cinque >> "$d/log.txt" 2>&1
}
lead_ba() { lead_set "$1" "$2" carla_p5v1_ba; }
lead_p6() { lead_set "$1" "$2" carla_p6; }
labels() { taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k labels >> "$1/log.txt" 2>&1; }
fit() { wait_gpu 12000; CUDA_VISIBLE_DEVICES=$GPU taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k fit >> "$1/log.txt" 2>&1; }
export_p6() { taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k export-p6 >> "$1/log.txt" 2>&1; }
check_eigh() { wait_gpu 12000; CUDA_VISIBLE_DEVICES=$GPU taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k check-eigh >> "$1/log.txt" 2>&1; }
ready() { "$PY" -m jevdrive.nq4_k ready >> "$1/log.txt" 2>&1; }

# ---------------------------------------------------------------- K5: closed-loop rule-8 check (a borrowed card, 2 servers)
XML=$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml
P7=$(pwd)/todos/2026-09-23-tfv6-controller/controller-eval/P7.json
PY_CARLA=$DATA_DIR/envs/carla/bin/python PY_SCOUT=$DATA_DIR/envs/scout-tfv6/bin/python
CL_ROUTES=${K_CL_ROUTES:-2416,3540,17752}      # [K] 17:30 (8): stop sign, HardBreakRoute, DynamicObjectCrossing
K_CL_WORKERS=${K_CL_WORKERS:-1}            # [K] 18:37: 2 servers on 4 cores starved the render thread (UE4 "GameThread timed out waiting for RenderThread")
K_CL_INDEX=${K_CL_INDEX:-170}                   # [K] 17:58: 170-179 (i, i + 120, i - 120 all unused; 490-499 collided with A's RPC)
export B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000} B2D_SENSOR_TICK=1
emptiest_gpu() {  # the card among 0-5 with the fewest CARLA servers (ties: the lowest index)
    local g best=0 n bn=999
    for g in 0 1 2 3 4 5; do
        n=$(nvidia-smi -i "$g" --query-compute-apps=process_name --format=csv,noheader | grep -c CarlaUE4)
        (( n < bn )) && { bn=$n; best=$g; }
    done
    echo "$best"
}
cl() {
    local d=$1 est=$2 g=${K_CL_GPU:-$(emptiest_gpu)} arm out cfg srv=$1/srv pids=()
    local i port busy=""
    for (( i = K_CL_INDEX; i < K_CL_INDEX + 10; i++ )); do   # RPC 2000 + 50 i (+1, +2) and TM 8000 + 50 i must be free
        for port in $((2000 + 50 * i)) $((2001 + 50 * i)) $((2002 + 50 * i)) $((8000 + 50 * i)); do
            ss -ltn "( sport = :$port )" | grep -q LISTEN && busy+=" $port"
        done
    done
    [[ -n $busy ]] && { log "cl: ports in use:$busy"; return 1; }
    mkdir -p "$srv"; log "cl: GPU $g ($(nvidia-smi -i "$g" --query-compute-apps=process_name --format=csv,noheader | grep -c CarlaUE4) CARLA servers there), cores $K_CPUS"
    echo "$g" > "$d/gpu"
    ( CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 setsid taskset -c "$K_CPUS" "$PY_OP" scripts/nq3_cl_server.py --pool "$K_CL_WORKERS" \
          --socket "$srv/head.sock" --ready-file "$srv/head.ready" >> "$srv/head.log" 2>&1 & echo $! > "$srv/head.pid"; wait ) &
    local t0=$SECONDS
    until [[ -e $srv/head.ready ]]; do (( SECONDS - t0 > 900 )) && return 1; sleep 5; done
    for arm in k0 k1 k2 k3; do
        out=$d/$arm; mkdir -p "$out"; cfg=$d/$arm.json
        printf '{"model": "head", "warmup_s": 5.0, "desire": true, "head_cam_tick": 0.0, "arm": "%s", "k_view": "unseen", "k_split": "%s", "socket": "%s", "controller": "fixed", "controller_preset": "pursuit", "controller_config": "%s", "seed": 0, "dump_every": 1}\n' \
            "$arm" "$K/route_split.json" "$srv/head.sock" "$P7" > "$cfg"
        [[ -e $out/DONE ]] && continue
        taskset -c "$K_CPUS" "$PY_CARLA" scripts/b2d_run.py --routes "$XML" --route-ids "$CL_ROUTES" --workers "$K_CL_WORKERS" \
            --server-index "$K_CL_INDEX" --index-span 10 --gpu-rank "$g" --tm-seed 0 --no-spectator --no-reap --client-threads 8 \
            --max-attempts 2 --stall-s 480 --route-timeout-s 3600 --out "$out" --python "$PY_SCOUT" \
            --agent scripts/b2d_zeroshot_agent.py --agent-config "$cfg" --fast-copy --cache-lights >> "$out/runner.log" 2>&1 &
        echo $! > "$out/runner.pid"
        guard "$!" "$est" "$out/runner.log" || return 1
        date '+%F %T' > "$out/DONE"
    done
    local p; p=$(cat "$srv/head.pid"); kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null
    local adirs=() r a
    for arm in k0 k1 k2 k3; do
        for r in "$d/$arm"/done/*.json; do
            a=$(python3 -c "import json;print(json.load(open('$r'))['attempt'])"); adirs+=("$d/$arm/attempts/$(basename "$r" .json)/$a")
        done
    done
    CUDA_VISIBLE_DEVICES=$g taskset -c "$K_CPUS" "$PY_OP" scripts/nq3_cl_check.py op "${adirs[@]}" --out "$d/check_op.json" >> "$d/log.txt" 2>&1 || return 1
    "$PY" -m jevdrive.nq4_k cl-verdict --dir "$d" >> "$d/log.txt" 2>&1
}

case ${1:-all} in
    lead) step lead_ba 1.5 lead_ba; step lead_p6 1.0 lead_p6 ;;
    labels) step labels 0.2 labels ;;
    fit) step fit 0.5 fit ;;
    labels-fit) step labels 0.2 labels; step fit 0.5 fit ;;
    cl) step cl 1.5 cl ;;
    check-eigh) step check_eigh 0.5 check_eigh ;;
    ready) step ready 0.05 ready ;;
    all) trap 'exit 129' HUP INT TERM
         step lead_ba 1.5 lead_ba; step labels 0.2 labels; step fit 0.5 fit; step check_eigh 0.5 check_eigh
         step lead_p6 1.0 lead_p6; step export_p6 0.1 export_p6; step cl 1.5 cl; step ready 0.05 ready
         status done "lead, labels, fit done; closed-loop rule-8 step: scripts/nq4_k.sh cl <gpu>" ;;
    *) sed -n 2,11p "$0"; exit 1 ;;
esac
