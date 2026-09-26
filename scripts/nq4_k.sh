#!/usr/bin/env bash
# Night queue 4, K-prep chain (todos/2026-09-26-night-queue-4.md, section K and its [K] entries). One chain, started in
# tmux jev:nq4-k; every step writes runs/nq4/k/steps/<step>/DONE and is skipped when that exists (resumable).
#
#   scripts/nq4_k.sh [all]     lead -> labels -> fit -> ready (the closed-loop rule-8 step `cl` is run by hand:
#                              scripts/nq4_k.sh cl <gpu>, it needs a borrowed card)
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
lead() {
    local d=$1 est=$2 pids=() s set
    for set in carla_p5v1_ba carla_p6; do
        wait_gpu 6000
        for s in 0 1; do
            P5_SET=$set CUDA_VISIBLE_DEVICES=$GPU setsid taskset -c "$K_CPUS" "$PY_OP" scripts/p5_openpilot.py --models cinque \
                --arrays temporal lead lead_prob --out-sub op_streams_lead --shard $s/2 --workers 2 >> "$d/log-$set-$s.txt" 2>&1 &
            pids+=($!); echo "${pids[*]}" > "$d/pids"
        done
        for s in "${pids[@]}"; do guard "$s" "$est" "$d/log-$set-0.txt" || return 1; done
        pids=()
        P5_SET=$set taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k check-lead --set "$set" >> "$d/log.txt" 2>&1 || return 1
        P5_SET=$set taskset -c "$K_CPUS" "$PY" -m jevdrive.p5_openpilot finalize --arrays temporal,lead,lead_prob \
            --sub op_streams_lead --models cinque >> "$d/log.txt" 2>&1 || return 1
    done
}
labels() { taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k labels >> "$1/log.txt" 2>&1; }
fit() { wait_gpu 12000; CUDA_VISIBLE_DEVICES=$GPU taskset -c "$K_CPUS" "$PY" -m jevdrive.nq4_k fit >> "$1/log.txt" 2>&1; }

case ${1:-all} in
    lead) step lead 1.0 lead ;;
    labels) step labels 0.2 labels ;;
    fit) step fit 0.5 fit ;;
    all) trap 'exit 129' HUP INT TERM
         step lead 1.0 lead; step labels 0.2 labels; step fit 0.5 fit
         status done "lead, labels, fit done; closed-loop rule-8 step: scripts/nq4_k.sh cl <gpu>" ;;
    *) sed -n 2,11p "$0"; exit 1 ;;
esac
