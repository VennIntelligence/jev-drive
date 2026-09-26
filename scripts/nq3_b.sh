#!/usr/bin/env bash
# Night queue 3, lane B: the closed-loop CL queue on Bench2Drive 220 (todos/2026-09-26-night-queue-3.md, section CL and
# "timetable / one-shot scripts"). One chain script, started once in tmux jev:nq3-b; nothing is advanced by hand.
#
#   scripts/nq3_b.sh expert        CL1a: PDM-Lite expert logs on all 220 routes (SimLingo tree, as the acceptance)
#   scripts/nq3_b.sh chain         the whole queue: CL0 gate -> CL1 ... CL10 by priority, CL5 / CL5d inserted when
#                                  runs/nq3/q2/closed_loop_head/READY appears, mc_real0 appended on runs/nq3/q4a/PASS
#   scripts/nq3_b.sh step <name>   one queue step (resumable), for debugging
#
# Resources (fixed by main): GPUs 0,1,2 at <= 6 CARLA servers each, 50 cores (taskset $B_CPUS), --client-threads 8;
# after runs/nq3/a/v1/DONE and once lane A's servers are gone: GPUs 0-5 and 90 cores. A new server waits while the
# cgroup's pids.current > 17000 (B2D_PIDS_WAIT, scripts/b2d_run.py). CARLA server indices 300-479 (below the
# ephemeral port range), 30 per GPU.
# Hand-offs: runs/nq3/b/<step>/DONE, runs/nq3/b/ERROR, runs/nq3/b/STATUS.md (every 10 min), runs/nq3/b/events.jsonl.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
REPO=$(pwd)
NQ=$DATA_DIR/runs/nq3
B=$NQ/b
mkdir -p "$B"
B_CPUS=${B_CPUS:-60-109}
export B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
SIM=$DATA_DIR/third_party/simlingo
P7=$REPO/todos/2026-09-23-tfv6-controller/controller-eval/P7.json

ev() {  # ev <kind> [json fields without braces]
    printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N | cut -c1-14)" "$1" "${2:+, $2}" >> "$B/events.jsonl"
}
log() { echo "$(date '+%F %T') $*" | tee -a "$B/log.txt" >&2; }

expert() {  # CL1a: expert logs, GPUs $1 (comma list), 6 workers per GPU; resumable
    local out=$B/cl1_expert gpus=${1:-0,1} pids=() g k=0
    mkdir -p "$out"
    ev step_start '"step": "cl1_expert"'
    log "cl1_expert on GPUs $gpus"
    for g in ${gpus//,/ }; do
        BENCH2DRIVE_ROOT=$SIM/Bench2Drive WORK_DIR=$SIM taskset -c "$B_CPUS" "$DATA_DIR/envs/carla/bin/python" \
            scripts/b2d_run.py --routes "$SIM/leaderboard/data/bench2drive220.xml" --towns all --workers 6 \
            --server-index $((300 + 30 * g)) --index-span 30 --gpu-rank "$g" --tm-seed 0 --no-spectator --no-reap \
            --client-threads 8 --max-attempts 3 --stall-s 480 --route-timeout-s 3600 --out "$out" \
            --python "$DATA_DIR/envs/simlingo/bin/python" --agent scripts/b2d_expert_agent.py \
            --agent-config "expert+nq3" >> "$out/runner-g$g.log" 2>&1 &
        pids+=($!)
        sleep $((60 + 30 * k)); k=$((k + 1))
    done
    local bad=0
    for p in "${pids[@]}"; do wait "$p" || bad=1; done
    mkdir -p "$out/logs"
    for d in "$out"/done/*.json; do         # the expert log of each route's finished attempt -> logs/<id>.jsonl
        rid=$(basename "$d" .json)
        a=$(python3 -c "import json;print(json.load(open('$d'))['attempt'])")
        cp "$out/attempts/$rid/$a/expert.jsonl" "$out/logs/$rid.jsonl"
    done
    ev step_end "\"step\": \"cl1_expert\", \"logs\": $(ls "$out/logs" | wc -l), \"runner_bad\": $bad"
    log "cl1_expert: $(ls "$out/logs" | wc -l) route logs (runner status $bad)"
}

case ${1:-} in
    expert) expert "${2:-0,1}" ;;
    *) sed -n 2,16p "$0"; exit 1 ;;
esac
