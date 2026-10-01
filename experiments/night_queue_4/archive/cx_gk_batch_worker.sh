#!/usr/bin/env bash
# Reuse the registered evaluator/functions without altering the running pilot shell.
set -uo pipefail
cd "$(dirname "$0")/../../.."
cand=$1 seed=$2 gpu=$3 workers=$4 index=$5 span=$6 cpus=$7 runtime=$8
python3 experiments/night_queue_4/archive/cx_gk_batch.py gate "$cand" || exit 3
export ARMS=arms CX_GK_PILOT_GUARD=0 B2D_NQ4_REUSE_WORLD=0 B2D_PIDS_WAIT=16000
source <(sed '/^case ${1:-}/,$d' experiments/night_queue_4/archive/nq4_gk.sh)
CANONICAL=$DATA_DIR/runs/nq4/gk
G=$runtime
GPUS=$gpu WORKERS=$workers SIDX0=$index SPAN=$span CPUS=$cpus CTX=batch
ERRP=$G/ERROR
mkdir -p "$G/srv" "$G/cfg" "$G/steps"
arm_dir() { echo "$CANONICAL/arms_k/$1/s$3"; }
# Isolate each worker's cleanup even when later work shares canonical outputs.
eval "$(declare -f launch | sed '1s/launch/original_launch/')"
launch() {
    local pid; pid=$(original_launch "$@")
    echo "$6/runner-$pid.owned.json" >> "$G/owned-runners"
    echo "$pid"
}
kill_runs() {
    local record
    [[ -f $G/owned-runners ]] || return 0
    while read -r record; do python3 experiments/night_queue_4/archive/cx_owned_process.py stop "$record" || return 1; done < "$G/owned-runners"
}
cleanup() { kill_runs; srv_stop_gpus "$GPUS"; }
trap cleanup EXIT
trap 'exit 129' HUP INT TERM
ids=$(step_ids "$cand" k "$seed" k220) || exit 4
nw=$(n_ids "$ids")
est=$(python3 -c "print(max(0.3, round($nw * 10 / 60 / $workers * 1.3, 2)))")
execute "$cand" k "$est" "$workers" "$seed=$ids"
rc=$?
srv_stop_gpus "$GPUS"
exit "$rc"
