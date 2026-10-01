#!/usr/bin/env bash
# SCH GPU helper launcher: pins a helper to one card and the scheduler's helper cores, records its PID.
# Usage (inside a tmux window, via scripts/tmux_run.sh sch-gpu-<job> ...): launch.sh <job> <card> <cmd...>
# Writes $DATA_DIR/runs/sched/gpu_helpers/<job>/pid.<card>.<n> ("pid pgid card start cmd"); the process is the helper itself.
set -euo pipefail
: "${DATA_DIR:?}"
job=$1 card=$2; shift 2
O=$DATA_DIR/runs/sched/gpu_helpers/$job
mkdir -p "$O"
f=$O/pid.$card.$(date +%H%M%S)
echo "$$ $(ps -o pgid= -p $$ | tr -d ' ') $card $(date '+%F %T') $*" > "$f"
ln -sfn "$(basename "$f")" "$O/pid"
export CUDA_VISIBLE_DEVICES=$card OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
exec taskset -c "${SCH_CPUS:-118-133}" "$@"
