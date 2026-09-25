#!/usr/bin/env bash
# Gated job launcher for the shared box schedule ($DATA_DIR/runs/schedule.md).
# Waits (bash sleep only, no agent polling) until every dependency slot has a .done sentinel and the chosen GPU
# has enough free VRAM, then runs the command, and finally writes <slot>.done or <slot>.failed and a line in
# gpu-plan.md. A dependency that ends in .failed stops the wait and fails this slot too.
#
# Usage (run it inside tmux via scripts/tmux_run.sh):
#   scripts/tmux_run.sh <slot> scripts/slot_run.sh <slot> [--after a,b] [--gpu N --vram-gb G] [--not-before HH:MM] -- cmd args...
# Sentinels live in $DATA_DIR/runs/sched/. Other agents wait on your slot by name, so pick the name in schedule.md.
# A job killed by a signal (rc > 128) also gets <slot>.death-<HHMMSS>.txt there: container memory, largest
# processes and the last 2 min of scripts/boxwatch.sh, which this script starts (once per box).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."

slot=$1; shift
after="" gpu="" vram_gb=0 not_before=""
while [[ $# -gt 0 && $1 != -- ]]; do
    case $1 in
        --after) after=$2; shift 2 ;;
        --gpu) gpu=$2; shift 2 ;;
        --vram-gb) vram_gb=$2; shift 2 ;;
        --not-before) not_before=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
shift  # the --
S=$DATA_DIR/runs/sched
PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
mkdir -p "$S"
rm -f "$S/$slot.done" "$S/$slot.failed"
note() { echo "$(date '+%Y-%m-%d %H:%M') [slot $slot] $*" | tee -a "$PLAN"; }
# The box-wide memory/process sampler (one per box, flock) for SIGKILL forensics; detached so it outlives this slot.
setsid nohup scripts/boxwatch.sh > /dev/null 2>&1 < /dev/null &

post_mortem() {  # rc -> $S/<slot>.death-<HHMMSS>.txt: container memory right after a signal death, largest processes,
    # and the last 2 min of the boxwatch samples (todos/2026-09-25-closed-loop-infra-acceptance/sigkill.md)
    local rc=$1 f=$S/$slot.death-$(date +%H%M%S).txt g; shift
    {
        echo "$(date '+%F %T.%3N') slot $slot: job exited rc=$rc (signal $(( rc - 128 ))); command: $*"
        for g in memory.current memory.high memory.max memory.events pids.current memory.pressure; do
            echo "$g: $(tr '\n' ' ' < /sys/fs/cgroup/$g 2>/dev/null)"
        done
        grep -E '^(anon|file|shmem|file_mapped|slab) ' /sys/fs/cgroup/memory.stat 2>/dev/null
        echo "--- largest processes"
        ps -eo pid,ppid,pgid,user,etimes,rss,nlwp,args --sort=-rss 2>/dev/null | head -16 | cut -c1-240
        echo "--- boxwatch, last 2 min"
        tail -24 "$DATA_DIR/runs/boxwatch/$(date +%Y%m%d).tsv" 2>/dev/null
    } > "$f"
    echo "$f"
}

ready() {
    local d
    # Check every dependency for failure before checking completion, so a failed later dependency is not
    # hidden behind an earlier one that is still running.
    for d in ${after//,/ }; do
        [[ -f $S/$d.failed ]] && { note "dependency $d failed; not starting"; touch "$S/$slot.failed"; exit 1; }
    done
    for d in ${after//,/ }; do
        [[ -f $S/$d.done ]] || return 1
    done
    if [[ -n $not_before ]] && [[ $(date +%H%M) < ${not_before/:/} ]]; then return 1; fi
    if [[ -n $gpu ]] && (( vram_gb > 0 )); then
        local free
        free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        (( free >= vram_gb * 1024 )) || return 1
    fi
}

echo "$(date +%T) waiting: after=[${after}] gpu=${gpu:-any} vram>=${vram_gb}GB not_before=${not_before:-now}"
until ready; do sleep 60; done
note "start (gpu=${gpu:-any}): $*"
[[ -n $gpu ]] && export CUDA_VISIBLE_DEVICES=$gpu
"$@"; rc=$?
pm=""
(( rc > 128 )) && pm=" (killed by signal $(( rc - 128 )); post-mortem $(post_mortem "$rc" "$@"))"
if (( rc == 0 )); then touch "$S/$slot.done"; note "done"; else touch "$S/$slot.failed"; note "FAILED rc=$rc$pm"; fi
exit $rc
