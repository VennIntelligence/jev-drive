#!/usr/bin/env bash
# Gated job launcher for the shared box schedule ($DATA_DIR/runs/schedule.md).
# Waits (bash sleep only, no agent polling) until every dependency slot has a .done sentinel and the chosen GPU
# has enough free VRAM, then runs the command, and finally writes <slot>.done or <slot>.failed and a line in
# gpu-plan.md. A dependency that ends in .failed stops the wait and fails this slot too.
#
# Usage (run it inside tmux via scripts/tmux_run.sh):
#   scripts/tmux_run.sh <slot> scripts/slot_run.sh <slot> [--after a,b] [--gpu N --vram-gb G] [--not-before HH:MM] -- cmd args...
# Sentinels live in $DATA_DIR/runs/sched/. Other agents wait on your slot by name, so pick the name in schedule.md.
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

ready() {
    local d
    for d in ${after//,/ }; do
        [[ -f $S/$d.failed ]] && { note "dependency $d failed; not starting"; touch "$S/$slot.failed"; exit 1; }
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
if (( rc == 0 )); then touch "$S/$slot.done"; note "done"; else touch "$S/$slot.failed"; note "FAILED rc=$rc"; fi
exit $rc
