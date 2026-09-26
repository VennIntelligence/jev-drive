#!/usr/bin/env bash
# SCH GPU helper: one-off cutover of lane C's Qwen feats (approved by SCH 2026-09-26 18:50 CST).
# 1. wait for features/c003/meta.json (the chunk lane C's process 801105 is on);
# 2. re-check both PIDs' cmdlines, SIGSTOP the `retry 200` subshell 801102 (so it can neither relaunch nor return),
#    SIGTERM 801105;
# 3. wait for the staged chunks (features_sch/c004..c008) and install them (nq3_feats_qwen_chunks.py install checks
#    rows and completeness);
# 4. SIGCONT 801102: retry relaunches nq3_feats qwen after 120 s, which finds no chunk left -> feats/OK.
set -uo pipefail
: "${DATA_DIR:?}"
RETRY=${1:-801102} OWNER=${2:-801105}
repo=$(cd "$(dirname "$0")/../.." && pwd)
P6=$DATA_DIR/processed/carla_p6
R=$DATA_DIR/runs/nq3/c
O=$DATA_DIR/runs/sched/gpu_helpers/feats
log() { echo "$(date '+%F %T') [cutover] $*" | tee -a "$O/cutover.log"; }
cmd() { tr '\0' ' ' < "/proc/$1/cmdline" 2> /dev/null; }
log "waiting for $P6/features/c003/meta.json"
until [[ -f $P6/features/c003/meta.json ]]; do
  [[ -f $R/feats/OK || -f $R/feats/FAIL ]] && { log "feats ended on its own: abort"; exit 1; }
  sleep 5
done
c1=$(cmd "$RETRY") c2=$(cmd "$OWNER")
ppid=$(awk '{print $4}' "/proc/$OWNER/stat" 2> /dev/null)
log "retry $RETRY: $c1"; log "owner $OWNER (ppid $ppid): $c2"
[[ $c1 == bash* && $c2 == *"jevdrive.nq3_feats qwen --shard 0/1"* && $ppid == "$RETRY" ]] || { log "PID check failed: abort"; exit 1; }
kill -STOP "$RETRY" && log "kill -STOP $RETRY done"
kill -TERM "$OWNER" && log "kill -TERM $OWNER done"
until [[ $(awk '{print $3}' "/proc/$OWNER/stat" 2> /dev/null) != [RSD] ]]; do sleep 1; done
log "$OWNER gone (state $(awk '{print $3}' "/proc/$OWNER/stat" 2> /dev/null || echo reaped))"
log "waiting for the staged chunks"
until [[ -f $P6/features_sch/c004/meta.json && -f $P6/features_sch/c005/meta.json && -f $P6/features_sch/c006/meta.json \
         && -f $P6/features_sch/c007/meta.json && -f $P6/features_sch/c008/meta.json ]]; do sleep 10; done
if "$repo/.venv/bin/python" "$repo/scripts/sch_gpu_helpers/nq3_feats_qwen_chunks.py" install --owner-pid "$OWNER" 2>&1 | tee -a "$O/cutover.log"; then
  log "install ok"
else
  log "install FAILED: $RETRY stays stopped, needs a human"; exit 1
fi
kill -CONT "$RETRY" && log "kill -CONT $RETRY done"
until [[ -f $R/feats/OK || -f $R/feats/FAIL ]]; do sleep 10; done
[[ -f $R/feats/OK ]] && log "feats/OK written" || log "feats/FAIL written"
tail -n 5 "$R/feats/qwen0.log" | tee -a "$O/cutover.log"
