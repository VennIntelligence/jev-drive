#!/usr/bin/env bash
# BODY1 Amendment 7: run one command on the lane's whole-held card (outside the pool, as rp_direct.py does for closed-loop stacks).
#   scripts/tmux_run.sh body1-r-<name> bash experiments/body1/scripts/route_run.sh <name> <cores> <command> [args ...]
# <cores>: a taskset list no pool job holds (the pool fills from core 0 upward; this lane uses the top of the range).
# State: $DATA_DIR/runs/body1/route/<name>/{log.txt, DONE | ERROR}. Refuses to start when the memory margin under the platform's kill line
# (cl top) is below $MARGIN GiB after a page-cache trim: jobs outside the pool are not covered by its memory guard.
set -uo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; cores=${2:?cores}; shift 2
O=$DATA_DIR/runs/body1/route/$name; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
VPY=$PWD/.venv/bin/python
$VPY -m jevdrive.cl trim --margin 150 >/dev/null 2>&1
m=$($VPY -m jevdrive.cl top 2>/dev/null | sed -n 's/.*margin \([0-9]*\).*/\1/p' | head -1)
if (( ${m:-0} < ${MARGIN:-90} )); then echo "memory margin ${m:-?} GiB < ${MARGIN:-90}: not started" | tee "$O/ERROR"; exit 3; fi
{ echo "==> $(date '+%F %T') card ${RP_CARD:-2} cores $cores margin $m"; echo "\$ $*"; } >> "$O/log.txt"
CUDA_VISIBLE_DEVICES=${RP_CARD:-2} taskset -c "$cores" "$@" >> "$O/log.txt" 2>&1
rc=$?
if (( rc == 0 )); then date '+%F %T' > "$O/DONE"; else echo "rc $rc" > "$O/ERROR"; fi
echo "==> $(date '+%F %T') rc $rc" >> "$O/log.txt"
exit $rc
