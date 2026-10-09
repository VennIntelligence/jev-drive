#!/usr/bin/env bash
# Lane CF1 (plans/2026-10-09-cf1-confirm-prereg.md): P2H10 / YR10m10 / APY10m10 on the part012-015 scenes (cf1fresh) and APY10m10 on OT3's 400 later
# scenes (ot3new), as GPU-pool jobs through ot2_loop.py (OT_LANE=cf1). One self-advancing chain:
#   scripts/tmux_run.sh cf1 bash experiments/alpasim/scripts/cf1_chain.sh        (box)
# State: $DATA_DIR/runs/alpasim/cf1/{STATUS, DONE, ERROR, log.txt}; the loop's own state is under cf1/a/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
RA=$DATA_DIR/runs/alpasim; O=$RA/cf1; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
VPY=$PWD/.venv/bin/python
status() { echo "$(date '+%F %T') cf1: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD); waiting for room (other lanes' running simulator stacks <= 3)"
# Box total of simulator stacks must stay <= 6 and this lane runs 3: wait until at most 3 stacks of other lanes run.
until [[ $($VPY -m jevdrive.cl queue 2>/dev/null | awk '$2 == "running" && $4 ~ /^alpasim-/ && $4 !~ /^alpasim-cf1/' | wc -l) -le 3 ]]; do sleep 60; done
status "loop starts, code $(git rev-parse --short HEAD)"
J=()
for s in 0 1; do J+=("APY10m10-AB-s$s@new:ap2:AP2_TAG=APY10m10-AB-s$s:APY10m10-AB-s$s:ot3new"); done
for r in P2H10-F YR10m10-F; do for s in 0 1; do J+=("$r-s$s@fresh:sh30:SH30_TAG=$r-s$s:$r-s$s:cf1fresh"); done; done
for s in 0 1; do J+=("APY10m10-AB-s$s@fresh:ap2:AP2_TAG=APY10m10-AB-s$s:APY10m10-AB-s$s:cf1fresh"); done
OT_LANE=cf1 OT_PRIO=13 OT2_MAX_ACTIVE=3 python3 experiments/alpasim/scripts/ot2_loop.py a "${J[@]}" || die "ot2_loop rc $? (see $RA/cf1/a/ERROR)"
status "loop done; report"
$VPY experiments/alpasim/scripts/cf1_report.py --out "$O/results" > "$O/report.log" 2>&1 || die "report failed (see $O/report.log)"
date '+%F %T' > "$O/DONE"; status "all done"
