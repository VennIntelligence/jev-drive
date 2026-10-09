#!/usr/bin/env bash
# CF1 amendment 1: AP2H10-AB-s0/s1 on the 791 fresh scenes (cf1fresh, ot3new), pool jobs through ot2_loop.py (OT_LANE=cf1, stage b), then the report.
#   scripts/tmux_run.sh cf1b bash experiments/alpasim/scripts/cf1b_chain.sh        (box)
set -uo pipefail
cd "$(dirname "$0")/../../.."
RA=$DATA_DIR/runs/alpasim; O=$RA/cf1/b0; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
VPY=$PWD/.venv/bin/python
status() { echo "$(date '+%F %T') cf1b: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
H=$(git rev-parse --short HEAD); status "checkout $H"
[[ -z $(git diff c9d2e07b HEAD --stat -- experiments/alpasim/lib experiments/alpasim/scripts/run.sh experiments/alpasim/scripts/drivers jevdrive) ]] || die "driver code differs from c9d2e07b"
J=()
for s in 0 1; do J+=("AP2H10-AB-s$s@new:ap2:AP2_TAG=AP2H10-AB-s$s:AP2H10-AB-s$s:ot3new" "AP2H10-AB-s$s@fresh:ap2:AP2_TAG=AP2H10-AB-s$s:AP2H10-AB-s$s:cf1fresh"); done
OT_LANE=cf1 OT_PRIO=13 OT2_MAX_ACTIVE=3 python3 experiments/alpasim/scripts/ot2_loop.py b "${J[@]}" || die "ot2_loop rc $?"
for d in $(ls -d $RA/cf1/b/runs/*/*/ 2>/dev/null); do [[ -f $d/checkout.txt ]] || echo "$H" > "$d/checkout.txt"; done
for d in $(ls -d $RA/cf1/a/runs/*/*/ 2>/dev/null); do [[ -f $d/checkout.txt ]] || echo "c9d2e07b" > "$d/checkout.txt"; done
status "loop done; report"
$VPY experiments/alpasim/scripts/cf1_report.py --out "$RA/cf1/results" > "$O/report.log" 2>&1 || die "report failed"
date '+%F %T' > "$O/DONE"; status "all done"
