#!/usr/bin/env bash
# Lane nuplan1491: P2H10S-F seeds 0-3 on the 791 public scenes the BODY1 closed-loop read did not cover (cf1fresh 391 + ot3new 400),
# unchanged nuPlan `sh30` driver (SH30_TAG=<tag>, no serving switch), same lists and run script as cf1_chain.sh. 8 independent pool jobs
# (4 seeds x 2 lists) through ot2_loop.py (OT_LANE=nuplan1491). One self-advancing chain:
#   scripts/tmux_run.sh nuplan1491 bash experiments/alpasim/scripts/nuplan1491_chain.sh        (box)
# State: $DATA_DIR/runs/alpasim/nuplan1491/{STATUS, DONE, ERROR, log.txt}; the loop's own state is under nuplan1491/a/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
RA=$DATA_DIR/runs/alpasim; O=$RA/nuplan1491; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
VPY=$PWD/.venv/bin/python
status() { echo "$(date '+%F %T') nuplan1491: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD); loop starts"
J=()
for s in 0 1 2 3; do for l in cf1fresh ot3new; do J+=("P2H10S-F-s$s@$l:sh30:SH30_TAG=P2H10S-F-s$s:P2H10S-F-s$s:$l"); done; done
OT_LANE=nuplan1491 OT_PRIO=13 OT2_MAX_ACTIVE=6 python3 experiments/alpasim/scripts/ot2_loop.py a "${J[@]}" || die "ot2_loop rc $? (see $RA/nuplan1491/a/ERROR)"
status "all done"
date '+%F %T' > "$O/DONE"
