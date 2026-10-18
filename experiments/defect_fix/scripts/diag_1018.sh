#!/usr/bin/env bash
# Route-memory reliance of JTN-L18-s0 vs RSG-L18-s0 in closed loop (SH30_RMEM=0 reads, results/jtb/diag_1018.md). Box, repo root; pool CPU job body.
# Memory-off reads: route_mem/read/<tag>-nomem (RSG: stored 10-13 read; JTN: rm_chain.sh submit-read with READ_ENV=SH30_RMEM=0 READ_SFX=nomem).
# Output $O: nomem/ (rf_nomem.py: held 460 zeros / mean / offroad, memory on vs off) and follow/ (rf_follow.py on the 24 held turns, on and off).
set -uo pipefail; cd "$(dirname "$0")/../../.."
D=${DATA_DIR:-/root/autodl-tmp/ujs}; M=$D/runs/route_mem/read; O=$D/runs/defect_fix/diag_1018; AP=$D/third_party/alpasim/.venv/bin/python; S=experiments/route_follow/scripts
SC=$D/runs/defect_fix/jt/turns24/scenes24.txt
dirs() { ls -d $M/$1/c*/run | paste -sd,; }
mkdir -p $O; rm -f $O/DONE $O/ERROR
{ $AP $S/rf_nomem.py --runs RSG-on=$(dirs RSG-L18-s0) RSG-off=$(dirs RSG-L18-s0-nomem) JTN-on=$(dirs JTN-L18-s0) JTN-off=$(dirs JTN-L18-s0-nomem) \
    --pairs RSG-off:RSG-on JTN-off:JTN-on JTN-on:RSG-on JTN-off:RSG-off --did JTN-off:JTN-on,RSG-off:RSG-on --out $O/nomem > $O/nomem.log 2>&1 \
  && $AP $S/rf_follow.py --runs RSG-on@18=$(dirs RSG-L18-s0) RSG-off@18=$(dirs RSG-L18-s0-nomem) JTN-on@18=$(dirs JTN-L18-s0) JTN-off@18=$(dirs JTN-L18-s0-nomem) \
    --scenes $SC --cover none --out $O/follow --workers 16 --pairs RSG-off@18:RSG-on@18 JTN-off@18:JTN-on@18 JTN-off@18:RSG-off@18 > $O/follow.log 2>&1; } \
  && date '+%F %T' > $O/DONE || { tail -20 $O/nomem.log $O/follow.log > $O/ERROR 2>&1; exit 1; }
