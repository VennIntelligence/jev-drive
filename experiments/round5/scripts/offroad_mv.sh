#!/usr/bin/env bash
# ROUND5 OFFROAD-MV (diagnosis, CPU only, PRIO 15): offroad_mv.py events / data / report through the pool. Box, repo root.
#   offroad_mv.sh launch   events + data in parallel -> report (--after both) -> $O/DONE
#   offroad_mv.sh one events|data|report   a single stage (no chain)
#   offroad_mv.sh wait     blocks until $O/DONE or a failed / cancelled job of this lane (one run_in_background waiter)
#   job bodies: events | data | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round5/offroad_mv; ME=experiments/round5/scripts/offroad_mv.sh; PY=experiments/round5/scripts/offroad_mv.py
TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; PRIO=15
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-offroad --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name-$(date +%H%M%S) "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
sub_events() { sub offmv-events "$@" --vram 0.1 --cpu 4 --ram 24 --timeout-h 1 -- bash $ME events; }
sub_data() { sub offmv-data "$@" --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME data; }
sub_report() { sub offmv-report "$@" --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME report; }
case ${1:-} in
launch)
  rm -f $O/DONE; e=$(sub_events) || exit 1; d=$(sub_data) || exit 1; r=$(sub_report --after $e,$d) || exit 1
  echo "$(date '+%F %T') launch (code $(git rev-parse --short HEAD)): events $e data $d report $r" | tee -a $O/stages.txt ;;
one) rm -f $O/DONE; "sub_$2" ;;
events) $V $PY events || die events ;;
data) $TR $PY data || die data ;;
report) $V $PY report || die report; date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 120; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; true ;;
*) sed -n 2,6p "$0"; exit 2 ;;
esac
