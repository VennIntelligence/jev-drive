#!/usr/bin/env bash
# ROUND6 ROUTE-CURV (diagnosis, CPU only, PRIO 15): rmc.py extract / scenes / rows / report through the pool. Box, repo root.
#   rmc.sh launch       extract + rows in parallel -> scenes (--after extract) -> report (--after scenes, rows) -> $O/DONE
#   rmc.sh smoke        extract -> scenes on 3 test + 3 train scenes, inline-sized (tag smoke): a code check before launch
#   rmc.sh one <stage> [--after ID]  a single stage
#   rmc.sh wait         blocks until $O/DONE or a failed / cancelled job of this lane (one run_in_background waiter)
#   job bodies: extract | scenes [args] | rows | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round6/route_curv; ME=experiments/round6/scripts/rmc.sh; PY=experiments/round6/scripts/rmc.py
V=.venv/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; PRIO=15
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round6-rmc --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name-$(date +%H%M%S) "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
sub_extract() { sub rmc-extract "$@" --vram 0.1 --cpu 8 --ram 48 --timeout-h 1 -- bash $ME extract; }
sub_scenes() { sub rmc-scenes "$@" --vram 0.1 --cpu 32 --ram 120 --timeout-h 3 -- bash $ME scenes --workers 32; }
sub_rows() { sub rmc-rows "$@" --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME rows; }
sub_report() { sub rmc-report "$@" --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME report; }
case ${1:-} in
launch)
  rm -f $O/DONE; e=$(sub_extract) || exit 1; w=$(sub_rows) || exit 1; s=$(sub_scenes --after $e) || exit 1; r=$(sub_report --after $s,$w) || exit 1
  echo "$(date '+%F %T') launch (code $(git rev-parse --short HEAD)): extract $e rows $w scenes $s report $r" | tee -a $O/stages.txt ;;
smoke)
  e=$(sub_extract) || exit 1
  s=$(sub rmc-smoke --after $e --vram 0.1 --cpu 6 --ram 32 --timeout-h 1 -- bash $ME scenes --workers 6 --limit 3 --tag smoke) || exit 1
  echo "$(date '+%F %T') smoke (code $(git rev-parse --short HEAD)): extract $e scenes $s" | tee -a $O/stages.txt ;;
one) rm -f $O/DONE; "sub_$2" "${@:3}" ;;
extract) $V $PY extract --workers 8 || die extract ;;
scenes) shift; "${EA[@]}" $AP $PY scenes "$@" || die scenes ;;
rows) $V $PY rows || die rows ;;
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
*) sed -n 2,7p "$0"; exit 2 ;;
esac
