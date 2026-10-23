#!/usr/bin/env bash
# ROUND9 SNAP10K (plans/2026-10-24-snap10k.md): one PAI 281 ACC + GAP read, Harmonizer OFF, of the step-10 000 snapshot PCL1-F-s0x10000 (= RB0L-F-s0x10000),
# with P1L's serve and chunk lists. No training. Box, repo root; every job through the pool, the whole chain in one submit; lane dir $O = $DATA_DIR/runs/round9/snap10k.
#   snap10k.sh all    8 read chunks + tax -> scorer_v2 new -> report -> $O/DONE
#   snap10k.sh wait   one waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: sv2 | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/snap10k; ME=experiments/round9/scripts/snap10k.sh; PY=experiments/round9/scripts/snap10k.py
V=.venv/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; F_=$DATA_DIR/runs/pai_fix/facts
P1L=$DATA_DIR/runs/round9/p1l/pai281/PCL1L-F-s0-ag; TAG=PCL1-F-s0x10000
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = p1l.sh
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-snap10k --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
all)
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in PCL1L-F-s0 PCL1-F-s0 RB0L-F-s0 RB0-F-s0; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f"; done
  C=$DATA_DIR/runs/op_parity/runs/$TAG; for f in ckpt-final.pt adapter.pt route_mem.json; do [[ -s $C/$f ]] || die "snapshot lacks $f"; done
  grep -qx "SERVE=$SERVE" $P1L/config.txt && grep -qx "WIZ=$WIZ" $P1L/config.txt && grep -qx "ALPASIM_FAST=1" $P1L/config.txt && grep -qx "CONC=4" $P1L/config.txt || die "serve differs from $P1L"
  [[ $(ls $P1L/chunks/*.txt | wc -l) == 8 ]] || die "not 8 chunk lists"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  R=$O/pai281/$TAG-ag; [[ ! -e $R ]] || die "$R exists"
  CHUNKS_FROM=$P1L/chunks SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $TAG > $R.submit.txt || die "pai submit"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $TAG-281ag $R | tail -1) && [[ -n $t ]] || die "tax"
  echo "$t pai_fix-tax-$TAG-281ag" >> $O/jobs.txt
  v=$(sub snap10k-sv2 --after $t --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- bash $ME sv2) || exit 1
  r=$(sub snap10k-report --after $v --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): chunks $(echo $ids) tax $t sv2 $v report $r" | tee -a $O/stages.txt ;;
sv2)
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $TAG-ag --workers 16 \
    > $O/sv2.log 2>&1 || die "scorer_v2 new (log $O/sv2.log)" ;;
report)
  $V $PY report --dir $O > $O/results/report.log 2>&1 || die "report (log $O/results/report.log)"; date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -5 $O/stages.txt; true ;;
*) sed -n 2,7p "$0"; exit 2 ;;
esac
