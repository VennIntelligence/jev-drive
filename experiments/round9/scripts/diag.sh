#!/usr/bin/env bash
# ROUND9 DIAG-READS (plans/2026-10-24-diag-reads.md): two PAI 281 ACC + GAP read lanes, no training.
#   PCL1L-F-s0 with the Harmonizer ON (P1L's stored OFF read's serve and chunk lists + HARMONIZER=1, as the agH reads)
#   INTP75-NR re-read, Harmonizer OFF, identical serve and chunk lists to its stored read (same-checkpoint noise floor)
# Box, repo root; every job through the pool, the whole chain in one submit; lane dir $O = $DATA_DIR/runs/round9/diag_reads.
#   diag.sh all    8 read chunks + tax per lane -> harmonizer-loaded check -> scorer_v2 new -> report -> $O/DONE
#   diag.sh wait   one waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: sv2 | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/diag_reads; ME=experiments/round9/scripts/diag.sh; PY=experiments/round9/scripts/diag.py
V=.venv/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; F_=$DATA_DIR/runs/pai_fix/facts
P1L=$DATA_DIR/runs/round9/p1l/pai281/PCL1L-F-s0-ag; INT=$DATA_DIR/runs/round9/interp/pai281/INTP75-NR-ag
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-diag --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <suffix> <chunks dir> <serve> <facts name> <pai_box extra env...> -> "<chunk ids...> <tax id>"
  local tag=$1 sfx=$2 ch=$3 sv=$4 fn=$5; shift 5
  local R=$O/pai281/$tag-$sfx ids t; [[ ! -e $R ]] || die "$R exists"
  env "$@" CHUNKS_FROM=$ch SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $tag > $R.submit.txt || die "pai submit $tag"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$tag: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $fn $R | tail -1) && [[ -n $t ]] || die "tax $tag"
  echo "$t pai_fix-tax-$fn" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
all)
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in PCL1L-F-s0-281ag NTV0-L18-s0-281ag NTV0-L18-s0-281agH R8K-L18-s2-281ag R8K-L18-s2-281agH INTP75-NR-281ag; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  grep -qx "SERVE=$SERVE" $P1L/config.txt && grep -qx "WIZ=$WIZ" $P1L/config.txt && grep -qx "ALPASIM_FAST=1" $P1L/config.txt && grep -qx "CONC=4" $P1L/config.txt || die "serve differs from $P1L"
  grep -qx "SERVE=$SERVE" $INT/config.txt && grep -qx "WIZ=$WIZ" $INT/config.txt && grep -qx "ALPASIM_FAST=1" $INT/config.txt && grep -qx "CONC=4" $INT/config.txt || die "serve differs from $INT"
  [[ -s $DATA_DIR/datasets/nurec/harmonizer/harmonizer_nontemporal.pt ]] || die "no harmonizer weights"
  [[ -d $DATA_DIR/runs/op_parity/runs/INTP75-NR ]] || die "no INTP75-NR checkpoint"
  [[ $(ls $P1L/chunks/*.txt | wc -l) == 8 && $(ls $INT/chunks/*.txt | wc -l) == 8 ]] || die "not 8 chunk lists"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  a=($(read281 PCL1L-F-s0 agH $P1L/chunks "$SERVE HARMONIZER=1" PCL1L-F-s0-281agH VRAM=30 RAM=40)); [[ ${#a[@]} == 9 ]] || die "read281 P1L ON"
  b=($(read281 INTP75-NR ag-rr $INT/chunks "$SERVE" INTP75-NR-281ag-rr)); [[ ${#b[@]} == 9 ]] || die "read281 INTP75 rr"
  v=$(sub diag-sv2 --after ${a[-1]},${b[-1]} --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- bash $ME sv2) || exit 1
  r=$(sub diag-report --after $v --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): P1L-ON ${a[*]}; INTP75-rr ${b[*]}; sv2 $v report $r" | tee -a $O/stages.txt ;;
sv2)
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads PCL1L-F-s0-agH,INTP75-NR-ag-rr --workers 16 \
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
*) sed -n 2,9p "$0"; exit 2 ;;
esac
