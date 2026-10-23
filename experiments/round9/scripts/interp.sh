#!/usr/bin/env bash
# ROUND9 INTERP (plans/2026-10-24-interp.md): theta = (1 - a) NTV0-L18-s0 + a R8K-L18-s2, a = 0.5 / 0.75, read on PAI 281 ACC + GAP (Harmonizer OFF)
# with R8K-L18-s2's stored read's serve and chunk lists. Box, repo root; every job through the pool, the whole chain in one submit; ids in
# $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/interp.
#   interp.sh all    merge (CPU: INTP50-NR, INTP75-NR + the a = 0 / 1 check points) -> check (small GPU job: serve load, frozen params, forwards;
#                    failing stops the chain) -> per point 8 read chunks + pai_fix tax -> scorer_v2 new -> report -> $O/DONE
#   interp.sh wait   one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: merge | check | sv2 | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/interp; ME=experiments/round9/scripts/interp.sh; PY=experiments/round9/scripts/interp.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; SRC=$DATA_DIR/runs/round8/s2/pai281/R8K-L18-s2-ag; CH=$SRC/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
PTS=(INTP50-NR:0.5 INTP75-NR:0.75); CHK=(INTPCHK0-NR:0 INTPCHK1-NR:1)
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-interp --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= pcl.sh read281, R8K-L18-s2's chunk lists)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
all)
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for s in "${PTS[@]}" "${CHK[@]}"; do [[ ! -e $RUNS/${s%:*} ]] || die "${s%:*} exists"; done
  for f in NTV0-L18-s0 R8K-L18-s2; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f facts"; done
  grep -qx "SERVE=$SERVE" $SRC/config.txt && grep -qx "WIZ=$WIZ" $SRC/config.txt && grep -qx "ALPASIM_FAST=1" $SRC/config.txt && grep -qx "CONC=4" $SRC/config.txt \
    || die "serve differs from $SRC/config.txt"
  [[ $(ls $CH/*.txt | wc -l) == 8 ]] || die "not 8 chunk lists in $CH"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  m=$(sub interp-merge --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME merge) || exit 1
  c=$(sub interp-check --after $m --vram 8 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME check) || exit 1
  TAX=()
  for s in "${PTS[@]}"; do y=($(read281 ${s%:*} $c)); [[ ${#y[@]} == 9 ]] || die "read281 ${s%:*}"; TAX+=(${y[-1]})
    echo "$(date '+%F %T') ${s%:*}: 281 ${y[*]}" >> $O/stages.txt; done
  v=$(sub interp-sv2 --after $(IFS=,; echo "${TAX[*]}") --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- bash $ME sv2) || exit 1
  r=$(sub interp-report --after $v --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): merge $m check $c tax ${TAX[*]} sv2 $v report $r" | tee -a $O/stages.txt ;;
merge)
  for s in "${CHK[@]}" "${PTS[@]}"; do $TR $PY merge --a ${s#*:} --tag ${s%:*} > $O/results/merge_${s%:*}.json || die "merge ${s%:*}"; done ;;
check)
  $TR $PY check --tags ${PTS[@]%:*} --chk0 ${CHK[0]%:*} --chk1 ${CHK[1]%:*} --out $O/results/check.json > $O/results/check.log 2>&1 \
    || die "check (log $O/results/check.log)"
  rm -rf $RUNS/${CHK[0]%:*} $RUNS/${CHK[1]%:*} ;;
sv2)
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads INTP50-NR-ag,INTP75-NR-ag --workers 16 \
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
