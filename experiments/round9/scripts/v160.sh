#!/usr/bin/env bash
# ROUND9 VAL160 (plans/2026-10-24-val160.md): read NT3D1-F-s0 (step 3 D1), R8K-L18-s2 and PCL1L-F-s0 (P1L) on 152 PAI val160 scenes (val160 minus the 8
# drive-linked ones), serve = the 281 ACC + GAP serve. Box, repo root; every job through the pool, the whole chain in one submit; ids in $O/jobs.txt.
#   v160.sh all    21 chunk jobs + 3 tax jobs -> report (scorer_v2 new on the 3 reads, then v160.py) -> $O/DONE
#   v160.sh wait   one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/val160; ME=experiments/round9/scripts/v160.sh; PY=experiments/round9/scripts/v160.py
V=.venv/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
CH=experiments/round9/results/val160/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = the 281 reads
NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
TAGS=(NT3D1-F-s0 R8K-L18-s2 PCL1L-F-s0); PRIO=${PRIO:-12}
mkdir -p $O/pool $O/results $O/reads
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-val160 --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
all)
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for t in "${TAGS[@]}"; do [[ -s $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "no checkpoint $t"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt; X=()
  for t in "${TAGS[@]}"; do
    R=$O/reads/$t-ag; [[ ! -e $R ]] || die "$R exists"
    CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $t > $R.submit.txt || die "pai submit $t"
    ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 7 ]] || die "$t: not 7 chunks"
    awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
    x=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $t-v160ag $R | tail -1) && [[ -n $x ]] || die "tax $t"
    echo "$x pai_fix-tax-$t-v160ag" >> $O/jobs.txt; X+=($x)
  done
  rp=$(sub v160-report --after $(IFS=,; echo "${X[*]}") --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): tax ${X[*]} report $rp" | tee -a $O/stages.txt ;;
report)
  rm -rf $O/sv2
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads ${TAGS[0]}-v160,${TAGS[1]}-v160,${TAGS[2]}-v160 \
      --workers 16 > $O/sv2.log 2>&1 || die "scorer_v2 new (log $O/sv2.log)"
  PYTHONPATH=. $V $PY report --dir $O --out $O/results || die "report"; date '+%F %T' > $O/DONE ;;
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
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)" || echo "FAILED job(s), see above"; tail -5 $O/stages.txt; true ;;
*) sed -n 2,6p "$0"; exit 2 ;;
esac
