#!/usr/bin/env bash
# One full PAI 281 read of R3MV-L18-s0 under the ACC + GAP serve, as round 2's read281 + tax281 + r2.py r281 path (r2.sh s2c / s2creport):
# NTV0's chunk lists, g10, FAST=1, CONC 4, JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7. Box, repo root; every job through the GPU pool (PRIO 25).
# Compared with the stored NTV0-L18-s0 and R2C-L18-s0 ACC + GAP 281 facts. Separate lane from r5.sh; state in $O (jobs.txt, stages.txt, DONE).
#   r5_281.sh launch   one submission: 8 chunk jobs -> facts (rf_tax + pai_fix tax) -> report job (--after) -> results/pai281_R3MV_ag.md + DONE
#   r5_281.sh wait     blocks until $O/DONE or a failed / cancelled job of this lane (one run_in_background waiter)
#   job body: report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round5/r5_281; ME=experiments/round5/scripts/r5_281.sh; R2PY=experiments/round2/scripts/r2.py; RES=experiments/round5/results
TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; F_=$DATA_DIR/runs/pai_fix/facts; T=R3MV-L18-s0; CTRL=R2C-L18-s0; PRIO=${PRIO:-25}
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-r5281 --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
launch)
  [[ ! -e $O/pai281 && ! -s $O/jobs.txt ]] || die "already submitted ($O)"; [[ -s $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "no $T checkpoint"
  for f in $F_/NTV0-L18-s0-281ag.csv $F_/$CTRL-281ag.csv; do [[ -s $f ]] || die "no $f"; done
  mkdir -p $O/pai281; R=$O/pai281/$T-ag
  CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" \
    WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R.submit.txt \
    || die "pai submit $T"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$T: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281ag $R | tail -1) && [[ -n $t ]] || die "tax $T"
  echo "$t pai_fix-tax-$T-281ag" >> $O/jobs.txt
  r=$(sub r5-281-report --after $t --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') launch (code $(git rev-parse --short HEAD)): chunks $(paste -sd' ' <<< "$ids") facts $t report $r" | tee -a $O/stages.txt ;;
report)
  $TR $R2PY r281 --read NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R2C-ag=$F_/$CTRL-281ag.csv R3MV-ag=$F_/$T-281ag.csv \
      --pairs R3MV-ag:NTV0-ag R3MV-ag:R2C-ag --out $RES/pai281_R3MV_ag.md \
      --title "PAI 281, ACC + GAP: R3MV vs stored NTV0-L18-s0 and R2C-L18-s0" || die "r281 $T"
  date '+%F %T' > $O/DONE ;;
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
  do sleep 300; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; true ;;
*) sed -n 2,7p "$0"; exit 2 ;;
esac
