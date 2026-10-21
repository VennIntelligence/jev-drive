#!/usr/bin/env bash
# TGT=R5A (default R3MV, unchanged): the same read of R5A-L18-s0 (own lane dir $DATA_DIR/runs/round5/r5_281_R5A, PRIO 20), reported against stored NTV0 and
# R3MV-s0 (r5.py newbreaks adds the new zeros / partials on the scenes NTV0 scores 1.0), then the out-of-pool local PAI PCS ruler (as r5.sh pcsL did for
# MV-LC: pai_pcs_local.py collect / fit boot / r5, CPU jobs PRIO 15) appended by r5.py pcsrow -> results/pai281_R5A_ag.md + DONE. One submission:
#   TGT=R5A r5_281.sh launch; TGT=R5A r5_281.sh wait
# One full PAI 281 read of R3MV-L18-s0 under the ACC + GAP serve, as round 2's read281 + tax281 + r2.py r281 path (r2.sh s2c / s2creport):
# NTV0's chunk lists, g10, FAST=1, CONC 4, JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7. Box, repo root; every job through the GPU pool (PRIO 25).
# Compared with the stored NTV0-L18-s0 and R2C-L18-s0 ACC + GAP 281 facts. Separate lane from r5.sh; state in $O (jobs.txt, stages.txt, DONE).
#   r5_281.sh launch   one submission: 8 chunk jobs -> facts (rf_tax + pai_fix tax) -> report job (--after) -> results/pai281_R3MV_ag.md + DONE
#   r5_281.sh wait     blocks until $O/DONE or a failed / cancelled job of this lane (one run_in_background waiter)
#   job body: report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round5/r5_281; ME=experiments/round5/scripts/r5_281.sh; R2PY=experiments/round2/scripts/r2.py; RES=experiments/round5/results
TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; F_=$DATA_DIR/runs/pai_fix/facts; T=R3MV-L18-s0; CTRL=R2C-L18-s0; PRIO=${PRIO:-25}
TGT=${TGT:-R3MV}; PY=experiments/round5/scripts/r5.py; MV=R3MV-L18-s0
case $TGT in R3MV) ;; R5A) O=${O}_R5A; T=R5A-L18-s0; PRIO=${PRIO_R5A:-20} ;; *) echo "TGT must be R3MV or R5A" >&2; exit 2 ;; esac
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-r5281 --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
launch)
  [[ ! -e $O/pai281 && ! -s $O/jobs.txt ]] || die "already submitted ($O)"; [[ -s $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "no $T checkpoint"
  for f in $F_/NTV0-L18-s0-281ag.csv $F_/$CTRL-281ag.csv $F_/$MV-281ag.csv; do [[ -s $f ]] || die "no $f"; done
  mkdir -p $O/pai281; R=$O/pai281/$T-ag
  CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" \
    WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R.submit.txt \
    || die "pai submit $T"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$T: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281ag $R | tail -1) && [[ -n $t ]] || die "tax $T"
  echo "$t pai_fix-tax-$T-281ag" >> $O/jobs.txt
  r=$(sub r5-281-report --after $t --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- env TGT=$TGT bash $ME report) || exit 1
  echo "$(date '+%F %T') launch (code $(git rev-parse --short HEAD)): chunks $(paste -sd' ' <<< "$ids") facts $t report $r" | tee -a $O/stages.txt
  [[ $TGT == R5A ]] || exit 0
  PRIO=15; P0=$DATA_DIR/tmp/pcs_r5; PT=$DATA_DIR/tmp/pcs_r5a; [[ ! -e $PT ]] || die "$PT exists"; for f in $P0/rows.csv.gz $P0/out/fits/paimix.npz; do [[ -s $f ]] || die "no $f"; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$P0/site)
  c=$(sub r5-pcs-collect --after $t --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz --only R5A-s0+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub r5-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  p=$(sub r5-pcs-report --after "$(IFS=,; echo "${b[*]}"),$r" --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" env TGT=$TGT bash $ME pcsrep) || exit 1
  echo "$(date '+%F %T') pcs (code $(git rev-parse --short HEAD)): collect $c boot ${b[*]} report $p" | tee -a $O/stages.txt ;;
pcsrep)
  P0=$DATA_DIR/tmp/pcs_r5; PT=$DATA_DIR/tmp/pcs_r5a; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  python3 - $P0/out $PT/out <<'PY' || die "bootstrap draws of the old reads differ from d429's"
import csv, glob, sys                                            # same seeds, same pool: the old subjects' draws must repeat exactly
rd = lambda d: [r for f in sorted(glob.glob(d + "/boot_*.csv")) for r in csv.DictReader(open(f))]  # noqa: E731
a, b = rd(sys.argv[1]), rd(sys.argv[2])
d = max(abs(float(x[k]) - float(y[k])) for x, y in zip(a, b) for k in x if x[k] and y[k])
print("old draws repeat:", len(a), "draws,", len(a[0]), "subjects, max |diff|", d, "PCS")
assert len(a) == len(b) == 100 and d < 0.5, "boot mismatch"
PY
  sj="R2C-s0+AG=$CTRL-281ag.csv,R2J-s0+AG=R2J-L18-s0-281ag.csv,R3MV-s0+AG=$MV-281ag.csv,R5A-s0+AG=$T-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs R5A-s0+AG:R3MV-s0+AG,R5A-s0+AG:R2C-s0+AG > $PT/out/r5.log 2>&1 \
      || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_r5a.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_r5a.json || die "copy ruler report"
  $V $PY pcsrow --pcs $RES/pcs_r5a.json --json $RES/pai281_R5A_ag.json --md $RES/pai281_R5A_ag.md --subject R3MV-s0+AG R5A-s0+AG --report pcs_r5a.md || die "pcsrow"
  date '+%F %T' > $O/DONE ;;
report)
  if [[ $TGT == R5A ]]; then rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R3MV-s0-ag=$F_/$MV-281ag.csv R5A-s0-ag=$F_/$T-281ag.csv)
    $TR $R2PY r281 --read "${rd[@]}" --pairs R5A-s0-ag:R3MV-s0-ag R5A-s0-ag:NTV0-ag --out $RES/pai281_R5A_ag.md \
        --title "PAI 281, ACC + GAP: R5A (S3 outcome rows) s0 vs stored R3MV-L18-s0 and NTV0-L18-s0" || die "r281 $T"
    $V $PY newbreaks --json $RES/pai281_R5A_ag.json --md $RES/pai281_R5A_ag.md --read "${rd[@]}" || die "newbreaks"
    exit 0; fi
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
