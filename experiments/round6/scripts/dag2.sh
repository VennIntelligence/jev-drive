#!/usr/bin/env bash
# ROUND6 DAGGER-MAP iteration 2 (plans/2026-10-21-dagger-map.md amendment 4). The two student chains run through dag.sh with DAG_IT2=1
# (R6DAG-L18-s0 -> $DATA_DIR/runs/round6/dag2/s0, pool on1_PAID2A; R6DAG-L18-s1 -> dag2/s1, on1_PAID2B; teacher b4). This script joins them:
# iteration-1 rows re-taught with b4 -> on1_PAIDL (tab.fut only, tokens zero-copy) -> zero-copy merge on1_JTPHD2 = on1_JTPH + PAIDL + PAID2A + PAID2B
# -> draws R6D20 (identity) / R6DAG2 (8 union rows a batch, uniform over the drawable rows) -> smoke + T3 -> BUILT; T2 re-read under b4 beside it.
# launch: R6DAG2-L18-s0 / s1 (R5LC's argv + the union, from NTV0-L18-s0, 9 500 steps) -> PAI 281 ACC + GAP -> r281 + verdict -> ruler -> DONE.
# Box, repo root; every job through the pool at PRIO 22; ids in $O/jobs.txt, stages in $O/stages.txt, a stop reason in $O/VERDICT.
#   dag2.sh join        (after both students' ROWSDONE) the join chain in one submission
#   dag2.sh launch      (after BUILT) trainings + reads + report + ruler
#   dag2.sh wait [M]    one run_in_background waiter: $O/VERDICT, $O/<M> (default DONE) or a failed job of this lane
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round6/dag2; O1=$DATA_DIR/runs/round6/dag; ME=experiments/round6/scripts/dag2.sh; DS=experiments/round6/scripts/dag.sh
PY=experiments/round6/scripts/dag.py; V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python
RF=experiments/route_follow/scripts; CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts
RES=experiments/round6/results/dagger2; PRIO=${PRIO:-22}; N=9500; KDAG=8; NM=JTPHD2; R2PY=experiments/round2/scripts/r2.py
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both on1_PAIH:ph:both on1_PAIH05:ph05:both on1_PAIH15:ph15:both"
DAGP="on1_PAIDL,on1_PAID2A,on1_PAID2B"; EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD DAG_RES=$RES)
EV2=(--env DAG_NM=$NM --env DAG_W0=R6D20 --env DAG_WF=R6DAG2 --env DAG_RES=$RES)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $O/checks $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round6-dag --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "${EV2[@]}" "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
run() { local X; X=$(env DAG_NM=$NM DAG_W0=R6D20 DAG_WF=R6DAG2 bash $DS printargv "$@") || die "argv $*"; mapfile -t A <<< "$X"; }
read281() {  # as dag.sh read281, under $O/pai281
  local R=$O/pai281/$1-ag ids t; mkdir -p $O/pai281
  AFTER=${2:-} CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO \
    bash experiments/alpasim/scripts/pai_box.sh submit $R $1 > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
join)
  for s in s0 s1; do [[ -f $O/$s/ROWSDONE ]] || die "student $s rows not done"; done; [[ ! -e $CR/on1_PAIDL && ! -e $CR/on1_$NM ]] || die "join already built"
  h=$(git rev-parse --short HEAD); echo "$h join" >> $O/code.txt
  t=$(sub dag2-teach1 --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- "${EA[@]}" $AP $PY teach --rows $O1/rows/PAID \
      --cfg-glob "$O1/roll/runs/*/sim/generated-user-config-0.yaml" --out $O1/teach_b4 --workers 16) || exit 1
  l=$(sub dag2-relabel --after $t --vram 0.1 --cpu 4 --ram 64 --timeout-h 2 -- env DAG_RES=$RES $V $PY label --root $O1 --pool on1_PAIDL --copy-from on1_PAID \
      --teach $O1/teach_b4) || exit 1
  m=$(sub dag2-merge --after $l --vram 0.1 --cpu 4 --ram 200 --timeout-h 3 -- bash $ME merge) || exit 1
  w=$(sub dag2-weights --after $m --vram 0.1 --cpu 2 --ram 48 --timeout-h 1 -- bash $ME weights) || exit 1
  sm=$(sub dag2-smoke --after $w "${FT[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  ia=$(sub dag2-ident-lc --after $w "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun lc) || exit 1
  ib=$(sub dag2-ident-d0 --after $w "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun d0) || exit 1
  ic=$(sub dag2-ident --after $ia,$ib --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  bd=$(sub dag2-build --after $sm,$ic --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME build) || exit 1
  t2=$(sub dag2-t2 --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- "${EA[@]}" $AP $PY t2 --workers 16) || exit 1
  echo "$(date '+%F %T') join (code $h): teach1 $t relabel $l merge $m weights $w smoke $sm identity $ia $ib $ic build $bd T2 $t2" | tee -a $O/stages.txt ;;
merge)
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 120 )) || die "free $free GB < 120 GB"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_PAIDL:pdl:both on1_PAID2A:pda:both on1_PAID2B:pdb:both --out $NM --nan-keys fut_reg || die merge
  $TR $PY mfix --root $O --pool on1_$NM --ref on1_JTPH || die "mfix: the merged prefix differs from on1_JTPH" ;;
weights)
  $TR $PY weights --pool on1_$NM --base R5LC --k-on 64 --k-dag 0 --out R6D20 --src $DAGP || die R6D20
  $TR $PY weights --pool on1_$NM --base R5LC --k-on 64 --k-dag $KDAG --out R6DAG2 --src $DAGP || die R6DAG2 ;;
smoke)
  X=R6SM2-L18-s0; [[ ! -e $RUNS/$X ]] || die "$X exists"; run dag 60 $X 0
  echo "$(date '+%F %T') smoke: ${A[*]} --diag-every 20 --save-at 20,40" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --save-at 20,40 > $O/smoke.log 2>&1 || die "smoke run (log $O/smoke.log)"
  $TR $PY smoke --tag $X --wfile R6DAG2 --k-dag $KDAG; r=$?
  rm -rf $RUNS/$X $RUNS/R6SM2x20-L18-s0 $RUNS/R6SM2x40-L18-s0; exit $r ;;
identrun)
  declare -A IX=([lc]=R6ID2LC [d0]=R6ID2D0); X=${IX[$2]}-L18-s0; [[ ! -e $RUNS/$X ]] || die "$X exists"; run $2 60 $X
  echo "$(date '+%F %T') identity $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R6ID2LC-L18-s0 --b R6ID2D0-L18-s0 --out round6_dag2_identity \
      --note "R5LC-L18-s0 argv 60 steps eager: on1_JTPH / R5LC vs on1_JTPHD2 / R6D20 (the DAgger union at weight 0)"; r=$?
  cp experiments/defect_fix/results/round6_dag2_identity.json $RES/identity.json || r=4
  rm -rf $RUNS/R6ID2LC-L18-s0 $RUNS/R6ID2D0-L18-s0; exit $r ;;
build)
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke.json'))['pass_'] and json.load(open('$RES/identity.json'))['identical'] else 1)" \
    || { echo "STOP smoke / identity (T3)" > $O/VERDICT; exit 1; }
  date '+%F %T' > $O/BUILT ;;
launch)
  [[ -f $O/BUILT && ! -f $O/VERDICT ]] || die "not built (or VERDICT)"
  for s in 0 1; do for X in R6DAG2-L18-s$s R6DAG2x2500-L18-s$s R6DAG2x5000-L18-s$s R6DAG2x7500-L18-s$s; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done; done
  for f in $F_/NTV0-L18-s0-281ag.csv $F_/R6DAG-L18-s0-281ag.csv $F_/R6DAG-L18-s1-281ag.csv; do [[ -s $f ]] || die "no stored $f"; done
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt; x=()
  for s in 0 1; do t=$(sub dag2-train-s$s "${FT[@]}" --timeout-h 4 -- bash $ME train $s) || exit 1
    y=($(read281 R6DAG2-L18-s$s $t)) || exit 1; x+=(${y[-1]})
    echo "$(date '+%F %T') launch s$s (code $h): train $t PAI 281 chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]}" | tee -a $O/stages.txt; done
  r=$(sub dag2-r281 --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281) || exit 1
  P0=$DATA_DIR/tmp/pcs_r6d; PT=$DATA_DIR/tmp/pcs_r6d2; [[ ! -e $PT ]] || die "$PT exists"; for f in $P0/rows.csv.gz $P0/out/fits/paimix.npz; do [[ -s $f ]] || die "no $f"; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$DATA_DIR/tmp/pcs_r5/site)
  c=$(sub dag2-pcs-collect --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only DAG2-s0+AG,DAG2-s1+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub dag2-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  p=$(sub dag2-pcs-report --after $(csv ${b[@]} $r) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME pcsrep) || exit 1
  echo "$(date '+%F %T') launch: r281 $r ruler collect $c boot ${b[*]} report $p" | tee -a $O/stages.txt ;;
train) s=$2; run dag $N R6DAG2-L18-s$s $s; echo "$(date '+%F %T') train s$s: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
r281)
  rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R5LC-s0-ag=$F_/R5LC-L18-s0-281ag.csv R5LC-s1-ag=$F_/R5LC-L18-s1-281ag.csv
      R6DAG-s0-ag=$F_/R6DAG-L18-s0-281ag.csv R6DAG-s1-ag=$F_/R6DAG-L18-s1-281ag.csv DAG2-s0-ag=$F_/R6DAG2-L18-s0-281ag.csv DAG2-s1-ag=$F_/R6DAG2-L18-s1-281ag.csv)
  $TR $R2PY r281 --read "${rd[@]}" --pairs DAG2-s0-ag:NTV0-ag DAG2-s1-ag:NTV0-ag DAG2-s0-ag:R6DAG-s0-ag DAG2-s1-ag:R6DAG-s1-ag DAG2-s0-ag:R5LC-s0-ag \
      DAG2-s1-ag:R5LC-s1-ag DAG2-s1-ag:DAG2-s0-ag --out $RES/pai281_DAG2_ag.md \
      --title "PAI 281, ACC + GAP: R6DAG2 (DAgger iteration 2) s0 / s1 vs stored R6DAG s0 / s1, R5LC (MV-LC) s0 / s1 and NTV0-L18-s0" || die "r281"
  $V $PY verdict --json $RES/pai281_DAG2_ag.json --md $RES/pai281_DAG2_ag.md --read "${rd[@]}" --arm DAG2 --ctrl R6DAG,R5LC || die "verdict"
  date '+%F %T' > $O/R281DONE ;;
pcsrep)
  P0=$DATA_DIR/tmp/pcs_r6d; PT=$DATA_DIR/tmp/pcs_r6d2; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  python3 - $P0/out $PT/out <<'PY' || die "bootstrap draws of the old reads differ from iteration 1's"
import csv, glob, sys
rd = lambda d: [r for f in sorted(glob.glob(d + "/boot_*.csv")) for r in csv.DictReader(open(f))]  # noqa: E731
a, b = rd(sys.argv[1]), rd(sys.argv[2])
d = max(abs(float(x[k]) - float(y[k])) for x, y in zip(a, b) for k in x if x[k] and y[k])
print("old draws repeat:", len(a), "draws,", len(a[0]), "subjects, max |diff|", d, "PCS")
assert len(a) == len(b) == 100 and d < 0.5, "boot mismatch"
PY
  sj="R3MV-s0+AG=R3MV-L18-s0-281ag.csv,R5LC-s0+AG=R5LC-L18-s0-281ag.csv,R5LC-s1+AG=R5LC-L18-s1-281ag.csv,DAG-s0+AG=R6DAG-L18-s0-281ag.csv"
  sj+=",DAG-s1+AG=R6DAG-L18-s1-281ag.csv,DAG2-s0+AG=R6DAG2-L18-s0-281ag.csv,DAG2-s1+AG=R6DAG2-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" \
      --pairs DAG2-s0+AG:DAG-s0+AG,DAG2-s1+AG:DAG-s1+AG,DAG2-s1+AG:DAG2-s0+AG > $PT/out/r5.log 2>&1 || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_dag2.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_dag2.json || die "copy ruler report"
  $V $PY pcs --pcs $RES/pcs_dag2.json --json $RES/pai281_DAG2_ag.json --md $RES/pai281_DAG2_ag.md --arm DAG2 --ctrl DAG,R5LC || die "pcs rows"
  date '+%F %T' > $O/DONE ;;
wait)
  M=${2:-DONE}
  until [[ -f $O/VERDICT || -f $O/$M ]] || ! $V - $O/jobs.txt <<'PY'
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
  cat $O/VERDICT 2>/dev/null; for f in BUILT R281DONE DONE; do [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; done; tail -3 $O/stages.txt; true ;;
*) sed -n 2,11p "$0"; exit 2 ;;
esac
