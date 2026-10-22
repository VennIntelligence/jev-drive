#!/usr/bin/env bash
# ROUND8 (plans/2026-10-22-r8k.md). Box, repo root; every job through the pool; ids in <dir>/jobs.txt, stages in <dir>/stages.txt.
#  R8K   R6DAG's recipe exactly (dag.sh printargv dag: NTV0-L18-s0 init, 9 500 steps, navtrain 51, S 32, ho_w 3) + round7's keep-pace floor and F2
#        (pool on1_JTPHDK = b5 labels on the DAgger rows, draw file round7/joint/w/R7JNS.npz: F2 on human + DAgger rows, on-k 72) + Harmonizer-ON
#        twin tokens (rm_train --twin PAIHTW:1.0). No S-MIX, no command rule; serve = R6DAG's (ACC + GAP, g10), no JEV_CMD.
#  NOISE R6DAG-L18-s1 re-read with HARMONIZER=1 on chunks c000 + c001 of NTV0's chunk table, compared per scene with the stored ON read.
#   r8k.sh noise          (PRIO 22) the two ON chunks -> pai_fix tax -> r8k.py noise -> $N_/DONE
#   r8k.sh all            (PRIO 20) gates (identity A / B -> bitcmp; smokes R8KSM / R8KSC -> r8k.py smoke) -> $K/GATES, and chained on GATES:
#                         R8K-L18-s0 / s1 -> PAI 281 OFF + ON each -> ruler (collect + 20 boots) -> report -> $K/DONE
#   r8k.sh s2             (PRIO 20, amendment 1) third seed of both recipes: R6DAG-L18-s2 (R6DAG argv) and R8K-L18-s2 (R8K argv), each -> PAI 281
#                         OFF + ON -> r8k.py seeds (3-seed recipe means, after R8K-s1's ON tax too) -> $S2/DONE
#   r8k.sh ruler2         (amendment 1) DAG-s2 / R8K-s2 into the ruler: collect -> the same 20 boot seeds as R8K's ruler (same draws) -> r5
#                         -> $S2/results/pcs_s2.{md,json}, $S2/RULER
#   r8k.sh wait <K|N|S> [M] one run_in_background waiter: the marker (default DONE), $K/VERDICT or a failed / cancelled job of that lane
#   r8k.sh printargv <variant> <steps> <tag> <seed>   r6 = R6DAG's argv; id = HEAD with every switch off (on1_JTPHDI / R7J0); k = the arm
#   job bodies: identrun a|b | ident | smokerun k|r6 | smoke | train <seed> | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
K=$DATA_DIR/runs/round8/r8k; N_=$DATA_DIR/runs/round8/noise; S2=$DATA_DIR/runs/round8/s2; ME=experiments/round8/scripts/r8k.sh; PY=experiments/round8/scripts/r8k.py
DS=experiments/round6/scripts/dag.sh; V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; JW=$DATA_DIR/runs/round7/joint/w; N=9500
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks; ST=$DATA_DIR/runs/round7/harmcl/R6DAG-L18-s1-agH
PRE=${R8K_PRE:-ba54017b}; WT=$DATA_DIR/tmp/joint_wt_$PRE                # identity run A: R6DAG's argv at the pre-round7 commit (round7's worktree)
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
TWIN=${R8K_TWIN:-PAIHTW:1.0}
FT=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train)
mkdir -p $K/pool $K/results $N_/pool $N_/results $S2/pool $S2/results
die() { echo "$(date '+%F %T') ERROR $*" | tee -a ${LANE:-$K}/stages.txt >&2; exit 1; }
sub() {  # sub <lane dir> <prio> <name> <pool args...>
  local d=$1 p=$2 name=$3; shift 3; local id
  id=$($V -m jevdrive.cl submit --owner round8 --name "$name" --tries 1 --priority $p --no-check --log-dir $d/pool/$name "$@" < /dev/null | awk '{print $NF}') \
    && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $d/jobs.txt; echo "$id"; }
ks() { sub $K 20 "$@"; }
csv() { local IFS=,; echo "$*"; }
argv() {  # argv <variant> <steps> <tag> <seed>: one argument a line
  local X; X=$(bash $DS printargv dag "$2" "$3" "$4") || die "R6DAG argv"
  ARGV_X="$X" python3 - "$1" "$JW" "$TWIN" <<'PY'
import os, sys
var, w, twin = sys.argv[1:4]
a = [x for x in os.environ["ARGV_X"].split("\n") if x != ""]
last = lambda q: len(a) - 1 - a[::-1].index(q)  # noqa: E731
assert a[last("--on") + 1] == "on1_JTPHD" and a[last("--on-wfile") + 1].endswith("/round6/dag/w/R6DAG.npz") and a[last("--on-k") + 1] == "72"
assert a[a.index("--ho-w") + 1] == "3" and a.count("--ho-w") == 1 and a[a.index("--init") + 1] == "NTV0-L18-s0" and "--cmd-rule" not in a and "--twin" not in a
if var == "id":
    a[last("--on") + 1], a[last("--on-wfile") + 1] = "on1_JTPHDI", f"{w}/R7J0.npz"
elif var == "k":
    a[last("--on") + 1], a[last("--on-wfile") + 1] = "on1_JTPHDK", f"{w}/R7JNS.npz"
    a += ["--twin", twin]
else:
    assert var == "r6", var
print("\n".join(a))
PY
}
run() { local X; X=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$X"; }
read281() {  # read281 <tag> <ag|agH> <after> -> "<chunk ids...> <tax id>" (NTV0's chunk table, ACC + GAP, g10, FAST, CONC 4; agH: HARMONIZER=1)
  local L=${LD:-$K}; local R=$L/pai281/$1-$2 ids t sv="$SERVE" ex=(); mkdir -p $L/pai281; [[ ! -e $R ]] || die "$R exists"
  [[ $2 == agH ]] && { sv="$sv HARMONIZER=1"; ex=(VRAM=30 RAM=40); }
  env "${ex[@]}" AFTER=$3 CHUNKS_FROM=$CH SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=20 bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1 $2"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1 $2: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt | sed "s/\$/-$2/" >> $L/jobs.txt
  t=$(PRIO=20 AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281$2 $R | tail -1) && [[ -n $t ]] || die "tax $1 $2"
  echo "$t pai_fix-tax-$1-281$2" >> $L/jobs.txt; echo $ids $t; }
waiter() {  # waiter <lane dir> <marker>: returns at the marker, a VERDICT, or a failed / cancelled job (the last id of each name)
  local d=$1 m=$2
  until [[ -f $d/VERDICT || -f $d/$m ]] || ! $V - $d/jobs.txt <<'PY'
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
  cat $d/VERDICT 2>/dev/null; for f in GATES DONE; do [[ -f $d/$f ]] && echo "$f $(cat $d/$f)"; done; tail -3 $d/stages.txt; true; }
case ${1:-} in
noise)
  LANE=$N_; [[ ! -e $N_/DONE ]] || die "noise already ran"; R=$N_/R6DAG-L18-s1-agH; [[ ! -e $R ]] || die "$R exists"
  for f in $F_/R6DAG-L18-s1-281agH.csv $F_/R6DAG-L18-s1-281ag.csv; do [[ -s $f ]] || die "no $f"; done
  h=$(git rev-parse --short HEAD); mkdir -p $N_/chunks && cp $CH/c000.txt $CH/c001.txt $N_/chunks/
  CHUNKS_FROM=$N_/chunks SERVE="$SERVE HARMONIZER=1" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=22 VRAM=30 RAM=40 \
    bash experiments/alpasim/scripts/pai_box.sh submit $R R6DAG-L18-s1 > $R.submit.txt || die "pai submit noise"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 2 ]] || die "noise: not 2 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $N_/jobs.txt
  t=$(PRIO=22 AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax R6DAG-L18-s1-281agH-rr $R | tail -1) && [[ -n $t ]] || die "noise tax"
  echo "$t pai_fix-tax-noise" >> $N_/jobs.txt
  c=$(sub $N_ 22 noise-cmp --after $t --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash -c "$V $PY noise --stored $ST --new $R \
      --facts-stored $F_/R6DAG-L18-s1-281agH.csv --facts-new $F_/R6DAG-L18-s1-281agH-rr.csv --facts-off $F_/R6DAG-L18-s1-281ag.csv \
      --out $N_/results/noise && date '+%F %T' > $N_/DONE") || exit 1
  echo "$(date '+%F %T') noise (code $h): chunks $(echo $ids) tax $t cmp $c" | tee -a $N_/stages.txt ;;
all)
  [[ ! -e $K/GATES && ! -e $K/DONE ]] || die "already ran"
  for s in 0 1; do X=R8K-L18-s$s; [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  for f in NTV0-L18-s0-281ag NTV0-L18-s0-281agH R6DAG-L18-s0-281ag R6DAG-L18-s1-281ag R6DAG-L18-s0-281agH R6DAG-L18-s1-281agH \
           R7J-L18-s0-281ag R7J-L18-s1-281ag R7J-L18-s0-281agH R7J-L18-s1-281agH; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  for f in $JW/R7JNS.npz $JW/R7J0.npz $DATA_DIR/runs/round7/joint/results/w_R7JNS.json; do [[ -s $f ]] || die "no $f"; done
  [[ $(git -C $WT rev-parse --short HEAD) == ${PRE:0:8} ]] || die "worktree $WT is not at $PRE"
  P0=$DATA_DIR/tmp/pcs_r7j; PT=$DATA_DIR/tmp/pcs_r8k; [[ ! -e $PT ]] || die "$PT exists"
  for f in $P0/rows.csv.gz $P0/out/fits/paimix.npz; do [[ -s $f ]] || die "no $f"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $K/code.txt
  ia=$(ks r8k-ident-a "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun a) || exit 1
  ib=$(ks r8k-ident-b "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun b) || exit 1
  ic=$(ks r8k-ident --after $ia,$ib --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  sk=$(ks r8k-smoke-k "${FT[@]}" --timeout-h 1 -- bash $ME smokerun k) || exit 1
  sc=$(ks r8k-smoke-c "${FT[@]}" --timeout-h 1 -- bash $ME smokerun r6) || exit 1
  sm=$(ks r8k-smoke --after $sk,$sc --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME smoke) || exit 1
  g=$(ks r8k-gates-done --after $ic,$sm --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $K/GATES") || exit 1
  echo "$(date '+%F %T') gates (code $h): identity $ia $ib $ic smoke $sk $sc $sm GATES $g" | tee -a $K/stages.txt
  x=()
  for s in 0 1; do t=$(ks r8k-train-s$s "${FT[@]}" --after $g --timeout-h 4 -- bash $ME train $s) || exit 1
    y=($(read281 R8K-L18-s$s ag $t)) || exit 1; z=($(read281 R8K-L18-s$s agH $t)) || exit 1; x+=(${y[-1]} ${z[-1]})
    echo "$(date '+%F %T') launch s$s: train $t OFF ${y[*]} ON ${z[*]}" | tee -a $K/stages.txt; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$DATA_DIR/tmp/pcs_r5/site)
  c=$(ks r8k-pcs-collect --after $(csv ${x[0]} ${x[2]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only R8K-s0+AG,R8K-s1+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(ks r8k-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  r=$(ks r8k-report --after $(csv ${x[@]} ${b[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME report) || exit 1
  echo "$(date '+%F %T') launch: ruler collect $c boot ${b[*]} report $r" | tee -a $K/stages.txt ;;
identrun)  # identrun a | b: 60 steps eager; a = R6DAG's argv from the worktree at $PRE, b = HEAD with every switch off
  if [[ $2 == a ]]; then X=R8IDA-L18-s0; run r6 60 $X 0; cd $WT; else X=R8IDB-L18-s0; run id 60 $X 0; fi
  [[ ! -e $RUNS/$X ]] || die "$X exists"
  echo "$(date '+%F %T') identity $2 ($(git rev-parse --short HEAD)): ${A[*]}" | tee -a $K/stages.txt; $TR "${A[@]}" > $K/ident_$2.log 2>&1 || die "identity $2 (log $K/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R8IDA-L18-s0 --b R8IDB-L18-s0 --out round8_r8k_identity \
      --note "R6DAG-L18-s0 argv 60 steps eager: code $PRE on on1_JTPHD / R6DAG vs HEAD with every R8K switch off (on1_JTPHDI / R7J0, no --twin)"; r=$?
  mv experiments/defect_fix/results/round8_r8k_identity.json $K/results/identity.json || r=4
  rm -rf $RUNS/R8IDA-L18-s0 $RUNS/R8IDB-L18-s0; exit $r ;;
smokerun)  # smokerun k | r6: 300 steps, diag every 20, dev every 100 (k = the arm, r6 = R6DAG's argv: the gradient-ratio control)
  declare -A SX=([k]=R8KSM [r6]=R8KSC); X=${SX[$2]}-L18-s0; [[ ! -e $RUNS/$X ]] || die "$X exists"; run $2 300 $X 0
  echo "$(date '+%F %T') smoke $2: ${A[*]} --diag-every 20 --eval-every 100" | tee -a $K/stages.txt
  $TR "${A[@]}" --diag-every 20 --eval-every 100 > $K/smoke_$2.log 2>&1 || die "smoke $2 (log $K/smoke_$2.log)" ;;
smoke)
  $V $PY smoke --tag R8KSM-L18-s0 --ctrl R8KSC-L18-s0 --wjson $DATA_DIR/runs/round7/joint/results/w_R7JNS.json --out $K/results/smoke.json; r=$?
  (( r == 0 )) && rm -rf $RUNS/R8KSM-L18-s0 $RUNS/R8KSC-L18-s0; exit $r ;;               # kept for inspection when a check fails
s2)
  LANE=$S2; LD=$S2; [[ ! -e $S2/DONE ]] || die "s2 already ran"; for X in R6DAG-L18-s2 R8K-L18-s2; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  for f in R6DAG-L18-s2-281ag R8K-L18-s2-281ag; do [[ ! -e $F_/$f.csv ]] || die "$f exists"; done
  t1=$(awk '$2 == "pai_fix-tax-R8K-L18-s1-281agH" {x = $1} END {print x}' $K/jobs.txt); [[ -n $t1 ]] || die "no R8K-s1 ON tax id"
  h=$(git rev-parse --short HEAD); echo "$h s2" >> $S2/code.txt; x=($t1)
  for v in r6:R6DAG k:R8K; do t=$(sub $S2 20 s2-train-${v#*:} "${FT[@]}" --timeout-h 4 -- bash $ME train2 ${v%:*}) || exit 1
    y=($(read281 ${v#*:}-L18-s2 ag $t)) || exit 1; z=($(read281 ${v#*:}-L18-s2 agH $t)) || exit 1; x+=(${y[-1]} ${z[-1]})
    echo "$(date '+%F %T') s2 ${v#*:} (code $h): train $t OFF ${y[*]} ON ${z[*]}" | tee -a $S2/stages.txt; done
  r=$(sub $S2 20 s2-seeds --after $(csv ${x[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME seeds) || exit 1
  echo "$(date '+%F %T') s2: seeds report $r after ${x[*]}" | tee -a $S2/stages.txt ;;
train2) declare -A TG=([r6]=R6DAG [k]=R8K); X=${TG[$2]}-L18-s2; run $2 $N $X 2; echo "$(date '+%F %T') train $X: ${A[*]} --save-at 2500,5000,7500" | tee -a $S2/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
seeds)
  rd=(); for m in ag agH; do for a in R6DAG R8K; do for s in 0 1 2; do rd+=($a-s$s-$m=$F_/$a-L18-s$s-281$m.csv); done; done; done
  $V $PY seeds --read "${rd[@]}" --out $S2/results/seeds3 || die seeds; date '+%F %T' > $S2/DONE ;;
train) s=$2; run k $N R8K-L18-s$s $s; echo "$(date '+%F %T') train s$s: ${A[*]} --save-at 2500,5000,7500" | tee -a $K/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
report)
  PT=$DATA_DIR/tmp/pcs_r8k; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; RES=$K/results
  sj="R3MV-s0+AG=R3MV-L18-s0-281ag.csv,DAG-s0+AG=R6DAG-L18-s0-281ag.csv,DAG-s1+AG=R6DAG-L18-s1-281ag.csv,R7J-s0+AG=R7J-L18-s0-281ag.csv"
  sj="$sj,R7J-s1+AG=R7J-L18-s1-281ag.csv,R8K-s0+AG=R8K-L18-s0-281ag.csv,R8K-s1+AG=R8K-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs R8K-s0+AG:DAG-s0+AG,R8K-s1+AG:DAG-s1+AG > $PT/out/r5.log 2>&1 \
    || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_r8k.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_r8k.json || die "copy ruler report"
  rd=(); for m in ag agH; do rd+=(NTV0-$m=$F_/NTV0-L18-s0-281$m.csv); for a in R6DAG R7J R8K; do for s in 0 1; do rd+=($a-s$s-$m=$F_/$a-L18-s$s-281$m.csv); done; done; done
  pr=(); for s in 0 1; do pr+=(R8K-s$s-ag:R6DAG-s$s-ag R8K-s$s-agH:R6DAG-s$s-agH R8K-s$s-ag:R7J-s$s-ag R8K-s$s-agH:R7J-s$s-agH R8K-s$s-ag:NTV0-ag
      R8K-s$s-agH:NTV0-agH R8K-s$s-agH:R8K-s$s-ag R6DAG-s$s-agH:R6DAG-s$s-ag); done; pr+=(R8K-s1-ag:R8K-s0-ag)
  $TR experiments/round2/scripts/r2.py r281 --read "${rd[@]}" --pairs "${pr[@]}" --out $RES/pai281_R8K.md \
      --title "PAI 281, ACC + GAP: round8 R8K s0 / s1 (R6DAG + keep-pace floor / F2 + Harmonizer twins) vs R6DAG, R7J and NTV0; Harmonizer OFF and ON" || die r281
  $V $PY verdict --json $RES/pai281_R8K.json --md $RES/pai281_R8K.md --pcs $RES/pcs_r8k.json --read "${rd[@]}" || die verdict
  date '+%F %T' > $K/DONE ;;
ruler2)
  LANE=$S2; P0=$DATA_DIR/tmp/pcs_r8k; PT=$DATA_DIR/tmp/pcs_r8k2; [[ ! -e $PT/rows.csv.gz ]] || die "$PT exists"; mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$DATA_DIR/tmp/pcs_r5/site); X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  c=$(sub $S2 20 s2-pcs-collect --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz --only DAG-s2+AG,R8K-s2+AG \
      --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub $S2 20 s2-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  r=$(sub $S2 20 s2-ruler --after $(csv ${b[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME ruler2rep) || exit 1
  echo "$(date '+%F %T') ruler2: collect $c boot ${b[*]} report $r" | tee -a $S2/stages.txt ;;
ruler2rep)
  PT=$DATA_DIR/tmp/pcs_r8k2; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  sj="R3MV-s0+AG=R3MV-L18-s0-281ag.csv"; for s in 0 1 2; do sj="$sj,DAG-s$s+AG=R6DAG-L18-s$s-281ag.csv,R8K-s$s+AG=R8K-L18-s$s-281ag.csv"; done
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs R8K-s0+AG:DAG-s0+AG,R8K-s1+AG:DAG-s1+AG,R8K-s2+AG:DAG-s2+AG \
    > $PT/out/r5.log 2>&1 || die "ruler2 report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $S2/results/pcs_s2.md && cp $PT/out/pai_pcs_local_r5.json $S2/results/pcs_s2.json && date '+%F %T' > $S2/RULER ;;
printargv) argv "${@:2}" ;;
wait) case ${2:-K} in K) waiter $K ${3:-DONE} ;; N) waiter $N_ ${3:-DONE} ;; S) waiter $S2 ${3:-DONE} ;; *) die "wait K|N|S" ;; esac ;;
*) sed -n 2,18p "$0"; exit 2 ;;
esac
