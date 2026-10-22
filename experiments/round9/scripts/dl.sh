#!/usr/bin/env bash
# ROUND9 DAG-LANE (plans/2026-10-22-dag-lane.md): R6DAG's recipe + (a) the heading command (rm_train --cmd-rule heading; serve JEV_CMD=heading)
# + (b) the DAgger teacher's human-lane rule (dag.py DAG_TV=b3L) on the iteration-1 states, pool on1_JTPHDL, draw file R6DAG.npz. Lane code dl.py.
# Box, repo root; every job through the pool at PRIO 20; ids in $O/jobs.txt, stages in $O/stages.txt; results in $O/results (copied to the Mac).
#   dl.sh all         the whole chain in one submit: teach c000 -> stage-1 check -> teach all -> label (G-L0 / G-L1 / G-L4, pool) | T1 / T2 with b3
#                     and b3L -> tcmp (G-L2 / G-L3) | identity A / B -> bitcmp (G0) | smokes -> check (G-S) -> GATES -> R9DL-L18-s0 / s1 ->
#                     PAI 281 OFF + ON (JEV_CMD=heading) -> cmdagree (G-C) -> ruler -> report -> DONE. A failed gate job fails everything after it.
#   dl.sh wait [M]    one run_in_background waiter: $O/<M> (default DONE) or a failed / cancelled job of this lane
#   dl.sh argv <variant> <steps> <tag> <seed>   r6 = R6DAG-L18-s0's recorded argv; dl = the arm (pool on1_JTPHDL, --cmd-rule heading)
#   job bodies: teach <list|all> | tcheck | label | t1 <tv> | t2 <tv> | tcmp | identrun a|b | ident | smokerun dl|r6 | smoke | train <s> | cmdagree <s> | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/dag_lane; ME=experiments/round9/scripts/dl.sh; PY=experiments/round9/scripts/dl.py; DG=experiments/round6/scripts/dag.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; RES=$O/results; PRIO=${PRIO:-20}; N=9500
META=$DATA_DIR/runs/route_mem/train-R6DAG-L18-s0/20261021-134807/meta.json; PRE=4d8d1a33; WT=$DATA_DIR/tmp/dl_wt_$PRE
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_CMD=heading"; WIZ=controller.gains.rel_front_steering_angle_weight=10
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
TCFG="$DATA_DIR/runs/round5/s3/roll/A/runs/*/sim/generated-user-config-0.yaml,$DATA_DIR/runs/round5/d3/roll/A/runs/*/sim/generated-user-config-0.yaml"
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD DL_RES=$RES)
FT=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round9-dl --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
argv() {  # argv <variant> <steps> <tag> <seed>: R6DAG-L18-s0's recorded argv (its meta.json), one argument a line
  python3 - "$META" "$@" <<'PY'
import json, sys
meta, var, steps, tag, seed = sys.argv[1:6]
a = json.load(open(meta))["argv"]
assert a[0].endswith("rm_train.py") and a[1] == "train" and json.load(open(meta))["git"]["sha"].startswith("4d8d1a33")
i = a.index("--save-at"); del a[i:i + 2]
last = lambda q: len(a) - 1 - a[::-1].index(q)  # noqa: E731
assert a[last("--on") + 1] == "on1_JTPHD" and a[last("--on-wfile") + 1].endswith("/round6/dag/w/R6DAG.npz") and a[a.index("--init") + 1] == "NTV0-L18-s0"
assert "--cmd-rule" not in a and a.count("--steps") == 1 and a.count("--seed") == 1 and a.count("--tag") == 1
a[a.index("--tag") + 1], a[a.index("--steps") + 1], a[a.index("--seed") + 1] = tag, steps, seed
if var == "dl":
    a[last("--on") + 1] = "on1_JTPHDL"
    a += ["--cmd-rule", "heading"]
else:
    assert var == "r6", var
print("\n".join(a))
PY
}
run() { local X; X=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$X"; }
read281() {  # read281 <tag> <ag|agH> <after> -> "<chunk ids...> <tax id>" (NTV0's chunk table, ACC + GAP + JEV_CMD, g10, FAST, CONC 4; agH: HARMONIZER=1)
  local R=$O/pai281/$1-$2 ids t sv="$SERVE" ex=(); mkdir -p $O/pai281; [[ ! -e $R ]] || die "$R exists"
  [[ $2 == agH ]] && { sv="$sv HARMONIZER=1"; ex=(VRAM=30 RAM=40); }
  env "${ex[@]}" AFTER=$3 CHUNKS_FROM=$CH SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1 $2"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1 $2: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt | sed "s/\$/-$2/" >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281$2 $R | tail -1) && [[ -n $t ]] || die "tax $1 $2"
  echo "$t pai_fix-tax-$1-281$2" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
all)
  [[ ! -e $O/GATES && ! -e $O/DONE ]] || die "already ran"
  for s in 0 1; do [[ ! -e $RUNS/R9DL-L18-s$s ]] || die "R9DL-L18-s$s exists"; done
  [[ ! -e $DATA_DIR/runs/op_parity/cache/on1_JTPHDL ]] || die "on1_JTPHDL exists"
  for f in NTV0-L18-s0-281ag NTV0-L18-s0-281agH R6DAG-L18-s0-281ag R6DAG-L18-s1-281ag R6DAG-L18-s2-281ag R6DAG-L18-s0-281agH R6DAG-L18-s1-281agH \
           R6DAG-L18-s2-281agH; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  P0=$DATA_DIR/tmp/pcs_r8k2; PT=$DATA_DIR/tmp/pcs_r9dl; [[ ! -e $PT ]] || die "$PT exists"; [[ -s $P0/rows.csv.gz ]] || die "no $P0/rows.csv.gz"
  [[ -d $WT ]] || git worktree add --detach $WT $PRE > /dev/null 2>&1 || die "worktree $PRE"
  [[ $(git -C $WT rev-parse --short=8 HEAD) == $PRE ]] || die "worktree $WT is not at $PRE"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  mkdir -p $O/lists && cp $DATA_DIR/runs/round6/dag/lists/c000.txt $O/lists/c000.txt
  CP=(--vram 0.1 --ram 96 --timeout-h 3)
  t0=$(sub dl-teach-c000 "${CP[@]}" --cpu 6 -- bash $ME teach $O/lists/c000.txt 6) || exit 1
  k0=$(sub dl-tcheck --after $t0 --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME tcheck) || exit 1
  ta=$(sub dl-teach-all --after $k0 "${CP[@]}" --cpu 48 -- bash $ME teach all 48) || exit 1
  lb=$(sub dl-label --after $ta --vram 0.1 --cpu 4 --ram 160 --timeout-h 2 -- bash $ME label) || exit 1
  T=(); for tv in b3 b3L; do T+=($(sub dl-t1-$tv --after $k0 "${CP[@]}" --cpu 24 -- bash $ME t1 $tv)) || exit 1
    T+=($(sub dl-t2-$tv --after $k0 "${CP[@]}" --cpu 8 -- bash $ME t2 $tv)) || exit 1; done
  tc=$(sub dl-tcmp --after $(csv ${T[@]}) --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME tcmp) || exit 1
  ia=$(sub dl-ident-a "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun a) || exit 1
  ib=$(sub dl-ident-b "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun b) || exit 1
  ic=$(sub dl-ident --after $ia,$ib --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  sd=$(sub dl-smoke-dl --after $lb "${FT[@]}" --timeout-h 1 -- bash $ME smokerun dl) || exit 1
  sc=$(sub dl-smoke-r6 "${FT[@]}" --timeout-h 1 -- bash $ME smokerun r6) || exit 1
  sk=$(sub dl-smoke --after $sd,$sc --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME smoke) || exit 1
  g=$(sub dl-gates-done --after $lb,$tc,$ic,$sk --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/GATES") || exit 1
  echo "$(date '+%F %T') gates (code $h): teach $t0 check $k0 all $ta label $lb t1/t2 ${T[*]} tcmp $tc identity $ia $ib $ic smoke $sd $sc $sk GATES $g" \
    | tee -a $O/stages.txt
  x=(); off=(); ca=()
  for s in 0 1; do t=$(sub dl-train-s$s --after $g "${FT[@]}" --timeout-h 4 -- bash $ME train $s) || exit 1
    y=($(read281 R9DL-L18-s$s ag $t)) || exit 1; z=($(read281 R9DL-L18-s$s agH $t)) || exit 1; x+=(${y[-1]} ${z[-1]}); off+=(${y[-1]})
    ca+=($(sub dl-cmdagree-s$s --after ${y[-1]},${z[-1]} --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME cmdagree $s)) || exit 1
    echo "$(date '+%F %T') launch s$s (code $h): train $t OFF ${y[*]} ON ${z[*]} cmdagree ${ca[-1]}" | tee -a $O/stages.txt; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$DATA_DIR/tmp/pcs_r5/site)
  c=$(sub dl-pcs-collect --after $(csv ${off[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only R9DL-s0+AG,R9DL-s1+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub dl-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  r=$(sub dl-report --after $(csv ${x[@]} ${ca[@]} ${b[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME report) || exit 1
  echo "$(date '+%F %T') launch: ruler collect $c boot ${b[*]} report $r" | tee -a $O/stages.txt ;;
teach)  # teach <list file | all> <workers>: AlpaSim venv, teacher b3L (+ b3 alongside)
  sc=(); [[ $2 != all ]] && sc=(--scenes $2)
  "${EA[@]}" DAG_TV=b3L $AP $PY teach "${sc[@]}" --workers $3 ;;
tcheck) "${EA[@]}" $V $PY tcheck --scenes $O/lists/c000.txt ;;
label) "${EA[@]}" $V $PY label ;;
t1)  # t1 <b3|b3L>: dag.py t1 (human states) with that teacher, results in $O/t_<tv>
  mkdir -p $O/t_$2 && "${EA[@]}" DAG_TV=$2 DAG_RES=$O/t_$2 DAG_O=$O/t_$2 $AP $DG t1 --cfg-glob "$TCFG" --workers 24 ;;
t2) mkdir -p $O/t_$2 && "${EA[@]}" DAG_TV=$2 DAG_RES=$O/t_$2 DAG_O=$O/t_$2 $AP $DG t2 --workers 8 ;;
tcmp) "${EA[@]}" $V $PY tcmp --b3 $O/t_b3 --b3l $O/t_b3L ;;
identrun)  # identrun a | b: 60 steps eager; a = R6DAG-L18-s0's argv run from the worktree at $PRE, b = HEAD
  if [[ $2 == a ]]; then X=R9IDA-L18-s0; run r6 60 $X 0; cd $WT; else X=R9IDB-L18-s0; run r6 60 $X 0; fi
  [[ ! -e $RUNS/$X ]] || die "$X exists"
  echo "$(date '+%F %T') identity $2 ($(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity $2 (log $O/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R9IDA-L18-s0 --b R9IDB-L18-s0 --out round9_dl_identity \
      --note "R6DAG-L18-s0 argv 60 steps eager: code $PRE (R6DAG's) vs HEAD (dag_lane + window layout code, every switch off)"; r=$?
  mv experiments/defect_fix/results/round9_dl_identity.json $RES/identity.json || r=4
  rm -rf $RUNS/R9IDA-L18-s0 $RUNS/R9IDB-L18-s0; exit $r ;;
smokerun)  # smokerun dl | r6: 300 steps, diag every 20, dev every 100
  declare -A SX=([dl]=R9DLSM [r6]=R9DLSC); X=${SX[$2]}-L18-s0; [[ ! -e $RUNS/$X ]] || die "$X exists"; run $2 300 $X 0
  echo "$(date '+%F %T') smoke $2: ${A[*]} --diag-every 20 --eval-every 100" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --eval-every 100 > $O/smoke_$2.log 2>&1 || die "smoke $2 (log $O/smoke_$2.log)" ;;
smoke)
  "${EA[@]}" $V $PY smoke --tag R9DLSM-L18-s0 --ctrl R9DLSC-L18-s0; r=$?
  (( r == 0 )) && rm -rf $RUNS/R9DLSM-L18-s0 $RUNS/R9DLSC-L18-s0; exit $r ;;
train) s=$2; run dl $N R9DL-L18-s$s $s; echo "$(date '+%F %T') train s$s: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
cmdagree) "${EA[@]}" $V $PY cmdagree --label s$2 --read $O/pai281/R9DL-L18-s$2-ag $O/pai281/R9DL-L18-s$2-agH ;;
report)
  PT=$DATA_DIR/tmp/pcs_r9dl; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  sj="R3MV-s0+AG=R3MV-L18-s0-281ag.csv"; for s in 0 1 2; do sj="$sj,DAG-s$s+AG=R6DAG-L18-s$s-281ag.csv"; done
  sj="$sj,R9DL-s0+AG=R9DL-L18-s0-281ag.csv,R9DL-s1+AG=R9DL-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs R9DL-s0+AG:DAG-s0+AG,R9DL-s1+AG:DAG-s1+AG > $PT/out/r5.log 2>&1 \
    || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_dl.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_dl.json || die "copy ruler report"
  rd=(); for m in ag agH; do rd+=(NTV0-$m=$F_/NTV0-L18-s0-281$m.csv); for s in 0 1 2; do rd+=(R6DAG-s$s-$m=$F_/R6DAG-L18-s$s-281$m.csv); done
    for s in 0 1; do rd+=(R9DL-s$s-$m=$F_/R9DL-L18-s$s-281$m.csv); done; done
  pr=(); for s in 0 1; do pr+=(R9DL-s$s-ag:R6DAG-s$s-ag R9DL-s$s-agH:R6DAG-s$s-agH R9DL-s$s-ag:R6DAG-s2-ag R9DL-s$s-agH:R6DAG-s2-agH R9DL-s$s-ag:NTV0-ag
      R9DL-s$s-agH:NTV0-agH R9DL-s$s-agH:R9DL-s$s-ag); done; pr+=(R9DL-s1-ag:R9DL-s0-ag)
  $TR experiments/round2/scripts/r2.py r281 --read "${rd[@]}" --pairs "${pr[@]}" --out $RES/pai281_R9DL.md \
      --title "PAI 281, ACC + GAP (+ JEV_CMD=heading for R9DL): round9 DAG-LANE R9DL s0 / s1 vs R6DAG s0 / s1 / s2 and NTV0; Harmonizer OFF and ON" || die r281
  "${EA[@]}" $V $PY verdict --read "${rd[@]}" || die verdict
  date '+%F %T' > $O/DONE ;;
argv) argv "${@:2}" ;;
wait)
  M=${2:-DONE}
  until [[ -f $O/$M ]] || ! $V - $O/jobs.txt <<'PY'
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
  do sleep 600; done
  for f in GATES DONE; do [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; done; tail -3 $O/stages.txt; true ;;
*) sed -n 2,11p "$0"; exit 2 ;;
esac
