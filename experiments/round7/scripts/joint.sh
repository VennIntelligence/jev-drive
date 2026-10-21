#!/usr/bin/env bash
# ROUND7 joint arm (plans/2026-10-22-joint-arm.md): R6DAG's recipe + (1) F2 and the keep-the-human's-pace DAgger labels (teacher b5), (2) S-MIX,
# (3) the command from the route's heading change (training rows: rm_train --cmd-rule heading; serve: JEV_CMD=heading). Lane code joint.py.
# Box, repo root; every job through the pool at PRIO 20; ids in $O/jobs.txt, stages in $O/stages.txt, a stop reason in $O/VERDICT.
#   joint.sh gates     cmdcheck (G2) | teach b5 -> label (G1, pools on1_JTPHDK / on1_JTPHDI) -> draw files R7J0 / R7J / R7JNS -> $O/GATES
#   joint.sh prep      identity (G4: R6DAG argv at the pre-change commit vs the joint code with every switch off), the two 300-step smokes (G3 / G5),
#                      the serve gates (chunk c000 of R6DAG-L18-s1: switch off vs the stored read, switch on) -> $O/PREP
#   joint.sh launch    R7J-L18-s0 / s1 (from NTV0-L18-s0, 9 500 steps) -> PAI 281 ACC + GAP + JEV_CMD, Harmonizer OFF and ON -> r281, verdict, ruler -> $O/DONE
#   joint.sh wait [M]  one run_in_background waiter: $O/VERDICT, $O/<M> or a failed job of this lane
#   joint.sh printargv <variant> <steps> <tag> <seed>   r6 = R6DAG's argv; id = the joint code with every switch off; j = the joint arm
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round7/joint; ME=experiments/round7/scripts/joint.sh; PY=experiments/round7/scripts/joint.py; DS=experiments/round6/scripts/dag.sh
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; RES=$O/results; PRIO=${PRIO:-20}; N=9500   # RES outside the repo tree (pulls never blocked)
PRE=${JOINT_PRE:-ba54017b}                                         # the last commit before this lane's code (identity run A)
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
J_POOL=${J_POOL:-on1_JTPHDK}; J_WF=${J_WF:-R7J}; J_K=${J_K:-98}; J_HOW=${J_HOW:-1.3}; J_CMD=${J_CMD-heading}   # fallbacks (a change dropped) override these
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round7-joint --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name --env J_POOL=$J_POOL --env J_WF=$J_WF --env J_K=$J_K --env J_HOW=$J_HOW --env J_CMD=$J_CMD "$@" < /dev/null | awk '{print $NF}') \
     && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
argv() {  # argv <variant> <steps> <tag> <seed>: one argument a line
  local X; X=$(bash $DS printargv dag "$2" "$3" "$4") || die "R6DAG argv"
  ARGV_X="$X" python3 - "$1" "$O/w" "$J_POOL" "$J_WF" "$J_K" "$J_HOW" "$J_CMD" <<'PY'
import os, sys
var, w, pool, wf, k, how, cmd = sys.argv[1:8]
a = os.environ["ARGV_X"].split("\n")
a = [x for x in a if x != ""]
last = lambda q: len(a) - 1 - a[::-1].index(q)  # noqa: E731
assert a[last("--on") + 1] == "on1_JTPHD" and a[last("--on-wfile") + 1].endswith("/round6/dag/w/R6DAG.npz") and a[last("--on-k") + 1] == "72"
assert a[a.index("--ho-w") + 1] == "3" and a.count("--ho-w") == 1 and a[a.index("--init") + 1] == "NTV0-L18-s0" and "--cmd-rule" not in a
if var == "id":
    a[last("--on") + 1], a[last("--on-wfile") + 1] = "on1_JTPHDI", f"{w}/R7J0.npz"
elif var == "j":
    a[last("--on") + 1], a[last("--on-wfile") + 1], a[last("--on-k") + 1], a[a.index("--ho-w") + 1] = pool, f"{w}/{wf}.npz", k, how
    if cmd:
        a += ["--cmd-rule", cmd]
else:
    assert var == "r6", var
print("\n".join(a))
PY
}
run() { local X; X=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$X"; }
read281() {  # read281 <tag> <suffix ag|agH> <after> -> "<chunk ids...> <tax id>"
  local R=$O/pai281/$1-$2 ids t sv="$SERVE${J_CMD:+ JEV_CMD=$J_CMD}" ex=(); mkdir -p $O/pai281
  [[ $2 == agH ]] && { sv="$sv HARMONIZER=1"; ex=(VRAM=30 RAM=40); }
  env "${ex[@]}" AFTER=$3 CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO \
    bash experiments/alpasim/scripts/pai_box.sh submit $R $1 > $R.submit.txt || die "pai submit $1 $2"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1 $2: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt | sed "s/\$/-$2/" >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281$2 $R | tail -1) && [[ -n $t ]] || die "tax $1 $2"
  echo "$t pai_fix-tax-$1-281$2" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
gates)
  [[ ! -e $O/GATES ]] || die "gates already ran"
  h=$(git rev-parse --short HEAD); echo "$h gates" >> $O/code.txt
  c=$(sub joint-cmdcheck --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $V $PY cmdcheck) || exit 1
  t=$(sub joint-teach --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- "${EA[@]}" $AP $PY teach --workers 32) || exit 1
  l=$(sub joint-label --after $t --vram 0.1 --cpu 4 --ram 96 --timeout-h 2 -- $V $PY label) || exit 1
  w=$(sub joint-weights --after $l --vram 0.1 --cpu 4 --ram 64 --timeout-h 1 -- bash $ME weights) || exit 1
  d=$(sub joint-gates-done --after $c,$w --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/GATES") || exit 1
  echo "$(date '+%F %T') gates (code $h): cmdcheck $c teach $t label $l weights $w done $d" | tee -a $O/stages.txt ;;
weights)
  $V $PY weights --out R7J0 --pool on1_JTPHDI || die R7J0
  $V $PY weights --out R7J --pool on1_JTPHDK --f2 1 --smix 1 || die R7J
  $V $PY weights --out R7JNS --pool on1_JTPHDK --f2 1 || die "R7JNS (fallback without S-MIX)" ;;
prep)
  [[ -f $O/GATES && ! -f $O/VERDICT && ! -e $O/PREP ]] || die "gates not done (or VERDICT / PREP)"
  h=$(git rev-parse --short HEAD); echo "$h prep" >> $O/code.txt
  ia=$(sub joint-ident-a "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun a) || exit 1
  ib=$(sub joint-ident-b "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun b) || exit 1
  ic=$(sub joint-ident --after $ia,$ib --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  sj=$(sub joint-smoke-j "${FT[@]}" --timeout-h 1 -- bash $ME smokerun j) || exit 1
  sc=$(sub joint-smoke-c "${FT[@]}" --timeout-h 1 -- bash $ME smokerun r6) || exit 1
  sk=$(sub joint-smoke --after $sj,$sc --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME smoke) || exit 1
  mkdir -p $O/serve/chunks && cp $DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks/c000.txt $O/serve/chunks/
  x=()
  for v in off on; do sv="$SERVE"; [[ $v == on ]] && sv="$SERVE JEV_CMD=heading"
    CHUNKS_FROM=$O/serve/chunks SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/serve/$v R6DAG-L18-s1 \
      > $O/serve/$v.submit.txt || die "serve $v"
    i=$(awk '$2 ~ /^pai-/ {print $1}' $O/serve/$v.submit.txt); [[ $(wc -w <<< "$i") == 1 ]] || die "serve $v: not 1 chunk"; echo "$i pai-serve-$v" >> $O/jobs.txt; x+=($i); done
  sg=$(sub joint-serve --after $(csv ${x[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY serve --ref $DATA_DIR/runs/round6/dag/pai281/R6DAG-L18-s1-ag \
       --ident $O/serve/off --on $O/serve/on) || exit 1
  d=$(sub joint-prep-done --after $ic,$sk,$sg --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/PREP") || exit 1
  echo "$(date '+%F %T') prep (code $h): identity $ia $ib $ic smoke $sj $sc $sk serve ${x[*]} $sg done $d" | tee -a $O/stages.txt ;;
identrun)  # identrun a | b: 60 steps eager; a = R6DAG's argv run from a worktree at $PRE, b = the joint code (HEAD) with every switch off
  if [[ $2 == a ]]; then X=R7IDA-L18-s0; run r6 60 $X 0; WT=$DATA_DIR/tmp/joint_wt_$PRE
    [[ -d $WT ]] || git worktree add --detach $WT $PRE || die "worktree $PRE"
    [[ $(git -C $WT rev-parse --short HEAD) == ${PRE:0:8} ]] || die "worktree is not at $PRE"; cd $WT
  else X=R7IDB-L18-s0; run id 60 $X 0; fi
  [[ ! -e $RUNS/$X ]] || die "$X exists"
  echo "$(date '+%F %T') identity $2 ($(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R7IDA-L18-s0 --b R7IDB-L18-s0 --out round7_joint_identity \
      --note "R6DAG-L18-s0 argv 60 steps eager: code $PRE on on1_JTPHD / R6DAG vs HEAD with every joint switch off on on1_JTPHDI / R7J0"; r=$?
  cp experiments/defect_fix/results/round7_joint_identity.json $RES/identity.json || r=4
  rm -rf $RUNS/R7IDA-L18-s0 $RUNS/R7IDB-L18-s0; exit $r ;;
smokerun)  # smokerun j | r6: 300 steps, diag every 20, dev every 100 (j = the joint arm, r6 = R6DAG's argv, the gradient-ratio control)
  declare -A SX=([j]=R7JSM [r6]=R7JSC); X=${SX[$2]}-L18-s0; [[ ! -e $RUNS/$X ]] || die "$X exists"; run $2 300 $X 0
  echo "$(date '+%F %T') smoke $2: ${A[*]} --diag-every 20 --eval-every 100" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --eval-every 100 > $O/smoke_$2.log 2>&1 || die "smoke $2 (log $O/smoke_$2.log)" ;;
smoke)
  $V $PY smoke --tag R7JSM-L18-s0 --ctrl R7JSC-L18-s0 --wfile $J_WF --nav $((136 - 13 - J_K)) --cmd-on $([[ -n $J_CMD ]] && echo 1 || echo 0); r=$?
  (( r == 0 )) && rm -rf $RUNS/R7JSM-L18-s0 $RUNS/R7JSC-L18-s0; exit $r ;;                  # kept for inspection when a check fails
resmoke)  # the smokes and their check again (after a smoke-check fix), then PREP after them and the serve gate already submitted
  [[ -f $O/GATES && ! -f $O/PREP ]] || die "gates not done (or PREP)"; h=$(git rev-parse --short HEAD); echo "$h resmoke" >> $O/code.txt
  rm -rf $RUNS/R7JSM-L18-s0 $RUNS/R7JSC-L18-s0
  sj=$(sub joint-smoke-j "${FT[@]}" --timeout-h 1 -- bash $ME smokerun j) || exit 1
  sc=$(sub joint-smoke-c "${FT[@]}" --timeout-h 1 -- bash $ME smokerun r6) || exit 1
  sk=$(sub joint-smoke --after $sj,$sc --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME smoke) || exit 1
  sg=$(awk '$2 == "joint-serve" {x = $1} END {print x}' $O/jobs.txt); ic=$(awk '$2 == "joint-ident" {x = $1} END {print x}' $O/jobs.txt)
  d=$(sub joint-prep-done --after $ic,$sk,$sg --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/PREP") || exit 1
  echo "$(date '+%F %T') resmoke (code $h): smoke $sj $sc $sk done $d (after identity $ic serve $sg)" | tee -a $O/stages.txt ;;
launch)
  [[ -f $O/PREP && ! -f $O/VERDICT ]] || die "prep not done (or VERDICT)"
  for s in 0 1; do for X in R7J-L18-s$s R7Jx2500-L18-s$s R7Jx5000-L18-s$s R7Jx7500-L18-s$s; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done; done
  for f in NTV0-L18-s0-281ag R6DAG-L18-s0-281ag R6DAG-L18-s1-281ag R6DAG-L18-s1-281agH NTV0-L18-s0-281agH; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt; x=()
  for s in 0 1; do t=$(sub joint-train-s$s "${FT[@]}" --timeout-h 4 -- bash $ME train $s) || exit 1
    y=($(read281 R7J-L18-s$s ag $t)) || exit 1; z=($(read281 R7J-L18-s$s agH $t)) || exit 1; x+=(${y[-1]} ${z[-1]})
    echo "$(date '+%F %T') launch s$s (code $h): train $t OFF ${y[*]} ON ${z[*]}" | tee -a $O/stages.txt; done
  P0=$DATA_DIR/tmp/pcs_r6d; PT=$DATA_DIR/tmp/pcs_r7j; [[ ! -e $PT ]] || die "$PT exists"; for f in $P0/rows.csv.gz $P0/out/fits/paimix.npz; do [[ -s $f ]] || die "no $f"; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$DATA_DIR/tmp/pcs_r5/site)
  off=$(csv ${x[0]} ${x[2]})
  c=$(sub joint-pcs-collect --after $off --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only R7J-s0+AG,R7J-s1+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub joint-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  r=$(sub joint-report --after $(csv ${x[@]} ${b[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME report) || exit 1
  echo "$(date '+%F %T') launch: ruler collect $c boot ${b[*]} report $r" | tee -a $O/stages.txt ;;
train) s=$2; run j $N R7J-L18-s$s $s; echo "$(date '+%F %T') train s$s: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
report)
  P0=$DATA_DIR/tmp/pcs_r6d; PT=$DATA_DIR/tmp/pcs_r7j; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  sj="R3MV-s0+AG=R3MV-L18-s0-281ag.csv,DAG-s0+AG=R6DAG-L18-s0-281ag.csv,DAG-s1+AG=R6DAG-L18-s1-281ag.csv,R7J-s0+AG=R7J-L18-s0-281ag.csv,R7J-s1+AG=R7J-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs R7J-s0+AG:DAG-s0+AG,R7J-s1+AG:DAG-s1+AG > $PT/out/r5.log 2>&1 \
    || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_joint.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_joint.json || die "copy ruler report"
  rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R6DAG-s0-ag=$F_/R6DAG-L18-s0-281ag.csv R6DAG-s1-ag=$F_/R6DAG-L18-s1-281ag.csv
      R7J-s0-ag=$F_/R7J-L18-s0-281ag.csv R7J-s1-ag=$F_/R7J-L18-s1-281ag.csv NTV0-agH=$F_/NTV0-L18-s0-281agH.csv R6DAG-s1-agH=$F_/R6DAG-L18-s1-281agH.csv
      R7J-s0-agH=$F_/R7J-L18-s0-281agH.csv R7J-s1-agH=$F_/R7J-L18-s1-281agH.csv)
  pr=(); [[ -s $F_/R6DAG-L18-s0-281agH.csv ]] && { rd+=(R6DAG-s0-agH=$F_/R6DAG-L18-s0-281agH.csv); pr=(R7J-s0-agH:R6DAG-s0-agH R6DAG-s0-agH:R6DAG-s0-ag); }   # main session's same-seed ON reference
  $TR experiments/round2/scripts/r2.py r281 --read "${rd[@]}" --pairs R7J-s0-ag:R6DAG-s0-ag R7J-s1-ag:R6DAG-s1-ag R7J-s0-ag:NTV0-ag R7J-s1-ag:NTV0-ag \
      R7J-s0-agH:R6DAG-s1-agH R7J-s1-agH:R6DAG-s1-agH R7J-s0-agH:NTV0-agH R7J-s1-agH:NTV0-agH R7J-s0-agH:R7J-s0-ag R7J-s1-agH:R7J-s1-ag R7J-s1-ag:R7J-s0-ag "${pr[@]}" \
      --out $RES/pai281_R7J.md --title "PAI 281, ACC + GAP (+ JEV_CMD=heading for R7J): round7 joint arm R7J s0 / s1 vs stored R6DAG s0 / s1 and NTV0; Harmonizer OFF and ON" \
      || die r281
  $V $PY verdict --json $RES/pai281_R7J.json --md $RES/pai281_R7J.md --pcs $RES/pcs_joint.json --read "${rd[@]}" || die verdict
  date '+%F %T' > $O/DONE ;;
printargv) argv "${@:2}" ;;
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
  cat $O/VERDICT 2>/dev/null; for f in GATES PREP DONE; do [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; done; tail -3 $O/stages.txt; true ;;
*) sed -n 2,11p "$0"; exit 2 ;;
esac
