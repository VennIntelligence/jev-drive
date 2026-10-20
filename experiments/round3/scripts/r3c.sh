#!/usr/bin/env bash
# ROUND3c (plans/2026-10-21-round3c-prereg.md): two class-balanced continuations of NTV0-L18-s0 on top of round 3b's MV (R3MV-L18-s0's command,
# only the draw file changed), one seed, 9 500 steps; control = round 2's stored R2C-L18-s0. MVL = MV + navtrain launch rows at draw weight 3
# (draw file R3MVL: R3MV's pool weights + nav_names / nav_w, rm_train renormalises inside navtrain); MVE = MV restricted to PAI human rows with an
# exam event ahead (route turn / stop; R3MVE). No new flag, no new loss (r3.py weights --rule mvl|mve). Then the PAI-FIX native read (vl, "v t")
# of both, the registered lines (r3.py gate --stem gate3c) and, for an arm that passes all three, its PAI 281 ACC + GAP read at PRIO 30 paired
# against the stored NTV0 + ACC + GAP and R2C + ACC + GAP; beside it the inertia gate (report only). Snapshots at 2 500 / 5 000 / 7 500 steps
# (--save-at; tags <arm>x<step>-L18-s0, not read here). Box, repo root; every job through the GPU pool at PRIO 20 (281: 30); ids in $O/jobs.txt.
#   r3c.sh dry           the two training commands and their diffs against round 3b's MV
#   r3c.sh smoke <rev>   MVL and MVE 60 steps each (--diag-every 20, --save-at 20,40) -> r3.py smoke -> results/smoke_{MVL,MVE}.json; + identity of
#                        the trainer change (MV's command, draw R3MV, 60 steps eager: code <rev> vs this code -> df.py bitcmp -> results/identity3c.json)
#   r3c.sh launch        (smokes + identity passed) 2 trainings -> per arm PAI-FIX vl -> report + gate -> per passing arm PAI 281 ag (PRIO 30) -> r281
#                        -> $O/PAIDONE; trainings -> inertia cf -> inertia report -> $O/INDONE; $O/DONE when both exist
#   r3c.sh wait          blocks until $O/DONE or a failed job of this lane (one run_in_background waiter); swait: until every listed job ended
#   job bodies: smoketrain <arm> | identrun <old|new> <rev> | ident <rev> | train <arm> | paifix | r281 <tag> | cf | inreport | fin
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round3c; ME=experiments/round3/scripts/r3c.sh; PY=experiments/round3/scripts/r3.py; R2PY=experiments/round2/scripts/r2.py
RES=experiments/round3/results; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; JT=experiments/defect_fix/scripts/jt.sh
IN=experiments/defect_fix/scripts/inertia.py; LA=experiments/defect_fix/scripts/lon_anchor.py; DFR=experiments/defect_fix/results
RUNS=$DATA_DIR/runs/op_parity/runs; LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2; INIT=NTV0-L18-s0; N=9500; F_=$DATA_DIR/runs/pai_fix/facts; CTRL=R2C-L18-s0
MV=R3MV-L18-s0
declare -A TAG=([MVL]=R3MVL-L18-s0 [MVE]=R3MVE-L18-s0 [MV]=$MV) WF=([MVL]=R3MVL [MVE]=R3MVE [MV]=R3MV)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train); SAVE="--save-at 2500,5000,7500"; PRIO=${PRIO:-20}
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round3c --name "$name" --tries ${TRIES:-1} --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() {  # cmd <arm> <steps> [tag]: round 3b's MV command (round 2's J via jt.sh train DRY, no flattening) from NTV0-L18-s0, the arm's draw file
  local A=$1 S=$2 T=${3:-${TAG[$1]}} C; C=$(DRY=1 ONK=64 ARCROWS=brake bash $JT train $T on1_JTPH $LAT ${WF[$A]} $S 0 | tail -1)
  [[ $C == *"--init RB0-F-s0 "* && $C == *"--on-k 64 "* && $C == *"--warmup 500 "* && $C == *"--steps $S "* && $C == *"w/${WF[$A]}.npz "* \
     && $C == *"--on on1_JTPH "* && $C == *"--on-arc 0 "* && $C == *"--lr-scale 0.1 "* && $C != *"--flat-"* ]] || die "unexpected command: $C"
  echo "${C/--init RB0-F-s0 /--init $INIT }" | sed 's/ *$//'; }
read281() {  # read281 <tag> <prio>: PAI 281 ACC + GAP (NTV0 + ACC + GAP's chunk lists, g10, FAST=1, CONC 4, as rounds 2 / 3) -> "<chunk ids...> <facts job id>"
  local T=$1 R=$O/pai281/$1-ag ids t; mkdir -p $O/pai281
  CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" \
    WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 ALPASIM_FAST=1 PRIO=$2 bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R.submit.txt \
    || die "pai submit $T"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$T: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$2 AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281ag $R | tail -1) && [[ -n $t ]] || die "tax $T"
  echo "$t pai_fix-tax-$T-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
dry)
  for A in MVL MVE; do echo "$A: $(cmd $A $N) $SAVE"; echo "== MV -> $A"; diff <(cmd MV $N | tr ' ' '\n') <(echo "$(cmd $A $N) $SAVE" | tr ' ' '\n'); done; true ;;
smoke)
  rev=${2:?old rev (before the rm_train nav_w change)}
  for A in MVL MVE; do [[ ! -e $RUNS/R3SM$A-L18-s0 ]] || die "R3SM$A-L18-s0 exists"; [[ -s $DATA_DIR/runs/defect_fix/w/${WF[$A]}.npz ]] || die "no ${WF[$A]} draw file"; done
  for X in R3CIDOLD R3CIDNEW; do [[ ! -e $RUNS/$X-L18-s0 ]] || die "$X-L18-s0 exists"; done
  WT=$DATA_DIR/tmp/r3c_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || die "worktree $rev"
  h=$(git rev-parse --short HEAD); echo "$h smoke (old $rev)" >> $O/code.txt
  for A in MVL MVE; do s=$(sub r3c-smoke-$A "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain $A) || exit 1; echo "$(date '+%F %T') smoke $A (code $h): $s" | tee -a $O/stages.txt; done
  a=$(sub r3c-ident-old "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun old $rev) || exit 1
  b=$(sub r3c-ident-new "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun new $rev) || exit 1
  c=$(sub r3c-ident --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident $rev) || exit 1
  echo "$(date '+%F %T') identity (code $h vs $rev): $a $b bitcmp $c" | tee -a $O/stages.txt ;;
smoketrain)
  A=$2 X=R3SM$2-L18-s0; C="$(cmd $A 60 $X) --diag-every 20 --save-at 20,40" || exit 1
  echo "$(date '+%F %T') smoke $A: $C" | tee -a $O/stages.txt
  $C > $O/smoke_$A.log 2>&1 || die "smoke run $A (log $O/smoke_$A.log)"
  $TR $PY smoke --tag $X --arm $A; r=$?
  for k in 20 40; do [[ -s $RUNS/R3SM${A}x$k-L18-s0/ckpt-final.pt ]] || { echo "$(date '+%F %T') ERROR smoke $A: no snapshot x$k" | tee -a $O/stages.txt; r=3; }; done
  rm -rf $RUNS/$X $RUNS/R3SM${A}x20-L18-s0 $RUNS/R3SM${A}x40-L18-s0; exit $r ;;
identrun)  # MV's command (draw R3MV: no nav_w), 60 steps, eager (pool --env JEV_RM_EAGER=1): old = the trainer at <rev>, new = this code
  X=R3CID${2^^}-L18-s0; C="$(cmd MV 60 $X)" || exit 1; D=.; [[ $2 == old ]] && D=$DATA_DIR/tmp/r3c_old_$3
  echo "$(date '+%F %T') identity $2 (in $D): $C" | tee -a $O/stages.txt; (cd $D && $C) > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R3CIDOLD-L18-s0 --b R3CIDNEW-L18-s0 --out round3c_identity_navw \
      --note "MV's command (draw R3MV, no nav_w) 60 steps eager: rm_train at $2 vs with the ROUND3c nav_w draw (off without nav_w)"; r=$?
  cp experiments/defect_fix/results/round3c_identity_navw.json $RES/identity3c.json || r=4
  rm -rf $RUNS/R3CIDOLD-L18-s0 $RUNS/R3CIDNEW-L18-s0; exit $r ;;
launch)
  for A in MVL MVE; do for X in ${TAG[$A]} ${TAG[$A]/-L18-/x2500-L18-} ${TAG[$A]/-L18-/x5000-L18-} ${TAG[$A]/-L18-/x7500-L18-}; do
    [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done; done
  python3 -c "import json, sys; sys.exit(0 if all(json.load(open(f'$RES/smoke_{a}.json'))['pass_'] for a in ('MVL', 'MVE')) and json.load(open('$RES/identity3c.json'))['identical'] else 1)" \
    || die "smoke / identity not passed"
  for f in $F_/$CTRL-vl.csv $F_/$MV-vl.csv $F_/NTV0-L18-s0-281ag.csv $F_/$CTRL-281ag.csv; do [[ -s $f ]] || die "no stored $f"; done
  rm -f $O/DONE $O/PAIDONE $O/INDONE; h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt
  declare -A T_; x=()
  for A in MVL MVE; do T_[$A]=$(sub r3c-train-${TAG[$A]} "${FT[@]}" --timeout-h 4 -- bash $ME train $A) || exit 1
    o=$(PRIO=$PRIO AFTER=${T_[$A]} bash experiments/pai_fix/scripts/pai_fix.sh read ${TAG[$A]} vl "v t" | tail -1) || die "pai_fix read ${TAG[$A]}"
    x+=(${o%% *}); echo "${o%% *} pai_fix-tax-${TAG[$A]}-vl" >> $O/jobs.txt
    awk '{print $1, $2}' $DATA_DIR/runs/pai_fix/reads/${TAG[$A]}-vl/jobs.txt >> $O/jobs.txt; done
  p=$(sub r3c-paifix --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME paifix) || exit 1
  ta=$(csv ${T_[MVL]} ${T_[MVE]})
  f=$(sub r3c-cf --after $ta --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME cf) || exit 1
  r=$(sub r3c-inreport --after $f --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME inreport) || exit 1
  echo "$(date '+%F %T') launch (code $h): train ${T_[MVL]} ${T_[MVE]} PAI-FIX ${x[*]} report + gate $p inertia $f $r" | tee -a $O/stages.txt ;;
train)
  A=$2; CMD="$(cmd $A $N) $SAVE" || exit 1; echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $O/stages.txt; exec $CMD ;;
paifix)  # report against CTRL and NTV0, the registered lines; every passing arm's PAI 281 ACC + GAP read at PRIO 30 -> r281 -> PAIDONE; else PAIDONE
  rd=(NTV0-L18-s0-vl=$F_/NTV0-L18-s0-vl.csv $CTRL-vl=$F_/$CTRL-vl.csv R2J-L18-s0-vl=$F_/R2J-L18-s0-vl.csv $MV-vl=$F_/$MV-vl.csv)
  for A in MVL MVE; do rd+=(${TAG[$A]}-vl=$F_/${TAG[$A]}-vl.csv); done
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $CTRL-vl --out $RES/paifix3c_vl_vs_ctrl.md --scores $RES/paifix3c_vl.csv > /dev/null \
    || die "pai_fix report vs CTRL"
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $MV-vl --out $RES/paifix3c_vl_vs_mv.md > /dev/null || die "pai_fix report vs MV"
  ps=$($V $PY gate --arms ${TAG[MVL]} ${TAG[MVE]} $MV --ctrl $CTRL --stem gate3c) || die "gate"
  ps=$(tr ' ' '\n' <<< "$ps" | grep -v "^$MV\$" | xargs)                    # MV is listed for reference only (stored, failed in 3b)
  echo "$(date '+%F %T') gate: passing [${ps}]" | tee -a $O/stages.txt
  [[ -z $ps ]] && { date '+%F %T' > $O/PAIDONE; bash $ME fin; exit 0; }
  r=(); for T in $ps; do y=($(read281 $T 30)) || exit 1
    r+=($(PRIO=30 sub r3c-r281-$T --after ${y[-1]} --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281 $T)) || exit 1
    echo "$(date '+%F %T') PAI 281 ag $T: chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]} report ${r[-1]}" | tee -a $O/stages.txt; done
  PRIO=30 sub r3c-paidone --after $(csv ${r[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/PAIDONE && bash $ME fin" > /dev/null ;;
r281)
  T=$2; $TR $R2PY r281 --read NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R2C-ag=$F_/$CTRL-281ag.csv R2J-ag=$F_/R2J-L18-s0-281ag.csv $T-ag=$F_/$T-281ag.csv \
      --pairs $T-ag:NTV0-ag $T-ag:R2C-ag $T-ag:R2J-ag --out $RES/pai281_${T%-L18-s0}_ag.md \
      --title "PAI 281, ACC + GAP: $T vs stored NTV0-L18-s0 and R2C-L18-s0 (and R2J)" || die "r281 $T" ;;
cf)     $TR $IN cf --tags NTV0-L18-s0,$CTRL,$MV,${TAG[MVL]},${TAG[MVE]},RSG-L18-s0 --gate-tag RSG-L18-s0 --name round3c ;;
inreport)
  $TR $IN report --name round3c > $O/inertia_report.log 2>&1 || die "inertia report"
  cp $DFR/inertia/report_round3c.md $DFR/inertia/report_round3c.json $RES/ || die "no inertia report"
  $TR $LA inertia --name round3c --ctl $CTRL --arms ${TAG[MVL]} ${TAG[MVE]} $MV --out-stem round3c_inertia || die "inertia read-out"
  cp $DFR/lon_anchor/round3c_inertia.md $RES/inertia3c.md && cp $DFR/lon_anchor/round3c_inertia.json $RES/inertia3c.json || die "copy inertia"
  date '+%F %T' > $O/INDONE; bash $ME fin ;;
fin)  # DONE once both branches are done (whichever finishes second writes it)
  [[ -f $O/PAIDONE && -f $O/INDONE ]] && date '+%F %T' > $O/DONE; true ;;
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
swait)  # blocks until every job in $O/jobs.txt is finished (done / failed / cancelled): the smoke stage's waiter
  until $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
st = {n: jobs.get(i, {}).get("state", "gone") for n, i in last.items()}
open_ = [n for n, s in st.items() if s not in ("done", "failed", "cancelled", "gone")]
print(st if not open_ else "", file=sys.stderr)
sys.exit(1 if open_ else 0)
PY
  do sleep 120; done; true ;;
*) sed -n 2,16p "$0"; exit 2 ;;
esac
