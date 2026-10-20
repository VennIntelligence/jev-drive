#!/usr/bin/env bash
# ROUND3 (plans/2026-10-21-round3-prereg.md): two continuations of NTV0-L18-s0 with round 2's J command, one seed, 9 500 steps; control = round 2's
# stored R2C-L18-s0. J-W = J with every PAI human row whose own 4 s label stands throughout at draw weight 0 (draw file R3JW, r3.py weights; no
# new flag, no new loss); J-NF = J without the onset flattening (R2JF, no --flat-brk / --flat-p). Then the PAI-FIX native read (vl, "v t") of
# both, the registered lines (r3.py gate) and, for an arm that passes all three, its PAI 281 ACC + GAP read at the top priority, paired against
# the stored NTV0-L18-s0 + ACC + GAP. Box, repo root; every job through the GPU pool; ids in $O/jobs.txt.
#   r3.sh dry            the two training commands and their diffs against round 2's J
#   r3.sh smoke          J-W and J-NF 60 steps each (--diag-every 20) -> r3.py smoke -> results/smoke_{JW,JNF}.json; smoke checkpoints removed
#   r3.sh launch         (smoke passed) 2 trainings -> per arm PAI-FIX vl -> report + gate -> per passing arm PAI 281 ag (PRIO 30) -> r281 -> $O/DONE
#   r3.sh wait           blocks until $O/DONE or a failed job of this lane (one run_in_background waiter)
#   job bodies: smoketrain <arm> | train <arm> | paifix | r281 <arm>
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round3; ME=experiments/round3/scripts/r3.sh; PY=experiments/round3/scripts/r3.py; R2PY=experiments/round2/scripts/r2.py
RES=experiments/round3/results; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; JT=experiments/defect_fix/scripts/jt.sh
RUNS=$DATA_DIR/runs/op_parity/runs; LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2; FLATG=$DATA_DIR/runs/defect_fix/simp/flat_onset.txt
INIT=NTV0-L18-s0; N=9500; F_=$DATA_DIR/runs/pai_fix/facts; CTRL=R2C-L18-s0
declare -A TAG=([JW]=R3JW-L18-s0 [JNF]=R3JNF-L18-s0) WF=([JW]=R3JW [JNF]=R2JF) FL=([JW]="--flat-brk $FLATG --flat-p 0.5" [JNF]="")
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round3 --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() {  # cmd <arm> <steps> [tag]: round 2's J command (jt.sh train DRY, ARCROWS=brake, on-k 64, on1_JTPH) from NTV0-L18-s0; the arm's draw file / flattening
  local A=$1 S=$2 T=${3:-${TAG[$1]}} C; C=$(DRY=1 ONK=64 ARCROWS=brake bash $JT train $T on1_JTPH $LAT ${WF[$A]} $S 0 | tail -1)
  [[ $C == *"--init RB0-F-s0 "* && $C == *"--on-k 64 "* && $C == *"--warmup 500 "* && $C == *"--steps $S "* && $C == *"w/${WF[$A]}.npz "* \
     && $C == *"--on on1_JTPH "* && $C == *"--on-arc 0 "* && $C == *"--lr-scale 0.1 "* ]] || die "unexpected command: $C"
  echo "${C/--init RB0-F-s0 /--init $INIT } ${FL[$A]}" | sed 's/ *$//'; }
read281() {  # read281 <tag> <prio>: PAI 281 ACC + GAP (NTV0 + ACC + GAP's chunk lists, g10, FAST=1, CONC 4, as round 2) -> "<chunk ids...> <facts job id>"
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
  R2J=$(bash experiments/round2/scripts/r2.sh dry $N | sed -n 's/^J: //p')
  for A in JW JNF; do echo "$A: $(cmd $A $N)"; echo "== R2J -> $A"; diff <(tr ' ' '\n' <<< "$R2J") <(cmd $A $N | tr ' ' '\n'); done; true ;;
smoke)
  for A in JW JNF; do [[ ! -e $RUNS/R3SM$A-L18-s0 ]] || die "R3SM$A-L18-s0 exists"; done
  [[ -s $DATA_DIR/runs/defect_fix/w/R3JW.npz ]] || die "no R3JW draw file (r3.py weights)"
  h=$(git rev-parse --short HEAD); echo "$h smoke" >> $O/code.txt
  for A in JW JNF; do s=$(sub r3-smoke-$A "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain $A) || exit 1; echo "$(date '+%F %T') smoke $A (code $h): $s" | tee -a $O/stages.txt; done ;;
smoketrain)
  A=$2 X=R3SM$2-L18-s0; C="$(cmd $A 60 $X) --diag-every 20" || exit 1
  echo "$(date '+%F %T') smoke $A: $C" | tee -a $O/stages.txt
  $C > $O/smoke_$A.log 2>&1 || die "smoke run $A (log $O/smoke_$A.log)"
  $TR $PY smoke --tag $X --arm $A; r=$?; rm -rf $RUNS/$X; exit $r ;;
launch)
  for A in JW JNF; do [[ ! -e $RUNS/${TAG[$A]}/ckpt-final.pt ]] || die "${TAG[$A]} exists"; done
  python3 -c "import json, sys; sys.exit(0 if all(json.load(open(f'$RES/smoke_{a}.json'))['pass_'] for a in ('JW', 'JNF')) else 1)" || die "smoke not passed"
  [[ -s $F_/$CTRL-vl.csv ]] || die "no stored $CTRL-vl facts"
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt; $TR $PY steps --n $N > /dev/null || die "steps"
  declare -A T_; x=()
  for A in JW JNF; do T_[$A]=$(sub r3-train-${TAG[$A]} "${FT[@]}" --timeout-h 4 -- bash $ME train $A) || exit 1
    o=$(PRIO=13 AFTER=${T_[$A]} bash experiments/pai_fix/scripts/pai_fix.sh read ${TAG[$A]} vl "v t" | tail -1) || die "pai_fix read ${TAG[$A]}"
    x+=(${o%% *}); echo "${o%% *} pai_fix-tax-${TAG[$A]}-vl" >> $O/jobs.txt
    awk '{print $1, $2}' $DATA_DIR/runs/pai_fix/reads/${TAG[$A]}-vl/jobs.txt >> $O/jobs.txt; done
  p=$(sub r3-paifix --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME paifix) || exit 1
  echo "$(date '+%F %T') launch (code $h): train ${T_[JW]} ${T_[JNF]} PAI-FIX ${x[*]} report + gate $p" | tee -a $O/stages.txt ;;
train)
  A=$2; CMD="$(cmd $A $N)" || exit 1; echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $O/stages.txt; exec $CMD ;;
paifix)  # report against CTRL and NTV0, the registered lines; every passing arm's PAI 281 ACC + GAP read at PRIO 30 -> r281; else DONE
  rd=(NTV0-L18-s0-vl=$F_/NTV0-L18-s0-vl.csv $CTRL-vl=$F_/$CTRL-vl.csv R2J-L18-s0-vl=$F_/R2J-L18-s0-vl.csv)
  for A in JW JNF; do rd+=(${TAG[$A]}-vl=$F_/${TAG[$A]}-vl.csv); done
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $CTRL-vl --out $RES/paifix_vl_vs_ctrl.md --scores $RES/paifix_vl.csv > /dev/null \
    || die "pai_fix report vs CTRL"
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref NTV0-L18-s0-vl --out $RES/paifix_vl_vs_ntv0.md > /dev/null || die "pai_fix report vs NTV0"
  ps=$($V $PY gate --arms ${TAG[JW]} ${TAG[JNF]} --ctrl $CTRL) || die "gate"
  echo "$(date '+%F %T') gate: passing [${ps}]" | tee -a $O/stages.txt
  [[ -z $ps ]] && { date '+%F %T' > $O/DONE; exit 0; }
  r=(); for T in $ps; do y=($(read281 $T 30)) || exit 1
    r+=($(PRIO=30 sub r3-r281-$T --after ${y[-1]} --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281 $T)) || exit 1
    echo "$(date '+%F %T') PAI 281 ag $T: chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]} report ${r[-1]}" | tee -a $O/stages.txt; done
  PRIO=30 sub r3-done --after $(csv ${r[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/DONE" > /dev/null ;;
r281)
  T=$2; $TR $R2PY r281 --read NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R2J-ag=$F_/R2J-L18-s0-281ag.csv $T-ag=$F_/$T-281ag.csv --pairs $T-ag:NTV0-ag $T-ag:R2J-ag \
      --out $RES/pai281_${T%-L18-s0}_ag.md --title "PAI 281, ACC + GAP: $T vs stored NTV0-L18-s0 (and R2J)" || die "r281 $T" ;;
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
*) sed -n 2,11p "$0"; exit 2 ;;
esac
