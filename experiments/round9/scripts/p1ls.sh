#!/usr/bin/env bash
# ROUND9 P1L-SPLIT (plans/2026-10-24-p1l-split.md = registration): which half of P1L's switch at T = 10 000 matters. P1La = P1L's argv
# + --lr2-groups net,adapter (route branch keeps P1's schedule); P1Lb = + --lr2-groups route (net / adapter keep P1's schedule). Seed 0, shipped
# weights. Box, repo root; every job through the pool, both chains in one submit; ids $O/jobs.txt, stages $O/stages.txt, $O = runs/round9/p1l_split.
# Reuses pcl.py (ident, tcheck, smoke, ratio, post), p1ls.py (argv, lrcheck, verdict) and p1l.sh's read path; edits neither p1l.* nor pcl.*.
#   p1ls.sh all    flag-off gates (P1's argv vs PCL1-F-s0; P1L's argv vs PCL1L-F-s0 at T = 10 000, vs P1LT30-F-s0 at T = 30 and, a whole
#                  60-step run, vs P1LSM-F-s0; lr included)
#                  -> per arm: pre-T identity vs PCL1L-F-s0 + lr, T = 30 + lr, 60-step smoke + lr -> training -> ratio gate -> post (ADE,
#                  pre-T vs P1L, whole-run lr) -> 8 PAI 281 chunks + tax -> scorer_v2 new -> both arms: verdict -> $O/DONE
#   p1ls.sh wait   one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: ident ... | smoke ... | train ARM | post ARM | sv2 ARM | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/p1l_split; ME=experiments/round9/scripts/p1ls.sh; C=experiments/round9/scripts/pcl.py; PY=experiments/round9/scripts/p1ls.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; P1LR=$DATA_DIR/runs/round9/p1l/pai281/PCL1L-F-s0-ag; CH=$P1LR/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = p1l.sh
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                                  # eager, as P1 / P1L
declare -A TAGOF=([P1LA]=PCL1LA-F-s0 [P1LB]=PCL1LB-F-s0); T=10000
PRIO=${PRIO:-12}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-p1ls --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= p1l.sh read281)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
all)
  for X in ${TAGOF[*]} PCL1LA-F-s0x10000 PCL1LB-F-s0x10000 P1SID0-F-s0 P1SID1-F-s0 P1SID2-F-s0 P1SID3-F-s0 P1SID3-F-s0x30 P1SAP-F-s0 P1SBP-F-s0 P1SAT-F-s0 P1SBT-F-s0 \
      P1SASM-F-s0 P1SBSM-F-s0 P1SASM-F-s0x30 P1SBSM-F-s0x30; do
    [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in PCL1L-F-s0 PCL1-F-s0 NTV0-L18-s0 RB0-F-s0; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f facts"; done
  grep -qx "SERVE=$SERVE" $P1LR/config.txt && grep -qx "WIZ=$WIZ" $P1LR/config.txt && grep -qx "ALPASIM_FAST=1" $P1LR/config.txt \
    && grep -qx "CONC=4" $P1LR/config.txt || die "serve differs from $P1LR"
  [[ $(ls $CH/*.txt | wc -l) == 8 ]] || die "not 8 chunk lists"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  G=(); g() { local x; x=$(sub "$@") && [[ -n $x ]] || die "submit $1"; G+=($x); }
  g p1ls-g0-flagoff-P1 "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1-F-s0 100 125 strict g0_flagoff_P1 P1 P1SID0-F-s0 -
  g p1ls-g0-flagoff-P1L "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1L-F-s0 100 125 strict g0_flagoff_P1L P1L P1SID1-F-s0 -
  g p1ls-g0-flagoff-P1LT30 "${FE[@]}" --timeout-h 1 -- bash $ME ident P1LT30-F-s0 75 100 strict g0_flagoff_P1LT30 P1L P1SID2-F-s0 - --on-start 30
  g p1ls-g0-flagoff-P1LSM "${FE[@]}" --timeout-h 1 -- bash $ME ident P1LSM-F-s0 60 61 strict g0_flagoff_P1LSM P1L P1SID3-F-s0 - \
    --steps 60 --on-start 30 --save-at 30                                   # a whole run (lr before and after T) vs P1L's stored smoke
  for arm in P1LA P1LB; do x=${arm: -1}
    g p1ls-g1-preT-$arm "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1L-F-s0 100 125 strict g1_preT_$arm $arm P1S${x}P-F-s0 $T
    g p1ls-g2-T30-$arm "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1-F-s0 25 75 new g2_T30_$arm $arm P1S${x}T-F-s0 30 --on-start 30
    g p1ls-smoke-$arm "${FE[@]}" --timeout-h 1 -- bash $ME smoke $arm P1S${x}SM-F-s0 --steps 60 --on-start 30 --save-at 30
  done
  GA=$(IFS=,; echo "${G[*]}"); RP=()
  for arm in P1LA P1LB; do tag=${TAGOF[$arm]}
    t=$(sub p1ls-train-$tag --after $GA "${FE[@]}" --timeout-h 4 -- bash $ME train $arm) || exit 1
    r=$(sub p1ls-ratio-$arm --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 6 -- $V $C ratio --tag $tag --on-start $T --train-id $t --out $O/results/ratio_$arm.json) || exit 1
    p=$(sub p1ls-post-$arm --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $arm) || exit 1
    y=($(read281 $tag $p)); [[ ${#y[@]} == 9 ]] || die "read281 $tag"
    v=$(sub p1ls-sv2-$arm --after ${y[-1]} --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- bash $ME sv2 $arm) || exit 1
    RP+=($v); echo "$(date '+%F %T') $arm: train $t ratio $r post $p 281 ${y[*]} sv2 $v" | tee -a $O/stages.txt
  done
  rp=$(sub p1ls-report --after $(IFS=,; echo "${RP[*]}") --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): gates ${G[*]} report $rp" | tee -a $O/stages.txt ;;
train)
  tag=${TAGOF[$2]}; mapfile -t A < <($V $PY argv $2 $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $tag (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
ident)  # ident REF UPTO STOP strict|new NAME ARM TAG LR_T|- [argv opts]: the arm's argv until step STOP, scalars <= UPTO vs REF; LR_T: lr check at T
  ref=$2 upto=$3 stop=$4 mode=$5 nm=$6 arm=$7 tag=$8 lt=$9; shift 9
  mapfile -t A < <($V $PY argv $arm $tag "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $C ident --tag $tag --ref $ref --upto $upto --stop $stop $([[ $mode == new ]] && echo --allow-new) --out $O/results/$nm.json -- $TR "${A[@]}"; r=$?
  if (( r == 0 )) && [[ $nm == g2_T30_* ]]; then $V $C tcheck --tag $tag --on-start 30 --out $O/results/${nm}_batch.json; r=$?; fi
  if (( r == 0 )) && [[ $lt != - ]]; then $V $PY lrcheck --tag $tag --on-start $lt --out $O/results/${nm}_lr.json; r=$?; fi
  rm -rf $RUNS/$tag $RUNS/${tag}x30; exit $r ;;
smoke)  # smoke ARM TAG [argv opts]: 60 steps (T = 30) + pcl.py smoke checks + lr check
  arm=$2 tag=$3; shift 3
  mapfile -t A < <($V $PY argv $arm $tag "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" || die "smoke $tag trainer"
  $V $C smoke --tag $tag --snap ${tag}x30 --on-start 30 --out $O/results/smoke_$arm.json && \
    $V $PY lrcheck --tag $tag --on-start 30 --out $O/results/smoke_lr_$arm.json; r=$?
  (( r == 0 )) && rm -rf $RUNS/$tag $RUNS/${tag}x30; exit $r ;;
post)
  tag=${TAGOF[$2]}
  $V $C post --tag $tag --ref PCL1L-F-s0 --on-start $T --out $O/results/post_$2.json || die "post $2"
  $V $PY lrcheck --tag $tag --on-start $T --full --out $O/results/lr_full_$2.json || die "lr_full $2" ;;
sv2)
  tag=${TAGOF[$2]}; rm -rf $O/sv2_$2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src   # new scorer (decision 453)
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2_$2 --reads $tag-ag --workers 16 > $O/sv2_$2.log 2>&1 \
    || die "scorer_v2 new $2 (log $O/sv2_$2.log)" ;;
report)
  mkdir -p $O/sv2/new; $V - $O <<'PY' || die "merge sv2"
import sys, pandas as pd
o = sys.argv[1]
pd.concat([pd.read_csv(f"{o}/sv2_{a}/new/rollouts.csv") for a in ("P1LA", "P1LB")]).to_csv(f"{o}/sv2/new/rollouts.csv", index=False)
PY
  $TR experiments/round2/scripts/r2.py r281 --read P1La=$F_/PCL1LA-F-s0-281ag.csv P1Lb=$F_/PCL1LB-F-s0-281ag.csv P1L=$F_/PCL1L-F-s0-281ag.csv \
      P1=$F_/PCL1-F-s0-281ag.csv NTV0s0=$F_/NTV0-L18-s0-281ag.csv RB0F=$F_/RB0-F-s0-281ag.csv --pairs P1La:P1L P1La:P1 P1Lb:P1L P1Lb:P1 P1La:P1Lb \
      --out $O/results/pai281.md --title "PAI 281, ACC + GAP, Harmonizer OFF: P1La / P1Lb vs P1L, P1 (old scorer)" || die r281
  $V $PY verdict --dir $O --out $O/results/verdict > $O/results/verdict.log 2>&1 || die "verdict (log $O/results/verdict.log)"
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
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -5 $O/stages.txt; true ;;
*) sed -n 2,12p "$0"; exit 2 ;;
esac
