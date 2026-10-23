#!/usr/bin/env bash
# ROUND9 P1L (plans/2026-10-24-p1l.md = registration): P1's argv (PCL1-F-s0) + NTV0-L18's lr schedule from step 10 000 (rm_train --lr2-at),
# seed 0, then PAI 281 ACC + GAP (Harmonizer OFF, P1's serve and chunk lists). Box, repo root; every job through the pool, the whole chain in one
# submit; ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/p1l. Reuses pcl.py (argv P1L, ident, tcheck, smoke,
# ratio, post) and pcl.sh's read path.
#   p1l.sh dry     the argv and its token diff against RB0L-F-s0 (no job)
#   p1l.sh all     gates (flag off: RB0-F vs RB0-F-s0, P1 vs PCL1-F-s0; flag on: pre-T identity + lr, T = 30 post-T + lr, 60-step smoke + lr)
#                  -> training -> S / imit ratio gate (T .. T + 500) + post gate (DONE, dev ADE, pre-T vs P1, whole-run lr) -> PAI 281 reads + tax
#                  -> report (r281, scorer_v2 new, behavior_stats, verdict) -> $O/DONE
#   p1l.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: ident ... | smoke ... | train | post | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/p1l; ME=experiments/round9/scripts/p1l.sh; PY=experiments/round9/scripts/pcl.py; PL=experiments/round9/scripts/p1l.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = pcl.sh
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                                  # eager, as P1
TAG=PCL1L-F-s0; T=10000
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-p1l --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= pcl.sh read281)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
dry) $V $PY diff P1L || die "diff" ;;
all)
  for X in $TAG ${TAG}x10000 P1LID0-F-s0 P1LID1-F-s0 P1LID2-F-s0 P1LT30-F-s0 P1LSM-F-s0 P1LSM-F-s0x30; do
    [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  [[ -s $DATA_DIR/runs/round9/pool_pilot/w/JT00S-rb0.npz ]] || die "no JT00S-rb0 wfile"
  for f in PCL1-F-s0 PCL0-L18-s0 NTV0-L18-s0 RB0-F-s0; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f facts"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  G=(); g() { local x; x=$(sub "$@") && [[ -n $x ]] || die "submit $1"; G+=($x); }
  g p1l-g0-flagoff-RB0F "${FE[@]}" --timeout-h 1 -- bash $ME ident RB0-F-s0 75 100 strict g0_flagoff_RB0F RB0F P1LID0-F-s0 -
  g p1l-g0-flagoff-P1 "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1-F-s0 100 125 strict g0_flagoff_P1 P1 P1LID1-F-s0 -
  g p1l-g1-preT "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1-F-s0 100 125 new g1_preT P1L P1LID2-F-s0 $T
  g p1l-g2-T30 "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1-F-s0 25 75 new g2_T30 P1L P1LT30-F-s0 30 --on-start 30
  g p1l-smoke "${FE[@]}" --timeout-h 1 -- bash $ME smoke P1LSM-F-s0 --steps 60 --on-start 30 --save-at 30
  GA=$(IFS=,; echo "${G[*]}")
  t=$(sub p1l-train-$TAG --after $GA "${FE[@]}" --timeout-h 4 -- bash $ME train) || exit 1
  r=$(sub p1l-ratio --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 6 -- $V $PY ratio --tag $TAG --on-start $T --train-id $t --out $O/results/ratio_P1L.json) || exit 1
  p=$(sub p1l-post --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post) || exit 1
  y=($(read281 $TAG $p)); [[ ${#y[@]} == 9 ]] || die "read281 $TAG"
  rp=$(sub p1l-report --after ${y[-1]} --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): gates ${G[*]} train $t ratio $r post $p 281 ${y[*]} report $rp" | tee -a $O/stages.txt ;;
train)
  mapfile -t A < <($V $PY argv P1L $TAG); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $TAG (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
ident)  # ident REF UPTO STOP strict|new NAME ARM TAG LR_T|- [argv opts]: the arm's argv until step STOP, scalars <= UPTO vs REF; LR_T: lr check at T
  ref=$2 upto=$3 stop=$4 mode=$5 nm=$6 arm=$7 tag=$8 lt=$9; shift 9
  mapfile -t A < <($V $PY argv $arm $tag "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PY ident --tag $tag --ref $ref --upto $upto --stop $stop $([[ $mode == new ]] && echo --allow-new) --out $O/results/$nm.json -- $TR "${A[@]}"; r=$?
  if (( r == 0 )) && [[ $nm == g2_T30 ]]; then $V $PY tcheck --tag $tag --on-start 30 --out $O/results/g2_T30_batch.json; r=$?; fi
  if (( r == 0 )) && [[ $lt != - ]]; then $V $PL lrcheck --tag $tag --on-start $lt --out $O/results/${nm}_lr.json; r=$?; fi
  rm -rf $RUNS/$tag; exit $r ;;
smoke)  # smoke TAG [argv opts]: 60 steps (T = 30) + pcl.py smoke checks + lr check
  tag=$2; shift 2
  mapfile -t A < <($V $PY argv P1L $tag "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" || die "smoke $tag trainer"
  $V $PY smoke --tag $tag --snap ${tag}x30 --on-start 30 --out $O/results/smoke_P1L.json && \
    $V $PL lrcheck --tag $tag --on-start 30 --out $O/results/smoke_lr.json; r=$?
  (( r == 0 )) && rm -rf $RUNS/$tag $RUNS/${tag}x30; exit $r ;;
post)
  $V $PY post --tag $TAG --ref PCL1-F-s0 --on-start $T --out $O/results/post_P1L.json || die "post"
  $V $PL lrcheck --tag $TAG --on-start $T --full --out $O/results/lr_full.json || die "lr_full" ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read P1L=$F_/$TAG-281ag.csv P1=$F_/PCL1-F-s0-281ag.csv P0=$F_/PCL0-L18-s0-281ag.csv \
      NTV0s0=$F_/NTV0-L18-s0-281ag.csv RB0F=$F_/RB0-F-s0-281ag.csv --pairs P1L:P1 P1L:P0 P1L:NTV0s0 P1L:RB0F \
      --out $O/results/pai281.md --title "PAI 281, ACC + GAP, Harmonizer OFF: P1L (PCL1L-F-s0) vs P1, P0, NTV0-s0, RB0-F" || die r281
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src   # new scorer (decision 453)
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $TAG-ag --workers 16 > $O/sv2.log 2>&1 \
    || die "scorer_v2 new (log $O/sv2.log)"
  $V $PL bs --out $O/bs > $O/bs.log 2>&1 || echo "$(date '+%F %T') behavior_stats failed (log $O/bs.log)" | tee -a $O/stages.txt
  $V $PL verdict --dir $O --out $O/results/verdict || die verdict
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
*) sed -n 2,13p "$0"; exit 2 ;;
esac
