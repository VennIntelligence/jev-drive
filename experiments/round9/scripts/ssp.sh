#!/usr/bin/env bash
# ROUND9 S-SPLIT (plans/2026-10-24-s-split.md = the registration): which part of the S rows carries NTV0's PAI gain, the hinges or distill_ho?
# Two arms, seed 0, both = P0 (PCL0-L18-s0's stored argv, compiled) with one switch:
#   NH  SSNH-L18-s0   --ho-w 3 -> 0: agent_ho, road_ho and the nr1_lat terms of the S rows x 0; distill_ho kept (32 S rows per batch still drawn)
#   ND  SSND-L18-s0   + --ho-distill 0 (on_train, off by default): distill_ho of the S rows x 0; every hinge kept
# then PAI 281 ACC + GAP (Harmonizer OFF, P0's serve and chunk lists). Box, repo root; every job through the pool, the whole chain in one submit;
# ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/s_split. Job bodies follow pcl.sh's.
#   ssp.sh dry     the two argvs and their token diffs against PCL0-L18-s0's stored argv (exit 1 on any other diff)
#   ssp.sh all     gates (flag-off identity: P0's argv on HEAD vs the same argv on the commit before --ho-distill, steps <= 50; 60-step smokes with
#                  the switch checks) -> 2 trainings -> post gates (DONE, dev ADE <= 0.566) -> PAI 281 reads + tax -> report -> $O/DONE
#   ssp.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: flagoff | smoke ARM | train ARM | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
REPO=$PWD; O=$DATA_DIR/runs/round9/s_split; ME=experiments/round9/scripts/ssp.sh; PY=experiments/round9/scripts/ssp.py; PC=experiments/round9/scripts/pcl.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = pcl.sh (P0's read)
FC=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)                                               # compiled, as P0
NH=SSNH-L18-s0; ND=SSND-L18-s0
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-ssp --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= pcl.sh read281)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
base() {  # the commit before the --ho-distill commit (flag-off reference code)
  local f; f=$(git log -1 --format=%H --grep='^round9 S-SPLIT: on_train --ho-distill') && [[ -n $f ]] || die "no --ho-distill commit"; git rev-parse $f^; }
case ${1:-} in
dry) for a in NH ND; do echo "== $a"; $V $PY diff $a || die "diff $a"; done ;;
all)
  bash $ME dry || die "argv differs"
  for X in $NH $ND SSNHSM-L18-s0 SSNDSM-L18-s0 SSID-L18-s0 SSIDB-L18-s0; do
    [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in PCL0-L18-s0 NTV0NS-L18-s0 NTV0-L18-s0 RB0-F-s0; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f facts"; done
  h=$(git rev-parse --short HEAD); b=$(base) || exit 1; echo "$h all, flag-off base $b" >> $O/code.txt
  G=(); g() { local x; x=$(sub "$@") && [[ -n $x ]] || die "submit $1"; G+=($x); }
  g ssp-g0-flagoff "${FC[@]}" --timeout-h 2 -- bash $ME flagoff
  g ssp-smoke-NH "${FC[@]}" --timeout-h 1 -- bash $ME smoke NH
  g ssp-smoke-ND "${FC[@]}" --timeout-h 1 -- bash $ME smoke ND
  GA=$(IFS=,; echo "${G[*]}"); TAX=()
  for arm in NH ND; do
    tag=${!arm}
    t=$(sub ssp-train-$tag --after $GA "${FC[@]}" --timeout-h 3 -- bash $ME train $arm) || exit 1
    p=$(sub ssp-post-$arm --after $t --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PC post --tag $tag --out $O/results/post_$arm.json) || exit 1
    y=($(read281 $tag $p)); [[ ${#y[@]} == 9 ]] || die "read281 $tag"
    TAX+=(${y[-1]})
    echo "$(date '+%F %T') $arm $tag: train $t post $p 281 ${y[*]}" >> $O/stages.txt
  done
  r=$(sub ssp-report --after $(IFS=,; echo "${TAX[*]}") --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h, base $b): gates ${G[*]} report $r" | tee -a $O/stages.txt ;;
flagoff)  # P0's argv (flag off) to step 75 on HEAD and on the base commit (git worktree); every scalar <= 50 bit-identical; vs the stored P0: recorded
  b=$(base) || exit 1; WT=$DATA_DIR/tmp/ssp_wt_${b:0:8}
  mapfile -t A < <($V $PY argv P0 SSID-L18-s0); mapfile -t B < <($V $PY argv P0 SSIDB-L18-s0); [[ ${#A[@]} -gt 5 && ${#B[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') flagoff HEAD $(git rev-parse --short HEAD) vs base ${b:0:8}: ${A[*]}" | tee -a $O/stages.txt
  $V $PC ident --tag SSID-L18-s0 --ref PCL0-L18-s0 --upto 50 --stop 75 --out $O/results/g0_head_vs_stored.json -- $TR "${A[@]}"
  [[ -d $WT ]] || git worktree add --detach $WT $b > /dev/null 2>&1 || die "worktree $b"
  (cd $WT && [[ $(git rev-parse HEAD) == $b ]] && $REPO/$V $REPO/$PC ident --tag SSIDB-L18-s0 --ref PCL0-L18-s0 --upto 50 --stop 75 \
      --out $O/results/g0_base_vs_stored.json -- $TR "${B[@]}")
  $V $PY offcmp --a SSID-L18-s0 --b SSIDB-L18-s0 --upto 50 --out $O/results/g0_flagoff.json; r=$?
  rm -rf $RUNS/SSID-L18-s0 $RUNS/SSIDB-L18-s0; git worktree remove --force $WT || true; exit $r ;;
smoke)  # smoke ARM: 60 steps (--diag-every 20) + pcl.py smoke checks + the switch check
  arm=$2; tag=SS${arm}SM-L18-s0
  mapfile -t A < <($V $PY argv $arm $tag --steps 60 --diag-every 20); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" || die "smoke $tag trainer"
  $V $PC smoke --tag $tag --out $O/results/smoke_$arm.json || exit 2
  $V $PY gcheck --arm $arm --tag $tag --out $O/results/gcheck_$arm.json; r=$?
  (( r == 0 )) && rm -rf $RUNS/$tag; exit $r ;;
train)  # train ARM
  tag=SS$2-L18-s0; mapfile -t A < <($V $PY argv $2 $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $tag (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read NH=$F_/$NH-281ag.csv ND=$F_/$ND-281ag.csv P0=$F_/PCL0-L18-s0-281ag.csv NTV0NS=$F_/NTV0NS-L18-s0-281ag.csv \
      NTV0s0=$F_/NTV0-L18-s0-281ag.csv RB0F=$F_/RB0-F-s0-281ag.csv --pairs NH:P0 NH:NTV0NS ND:P0 ND:NTV0NS NH:ND NH:NTV0s0 ND:NTV0s0 P0:NTV0NS \
      --out $O/results/pai281.md --title "PAI 281, ACC + GAP, Harmonizer OFF: S-SPLIT (NH / ND) vs P0, NTV0NS, NTV0-s0, RB0-F" || die r281
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src   # new scorer (decision 453)
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $NH-ag,$ND-ag --workers 16 > $O/sv2.log 2>&1 \
    || die "scorer_v2 new (log $O/sv2.log)"
  $V $PY verdict --dir $O --out $O/results/verdict || die verdict
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
*) sed -n 2,14p "$0"; exit 2 ;;
esac
