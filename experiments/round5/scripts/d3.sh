#!/usr/bin/env bash
# D3 (round 5 diagnostic, plans/2026-10-21-round5-framework-draft.md): is there outcome signal for closed-loop learning on PAI? NTV0-L18-s0 with
# the PAI candidate serve (ACC + GAP, steering weight 10, ALPASIM_FAST=1) rolled K = 4 ways on 150 curated train scenes (results/d3_scenes.tsv:
# stop without lead / sharp turn / launch 38 each, other 36, non-held, one per drive group, never test281 / val160 or their drives). Arms (d3.py):
#   A  the served config            F  ACC targets x1.15 (JEV_ACC_VSET 1.15, JEV_ACC_ALAT 2.4 x 1.15^2)
#   S  ACC targets x0.85 (VSET 0.85, ALAT 2.4 x 0.85^2)               L  route-seeking choice among the model's own TTS candidates (JEV_TTS)
# Staged (docs/long-runs.md): smoke = 1 launch scene x 4 -> chk1 (knobs reach the driver, scores, A against the stored d402 roll) -> c000 (25 x 4)
# -> chk2 (scored >= 23 / 25 per arm, projected cost <= 6 card-h) -> rest (c001..c005 x 4, submitted by a pool job) -> report -> $O/DONE.
# Every GPU job through the pool at PRIO 8; ids in $O/jobs.txt, stages in $O/stages.txt, a stop reason in $O/VERDICT.
#   d3.sh chain      submit the whole chain once (never overwrites $O)
#   d3.sh wait       blocks until $O/DONE / $O/VERDICT or a failed job of this lane (the one run_in_background waiter)
#   job bodies: chk1 | chk2 | rest | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round5/d3; ME=experiments/round5/scripts/d3.sh; PY=experiments/round5/scripts/d3.py; V=.venv/bin/python
T=NTV0-L18-s0; PRIO=${PRIO:-8}; TSV=experiments/round5/results/d3_scenes.tsv; ARMS="A F S L"
BASE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
declare -A SV=([A]="$BASE" [F]="$BASE JEV_ACC_VSET=1.15 JEV_ACC_ALAT=3.174" [S]="$BASE JEV_ACC_VSET=0.85 JEV_ACC_ALAT=1.734"
               [L]="$BASE JEV_TTS=1 JEV_TTS_W=$PWD/experiments/round5/scripts/d3_tts_route.json")
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-d3 --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
roll() {  # roll <arm> <root> [after] -> the ids of the chunks submitted now (comma separated)
  local f=$2/submit.$(date +%s%N).txt
  SERVE="${SV[$1]}" WIZ=$WIZ CONC=4 PRIO=$PRIO AFTER=${3:-} bash experiments/alpasim/scripts/pai_box.sh submit $2 $T > $f || die "roll submit $1 $2"
  grep -q 'not submitted' $f && die "roll $1 $2: $(grep 'not submitted' $f)"
  awk -v a=$1 '$2 ~ /^pai-/ {print $1, "roll-" a "-" substr($2, length($2) - 3)}' $f >> $O/jobs.txt
  awk '$2 ~ /^pai-/ {print $1}' $f | paste -sd, -; }
case ${1:-} in
chain)
  [[ ! -e $O/roll && ! -e $O/smoke && ! -e $O/DONE ]] || die "$O has a chain: never overwritten"
  [[ -f $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "no $T"
  git rev-parse --short HEAD > $O/code.txt; $V $PY lists --scenes $TSV --root $O || die lists
  s=(); for a in $ARMS; do mkdir -p $O/smoke/$a/chunks $O/roll/$a/chunks; cp $O/lists/s000.txt $O/smoke/$a/chunks/; cp $O/lists/c000.txt $O/roll/$a/chunks/
    s+=($(roll $a $O/smoke/$a)) || exit 1; done
  k1=$(sub d3-chk1 --after $(IFS=,; echo "${s[*]}") --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME chk1) || exit 1
  c=(); for a in $ARMS; do c+=($(roll $a $O/roll/$a $k1)) || exit 1; done
  k2=$(sub d3-chk2 --after $(IFS=,; echo "${c[*]}") --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME chk2) || exit 1
  rs=$(sub d3-rest --after $k2 --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt)): smoke ${s[*]} chk1 $k1; c000 ${c[*]} chk2 $k2; rest $rs" | tee -a $O/stages.txt ;;
chk1) $V $PY check --root $O --stage smoke ;;
chk2) $V $PY check --root $O --stage c000 ;;
rest)  # the remaining chunk lists into every arm root, one submit per arm (pai_box.sh skips the finished c000), then the report after all
  r=(); for a in $ARMS; do cp $O/lists/c00[1-9].txt $O/roll/$a/chunks/; r+=($(roll $a $O/roll/$a)) || exit 1; done
  p=$(sub d3-report --after $(IFS=,; echo "${r[*]}") --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') rest: rolls ${r[*]} report $p" | tee -a $O/stages.txt ;;
report)
  $V $PY report --root $O --scenes $TSV --out $O/report > $O/report.log 2>&1 || { echo "STOP report crashed ($O/report.log)" > $O/VERDICT; exit 1; }
  date '+%F %T' > $O/DONE ;;
wait)  # returns when the lane has an outcome (DONE / VERDICT) or one of its jobs failed / was cancelled
  until [[ -f $O/VERDICT || -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
ids = [l.split()[0] for l in open(sys.argv[1]) if l.split()]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for i in ids if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  cat $O/VERDICT 2>/dev/null; [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,15p "$0"; exit 2 ;;
esac
