#!/usr/bin/env bash
# ROUND9 P1 SEED 1 (plans/2026-10-23-pool-design.md section 6, supplement 2026-10-24): PCL1-F-s1 = PCL1-F-s0's argv with only --seed 1 and the tag
# changed (pcl.py argv P1 TAG --seed 1), then PAI 281 ACC + GAP (Harmonizer OFF, P1's serve and chunk lists). Box, repo root; every job through the
# pool, the whole chain in one submit; ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/p1_s1. Job bodies = pcl.sh's.
#   p1s1.sh dry    the argv and its token diff against PCL1-F-s0's stored meta argv (only --seed and --tag may differ)
#   p1s1.sh all    60-step smoke -> training -> S / imit ratio gate + post gate (DONE, dev ADE) -> PAI 281 reads + tax -> report -> $O/DONE
#   p1s1.sh wait   one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/p1_s1; ME=experiments/round9/scripts/p1s1.sh; PY=experiments/round9/scripts/pcl.py; PS=experiments/round9/scripts/p1s1.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = pcl.sh
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                          # eager, as P1
TAG=PCL1-F-s1; T=10000
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-p1s1 --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
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
dry)
  $V $PY argv P1 $TAG --seed 1 > $O/argv.txt || die "argv"
  $V - $O/argv.txt <<'PY'
import sys
sys.path.insert(0, "experiments/round9/scripts")
import pcl as C
a, b = C.stored_argv("PCL1-F-s0"), open(sys.argv[1]).read().split("\n")[:-1]
d = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
print(len(a), len(b), d)
ok = len(a) == len(b) and [x for _, x, _ in d] == ["PCL1-F-s0", "0"] and [y for _, _, y in d] == ["PCL1-F-s1", "1"] and a[d[0][0] - 1] == "--tag" and a[d[1][0] - 1] == "--seed"
print("ONLY --tag AND --seed DIFFER" if ok else "DIFFERS BEYOND --tag / --seed")
sys.exit(0 if ok else 1)
PY
  ;;
all)
  $0 dry || die "argv differs"
  for X in $TAG ${TAG}x10000 PCL1SM-F-s1 PCL1SM-F-s1x30; do
    [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in PCL1-F-s0 NTV0-L18-s0 NTV0-L18-s1 RB0-F-s0; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f facts"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  sm=$(sub p1s1-smoke "${FE[@]}" --timeout-h 1 -- bash $ME smoke PCL1SM-F-s1 30 --steps 60 --on-start 30 --save-at 30) || exit 1
  t=$(sub p1s1-train-$TAG --after $sm "${FE[@]}" --timeout-h 4 -- bash $ME train) || exit 1
  r=$(sub p1s1-ratio --after $sm --vram 0.1 --cpu 1 --ram 4 --timeout-h 6 -- $V $PY ratio --tag $TAG --on-start $T --train-id $t --out $O/results/ratio_P1.json) || exit 1
  p=$(sub p1s1-post --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY post --tag $TAG --ref RB0L-F-s0 --on-start $T --out $O/results/post_P1.json) || exit 1
  y=($(read281 $TAG $p)); [[ ${#y[@]} == 9 ]] || die "read281 $TAG"
  rp=$(sub p1s1-report --after ${y[-1]} --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): smoke $sm train $t ratio $r post $p 281 ${y[*]} report $rp" | tee -a $O/stages.txt ;;
train)
  mapfile -t A < <($V $PY argv P1 $TAG --seed 1); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $TAG (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
smoke)  # smoke TAG SNAP [argv opts]: 60 steps (T = 30) + pcl.py smoke checks
  tag=$2 snap=$3; shift 3
  mapfile -t A < <($V $PY argv P1 $tag --seed 1 "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" || die "smoke $tag trainer"
  $V $PY smoke --tag $tag --snap ${tag}x$snap --on-start 30 --out $O/results/smoke_P1.json; r=$?
  (( r == 0 )) && rm -rf $RUNS/$tag $RUNS/${tag}x$snap; exit $r ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read S1=$F_/$TAG-281ag.csv S0=$F_/PCL1-F-s0-281ag.csv NTV0s0=$F_/NTV0-L18-s0-281ag.csv \
      NTV0s1=$F_/NTV0-L18-s1-281ag.csv RB0F=$F_/RB0-F-s0-281ag.csv --pairs S1:S0 S1:NTV0s0 S1:NTV0s1 S1:RB0F \
      --out $O/results/pai281.md --title "PAI 281, ACC + GAP, Harmonizer OFF: PCL1-F-s1 vs PCL1-F-s0, NTV0-s0, NTV0-s1, RB0-F" || die r281
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src   # new scorer (decision 453)
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $TAG-ag --workers 16 > $O/sv2.log 2>&1 \
    || die "scorer_v2 new (log $O/sv2.log)"
  $V $PS verdict --dir $O --out $O/results/p1_s1 || die verdict
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
*) sed -n 2,9p "$0"; exit 2 ;;
esac
