#!/usr/bin/env bash
# ROUND9 NTV1 step 1 (plans/2026-10-24-ntv1-step1.md = registration): three arms on P1L's recipe (PCL1L-F-s0), seed 0, eager, 19 500 steps:
#   N1  = P1L + --cmd-rule heading-all                                  (S rows: on1_JT rb0, 32 per batch from step 10 000)
#   N2a = N1 + merged pool on1_JTPH (NTV1M: rb0 S 0.5 / RB0P PAI human 0.5, 64 per batch from T) + 58 RB0P PAI rows per batch before T
#   N2b = N1 + the same merged pool from T only (before T = N1)
# then PAI 281 ACC + GAP (Harmonizer OFF, P1L's serve + JEV_CMD=heading, NTV0's chunk lists). Box, repo root; every job through the pool, the
# whole chain in one submit; ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/ntv1_step1.
#   ntv.sh dry     each arm's argv and its token diff against PCL1L-F-s0's stored argv (no job)
#   ntv.sh all     gates (wfile G-W; flag off: RB0-F vs RB0-F-s0, P1L vs PCL1L-F-s0; per arm a 60-step smoke T = 30 with composition, command and lr
#                  checks; N2b vs N1 before T, reported) -> three trainings in parallel -> per arm S / imit ratio gate + post gate -> PAI 281 reads +
#                  tax + served-command check -> report (r281, scorer_v2 new, ntv.py verdict) -> $O/DONE
#   ntv.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: ident ... | smoke <arm> | same | train <arm> | post <arm> | cmdagree <arm> | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv1_step1; ME=experiments/round9/scripts/ntv.sh; PY=experiments/round9/scripts/ntv.py; PC=experiments/round9/scripts/pcl.py
PL=experiments/round9/scripts/p1l.py; CP=experiments/round9/scripts/cpl.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_CMD=heading"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = P1L + heading
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                                                # eager, as P1L
declare -A TAG=([N1]=NT1N1-F-s0 [N2A]=NT1N2A-F-s0 [N2B]=NT1N2B-F-s0); ARMS="N1 N2A N2B"; T=10000
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281 $O/w
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-ntv1 --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
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
dry) for m in $ARMS; do $V $PY diff $m || die "diff $m"; done ;;
all)
  for X in ${TAG[N1]} ${TAG[N2A]} ${TAG[N2B]} NT1ID0-F-s0 NT1ID1-F-s0 NT1SM-N1-F-s0 NT1SM-N2A-F-s0 NT1SM-N2B-F-s0; do
    [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in $DATA_DIR/runs/round9/pool_pilot/w/JT00S-rb0.npz $DATA_DIR/runs/round9/pai_base/w/RB0P.npz $DATA_DIR/runs/round9/cmd_design/rows.parquet \
           $DATA_DIR/runs/round9/p1l/sv2/new/rollouts.csv; do [[ -s $f ]] || die "no $f"; done
  for f in PCL1L-F-s0 NTV0-L18-s0 R8K-L18-s2; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f facts"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  G=(); g() { local x; x=$(sub "$@") && [[ -n $x ]] || die "submit $1"; G+=($x); }
  wf=$(sub ntv-wfile --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- $V $PY wfile) || exit 1; G+=($wf)
  g ntv-g0-flagoff-RB0F "${FE[@]}" --timeout-h 1 -- bash $ME ident RB0-F-s0 75 100 g0_flagoff_RB0F RB0F NT1ID0-F-s0
  g ntv-g0-flagoff-P1L "${FE[@]}" --timeout-h 1 -- bash $ME ident PCL1L-F-s0 100 125 g0_flagoff_P1L P1L NT1ID1-F-s0
  declare -A SM; for m in $ARMS; do SM[$m]=$(sub ntv-smoke-$m --after $wf "${FE[@]}" --timeout-h 1 -- bash $ME smoke $m) || exit 1; G+=(${SM[$m]}); done
  g ntv-same --after ${SM[N1]},${SM[N2B]} --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME same
  GA=$(IFS=,; echo "${G[*]}")
  X=(); declare -A TT
  for m in $ARMS; do TT[$m]=$(sub ntv-train-${TAG[$m]} --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train $m) || exit 1; done
  for m in $ARMS; do t=${TT[$m]}
    r=$(sub ntv-ratio-$m --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag ${TAG[$m]} --on-start $T --train-id $t \
        --out $O/results/ratio_$m.json) || exit 1
    aft=$t,$r; [[ $m != N1 ]] && aft+=,${TT[N1]}
    p=$(sub ntv-post-$m --after $aft --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $m) || exit 1
    y=($(read281 ${TAG[$m]} $p)); [[ ${#y[@]} == 9 ]] || die "read281 $m"
    c=$(sub ntv-cmdagree-$m --after $(IFS=,; echo "${y[*]:0:8}") --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME cmdagree $m) || exit 1
    X+=(${y[-1]} $c)
    echo "$(date '+%F %T') $m (code $h): train $t ratio $r post $p 281 ${y[*]} cmdagree $c" | tee -a $O/stages.txt; done
  rp=$(sub ntv-report --after $(IFS=,; echo "${X[*]}") --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): gates ${G[*]} report $rp" | tee -a $O/stages.txt ;;
train)
  m=$2; mapfile -t A < <($V $PY argv $m ${TAG[$m]}); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train ${TAG[$m]} (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
ident)  # ident REF UPTO STOP NAME ARM TAG: flag off, the arm's argv until step STOP, every scalar <= UPTO vs REF bit for bit, no extra / missing key
  ref=$2 upto=$3 stop=$4 nm=$5 arm=$6 tag=$7
  mapfile -t A < <($V $PY argv $arm $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PC ident --tag $tag --ref $ref --upto $upto --stop $stop --out $O/results/$nm.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$tag; exit $r ;;
smoke)  # smoke ARM: 60 steps, T = 30, snapshot at 30; composition (ntv.py), command (cpl.py), lr (p1l.py)
  m=$2; tag=NT1SM-$m-F-s0
  mapfile -t A < <($V $PY argv $m $tag --steps 60 --on-start 30 --save-at 30); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" || die "smoke $tag trainer"
  $V $PY smoke --arm $m --tag $tag --on-start 30 --out $O/results/smoke_$m.json; r1=$?
  $V $CP smoke --tag $tag --out $O/results/smoke_cmd_$m.json; r2=$?
  $V $PL lrcheck --tag $tag --on-start 30 --out $O/results/smoke_lr_$m.json; r3=$?
  rm -rf $RUNS/$tag $RUNS/${tag}x30; exit $(( r1 || r2 || r3 )) ;;
same) $V $PY same --a NT1SM-N2B-F-s0 --b NT1SM-N1-F-s0 --upto 25 --out $O/results/same_N2B_N1_smoke.json || true ;;
post)
  m=$2; tag=${TAG[$m]}; ref=; [[ $m == N2B ]] && ref="--ref ${TAG[N1]} --on-start $T"
  $V $PC post --tag $tag $ref --out $O/results/post_$m.json || die "post $m"
  $V $PL lrcheck --tag $tag --on-start $T --full --out $O/results/lr_full_$m.json || die "lr_full $m" ;;
cmdagree) DL_RES=$O/results $V experiments/round9/scripts/dl.py cmdagree --label $2 --read $O/pai281/${TAG[$2]}-ag; r=$?
  [[ -s $O/results/cmdagree_$2.json ]] || die "cmdagree $2: no record"; exit $r ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read N1=$F_/${TAG[N1]}-281ag.csv N2a=$F_/${TAG[N2A]}-281ag.csv N2b=$F_/${TAG[N2B]}-281ag.csv \
      P1L=$F_/PCL1L-F-s0-281ag.csv NTV0s0=$F_/NTV0-L18-s0-281ag.csv R8Ks2=$F_/R8K-L18-s2-281ag.csv \
      --pairs N1:P1L N1:NTV0s0 N1:R8Ks2 N2a:N1 N2b:N1 N2a:N2b --out $O/results/pai281.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF (old scorer): NTV1 step 1 N1 / N2a / N2b vs P1L, NTV0-s0, R8K-s2" || die r281
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src   # new scorer (decision 453)
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads ${TAG[N1]}-ag,${TAG[N2A]}-ag,${TAG[N2B]}-ag \
      --workers 16 > $O/sv2.log 2>&1 || die "scorer_v2 new (log $O/sv2.log)"
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
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -6 $O/stages.txt; true ;;
*) sed -n 2,15p "$0"; exit 2 ;;
esac
