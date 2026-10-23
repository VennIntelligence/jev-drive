#!/usr/bin/env bash
# ROUND9 POOL-PILOT (plans/2026-10-23-pool-design.md, section 6 = registration): three arms, seed 0, the S pool limited to on1_L18RB0
# (wfile JT00S-rb0), then PAI 281 ACC + GAP (Harmonizer OFF). Box, repo root; every job through the pool, the whole chain in one submit;
# ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/pool_pilot.
#   P1  PCL1-F-s0    RB0L-F-s0's argv (19 500 steps from the shipped weights, eager) + NTV0's S flags, --on-start 10000
#   P1B PCL1B-F-s0   RB0-F-s0's argv (10 000 steps from the shipped weights, eager) + the same S flags, --on-start 5000
#   P0  PCL0-L18-s0  NTV0-L18-s0's argv (init RB0-F-s0, 9 500 steps, compiled), only --on-wfile -> JT00S-rb0 (diagnostic, not a candidate)
#   pcl.sh dry     the three argvs and their token diffs (no job)
#   pcl.sh all     wfile (CPU) -> gates (flag-off identity, pre-T identity of P1 / P1B, T = 30 post-T check, smokes, P0 determinism) ->
#                  3 trainings, each with an in-queue S / imit gradient-ratio gate (T .. T + 500) and a post gate (DONE, dev ADE <= 0.566)
#                  -> PAI 281 reads (NTV0's 8 chunk lists, g10, FAST, CONC 4) + tax -> report (scorer_v2 new, behavior_stats, verdict) -> $O/DONE
#   pcl.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: wfile | train ARM TAG [argv opts] | ident ... | smoke ... | det | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/pool_pilot; ME=experiments/round9/scripts/pcl.sh; PY=experiments/round9/scripts/pcl.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)   # P1 / P1B: eager, as RB0L-F / RB0-F (rbl.sh)
FC=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)                         # P0: compiled, as NTV0 (nos.sh)
P1=PCL1-F-s0; P1B=PCL1B-F-s0; P0=PCL0-L18-s0
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-pcl --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= nos.sh read281)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
dry) for a in P1 P1B P0; do echo "== $a"; $V $PY diff $a || die "diff $a"; done ;;
all)
  for X in $P1 ${P1}x10000 $P1B $P0 PCLID0-F-s0 PCLID1-F-s0 PCLID1B-F-s0 PCLT30-F-s0 PCL1SM-F-s0 PCL1SM-F-s0x30 PCL1BSM-F-s0 PCL0SM-L18-s0 PCL0SM2-L18-s0; do
    [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in NTV0-L18-s0 RB0-F-s0 RB0L-F-s0 NTV0NS-L18-s0; do [[ -s $F_/$f-281ag.csv ]] || die "no stored $f facts"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  w=$(sub pcl-wfile --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME wfile) || exit 1
  G=(); g() { local x; x=$(sub "$@") && [[ -n $x ]] || die "submit $1"; G+=($x); }
  g pcl-g0-flagoff --after $w "${FE[@]}" --timeout-h 1 -- bash $ME ident RB0-F-s0 75 100 strict g0_flagoff RB0F PCLID0-F-s0
  g pcl-g1-preT-P1 --after $w "${FE[@]}" --timeout-h 1 -- bash $ME ident RB0L-F-s0 100 125 new g1_preT_P1 P1 PCLID1-F-s0
  g pcl-g1-preT-P1B --after $w "${FE[@]}" --timeout-h 1 -- bash $ME ident RB0-F-s0 100 125 new g1_preT_P1B P1B PCLID1B-F-s0
  g pcl-g2-T30 --after $w "${FE[@]}" --timeout-h 1 -- bash $ME ident RB0L-F-s0 25 75 new g2_T30 P1 PCLT30-F-s0 --on-start 30
  g pcl-smoke-P1 --after $w "${FE[@]}" --timeout-h 1 -- bash $ME smoke P1 PCL1SM-F-s0 30 --steps 60 --on-start 30 --save-at 30
  g pcl-smoke-P1B --after $w "${FE[@]}" --timeout-h 1 -- bash $ME smoke P1B PCL1BSM-F-s0 - --steps 60 --on-start 30
  a0=$(sub pcl-smoke-P0 --after $w "${FC[@]}" --timeout-h 1 -- bash $ME smoke P0 PCL0SM-L18-s0 - --steps 60) || exit 1
  G+=($a0); g pcl-det-P0 --after $a0 "${FC[@]}" --timeout-h 1 -- bash $ME det
  GA=$(IFS=,; echo "${G[*]}")
  TAX=()
  for spec in "P1 $P1 10000 RB0L-F-s0 4" "P1B $P1B 5000 RB0-F-s0 3" "P0 $P0 0 - 2"; do
    read -r arm tag T ref th <<< "$spec"; [[ $arm == P0 ]] && F=("${FC[@]}") || F=("${FE[@]}")
    t=$(sub pcl-train-$tag --after $GA "${F[@]}" --timeout-h $th -- bash $ME train $arm $tag) || exit 1
    r=$(sub pcl-ratio-$arm --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h $((th + 2)) -- \
        $V $PY ratio --tag $tag --on-start $T --train-id $t --out $O/results/ratio_$arm.json) || exit 1
    [[ $ref == - ]] && RF=() || RF=(--ref $ref --on-start $T)
    p=$(sub pcl-post-$arm --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY post --tag $tag "${RF[@]}" --out $O/results/post_$arm.json) || exit 1
    y=($(read281 $tag $p)); [[ ${#y[@]} == 9 ]] || die "read281 $tag"
    TAX+=(${y[-1]})
    echo "$(date '+%F %T') $arm $tag: train $t ratio $r post $p 281 ${y[*]}" >> $O/stages.txt
  done
  r=$(sub pcl-report --after $(IFS=,; echo "${TAX[*]}") --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): wfile $w gates ${G[*]} report $r" | tee -a $O/stages.txt ;;
wfile) $V $PY wfile > $O/results/wfile.log 2>&1 || die wfile; cp $DATA_DIR/runs/round9/pool_pilot/w/JT00S-rb0.json $O/results/wfile.json ;;
train)  # train ARM TAG [argv opts]
  mapfile -t A < <($V $PY argv "${@:2}"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $3 (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
ident)  # ident REF UPTO STOP strict|new NAME ARM TAG [argv opts]: the arm's argv run until step STOP, scalars <= UPTO vs REF's stored run
  ref=$2 upto=$3 stop=$4 mode=$5 nm=$6 arm=$7 tag=$8; shift 8
  mapfile -t A < <($V $PY argv $arm $tag "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PY ident --tag $tag --ref $ref --upto $upto --stop $stop $([[ $mode == new ]] && echo --allow-new) --out $O/results/$nm.json -- $TR "${A[@]}"; r=$?
  if (( r == 0 )) && [[ $nm == g2_T30 ]]; then $V $PY tcheck --tag $tag --on-start 30 --out $O/results/g2_T30_batch.json; r=$?; fi
  rm -rf $RUNS/$tag; exit $r ;;
smoke)  # smoke ARM TAG SNAP|- [argv opts]: 60 steps + checks (pool rows per window when --on-start is given)
  arm=$2 tag=$3 snap=$4; shift 4
  mapfile -t A < <($V $PY argv $arm $tag "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" || die "smoke $tag trainer"
  T=$(sed -n 's/.*--on-start \([0-9]*\).*/\1/p' <<< "$*")
  $V $PY smoke --tag $tag $([[ $snap != - ]] && echo --snap ${tag}x$snap) ${T:+--on-start $T} --out $O/results/smoke_$arm.json; r=$?
  (( r == 0 )) && [[ $arm != P0 ]] && rm -rf $RUNS/$tag $RUNS/${tag}x$snap; exit $r ;;
det)  # P0: its argv twice on HEAD (compiled), every scalar and on_hash_60 equal
  mapfile -t A < <($V $PY argv P0 PCL0SM2-L18-s0 --steps 60); [[ ${#A[@]} -gt 5 ]] || die "argv"
  $TR "${A[@]}" || die "det trainer"
  $V $PY det --a PCL0SM-L18-s0 --b PCL0SM2-L18-s0 --out $O/results/det_P0.json; r=$?
  rm -rf $RUNS/PCL0SM-L18-s0 $RUNS/PCL0SM2-L18-s0; exit $r ;;
report)
  E=experiments/round2/scripts/r2.py
  $TR $E r281 --read P1=$F_/$P1-281ag.csv P1B=$F_/$P1B-281ag.csv P0=$F_/$P0-281ag.csv NTV0s0=$F_/NTV0-L18-s0-281ag.csv RB0F=$F_/RB0-F-s0-281ag.csv \
      RB0L=$F_/RB0L-F-s0-281ag.csv NTV0NS=$F_/NTV0NS-L18-s0-281ag.csv \
      --pairs P1:NTV0s0 P1:RB0F P1:RB0L P1:NTV0NS P1B:NTV0s0 P1B:RB0F P1B:RB0L P1B:NTV0NS P0:NTV0s0 P0:RB0F P0:RB0L P0:NTV0NS P1:P1B P1:P0 \
      --out $O/results/pai281.md --title "PAI 281, ACC + GAP, Harmonizer OFF: pool pilot (PCL1 / PCL1B / PCL0) vs NTV0-s0, RB0-F, RB0L, NTV0NS" || die r281
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src   # new scorer (decision 453)
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $P1-ag,$P1B-ag,$P0-ag --workers 16 > $O/sv2.log 2>&1 \
    || echo "$(date '+%F %T') scorer_v2 new failed (log $O/sv2.log); verdict without it" | tee -a $O/stages.txt
  $V $PY bs --out $O/bs --workers 16 > $O/bs.log 2>&1 \
    || echo "$(date '+%F %T') behavior_stats failed (log $O/bs.log)" | tee -a $O/stages.txt
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
