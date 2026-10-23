#!/usr/bin/env bash
# ROUND9 NTV1 step 3 (plans/2026-10-24-ntv1-step3.md = registration): two arms on P1L's recipe (PCL1L-F-s0), seed 0, eager, 19 500 steps:
#   D1 = P1L + P1L's own DAgger rows (on1_JTDC, NTV1D: 32 S + 8 DAgger per batch from step 10 000, the batch size kept: 75 navtrain after T)
#   D2 = D1 + --ho-distill 0 (decision 464's switch: the S rows' distill_ho x 0)
# then PAI 281 ACC + GAP (Harmonizer OFF, P1L's serve, NTV0's chunk lists). Candidate rule: an arm with new mean >= 0.86 -> its seed 1 + Harmonizer-ON
# reads of both seeds, submitted from the report job. Box, repo root; every job through the pool, the whole chain in one submit; ids in $O/jobs.txt,
# stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/ntv1_step3.
#   dgt.sh dry     both arms' token diffs against PCL1L-F-s0's stored argv (exit 1 on an unregistered diff)
#   dgt.sh all     gates (G0a P1L flag off, G0b D1 pre-T vs stored P1L, D2 pre-T reported, G0c weight-0 pair, G-W draw file, G-R row treatment,
#                  60-step smokes T = 30 + lr, D2 switch) -> 2 trainings -> per arm S / imit ratio gate + post gate -> PAI 281 reads + tax ->
#                  report (r281, scorer_v2 new, behavior_stats, verdict) -> cand (no candidate: $O/DONE; else the tail -> creport -> $O/DONE)
#   dgt.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: ident | t3run | t3cmp | smoke | dsw | train | post | report | cand | sv2c | creport
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv1_step3; ME=experiments/round9/scripts/dgt.sh; PY=experiments/round9/scripts/dgt.py; PC=experiments/round9/scripts/pcl.py
GP=experiments/round9/scripts/dgc.py; PL=experiments/round9/scripts/p1l.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = P1L's read
NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src                                                       # new scorer (decision 453)
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                          # eager, as P1L
declare -A TAG=([D1]=NT3D1-F-s0 [D2]=NT3D2-F-s0); ARMS="D1 D2"; T=10000; P1L=PCL1L-F-s0
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-ntv1s3 --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <after> <ag|agH> -> "<chunk ids...> <tax id>" (= ntv.sh read281; agH = Harmonizer ON as diag.sh)
  local R=$O/pai281/$1-$3 ids t sv="$SERVE" ex=(); [[ ! -e $R ]] || die "$R exists"
  [[ $3 == agH ]] && sv="$SERVE HARMONIZER=1" && ex=(VRAM=30 RAM=40)
  env "${ex[@]}" AFTER=$2 CHUNKS_FROM=$CH SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1 $3"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1 $3: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281$3 $R | tail -1) && [[ -n $t ]] || die "tax $1 $3"
  echo "$t pai_fix-tax-$1-281$3" >> $O/jobs.txt; echo $ids $t; }
stag() { [[ ${2:-0} == 0 ]] && echo ${TAG[$1]} || echo ${TAG[$1]/-s0/-s$2}; }   # stag ARM [SEED]
case ${1:-} in
dry) for m in $ARMS; do $V $PY diff $m || die "diff $m"; done ;;
all)
  bash $ME dry > $O/dry.txt 2>&1 || die "argv differs (see $O/dry.txt)"
  for X in ${TAG[D1]} ${TAG[D2]} NT3D1-F-s1 NT3D2-F-s1 NT3ID0-F-s0 NT3ID1-F-s0 NT3ID2-F-s0 NT3T3A-F-s0 NT3T3B-F-s0 NT3SM-D1-F-s0 NT3SM-D2-F-s0; do
    [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in runs/round9/ntv1_step2/w/NTV1D.npz runs/round9/ntv1_step2/w/NTV1D0.npz runs/round9/pool_pilot/w/JT00S-rb0.npz runs/op_parity/cache/on1_JTDC/tab.npz \
           runs/round9/p1l/sv2/new/rollouts.csv runs/round9/interp/sv2/new/rollouts.csv runs/round9/p1l/bs/steps_$P1L.parquet; do
    [[ -s $DATA_DIR/$f ]] || die "no $f"; done
  for f in $P1L NTV0-L18-s0 R8K-L18-s2 INTP75-NR NTV0-L18-s0-281agH R8K-L18-s2-281agH; do [[ $f == *agH ]] && g=$f || g=$f-281ag; [[ -s $F_/$g.csv ]] || die "no stored $g facts"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  G=(); g() { local x; x=$(sub "$@") && [[ -n $x ]] || die "submit $1"; G+=($x); }
  g dgt-wcheck --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- $V $PY wcheck --out $O/results/gw_draw_file.json
  g dgt-rowcheck --vram 0.1 --cpu 2 --ram 96 --timeout-h 1 -- $V $PY rowcheck --out $O/results/gr_row_treatment.json
  g dgt-g0a-flagoff-P1L "${FE[@]}" --timeout-h 1 -- bash $ME ident g0a_flagoff_P1L NT3ID0-F-s0 $PC P1L hard
  g dgt-g0b-preT-D1 "${FE[@]}" --timeout-h 1 -- bash $ME ident g0b_preT_D1 NT3ID1-F-s0 $PY D1 hard
  x=$(sub dgt-rep-preT-D2 "${FE[@]}" --timeout-h 1 -- bash $ME ident rep_preT_D2 NT3ID2-F-s0 $PY D2 soft) || exit 1
  pa=$(sub dgt-t3run-a "${FE[@]}" --timeout-h 1 -- bash $ME t3run a) || exit 1
  pb=$(sub dgt-t3run-b "${FE[@]}" --timeout-h 1 -- bash $ME t3run b) || exit 1
  g dgt-g0c-w0 --after $pa,$pb --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME t3cmp
  declare -A SM; for m in $ARMS; do SM[$m]=$(sub dgt-smoke-$m "${FE[@]}" --timeout-h 1 -- bash $ME smoke $m) || exit 1; G+=(${SM[$m]}); done
  g dgt-dsw --after ${SM[D1]},${SM[D2]} --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME dsw
  GA=$(IFS=,; echo "${G[*]}")
  X=()
  for m in $ARMS; do
    t=$(sub dgt-train-${TAG[$m]} --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train $m 0) || exit 1
    r=$(sub dgt-ratio-$m --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag ${TAG[$m]} --on-start $T --train-id $t \
        --out $O/results/ratio_$m.json) || exit 1
    p=$(sub dgt-post-$m --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $m 0) || exit 1
    y=($(read281 ${TAG[$m]} $p ag)); [[ ${#y[@]} == 9 ]] || die "read281 $m"
    X+=(${y[-1]})
    echo "$(date '+%F %T') $m (code $h): train $t ratio $r post $p 281 ${y[*]}" | tee -a $O/stages.txt; done
  rp=$(sub dgt-report --after $(IFS=,; echo "${X[*]}") --vram 0.1 --cpu 16 --ram 48 --timeout-h 3 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): gates ${G[*]} D2-preT $x T3 runs $pa $pb report $rp" | tee -a $O/stages.txt ;;
ident)  # ident NAME TAG SCRIPT ARM hard|soft: the argv to step 125, every scalar <= 100 vs the stored PCL1L-F-s0 bit for bit, no extra / missing key
  nm=$2 tag=$3 scr=$4 arm=$5 mode=$6
  mapfile -t A < <($V $scr argv $arm $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PC ident --tag $tag --ref $P1L --upto 100 --stop 125 --out $O/results/$nm.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$tag; [[ $mode == soft ]] && exit 0; exit $r ;;
t3run)  # G0c = step 2a's T3 (b) on HEAD: P1L's argv, 60 steps, T = 30; a on on1_JT / JT00S-rb0, b on on1_JTDC / NTV1D0
  [[ $2 == a ]] && { X=NT3T3A-F-s0; v=p1l; } || { X=NT3T3B-F-s0; v=t3; }
  mapfile -t A < <($V $GP argv $v $X --steps 60 --on-start 30 --save-at none); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') t3run $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/t3_$2.log 2>&1 || die "t3run $2 (log $O/t3_$2.log)" ;;
t3cmp) $V $GP pair --a NT3T3A-F-s0 --b NT3T3B-F-s0 --upto 60 --out $O/results/g0c_w0.json; r=$?; rm -rf $RUNS/NT3T3A-F-s0 $RUNS/NT3T3B-F-s0; exit $r ;;
smoke)  # smoke ARM: 60 steps, T = 30, --on-mix-log; composition (dgc.py smoke --var b) + lr (p1l.py lrcheck)
  m=$2; tag=NT3SM-$m-F-s0
  mapfile -t A < <($V $PY argv $m $tag --steps 60 --on-start 30 --save-at none --mix); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" > $O/smoke_$m.log 2>&1 || die "smoke $tag trainer (log $O/smoke_$m.log)"
  $V $GP smoke --var b --tag $tag --on-start 30 --out $O/results/smoke_$m.json; r1=$?
  $V $PL lrcheck --tag $tag --on-start 30 --out $O/results/smoke_lr_$m.json; r2=$?
  rm -rf $RUNS/$tag; exit $(( r1 || r2 )) ;;
dsw) $V $PY dsw --a NT3SM-D1-F-s0 --b NT3SM-D2-F-s0 --on-start 30 --out $O/results/dsw_D2_switch.json ;;
train)  # train ARM SEED
  m=$2 s=$3; tag=$(stag $m $s); mapfile -t A < <($V $PY argv $m $tag $([[ $s != 0 ]] && echo --seed $s)); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $tag (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
post)  # post ARM SEED: DONE, dev ADE <= 0.566, lr over the whole run
  m=$2 s=$3; tag=$(stag $m $s); sf=$([[ $s != 0 ]] && echo _s$s)
  $V $PC post --tag $tag --out $O/results/post_$m$sf.json || die "post $tag"
  $V $PL lrcheck --tag $tag --on-start $T --full --out $O/results/lr_full_$m$sf.json || die "lr_full $tag" ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read D1=$F_/${TAG[D1]}-281ag.csv D2=$F_/${TAG[D2]}-281ag.csv P1L=$F_/$P1L-281ag.csv \
      NTV0s0=$F_/NTV0-L18-s0-281ag.csv R8Ks2=$F_/R8K-L18-s2-281ag.csv INTP75=$F_/INTP75-NR-281ag.csv \
      --pairs D1:P1L D2:P1L D2:D1 D1:NTV0s0 D1:R8Ks2 D1:INTP75 D2:NTV0s0 D2:R8Ks2 D2:INTP75 --out $O/results/pai281.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF (old scorer): NTV1 step 3 D1 / D2 vs P1L, NTV0-s0, R8K-s2, INTP75-NR" || die r281
  rm -rf $O/sv2
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads ${TAG[D1]}-ag,${TAG[D2]}-ag \
      --workers 16 > $O/sv2.log 2>&1 || die "scorer_v2 new (log $O/sv2.log)"
  $V $PY bs --out $O/bs --tags ${TAG[D1]},${TAG[D2]} > $O/bs.log 2>&1 \
    || echo "$(date '+%F %T') behavior_stats failed (log $O/bs.log); verdict without the arms' highway rows" | tee -a $O/stages.txt
  $V $PY verdict --dir $O --out $O/results/verdict || die verdict
  bash $ME cand ;;
cand)  # candidate rule (plan section 5): $O/CAND from the verdict; none -> DONE; else seed 1 + Harmonizer-ON reads of both seeds -> creport -> DONE
  c=$(cat $O/CAND); if [[ $c == none ]]; then echo "$(date '+%F %T') no candidate (new mean < 0.86): done" | tee -a $O/stages.txt; date '+%F %T' > $O/DONE; exit 0; fi
  [[ -n ${TAG[$c]:-} ]] || die "CAND $c"; s0=${TAG[$c]}; s1=$(stag $c 1)
  [[ ! -e $RUNS/$s1 && -z $(ls -d $DATA_DIR/runs/route_mem/train-$s1 2>/dev/null) ]] || die "$s1 exists"
  h=$(git rev-parse --short HEAD); echo "$h cand $c" >> $O/code.txt
  t=$(sub dgt-train-$s1 "${FE[@]}" --timeout-h 5 -- bash $ME train $c 1) || exit 1
  r=$(sub dgt-ratio-cand-s1 --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $s1 --on-start $T --train-id $t --out $O/results/ratio_cand_s1.json) || exit 1
  p=$(sub dgt-post-cand-s1 --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $c 1) || exit 1
  a=($(read281 $s0 "" agH)); [[ ${#a[@]} == 9 ]] || die "read281 $s0 ON"
  b=($(read281 $s1 $p ag)); [[ ${#b[@]} == 9 ]] || die "read281 $s1 OFF"
  d=($(read281 $s1 $p agH)); [[ ${#d[@]} == 9 ]] || die "read281 $s1 ON"
  v=$(sub dgt-sv2c --after ${a[-1]},${b[-1]},${d[-1]} --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- bash $ME sv2c $s0 $s1) || exit 1
  f=$(sub dgt-creport --after $v --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME creport) || exit 1
  echo "$(date '+%F %T') cand $c (code $h): train $s1 $t ratio $r post $p; $s0 ON ${a[*]}; $s1 OFF ${b[*]}; $s1 ON ${d[*]}; sv2c $v creport $f" | tee -a $O/stages.txt ;;
sv2c) rm -rf $O/sv2c
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2c --reads $2-agH,$3-ag,$3-agH --workers 16 > $O/sv2c.log 2>&1 \
    || die "scorer_v2 new cand (log $O/sv2c.log)" ;;
creport) $V $PY cverdict --dir $O --out $O/results/cand > $O/creport.log 2>&1 || die "cverdict (log $O/creport.log)"; date '+%F %T' > $O/DONE ;;
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
*) sed -n 2,14p "$0"; exit 2 ;;
esac
