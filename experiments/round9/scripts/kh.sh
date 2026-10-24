#!/usr/bin/env bash
# ROUND9 NTV1 K / H (plans/2026-10-24-ntv1-kh.md = registration): D1 (NT3D1-F-s0) + R8K's pacing floor (K), seed 0; H (twin tokens) not run (cost).
#   K: P1L's DAgger states (on1_PAIDC) relabelled with teacher b5 + F2 (joint.py, verbatim) -> pools on1_JTDCK / on1_JTDCI, draw NTV1K ->
#      NTK1-F-s0 = D1's argv with --on on1_JTDCK --on-wfile NTV1K, from the shipped weights, eager, 19 500 steps -> PAI 281 Harmonizer OFF and ON -> verdict.
# Box, repo root; every job through the pool, the whole chain in one submit; ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/ntv1_kh.
#   kh.sh dry     K's token diff against NT3D1-F-s0's stored argv (exit 1 on an unregistered diff)
#   kh.sh all     teach b5 -> label (G1, pools) -> weights -> G-W | twincov | D1 argv vs stored D1 | K pre-T vs stored D1 | T3 pairs (switch off, weight 0)
#                 | smokes D1 / K + K switch -> training -> ratio / post gates -> PAI 281 OFF + ON reads + tax -> report (scorer_v2 new, verdict) -> $O/DONE
#   kh.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: teach | ident | t3run | t3cmp | smoke | train | post | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv1_kh; ME=experiments/round9/scripts/kh.sh; PY=experiments/round9/scripts/kh.py; PC=experiments/round9/scripts/pcl.py
GP=experiments/round9/scripts/dgc.py; PL=experiments/round9/scripts/p1l.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = P1L's / D1's read
NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src                                                       # new scorer (decision 453)
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD DAG_TV=b3 DAG_O=$DATA_DIR/runs/round9/ntv1_step2)
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                          # eager, as P1L / D1
D1=NT3D1-F-s0; K=NTK1-F-s0; P1L=PCL1L-F-s0; T=10000; W0=runs/round9/ntv1_step2/w; R=$O/results
PRIO=${PRIO:-14}
mkdir -p $O/pool $R $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-ntv1kh --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read281() {  # read281 <tag> <after> <ag|agH> -> "<chunk ids...> <tax id>" (= dgt.sh read281)
  local D=$O/pai281/$1-$3 ids t sv="$SERVE" ex=(); [[ ! -e $D ]] || die "$D exists"
  [[ $3 == agH ]] && sv="$SERVE HARMONIZER=1" && ex=(VRAM=30 RAM=40)
  env "${ex[@]}" AFTER=$2 CHUNKS_FROM=$CH SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $D $1 \
    > $D.submit.txt || die "pai submit $1 $3"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $D.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1 $3: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $D.submit.txt | sed "s/\$/-$3/" >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281$3 $D | tail -1) && [[ -n $t ]] || die "tax $1 $3"
  echo "$t pai_fix-tax-$1-281$3" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
dry) $V $PY diff ;;
all)
  bash $ME dry > $O/dry.txt 2>&1 || die "argv differs (see $O/dry.txt)"
  for X in $K NTKID1-F-s0 NTKIDK-F-s0 NTKT3A-F-s0 NTKT3B-F-s0 NTKT3C-F-s0 NTKT3D-F-s0 NTKSM-D1-F-s0 NTKSM-K-F-s0; do
    [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for p in on1_JTDCK on1_JTDCI; do [[ ! -e $DATA_DIR/runs/op_parity/cache/$p ]] || die "$p exists"; done
  for f in $W0/NTV1D.npz $W0/NTV1D0.npz runs/op_parity/cache/on1_JTDC/tab.npz runs/op_parity/cache/on1_PAIHTW/tab.npz \
           runs/round9/ntv1_step3/sv2/new/rollouts.csv runs/round9/ntv1_step3/sv2c/new/rollouts.csv; do [[ -s $DATA_DIR/$f ]] || die "no $f"; done
  for f in $D1-281ag $D1-281agH $P1L-281ag NTV0-L18-s0-281ag R8K-L18-s2-281ag NTV0-L18-s0-281agH R8K-L18-s2-281agH; do
    [[ -s $F_/$f.csv ]] || die "no stored $f facts"; done
  grep -q "\"$K-agH\"" experiments/alpasim/scripts/scorer_v2.py || die "scorer_v2 READS lacks $K"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  G=(); g() { local x; x=$(sub "$@") && [[ -n $x ]] || die "submit $1"; G+=($x); }
  te=$(sub kh-teach --vram 0.1 --cpu 48 --ram 96 --timeout-h 4 -- bash $ME teach) || exit 1
  lb=$(sub kh-label --after $te --vram 0.1 --cpu 4 --ram 96 --timeout-h 2 -- $V $PY label) || exit 1
  wt=$(sub kh-weights --after $lb --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY weights) || exit 1
  G+=($te $lb $wt)
  g kh-wcheck --after $wt --vram 0.1 --cpu 2 --ram 48 --timeout-h 1 -- $V $PY wcheck --out $R/gw_draw_file.json
  g kh-twincov --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $V $PY twincov
  g kh-ident-D1 "${FE[@]}" --timeout-h 1 -- bash $ME ident d1_argv_vs_stored_D1 NTKID1-F-s0 D1
  g kh-ident-K --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME ident k_preT_vs_stored_D1 NTKIDK-F-s0 K
  ta=$(sub kh-t3run-a "${FE[@]}" --timeout-h 1 -- bash $ME t3run NTKT3A-F-s0 on1_JTDC $W0/NTV1D.npz) || exit 1
  tb=$(sub kh-t3run-b --after $lb "${FE[@]}" --timeout-h 1 -- bash $ME t3run NTKT3B-F-s0 on1_JTDCI $W0/NTV1D.npz) || exit 1
  tc=$(sub kh-t3run-c "${FE[@]}" --timeout-h 1 -- bash $ME t3run NTKT3C-F-s0 on1_JTDC $W0/NTV1D0.npz) || exit 1
  td=$(sub kh-t3run-d --after $lb "${FE[@]}" --timeout-h 1 -- bash $ME t3run NTKT3D-F-s0 on1_JTDCK $W0/NTV1D0.npz) || exit 1
  g kh-t3-switchoff --after $ta,$tb --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME t3cmp NTKT3A-F-s0 NTKT3B-F-s0 t3_switch_off
  g kh-t3-weight0 --after $tc,$td --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME t3cmp NTKT3C-F-s0 NTKT3D-F-s0 t3_weight0
  s1=$(sub kh-smoke-D1 --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke D1 w_NTV1D.json) || exit 1
  s2=$(sub kh-smoke-K --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke K w_NTV1K.json) || exit 1
  G+=($s1 $s2)
  g kh-ksw --after $s1,$s2 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash -c "$V $PY ksw --a NTKSM-D1-F-s0 --b NTKSM-K-F-s0 --on-start 30 --out $R/ksw_K_switch.json; r=\$?; rm -rf $RUNS/NTKSM-D1-F-s0 $RUNS/NTKSM-K-F-s0; exit \$r"
  GA=$(IFS=,; echo "${G[*]}")
  t=$(sub kh-train-$K --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train) || exit 1
  r=$(sub kh-ratio-K --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $K --on-start $T --train-id $t --out $R/ratio_K.json) || exit 1
  p=$(sub kh-post-K --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post) || exit 1
  a=($(read281 $K $p ag)); [[ ${#a[@]} == 9 ]] || die "read281 OFF"
  b=($(read281 $K $p agH)); [[ ${#b[@]} == 9 ]] || die "read281 ON"
  rp=$(sub kh-report --after ${a[-1]},${b[-1]} --vram 0.1 --cpu 16 --ram 48 --timeout-h 3 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): gates ${G[*]} T3 runs $ta $tb $tc $td train $t ratio $r post $p OFF ${a[*]} ON ${b[*]} report $rp" | tee -a $O/stages.txt ;;
teach) "${EA[@]}" $AP $PY teach --workers 46 ;;
ident)  # ident NAME TAG ARM: the argv to step 125, every scalar <= 100 vs the stored NT3D1-F-s0 bit for bit, no extra / missing key
  nm=$2 tag=$3 arm=$4
  mapfile -t A < <($V $PY argv $arm $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PC ident --tag $tag --ref $D1 --upto 100 --stop 125 --out $R/$nm.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$tag; exit $r ;;
t3run)  # t3run TAG POOL WFILE: D1's argv, 60 steps, T = 30, on POOL / WFILE
  mapfile -t A < <($V $PY argv D1 $2 --steps 60 --on-start 30 --save-at none --pool $3 --wfile $4); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') t3run $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/t3_$2.log 2>&1 || die "t3run $2 (log $O/t3_$2.log)" ;;
t3cmp) $V $GP pair --a $2 --b $3 --upto 60 --out $R/$4.json; r=$?; rm -rf $RUNS/$2 $RUNS/$3; exit $r ;;
smoke)  # smoke ARM WANT: 60 steps, T = 30, --on-mix-log; composition (dgc.py smoke --var b, --want the draw file's group shares) + lr; run dir kept for ksw
  m=$2; tag=NTKSM-$m-F-s0
  mapfile -t A < <($V $PY argv $m $tag --steps 60 --on-start 30 --save-at none --mix); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" > $O/smoke_$m.log 2>&1 || die "smoke $tag trainer (log $O/smoke_$m.log)"
  $V $GP smoke --var b --tag $tag --on-start 30 --want $R/$3 --out $R/smoke_$m.json; r1=$?
  $V $PL lrcheck --tag $tag --on-start 30 --out $R/smoke_lr_$m.json; r2=$?
  exit $(( r1 || r2 )) ;;
train) mapfile -t A < <($V $PY argv K $K); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $K (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
post) $V $PC post --tag $K --out $R/post_K.json || die "post $K"
  $V $PL lrcheck --tag $K --on-start $T --full --out $R/lr_full_K.json || die "lr_full $K" ;;
report)
  rm -rf $O/sv2
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $K-ag,$K-agH --workers 16 > $O/sv2.log 2>&1 \
    || die "scorer_v2 new (log $O/sv2.log)"
  $V $PY verdict --out $R/verdict > $O/verdict.log 2>&1 || die "verdict (log $O/verdict.log)"
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
*) sed -n 2,10p "$0"; exit 2 ;;
esac
