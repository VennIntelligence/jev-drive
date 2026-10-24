#!/usr/bin/env bash
# ROUND9 NTV1 S-NU (plans/2026-10-24-ntv1-snu.md = registration): refresh the S pool with the student's own nuPlan rollouts, 1 seed.
# Build: D3 = NT4D3-F-s0 rolls onpol-train at 18 decisions exactly as RB0-F-s0 did for on1_L18RB0 (ol_chain.sh roll1 / geo / tok / strip and
#   rm_build.py pool, the job bodies and declarations of the 10-13 pool records, empty job env) -> pool on1_L18D3 (part d3s); arm pool on1_JTSNU =
#   d3s + on1_JTDC2 without rb0; identity pool on1_JTSNUZ = on1_JTDC2 + d3s; lateral raster SNUv2 (df_lat build as L18NR1v2, d3s + rtb1 prefix);
#   draw files NSNU / NSNUZ. Gates: c0 (configs vs RB0's c0) before chunks 1-5, R0 (today's geo / rm_build on RB0's own roll = the stored rb0
#   files), R1 / R2, poolgate, merge checks, raster checks, draw files, argv diff, arm pre-T, weight-0 pair, smoke; ratio / post.
# Arm: NT5SNU-F-s0 = D3's argv with only --on / --on-wfile / --lat-file changed -> PAI 281 ACC + GAP (Harmonizer off, D3's serve) + val160 -> report.
# Box, repo root; every job through the pool; ids in $O/jobs.txt (+ $O/v160/jobs.txt), stages in $O/stages.txt.
#   snu.sh chain     preflight + R0 reproduction + roll c0 -> c0chk -> rest (submits everything else)
#   snu.sh wait      one run_in_background waiter: $O/DONE or a failed / cancelled job of this lane
#   snu.sh relat     plan note 2: attempt 1's raster aside, raster rebuilt with PYTHONHASHSEED=1 -> latchk / latgate -> the registered tail
#   job bodies: rest | lat | repro | merge A|Z | ident | t3run a|b | t3cmp | smoke | train | post | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv1_snu; O4=$DATA_DIR/runs/round9/ntv1_step4; ME=experiments/round9/scripts/snu.sh; PY=experiments/round9/scripts/snu.py
OL=experiments/onpol_long/scripts/ol_chain.sh; ONR=experiments/onpol_nuplan/scripts/on_rows.py; RB=experiments/route_mem/scripts/rm_build.py
LT=experiments/defect_fix/scripts/df_lat.py; RF=experiments/route_follow/scripts; PC=experiments/round9/scripts/pcl.py; GP=experiments/round9/scripts/dgc.py
PL=experiments/round9/scripts/p1l.py; VS=experiments/round9/scripts/v160.sh
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python
CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs; W=$DATA_DIR/runs/onpol_long; F_=$DATA_DIR/runs/pai_fix/facts
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
T=NT4D3-F-s0; N=L18D3; X=L18RB0X; ARM=NT5SNU-F-s0; TON=10000; PRIO=${PRIO:-14}
PA=on1_JTSNU; PZ=on1_JTSNUZ; LATA=$O/lat/SNUv2; FEAT=$DATA_DIR/runs/route_mem/feat
REST="on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both on1_PAIDC:pd:both on1_PAIDC2:pd2:both"   # on1_JTDC2's parts after rb0
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = dg4.sh (D3's 281 read)
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                    # eager, as D3
GT="NT5NUID-F-s0 NT5NUT3A-F-s0 NT5NUT3B-F-s0 NT5NUSM-F-s0"
mkdir -p $O/pool $O/checks $O/w $O/results $O/pai281 $O/lat
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round9-snu --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() { sub snu-roll-c$1 --vram 32 --cpu 8 --ram 60 --timeout-h 3 -- bash $OL roll1 $T $1; }   # = the 10-13 ol-roll-RB0-<i> jobs (env {})
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= dg4.sh read281, Harmonizer off)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
chain)
  [[ ! -e $O/DONE && ! -s $O/jobs.txt && ! -e $W/roll/$T ]] || die "$O / $W/roll/$T has a chain: never overwritten"
  for f in $RUNS/$T/ckpt-final.pt $RUNS/NT4D3-F-s1/ckpt-final.pt $W/roll/RB0-F-s0/manifest.json $DATA_DIR/runs/onpol_nuplan/rows/smoke10eY/r1.json \
           $DATA_DIR/runs/route_follow/poolgate.py $DATA_DIR/runs/defect_fix/lat/L18NR1v2/lat_gate_v2.json $O4/w/NTV1D2.npz $CR/on1_JTDC2/tab.npz \
           $O4/sv2/new/rollouts.csv $O4/sv2s1/new/rollouts.csv $O4/v160/sv2/new/rollouts.csv $F_/$T-281ag.csv $F_/NT4D3-F-s1-281ag.csv \
           $F_/$T-v160ag.csv $F_/NT4D3-F-s1-v160ag.csv; do [[ -s $f ]] || die "no $f"; done
  for i in 0 1 2 3 4 5; do [[ -s $W/lists/c$i.txt ]] || die "no list c$i"; done
  for X_ in $ARM $GT; do [[ ! -e $RUNS/$X_ && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X_ 2>/dev/null) ]] || die "$X_ exists"; done
  for p in $CR/on1_$N $CR/on1_$X $CR/$PA $CR/$PZ $DATA_DIR/runs/onpol_nuplan/rows/$N $DATA_DIR/runs/onpol_nuplan/rows/$X $W/rows/$N $LATA $FEAT/on1_$N.npz; do
    [[ ! -e $p ]] || die "$p exists"; done
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 200 )) || die "free $free GB < 200 GB"
  git rev-parse --short HEAD > $O/code.txt
  rp=$(sub snu-repro --vram 0.1 --cpu 48 --ram 96 --timeout-h 2 -- bash $ME repro) || exit 1; echo $rp > $O/repro.id
  r0=$(roll 0) || exit 1; echo $r0 > $O/roll0.id
  k=$(sub snu-c0chk --after $r0 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY c0chk) || exit 1
  r=$(sub snu-rest --after $k --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt)): repro $rp roll c0 $r0 c0chk $k rest $r (free $free GB)" | tee -a $O/stages.txt ;;
rest)
  h=$(git rev-parse --short HEAD); echo "$h rest" >> $O/code.txt
  ro=($(cat $O/roll0.id)); for i in 1 2 3 4 5; do ro+=($(roll $i)) || exit 1; done
  rs=$(sub snu-rollstats --after $(csv ${ro[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY rollstats) || exit 1
  g=$(sub snu-geo --after $(csv ${ro[@]}) --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $OL geo $T $N) || exit 1
  k=$(sub snu-tok --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash -c "bash $OL tok $N smoke10eY; echo tok-and-gate rc \$?; \
      test -f $CR/on1_$N@warp/front.npy -a -f $W/rows/$N/gates.json") || exit 1
  st=$(sub snu-strip --after $k --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $OL strip $T $N) || exit 1
  po=$(sub snu-pool --after $k --vram 0.5 --cpu 32 --ram 80 --timeout-h 1 -- $AP $RB pool --name $N --workers 32) || exit 1
  pg=$(sub snu-pgate --after $po --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $V $PY pgate) || exit 1
  mA=$(sub snu-merge-A --after $pg --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME merge A) || exit 1
  mZ=$(sub snu-merge-Z --after $pg --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME merge Z) || exit 1
  af=$(sub snu-afix --after $mA --vram 0.1 --cpu 4 --ram 120 --timeout-h 2 -- $TR $PY afix) || exit 1
  zf=$(sub snu-zfix --after $mZ --vram 0.1 --cpu 4 --ram 120 --timeout-h 2 -- $TR $PY zfix) || exit 1
  rw=$(sub snu-rowstats --after $mA --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $V $PY rowstats) || exit 1
  lb=$(sub snu-lat --after $af --vram 0.1 --cpu 64 --ram 160 --timeout-h 2 -- bash $ME lat) || exit 1
  lc=$(sub snu-latchk --after $lb --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY latchk) || exit 1
  lg=$(sub snu-latgate --after $lb --vram 0.1 --cpu 16 --ram 200 --timeout-h 2 -- $TR $PY latgate) || exit 1
  wt=$(sub snu-weights --after $af,$zf --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY weights) || exit 1
  wc=$(sub snu-wcheck --after $wt --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY wcheck --out $O/results/gw_draw_files.json) || exit 1
  df_=$(sub snu-diff --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- $V $PY diff) || exit 1
  gb=$(sub snu-g0b --after $wt,$lc "${FE[@]}" --timeout-h 1 -- bash $ME ident) || exit 1
  pa=$(sub snu-t3run-a "${FE[@]}" --timeout-h 1 -- bash $ME t3run a) || exit 1
  pb=$(sub snu-t3run-b --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3run b) || exit 1
  tc=$(sub snu-t3cmp --after $pa,$pb --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME t3cmp) || exit 1
  sm=$(sub snu-smoke --after $wc,$lc "${FE[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  GA=$(csv $(cat $O/repro.id) $pg $af $zf $lc $lg $wc $df_ $gb $tc $sm)
  tr=$(sub snu-train-$ARM --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train) || exit 1
  ra=$(sub snu-ratio --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $ARM --on-start $TON --train-id $tr \
       --out $O/results/ratio_arm.json) || exit 1
  pt=$(sub snu-post --after $tr,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post) || exit 1
  y=($(read281 $ARM $pt)); [[ ${#y[@]} == 9 ]] || die "read281 $ARM"
  v=$(V160_O=$O/v160 V160_TAGS=$ARM V160_AFTER=$pt PRIO=$PRIO bash $VS reads | tail -1) && [[ -n $v ]] || die "v160 $ARM"
  rp=$(sub snu-report --after ${y[-1]},$v,$rs,$rw,$st --vram 0.1 --cpu 16 --ram 48 --timeout-h 3 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') rest (code $h): rolls ${ro[*]} rollstats $rs geo $g tok $k strip $st pool $po pgate $pg merge $mA / $mZ afix $af zfix $zf" \
       "rowstats $rw lat $lb latchk $lc latgate $lg weights $wt wcheck $wc diff $df_ g0b $gb t3 $pa $pb $tc smoke $sm train $tr ratio $ra post $pt" \
       "281 ${y[*]} v160 $v report $rp" | tee -a $O/stages.txt ;;
lat)  # the lateral raster SNUv2 (df_lat build as L18NR1v2); PYTHONHASHSEED=1 (plan note 2: the build's own-lane pick follows the string-hash order on
      # a few rows; seed 1 reproduces L18NR1v2's rtb1 rows bit for bit)
  [[ ! -e $LATA ]] || die "$LATA exists"
  env PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 \
    $AP $LT build --pool $PA --rows $N:d3s,L18RTB1:rtb1 --out $LATA --workers 64 --dilate 6 --points ra --prefix ;;
relat)  # plan note 2: attempt 1's raster (built without a fixed hash seed; latchk failed) and its never-run read dirs go to $O/attempt1, then the
        # raster rebuilt (lat, PYTHONHASHSEED=1) -> latchk / latgate unchanged -> the registered tail (G0b, smoke, train, ratio, post, 281, val160, report)
  [[ ! -e $O/attempt1 ]] || die "$O/attempt1 exists"
  for id in $(awk '$2 ~ /^(snu-g0b|snu-smoke|snu-train-|snu-post|pai-nt5snu|pai_fix-tax-NT5SNU)/ {print $1}' $O/jobs.txt $O/v160/jobs.txt); do
    st=$($V -c "from jevdrive.cl import pool as P; print(P.snapshot()[0]['jobs'].get('$id', {}).get('state'))"); [[ $st == failed ]] || die "job $id is $st"; done
  mkdir -p $O/attempt1/v160 $O/attempt1/results
  mv $LATA $O/attempt1/SNUv2; mv $O/pai281/$ARM-ag $O/pai281/$ARM-ag.submit.txt $O/attempt1/; mv $O/v160/reads $O/attempt1/v160/
  cp $O/results/latchk.json $O/attempt1/results/latchk_attempt1.json; rm -f $O/results/latchk.json $O/results/latgate.json
  h=$(git rev-parse --short HEAD); echo "$h relat" >> $O/code.txt
  wc -l < $O/jobs.txt > $O/relat.start; wc -l < $O/v160/jobs.txt > $O/v160/relat.start      # wait: only the jobs submitted from here on
  lb=$(sub snu-lat2 --vram 0.1 --cpu 64 --ram 160 --timeout-h 2 -- bash $ME lat) || exit 1
  lc=$(sub snu-latchk2 --after $lb --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY latchk) || exit 1
  lg=$(sub snu-latgate2 --after $lb --vram 0.1 --cpu 16 --ram 200 --timeout-h 2 -- $TR $PY latgate) || exit 1
  gb=$(sub snu-g0b --after $lc "${FE[@]}" --timeout-h 1 -- bash $ME ident) || exit 1
  sm=$(sub snu-smoke --after $lc "${FE[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  GA=$(csv $lc $lg $gb $sm)
  tr=$(sub snu-train-$ARM --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train) || exit 1
  ra=$(sub snu-ratio --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $ARM --on-start $TON --train-id $tr \
       --out $O/results/ratio_arm.json) || exit 1
  pt=$(sub snu-post --after $tr,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post) || exit 1
  y=($(read281 $ARM $pt)); [[ ${#y[@]} == 9 ]] || die "read281 $ARM"
  v=$(V160_O=$O/v160 V160_TAGS=$ARM V160_AFTER=$pt PRIO=$PRIO bash $VS reads | tail -1) && [[ -n $v ]] || die "v160 $ARM"
  rp=$(sub snu-report --after ${y[-1]},$v --vram 0.1 --cpu 16 --ram 48 --timeout-h 3 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') relat (code $h): lat $lb latchk $lc latgate $lg g0b $gb smoke $sm train $tr ratio $ra post $pt 281 ${y[*]} v160 $v report $rp" \
    | tee -a $O/stages.txt ;;
repro)  # R0: today's on_rows geo + rm_build pool on RB0-F-s0's own stored roll (scratch name L18RB0X) vs the stored L18RB0 files
  $NAV $ONR geo --name $X --roll $W/roll/RB0-F-s0 --workers 48 || die "repro geo"
  $AP $RB pool --name $X --workers 32 || die "repro rm_build pool"
  $TR $PY repro ;;
merge)  # merge A (arm pool: d3s first) | merge Z (identity pool: on1_JTDC2's parts + d3s last)
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 100 )) || die "free $free GB < 100 GB"
  if [[ $2 == A ]]; then [[ ! -e $CR/$PA ]] || die "$PA exists"
    $TR $RF/rf_pait.py merge --pools on1_$N:d3s:self $REST --out ${PA#on1_} --nan-keys fut_reg || die "merge A"
  else [[ ! -e $CR/$PZ ]] || die "$PZ exists"
    $TR $RF/rf_pait.py merge --pools on1_L18RB0:rb0:self $REST on1_$N:d3s:self --out ${PZ#on1_} --nan-keys fut_reg || die "merge Z"; fi ;;
ident)  # G0b: the arm's argv to step 125, every scalar <= 100 vs the stored D3 bit for bit (held-out pool diagnostics excluded)
  tag=NT5NUID-F-s0; mapfile -t A < <($V $PY argv S $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $tag (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PY ident --tag $tag --ref $T --upto 100 --stop 125 --out $O/results/g0b_arm_preT.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$tag; exit $r ;;
t3run)  # weight-0 pair: D3's argv, 60 steps, T = 30; a on on1_JTDC2 / NTV1D2, b on on1_JTSNUZ / NSNUZ
  [[ $2 == a ]] && { X_=NT5NUT3A-F-s0; v=D3; } || { X_=NT5NUT3B-F-s0; v=Z; }
  mapfile -t A < <($V $PY argv $v $X_ --steps 60 --on-start 30 --save-at none); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') t3run $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/t3_$2.log 2>&1 || die "t3run $2 (log $O/t3_$2.log)" ;;
t3cmp)  # the pair; on a pass the identity pool (this lane's scratch) is removed
  $V $PY pair --a NT5NUT3A-F-s0 --b NT5NUT3B-F-s0 --upto 60 --out $O/results/t3_w0.json; r=$?; rm -rf $RUNS/NT5NUT3A-F-s0 $RUNS/NT5NUT3B-F-s0
  if [[ $r == 0 ]]; then rm -rf $CR/$PZ $CR/$PZ@warp; rm -f $FEAT/$PZ.npz; echo "$(date '+%F %T') t3cmp pass: removed $PZ" | tee -a $O/stages.txt; fi; exit $r ;;
smoke)  # the arm's argv, 60 steps, T = 30, --on-mix-log: composition (dgc.py smoke --var b, NSNU's shares) + lr (p1l.py lrcheck)
  tag=NT5NUSM-F-s0; mapfile -t A < <($V $PY argv S $tag --steps 60 --on-start 30 --save-at none --mix); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" > $O/smoke.log 2>&1 || die "smoke $tag trainer (log $O/smoke.log)"
  $V $GP smoke --var b --tag $tag --on-start 30 --want $O/results/w_NSNU.json --out $O/results/smoke_arm.json; r1=$?
  $V $PL lrcheck --tag $tag --on-start 30 --out $O/results/smoke_lr_arm.json; r2=$?
  rm -rf $RUNS/$tag; exit $(( r1 || r2 )) ;;
train)
  mapfile -t A < <($V $PY argv S $ARM); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $ARM (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
post)
  $V $PC post --tag $ARM --out $O/results/post_arm.json || die "post $ARM"
  $V $PL lrcheck --tag $ARM --on-start $TON --full --out $O/results/lr_full_arm.json || die "lr_full $ARM" ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read SNU=$F_/$ARM-281ag.csv D3s0=$F_/$T-281ag.csv D3s1=$F_/NT4D3-F-s1-281ag.csv \
      R8Ks2=$F_/R8K-L18-s2-281ag.csv P1L=$F_/PCL1L-F-s0-281ag.csv --pairs SNU:D3s0 SNU:D3s1 SNU:R8Ks2 SNU:P1L --out $O/results/pai281.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF (old scorer): NTV1 S-NU vs D3-s0 / s1, R8K-s2, P1L" || die r281
  rm -rf $O/sv2 $O/v160/sv2
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $ARM-ag --workers 16 > $O/sv2.log 2>&1 \
    || die "scorer_v2 new 281 (log $O/sv2.log)"
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/v160/sv2 --reads $ARM-v160 --workers 16 > $O/v160/sv2.log 2>&1 \
    || die "scorer_v2 new val160 (log $O/v160/sv2.log)"
  $V experiments/round9/scripts/dg4.py bs --out $O/bs --tags $ARM > $O/bs.log 2>&1 \
    || echo "$(date '+%F %T') behavior_stats failed (log $O/bs.log); verdict without the highway row" | tee -a $O/stages.txt
  $V $PY verdict --out $O/results/verdict > $O/verdict.log 2>&1 || die "verdict (log $O/verdict.log)"
  $V $PY vverdict --out $O/results/v160 > $O/vverdict.log 2>&1 || die "vverdict (log $O/vverdict.log)"
  $V $PY build > $O/build.log 2>&1 || die "build (log $O/build.log)"
  date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt $O/v160/jobs.txt <<'PY'
import os, sys
from jevdrive.cl import pool as P
last = {}
for f in sys.argv[1:]:
    s0 = os.path.join(os.path.dirname(f), "relat.start")          # plan note 2: attempt 1's failed tail is history
    k0 = int(open(s0).read()) if os.path.exists(s0) else 0
    for l in (open(f).readlines()[k0:] if os.path.exists(f) else []):
        if len(l.split()) > 1:
            last[f + ":" + l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -4 $O/stages.txt; true ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
