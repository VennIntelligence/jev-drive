#!/usr/bin/env bash
# ROUND9 NTV1 step 4 (plans/2026-10-24-ntv1-step4.md = registration): DAgger iteration 2, 1 seed.
# 4a: D1 = NT3D1-F-s0 rolled under step 2a's submission serve (ACC + GAP, g10, ALPASIM_FAST=1, CONC 4, frame dumps, KEEP_ASL) on the same 719 curated
#     train scenes and chunk lists; round 1's teacher b3. The roll -> rows -> teach -> clean -> cache -> pool -> label (T4) -> fwd / r1in job bodies are
#     round6 dag.sh's, unchanged, under this lane's env (as dgc.sh) -> pool on1_PAIDC2 (part pd2) -> merge on1_JTDC2 = on1_JTDC's parts + pd2 -> draw files
#     NTV1D2Z (weight 0) / NTV1D2 (32 S + 8 DAgger, the 8 uniform over both iterations' drawable rows) -> gates (T1 / T2, chk1, T4, input path, mfix,
#     draw files, D1 argv vs its stored run, weight-0 pair, D3 pre-T, smoke, argv diff).
# 4b: D3 = NT4D3-F-s0 = D1's argv with only the pool / draw file changed, from the shipped weights, seed 0 -> ratio / post gates -> PAI 281 ACC + GAP
#     (Harmonizer off, P1L's serve) -> report (r281, scorer_v2 new, behavior_stats, verdict) -> cand (D3 new >= 0.875: seed 1 + val160 of both seeds).
# Box, repo root; every job through the pool; ids in $O/jobs.txt (+ $O/v160/jobs.txt), stages in $O/stages.txt.
#   dg4.sh chain     preflight + lists check + T1 / T2 + c000 (6 scenes) roll -> rows -> teach / clean -> chk1 -> rest (submits all the rest)
#   dg4.sh wait      one run_in_background waiter: $O/DONE, $O/VERDICT or a failed / cancelled job of this lane
#   job bodies: rest | merge | ident | t3run | t3cmp | smoke | tax | train | post | report | cand | vreport
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv1_step4; O2=$DATA_DIR/runs/round9/ntv1_step2; O3=$DATA_DIR/runs/round9/ntv1_step3; ME=experiments/round9/scripts/dg4.sh
DS=experiments/round6/scripts/dag.sh; DG=experiments/round6/scripts/dag.py; PY=experiments/round9/scripts/dg4.py; PC=experiments/round9/scripts/pcl.py
GP=experiments/round9/scripts/dgc.py; PL=experiments/round9/scripts/p1l.py; VS=experiments/round9/scripts/v160.sh
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; RF=experiments/route_follow/scripts
CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src; RES=experiments/round9/results/ntv1_step4
T=NT3D1-F-s0; L=${T,,}; NP=PAIDC2; NM=JTDC2; TV=b3; D3=NT4D3-F-s0; D3S1=NT4D3-F-s1; TON=10000; PRIO=${PRIO:-14}
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = dgc.sh (step 2a) / P1L's read
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both on1_PAIDC:pd:both"   # = on1_JTDC (dgc.sh merge)
TCFG="$DATA_DIR/runs/round5/s3/roll/A/runs/*/sim/generated-user-config-0.yaml,$DATA_DIR/runs/round5/d3/roll/A/runs/*/sim/generated-user-config-0.yaml"
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
export DAG_O=$O DAG_T=$T DAG_NP=$NP DAG_RES=$RES DAG_TV=$TV
DENV=(--env DAG_O=$O --env DAG_T=$T --env DAG_NP=$NP --env DAG_RES=$RES --env DAG_TV=$TV)
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                    # eager, as P1L / D1
GT="NT4ID1-F-s0 NT4ID3-F-s0 NT4T3A-F-s0 NT4T3B-F-s0 NT4SM-F-s0"
mkdir -p $O/pool $O/checks $O/w $O/tax $O/results $O/pai281 $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round9-ntv1s4 --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "${DENV[@]}" "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() {  # -> "<id> <chunk>" lines of the chunks submitted now (pai_box.sh skips finished / pending chunks); = dgc.sh roll
  local f=$O/roll/submit.$(date +%s%N).txt
  SERVE="$SERVE PAI_DUMP=100000" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO KEEP_ASL=1 bash experiments/alpasim/scripts/pai_box.sh submit $O/roll $T > $f \
    || die "roll submit"
  grep -q 'not submitted' $f && die "roll: $(grep 'not submitted' $f)"
  awk '$2 ~ /^pai-/ {print $1, "roll-" substr($2, length($2) - 3)}' $f >> $O/jobs.txt
  awk '$2 ~ /^pai-/ {print $1, substr($2, length($2) - 3)}' $f; }
chunk_tail() {  # chunk_tail <roll id> <chunk> -> "<rows id> <teach id> <clean id>" (clean only after rows: rows checks every file's tag == $T)
  local x y z
  x=$(sub dg4-rows-$2 --after $1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $DS rows $2) || exit 1
  y=$(sub dg4-teach-$2 --after $x --vram 0.1 --cpu 8 --ram 32 --timeout-h 2 -- bash $DS teach $2) || exit 1
  z=$(sub dg4-clean-$2 --after $x --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $DS clean $2) || exit 1
  echo "$x $y $z"; }
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= dgt.sh read281, Harmonizer off)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
v160() {  # v160 <tag> [after] -> tax id (v160.sh reads: its chunk lists and serve, under $O/v160)
  local x; x=$(V160_O=$O/v160 V160_TAGS=$1 V160_AFTER=${2:-} PRIO=$PRIO bash $VS reads | tail -1) && [[ -n $x ]] || die "v160 $1"; echo $x; }
case ${1:-} in
chain)
  [[ ! -e $O/roll && ! -e $O/DONE && ! -s $O/jobs.txt ]] || die "$O has a chain: never overwritten"
  [[ -f $RUNS/$T/ckpt-final.pt ]] || die "no $T"
  for X in $D3 $D3S1 $GT; do [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  for p in on1_$NP on1_$NM; do [[ ! -e $CR/$p ]] || die "$p exists"; done
  [[ -z $(ls -d $DATA_DIR/tmp/dag_stage/${L}_* 2>/dev/null) ]] || die "staging dirs ${L}_* exist"
  for f in $F_/PCL1L-F-s0-281ag.csv $F_/NTV0-L18-s0-281ag.csv $F_/R8K-L18-s2-281ag.csv $F_/INTP75-NR-281ag.csv $F_/$T-281ag.csv $F_/NT3D1-F-s1-281ag.csv \
           $F_/R8K-L18-s2-v160ag.csv $F_/$T-v160ag.csv $F_/PCL1L-F-s0-v160ag.csv $O3/sv2/new/rollouts.csv $O3/sv2c/new/rollouts.csv \
           $DATA_DIR/runs/round9/val160/sv2/new/rollouts.csv $O2/tax/roll.csv $O2/w/NTV1D.npz $CR/on1_JTDC/tab.npz $CR/on1_PAIDC/dagger.npz; do
    [[ -s $f ]] || die "no $f"; done
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 200 )) || die "free $free GB < 200 GB"
  git rev-parse --short HEAD > $O/code.txt
  mkdir -p $O/lists $O/tmp_lists; cp $O2/lists/c*.txt $O/lists/; $V $DG lists --root $O/tmp_lists > /dev/null || die lists
  diff -r $O/lists $O/tmp_lists/lists > /dev/null || die "chunk lists differ from dag.py lists"; diff -r $O/lists $O2/lists > /dev/null || die "lists copy"
  rm -rf $O/tmp_lists; echo "$(date '+%F %T') lists: $(ls $O/lists | wc -l) chunk files = step 2a's = dag.py lists" | tee -a $O/stages.txt
  mkdir -p $O/roll/chunks; cp $O/lists/c000.txt $O/roll/chunks/
  t1=$(sub dg4-t1 --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- "${EA[@]}" $AP $DG t1 --cfg-glob "$TCFG" --workers 32) || exit 1
  t2=$(sub dg4-t2 --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- "${EA[@]}" $AP $DG t2 --workers 16) || exit 1
  t12=$(sub dg4-t12 --after $t1,$t2 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- $V $PY t12) || exit 1; echo $t12 > $O/t12.id
  read -r id c < <(roll) || exit 1; [[ $c == c000 ]] || die "first roll chunk $c"
  read -r x y z < <(chunk_tail $id c000) || exit 1
  k=$(sub dg4-chk1 --after $y,$z --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $DS chk1) || exit 1
  r=$(sub dg4-rest --after $k --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt)): T1 $t1 T2 $t2 t12 $t12 c000 roll $id rows $x teach $y clean $z chk1 $k rest $r" | tee -a $O/stages.txt ;;
rest)
  cp $O/lists/c[0-9][0-9][0-9].txt $O/roll/chunks/; out=$(roll) || exit 1; w=(); ro=()
  while read -r id c; do [[ -n $id ]] || continue; ro+=($id); w+=($(chunk_tail $id $c)) || exit 1; done <<< "$out"
  h=$(git rev-parse --short HEAD); echo "$h rest" >> $O/code.txt; t12=$(cat $O/t12.id)
  l=$(sub dg4-link --after $(csv ${w[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $DS link) || exit 1
  ca=$(sub dg4-cache --after $l --vram 20 --cpu 8 --ram 64 --timeout-h 8 -- bash $DS cache) || exit 1
  po=$(sub dg4-pool --after $ca --vram 0.1 --cpu 8 --ram 96 --timeout-h 3 -- bash $DS pool) || exit 1
  lb=$(sub dg4-label --after $po --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $DS label) || exit 1
  fw=$(sub dg4-fwd --after $lb --vram 16 --cpu 8 --ram 120 --timeout-h 3 -- bash $DS fwd) || exit 1
  gi=$(sub dg4-r1in --after $fw --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $DS r1in) || exit 1
  m=$(sub dg4-merge --after $gi --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME merge) || exit 1
  wt=$(sub dg4-weights --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY weights) || exit 1
  wc=$(sub dg4-wcheck --after $wt --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY wcheck --out $O/results/gw_draw_files.json) || exit 1
  df_=$(sub dg4-diff --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- $V $PY diff --out $O/results/diff_D3.json) || exit 1
  ga=$(sub dg4-t3a-D1 "${FE[@]}" --timeout-h 1 -- bash $ME ident t3a_D1_preT NT4ID1-F-s0 D1) || exit 1
  pa=$(sub dg4-t3run-a "${FE[@]}" --timeout-h 1 -- bash $ME t3run a) || exit 1
  pb=$(sub dg4-t3run-b --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3run b) || exit 1
  tc=$(sub dg4-t3cmp --after $pa,$pb --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME t3cmp) || exit 1
  gb=$(sub dg4-g0b-D3 --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME ident g0b_D3_preT NT4ID3-F-s0 D3) || exit 1
  sm=$(sub dg4-smoke --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  tx=$(sub dg4-tax --after $(csv ${ro[@]}) --vram 0.1 --cpu 16 --ram 48 --timeout-h 4 -- bash $ME tax) || exit 1
  GA=$(csv $t12 $wc $df_ $ga $tc $gb $sm)
  bd=$(sub dg4-build --after $GA,$tx --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- $V $PY build) || exit 1
  tr=$(sub dg4-train-$D3 --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train 0) || exit 1
  ra=$(sub dg4-ratio-D3 --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $D3 --on-start $TON --train-id $tr \
       --out $O/results/ratio_D3.json) || exit 1
  pt=$(sub dg4-post-D3 --after $tr,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post 0) || exit 1
  y=($(read281 $D3 $pt)); [[ ${#y[@]} == 9 ]] || die "read281 $D3"
  rp=$(sub dg4-report --after ${y[-1]} --vram 0.1 --cpu 16 --ram 48 --timeout-h 3 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') rest (code $h): rolls ${ro[*]} chunk tails ${w[*]} link $l cache $ca pool $po label $lb fwd $fw r1in $gi merge $m weights $wt" \
       "wcheck $wc diff $df_ t3a $ga t3b $pa $pb $tc g0b $gb smoke $sm tax $tx build $bd train $tr ratio $ra post $pt 281 ${y[*]} report $rp" | tee -a $O/stages.txt ;;
merge)
  [[ ! -e $CR/on1_$NM ]] || die "on1_$NM exists: never overwritten"
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 100 )) || die "free $free GB < 100 GB"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_$NP:pd2:both --out $NM --nan-keys fut_reg || die merge
  $TR $PY mfix || die "mfix: the merged prefix differs from on1_JTDC" ;;
ident)  # ident NAME TAG ARM: the argv to step 125, every scalar <= 100 vs the stored NT3D1-F-s0 bit for bit, no extra / missing key
  nm=$2 tag=$3 arm=$4; mapfile -t A < <($V $PY argv $arm $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PC ident --tag $tag --ref $T --upto 100 --stop 125 --out $O/results/$nm.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$tag; exit $r ;;
t3run)  # T3 (b), weight-0 pair: D1's argv, 60 steps, T = 30; a on on1_JTDC / NTV1D, b on on1_JTDC2 / NTV1D2Z
  [[ $2 == a ]] && { X=NT4T3A-F-s0; v=D1; } || { X=NT4T3B-F-s0; v=D1Z; }
  mapfile -t A < <($V $PY argv $v $X --steps 60 --on-start 30 --save-at none); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') t3run $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/t3_$2.log 2>&1 || die "t3run $2 (log $O/t3_$2.log)" ;;
t3cmp) $V $GP pair --a NT4T3A-F-s0 --b NT4T3B-F-s0 --upto 60 --out $O/results/t3b_w0.json; r=$?; rm -rf $RUNS/NT4T3A-F-s0 $RUNS/NT4T3B-F-s0; exit $r ;;
smoke)  # D3's argv, 60 steps, T = 30, --on-mix-log: composition (dgc.py smoke --var b, NTV1D2's shares) + lr (p1l.py lrcheck)
  tag=NT4SM-F-s0; mapfile -t A < <($V $PY argv D3 $tag --steps 60 --on-start 30 --save-at none --mix); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" > $O/smoke.log 2>&1 || die "smoke $tag trainer (log $O/smoke.log)"
  $V $GP smoke --var b --tag $tag --on-start 30 --want $O/results/w_NTV1D2.json --out $O/results/smoke_D3.json; r1=$?
  $V $PL lrcheck --tag $tag --on-start 30 --out $O/results/smoke_lr_D3.json; r2=$?
  rm -rf $RUNS/$tag; exit $(( r1 || r2 )) ;;
tax)  # 719 train-scene scores of D1's roll (rf_tax + pai_fix facts = dgc.sh tax); P1L's step-2a roll is stored ($O2/tax/roll.csv)
  $AP $RF/rf_tax.py extract --runs $(ls -d $O/roll/runs/*/sim) --out $O/tax/roll.pkl --workers 16 > $O/tax/roll.log 2>&1 || die "extract"
  $V experiments/pai_fix/scripts/pai_fix.py facts --pkl $O/tax/roll.pkl --runs $O/roll --out $O/tax/roll.csv >> $O/tax/roll.log 2>&1 || die "facts" ;;
train)  # train SEED
  s=$2; tag=$([[ $s == 0 ]] && echo $D3 || echo $D3S1); mapfile -t A < <($V $PY argv D3 $tag $([[ $s != 0 ]] && echo --seed $s)); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $tag (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
post)  # post SEED: DONE, dev ADE <= 0.566, lr over the whole run
  s=$2; tag=$([[ $s == 0 ]] && echo $D3 || echo $D3S1); sf=$([[ $s != 0 ]] && echo _s$s)
  $V $PC post --tag $tag --out $O/results/post_D3$sf.json || die "post $tag"
  $V $PL lrcheck --tag $tag --on-start $TON --full --out $O/results/lr_full_D3$sf.json || die "lr_full $tag" ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read D3=$F_/$D3-281ag.csv D1s0=$F_/$T-281ag.csv D1s1=$F_/NT3D1-F-s1-281ag.csv P1L=$F_/PCL1L-F-s0-281ag.csv \
      NTV0s0=$F_/NTV0-L18-s0-281ag.csv R8Ks2=$F_/R8K-L18-s2-281ag.csv INTP75=$F_/INTP75-NR-281ag.csv \
      --pairs D3:D1s0 D3:D1s1 D3:P1L D3:NTV0s0 D3:R8Ks2 D3:INTP75 --out $O/results/pai281.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF (old scorer): NTV1 step 4 D3 vs D1-s0 / s1, P1L, NTV0-s0, R8K-s2, INTP75-NR" || die r281
  rm -rf $O/sv2
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $D3-ag --workers 16 > $O/sv2.log 2>&1 \
    || die "scorer_v2 new (log $O/sv2.log)"
  $V $PY bs --out $O/bs --tags $D3 > $O/bs.log 2>&1 \
    || echo "$(date '+%F %T') behavior_stats failed (log $O/bs.log); verdict without D3's highway row" | tee -a $O/stages.txt
  $V $PY verdict --dir $O --out $O/results/verdict > $O/verdict.log 2>&1 || die "verdict (log $O/verdict.log)"
  bash $ME cand ;;
cand)  # candidate rule (plan section 6): $O/CAND from the verdict; none -> DONE; else D3 seed 1 + val160 of both seeds -> vreport -> DONE
  c=$(cat $O/CAND); if [[ $c == none ]]; then echo "$(date '+%F %T') no candidate (D3 new mean < 0.875): done" | tee -a $O/stages.txt; date '+%F %T' > $O/DONE; exit 0; fi
  [[ $c == D3 ]] || die "CAND $c"; [[ ! -e $RUNS/$D3S1 && -z $(ls -d $DATA_DIR/runs/route_mem/train-$D3S1 2>/dev/null) ]] || die "$D3S1 exists"
  h=$(git rev-parse --short HEAD); echo "$h cand" >> $O/code.txt
  t=$(sub dg4-train-$D3S1 "${FE[@]}" --timeout-h 5 -- bash $ME train 1) || exit 1
  r=$(sub dg4-ratio-D3-s1 --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $D3S1 --on-start $TON --train-id $t --out $O/results/ratio_D3_s1.json) || exit 1
  p=$(sub dg4-post-D3-s1 --after $t,$r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post 1) || exit 1
  a=$(v160 $D3) || exit 1; b=$(v160 $D3S1 $p) || exit 1
  f=$(sub dg4-vreport --after $a,$b --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME vreport) || exit 1
  echo "$(date '+%F %T') cand (code $h): train $D3S1 $t ratio $r post $p; val160 tax $D3 $a / $D3S1 $b; vreport $f" | tee -a $O/stages.txt ;;
vreport)
  rm -rf $O/v160/sv2
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/v160/sv2 --reads $D3-v160,$D3S1-v160 --workers 16 \
      > $O/v160/sv2.log 2>&1 || die "scorer_v2 new val160 (log $O/v160/sv2.log)"
  $V $PY vverdict --dir $O --out $O/results/cand_v160 > $O/vverdict.log 2>&1 || die "vverdict (log $O/vverdict.log)"; date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/VERDICT || -f $O/DONE ]] || ! $V - $O/jobs.txt $O/v160/jobs.txt <<'PY'
import os, sys
from jevdrive.cl import pool as P
last = {}
for f in sys.argv[1:]:
    for l in (open(f) if os.path.exists(f) else []):
        if len(l.split()) > 1:
            last[f + ":" + l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  cat $O/VERDICT 2>/dev/null; [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -4 $O/stages.txt; true ;;
*) sed -n 2,15p "$0"; exit 2 ;;
esac
