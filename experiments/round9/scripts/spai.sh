#!/usr/bin/env bash
# ROUND9 NTV1 S-PAI (plans/2026-10-24-ntv1-spai.md = registration): S rows on the student's own PAI states, 1 seed.
# Student D3 = NT4D3-F-s0 rolled under step 2a's submission serve (ACC + GAP, g10, ALPASIM_FAST=1, CONC 4, frame dumps, KEEP_ASL) on the same 719
# curated train scenes and chunk lists; round 6 dag.sh's roll -> rows -> teach -> clean -> cache -> pool -> label -> fwd / r1in bodies, unchanged,
# under this lane's env -> pool on1_PAIDC3 (DAgger iteration 3: kept, not used). Per chunk after its rows: spai.py raster (2 Hz candidate rows + the
# pre-offroad decisions, stretched road_pai canvas). Then spai.py index (G1-G3) -> merge on1_JTDC2S = on1_JTDC2's parts + on1_PAIDC3 as `ps:self` ->
# spai.py weights (NTV1D2S 16 rb0 + 16 ps + 8 DAgger; NTV1D2SZ weight 0; G4) -> wcheck, diff, code identity (old worktree vs HEAD), weight-0 pair,
# pre-T identity, smoke -> train NT5SP-F-s0 -> ratio / post -> PAI 281 + val160 -> report -> $O/DONE.
# Box, repo root; every job through the pool; ids in $O/jobs.txt (+ $O/v160/jobs.txt), stages in $O/stages.txt.
#   spai.sh chain <old rev>   preflight + lists + code-identity pair (old rev worktree vs HEAD) + c000 roll -> rows -> teach / clean / raster -> chk1 -> rest
#   spai.sh wait              one run_in_background waiter: $O/DONE, $O/VERDICT or a failed / cancelled job of this lane
#   job bodies: rest | raster <c> | merge | ident | crun <old|new> | w0run | smoke | tax | train | post | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv1_spai; O4=$DATA_DIR/runs/round9/ntv1_step4; O2=$DATA_DIR/runs/round9/ntv1_step2; ME=experiments/round9/scripts/spai.sh
DS=experiments/round6/scripts/dag.sh; DG=experiments/round6/scripts/dag.py; PY=experiments/round9/scripts/spai.py; PC=experiments/round9/scripts/pcl.py
VS=experiments/round9/scripts/v160.sh; PL=experiments/round9/scripts/p1l.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; RF=experiments/route_follow/scripts
CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src; RES=experiments/round9/results/ntv1_spai
T=NT4D3-F-s0; L=${T,,}; NP=PAIDC3; NM=JTDC2S; TV=b3; SP=NT5SP-F-s0; TON=10000; PRIO=${PRIO:-14}
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = dgc.sh / dg4.sh / D3's read
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both on1_PAIDC:pd:both on1_PAIDC2:pd2:both"  # = on1_JTDC2
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
export DAG_O=$O DAG_T=$T DAG_NP=$NP DAG_RES=$RES DAG_TV=$TV
DENV=(--env DAG_O=$O --env DAG_T=$T --env DAG_NP=$NP --env DAG_RES=$RES --env DAG_TV=$TV)
FE=(--train --vram 40 --cpu 6 --ram 48 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                    # eager, as D3
GT="NT5CO-F-s0 NT5CN-F-s0 NT5W0-F-s0 NT5IDP-F-s0 NT5SM-F-s0"
mkdir -p $O/pool $O/checks $O/w $O/tax $O/results $O/pai281 $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round9-spai --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "${DENV[@]}" "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() {  # -> "<id> <chunk>" lines of the chunks submitted now (= dg4.sh roll)
  local f=$O/roll/submit.$(date +%s%N).txt
  SERVE="$SERVE PAI_DUMP=100000" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO KEEP_ASL=1 bash experiments/alpasim/scripts/pai_box.sh submit $O/roll $T > $f \
    || die "roll submit"
  grep -q 'not submitted' $f && die "roll: $(grep 'not submitted' $f)"
  awk '$2 ~ /^pai-/ {print $1, "roll-" substr($2, length($2) - 3)}' $f >> $O/jobs.txt
  awk '$2 ~ /^pai-/ {print $1, substr($2, length($2) - 3)}' $f; }
chunk_tail() {  # chunk_tail <roll id> <chunk> -> "<rows id> <teach id> <clean id> <raster id>"
  local x y z r
  x=$(sub sp-rows-$2 --after $1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $DS rows $2) || exit 1
  y=$(sub sp-teach-$2 --after $x --vram 0.1 --cpu 8 --ram 32 --timeout-h 2 -- bash $DS teach $2) || exit 1
  z=$(sub sp-clean-$2 --after $x --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $DS clean $2) || exit 1
  r=$(sub sp-raster-$2 --after $x --vram 0.1 --cpu 8 --ram 32 --timeout-h 2 -- bash $ME raster $2) || exit 1
  echo "$x $y $z $r"; }
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
  rev=${2:?old rev (the commit before the drivable_hinge geom change)}
  [[ ! -e $O/roll && ! -e $O/DONE && ! -s $O/jobs.txt ]] || die "$O has a chain: never overwritten"
  [[ -f $RUNS/$T/ckpt-final.pt ]] || die "no $T"
  for X in $SP $GT; do [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  for p in on1_$NP on1_$NM; do [[ ! -e $CR/$p ]] || die "$p exists"; done
  [[ -z $(ls -d $DATA_DIR/tmp/dag_stage/${L}_* 2>/dev/null) ]] || die "staging dirs ${L}_* exist"
  for f in $F_/$T-281ag.csv $F_/NT4D3-F-s1-281ag.csv $F_/$T-v160ag.csv $F_/NT4D3-F-s1-v160ag.csv $O4/sv2/new/rollouts.csv $O4/sv2s1/new/rollouts.csv \
           $O4/v160/sv2/new/rollouts.csv $O4/tax/roll.csv $O4/w/NTV1D2.npz $CR/on1_JTDC2/tab.npz $CR/on1_PAIDC2/dagger.npz \
           $DATA_DIR/runs/round9/pool_design/stations_train.parquet; do [[ -s $f ]] || die "no $f"; done
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 250 )) || die "free $free GB < 250 GB"
  git rev-parse --short HEAD > $O/code.txt
  WT=$DATA_DIR/tmp/spai_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || die "worktree $rev"
  mkdir -p $O/lists $O/tmp_lists; cp $O2/lists/c*.txt $O/lists/; $V $DG lists --root $O/tmp_lists > /dev/null || die lists
  diff -r $O/lists $O/tmp_lists/lists > /dev/null || die "chunk lists differ from dag.py lists"; diff -r $O/lists $O2/lists > /dev/null || die "lists copy"
  rm -rf $O/tmp_lists; echo "$(date '+%F %T') lists: $(ls $O/lists | wc -l) chunk files = step 2a's = dag.py lists" | tee -a $O/stages.txt
  co=$(sub sp-crun-old "${FE[@]}" --timeout-h 1 -- bash $ME crun old $rev) || exit 1
  cn=$(sub sp-crun-new "${FE[@]}" --timeout-h 1 -- bash $ME crun new) || exit 1
  cc=$(sub sp-gcode --after $co,$cn --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY pair --a NT5CO-F-s0 --b NT5CN-F-s0 --upto 60 --strict \
       --out $O/results/gcode.json) || exit 1
  echo $cn > $O/cn.id; echo $cc > $O/gcode.id
  mkdir -p $O/roll/chunks; cp $O/lists/c000.txt $O/roll/chunks/
  read -r id c < <(roll) || exit 1; [[ $c == c000 ]] || die "first roll chunk $c"
  read -r x y z r < <(chunk_tail $id c000) || exit 1; echo $r > $O/raster_c000.id
  k=$(sub sp-chk1 --after $y,$z --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $DS chk1) || exit 1
  q=$(sub sp-rest --after $k --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt), old $rev): code pair $co $cn $cc c000 roll $id rows $x teach $y clean $z raster $r chk1 $k rest $q" \
    | tee -a $O/stages.txt ;;
rest)
  cp $O/lists/c[0-9][0-9][0-9].txt $O/roll/chunks/; out=$(roll) || exit 1; w=(); ro=(); ras=($(cat $O/raster_c000.id))
  while read -r id c; do [[ -n $id ]] || continue; ro+=($id); read -r x y z r < <(chunk_tail $id $c) || exit 1; w+=($x $y $z); ras+=($r); done <<< "$out"
  h=$(git rev-parse --short HEAD); echo "$h rest" >> $O/code.txt; cn=$(cat $O/cn.id); cc=$(cat $O/gcode.id)
  l=$(sub sp-link --after $(csv ${w[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $DS link) || exit 1
  ca=$(sub sp-cache --after $l --vram 20 --cpu 8 --ram 64 --timeout-h 8 -- bash $DS cache) || exit 1
  po=$(sub sp-pool --after $ca --vram 0.1 --cpu 8 --ram 96 --timeout-h 3 -- bash $DS pool) || exit 1
  ix=$(sub sp-index --after $po,$(csv ${ras[@]}) --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- $TR $PY index) || exit 1
  lb=$(sub sp-label --after $po --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $DS label) || exit 1
  fw=$(sub sp-fwd --after $lb --vram 16 --cpu 8 --ram 120 --timeout-h 3 -- bash $DS fwd) || exit 1
  gi=$(sub sp-r1in --after $fw --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $DS r1in) || exit 1
  m=$(sub sp-merge --after $lb,$ix --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME merge) || exit 1
  wt=$(sub sp-weights --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY weights) || exit 1
  wc=$(sub sp-wcheck --after $wt --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY wcheck --out $O/results/gw_draw_files.json) || exit 1
  df_=$(sub sp-diff --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- $V $PY diff --out $O/results/diff_SP.json) || exit 1
  wb=$(sub sp-w0run --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME w0run) || exit 1
  wp=$(sub sp-gw0 --after $wb,$cn --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY pair --a NT5CN-F-s0 --b NT5W0-F-s0 --upto 60 --out $O/results/gw0.json) || exit 1
  gp=$(sub sp-gpre --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME ident gpre_SP NT5IDP-F-s0 SP) || exit 1
  sm=$(sub sp-smoke --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  tx=$(sub sp-tax --after $(csv ${ro[@]}) --vram 0.1 --cpu 16 --ram 48 --timeout-h 4 -- bash $ME tax) || exit 1
  GA=$(csv $cc $wc $df_ $wp $gp $sm $gi)
  bd=$(sub sp-build --after $GA,$tx --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- $V $PY build) || exit 1
  tr=$(sub sp-train-$SP --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train) || exit 1
  ra=$(sub sp-ratio --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $SP --on-start $TON --train-id $tr \
       --out $O/results/ratio_SP.json) || exit 1
  pt=$(sub sp-post --after $tr,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post) || exit 1
  y=($(read281 $SP $pt)); [[ ${#y[@]} == 9 ]] || die "read281 $SP"
  v=$(V160_O=$O/v160 V160_TAGS=$SP V160_AFTER=$pt PRIO=$PRIO bash $VS reads | tail -1) && [[ -n $v ]] || die "v160 $SP"
  rp=$(sub sp-report --after ${y[-1]},$v --vram 0.1 --cpu 16 --ram 48 --timeout-h 3 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') rest (code $h): rolls ${ro[*]} tails ${w[*]} rasters ${ras[*]} link $l cache $ca pool $po index $ix label $lb fwd $fw r1in $gi" \
       "merge $m weights $wt wcheck $wc diff $df_ w0 $wb $wp pre-T $gp smoke $sm tax $tx build $bd train $tr ratio $ra post $pt 281 ${y[*]} v160 $v" \
       "report $rp" | tee -a $O/stages.txt ;;
resume)  # amendment 1 (plan): the drivable_hinge geom fix -> gcode / gw0 / pre-T / smoke again on this code, then the tail (attempt 1's gw0 failed)
  rev=${2:?old rev}; h=$(git rev-parse --short HEAD); echo "$h resume" >> $O/code.txt
  for X in NT5CO-F-s0 NT5CN-F-s0 NT5W0-F-s0 NT5IDP-F-s0 NT5SM-F-s0 NT5BA-F-s0 NT5BB-F-s0; do rm -rf $RUNS/$X $DATA_DIR/runs/route_mem/train-$X; done
  for d in $O/pai281/$SP-ag $O/v160/reads/$SP-ag; do [[ ! -e $d/runs ]] || die "$d has runs"; rm -rf $d $d.submit.txt; done
  mkdir -p $O/results/attempt1; mv $O/results/{gcode,gw0,gpre_SP,smoke_SP,smoke_SP.json.dgc,smoke_lr_SP}.json $O/results/attempt1/ 2>/dev/null
  co=$(sub sp-crun-old "${FE[@]}" --timeout-h 1 -- bash $ME crun old $rev) || exit 1
  cn=$(sub sp-crun-new "${FE[@]}" --timeout-h 1 -- bash $ME crun new) || exit 1
  cc=$(sub sp-gcode --after $co,$cn --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY pair --a NT5CO-F-s0 --b NT5CN-F-s0 --upto 60 --strict \
       --out $O/results/gcode.json) || exit 1
  wb=$(sub sp-w0run "${FE[@]}" --timeout-h 1 -- bash $ME w0run) || exit 1
  wp=$(sub sp-gw0 --after $wb,$cn --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY pair --a NT5CN-F-s0 --b NT5W0-F-s0 --upto 60 --out $O/results/gw0.json) || exit 1
  gp=$(sub sp-gpre "${FE[@]}" --timeout-h 1 -- bash $ME ident gpre_SP NT5IDP-F-s0 SP) || exit 1
  sm=$(sub sp-smoke "${FE[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  GA=$(csv $cc $wp $gp $sm)
  bd=$(sub sp-build --after $GA --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- $V $PY build) || exit 1
  tr=$(sub sp-train-$SP --after $GA "${FE[@]}" --timeout-h 5 -- bash $ME train) || exit 1
  ra=$(sub sp-ratio --after $GA --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $SP --on-start $TON --train-id $tr \
       --out $O/results/ratio_SP.json) || exit 1
  pt=$(sub sp-post --after $tr,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post) || exit 1
  y=($(read281 $SP $pt)); [[ ${#y[@]} == 9 ]] || die "read281 $SP"
  v=$(V160_O=$O/v160 V160_TAGS=$SP V160_AFTER=$pt PRIO=$PRIO bash $VS reads | tail -1) && [[ -n $v ]] || die "v160 $SP"
  rp=$(sub sp-report --after ${y[-1]},$v --vram 0.1 --cpu 16 --ram 48 --timeout-h 3 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') resume (code $h): code pair $co $cn $cc w0 $wb $wp pre-T $gp smoke $sm build $bd train $tr ratio $ra post $pt 281 ${y[*]} v160 $v report $rp" \
    | tee -a $O/stages.txt ;;
raster) "${EA[@]}" $AP $PY raster --chunk $2 --rows $O/rows/$2 --cfg-glob "$O/roll/runs/${L}_$2/sim/generated-user-config-0.yaml" --workers 8 ;;
merge)
  [[ ! -e $CR/on1_$NM ]] || die "on1_$NM exists: never overwritten"
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 100 )) || die "free $free GB < 100 GB"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_$NP:ps:self --out $NM --nan-keys fut_reg || die merge ;;
crun)  # crun old <rev> | new: D3's argv, 60 steps, T = 30 (lr2-at follows), no checkpoints; old = the pre-change commit's worktree
  [[ $2 == old ]] && X=NT5CO-F-s0 || X=NT5CN-F-s0
  mapfile -t A < <($V $PY argv D3 $X --steps 60 --on-start 30 --save-at none); [[ ${#A[@]} -gt 5 ]] || die "argv"
  [[ $2 == old ]] && { cd $DATA_DIR/tmp/spai_old_$3 || die "no worktree $3"; }
  echo "$(date '+%F %T') crun $2 in $PWD ($(git rev-parse --short HEAD)): ${A[*]}" >> $O/stages.txt
  $TR "${A[@]}" > $O/crun_$2.log 2>&1 || die "crun $2 (log $O/crun_$2.log)" ;;
w0run)  # the weight-0 side: D3's argv on on1_JTDC2S / NTV1D2SZ + the raster, 60 steps, T = 30
  mapfile -t A < <($V $PY argv SPZ NT5W0-F-s0 --steps 60 --on-start 30 --save-at none); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') w0run: ${A[*]}" >> $O/stages.txt; $TR "${A[@]}" > $O/w0run.log 2>&1 || die "w0run (log $O/w0run.log)" ;;
ident)  # ident NAME TAG ARM: the argv to step 125, every scalar <= 100 vs the stored D3 bit for bit (pool_held / pool_fit slices reported apart)
  nm=$2 tag=$3 arm=$4; mapfile -t A < <($V $PY argv $arm $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $nm (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PY ident --tag $tag --ref $T --upto 100 --stop 125 --out $O/results/$nm.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$tag; exit $r ;;
smoke)  # the arm's argv, 60 steps, T = 30, --on-mix-log: composition, draw shares, raster rows taken, road hinge after T; lr (p1l.py lrcheck)
  tag=NT5SM-F-s0; mapfile -t A < <($V $PY argv SP $tag --steps 60 --on-start 30 --save-at none --mix); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $tag (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" > $O/smoke.log 2>&1 || die "smoke $tag trainer (log $O/smoke.log)"
  $V $PY smoke --tag $tag --on-start 30 --out $O/results/smoke_SP.json; r1=$?
  $V $PL lrcheck --tag $tag --on-start 30 --out $O/results/smoke_lr_SP.json; r2=$?
  rm -rf $RUNS/$tag; exit $(( r1 || r2 )) ;;
tax)  # 719 train-scene scores of D3's roll (= dg4.sh tax)
  $AP $RF/rf_tax.py extract --runs $(ls -d $O/roll/runs/*/sim) --out $O/tax/roll.pkl --workers 16 > $O/tax/roll.log 2>&1 || die "extract"
  $V experiments/pai_fix/scripts/pai_fix.py facts --pkl $O/tax/roll.pkl --runs $O/roll --out $O/tax/roll.csv >> $O/tax/roll.log 2>&1 || die "facts" ;;
train)
  mapfile -t A < <($V $PY argv SP $SP); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $SP (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
post)
  $V $PC post --tag $SP --out $O/results/post_SP.json || die "post $SP"
  $V $PL lrcheck --tag $SP --on-start $TON --full --out $O/results/lr_full_SP.json || die "lr_full $SP" ;;
report)
  rm -rf $O/sv2 $O/v160/sv2
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $SP-ag --workers 16 > $O/sv2.log 2>&1 \
    || die "scorer_v2 new (log $O/sv2.log)"
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/v160/sv2 --reads $SP-v160 --workers 16 > $O/v160/sv2.log 2>&1 \
    || die "scorer_v2 new val160 (log $O/v160/sv2.log)"
  $V $PY bs --out $O/bs --tags $SP > $O/bs.log 2>&1 \
    || echo "$(date '+%F %T') behavior_stats failed (log $O/bs.log); verdict without the arm's highway row" | tee -a $O/stages.txt
  $V $PY verdict --dir $O --out $O/results/verdict > $O/verdict.log 2>&1 || die "verdict (log $O/verdict.log)"
  date '+%F %T' > $O/DONE ;;
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
*) sed -n 2,13p "$0"; exit 2 ;;
esac
