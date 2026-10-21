#!/usr/bin/env bash
# ROUND6 line A, DAGGER-MAP (plans/2026-10-21-dagger-map.md): the student R5LC-L18-s0 (MV-LC) rolled under the submission serve (ACC + GAP, g10,
# ALPASIM_FAST=1, CONC 4) with frame dumps on the 719 non-held curated train scenes (round5 results/s3_scenes.tsv) -> onp_rows (rows + cold) ->
# the map teacher on every decision (dag.py teach) -> frame dumps / rollout.asl removed -> onp_cache -> rf_pai_pool --fut poses (pool on1_PAID) ->
# dag.py label (tab.fut := teacher label on rows fed >= 2 m/s; T4) -> input-path check (rf_recov fwd + dag.py r1in) -> zero-copy merge on1_JTPHD =
# on1_JTPH + on1_PAID -> draw files R6D0 (identity) / R6DAG (8 DAgger rows a batch) -> smoke + T3 identity -> trainings R6DAG-L18-s0 / s1 (MV-LC's argv
# + the DAgger rows, from NTV0-L18-s0, 9 500 steps) -> PAI 281 ACC + GAP -> r281 + verdict -> local PCS ruler -> DONE.
# Box, repo root; every job through the pool at PRIO 22; ids in $O/jobs.txt, stages in $O/stages.txt, a stop reason in $O/VERDICT.
#   dag.sh gates      T1 / T2 (CPU, AlpaSim venv) -> results/dagger/t1.*, t2.*
#   dag.sh chain      c000 (6 scenes) roll -> rows -> teach -> clean -> chk1 -> rest (submitter: every other chunk, then the build tail)
#   dag.sh wait [M]   blocks until $O/VERDICT, $O/<M> (default DONE) or a failed job of this lane (one run_in_background waiter)
#   dag.sh dry        the training / identity commands and their diff against R5LC-L18-s0's recorded argv
#   job bodies: chk1 | rest | rows <c> | teach <c> | clean <c> | link | cache | pool | label | fwd | r1in | merge | weights | smoke | identrun <v> |
#               ident | build | launch | train <seed> | r281 | pcsrep
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=${DAG_O:-$DATA_DIR/runs/round6/dag}; ME=experiments/round6/scripts/dag.sh; PY=experiments/round6/scripts/dag.py; V=.venv/bin/python
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; RF=experiments/route_follow/scripts
CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; RES=${DAG_RES:-experiments/round6/results/dagger}; export DAG_RES=$RES DAG_O=$O
T=${DAG_T:-R5LC-L18-s0}; L=${T,,}; WRONG=RB0-F-s1; PRIO=${PRIO:-22}; NP=${DAG_NP:-PAID}; export DAG_T=$T DAG_NP=$NP; NM=JTPHD; N=9500; KDAG=8; R2PY=experiments/round2/scripts/r2.py
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both on1_PAIH:ph:both on1_PAIH05:ph05:both on1_PAIH15:ph15:both"
TCFG="$DATA_DIR/runs/round5/s3/roll/A/runs/*/sim/generated-user-config-0.yaml,$DATA_DIR/runs/round5/d3/roll/A/runs/*/sim/generated-user-config-0.yaml"
RCFG="$O/roll/runs/*/sim/generated-user-config-0.yaml"
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $O/checks $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round6-dag --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name --env DAG_O=$O --env DAG_T=$T --env DAG_NP=$NP --env DAG_RES=$RES ${DAG_IT2:+--env DAG_IT2=1} "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() {  # roll [after] -> "<id> <chunk>" lines of the chunks submitted now (pai_box.sh skips finished / pending chunks)
  local f=$O/roll/submit.$(date +%s%N).txt
  SERVE="$SERVE PAI_DUMP=100000" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO AFTER=${1:-} KEEP_ASL=1 bash experiments/alpasim/scripts/pai_box.sh submit $O/roll $T > $f \
    || die "roll submit"
  grep -q 'not submitted' $f && die "roll: $(grep 'not submitted' $f)"
  awk '$2 ~ /^pai-/ {print $1, "roll-" substr($2, length($2) - 3)}' $f >> $O/jobs.txt
  awk '$2 ~ /^pai-/ {print $1, substr($2, length($2) - 3)}' $f; }
chunk_tail() {  # chunk_tail <roll id> <chunk> -> "<rows id> <teach id> <clean id>"
  local x y z
  x=$(sub dag-rows-$2 --after $1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $2) || exit 1
  y=$(sub dag-teach-$2 --after $x --vram 0.1 --cpu 8 --ram 32 --timeout-h 2 -- bash $ME teach $2) || exit 1
  z=$(sub dag-clean-$2 --after $x --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $2) || exit 1
  echo "$x $y $z"; }
argv() {  # argv <variant> <steps> <tag> [seed]: one argument a line. lc = R5LC-L18-s0's recorded argv (no --save-at); d0 = lc on on1_JTPHD / R6D0
          # (DAgger rows at weight 0: T3 identity); dag = lc on on1_JTPHD / R6DAG with 8 DAgger rows a batch (--on-k 72 --batch 136 --yr-mass 0.095:
          # 51 navtrain + 13 yaw-rate + 64 pool rows unchanged)
  local M; M=$(ls -d $DATA_DIR/runs/route_mem/train-$T/*/ | tail -1)meta.json; [[ -f $M ]] || die "no $M"
  python3 - "$M" "$@" "$DATA_DIR/runs/round6/dag/w" <<'PY'
import json, os, sys
m, var, steps, tag = sys.argv[1:5]
nm, wf = os.environ.get("DAG_NM", "JTPHD"), dict(d0=os.environ.get("DAG_W0", "R6D0"), dag=os.environ.get("DAG_WF", "R6DAG"))   # amendment 4: DAG2's pool / draws
seed, w6 = (sys.argv[5:-1] or [None])[0], sys.argv[-1]
a = json.load(open(m))["argv"]
last = lambda k: len(a) - 1 - a[::-1].index(k)  # noqa: E731
assert a[a.index("--init") + 1] == "NTV0-L18-s0" and a[a.index("--tag") + 1] == "R5LC-L18-s0" and a[last("--on-wfile") + 1].endswith("/round5/r5/w/R5LC.npz")
assert a[last("--on") + 1] == "on1_JTPH" and a[last("--on-k") + 1] == "64" and a[last("--seed") + 1] == "0" and "--batch" not in a and a[last("--yr-mass") + 1] == "0.1"
a[a.index("--tag") + 1] = tag
a[last("--steps") + 1] = steps
i = a.index("--save-at"); del a[i:i + 2]
if var in ("d0", "dag"):
    a[last("--on") + 1] = f"on1_{nm}"
    a[last("--on-wfile") + 1] = f"{w6}/{wf[var]}.npz"
if var == "dag":
    a[last("--on-k") + 1] = "72"
    a[last("--yr-mass") + 1] = "0.095"
    a += ["--batch", "136"]
if seed is not None:
    a[last("--seed") + 1] = seed
print("\n".join(a))
PY
}
run() { local X; X=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$X"; }
read281() {  # read281 <tag> [after] -> "<chunk ids...> <facts id>" (round 5's read281: NTV0's chunk lists, ACC + GAP, g10, FAST=1, CONC 4)
  local R=$O/pai281/$1-ag ids t; mkdir -p $O/pai281
  AFTER=${2:-} CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO \
    bash experiments/alpasim/scripts/pai_box.sh submit $R $1 > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
gates)
  h=$(git rev-parse --short HEAD); echo "$h gates" >> $O/code.txt
  a=$(sub dag-t1 --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- "${EA[@]}" $AP $PY t1 --cfg-glob "$TCFG" --workers 32) || exit 1
  b=$(sub dag-t2 --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- "${EA[@]}" $AP $PY t2 --workers 16) || exit 1
  echo "$(date '+%F %T') gates (code $h): T1 $a T2 $b" | tee -a $O/stages.txt ;;
chain)
  [[ ! -e $O/roll && ! -e $O/DONE ]] || die "$O has a chain: never overwritten"
  [[ -f $RUNS/$T/ckpt-final.pt ]] || die "no $T"
  git rev-parse --short HEAD > $O/code.txt; $V $PY lists --root $O || die lists
  mkdir -p $O/roll/chunks; cp $O/lists/c000.txt $O/roll/chunks/
  read -r id c < <(roll) || exit 1; [[ $c == c000 ]] || die "first roll chunk $c"
  read -r x y z < <(chunk_tail $id c000) || exit 1
  k=$(sub dag-chk1 --after $y,$z --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME chk1) || exit 1
  r=$(sub dag-rest --after $k --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt)): c000 roll $id rows $x teach $y clean $z chk1 $k rest $r" | tee -a $O/stages.txt ;;
chk1) $V $PY chk1 --root $O ;;
rest)
  cp $O/lists/c[0-9][0-9][0-9].txt $O/roll/chunks/; out=$(roll) || exit 1; w=(); ro=()
  while read -r id c; do [[ -n $id ]] || continue; ro+=($id); w+=($(chunk_tail $id $c)) || exit 1; done <<< "$out"
  h=$(git rev-parse --short HEAD); echo "$h rest" >> $O/code.txt
  l=$(sub dag-link --after $(csv ${w[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME link) || exit 1
  ca=$(sub dag-cache --after $l --vram 20 --cpu 8 --ram 64 --timeout-h 8 -- bash $ME cache) || exit 1
  po=$(sub dag-pool --after $ca --vram 0.1 --cpu 8 --ram 96 --timeout-h 3 -- bash $ME pool) || exit 1
  lb=$(sub dag-label --after $po --vram 0.1 --cpu 4 --ram 64 --timeout-h 1 -- bash $ME label) || exit 1
  fw=$(sub dag-fwd --after $lb --vram 16 --cpu 8 --ram 120 --timeout-h 3 -- bash $ME fwd) || exit 1
  gi=$(sub dag-r1in --after $fw --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME r1in) || exit 1
  if [[ -n ${DAG_IT2:-} ]]; then                                   # iteration 2 (amendment 4): this student's pool only; dag2.sh joins
    d=$(sub dag-rowsdone --after $gi --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/ROWSDONE") || exit 1
    echo "$(date '+%F %T') rest (code $h, iteration 2): rolls ${ro[*]} chunk tails ${w[*]} link $l cache $ca pool $po label $lb fwd $fw r1in $gi done $d" | tee -a $O/stages.txt
    exit 0; fi
  m=$(sub dag-merge --after $gi --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME merge) || exit 1
  wt=$(sub dag-weights --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME weights) || exit 1
  sm=$(sub dag-smoke --after $wt "${FT[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  ia=$(sub dag-ident-lc --after $wt "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun lc) || exit 1
  ib=$(sub dag-ident-d0 --after $wt "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun d0) || exit 1
  ic=$(sub dag-ident --after $ia,$ib --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  bd=$(sub dag-build --after $sm,$ic --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME build) || exit 1
  echo "$(date '+%F %T') rest (code $h): rolls ${ro[*]} chunk tails ${w[*]} link $l cache $ca pool $po label $lb fwd $fw r1in $gi merge $m" \
       "weights $wt smoke $sm identity $ia $ib $ic build $bd" | tee -a $O/stages.txt ;;
rows)  # rows <chunk>: the chunk's rollouts -> $O/rows/<chunk>/<scene>.npz, linked into $O/rows/PAID/
  c=$2; G=$DATA_DIR/tmp/dag_stage/${L}_$c; R=$O/rows; mkdir -p $G $R/$NP          # stage per student and chunk (two chains run at once)
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AP experiments/onpol_pai/scripts/onp_rows.py rows --pass $c \
      --runs "$O/roll/runs/${L}_$c" --out $R --stage $G --tag $T --jobs 8 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $c --out $R --stage $G --jobs 8 || exit 1
  rm -rf $G
  bad=$($V - $R/$c $T <<'PY'
import glob, sys
import numpy as np
print(sum(str(np.load(f)["tag"]) != sys.argv[2] for f in glob.glob(sys.argv[1] + "/*.npz")))
PY
) && [[ $bad == 0 ]] || die "rows $c: $bad row files are not $T's (dumps kept)"
  for f in $R/$c/*.npz; do [[ -e $f ]] && ln -sfn $f $R/$NP/$(basename $f); done
  n=$(ls $R/$c/*.npz 2>/dev/null | wc -l); m=$(wc -w < $O/roll/chunks/$c.txt); echo "$(date '+%F %T') rows $c: $n / $m scene files" | tee -a $O/stages.txt
  (( n >= m * 9 / 10 )) || die "rows $c: $n of $m scenes" ;;
teach) "${EA[@]}" $AP $PY teach --rows $O/rows/$2 --cfg-glob "$RCFG" --out $O/teach --workers 8 ;;
clean)  # clean <chunk>: this lane's own rollout.asl and frame dumps of the chunk (its rows are built); drive.jsonl and scores stay
  r=$O/roll/runs/${L}_$2; n=$(find $r -name rollout.asl | wc -l); find $r -name rollout.asl -delete; rm -rf $r/sim/driver-logs/dump
  echo "$(date '+%F %T') clean $2: removed $n rollout.asl and the frame dumps; $(du -sh $r | cut -f1) left; free $(df -h $DATA_DIR | awk 'NR == 2 {print $4}')" | tee -a $O/stages.txt ;;
link) n=$(ls $O/rows/$NP/*.npz | wc -l); m=$(ls $O/teach/*.npz | wc -l); echo "$(date '+%F %T') link: $n scene files, $m teacher files" | tee -a $O/stages.txt
  (( m >= n )) || die "teacher files $m < row files $n" ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $NP --rows $O/rows/$NP --out $O/cache ;;
pool)
  [[ ! -e $CR/on1_$NP ]] || die "on1_$NP exists: never overwritten"
  $TR $RF/rf_pai_pool.py --name $NP --rows $O/rows/$NP --cache $O/cache/$NP --runs "$O/roll/runs/*" --split experiments/round5/results/s3_scenes.tsv \
      --fut poses --prefix ${NP,,} --src dagger ;;
reteach) "${EA[@]}" $AP $PY teach --rows $O/rows/$NP --cfg-glob "$RCFG" --out $O/teach --workers ${2:-16} --stale-only 1 ;;   # amendment 1: files of an older teacher
tail)  # amendment 3: resubmit the build tail from the label step (after the T4 stop); the stop record kept as VERDICT.t4
  [[ -f $CR/on1_$NP/tab.npz && ! -e $CR/on1_$NM ]] || die "no on1_$NP or on1_$NM exists"; [[ -f $O/VERDICT ]] && mv $O/VERDICT $O/VERDICT.t4.$(date +%H%M%S)
  h=$(git rev-parse --short HEAD); echo "$h tail" >> $O/code.txt
  lb=$(sub dag-label --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME label) || exit 1
  fw=$(sub dag-fwd --after $lb --vram 16 --cpu 8 --ram 120 --timeout-h 3 -- bash $ME fwd) || exit 1
  gi=$(sub dag-r1in --after $fw --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME r1in) || exit 1
  if [[ -n ${DAG_IT2:-} ]]; then                                   # iteration 2 (amendment 4): this student's pool only; dag2.sh joins
    d=$(sub dag-rowsdone --after $gi --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/ROWSDONE") || exit 1
    echo "$(date '+%F %T') rest (code $h, iteration 2): rolls ${ro[*]} chunk tails ${w[*]} link $l cache $ca pool $po label $lb fwd $fw r1in $gi done $d" | tee -a $O/stages.txt
    exit 0; fi
  m=$(sub dag-merge --after $gi --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME merge) || exit 1
  wt=$(sub dag-weights --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME weights) || exit 1
  sm=$(sub dag-smoke --after $wt "${FT[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  ia=$(sub dag-ident-lc --after $wt "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun lc) || exit 1
  ib=$(sub dag-ident-d0 --after $wt "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun d0) || exit 1
  ic=$(sub dag-ident --after $ia,$ib --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  bd=$(sub dag-build --after $sm,$ic --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME build) || exit 1
  echo "$(date '+%F %T') tail (code $h): label $lb fwd $fw r1in $gi merge $m weights $wt smoke $sm identity $ia $ib $ic build $bd" | tee -a $O/stages.txt ;;
t4pilot)  # T4 pilot (CPU): teacher variants on c001 + c002 (50 scenes) -> position-based label lateral acceleration
  P=$O/pilot; mkdir -p $P; cat $O/lists/c001.txt $O/lists/c002.txt > $P/scenes.txt; d=()
  for v in "A:" "B:DAG_SIG=3" "C:DAG_SIG=3 DAG_TM=3" "D:DAG_SIG=3 DAG_HCLIP=0" "E:DAG_SIG=3 DAG_HSIG=20 DAG_TM=3" "F:DAG_SIG=3 DAG_HSIG=20 DAG_TM=2.5"; do
    n=${v%%:*}; rm -rf $P/$n; mkdir -p $P/$n
    env ${v#*:} "${EA[@]}" $AP $PY teach --rows $O/rows/$NP --cfg-glob "$RCFG" --out $P/$n --chunk-list $P/scenes.txt --workers 16 || die "pilot $n"; d+=($P/$n); done
  $V $PY alatpos --dirs ${d[@]} --out $RES/t4_pilot.json ;;
label) bash $ME reteach 16 || die "reteach"; $V $PY label --root $O --pool on1_$NP ;;
fwd)  $TR $RF/rf_recov.py fwd --pool $NP --rows train-rec --tags $T,$WRONG --out $O/fwd ;;
r1in) $V $PY r1in --root $O --pool on1_$NP --fwd $O/fwd/$NP-train-rec --tag $T --control $WRONG ;;
merge)
  [[ ! -e $CR/on1_$NM ]] || die "on1_$NM exists: never overwritten"
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 200 )) || die "free $free GB < 200 GB"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_$NP:pd:both --out $NM --nan-keys fut_reg || die merge
  $TR $PY mfix --root $O --pool on1_$NM --ref on1_JTPH || die "mfix: the merged prefix differs from on1_JTPH" ;;
weights)
  $TR $PY weights --pool on1_$NM --base R5LC --k-on 64 --k-dag 0 --out R6D0 --src on1_$NP || die "R6D0"
  $TR $PY weights --pool on1_$NM --base R5LC --k-on 64 --k-dag $KDAG --out R6DAG --src on1_$NP || die "R6DAG" ;;
smoke)
  X=R6SM-L18-s0; [[ ! -e $RUNS/$X ]] || die "$X exists"; run dag 60 $X 0
  echo "$(date '+%F %T') smoke: ${A[*]} --diag-every 20 --save-at 20,40" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --save-at 20,40 > $O/smoke.log 2>&1 || die "smoke run (log $O/smoke.log)"
  $TR $PY smoke --tag $X --wfile R6DAG --k-dag $KDAG; r=$?
  rm -rf $RUNS/$X $RUNS/R6SMx20-L18-s0 $RUNS/R6SMx40-L18-s0; exit $r ;;
identrun)  # identrun lc|d0: 60 steps eager (pool --env JEV_RM_EAGER=1)
  declare -A IX=([lc]=R6IDLC [d0]=R6IDD0); X=${IX[$2]}-L18-s0; [[ ! -e $RUNS/$X ]] || die "$X exists"; run $2 60 $X
  echo "$(date '+%F %T') identity $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R6IDLC-L18-s0 --b R6IDD0-L18-s0 --out round6_dag_identity \
      --note "R5LC-L18-s0 argv 60 steps eager: on1_JTPH / R5LC vs on1_JTPHD / R6D0 (on1_PAID rows at weight 0)"; r=$?
  cp experiments/defect_fix/results/round6_dag_identity.json $RES/identity.json || r=4
  rm -rf $RUNS/R6IDLC-L18-s0 $RUNS/R6IDD0-L18-s0; exit $r ;;
build)
  $V $PY report --root $O > $O/report.log 2>&1 || die "report"
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke.json'))['pass_'] and json.load(open('$RES/identity.json'))['identical'] else 1)" \
    || { echo "STOP smoke / identity (T3)" > $O/VERDICT; exit 1; }
  date '+%F %T' > $O/BUILT ;;
printargv) argv "${@:2}" ;;
dry) echo "R6DAG s0: $(argv dag $N R6DAG-L18-s0 0 | xargs) --save-at 2500,5000,7500"; echo "== R5LC -> R6DAG s0"; diff <(argv lc $N R5LC-L18-s0) <(argv dag $N R6DAG-L18-s0 0)
  echo "== R5LC -> R6DAG s1"; diff <(argv lc $N R5LC-L18-s0) <(argv dag $N R6DAG-L18-s1 1); echo "== identity lc -> d0"; diff <(argv lc 60 R6IDLC-L18-s0) <(argv d0 60 R6IDD0-L18-s0); true ;;
launch)  # (main session, after BUILT and T1 / T2) the two trainings in parallel, each one's PAI 281 --after it; r281 + verdict after both; the ruler
  [[ -f $O/BUILT && ! -f $O/VERDICT ]] || die "not built (or VERDICT)"
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/t1.json'))['passed']['all'] and json.load(open('$RES/t2.json'))['in_canvas']['passed'] else 1)" || die "T1 / T2 (in-canvas, amendment 1) not passed"
  for s in 0 1; do for X in R6DAG-L18-s$s R6DAGx2500-L18-s$s R6DAGx5000-L18-s$s R6DAGx7500-L18-s$s; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done; done
  for f in $F_/NTV0-L18-s0-281ag.csv $F_/R5LC-L18-s0-281ag.csv $F_/R5LC-L18-s1-281ag.csv; do [[ -s $f ]] || die "no stored $f"; done
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt; x=()
  for s in 0 1; do t=$(sub dag-train-s$s "${FT[@]}" --timeout-h 4 -- bash $ME train $s) || exit 1
    y=($(read281 R6DAG-L18-s$s $t)) || exit 1; x+=(${y[-1]})
    echo "$(date '+%F %T') launch s$s (code $h): train $t PAI 281 chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]}" | tee -a $O/stages.txt; done
  r=$(sub dag-r281 --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281) || exit 1
  P0=$DATA_DIR/tmp/pcs_r5l; PT=$DATA_DIR/tmp/pcs_r6d; [[ ! -e $PT ]] || die "$PT exists"; for f in $P0/rows.csv.gz $P0/out/fits/paimix.npz; do [[ -s $f ]] || die "no $f"; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$DATA_DIR/tmp/pcs_r5/site)
  c=$(sub dag-pcs-collect --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only DAG-s0+AG,DAG-s1+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub dag-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  p=$(sub dag-pcs-report --after $(csv ${b[@]} $r) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME pcsrep) || exit 1
  echo "$(date '+%F %T') launch: r281 $r ruler collect $c boot ${b[*]} report $p" | tee -a $O/stages.txt ;;
train) s=$2; run dag $N R6DAG-L18-s$s $s; echo "$(date '+%F %T') train s$s: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
r281)
  rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R5LC-s0-ag=$F_/R5LC-L18-s0-281ag.csv R5LC-s1-ag=$F_/R5LC-L18-s1-281ag.csv
      DAG-s0-ag=$F_/R6DAG-L18-s0-281ag.csv DAG-s1-ag=$F_/R6DAG-L18-s1-281ag.csv)
  $TR $R2PY r281 --read "${rd[@]}" --pairs DAG-s0-ag:NTV0-ag DAG-s1-ag:NTV0-ag DAG-s0-ag:R5LC-s0-ag DAG-s1-ag:R5LC-s1-ag DAG-s1-ag:DAG-s0-ag \
      R5LC-s0-ag:NTV0-ag R5LC-s1-ag:NTV0-ag R5LC-s1-ag:R5LC-s0-ag --out $RES/pai281_DAG_ag.md \
      --title "PAI 281, ACC + GAP: R6DAG (DAgger map teacher) s0 / s1 vs stored R5LC (MV-LC) s0 / s1 and NTV0-L18-s0" || die "r281"
  $V $PY verdict --json $RES/pai281_DAG_ag.json --md $RES/pai281_DAG_ag.md --read "${rd[@]}" || die "verdict"
  date '+%F %T' > $O/R281DONE ;;
pcsrep)
  P0=$DATA_DIR/tmp/pcs_r5l; PT=$DATA_DIR/tmp/pcs_r6d; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  python3 - $P0/out $PT/out <<'PY' || die "bootstrap draws of the old reads differ from stage L's"
import csv, glob, sys                                            # same seeds, same pool: the old subjects' draws must repeat exactly
rd = lambda d: [r for f in sorted(glob.glob(d + "/boot_*.csv")) for r in csv.DictReader(open(f))]  # noqa: E731
a, b = rd(sys.argv[1]), rd(sys.argv[2])
d = max(abs(float(x[k]) - float(y[k])) for x, y in zip(a, b) for k in x if x[k] and y[k])
print("old draws repeat:", len(a), "draws,", len(a[0]), "subjects, max |diff|", d, "PCS")
assert len(a) == len(b) == 100 and d < 0.5, "boot mismatch"
PY
  sj="R3MV-s0+AG=R3MV-L18-s0-281ag.csv,R5LC-s0+AG=R5LC-L18-s0-281ag.csv,R5LC-s1+AG=R5LC-L18-s1-281ag.csv,DAG-s0+AG=R6DAG-L18-s0-281ag.csv,DAG-s1+AG=R6DAG-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" \
      --pairs DAG-s0+AG:R5LC-s0+AG,DAG-s1+AG:R5LC-s1+AG,DAG-s1+AG:DAG-s0+AG > $PT/out/r5.log 2>&1 || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_dag.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_dag.json || die "copy ruler report"
  $V $PY pcs --pcs $RES/pcs_dag.json --json $RES/pai281_DAG_ag.json --md $RES/pai281_DAG_ag.md || die "pcs rows"
  date '+%F %T' > $O/DONE ;;
wait)  # returns when the lane has an outcome (VERDICT / the marker, default DONE) or one of its jobs failed / was cancelled
  M=${2:-DONE}
  until [[ -f $O/VERDICT || -f $O/$M ]] || ! $V - $O/jobs.txt <<'PY'
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
  cat $O/VERDICT 2>/dev/null; for f in BUILT R281DONE DONE; do [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; done; tail -3 $O/stages.txt; true ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
