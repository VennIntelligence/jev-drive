#!/usr/bin/env bash
# ROUND9 NTV1 step 2a (plans/2026-10-24-ntv1-step2.md = registration): the DAgger pool of P1L's own rolls. The student PCL1L-F-s0 rolled under the
# submission serve (ACC + GAP, g10, ALPASIM_FAST=1, CONC 4, frame dumps, KEEP_ASL) on the 719 non-held curated train scenes; round 1's teacher b3
# (DAG_TV=b3: route-lane centre + the human's lateral offset, the student's own served-plan arc lengths). The roll -> rows -> teach -> clean ->
# cache -> pool -> label (T4) -> fwd / r1in job bodies are round6 dag.sh's, unchanged, under this lane's env (DAG_O / DAG_T / DAG_NP / DAG_RES / DAG_TV);
# then merge on1_JTDC = on1_JT's parts + on1_PAIDC (pd, both) -> NTV1D0 / NTV1D -> T3 (a) vs the stored P1L, (b) T = 30 pair -> smokes A / B ->
# command fields -> tax (719 train scores, this roll and round 6's stored R5LC roll) -> report -> $O/DONE. No training, no 281 read.
# Box, repo root; every job through the pool; ids in $O/jobs.txt, stages in $O/stages.txt, a stop reason in $O/VERDICT.
#   dgc.sh dry       argvs and their diff against PCL1L-F-s0's stored argv (no job)
#   dgc.sh chain     T1 / T2 + c000 (6 scenes) roll -> rows -> teach / clean -> chk1 -> rest (submits every other chunk and the build tail)
#   dgc.sh wait      one run_in_background waiter: $O/DONE, $O/VERDICT or a failed / cancelled job of this lane
#   job bodies: rest | merge | weights | t3a | t3run <a|b> | t3cmp | smoke <a|b> | cmdf | tax | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv1_step2; ME=experiments/round9/scripts/dgc.sh; DS=experiments/round6/scripts/dag.sh; DG=experiments/round6/scripts/dag.py
PY=experiments/round9/scripts/dgc.py; PC=experiments/round9/scripts/pcl.py; V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; RF=experiments/route_follow/scripts; CR=$DATA_DIR/runs/op_parity/cache
RUNS=$DATA_DIR/runs/op_parity/runs; RES=experiments/round9/results/ntv1_step2; T=PCL1L-F-s0; L=${T,,}; NP=PAIDC; NM=JTDC; TV=b3; PRIO=${PRIO:-14}
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # = dag.sh round 1
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"            # = on1_JT (front.parts.json order)
TCFG="$DATA_DIR/runs/round5/s3/roll/A/runs/*/sim/generated-user-config-0.yaml,$DATA_DIR/runs/round5/d3/roll/A/runs/*/sim/generated-user-config-0.yaml"
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
export DAG_O=$O DAG_T=$T DAG_NP=$NP DAG_RES=$RES DAG_TV=$TV
DENV=(--env DAG_O=$O --env DAG_T=$T --env DAG_NP=$NP --env DAG_RES=$RES --env DAG_TV=$TV)
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)                    # eager, as P1L
declare -A TG=([t3]=NT1DT3-F-s0 [pa]=NT1DPA-F-s0 [pb]=NT1DPB-F-s0 [a]=NT1DSA-F-s0 [b]=NT1DSB-F-s0)
mkdir -p $O/pool $O/checks $O/w $O/tax $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round9-ntv1d --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "${DENV[@]}" "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() {  # -> "<id> <chunk>" lines of the chunks submitted now (pai_box.sh skips finished / pending chunks); = dag.sh roll
  local f=$O/roll/submit.$(date +%s%N).txt
  SERVE="$SERVE PAI_DUMP=100000" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO KEEP_ASL=1 bash experiments/alpasim/scripts/pai_box.sh submit $O/roll $T > $f \
    || die "roll submit"
  grep -q 'not submitted' $f && die "roll: $(grep 'not submitted' $f)"
  awk '$2 ~ /^pai-/ {print $1, "roll-" substr($2, length($2) - 3)}' $f >> $O/jobs.txt
  awk '$2 ~ /^pai-/ {print $1, substr($2, length($2) - 3)}' $f; }
chunk_tail() {  # chunk_tail <roll id> <chunk> -> "<rows id> <teach id> <clean id>" (clean only after rows: rows checks every file's tag)
  local x y z
  x=$(sub dgc-rows-$2 --after $1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $DS rows $2) || exit 1
  y=$(sub dgc-teach-$2 --after $x --vram 0.1 --cpu 8 --ram 32 --timeout-h 2 -- bash $DS teach $2) || exit 1
  z=$(sub dgc-clean-$2 --after $x --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $DS clean $2) || exit 1
  echo "$x $y $z"; }
argv() { mapfile -t A < <($V $PY argv "$@"); [[ ${#A[@]} -gt 5 ]] || die "argv $*"; }
case ${1:-} in
dry)
  for v in p1l t3 a b; do echo "== $v"; argv $v X; $V - "${A[@]}" <<'PY'
import difflib, sys
sys.path.insert(0, "experiments/round9/scripts")
import pcl as C
b, x = C.stored_argv("PCL1L-F-s0"), sys.argv[1:]
x = [("PCL1L-F-s0" if t == "X" else t) for t in x]
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, x, autojunk=False).get_opcodes():
    if op != "equal":
        print(f"  {op}: stored {b[i1:i2]} -> {x[j1:j2]}")
PY
  done ;;
chain)
  [[ ! -e $O/roll && ! -e $O/DONE ]] || die "$O has a chain: never overwritten"
  [[ -f $RUNS/$T/ckpt-final.pt ]] || die "no $T"
  for X in ${TG[@]}; do [[ ! -e $RUNS/$X && -z $(ls -d $DATA_DIR/runs/route_mem/train-$X 2>/dev/null) ]] || die "$X exists"; done
  for p in on1_$NP on1_$NM; do [[ ! -e $CR/$p ]] || die "$p exists"; done
  [[ ! -e $DATA_DIR/tmp/dag_stage/${L}_c000 ]] || die "staging dir ${L}_c000 exists"
  git rev-parse --short HEAD > $O/code.txt; $V $DG lists --root $O || die lists
  mkdir -p $O/roll/chunks; cp $O/lists/c000.txt $O/roll/chunks/
  t1=$(sub dgc-t1 --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- "${EA[@]}" $AP $DG t1 --cfg-glob "$TCFG" --workers 32) || exit 1
  t2=$(sub dgc-t2 --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- "${EA[@]}" $AP $DG t2 --workers 16) || exit 1
  read -r id c < <(roll) || exit 1; [[ $c == c000 ]] || die "first roll chunk $c"
  read -r x y z < <(chunk_tail $id c000) || exit 1
  k=$(sub dgc-chk1 --after $y,$z --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $DS chk1) || exit 1
  r=$(sub dgc-rest --after $k --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt)): T1 $t1 T2 $t2 c000 roll $id rows $x teach $y clean $z chk1 $k rest $r" | tee -a $O/stages.txt ;;
rest)
  cp $O/lists/c[0-9][0-9][0-9].txt $O/roll/chunks/; out=$(roll) || exit 1; w=(); ro=()
  while read -r id c; do [[ -n $id ]] || continue; ro+=($id); w+=($(chunk_tail $id $c)) || exit 1; done <<< "$out"
  h=$(git rev-parse --short HEAD); echo "$h rest" >> $O/code.txt
  l=$(sub dgc-link --after $(csv ${w[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $DS link) || exit 1
  ca=$(sub dgc-cache --after $l --vram 20 --cpu 8 --ram 64 --timeout-h 8 -- bash $DS cache) || exit 1
  po=$(sub dgc-pool --after $ca --vram 0.1 --cpu 8 --ram 96 --timeout-h 3 -- bash $DS pool) || exit 1
  lb=$(sub dgc-label --after $po --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $DS label) || exit 1
  fw=$(sub dgc-fwd --after $lb --vram 16 --cpu 8 --ram 120 --timeout-h 3 -- bash $DS fwd) || exit 1
  gi=$(sub dgc-r1in --after $fw --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $DS r1in) || exit 1
  m=$(sub dgc-merge --after $gi --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME merge) || exit 1
  wt=$(sub dgc-weights --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME weights) || exit 1
  ta=$(sub dgc-t3a --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3a) || exit 1
  pa=$(sub dgc-t3run-a --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3run pa) || exit 1
  pb=$(sub dgc-t3run-b --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3run pb) || exit 1
  tb=$(sub dgc-t3cmp --after $pa,$pb --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME t3cmp) || exit 1
  sa=$(sub dgc-smoke-a --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke a) || exit 1
  sb=$(sub dgc-smoke-b --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke b) || exit 1
  cf=$(sub dgc-cmdf --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME cmdf) || exit 1
  tx=$(sub dgc-tax --after $(csv ${ro[@]}) --vram 0.1 --cpu 16 --ram 48 --timeout-h 4 -- bash $ME tax) || exit 1
  rp=$(sub dgc-report --after $ta,$tb,$sa,$sb,$cf,$tx --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') rest (code $h): rolls ${ro[*]} chunk tails ${w[*]} link $l cache $ca pool $po label $lb fwd $fw r1in $gi merge $m" \
       "weights $wt T3a $ta T3b $pa $pb $tb smoke $sa $sb cmdf $cf tax $tx report $rp" | tee -a $O/stages.txt ;;
merge)
  [[ ! -e $CR/on1_$NM ]] || die "on1_$NM exists: never overwritten"
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 100 )) || die "free $free GB < 100 GB"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_$NP:pd:both --out $NM --nan-keys fut_reg || die merge
  bash $ME mfix ;;
mfix) $TR $PY mfix || die "mfix: the merged prefix differs from on1_JT" ;;   # attempt 1's dag.py mfix raised KeyError on on1_JT's extra keys
tail2)  # attempt 2 (after the mfix check-script fault): mfix -> weights -> T3 / smokes -> cmdf -> report; the merged pool on1_JTDC kept
  [[ -f $CR/on1_$NM/tab.npz && ! -f $O/DONE ]] || die "no on1_$NM or DONE"; h=$(git rev-parse --short HEAD); echo "$h tail2" >> $O/code.txt
  m=$(sub dgc-mfix --vram 0.1 --cpu 4 --ram 160 --timeout-h 2 -- bash $ME mfix) || exit 1
  wt=$(sub dgc-weights --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME weights) || exit 1
  ta=$(sub dgc-t3a --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3a) || exit 1
  pa=$(sub dgc-t3run-a --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3run pa) || exit 1
  pb=$(sub dgc-t3run-b --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME t3run pb) || exit 1
  tb=$(sub dgc-t3cmp --after $pa,$pb --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME t3cmp) || exit 1
  sa=$(sub dgc-smoke-a --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke a) || exit 1
  sb=$(sub dgc-smoke-b --after $wt "${FE[@]}" --timeout-h 1 -- bash $ME smoke b) || exit 1
  cf=$(sub dgc-cmdf --after $m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME cmdf) || exit 1
  rp=$(sub dgc-report --after $ta,$tb,$sa,$sb,$cf --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') tail2 (code $h): mfix $m weights $wt T3a $ta T3b $pa $pb $tb smoke $sa $sb cmdf $cf report $rp" | tee -a $O/stages.txt ;;
weights)
  $V $PY weights --k-dag 0 --out NTV1D0 || die NTV1D0
  $V $PY weights --k-dag 8 --out NTV1D || die NTV1D ;;
t3a)  # T3 (a): P1L's argv on on1_JTDC / NTV1D0 to step 125, every scalar <= 100 vs the stored PCL1L-F-s0 bit for bit (pre-T: the pool is loaded, not drawn)
  X=${TG[t3]}; argv t3 $X; echo "$(date '+%F %T') T3a (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $V $PC ident --tag $X --ref $T --upto 100 --stop 125 --out $RES/t3a.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$X; exit $r ;;
t3run)  # T3 (b): P1L's argv, 60 steps, T = 30: pa on on1_JT / JT00S-rb0, pb on on1_JTDC / NTV1D0
  X=${TG[$2]}; [[ $2 == pa ]] && argv p1l $X --steps 60 --on-start 30 --save-at none || argv t3 $X --steps 60 --on-start 30 --save-at none
  echo "$(date '+%F %T') T3b $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/t3b_$2.log 2>&1 || die "T3b run $2 (log $O/t3b_$2.log)" ;;
t3cmp) $V $PY pair --a ${TG[pa]} --b ${TG[pb]} --upto 60 --out $RES/t3b.json; r=$?; rm -rf $RUNS/${TG[pa]} $RUNS/${TG[pb]}; exit $r ;;
smoke)
  X=${TG[$2]}; argv $2 $X --steps 60 --on-start 30 --save-at none --mix; echo "$(date '+%F %T') smoke $2: ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" > $O/smoke_$2.log 2>&1 || die "smoke $2 (log $O/smoke_$2.log)"
  $V $PY smoke --var $2 --tag $X --on-start 30 --out $RES/smoke_$2.json; r=$?; rm -rf $RUNS/$X; exit $r ;;
cmdf) $V $PY cmdf ;;
tax)  # 719 train-scene scores: this roll and round 6's stored R5LC-L18-s0 roll (the same lists), rf_tax + pai_fix facts (= pai_fix.sh job-tax)
  for x in roll:$O/roll ref_r5lc:$DATA_DIR/runs/round6/dag/roll; do n=${x%%:*}; r=${x#*:}
    $AP $RF/rf_tax.py extract --runs $(ls -d $r/runs/*/sim) --out $O/tax/$n.pkl --workers 16 > $O/tax/$n.log 2>&1 || die "extract $n"
    $V experiments/pai_fix/scripts/pai_fix.py facts --pkl $O/tax/$n.pkl --runs $r --out $O/tax/$n.csv >> $O/tax/$n.log 2>&1 || die "facts $n"; done ;;
report) $V $PY report > $O/report.log 2>&1 || die "report (log $O/report.log)"; date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/VERDICT || -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
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
  cat $O/VERDICT 2>/dev/null; [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -4 $O/stages.txt; true ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
