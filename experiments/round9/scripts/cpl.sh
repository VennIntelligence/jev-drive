#!/usr/bin/env bash
# ROUND9 CMD-PILOT (plans/2026-10-23-command-design.md, section 5): the command = rmem.cmd_heading (20 deg, 40 m) on the fed route memory on EVERY
# training row (rm_train --cmd-rule heading-all) and at serve (JEV_CMD=heading). Arms from shipped weights, seed 0, eager:
#   HF   = RB0-F-s0's stored argv   + --cmd-rule heading-all
#   HP50 = RB0-P50-s0's stored argv + --cmd-rule heading-all
# Box, repo root; every job through the pool, the whole chain in one submit; ids in $O/jobs.txt, stages in $O/stages.txt, $O = runs/round9/cmd_pilot.
#   cpl.sh dry     each arm's argv and its token diff against the reference run's stored meta.json argv (no job)
#   cpl.sh all     identity gate (RB0-F's argv, flag off, stopped after step 75, vs RB0-F-s0's stored scalars; rbl.py ident) + per arm a 60-step
#                  smoke with the flag on (cpl.py smoke: finite, per-part change shares vs cmdd.py rows, fed command per row vs cmdd.py's offline
#                  recomputation) -> the two trainings -> PAI 281 ACC + GAP OFF + JEV_CMD=heading (NTV0's 8 chunk lists, g10, FAST, CONC 4) + tax
#                  -> cmdagree (served command vs the rule, dl.py) -> report (r2.py r281 + scorer_v2 new + cpl.py verdict) -> $O/DONE
#   cpl.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: train <arm> [tag] [steps] | ident | smoke <arm> | cmdagree <arm> | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/cmd_pilot; ME=experiments/round9/scripts/cpl.sh; PY=experiments/round9/scripts/cpl.py; RBL=experiments/round9/scripts/rbl.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks; ID=CPLID-F-s0
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_CMD=heading"; WIZ=controller.gains.rel_front_steering_angle_weight=10
FT=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)   # = rbl.sh / pb.sh (eager, as RB0-F / RB0-P50)
declare -A REF=([HF]=RB0-F-s0 [HP50]=RB0-P50-s0)
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-cpl --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
argv() {  # argv <ref> <tag> <flag 0|1> [steps]: the reference run's stored argv, one token a line; tag replaced, steps replaced, flag appended
  python3 - "$@" <<'PY'
import glob, json, os, sys
ref, tag, flag = sys.argv[1:4]
a = json.load(open(sorted(glob.glob(f"{os.environ['DATA_DIR']}/runs/route_mem/train-{ref}/*/meta.json"))[-1]))["argv"]
assert "--init" not in a and "--cmd-rule" not in a, a
def put(k, x):
    assert a.count(k) == 1, (k, a)
    a[a.index(k) + 1] = x
put("--tag", tag)
if sys.argv[4:5]:
    put("--steps", sys.argv[4])
if flag == "1":
    a += ["--cmd-rule", "heading-all"]
print("\n".join(a))
PY
}
case ${1:-} in
dry)
  for m in HF HP50; do mapfile -t A < <(argv ${REF[$m]} $m-s0 1); [[ ${#A[@]} -gt 5 ]] || die "argv $m"
    echo "$m: ${A[*]}"; python3 - ${REF[$m]} "${A[@]}" <<'PY'
import difflib, glob, json, os, sys
a = json.load(open(sorted(glob.glob(f"{os.environ['DATA_DIR']}/runs/route_mem/train-{sys.argv[1]}/*/meta.json"))[-1]))["argv"]
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, sys.argv[2:], autojunk=False).get_opcodes():
    if op != "equal":
        print(f"  {op}: {sys.argv[1]} {a[i1:i2]} -> arm {sys.argv[2:][j1:j2]}")
PY
  done ;;
all)
  for X in HF-s0 HP50-s0 CPLSM-HF-s0 CPLSM-HP50-s0 $ID; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in NTV0-L18-s0-281ag RB0-F-s0-281ag RB0-P50-s0-281ag; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  [[ -s $DATA_DIR/runs/round9/cmd_design/rows.parquet ]] || die "no cmdd rows.parquet"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  i=$(sub cpl-ident "${FT[@]}" --timeout-h 1 -- bash $ME ident) || exit 1
  X=(); for m in HF HP50; do
    s=$(sub cpl-smoke-$m "${FT[@]}" --timeout-h 1 -- bash -c "bash $ME train $m CPLSM-$m-s0 60 && bash $ME smoke $m") || exit 1
    t=$(sub cpl-train-$m --after $i,$s "${FT[@]}" --timeout-h 3 -- bash $ME train $m) || exit 1
    R=$O/pai281/$m-s0-ag; [[ ! -e $R ]] || die "$R exists"
    AFTER=$t CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $m-s0 \
      > $R.submit.txt || die "pai submit $m"
    ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$m: not 8 chunks"
    awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
    x=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $m-s0-281ag $R | tail -1) && [[ -n $x ]] || die "tax $m"
    echo "$x pai_fix-tax-$m-s0-281ag" >> $O/jobs.txt
    c=$(sub cpl-cmdagree-$m --after $(paste -sd, <<< "$ids") --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME cmdagree $m) || exit 1
    X+=($x $c)
    echo "$(date '+%F %T') $m (code $h): smoke $s train $t 281 $(echo $ids) tax $x cmdagree $c" | tee -a $O/stages.txt; done
  r=$(sub cpl-report --after $(IFS=,; echo "${X[*]}") --vram 0.1 --cpu 8 --ram 24 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): identity $i report $r" | tee -a $O/stages.txt ;;
train)  # train <arm> [tag] [steps]
  m=$2; [[ -n ${REF[$m]:-} ]] || die "arm $m: HF | HP50"; tag=${3:-$m-s0}
  mapfile -t A < <(argv ${REF[$m]} $tag 1 ${4:-}); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $tag (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
ident)  # flag off: RB0-F's argv (only the tag changed), stopped once step 100's scalars exist; steps <= 75 vs the stored RB0-F-s0 bit for bit
  mapfile -t A < <(argv RB0-F-s0 $ID 0); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $ID (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $RBL ident --tag $ID --ref RB0-F-s0 --upto 75 --out $O/results/identity.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$ID; exit $r ;;
smoke) $V $PY smoke --tag CPLSM-$2-s0 --out $O/results/smoke_$2.json; r=$?; (( r == 0 )) && rm -rf $RUNS/CPLSM-$2-s0; exit $r ;;
cmdagree) DL_RES=$O/results $V experiments/round9/scripts/dl.py cmdagree --label $2 --read $O/pai281/$2-s0-ag || true
  [[ -s $O/results/cmdagree_$2.json ]] || die "cmdagree $2: no record" ;;
report)
  $TR experiments/round2/scripts/r2.py r281 --read HF=$F_/HF-s0-281ag.csv HP50=$F_/HP50-s0-281ag.csv RB0F=$F_/RB0-F-s0-281ag.csv \
      RB0P50=$F_/RB0-P50-s0-281ag.csv NTV0=$F_/NTV0-L18-s0-281ag.csv --pairs HF:RB0F HP50:RB0P50 HP50:HF RB0P50:RB0F --out $O/results/pai281.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF (old scorer): HF / HP50 (heading command on every row and at serve) vs RB0-F-s0 / RB0-P50-s0" || die r281
  NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python   # new scorer (decision 453)
  for m in HF HP50; do rm -rf $O/sv2_$m
    PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2_$m --reads $m-s0-ag --workers 8 > $O/sv2_$m.log 2>&1 \
      || echo "$(date '+%F %T') scorer_v2 new $m failed (log $O/sv2_$m.log)" | tee -a $O/stages.txt; done
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
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -4 $O/stages.txt; true ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
