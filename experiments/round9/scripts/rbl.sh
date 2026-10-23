#!/usr/bin/env bash
# ROUND9 RB0-LONG (plans/2026-10-23-rb0-long.md): RB0-F-s0's stored argv (shipped weights, no --init, eager) with only --steps 19500, the tag and a
# --save-at 10000 snapshot, then PAI 281 ACC + GAP (Harmonizer OFF). Box, repo root; every job through the pool, the whole chain in one submit;
# ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/rb0_long.
#   rbl.sh dry     the arm's argv and its token diff against RB0-F-s0's stored meta.json argv (no job)
#   rbl.sh all     identity gate (RB0-F's argv, --steps 10000, stopped after step 75, vs RB0-F-s0's stored scalars) + smoke (arm argv, 60 steps,
#                  --save-at 30) -> RB0L-F-s0 (19 500 steps, snapshot RB0L-F-s0x10000) -> PAI 281 ACC + GAP OFF (NTV0's 8 chunk lists, g10, FAST,
#                  CONC 4) + tax -> report (r2.py r281 vs RB0-F-s0 / NTV0-L18-s0 [+ NTV0NS-L18-s0 if its facts exist], rbl.py verdict) -> $O/DONE
#   rbl.sh wait    one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: train <tag> [steps] [save-at] | ident | smoke | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/rb0_long; ME=experiments/round9/scripts/rbl.sh; PY=experiments/round9/scripts/rbl.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts
T=RB0L-F-s0; SM=RB0LSM-F-s0; ID=RB0LID-F-s0; CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
FT=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)   # = pb.sh's RB0-FR-s0 declaration (eager, as RB0-F)
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-rbl --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
argv() {  # argv <tag> [steps] [save-at]: RB0-F-s0's stored argv, one token a line; steps / save-at given = replaced / appended
  python3 - "$@" <<'PY'
import glob, json, os, sys
a = json.load(open(sorted(glob.glob(f"{os.environ['DATA_DIR']}/runs/route_mem/train-RB0-F-s0/*/meta.json"))[-1]))["argv"]
assert "--init" not in a and "--save-at" not in a, a
def put(k, x):
    assert a.count(k) == 1, (k, a)
    a[a.index(k) + 1] = x
put("--tag", sys.argv[1])
if sys.argv[2:3] and sys.argv[2]:
    put("--steps", sys.argv[2])
if sys.argv[3:4]:
    a += ["--save-at", sys.argv[3]]
print("\n".join(a))
PY
}
case ${1:-} in
dry)
  mapfile -t A < <(argv $T 19500 10000); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "${A[*]}"; python3 - "${A[@]}" <<'PY'
import difflib, glob, json, os, sys
a = json.load(open(sorted(glob.glob(f"{os.environ['DATA_DIR']}/runs/route_mem/train-RB0-F-s0/*/meta.json"))[-1]))["argv"]
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, sys.argv[1:], autojunk=False).get_opcodes():
    if op != "equal":
        print(f"{op}: RB0-F {a[i1:i2]} -> arm {sys.argv[1:][j1:j2]}")
PY
  ;;
all)
  for X in $T ${T}x10000 $SM ${SM}x30 $ID; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  [[ ! -e $O/DONE && ! -e $O/jobs.txt ]] || die "already ran"
  for f in NTV0-L18-s0-281ag RB0-F-s0-281ag; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  i=$(sub rbl-ident "${FT[@]}" --timeout-h 1 -- bash $ME ident) || exit 1
  s=$(sub rbl-smoke "${FT[@]}" --timeout-h 1 -- bash -c "bash $ME train $SM 60 30 && bash $ME smoke") || exit 1
  t=$(sub rbl-train-$T --after $i,$s "${FT[@]}" --timeout-h 3 -- bash $ME train $T 19500 10000) || exit 1
  R=$O/pai281/$T-ag; [[ ! -e $R ]] || die "$R exists"
  AFTER=$t CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $T \
    > $R.submit.txt || die "pai submit"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  x=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281ag $R | tail -1) && [[ -n $x ]] || die "tax"
  echo "$x pai_fix-tax-$T-281ag" >> $O/jobs.txt
  r=$(sub rbl-report --after $x --vram 0.1 --cpu 8 --ram 16 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): identity $i smoke $s train $t 281 $(echo $ids) tax $x report $r" | tee -a $O/stages.txt ;;
train)  # train <tag> [steps] [save-at]
  mapfile -t A < <(argv "${@:2}"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $2 (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
ident)  # RB0-F's argv (only the tag changed, --steps 10000 kept), stopped once step 100's scalars exist; steps <= 75 vs the stored RB0-F-s0
  mapfile -t A < <(argv $ID); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident $ID (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  $V $PY ident --tag $ID --ref RB0-F-s0 --upto 75 --out $O/results/identity.json -- $TR "${A[@]}"; r=$?
  rm -rf $RUNS/$ID; exit $r ;;
smoke) $V $PY smoke --tag $SM --snap ${SM}x30 --out $O/results/smoke.json; r=$?; (( r == 0 )) && rm -rf $RUNS/$SM $RUNS/${SM}x30; exit $r ;;
report)
  E=experiments/round2/scripts/r2.py
  $TR $E r281 --read RB0L-ag=$F_/$T-281ag.csv RB0F-ag=$F_/RB0-F-s0-281ag.csv NTV0s0-ag=$F_/NTV0-L18-s0-281ag.csv \
      --pairs RB0L-ag:RB0F-ag RB0L-ag:NTV0s0-ag NTV0s0-ag:RB0F-ag --out $O/results/pai281.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF: RB0L-F-s0 (RB0-F's argv, 19 500 steps) vs RB0-F-s0 and NTV0-L18-s0" || die r281
  rm -f $O/results/pai281_ns.*
  if [[ -s $F_/NTV0NS-L18-s0-281ag.csv ]]; then
    $TR $E r281 --read RB0L-ag=$F_/$T-281ag.csv NTV0NS-ag=$F_/NTV0NS-L18-s0-281ag.csv RB0F-ag=$F_/RB0-F-s0-281ag.csv NTV0s0-ag=$F_/NTV0-L18-s0-281ag.csv \
        --pairs RB0L-ag:NTV0NS-ag --out $O/results/pai281_ns.md \
        --title "PAI 281, ACC + GAP, Harmonizer OFF: RB0L-F-s0 vs NTV0NS-L18-s0 (NTV0 without S rows)" || die "r281 ns"; fi
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
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,10p "$0"; exit 2 ;;
esac
