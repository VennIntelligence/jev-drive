#!/usr/bin/env bash
# ROUND9 NTV0-NOS (plans/2026-10-23-ntv0-nos.md): NTV0-L18-s0's stored argv with the S rows removed (--on-k 32 -> 0, no --on-wfile), then
# PAI 281 ACC + GAP (Harmonizer OFF). Box, repo root; every job through the pool, the whole chain in one submit; ids in $O/jobs.txt, stages in
# $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/ntv0_nos.
#   nos.sh dry        the arm's argv and its token diff against NTV0-L18-s0's stored meta.json argv (no job)
#   nos.sh all        smoke NTV0NSSM-L18-s0 (60 steps) + gate -> NTV0NS-L18-s0 (9 500 steps) -> PAI 281 ACC + GAP OFF (NTV0's 8 chunk lists, g10,
#                     FAST, CONC 4) + tax; NTV0-L18-s1's stored read taxed (CPU) -> report (r2.py r281, scorer_v2 new, nos.py verdict) -> $O/DONE
#                     + (informational, not a gate) identity: NTV0's own argv 60 steps on HEAD (NTV0IDSM-L18-s0) vs NTV0's stored step 25 / 50
#                     scalars and on_hash_60 (NTV0 trained at 1599227, compiled)
#   nos.sh wait       one run_in_background waiter: $O/DONE, or a failed / cancelled job of this lane
#   job bodies: train <tag> [steps] (KEEPS=1: NTV0's argv unchanged but tag / steps) | smoke | ident | s1tax | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/ntv0_nos; ME=experiments/round9/scripts/nos.sh; PY=experiments/round9/scripts/nos.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; T=NTV0NS-L18-s0; SM=NTV0NSSM-L18-s0; ID=NTV0IDSM-L18-s0
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks; S1=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s1
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)        # = jt.sh's NTV0 training declaration
PRIO=${PRIO:-14}
mkdir -p $O/pool $O/results $O/pai281
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <name> <pool args...>
  local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-nos --name "$name" --tries 1 --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
    | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
argv() {  # argv <tag> [steps]: NTV0-L18-s0's stored argv, one token a line, with the S rows removed
  python3 - "$@" <<'PY'
import glob, json, os, sys
D = os.environ["DATA_DIR"]
a = json.load(open(sorted(glob.glob(f"{D}/runs/route_mem/train-NTV0-L18-s0/*/meta.json"))[-1]))["argv"]
def put(k, x, last=False):
    i = [j for j, t in enumerate(a) if t == k]
    assert i and (last or len(i) == 1), (k, a)
    a[i[-1] + 1] = x
put("--tag", sys.argv[1])
if len(sys.argv) > 2:
    put("--steps", sys.argv[2])
if os.environ.get("KEEPS") == "1":             # identity run: NTV0's argv unchanged but --tag / --steps
    print("\n".join(a)); sys.exit(0)
assert a.count("--on-k") == 2 and a[a.index("--on-k") + 1] == "13" and a[len(a) - 1 - a[::-1].index("--on-k") + 1] == "32", a
put("--on-k", "0", last=True)                  # the effective --on-k (RSG's 13 is overridden by the later 32): no pool row in any batch
i = a.index("--on-wfile"); del a[i:i + 2]      # rm_train asserts --on-wfile needs k_on > 0; with no pool row drawn it only set the draw / probes
print("\n".join(a))
PY
}
read281() {  # read281 <tag> <after> -> "<chunk ids...> <tax id>" (= pb.sh read281 ag: NTV0's chunk table, ACC + GAP, g10, FAST, CONC 4)
  local R=$O/pai281/$1-ag ids t; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281ag $R | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1-281ag" >> $O/jobs.txt; echo $ids $t; }
case ${1:-} in
dry)
  mapfile -t A < <(argv $T); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "${A[*]}"; python3 - "${A[@]}" <<'PY'
import difflib, glob, json, os, sys
a = json.load(open(sorted(glob.glob(f"{os.environ['DATA_DIR']}/runs/route_mem/train-NTV0-L18-s0/*/meta.json"))[-1]))["argv"]
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, sys.argv[1:], autojunk=False).get_opcodes():
    if op != "equal":
        print(f"{op}: NTV0 {a[max(0, i1 - 1):i2]} -> arm {sys.argv[1:][max(0, j1 - 1):j2]}")
PY
  ;;
all)
  for X in $T $SM $ID; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  [[ ! -e $O/DONE ]] || die "already ran"
  for f in NTV0-L18-s0-281ag RB0-F-s0-281ag; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  [[ -s $DATA_DIR/runs/alpasim/scorer_v2/report/per_scene.csv ]] || die "no scorer_v2 per_scene"
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  s=$(sub nos-smoke "${FT[@]}" --timeout-h 1 -- bash -c "bash $ME train $SM 60 && bash $ME smoke") || exit 1
  t=$(sub nos-train-$T --after $s "${FT[@]}" --timeout-h 3 -- bash $ME train $T) || exit 1
  i=$(sub nos-ident "${FT[@]}" --env KEEPS=1 --timeout-h 1 -- bash -c "bash $ME train $ID 60 && KEEPS= bash $ME ident") || exit 1
  y=($(read281 $T $t)) || exit 1
  x=$(sub nos-s1tax --vram 0.1 --cpu 2 --ram 8 --timeout-h 2 -- bash $ME s1tax) || exit 1
  r=$(sub nos-report --after ${y[-1]},$x --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') all (code $h): smoke $s identity $i train $t 281 ${y[*]} s1 tax $x report $r" | tee -a $O/stages.txt ;;
train)  # train <tag> [steps]
  mapfile -t A < <(argv "${@:2}"); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $2 (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
smoke) $V $PY smoke --tag $SM --out $O/results/smoke.json; r=$?; (( r == 0 )) && rm -rf $RUNS/$SM; exit $r ;;
ident) $V $PY ident --tag $ID --ref NTV0-L18-s0 --out $O/results/identity.json; r=$?; rm -rf $RUNS/$ID; exit $r ;;
s1tax)  # NTV0-L18-s1's stored read (no 281 facts yet): rf_tax + facts in this job (= pai_fix.sh job-tax), skipped if another lane wrote it
  [[ -s $F_/NTV0-L18-s1-281ag.csv ]] && { echo "exists: $F_/NTV0-L18-s1-281ag.csv"; exit 0; }
  bash experiments/pai_fix/scripts/pai_fix.sh job-tax NTV0-L18-s1-281ag $S1 ;;
report)
  E=experiments/round2/scripts/r2.py
  $TR $E r281 --read NTV0NS-ag=$F_/$T-281ag.csv NTV0s0-ag=$F_/NTV0-L18-s0-281ag.csv RB0F-ag=$F_/RB0-F-s0-281ag.csv \
      --pairs NTV0NS-ag:NTV0s0-ag NTV0NS-ag:RB0F-ag NTV0s0-ag:RB0F-ag --out $O/results/pai281.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF: NTV0NS-L18-s0 (NTV0 without S rows) vs NTV0-L18-s0 and RB0-F-s0" || die r281
  $TR $E r281 --read NTV0NS-ag=$F_/$T-281ag.csv NTV0s0-ag=$F_/NTV0-L18-s0-281ag.csv NTV0s1-ag=$F_/NTV0-L18-s1-281ag.csv \
      --pairs NTV0NS-ag:NTV0s1-ag NTV0s1-ag:NTV0s0-ag --out $O/results/pai281_s1.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF: NTV0NS-L18-s0 vs NTV0-L18-s1 (and s1 - s0)" || die "r281 s1"
  rm -rf $O/sv2; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
  PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py score --variant new --root $O/sv2 --reads $T-ag --workers 16 > $O/sv2.log 2>&1 \
    || echo "scorer_v2 new failed (log $O/sv2.log); verdict without it" | tee -a $O/stages.txt
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
*) sed -n 2,9p "$0"; exit 2 ;;
esac
