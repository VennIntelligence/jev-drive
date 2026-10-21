#!/usr/bin/env bash
# ROUND6 line C, FIX (plans/2026-10-21-fix-rows.md): R6FIX = R5LC-L18-s<seed>'s recorded argv (MV-LC) with draw R6FIX (R5LC with the F1 stretch
# rows and the F2 no-lead slow-down rows at weight 0, parts renormalised). Box, repo root; every job through the pool at PRIO 22 (PRIO env), the
# ruler's CPU jobs at PRIO 15. State in $O = $DATA_DIR/runs/round6/fix (jobs.txt, stages.txt, w/, pai281/, flags).
#   fix.sh rows             one CPU job: fix.py rows -> w/R6FIX.npz, w/R6FIXI.npz, results/fix/fix_rows.{md,json} -> $O/ROWSDONE
#   fix.sh chain            (after ROWSDONE and the prereg) one submission: smoke (60 steps) + identity (R5LC-s0 argv 60 steps eager: R5LC vs
#                           R6FIXI, df.py bitcmp) -> launch: trainings s0 / s1 in parallel (9 500 steps, --save-at 2500,5000,7500), each one's PAI 281
#                           ACC + GAP --after it -> r281 report + new breaks; local PCS ruler (collect, 20 bootstrap shards, report) -> verdict -> DONE
#   fix.sh dry              the training commands and the diff against R5LC's recorded argv
#   fix.sh wait <FLAG>      one blocking waiter (run_in_background): returns at $O/<FLAG>, $O/STOP or a failed / cancelled job of this lane
#   job bodies: rowsjob | smoketrain | identrun <lc|fixi> | ident | launch | train <seed> | r281 | pcsrep
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round6/fix; ME=experiments/round6/scripts/fix.sh; PY=experiments/round6/scripts/fix.py; R2PY=experiments/round2/scripts/r2.py
R5PY=experiments/round5/scripts/r5.py; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; RUNS=$DATA_DIR/runs/op_parity/runs
F_=$DATA_DIR/runs/pai_fix/facts; RES=experiments/round6/results/fix; DFR=experiments/defect_fix/results; PRIO=${PRIO:-22}; N=9500
G10=controller.gains.rel_front_steering_angle_weight=10; FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
stop() { echo "$(date '+%F %T') $1" | tee $O/STOP >> $O/stages.txt; exit 2; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round6-fix --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check \
     --log-dir $O/pool/$name.$(date +%m%d%H%M%S) "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
argv() {  # argv <variant> <steps> <tag> <seed>: R5LC-L18-s<seed>'s recorded argv (no --save-at); fix = --on-wfile R6FIX; fixi = R6FIXI; lc = as recorded
  local M; M=$(ls -d $DATA_DIR/runs/route_mem/train-R5LC-L18-s$4/*/ | tail -1)meta.json; [[ -f $M ]] || die "no $M"
  python3 - "$M" "$@" "$O/w" <<'PY'
import json, sys
m, var, steps, tag, seed, w6 = sys.argv[1:7]
a = json.load(open(m))["argv"]
last = lambda k: len(a) - 1 - a[::-1].index(k)  # noqa: E731
assert a[a.index("--init") + 1] == "NTV0-L18-s0" and a[a.index("--tag") + 1] == f"R5LC-L18-s{seed}" and a[last("--on-wfile") + 1].endswith("/R5LC.npz")
assert a[last("--on") + 1] == "on1_JTPH" and a[last("--on-k") + 1] == "64" and a[last("--seed") + 1] == seed and "--batch" not in a and "--on-road-pai" not in a
a[a.index("--tag") + 1] = tag
a[last("--steps") + 1] = steps
i = a.index("--save-at"); del a[i:i + 2]
if var in ("fix", "fixi"):
    a[last("--on-wfile") + 1] = f"{w6}/{'R6FIX' if var == 'fix' else 'R6FIXI'}.npz"
print("\n".join(a))
PY
}
run() { local L; L=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$L"; }
read281() {  # read281 <tag> [after]: PAI 281 ACC + GAP as round 5's read281 (NTV0's chunk lists, g10, FAST 1, CONC 4) -> "<chunk ids...> <facts id>"
  local t=$1 R=$O/pai281/$1-ag ids x; mkdir -p $O/pai281
  AFTER=${2:-} CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" \
    WIZ=$G10 CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $t > $R.submit.txt || die "pai submit $t"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$t: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  x=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $t-281ag $R | tail -1) && [[ -n $x ]] || die "tax $t"
  echo "$x pai_fix-tax-$t-281ag" >> $O/jobs.txt; echo $ids $x; }
case ${1:-} in
dry)
  for s in 0 1; do echo "R6FIX s$s: $(argv fix $N R6FIX-L18-s$s $s | xargs) --save-at 2500,5000,7500"; echo "== R5LC s$s -> R6FIX s$s"
    diff <(argv lc $N R5LC-L18-s$s $s) <(argv fix $N R6FIX-L18-s$s $s); done
  echo "== identity: R5LC s0 (lc) vs R6FIXI (fixi), 60 steps"; diff <(argv lc 60 R6IDFXLC-L18-s0 0) <(argv fixi 60 R6IDFXI-L18-s0 0); true ;;
rows)
  [[ ! -f $O/ROWSDONE ]] || die "rows done ($O/ROWSDONE)"; h=$(git rev-parse --short HEAD); echo "$h rows" >> $O/code.txt
  j=$(sub fix-rows --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- bash $ME rowsjob) || exit 1
  echo "$(date '+%F %T') rows (code $h): $j" | tee -a $O/stages.txt ;;
rowsjob) $TR $PY rows > $O/rows.log 2>&1 || die "rows (log $O/rows.log)"; date '+%F %T' > $O/ROWSDONE ;;
chain)
  [[ -f $O/ROWSDONE && ! -f $O/STOP ]] || die "no ROWSDONE (or STOP)"; [[ -z $(grep -v fix-rows $O/jobs.txt 2>/dev/null) ]] || die "chain already submitted"
  for f in $O/w/R6FIX.npz $O/w/R6FIXI.npz $RES/w_R6FIX.json; do [[ -s $f ]] || die "no $f"; done
  for X in R6FIXSM-L18-s0 R6IDFXLC-L18-s0 R6IDFXI-L18-s0; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
  h=$(git rev-parse --short HEAD); echo "$h chain" >> $O/code.txt
  s=$(sub fix-smoke "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain) || exit 1
  i1=$(sub fix-ident-lc "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun lc) || exit 1
  i2=$(sub fix-ident-fixi "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun fixi) || exit 1
  i=$(sub fix-ident --after $i1,$i2 --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  l=$(sub fix-launch --after $s,$i --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 --env PRIO=$PRIO -- bash $ME launch) || exit 1
  echo "$(date '+%F %T') chain (code $h): smoke $s identity $i1 $i2 bitcmp $i launch $l" | tee -a $O/stages.txt ;;
smoketrain)
  X=R6FIXSM-L18-s0; run fix 60 $X 0
  echo "$(date '+%F %T') smoke: ${A[*]} --diag-every 20 --save-at 20,40" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --save-at 20,40 > $O/smoke.log 2>&1 || die "smoke run (log $O/smoke.log)"
  $TR $PY smoke --tag $X; r=$?
  for k in 20 40; do [[ -s $RUNS/R6FIXSMx$k-L18-s0/ckpt-final.pt ]] || { echo "$(date '+%F %T') ERROR smoke: no snapshot x$k" | tee -a $O/stages.txt; r=3; }; done
  rm -rf $RUNS/$X $RUNS/R6FIXSMx20-L18-s0 $RUNS/R6FIXSMx40-L18-s0; exit $r ;;
identrun)  # identrun lc|fixi: 60 steps eager (pool --env JEV_RM_EAGER=1)
  declare -A IX=([lc]=R6IDFXLC [fixi]=R6IDFXI); X=${IX[$2]}-L18-s0; run $2 60 $X 0
  echo "$(date '+%F %T') identity $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R6IDFXLC-L18-s0 --b R6IDFXI-L18-s0 --out round6_identity_fix \
      --note "R5LC-L18-s0 argv 60 steps eager: R5LC vs R6FIXI (R6FIX's draw code, nothing dropped)"; r=$?
  cp $DFR/round6_identity_fix.json $RES/identity.json || r=4; rm -rf $RUNS/R6IDFXLC-L18-s0 $RUNS/R6IDFXI-L18-s0; exit $r ;;
launch)  # smoke + identity passed -> the two trainings in parallel, each one's PAI 281 --after it, one report after both facts, then the ruler
  [[ ! -f $O/STOP ]] || exit 2
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke_R6FIX.json'))['pass_'] and json.load(open('$RES/identity.json'))['identical'] else 1)" \
    || stop "smoke / identity not passed"
  for s in 0 1; do for X in R6FIX-L18-s$s R6FIXx2500-L18-s$s R6FIXx5000-L18-s$s R6FIXx7500-L18-s$s; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
    [[ -s $F_/R5LC-L18-s$s-281ag.csv ]] || die "no stored R5LC s$s facts"; done
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt; x=()
  for s in 0 1; do t=$(sub fix-train-s$s "${FT[@]}" --timeout-h 4 -- bash $ME train $s) || exit 1
    y=($(read281 R6FIX-L18-s$s $t)) || exit 1; x+=(${y[-1]})
    echo "$(date '+%F %T') launch s$s (code $h): train $t PAI 281 ag chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]}" | tee -a $O/stages.txt; done
  r=$(sub fix-r281 --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281) || exit 1
  P0=$DATA_DIR/tmp/pcs_r5; PB=$DATA_DIR/tmp/pcs_r5l; PT=$DATA_DIR/tmp/pcs_fix; [[ ! -e $PT ]] || die "$PT exists"   # r5 report: R3MV-s0+AG must be a subject (hardcoded columns)
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$P0/site)
  c=$(PRIO=15 sub fix-pcs-collect --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only R6FIX-s0+AG,R6FIX-s1+AG --base $PB/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(PRIO=15 sub fix-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit \
      --rows $PT/rows.csv.gz --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  p=$(PRIO=15 sub fix-pcs-report --after "$(csv ${b[@]}),$r" --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME pcsrep) || exit 1
  echo "$(date '+%F %T') launch: report $r ruler collect $c boot ${b[*]} report+verdict $p" | tee -a $O/stages.txt ;;
train)
  run fix $N R6FIX-L18-s$2 $2; echo "$(date '+%F %T') train s$2: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
r281)
  rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R5LC-s0-ag=$F_/R5LC-L18-s0-281ag.csv R5LC-s1-ag=$F_/R5LC-L18-s1-281ag.csv
      FIX-s0-ag=$F_/R6FIX-L18-s0-281ag.csv FIX-s1-ag=$F_/R6FIX-L18-s1-281ag.csv)
  $TR $R2PY r281 --read "${rd[@]}" --pairs FIX-s0-ag:R5LC-s0-ag FIX-s1-ag:R5LC-s1-ag FIX-s0-ag:NTV0-ag FIX-s1-ag:NTV0-ag R5LC-s0-ag:NTV0-ag \
      R5LC-s1-ag:NTV0-ag FIX-s1-ag:FIX-s0-ag --out $RES/pai281_FIX_ag.md \
      --title "PAI 281, ACC + GAP: FIX (MV-LC without F1 / F2 rows) s0 / s1 vs stored R5LC s0 / s1 (MV-LC) and NTV0-L18-s0" || die "r281"
  $V $R5PY newbreaks --json $RES/pai281_FIX_ag.json --md $RES/pai281_FIX_ag.md --read "${rd[@]}" || die "newbreaks" ;;
pcsrep)
  PB=$DATA_DIR/tmp/pcs_r5l; PT=$DATA_DIR/tmp/pcs_fix; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  python3 - $PB/out $PT/out <<'PY' || die "bootstrap draws of the old reads differ from pcs_r5l's"
import csv, glob, sys                                            # same seeds, same pool: the old subjects' draws must repeat exactly
rd = lambda d: [r for f in sorted(glob.glob(d + "/boot_*.csv")) for r in csv.DictReader(open(f))]  # noqa: E731
a, b = rd(sys.argv[1]), rd(sys.argv[2])
d = max(abs(float(x[k]) - float(y[k])) for x, y in zip(a, b) for k in x if x[k] and y.get(k))
print("old draws repeat:", len(a), "draws,", len(a[0]), "subjects, max |diff|", d, "PCS")
assert len(a) == len(b) == 100 and d < 0.5, "boot mismatch"
PY
  sj="R2C-s0+AG=R2C-L18-s0-281ag.csv,R3MV-s0+AG=R3MV-L18-s0-281ag.csv,R5LC-s0+AG=R5LC-L18-s0-281ag.csv,R5LC-s1+AG=R5LC-L18-s1-281ag.csv,R6FIX-s0+AG=R6FIX-L18-s0-281ag.csv,R6FIX-s1+AG=R6FIX-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs R6FIX-s0+AG:R5LC-s0+AG,R6FIX-s1+AG:R5LC-s1+AG,R6FIX-s1+AG:R6FIX-s0+AG \
      > $PT/out/r5.log 2>&1 || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_fix.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_fix.json || die "copy ruler report"
  $V $R5PY pcsrow --pcs $RES/pcs_fix.json --json $RES/pai281_FIX_ag.json --md $RES/pai281_FIX_ag.md --subject R5LC-s0+AG R5LC-s1+AG R6FIX-s0+AG R6FIX-s1+AG \
      --report pcs_fix.md || die "pcsrow"
  rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R5LC-s0-ag=$F_/R5LC-L18-s0-281ag.csv R5LC-s1-ag=$F_/R5LC-L18-s1-281ag.csv
      FIX-s0-ag=$F_/R6FIX-L18-s0-281ag.csv FIX-s1-ag=$F_/R6FIX-L18-s1-281ag.csv)
  $V $PY verdict --json $RES/pai281_FIX_ag.json --md $RES/pai281_FIX_ag.md --pcs $RES/pcs_fix.json --read "${rd[@]}" || die "verdict"
  date '+%F %T' > $O/DONE ;;
wait)  # wait <FLAG>: one blocking waiter
  f=${2:-DONE}
  until [[ -f $O/$f || -f $O/STOP ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  cat $O/STOP 2>/dev/null; [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; tail -4 $O/stages.txt; true ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
