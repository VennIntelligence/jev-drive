#!/usr/bin/env bash
# ROUND5 v1 recipe (plans/2026-10-21-r5-prereg.md). Stage R: R5-R = R3MV-L18-s0's recorded command (draw R3MV: PAI human rows fed >= 2 m/s, three
# pools, no flattening; from NTV0-L18-s0, 9 500 steps, --save-at 2500,5000,7500) + --on-road-pai road_on1_JTPH.npz (rule d, W 1 = the nuPlan pool
# rows' road hinge weight: road_lam 10 x ho_w 3). Stage A (prereg addendum 1: no road hinge): R5-A = R3MV's argv + S3 outcome rows (pool on1_JTPHO = on1_JTPH + on1_PAIO, the S3 lane's
# zero-copy merge) as 9 extra rows a batch (--on-k 73, --batch 137, --yr-mass 0.095: 51 navtrain + 13 yaw-rate + 64 pool rows unchanged; draw R5A),
# seeds 0 and 1 from NTV0-L18-s0. Each stage: PAI-FIX native read (vl, "v t") -> report + gate (r5.py gate: round 3's lines + class guard) ->
# per passing tag PAI 281 ACC + GAP (PRIO 30) paired vs stored NTV0 / R2C + ACC + GAP; inertia cf + report beside it. Control R2C-L18-s0 (stored).
# Box, repo root; every job through the GPU pool at PRIO 20 (281: 30); ids in $O/<stage>/jobs.txt, stages in $O/<stage>/stages.txt.
#   r5.sh dry <R|A>         the training command(s) and the diff against R3MV's recorded argv
#   r5.sh smoke R           R5-R 60 steps (--diag-every 20 --save-at 20,40) -> r5.py smoke -> results/smoke_R.json; identity: R3MV's argv 60 steps
#                           eager vs the same + --on-road-pai ... --on-road-pai-w 0 -> df.py bitcmp -> results/identity_r5r.json
#   r5.sh smoke A           (S3 DONE) R5A draw file -> R5-A s0 60 steps -> r5.py smoke; identity: R3MV's argv vs the same on on1_JTPHO /
#                           R3MV extended with S3 rows at weight 0 (R5R0) 60 steps eager -> results/identity_r5a.json
#   r5.sh launch <R|A>      (smoke + identity passed) trainings -> PAI-FIX -> paifix (report, gate, 281 for passing tags) -> PAIDONE;
#                           trainings -> inertia cf -> inreport -> INDONE; DONE when both
#   r5.sh wait <R|A>        blocks until $O/<stage>/DONE or a failed job of this stage (one run_in_background waiter); swait: every job ended
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
R5=$DATA_DIR/runs/round5/r5; ME=experiments/round5/scripts/r5.sh; PY=experiments/round5/scripts/r5.py; R2PY=experiments/round2/scripts/r2.py
RES=experiments/round5/results; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; RPD=$DATA_DIR/runs/round5/road_pai; S3O=$DATA_DIR/runs/round5/s3
IN=experiments/defect_fix/scripts/inertia.py; LA=experiments/defect_fix/scripts/lon_anchor.py; DFR=experiments/defect_fix/results
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CTRL=R2C-L18-s0; MV=R3MV-L18-s0; RT=R5R-L18-s0; N=9500
declare -A TAGS=([R]="$RT" [A]="R5A-L18-s0 R5A-L18-s1") NAME=([R]=round5r [A]=round5a)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train); PRIO=${PRIO:-20}
ST=${2:-R}; [[ $ST == R || $ST == A ]] || { sed -n 2,19p "$0"; exit 2; }; O=$R5/$ST; mkdir -p $O/pool $R5/w $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-r5 --name "$name" --tries ${TRIES:-1} --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
argv() {  # argv <variant> <steps> <tag> [seed]: one argument a line. Variants: mv = R3MV's recorded argv (no --save-at); r = mv + the road hinge;
          # r0 = mv + the road hinge at W 0; ro = mv on on1_JTPHO / R3MV + S3 rows at weight 0 (R5R0), no road hinge; a = ro with the R5A draw, 9 extra rows
  local M; M=$(ls -d $DATA_DIR/runs/route_mem/train-$MV/*/ | tail -1)meta.json; [[ -f $M ]] || die "no $M"
  python3 - "$M" "$@" "$RPD" "$R5/w" <<'PY'
import json, sys
m, var, steps, tag = sys.argv[1:5]
seed, rpd, w5 = (sys.argv[5:-2] or [None])[0], sys.argv[-2], sys.argv[-1]
a = json.load(open(m))["argv"]
last = lambda k: len(a) - 1 - a[::-1].index(k)  # noqa: E731
assert a[a.index("--init") + 1] == "NTV0-L18-s0" and a[a.index("--tag") + 1] == "R3MV-L18-s0" and a[last("--on-wfile") + 1] == "runs/defect_fix/w/R3MV.npz"
assert a[last("--on") + 1] == "on1_JTPH" and a[last("--on-k") + 1] == "64" and a[last("--seed") + 1] == "0" and "--on-road-pai" not in a and "--batch" not in a
a[a.index("--tag") + 1] = tag
a[last("--steps") + 1] = steps
i = a.index("--save-at"); del a[i:i + 2]
if var in ("r", "r0"):
    a += ["--on-road-pai", f"{rpd}/road_on1_JTPH.npz"] + (["--on-road-pai-w", "0"] if var == "r0" else [])
if var in ("ro", "a"):                                                   # stage A (addendum 1): R3MV's argv, no road hinge
    a[last("--on") + 1] = "on1_JTPHO"
    a[last("--on-wfile") + 1] = f"{w5}/{'R5A' if var == 'a' else 'R5R0'}.npz"
if var == "a":
    a[last("--on-k") + 1] = "73"
    a[last("--yr-mass") + 1] = "0.095"
    a += ["--batch", "137"]
if seed is not None:
    a[last("--seed") + 1] = seed
print("\n".join(a))
PY
}
run() { local L; L=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$L"; }
read281() {  # read281 <tag> <prio>: PAI 281 ACC + GAP as rounds 2 / 3 (NTV0 + ACC + GAP's chunk lists, g10, FAST=1, CONC 4) -> "<chunk ids...> <facts id>"
  local T=$1 R=$R5/pai281/$1-ag ids t; mkdir -p $R5/pai281
  CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" \
    WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 ALPASIM_FAST=1 PRIO=$2 bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R.submit.txt \
    || die "pai submit $T"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$T: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$2 AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281ag $R | tail -1) && [[ -n $t ]] || die "tax $T"
  echo "$t pai_fix-tax-$T-281ag" >> $O/jobs.txt; echo $ids $t; }
poolwait() {  # the waiter's check: 0 while no job of this stage failed / was cancelled
  $V - $O/jobs.txt <<'PY'
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
open_ = [n for n, i in last.items() if jobs.get(i, {}).get("state", "inbox") not in ("done", "failed", "cancelled")]   # not yet listed = still in the inbox
sys.exit(1 if bad else (3 if open_ else 0))
PY
}
case ${1:-} in
dry)
  if [[ $ST == R ]]; then echo "R5-R: $(argv r $N $RT | xargs) --save-at 2500,5000,7500"; echo "== R3MV -> R5-R"; diff <(argv mv $N $MV) <(argv r $N $RT)
  else for s in 0 1; do echo "R5-A s$s: $(argv a $N R5A-L18-s$s $s | xargs) --save-at 2500,5000,7500"; echo "== R3MV -> R5-A s$s"
    diff <(argv mv $N $MV) <(argv a $N R5A-L18-s$s $s); done; fi; true ;;
smoke)
  h=$(git rev-parse --short HEAD); echo "$h smoke $ST" >> $O/code.txt
  if [[ $ST == R ]]; then
    for X in R5SMR-L18-s0 R5IDMV-L18-s0 R5IDW0-L18-s0; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
    s=$(sub r5-smoke-R "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain R) || exit 1
    a=$(sub r5-ident-mv "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun R mv) || exit 1
    b=$(sub r5-ident-w0 "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun R r0) || exit 1
  else
    [[ -f $S3O/DONE && ! -f $S3O/VERDICT ]] || die "S3 not done (or VERDICT)"
    for X in R5SMA-L18-s0 R5IDMV-L18-s0 R5IDRO-L18-s0; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
    p=$(sub r5-prepA --vram 0.1 --cpu 16 --ram 64 --timeout-h 1 -- bash $ME prepA A) || exit 1
    s=$(sub r5-smoke-A --after $p "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain A) || exit 1
    a=$(sub r5-ident-mv --after $p "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun A mv) || exit 1
    b=$(sub r5-ident-ro --after $p "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun A ro) || exit 1
  fi
  c=$(sub r5-ident-$ST --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident $ST) || exit 1
  echo "$(date '+%F %T') smoke $ST (code $h): smoke $s identity $a $b bitcmp $c" | tee -a $O/stages.txt ;;
prepA)  # R5R0 (R3MV + S3 rows at weight 0) and R5A draw files
  [[ -s $DATA_DIR/runs/op_parity/cache/on1_JTPHO/tab.npz ]] || die "no on1_JTPHO"
  $TR $PY weights --k-s3 0 --out R5R0 > $O/w_R5R0.log 2>&1 || die "R5R0 weights"
  $TR $PY weights --k-s3 9 --out R5A > $O/w_R5A.log 2>&1 || die "R5A weights" ;;
smoketrain)
  X=R5SM$ST-L18-s0; if [[ $ST == R ]]; then run r 60 $X; else run a 60 $X 0; fi
  echo "$(date '+%F %T') smoke $ST: ${A[*]} --diag-every 20 --save-at 20,40" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --save-at 20,40 > $O/smoke.log 2>&1 || die "smoke run $ST (log $O/smoke.log)"
  if [[ $ST == R ]]; then $TR $PY smoke --tag $X --arm R; else $TR $PY smoke --tag $X --arm A --wfile R5A --k-s3 9; fi; r=$?
  for k in 20 40; do [[ -s $RUNS/R5SM${ST}x$k-L18-s0/ckpt-final.pt ]] || { echo "$(date '+%F %T') ERROR smoke $ST: no snapshot x$k" | tee -a $O/stages.txt; r=3; }; done
  rm -rf $RUNS/$X $RUNS/R5SM${ST}x20-L18-s0 $RUNS/R5SM${ST}x40-L18-s0; exit $r ;;
identrun)  # identrun <stage> <variant>: 60 steps eager (pool --env JEV_RM_EAGER=1), this code
  declare -A IX=([mv]=R5IDMV [r0]=R5IDW0 [r]=R5IDR [ro]=R5IDRO); X=${IX[$3]}-L18-s0; run $3 60 $X
  echo "$(date '+%F %T') identity $3: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$3.log 2>&1 || die "identity run $3 (log $O/ident_$3.log)" ;;
ident)
  if [[ $ST == R ]]; then a=R5IDMV-L18-s0 b=R5IDW0-L18-s0 n="R3MV-L18-s0 argv 60 steps eager: without vs with --on-road-pai road_on1_JTPH.npz --on-road-pai-w 0"
  else a=R5IDMV-L18-s0 b=R5IDRO-L18-s0 n="R3MV-L18-s0 argv 60 steps eager: on1_JTPH / R3MV vs on1_JTPHO / R5R0 (S3 rows weight 0)"; fi
  $TR experiments/defect_fix/scripts/df.py bitcmp --a $a --b $b --out round5_identity_$ST --note "$n"; r=$?
  cp $DFR/round5_identity_$ST.json $RES/identity_r5${ST,,}.json || r=4
  rm -rf $RUNS/$a $RUNS/$b; exit $r ;;
launch)
  for T in ${TAGS[$ST]}; do for X in $T ${T/-L18-/x2500-L18-} ${T/-L18-/x5000-L18-} ${T/-L18-/x7500-L18-}; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done; done
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke_$([[ $ST == R ]] && echo R || echo R5A).json'))['pass_'] and json.load(open('$RES/identity_r5${ST,,}.json'))['identical'] else 1)" \
    || die "smoke / identity not passed"
  for f in $F_/$CTRL-vl.csv $F_/$MV-vl.csv $F_/NTV0-L18-s0-281ag.csv $F_/$CTRL-281ag.csv; do [[ -s $f ]] || die "no stored $f"; done
  rm -f $O/DONE $O/PAIDONE $O/INDONE; h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt
  t_=(); x=()
  for T in ${TAGS[$ST]}; do t_+=($(sub r5-train-$T "${FT[@]}" --timeout-h 4 -- bash $ME train $ST $T)) || exit 1
    o=$(PRIO=$PRIO AFTER=${t_[-1]} bash experiments/pai_fix/scripts/pai_fix.sh read $T vl "v t" | tail -1) || die "pai_fix read $T"
    x+=(${o%% *}); echo "${o%% *} pai_fix-tax-$T-vl" >> $O/jobs.txt
    awk '{print $1, $2}' $DATA_DIR/runs/pai_fix/reads/$T-vl/jobs.txt >> $O/jobs.txt; done
  p=$(sub r5-paifix-$ST --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME paifix $ST) || exit 1
  f=$(sub r5-cf-$ST --after $(csv ${t_[@]}) --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME cf $ST) || exit 1
  r=$(sub r5-inreport-$ST --after $f --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME inreport $ST) || exit 1
  echo "$(date '+%F %T') launch $ST (code $h): train ${t_[*]} PAI-FIX ${x[*]} report + gate $p inertia $f $r" | tee -a $O/stages.txt ;;
train)
  T=$3; if [[ $ST == R ]]; then run r $N $T; else run a $N $T ${T##*-s}; fi
  echo "$(date '+%F %T') train $T: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt; exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
paifix)  # reports vs CTRL and vs the stage's reference (R: R3MV, A: R5-R); the gate; every passing tag's PAI 281 ACC + GAP (PRIO 30) -> r281 -> PAIDONE
  rd=(NTV0-L18-s0-vl=$F_/NTV0-L18-s0-vl.csv $CTRL-vl=$F_/$CTRL-vl.csv R2J-L18-s0-vl=$F_/R2J-L18-s0-vl.csv $MV-vl=$F_/$MV-vl.csv)
  [[ $ST == A ]] && rd+=($RT-vl=$F_/$RT-vl.csv); for T in ${TAGS[$ST]}; do rd+=($T-vl=$F_/$T-vl.csv); done
  REF=$MV; s=${ST,,}                                              # both stages pair against R3MV (addendum 1)
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $CTRL-vl --out $RES/paifix_r5${s}_vl_vs_ctrl.md --scores $RES/paifix_r5${s}_vl.csv > /dev/null \
    || die "pai_fix report vs CTRL"
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $REF-vl --out $RES/paifix_r5${s}_vl_vs_${REF%%-*}.md > /dev/null || die "pai_fix report vs $REF"
  ps=$($V $PY gate --arms ${TAGS[$ST]} $REF --ctrl $CTRL --stem gate_r5$s) || die "gate"
  ps=$(tr ' ' '\n' <<< "$ps" | grep -v "^$REF\$" | xargs)                    # the reference is listed for comparison only
  echo "$(date '+%F %T') gate: passing [${ps}]" | tee -a $O/stages.txt
  [[ -z $ps ]] && { date '+%F %T' > $O/PAIDONE; bash $ME fin $ST; exit 0; }
  r=(); for T in $ps; do y=($(read281 $T 30)) || exit 1
    r+=($(PRIO=30 sub r5-r281-$T --after ${y[-1]} --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281 $ST $T)) || exit 1
    echo "$(date '+%F %T') PAI 281 ag $T: chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]} report ${r[-1]}" | tee -a $O/stages.txt; done
  PRIO=30 sub r5-paidone-$ST --after $(csv ${r[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/PAIDONE && bash $ME fin $ST" > /dev/null ;;
r281)
  T=$3; rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R2C-ag=$F_/$CTRL-281ag.csv R2J-ag=$F_/R2J-L18-s0-281ag.csv); pr=($T-ag:NTV0-ag $T-ag:R2C-ag $T-ag:R2J-ag)
  $TR $R2PY r281 --read "${rd[@]}" $T-ag=$F_/$T-281ag.csv --pairs "${pr[@]}" --out $RES/pai281_${T%-L18-*}${T##*-L18}_ag.md \
      --title "PAI 281, ACC + GAP: $T vs stored NTV0-L18-s0 and R2C-L18-s0 (and R2J)" || die "r281 $T" ;;
cf)  $TR $IN cf --tags NTV0-L18-s0,$CTRL,$MV,$([[ $ST == A ]] && echo $RT,)$(csv ${TAGS[$ST]}),RSG-L18-s0 --gate-tag RSG-L18-s0 --name ${NAME[$ST]} ;;
inreport)
  n=${NAME[$ST]}; $TR $IN report --name $n > $O/inertia_report.log 2>&1 || die "inertia report"
  cp $DFR/inertia/report_$n.md $DFR/inertia/report_$n.json $RES/ || die "no inertia report"
  $TR $LA inertia --name $n --ctl $CTRL --arms ${TAGS[$ST]} $MV $([[ $ST == A ]] && echo $RT) --out-stem ${n}_inertia || die "inertia read-out"
  cp $DFR/lon_anchor/${n}_inertia.md $RES/inertia_r5${ST,,}.md && cp $DFR/lon_anchor/${n}_inertia.json $RES/inertia_r5${ST,,}.json || die "copy inertia"
  date '+%F %T' > $O/INDONE; bash $ME fin $ST ;;
fin) [[ -f $O/PAIDONE && -f $O/INDONE ]] && date '+%F %T' > $O/DONE; true ;;
wait)   # returns at $O/DONE or when a job of this stage failed / was cancelled
  until [[ -f $O/DONE ]]; do poolwait; rc=$?; (( rc != 3 )) && break; sleep 300; done   # failed (1) or every job ended without DONE (0)
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -3 $O/stages.txt; true ;;
swait)  # returns when every job of this stage has ended (the smoke stage's waiter)
  until poolwait; rc=$?; (( rc != 3 )); do sleep 120; done; tail -4 $O/stages.txt; true ;;
*) sed -n 2,19p "$0"; exit 2 ;;
esac
