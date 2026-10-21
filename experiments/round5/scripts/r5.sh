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
# Stage B (prereg addendum 2, s3-rows plan addendum 1): R5-B = R5-A's command on on1_JTPHOR (S3 rows relabelled with the winner's realised 4 s
# future; draw R5B). s0 first (stage B); only if s0 passes the four lines: PAI 281 for s0 and stage B1 (s1: train, PAI-FIX, gate, 281 if it passes).
#   r5.sh chainB B          one submission: relabel (s3.py relabel, L0) -> merge on1_JTPHOR (s3.sh merge, mfix) -> rcheck -> prepB (R5B0, R5B) ->
#                           smoke B + identity (R3MV argv: on1_JTPH / R3MV vs on1_JTPHOR / R5B0 = L2) -> launch B
# Stage N (prereg addendum 3): R5-N = R3MV's argv with draw R5N (every PAI human row whose state has a lead vehicle -> 0, r5.py LEAD rule).
#   r5.sh leadrows N        (CPU, offline, before the prereg) r5.py leadrows -> R5N / R5NI draw files + results/lead_rows.{md,json}
#   r5.sh chainN N          one submission: smoke N + identity (R3MV argv: R3MV vs R5NI, bit-equal file via the new code) -> launch N;
#                           s0 passing the four lines -> PAI 281 for s0 and stage N1 (s1: train, PAI-FIX, gate, 281 if it passes)
# Stage G (prereg addendum 4): R5-G = R3MV's argv with draw R5G (PAI GO-NOW rows kept, navtrain launch rows x 3 as R3MVL); LAUNCH-37 exam.
#   r5.sh gonow G           (CPU, offline, before the prereg) r5.py gonow -> w/R5G.npz + results/{w_R5G,gonow_rows}.{json,md}
#   r5.sh chainG G          one submission: LAUNCH-37 reads of R3MV / R3MVL / R5N (exist) + smoke G -> launchG: train R5G -> PAI-FIX (vl) ->
#                           report + gate (no 281 here); LAUNCH-37 read of R5G -> l37fin: results/launch37.md, R5G vs R5N; R5-G passing the
#                           gate and launching >= 30 / 37 -> its PAI 281 ACC + GAP (PRIO 30) -> r281 -> DONE
# Stage G2 (addendum 4b): R5-G2 = R5-G with the GO-NOW rows (ph, ph15; ph05's dropped) at 4 rows a batch (draw R5G2).
#   r5.sh gonow G2          (CPU) r5.py gonow --go-batch 4 --go-drop ph05 --out R5G2;  r5.sh chainG2 G2: smoke G2 -> launchG G2 (as G; its
#                           l37fin waits for stage G's gate and LAUNCH-37, writes results/launch37_g2.md with R5-G2 and R5-G vs R5-N)
# Stage L (addendum 5): R5-LC (MV-LC) = R3MV's argv with draw R5LC (PAI human rows whose 4 s label is a lane change off the route lane -> 0),
# seeds 0 / 1, plus R3MV-L18-s1 (R3MV's argv, seed 1: seed noise); exam = PAI 281 ACC + GAP directly (read281), no PAI-FIX, no inertia.
#   r5.sh lcrows L          (CPU, offline, before the prereg) r5.py lcrows -> w/R5LC.npz, w/R5LCI.npz + results/{lc_rows,w_R5LC,w_R5LCI}.{md,json}
#   r5.sh chainL L          one submission: smoke L + identity (R3MV argv: R3MV vs R5LCI) -> launchL: the three trainings in parallel, each one's
#                           281 read --after it -> r281L (results/pai281_R5LC_ag.md + lcverdict) -> DONE
#   r5.sh pcsL L <after id>  (addendum 5, secondary) the local PAI PCS ruler (d392 pool / anchors) on the three new reads, CPU pool jobs PRIO 15:
#                           collect (new rows onto tmp/pcs_r5's) -> 20 bootstrap shards x 5 (d429's fits reused: the new reads are out of pool)
#                           -> pcsrep: pai_pcs_local.py r5 -> results/pcs_r5l.{md,json} + r5.py lcpcs -> PCSDONE
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
R5=$DATA_DIR/runs/round5/r5; ME=experiments/round5/scripts/r5.sh; PY=experiments/round5/scripts/r5.py; R2PY=experiments/round2/scripts/r2.py
RES=experiments/round5/results; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; RPD=$DATA_DIR/runs/round5/road_pai; S3O=$DATA_DIR/runs/round5/s3
IN=experiments/defect_fix/scripts/inertia.py; LA=experiments/defect_fix/scripts/lon_anchor.py; DFR=experiments/defect_fix/results
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; CTRL=R2C-L18-s0; MV=R3MV-L18-s0; RT=R5R-L18-s0; N=9500
declare -A TAGS=([R]="$RT" [A]="R5A-L18-s0 R5A-L18-s1" [B]="R5B-L18-s0" [B1]="R5B-L18-s1" [N]="R5N-L18-s0" [N1]="R5N-L18-s1" [G]="R5G-L18-s0" [G2]="R5G2-L18-s0" [L]="R5LC-L18-s0 R5LC-L18-s1 R3MV-L18-s1")
declare -A NAME=([R]=round5r [A]=round5a [B]=round5b [B1]=round5b1 [N]=round5n [N1]=round5n1 [G]=round5g [G2]=round5g2 [L]=round5l)
declare -A SMK=([R]=R [A]=R5A [B]=R5B [B1]=R5B [N]=R5N [N1]=R5N [G]=R5G [G2]=R5G2 [L]=R5LC) IDN=([R]=r [A]=a [B]=b [B1]=b [N]=n [N1]=n [G]=g [G2]=g2 [L]=lc); A0=R5A-L18-s0; B0=R5B-L18-s0; N0=R5N-L18-s0
MVL=R3MVL-L18-s0; GT=R5G-L18-s0; MV1=R3MV-L18-s1; L37=experiments/round5/results/launch37.txt
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train); PRIO=${PRIO:-20}
ST=${2:-R}; [[ -n ${TAGS[$ST]:-} ]] || { sed -n 2,40p "$0"; exit 2; }; O=$R5/$ST; mkdir -p $O/pool $R5/w $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-r5 --name "$name" --tries ${TRIES:-1} --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
argv() {  # argv <variant> <steps> <tag> [seed]: one argument a line. Variants: mv = R3MV's recorded argv (no --save-at); r = mv + the road hinge;
          # r0 = mv + the road hinge at W 0; ro = mv on on1_JTPHO / R3MV + S3 rows at weight 0 (R5R0), no road hinge; a = ro with the R5A draw, 9 extra rows;
          # bo / b = ro / a on on1_JTPHOR (relabelled S3 rows) with R5B0 / R5B; n / ni = mv with the R5N / R5NI draw (stage N); g = mv with R5G;
          # lc / lci = mv with the R5LC / R5LCI draw (stage L)
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
if var in ("ro", "a", "bo", "b"):                                       # stages A / B (addenda 1 / 2): R3MV's argv, no road hinge
    a[last("--on") + 1] = "on1_JTPHOR" if var in ("bo", "b") else "on1_JTPHO"
    a[last("--on-wfile") + 1] = f"{w5}/{dict(ro='R5R0', a='R5A', bo='R5B0', b='R5B')[var]}.npz"
if var in ("n", "ni", "g", "g2", "lc", "lci"):                               # stages N / G / L (addenda 3-5): only the draw file changes
    a[last("--on-wfile") + 1] = f"{w5}/{dict(n='R5N', ni='R5NI', g='R5G', g2='R5G2', lc='R5LC', lci='R5LCI')[var]}.npz"
if var in ("a", "b"):
    a[last("--on-k") + 1] = "73"
    a[last("--yr-mass") + 1] = "0.095"
    a += ["--batch", "137"]
if seed is not None:
    a[last("--seed") + 1] = seed
print("\n".join(a))
PY
}
run() { local L; L=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$L"; }
read281() {  # read281 <tag> <prio> [after ids]: PAI 281 ACC + GAP as rounds 2 / 3 (NTV0 + ACC + GAP's chunk lists, g10, FAST=1, CONC 4) -> "<chunk ids...> <facts id>"
  local T=$1 R=$R5/pai281/$1-ag ids t; mkdir -p $R5/pai281
  AFTER=${3:-} CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" \
    WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 ALPASIM_FAST=1 PRIO=$2 bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R.submit.txt \
    || die "pai submit $T"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$T: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$2 AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281ag $R | tail -1) && [[ -n $t ]] || die "tax $T"
  echo "$t pai_fix-tax-$T-281ag" >> $O/jobs.txt; echo $ids $t; }
l37read() {  # l37read <tag> [after ids]: LAUNCH-37 under read281's serve (g10, FAST=1, CONC 4), 4 chunk lists of <= 10 scenes -> "<chunk ids...> <facts id>"
  local T=$1 R=$R5/l37/$1-ag ids t; mkdir -p $R5/l37
  CHUNK=10 SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 ALPASIM_FAST=1 \
    PRIO=$PRIO AFTER=${2:-} bash experiments/alpasim/scripts/pai_box.sh submit $R $T $L37 > $R.submit.txt || die "l37 submit $T"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 4 ]] || die "$T: not 4 LAUNCH-37 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-l37ag $R | tail -1) && [[ -n $t ]] || die "tax l37 $T"
  echo "$t pai_fix-tax-$T-l37ag" >> $O/jobs.txt; echo $ids $t; }
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
  elif [[ $ST == A ]]; then for s in 0 1; do echo "R5-A s$s: $(argv a $N R5A-L18-s$s $s | xargs) --save-at 2500,5000,7500"; echo "== R3MV -> R5-A s$s"
    diff <(argv mv $N $MV) <(argv a $N R5A-L18-s$s $s); done
  elif [[ $ST == G ]]; then echo "R5-G: $(argv g $N $GT 0 | xargs) --save-at 2500,5000,7500"; echo "== R3MV -> R5-G"; diff <(argv mv $N $MV) <(argv g $N $GT 0)
  elif [[ $ST == L ]]; then for s in 0 1; do echo "R5-LC s$s: $(argv lc $N R5LC-L18-s$s $s | xargs) --save-at 2500,5000,7500"; echo "== R3MV -> R5-LC s$s"
    diff <(argv mv $N $MV) <(argv lc $N R5LC-L18-s$s $s); done; echo "R3MV s1: $(argv mv $N $MV1 1 | xargs) --save-at 2500,5000,7500"; echo "== R3MV -> R3MV s1"
    diff <(argv mv $N $MV) <(argv mv $N $MV1 1); echo "== R3MV -> identity (lci)"; diff <(argv mv 60 R5IDMV-L18-s0) <(argv lci 60 R5IDLCI-L18-s0)
  elif [[ $ST == N* ]]; then for s in 0 1; do echo "R5-N s$s: $(argv n $N R5N-L18-s$s $s | xargs) --save-at 2500,5000,7500"; echo "== R3MV -> R5-N s$s"
    diff <(argv mv $N $MV) <(argv n $N R5N-L18-s$s $s); done; echo "== R3MV -> identity (ni)"; diff <(argv mv 60 R5IDMV-L18-s0) <(argv ni 60 R5IDNI-L18-s0)
  else for s in 0 1; do echo "== R5-A s$s -> R5-B s$s"; diff <(argv a $N R5A-L18-s$s $s) <(argv b $N R5B-L18-s$s $s); done
    echo "== R3MV -> L2 identity (bo)"; diff <(argv mv 60 R5IDMV-L18-s0) <(argv bo 60 R5IDBO-L18-s0); fi; true ;;
smoke)
  h=$(git rev-parse --short HEAD); echo "$h smoke $ST" >> $O/code.txt
  if [[ $ST == R ]]; then
    for X in R5SMR-L18-s0 R5IDMV-L18-s0 R5IDW0-L18-s0; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
    s=$(sub r5-smoke-R "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain R) || exit 1
    a=$(sub r5-ident-mv "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun R mv) || exit 1
    b=$(sub r5-ident-w0 "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun R r0) || exit 1
  elif [[ $ST == A ]]; then
    [[ -f $S3O/DONE && ! -f $S3O/VERDICT ]] || die "S3 not done (or VERDICT)"
    for X in R5SMA-L18-s0 R5IDMV-L18-s0 R5IDRO-L18-s0; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
    p=$(sub r5-prepA --vram 0.1 --cpu 16 --ram 64 --timeout-h 1 -- bash $ME prepA A) || exit 1
    s=$(sub r5-smoke-A --after $p "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain A) || exit 1
    a=$(sub r5-ident-mv --after $p "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun A mv) || exit 1
    b=$(sub r5-ident-ro --after $p "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun A ro) || exit 1
  else die "stage $ST: use chainB"; fi
  c=$(sub r5-ident-$ST --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident $ST) || exit 1
  echo "$(date '+%F %T') smoke $ST (code $h): smoke $s identity $a $b bitcmp $c" | tee -a $O/stages.txt ;;
chainB)  # stage B in one submission (see the header); the gates inside the jobs stop the chain
  [[ $ST == B ]] || die "chainB is stage B"; [[ ! -e $DATA_DIR/runs/op_parity/cache/on1_PAIOR && ! -s $O/jobs.txt ]] || die "stage B already submitted"
  for X in R5SMB-L18-s0 R5IDMV-L18-s0 R5IDBO-L18-s0; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  h=$(git rev-parse --short HEAD); echo "$h chainB" >> $O/code.txt
  r=$(sub r5-relabel --vram 0.1 --cpu 8 --ram 32 --timeout-h 1 -- bash $ME relabel B) || exit 1
  m=$(sub r5-merge-B --after $r --vram 0.1 --cpu 4 --ram 120 --timeout-h 3 -- env S3_NP=PAIOR S3_NM=JTPHOR bash experiments/round5/scripts/s3.sh merge) || exit 1
  c=$(sub r5-rcheck --after $m --vram 0.1 --cpu 4 --ram 120 --timeout-h 2 -- bash $ME rcheck B) || exit 1
  p=$(sub r5-prepB --after $c --vram 0.1 --cpu 16 --ram 64 --timeout-h 1 -- bash $ME prepB B) || exit 1
  s=$(sub r5-smoke-B --after $p "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain B) || exit 1
  a=$(sub r5-ident-mv-B --after $p "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun B mv) || exit 1
  b=$(sub r5-ident-bo --after $p "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun B bo) || exit 1
  i=$(sub r5-ident-B --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident B) || exit 1
  l=$(sub r5-launch-B --after $s,$i --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME launch B) || exit 1
  echo "$(date '+%F %T') chainB (code $h): relabel $r merge $m rcheck $c prepB $p smoke $s identity $a $b bitcmp $i launch $l" | tee -a $O/stages.txt ;;
chainN)  # stage N in one submission: smoke + identity -> launch (the draw files come from leadrows, run before the prereg addendum)
  [[ $ST == N ]] || die "chainN is stage N"; [[ ! -s $O/jobs.txt || -z $(grep -v leadrows $O/jobs.txt) ]] || die "stage N already submitted"
  for f in $R5/w/R5N.npz $R5/w/R5NI.npz $RES/w_R5N.json; do [[ -s $f ]] || die "no $f (run leadrows)"; done
  for X in R5SMN-L18-s0 R5IDMV-L18-s0 R5IDNI-L18-s0; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  h=$(git rev-parse --short HEAD); echo "$h chainN" >> $O/code.txt
  s=$(sub r5-smoke-N "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain N) || exit 1
  a=$(sub r5-ident-mv-N "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun N mv) || exit 1
  b=$(sub r5-ident-ni "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun N ni) || exit 1
  i=$(sub r5-ident-N --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident N) || exit 1
  l=$(sub r5-launch-N --after $s,$i --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME launch N) || exit 1
  echo "$(date '+%F %T') chainN (code $h): smoke $s identity $a $b bitcmp $i launch $l" | tee -a $O/stages.txt ;;
gonow)
  [[ $ST == G* ]] || die "gonow is stage G / G2"
  if [[ ${3:-} != run ]]; then h=$(git rev-parse --short HEAD); echo "$h gonow" >> $O/code.txt
    j=$(sub r5-gonow-$ST --vram 0.1 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME gonow $ST run) || exit 1
    echo "$(date '+%F %T') gonow $ST (code $h): $j" | tee -a $O/stages.txt; exit 0; fi
  if [[ $ST == G ]]; then $TR $PY gonow > $O/gonow.log 2>&1; else $TR $PY gonow --go-batch 4 --go-drop ph05 --out R5G2 > $O/gonow.log 2>&1; fi \
    || die "gonow (log $O/gonow.log)"; date '+%F %T' > $O/GODONE ;;
chainG2)  # stage G2: smoke -> launchG G2 (LAUNCH-37 references come from stage G)
  [[ $ST == G2 ]] || die "chainG2 is stage G2"; [[ ! -s $O/jobs.txt || -z $(grep -v gonow $O/jobs.txt) ]] || die "stage G2 already submitted"
  for f in $R5/w/R5G2.npz $RES/w_R5G2.json $R5/G/l37_ref_tax.txt; do [[ -s $f ]] || die "no $f"; done; [[ ! -e $RUNS/R5SMG2-L18-s0 ]] || die "R5SMG2-L18-s0 exists"
  h=$(git rev-parse --short HEAD); echo "$h chainG2" >> $O/code.txt
  s=$(sub r5-smoke-G2 "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain G2) || exit 1
  l=$(sub r5-launch-G2 --after $s --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME launchG G2) || exit 1
  echo "$(date '+%F %T') chainG2 (code $h): smoke $s launch $l" | tee -a $O/stages.txt ;;
chainG)  # stage G in one submission (see the header)
  [[ $ST == G ]] || die "chainG is stage G"; [[ ! -s $O/jobs.txt || -z $(grep -v gonow $O/jobs.txt) ]] || die "stage G already submitted"
  for f in $R5/w/R5G.npz $RES/w_R5G.json $L37; do [[ -s $f ]] || die "no $f (run gonow)"; done
  [[ ! -e $RUNS/R5SMG-L18-s0 ]] || die "R5SMG-L18-s0 exists"
  for T in $MV $MVL $N0; do [[ -s $RUNS/$T/ckpt-final.pt ]] || die "no $T"; [[ ! -e $R5/l37/$T-ag ]] || die "LAUNCH-37 of $T exists"; done
  h=$(git rev-parse --short HEAD); echo "$h chainG" >> $O/code.txt
  x=(); for T in $MV $MVL $N0; do y=($(l37read $T)) || exit 1; x+=(${y[-1]}); echo "$(date '+%F %T') LAUNCH-37 $T: chunks ${y[*]:0:4} facts ${y[-1]}" | tee -a $O/stages.txt; done
  echo "${x[*]}" > $O/l37_ref_tax.txt
  s=$(sub r5-smoke-G "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain G) || exit 1
  l=$(sub r5-launch-G --after $s --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME launchG G) || exit 1
  echo "$(date '+%F %T') chainG (code $h): LAUNCH-37 refs facts ${x[*]} smoke $s launch $l" | tee -a $O/stages.txt ;;
launchG)
  T=${TAGS[$ST]}; [[ $ST == G* ]] || die "launchG is stage G / G2"
  for X in $T ${T/-L18-/x2500-L18-} ${T/-L18-/x5000-L18-} ${T/-L18-/x7500-L18-}; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke_${SMK[$ST]}.json'))['pass_'] else 1)" || die "smoke not passed"
  for f in $F_/$CTRL-vl.csv $F_/$MV-vl.csv $F_/$MVL-vl.csv $F_/$N0-vl.csv $F_/NTV0-L18-s0-281ag.csv $F_/$CTRL-281ag.csv $F_/R2J-L18-s0-281ag.csv; do
    [[ -s $f ]] || die "no stored $f"; done
  [[ -s $R5/G/l37_ref_tax.txt ]] || die "no LAUNCH-37 reference reads"
  ga=(); if [[ $ST == G2 ]]; then                                  # stage G's PAI-FIX facts and LAUNCH-37 read feed G2's references
    for n in pai_fix-tax-$GT-vl pai_fix-tax-$GT-l37ag; do i=$(awk -v n=$n '$2 == n {print $1}' $R5/G/jobs.txt | tail -1); [[ -n $i ]] || die "stage G: no $n"; ga+=($i); done; fi
  rm -f $O/DONE $O/PAIDONE; h=$(git rev-parse --short HEAD); echo "$h launchG" >> $O/code.txt
  t=$(sub r5-train-$T "${FT[@]}" --timeout-h 4 -- bash $ME train $ST $T) || exit 1
  o=$(PRIO=$PRIO AFTER=$t bash experiments/pai_fix/scripts/pai_fix.sh read $T vl "v t" | tail -1) || die "pai_fix read $T"
  x=${o%% *}; echo "$x pai_fix-tax-$T-vl" >> $O/jobs.txt; awk '{print $1, $2}' $DATA_DIR/runs/pai_fix/reads/$T-vl/jobs.txt >> $O/jobs.txt
  p=$(sub r5-paifix-$ST --after $(csv $x ${ga[@]:0:1}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME paifix $ST) || exit 1
  y=($(l37read $T $t)) || exit 1
  f=$(sub r5-l37-$ST --after $(csv $p ${y[-1]} $(cat $R5/G/l37_ref_tax.txt) ${ga[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME l37fin $ST) || exit 1
  echo "$(date '+%F %T') launchG $ST (code $h): train $t PAI-FIX $x report + gate $p LAUNCH-37 chunks ${y[*]:0:4} facts ${y[-1]} l37fin $f" | tee -a $O/stages.txt ;;
l37fin)  # LAUNCH-37 report (registered comparisons R5G vs R5N, G2: also R5G2 vs R5N); the stage's arm passing its gate and launching >= 30 / 37 -> its PAI 281
  T=${TAGS[$ST]}; k=${T%-L18-*}
  rd=(NTV0=$F_/NTV0-L18-s0-281ag.csv R2C=$F_/$CTRL-281ag.csv R2J=$F_/R2J-L18-s0-281ag.csv R3MV=$F_/$MV-l37ag.csv R3MVL=$F_/$MVL-l37ag.csv
      R5N=$F_/$N0-l37ag.csv R5G=$F_/$GT-l37ag.csv)
  if [[ $ST == G ]]; then
    $V $PY l37 --read "${rd[@]}" --pair R5G:R5N R5G:R3MVL R5G:R3MV R5N:R3MV R3MVL:R3MV R5G:NTV0 R5N:NTV0 R3MV:NTV0 R5G:R2C \
        --target $GT:$N0 $GT:$MVL $GT:$MV $N0:$MV > $O/l37.log 2>&1 || die "l37 report (log $O/l37.log)"; js=$RES/launch37.json
  else
    $V $PY l37 --read "${rd[@]}" R5G2=$F_/$T-l37ag.csv --pair R5G2:R5N R5G:R5N R5G2:R5G R5G2:R3MVL R5G:R3MVL R5N:R3MV R5G2:NTV0 R5G2:R2C \
        --target $T:$N0 $GT:$N0 $T:$GT $T:$MVL --sep R5G2:R5N=$T:$N0 R5G:R5N=$GT:$N0 --stem launch37_g2 > $O/l37.log 2>&1 || die "l37 report (log $O/l37.log)"
    js=$RES/launch37_g2.json; fi
  go=$(python3 -c "import json; g=json.load(open('$RES/gate_r5${ST,,}.json'))['arms']['$T']['pass_']; n=json.load(open('$js'))['reads']['$k']['launched']; print(int(bool(g) and n >= 30))") \
    || die "decision"
  echo "$(date '+%F %T') l37fin $ST: $k gate + LAUNCH-37 >= 30 -> PAI 281: $go" | tee -a $O/stages.txt
  if [[ $go == 1 ]]; then y=($(read281 $T 30)) || exit 1
    r=$(PRIO=30 sub r5-r281-$T --after ${y[-1]} --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281 $ST $T) || exit 1
    PRIO=30 sub r5-done-$ST --after $r --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/DONE" > /dev/null
    echo "$(date '+%F %T') PAI 281 ag $T: chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]} report $r" | tee -a $O/stages.txt
  else date '+%F %T' > $O/DONE; fi ;;
lcrows)
  [[ $ST == L ]] || die "lcrows is stage L"
  if [[ ${3:-} != run ]]; then h=$(git rev-parse --short HEAD); echo "$h lcrows" >> $O/code.txt
    j=$(sub r5-lcrows --vram 0.1 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME lcrows L run) || exit 1
    echo "$(date '+%F %T') lcrows (code $h): $j" | tee -a $O/stages.txt; exit 0; fi
  $TR $PY lcrows > $O/lcrows.log 2>&1 || die "lcrows (log $O/lcrows.log)"; date '+%F %T' > $O/LCDONE ;;
chainL)  # stage L in one submission: smoke + identity -> launchL (the draw files come from lcrows, run before the prereg addendum)
  [[ $ST == L ]] || die "chainL is stage L"; [[ ! -s $O/jobs.txt || -z $(grep -v lcrows $O/jobs.txt) ]] || die "stage L already submitted"
  for f in $R5/w/R5LC.npz $R5/w/R5LCI.npz $RES/w_R5LC.json; do [[ -s $f ]] || die "no $f (run lcrows)"; done
  for X in R5SML-L18-s0 R5IDMV-L18-s0 R5IDLCI-L18-s0; do [[ ! -e $RUNS/$X ]] || die "$X exists"; done
  h=$(git rev-parse --short HEAD); echo "$h chainL" >> $O/code.txt
  s=$(sub r5-smoke-L "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain L) || exit 1
  a=$(sub r5-ident-mv-L "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun L mv) || exit 1
  b=$(sub r5-ident-lci "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun L lci) || exit 1
  i=$(sub r5-ident-L --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident L) || exit 1
  l=$(sub r5-launch-L --after $s,$i --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 --env PRIO=$PRIO -- bash $ME launchL L) || exit 1   # PRIO rides into the jobs it submits
  echo "$(date '+%F %T') chainL (code $h): smoke $s identity $a $b bitcmp $i launch $l" | tee -a $O/stages.txt ;;
launchL)  # the three trainings in parallel, each one's PAI 281 ACC + GAP --after it, one report after the three facts
  [[ $ST == L ]] || die "launchL is stage L"
  for T in ${TAGS[$ST]}; do for X in $T ${T/-L18-/x2500-L18-} ${T/-L18-/x5000-L18-} ${T/-L18-/x7500-L18-}; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
    [[ ! -e $R5/pai281/$T-ag ]] || die "281 read of $T exists"; done
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke_R5LC.json'))['pass_'] and json.load(open('$RES/identity_r5l.json'))['identical'] else 1)" \
    || die "smoke / identity not passed"
  for f in $F_/NTV0-L18-s0-281ag.csv $F_/$CTRL-281ag.csv $F_/$MV-281ag.csv; do [[ -s $f ]] || die "no stored $f"; done
  rm -f $O/DONE; h=$(git rev-parse --short HEAD); echo "$h launchL" >> $O/code.txt
  x=(); for T in ${TAGS[$ST]}; do t=$(sub r5-train-$T "${FT[@]}" --timeout-h 4 -- bash $ME train $ST $T) || exit 1
    y=($(read281 $T $PRIO $t)) || exit 1; x+=(${y[-1]})
    echo "$(date '+%F %T') launchL $T (code $h): train $t PAI 281 ag chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]}" | tee -a $O/stages.txt; done
  r=$(sub r5-r281-L --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281L L) || exit 1
  echo "$(date '+%F %T') launchL: report $r" | tee -a $O/stages.txt ;;
r281L)
  rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R2C-ag=$F_/$CTRL-281ag.csv R3MV-s0-ag=$F_/$MV-281ag.csv R3MV-s1-ag=$F_/$MV1-281ag.csv
      R5LC-s0-ag=$F_/R5LC-L18-s0-281ag.csv R5LC-s1-ag=$F_/R5LC-L18-s1-281ag.csv)
  $TR $R2PY r281 --read "${rd[@]}" --pairs R5LC-s0-ag:R3MV-s0-ag R5LC-s1-ag:R3MV-s1-ag R5LC-s0-ag:NTV0-ag R5LC-s1-ag:NTV0-ag R3MV-s1-ag:R3MV-s0-ag \
      R3MV-s0-ag:NTV0-ag R3MV-s1-ag:NTV0-ag R5LC-s1-ag:R5LC-s0-ag R5LC-s0-ag:R2C-ag R5LC-s1-ag:R2C-ag --out $RES/pai281_R5LC_ag.md \
      --title "PAI 281, ACC + GAP: R5-LC (MV-LC) s0 / s1 vs R3MV s0 / s1 and stored NTV0-L18-s0 / R2C-L18-s0" || die "r281 L"
  $V $PY lcverdict --json $RES/pai281_R5LC_ag.json --md $RES/pai281_R5LC_ag.md --read "${rd[@]}" || die "lcverdict"
  date '+%F %T' > $O/DONE ;;
pcsL)  # pcsL L <after id>: submit the ruler chain (CPU only, PRIO 15)
  [[ $ST == L ]] || die "pcsL is stage L"; [[ -n ${3:-} ]] || die "pcsL needs the job id to wait for (r281L)"
  PRIO=15; P0=$DATA_DIR/tmp/pcs_r5; PT=$DATA_DIR/tmp/pcs_r5l; [[ ! -e $PT ]] || die "$PT exists"; for f in $P0/rows.csv.gz $P0/out/fits/paimix.npz; do [[ -s $f ]] || die "no $f"; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/; h=$(git rev-parse --short HEAD); echo "$h pcsL" >> $O/code.txt
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$P0/site)
  c=$(sub r5-pcs-collect --after $3 --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only R5LC-s0+AG,R5LC-s1+AG,R3MV-s1+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub r5-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  r=$(sub r5-pcs-report --after $(csv ${b[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME pcsrep L) || exit 1
  echo "$(date '+%F %T') pcsL (code $h): collect $c boot ${b[*]} report $r" | tee -a $O/stages.txt ;;
pcsrep)
  PT=$DATA_DIR/tmp/pcs_r5l; P0=$DATA_DIR/tmp/pcs_r5; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  python3 - $P0/out $PT/out <<'PY' || die "bootstrap draws of the old reads differ from d429's"
import csv, glob, sys                                            # same seeds, same pool: the old subjects' draws must repeat exactly
rd = lambda d: [r for f in sorted(glob.glob(d + "/boot_*.csv")) for r in csv.DictReader(open(f))]  # noqa: E731
a, b = rd(sys.argv[1]), rd(sys.argv[2])
assert len(a) == len(b) == 100 and all(abs(float(x[k]) - float(y[k])) < 1e-6 for x, y in zip(a, b) for k in x), "boot mismatch"
print("old draws repeat:", len(a), "draws,", len(a[0]), "subjects")
PY
  sj="R2C-s0+AG=R2C-L18-s0-281ag.csv,R2J-s0+AG=R2J-L18-s0-281ag.csv,R3MV-s0+AG=$MV-281ag.csv,R3MV-s1+AG=$MV1-281ag.csv,R5LC-s0+AG=R5LC-L18-s0-281ag.csv,R5LC-s1+AG=R5LC-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" \
      --pairs R5LC-s0+AG:R3MV-s0+AG,R5LC-s1+AG:R3MV-s1+AG,R3MV-s1+AG:R3MV-s0+AG,R5LC-s1+AG:R5LC-s0+AG,R5LC-s0+AG:R2C-s0+AG,R5LC-s1+AG:R2C-s0+AG > $PT/out/r5.log 2>&1 \
      || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_r5l.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_r5l.json || die "copy ruler report"
  $V $PY lcpcs --pcs $RES/pcs_r5l.json --json $RES/pai281_R5LC_ag.json --md $RES/pai281_R5LC_ag.md || die "lcpcs"
  date '+%F %T' > $O/PCSDONE ;;
leadrows)
  [[ $ST == N ]] || die "leadrows is stage N"
  if [[ ${3:-} != run ]]; then h=$(git rev-parse --short HEAD); echo "$h leadrows" >> $O/code.txt
    j=$(sub r5-leadrows --vram 0.1 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME leadrows N run) || exit 1
    echo "$(date '+%F %T') leadrows (code $h): $j" | tee -a $O/stages.txt; exit 0; fi
  $TR $PY leadrows > $O/leadrows.log 2>&1 || die "leadrows (log $O/leadrows.log)"; date '+%F %T' > $O/LEADDONE ;;
relabel) $TR experiments/round5/scripts/s3.py relabel --root $S3O --pool on1_PAIO --out PAIOR ;;
rcheck)  $TR experiments/round5/scripts/s3.py rcheck --pool on1_JTPHOR --ref on1_JTPHO --src on1_PAIOR ;;
prepB)
  [[ -s $DATA_DIR/runs/op_parity/cache/on1_JTPHOR/tab.npz ]] || die "no on1_JTPHOR"
  $TR $PY weights --pool on1_JTPHOR --s3 on1_PAIOR --k-s3 0 --out R5B0 > $O/w_R5B0.log 2>&1 || die "R5B0 weights"
  $TR $PY weights --pool on1_JTPHOR --s3 on1_PAIOR --k-s3 9 --out R5B > $O/w_R5B.log 2>&1 || die "R5B weights" ;;
prepA)  # R5R0 (R3MV + S3 rows at weight 0) and R5A draw files
  [[ -s $DATA_DIR/runs/op_parity/cache/on1_JTPHO/tab.npz ]] || die "no on1_JTPHO"
  $TR $PY weights --k-s3 0 --out R5R0 > $O/w_R5R0.log 2>&1 || die "R5R0 weights"
  $TR $PY weights --k-s3 9 --out R5A > $O/w_R5A.log 2>&1 || die "R5A weights" ;;
smoketrain)
  X=R5SM$ST-L18-s0; if [[ $ST == R ]]; then run r 60 $X; else run ${IDN[$ST]} 60 $X 0; fi
  echo "$(date '+%F %T') smoke $ST: ${A[*]} --diag-every 20 --save-at 20,40" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --save-at 20,40 > $O/smoke.log 2>&1 || die "smoke run $ST (log $O/smoke.log)"
  if [[ $ST == R ]]; then $TR $PY smoke --tag $X --arm R; elif [[ $ST == N ]]; then $TR $PY smoke --tag $X --arm N --wfile R5N
  elif [[ $ST == L ]]; then $TR $PY smoke --tag $X --arm L --wfile R5LC
  elif [[ $ST == G* ]]; then $TR $PY smoke --tag $X --arm G --wfile ${SMK[$ST]}
  else $TR $PY smoke --tag $X --arm A --wfile ${SMK[$ST]} --k-s3 9; fi; r=$?
  for k in 20 40; do [[ -s $RUNS/R5SM${ST}x$k-L18-s0/ckpt-final.pt ]] || { echo "$(date '+%F %T') ERROR smoke $ST: no snapshot x$k" | tee -a $O/stages.txt; r=3; }; done
  rm -rf $RUNS/$X $RUNS/R5SM${ST}x20-L18-s0 $RUNS/R5SM${ST}x40-L18-s0; exit $r ;;
identrun)  # identrun <stage> <variant>: 60 steps eager (pool --env JEV_RM_EAGER=1), this code
  declare -A IX=([mv]=R5IDMV [r0]=R5IDW0 [r]=R5IDR [ro]=R5IDRO [bo]=R5IDBO [ni]=R5IDNI [lci]=R5IDLCI); X=${IX[$3]}-L18-s0; run $3 60 $X
  echo "$(date '+%F %T') identity $3: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$3.log 2>&1 || die "identity run $3 (log $O/ident_$3.log)" ;;
ident)
  if [[ $ST == R ]]; then a=R5IDMV-L18-s0 b=R5IDW0-L18-s0 n="R3MV-L18-s0 argv 60 steps eager: without vs with --on-road-pai road_on1_JTPH.npz --on-road-pai-w 0"
  elif [[ $ST == A ]]; then a=R5IDMV-L18-s0 b=R5IDRO-L18-s0 n="R3MV-L18-s0 argv 60 steps eager: on1_JTPH / R3MV vs on1_JTPHO / R5R0 (S3 rows weight 0)"
  elif [[ $ST == N ]]; then a=R5IDMV-L18-s0 b=R5IDNI-L18-s0 n="R3MV-L18-s0 argv 60 steps eager: R3MV vs R5NI (stage N draw code, nothing dropped)"
  elif [[ $ST == L ]]; then a=R5IDMV-L18-s0 b=R5IDLCI-L18-s0 n="R3MV-L18-s0 argv 60 steps eager: R3MV vs R5LCI (stage L draw code, nothing dropped)"
  else a=R5IDMV-L18-s0 b=R5IDBO-L18-s0 n="R3MV-L18-s0 argv 60 steps eager: on1_JTPH / R3MV vs on1_JTPHOR / R5B0 (relabelled S3 rows weight 0)"; fi
  $TR experiments/defect_fix/scripts/df.py bitcmp --a $a --b $b --out round5_identity_$ST --note "$n"; r=$?
  cp $DFR/round5_identity_$ST.json $RES/identity_r5${ST,,}.json || r=4
  rm -rf $RUNS/$a $RUNS/$b; exit $r ;;
launch)
  for T in ${TAGS[$ST]}; do for X in $T ${T/-L18-/x2500-L18-} ${T/-L18-/x5000-L18-} ${T/-L18-/x7500-L18-}; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done; done
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke_${SMK[$ST]}.json'))['pass_'] and json.load(open('$RES/identity_r5${IDN[$ST]}.json'))['identical'] else 1)" \
    || die "smoke / identity not passed"
  for f in $F_/$CTRL-vl.csv $F_/$MV-vl.csv $F_/NTV0-L18-s0-281ag.csv $F_/$CTRL-281ag.csv $([[ $ST == B* ]] && echo $F_/$A0-vl.csv); do [[ -s $f ]] || die "no stored $f"; done
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
  T=$3; if [[ $ST == R ]]; then run r $N $T; elif [[ $T == $MV1 ]]; then run mv $N $T 1; else run ${IDN[$ST]} $N $T ${T##*-s}; fi
  echo "$(date '+%F %T') train $T: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt; exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
paifix)  # reports vs CTRL and vs the stage's reference (R: R3MV, A: R5-R); the gate; every passing tag's PAI 281 ACC + GAP (PRIO 30) -> r281 -> PAIDONE
  rd=(NTV0-L18-s0-vl=$F_/NTV0-L18-s0-vl.csv $CTRL-vl=$F_/$CTRL-vl.csv R2J-L18-s0-vl=$F_/R2J-L18-s0-vl.csv $MV-vl=$F_/$MV-vl.csv)
  [[ $ST == A ]] && rd+=($RT-vl=$F_/$RT-vl.csv); [[ $ST == G* ]] && rd+=($MVL-vl=$F_/$MVL-vl.csv $N0-vl=$F_/$N0-vl.csv); [[ $ST == G2 ]] && rd+=($GT-vl=$F_/$GT-vl.csv); [[ $ST == B* ]] && rd+=($A0-vl=$F_/$A0-vl.csv); [[ $ST == B1 ]] && rd+=($B0-vl=$F_/$B0-vl.csv); [[ $ST == N1 ]] && rd+=($N0-vl=$F_/$N0-vl.csv)
  for T in ${TAGS[$ST]}; do rd+=($T-vl=$F_/$T-vl.csv); done
  REF=$MV; s=${ST,,}                                              # every stage pairs against R3MV (addendum 1); B / B1 also against R5A-s0 (addendum 2)
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $CTRL-vl --out $RES/paifix_r5${s}_vl_vs_ctrl.md --scores $RES/paifix_r5${s}_vl.csv > /dev/null \
    || die "pai_fix report vs CTRL"
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $REF-vl --out $RES/paifix_r5${s}_vl_vs_${REF%%-*}.md > /dev/null || die "pai_fix report vs $REF"
  [[ $ST == G* ]] && { $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $MVL-vl --out $RES/paifix_r5${s}_vl_vs_R3MVL.md > /dev/null || die "pai_fix report vs $MVL"; }
  [[ $ST == G2 ]] && { $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $GT-vl --out $RES/paifix_r5g2_vl_vs_R5G.md > /dev/null || die "pai_fix report vs $GT"; }
  [[ $ST == B* ]] && { $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref $A0-vl --out $RES/paifix_r5${s}_vl_vs_R5A.md > /dev/null || die "pai_fix report vs $A0"; }
  ref=($REF); [[ $ST == B* ]] && ref+=($A0); [[ $ST == B1 ]] && ref+=($B0); [[ $ST == N1 ]] && ref+=($N0); [[ $ST == G* ]] && ref+=($MVL $N0); [[ $ST == G2 ]] && ref+=($GT)
  ps=$($V $PY gate --arms ${TAGS[$ST]} ${ref[@]} --ctrl $CTRL --stem gate_r5$s) || die "gate"
  ps=$(tr ' ' '\n' <<< "$ps" | grep -xF -f <(tr ' ' '\n' <<< "${TAGS[$ST]}") | xargs)   # the references are listed for comparison only
  echo "$(date '+%F %T') gate: passing [${ps}]" | tee -a $O/stages.txt
  [[ $ST == G* ]] && { date '+%F %T' > $O/PAIDONE; exit 0; }           # stage G: the 281 decision is l37fin's (gate + LAUNCH-37)
  [[ -z $ps ]] && { date '+%F %T' > $O/PAIDONE; bash $ME fin $ST; exit 0; }
  if [[ $ST == B || $ST == N ]]; then b1=$(bash $ME launch ${ST}1 2>&1 | tail -1); echo "$(date '+%F %T') s0 passes: stage ${ST}1 (s1) $b1" | tee -a $O/stages.txt; fi
  r=(); for T in $ps; do y=($(read281 $T 30)) || exit 1
    r+=($(PRIO=30 sub r5-r281-$T --after ${y[-1]} --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281 $ST $T)) || exit 1
    echo "$(date '+%F %T') PAI 281 ag $T: chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]} report ${r[-1]}" | tee -a $O/stages.txt; done
  PRIO=30 sub r5-paidone-$ST --after $(csv ${r[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash -c "date '+%F %T' > $O/PAIDONE && bash $ME fin $ST" > /dev/null ;;
r281)
  T=$3; rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R2C-ag=$F_/$CTRL-281ag.csv R2J-ag=$F_/R2J-L18-s0-281ag.csv); pr=($T-ag:NTV0-ag $T-ag:R2C-ag $T-ag:R2J-ag)
  $TR $R2PY r281 --read "${rd[@]}" $T-ag=$F_/$T-281ag.csv --pairs "${pr[@]}" --out $RES/pai281_${T%-L18-*}${T##*-L18}_ag.md \
      --title "PAI 281, ACC + GAP: $T vs stored NTV0-L18-s0 and R2C-L18-s0 (and R2J)" || die "r281 $T"
  if [[ $ST == N* || $ST == G* ]]; then $V $PY split281 --read "${rd[@]}" $T-ag=$F_/$T-281ag.csv --out $RES/pai281_${T%-L18-*}${T##*-L18}_lead.md || die "split281 $T"; fi ;;
cf)  $TR $IN cf --tags NTV0-L18-s0,$CTRL,$MV,$([[ $ST == A ]] && echo $RT,)$([[ $ST == B* ]] && echo $A0,)$([[ $ST == B1 ]] && echo $B0,)$([[ $ST == N1 ]] && echo $N0,)$(csv ${TAGS[$ST]}),RSG-L18-s0 --gate-tag RSG-L18-s0 --name ${NAME[$ST]} ;;
inreport)
  n=${NAME[$ST]}; $TR $IN report --name $n > $O/inertia_report.log 2>&1 || die "inertia report"
  cp $DFR/inertia/report_$n.md $DFR/inertia/report_$n.json $RES/ || die "no inertia report"
  $TR $LA inertia --name $n --ctl $CTRL --arms ${TAGS[$ST]} $MV $([[ $ST == A ]] && echo $RT) $([[ $ST == B* ]] && echo $A0) $([[ $ST == B1 ]] && echo $B0) $([[ $ST == N1 ]] && echo $N0) --out-stem ${n}_inertia || die "inertia read-out"
  cp $DFR/lon_anchor/${n}_inertia.md $RES/inertia_r5${ST,,}.md && cp $DFR/lon_anchor/${n}_inertia.json $RES/inertia_r5${ST,,}.json || die "copy inertia"
  date '+%F %T' > $O/INDONE; bash $ME fin $ST ;;
fin) [[ -f $O/PAIDONE && -f $O/INDONE ]] && date '+%F %T' > $O/DONE; true ;;
wait)   # returns at $O/DONE or when a job of this stage failed / was cancelled
  until [[ -f $O/DONE ]]; do poolwait; rc=$?; (( rc != 3 )) && break; sleep 300; done   # failed (1) or every job ended without DONE (0)
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -3 $O/stages.txt; true ;;
swait)  # returns when every job of this stage has ended (the smoke stage's waiter)
  until poolwait; rc=$?; (( rc != 3 )); do sleep 120; done; tail -4 $O/stages.txt; true ;;
*) sed -n 2,40p "$0"; exit 2 ;;
esac
