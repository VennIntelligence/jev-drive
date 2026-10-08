#!/usr/bin/env bash
# op_parity sh30-crossfit (plans/2026-10-08-sh30-crossfit-prereg.md): K = 5 log-disjoint folds of the SH30 recipe (decision 170), held-out plans and
# simulator labels for the navtrain turn tokens, diagnostic, then (step 4) the 33-candidate family around the held-out plans.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh sh30-crossfit experiments/op_parity/scripts/sh30_crossfit_chain.sh). Every GPU / CPU job
# through the pool, inference / scoring through jevdrive.bench.
#   folds check -> fold 0 preflight + alone (throughput gate) -> folds 1-4 -> per fold: navtest + navtrain plans/export -> recipe gate -> assemble -> score
#   -> diagnostic -> [gate failed: STOP] -> family -> stage 0 (300 tokens, identity gate, measured cost, family by the ladder) -> all tokens -> ceiling
# State: $DATA_DIR/runs/op_parity/sh30_crossfit/chain/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}. Rerunning resumes.
# STOP_AFTER=diag ends after the diagnostic (step 4 is then run by a second invocation).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/sh30_crossfit
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
X="$VPY $S/sh30_crossfit.py"
B=("$VPY" -m jevdrive.bench)
L=$D/pool
K=5
HF="--hinge-lam 30 --hinge-margin 0.5"
FULL=$(for i in $(seq 0 11); do echo -n "navtrain_full.s${i}of12 "; done)
status() { echo "$(date '+%F %T') op_parity sh30-crossfit: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
train() {   # train <fold>: pool job, SH30 recipe on the fold's training logs
  local j=$1 m="CF${K}f$1-F-s0"
  local smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split navsim/op-parity-cf${K}f$j --steps 3 --batch 16 --eval-every 3 $HF --tag smoke-cf$j"
  sub cf-t-$j $L/t-$j --train --vram 40 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed 0 --frames warp --host \
      --data $FULL --split navsim/op-parity-cf${K}f$j --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $HF --tag $m
}
post() {    # post <fold>: after training, navtest read and the navtrain plans of the fold model
  local j=$1 m="CF${K}f$1-F-s0"
  waitdirs $L/t-$j
  $PY $S/pp_full_check.py train --tag $m || { echo "training sanity $m" > "$D/post.$j.fail"; return; }
  "${B[@]}" run --model $m --bench navtest || { echo "bench navtest $m" > "$D/post.$j.fail"; return; }
  $X plans --fold $j || { echo "plans $m" > "$D/post.$j.fail"; return; }
  date > "$D/post.$j.ok"
}

status "folds check"
$X folds || die "folds"

# ---------------------------------------------------------------- 1. fold 0 alone: preflight smoke + throughput gate
status "fold 0: preflight and launch alone"
train 0
t0=""
for _ in $(seq 1 120); do   # up to 60 min for the job to start and log step 500
  t0=$(ls -d $DATA_DIR/runs/op_parity/train-CF${K}f0-F-s0/*/ 2>/dev/null | tail -n 1)
  [[ -n $t0 ]] && grep -q "step 500:" "$t0/log.txt" 2>/dev/null && break
  [[ -f $L/t-0/ERROR ]] && die "fold 0 failed: $L/t-0/ERROR"
  sleep 30
done
grep -q "step 500:" "$t0/log.txt" 2>/dev/null || die "fold 0 did not reach step 500 in 60 min"
its=$(grep "step 500:" "$t0/log.txt" | tail -n 1 | sed -E 's/.*; ([0-9.]+) it\/s.*/\1/')
proj=$(python3 -c "print(round(10000/$its/60,1))")
status "fold 0 throughput $its it/s, projected $proj min (SH30: 40 min; stop line 70 min)"
python3 -c "import sys; sys.exit(0 if 10000/$its/60 <= 70 else 1)" || die "fold 0 projected $proj min > 70 min; stop and report"

# ---------------------------------------------------------------- 2. folds 1-4, per-fold post-processing
status "folds 1-4 submitted; post-processing per fold"
for j in 1 2 3 4; do train $j; done
for j in 0 1 2 3 4; do [[ -f $D/post.$j.ok ]] || { rm -f "$D/post.$j.fail"; post $j & }; done
wait
[[ -f $D/ERROR ]] && exit 1
for j in 0 1 2 3 4; do [[ -f $D/post.$j.fail ]] && die "post-processing fold $j: $(cat $D/post.$j.fail)"; [[ -f $D/post.$j.ok ]] || die "post-processing fold $j did not finish"; done
"${B[@]}" status --model CF${K}f0-F-s0 CF${K}f1-F-s0 CF${K}f2-F-s0 CF${K}f3-F-s0 CF${K}f4-F-s0 --bench navtest --wait || die "navtest status"

# ---------------------------------------------------------------- 3. gate, held-out plans, labels, diagnostic
status "recipe gate (fold models on navtest)"
$X gate > "$D/gate.txt" 2>&1; rc=$?
tail -n 12 "$D/gate.txt"
(( rc == 0 || rc == 2 )) || die "gate"
(( rc == 2 )) && { status "recipe gate: NOT PASSED (diagnostic still read, step 4 not run)"; cp "$O/gate.json" "$D/GATE_STOP"; }
status "assemble held-out plans"
$X assemble || die "assemble"
status "score held-out plans (28 323 tokens, v2_navtrain)"
$X score || die "score"
status "diagnostic"
$X diag > "$D/diag.txt" 2>&1 || die "diag"
tail -n 40 "$D/diag.txt"
if [[ -f $D/GATE_STOP ]]; then status "done (gate stop; diagnostic written)"; date > "$D/DONE"; exit 0; fi
[[ ${STOP_AFTER:-} == diag ]] && { status "stopped after the diagnostic (STOP_AFTER)"; date > "$D/DONE"; exit 0; }

# ---------------------------------------------------------------- 4. the 33-candidate family around the held-out plans
status "step 4: family"
$X cands || die "cands"
status "step 4 stage 0: 300 tokens x 33 keys"
$X cscore --stage t300 --family F33 || die "cscore t300"
wall=0; for j in 0 1 2 3 4; do w=$(python3 -c "import json,glob; print(json.load(open(sorted(glob.glob('$DATA_DIR/runs/op_parity/train-CF${K}f$j-F-s0/*/DONE'))[-1]))['wall_s'])") || w=2400; wall=$(python3 -c "print($wall+$w)"); done
spent=$(python3 -c "print(round(6*$wall/3600 + 3.0, 2))")     # declared 6 cores per training + 3 core-h for plans / exports / held-out scoring / diagnostic
$X cgate --spent-core-h "$spent" || die "step-4 stage-0 gate"
fam=$(cat $O/labels/family_chosen.txt)
status "step 4: all tokens x family $fam"
$X cscore --stage all --family "$fam" || die "cscore all"
status "step 4: ceiling table"
$X ceiling --family "$fam" > "$D/ceiling.txt" 2>&1 || die "ceiling"
tail -n 40 "$D/ceiling.txt"
status "done"; date > "$D/DONE"
