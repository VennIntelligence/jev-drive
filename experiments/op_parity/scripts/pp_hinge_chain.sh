#!/usr/bin/env bash
# op_parity hinge lane (plans/2026-10-06-hinge-prereg.md): P2 (decision 144 recipe) + footprint drivable-area SDF hinge on the plan.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh); every GPU job goes through the pool. Staged:
#   labels (CPU, all navtrain) -> train s0 (lambda LAM0) -> navtest plans + scoring -> GATE (pp_hinge_report.py gate)
#   -> pass: s1 + navtest s1 + navhard (GIMM frames) + HUGSIM 64 exam / spec for both seeds, reports
#   -> EPDMS dropped but DAC moved: the same small read at LAM1, then continue with the lambda that passes
#   -> otherwise: STOP (state file GATE_STOP, STATUS says why).
# State: $DATA_DIR/runs/op_parity/hinge/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt, gate-*.json}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/hinge; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
NAV2=$DATA_DIR/envs/navsim2/bin/python
HPY=$DATA_DIR/envs/hugsim/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
K=12 SPLIT=navsim/op-parity-full STEPS=${STEPS:-10000} BATCH=${BATCH:-128}
LAM0=${LAM0:-10} LAM1=${LAM1:-3}
LABELS=$DATA_DIR/runs/op_probe/labels/navtrain_all.npz
DATA=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
LIST=experiments/hugsim/scripts/derot_all64.txt
status() { echo "$(date '+%F %T') op_parity hinge: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
aft() { [[ $1 == done || -z $1 ]] && echo "" || echo "--after $1"; }
lam_tag() { echo "P2H$1"; }

# ---------------------------------------------------------------- 0. labels: drivable SDF raster for every navtrain token (CPU, ~1-3 min)
status "labels"
if [[ ! -f $LABELS ]]; then
  j_lab=$(sub ppK-labels $L/labels --vram 1 --cpu 64 --ram 64 -- bash -c \
    "$NAV2 experiments/op_probe/scripts/opb_labels.py build --split navtrain --shards $(seq -s' ' 0 $((K - 1))) --workers 60 && \
     mv $DATA_DIR/runs/op_probe/labels/navtrain_s$(seq -s'' 0 $((K - 1))).npz $LABELS")
  waitdirs $L/labels
fi
[[ -f $LABELS ]] || die "labels missing"

# ---------------------------------------------------------------- training / navtest readout of one (lambda, seed)
train() {  # lam seed -> job id
  local lam=$1 s=$2 tag; tag=$(lam_tag $lam)
  local smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of$K --split $SPLIT --steps 3 --batch 16 --eval-every 3 --hinge-lam $lam --tag smoke-hinge"
  sub ppK-t-$tag-s$s $L/t-$tag-s$s --train --vram 40 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed $s \
      --frames warp --host --data $DATA --split $SPLIT --steps $STEPS --batch $BATCH --warmup 300 --eval-every 1000 --hinge-lam $lam --tag $tag-F-s$s >/dev/null
}
navtest_plans() {  # model... -> job id
  sub ppK-plans-$(echo "$@" | tr ' ' '_') $L/plans-$(echo "$@" | tr ' ' '_') --vram 30 --cpu 8 --ram 24 -- \
      $PY $S/pp_eval.py --data lb_navtest --frames warp plans --models "$@" --tag hinge >/dev/null
}
navtest_score() {
  sub ppK-score-$(echo "$@" | tr ' ' '_') $L/score-$(echo "$@" | tr ' ' '_') --vram 1 --cpu 24 --ram 48 --env NAVSIM_THREADS=22 -- \
      $PY $S/pp_eval.py --data lb_navtest --frames warp score --models "$@" >/dev/null
}
small_read() {  # lam -> 0 pass, 1 lower lambda, 2 stop
  local lam=$1 tag; tag=$(lam_tag $lam)
  status "small read lambda $lam: train $tag-F-s0"
  train $lam 0 >/dev/null; waitdirs $L/t-$tag-s0
  $PY $S/pp_full_check.py train --tag $tag-F-s0 || die "training sanity, $tag-F-s0"
  status "small read lambda $lam: navtest plans + scoring"
  navtest_plans $tag-F-s0 >/dev/null; waitdirs $L/plans-$tag-F-s0
  navtest_score $tag-F-s0 >/dev/null; waitdirs $L/score-$tag-F-s0
  $PY $S/pp_hinge_report.py gate --new $tag-F-s0 --ref P2-F-s0; return $?
}

# ---------------------------------------------------------------- 1. GATE (one seed, full recipe, navtest)
LAM=$LAM0
small_read $LAM0; rc=$?
if (( rc == 1 )); then
  status "gate: EPDMS fell but DAC moved at lambda $LAM0; small read at lambda $LAM1"
  LAM=$LAM1; small_read $LAM1; rc=$?
fi
if (( rc != 0 )); then
  status "GATE STOP (lambda $LAM, verdict code $rc): see $D/gate-$(lam_tag $LAM)-F-s0.json"
  cp "$D/gate-$(lam_tag $LAM)-F-s0.json" "$D/GATE_STOP"; exit 0
fi
TAG=$(lam_tag $LAM)
status "gate passed at lambda $LAM; launching s1, navtest s1, navhard, HUGSIM"

# ---------------------------------------------------------------- 2. everything else, in parallel
train $LAM 1 >/dev/null
navhard_g() {  # model -> last job id (plans -> export -> harness)
  local m=$1
  local jp; jp=$(sub ppK-nh-plans-$m $L/nh-plans-$m --vram 30 --cpu 8 --ram 24 $(aft ${2:-}) -- \
      $PY $S/pp_eval.py --data lb_navhard --frames gimm plans --models $m --tag navhard_gimm_hinge)
  local je; je=$(sub ppK-nh-export-$m $L/nh-export-$m --vram 1 --cpu 4 --ram 16 --env OPI_ROOT=op_lb $(aft $jp) -- \
      $JEV experiments/op_openloop/lib/op_interp.py nav-export --data lb_navhard --adapters base --plans gimm@cinque_PP$m)
  sub ppK-nh-h-$m $L/nh-h-$m --vram 1 --cpu 10 --ram 24 $(aft $je) -- \
      $NAV2 experiments/op_guard/scripts/nav_harness.py --poses $DATA_DIR/runs/op_lb/lb_navhard/preds/gimm-cinque_PP${m}__base.npz \
      --out $D/harness/$m --procs 10 >/dev/null
}
hug() {  # model preset
  local m=$1 pr=$2 dn=exam; [[ $pr == spec ]] && dn=spec
  sub ppK-h$dn-$m $L/h-$dn-$m --vram 40 --cpu 14 --ram 45 --env PRESET=$pr -- bash $S/pp_hugsim.sh arm $m $LIST 6 >/dev/null
}
m0=$TAG-F-s0; m1=$TAG-F-s1
navhard_g $m0 >/dev/null; hug $m0 exam; hug $m0 spec
waitdirs $L/t-$TAG-s1
$PY $S/pp_full_check.py train --tag $m1 || die "training sanity, $m1"
status "s1 trained; navtest s1, navhard s1, HUGSIM s1"
navtest_plans $m1 >/dev/null; waitdirs $L/plans-$m1
navtest_score $m1 >/dev/null
navhard_g $m1 >/dev/null; hug $m1 exam; hug $m1 spec
waitdirs $L/score-$m1 $L/nh-h-$m0 $L/nh-h-$m1 $L/h-exam-$m0 $L/h-spec-$m0 $L/h-exam-$m1 $L/h-spec-$m1

# ---------------------------------------------------------------- 3. reports
status "reports"
$PY $S/pp_hinge_report.py navtest --hinge $TAG || die "navtest report"
$PY $S/pp_hinge_report.py navhard --hinge $TAG || die "navhard report"
export FULL_TAGS="P0,P2-F-s0,P2-F-s1,$m0,$m1" FULL_OUT=$PWD/experiments/op_parity/results/hugsim_hinge
$HPY $S/pp_hugsim_report.py extract full || die "hugsim extract"
$PY $S/pp_hugsim_report.py report full || status "hugsim report failed (run it on the Mac from extract.csv)"
status "done (lambda $LAM)"
date > "$D/DONE"
