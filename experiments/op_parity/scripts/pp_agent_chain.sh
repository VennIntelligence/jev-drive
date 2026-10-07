#!/usr/bin/env bash
# op_parity agent-hinge lane (plans/2026-10-07-agent-hinge-prereg.md, deviations section): one self-advancing chain in tmux jev (scripts/tmux_run.sh).
#   labels (K 32, CPU) + pre-training check -> pilot HA-F-s0 (HP recipe + agent hinge lambda 10) -> navtest -> GATE (pp_agent_report.py gate vs HP-F-s0)
#   -> lambda3: the same small read at lambda 3 (HA3-F-s0) -> pass: full stage P2HA<lam>-F-s0 / s1 (P2H10 recipe + agent hinge), navtest, navhard (G),
#   HUGSIM 64 spec_plan_smooth -> reports. Stop: GATE_STOP, nothing else runs.
# Every GPU job goes through the pool, every evaluation through jevdrive.bench. State: $DATA_DIR/runs/op_parity/agent/{STATUS, DONE, ERROR, GATE_STOP,
# log.txt, jobs.txt, gate-*.json}. Rerunning resumes (finished stages are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/agent; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
L=$D/pool
LAB=$DATA_DIR/runs/op_parity/agent_labels
ALAB=runs/op_parity/agent_labels/navtrain_all-k32.npz
AG="--agent-labels $ALAB --agent-margin 0.5 --agent-side-margin 0"
SPLIT=navsim/op-parity-full
FULL=$(for i in $(seq 0 11); do echo -n "navtrain_full.s${i}of12 "; done)
status() { echo "$(date '+%F %T') op_parity agent: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

# ---------------------------------------------------------------- 0. labels (K 32) + the pre-training check (deviation 3)
status "labels + check"
if [[ ! -f $LAB/navtrain_all-k32.npz || ! -f $LAB/navtest-k32.npz ]]; then
  sub ppA-labels-k32 $L/labels-k32 --vram 1 --cpu 48 --ram 64 -- bash -c \
    "$PY lib/agent_hinge.py build --split navtest --k 32 --suffix=-k32 --workers 46 && $PY lib/agent_hinge.py build --split navtrain --k 32 --suffix=-k32 --workers 46" >/dev/null
  waitdirs $L/labels-k32
fi
if [[ ! -f $LAB/check-k32-m0.5-s0.0.json ]]; then
  sub ppA-check-k32 $L/check-k32 --vram 4 --cpu 8 --ram 32 -- $PY lib/agent_hinge.py check --labels=-k32 --side-margin 0 >/dev/null
  waitdirs $L/check-k32
fi
$PY -c "import json, sys; sys.exit(0 if json.load(open('$LAB/check-k32-m0.5-s0.0.json'))['pass'] else 1)" || die "pre-training geometry check failed"

# ---------------------------------------------------------------- training of one (tag, data, steps, batch, warmup, lambda, seed)
train() {  # name logdir tag lam seed data steps batch warmup vram
  local n=$1 ld=$2 tag=$3 lam=$4 s=$5 data=$6 steps=$7 batch=$8 wu=$9 vram=${10}
  local smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 --hinge-lam 10 --agent-lam $lam $AG --tag smoke-agent"
  sub $n $ld --train --vram $vram --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
      --data $data --split $SPLIT --steps $steps --batch $batch --warmup $wu --eval-every 1000 --hinge-lam 10 --agent-lam $lam $AG --tag $tag >/dev/null
}
small_read() {  # tag lam -> 0 pass, 1 lambda 3, 2 stop
  local tag=$1 lam=$2
  status "small read lambda $lam: train $tag"
  train ppA-t-$tag $L/t-$tag $tag $lam 0 "navtrain_full.s0of12 navtrain_full.s1of12" 3000 64 100 24
  waitdirs $L/t-$tag
  $PY $S/pp_full_check.py train --tag $tag || die "training sanity $tag"
  status "small read lambda $lam: navtest plans + scoring"
  "${B[@]}" run --model $tag --bench navtest --wait || die "bench navtest $tag"
  $PY $S/pp_agent_report.py gate --new $tag --ref HP-F-s0; return $?
}

# ---------------------------------------------------------------- 1. GATE (seed 0, pilot scale)
LAM=10
small_read HA-F-s0 10; rc=$?
if (( rc == 1 )); then
  status "gate: NC + TTC moved but EP fell below -0.2 at lambda 10; small read at lambda 3"
  LAM=3; small_read HA3-F-s0 3; rc=$?
fi
GT=HA-F-s0; [[ $LAM == 3 ]] && GT=HA3-F-s0
if (( rc != 0 )); then
  status "GATE STOP (lambda $LAM, code $rc): see $D/gate-$GT.json"
  cp "$D/gate-$GT.json" "$D/GATE_STOP"; exit 0
fi
status "gate passed at lambda $LAM: full stage"

# ---------------------------------------------------------------- 2. full stage: P2H10 recipe + agent hinge, 2 seeds, navtest / navhard (G) / HUGSIM 64 spec_plan_smooth
T=P2HA$LAM
for s in 0 1; do train ppA-t-$T-s$s $L/t-$T-s$s $T-F-s$s $LAM $s "$FULL" 10000 128 300 40; done
for s in 0 1; do
  waitdirs $L/t-$T-s$s
  $PY $S/pp_full_check.py train --tag $T-F-s$s || die "training sanity $T-F-s$s"
  "${B[@]}" run --model $T-F-s$s --bench navtest || die "bench navtest s$s"
  "${B[@]}" run --model "$T-F-s$s@gimm" --bench navhard || die "bench navhard s$s"
  "${B[@]}" run --model $T-F-s$s --bench hugsim --preset spec_plan_smooth --scenarios all64 || die "bench hugsim s$s"
  status "s$s trained; its evaluations submitted"
done
"${B[@]}" status --model $T-F-s0 $T-F-s1 --bench navtest --wait || die "navtest"
"${B[@]}" status --model "$T-F-s0@gimm" "$T-F-s1@gimm" --bench navhard --wait || die "navhard"
"${B[@]}" status --model $T-F-s0 $T-F-s1 --bench hugsim --preset spec_plan_smooth --wait || die "hugsim"

# ---------------------------------------------------------------- 3. reports (tables under results/agent_hinge/)
status "reports"
$PY $S/pp_agent_report.py navtest --arm "$T=$T-F-s0+$T-F-s1" --ref "P2H10=P2H10-F-s0+P2H10-F-s1" --out full_navtest || die "navtest report"
$PY $S/pp_agent_report.py navhard --arm "$T=$T-F-s0@gimm+$T-F-s1@gimm" --ref "P2H10=P2H10-F-s0@gimm+P2H10-F-s1@gimm" --out full_navhard || die "navhard report"
$PY $S/pp_agent_report.py hugsim --arm "$T=$T-F-s0+$T-F-s1" --ref "P2H10=P2H10-F-s0+P2H10-F-s1" --out full_hugsim || die "hugsim report"
status "done (lambda $LAM)"
date > "$D/DONE"
