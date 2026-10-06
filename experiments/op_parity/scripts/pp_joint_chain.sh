#!/usr/bin/env bash
# op_parity joint action-head lane, stage 1 small read (plans/2026-10-07-joint-action-prereg.md); one self-advancing chain in tmux jev.
#   train JC / JL / JW (seed 0, pilot scale = HP-F recipe + the J changes) -> sanity + offline probe (HP-F-s0 + the J arms)
#   -> HUGSIM turn23 then spin10, presets spec + spec_plan_smooth, for HP-F-s0 and the J arms (jevdrive.bench).
# Every GPU job goes through the pool. State: $DATA_DIR/runs/op_parity/joint/{STATUS, DONE, ERROR, log.txt, jobs.txt}; rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/joint; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
L=$D/pool
SEED=${SEED:-0}
SH="navtrain_full.s0of12 navtrain_full.s1of12"
SPLIT=navsim/op-parity-full STEPS=${STEPS:-3000} BATCH=${BATCH:-64}
J="--hinge-lam 10 --ego-lat-drop 0.5 --act-lam 3"
declare -A LAB=([JC]=plan [JL]=log [JW]=logwin)
ARMS=${ARMS:-"JC JL JW"}
OUT=experiments/op_parity/results/joint_action
HSZ=${HSZ:-}                                   # HUGSIM sizing, e.g. "--workers 2 --jobs 2" on a crowded box
status() { echo "$(date '+%F %T') op_parity joint: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

# ---------------------------------------------------------------- 1. training
status "training $ARMS seed $SEED"
for a in $ARMS; do
  smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 $J --act-lab ${LAB[$a]} --tag smoke-$a"
  sub ppJ-t-$a-s$SEED $L/t-$a-s$SEED --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed $SEED --frames warp --host \
      --data $SH --split $SPLIT --steps $STEPS --batch $BATCH --warmup 100 --eval-every 1000 $J --act-lab ${LAB[$a]} --tag $a-F-s$SEED >/dev/null
done
waitdirs $(for a in $ARMS; do echo $L/t-$a-s$SEED; done)
for a in $ARMS; do $PY $S/pp_full_check.py train --tag $a-F-s$SEED || die "training sanity $a-F-s$SEED"; done

# ---------------------------------------------------------------- 2. offline probe (action gain, plan agreement, history feedback gain)
status "offline probe"
TAGS="HP-F-s$SEED $(for a in $ARMS; do echo -n "$a-F-s$SEED "; done)"
sub ppJ-probe-s$SEED $L/probe-s$SEED --vram 12 --cpu 4 --ram 40 -- $PY $S/pp_joint_probe.py --tags $TAGS --out $OUT/probe_s$SEED.json >/dev/null
waitdirs $L/probe-s$SEED

# ---------------------------------------------------------------- 3. HUGSIM small read: turn23 first (primary), then spin10
for sc in turn23 spin10; do
  status "HUGSIM $sc: $TAGS x spec, spec_plan_smooth"
  "${B[@]}" run --model $TAGS --bench hugsim --preset spec spec_plan_smooth --scenarios $sc $HSZ --wait || die "bench hugsim $sc"
done
status "done (stage 1 small read; gate by pp_joint_report.py)"
date > "$D/DONE"
