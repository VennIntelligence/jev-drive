#!/usr/bin/env bash
# op_parity joint action-head lane, stage 2 (plans/2026-10-07-joint-action-prereg.md): the stage-1 winner at the P2H10-F full recipe, 2 seeds,
#   -> navtest EPDMS (guard vs P2H10-F) -> HUGSIM 64 spec + spec_plan_smooth. One self-advancing chain in tmux jev; every GPU job via the pool.
#   ARM=JL experiments/op_parity/scripts/pp_joint_full.sh
# State: $DATA_DIR/runs/op_parity/joint_full/{STATUS, DONE, ERROR, log.txt, jobs.txt}; rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
ARM=${ARM:?set ARM (JC|JL|JW) from results/joint_action/gate_s0.json}
D=$DATA_DIR/runs/op_parity/joint_full; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
L=$D/pool
K=12 SPLIT=navsim/op-parity-full STEPS=${STEPS:-10000} BATCH=${BATCH:-128}
DATA=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
declare -A LAB=([JC]=plan [JL]=log [JW]=logwin)
J="--hinge-lam 10 --ego-lat-drop 0.5 --act-lam 3 --act-lab ${LAB[$ARM]}"
OUT=experiments/op_parity/results/joint_action
status() { echo "$(date '+%F %T') op_parity joint_full $ARM: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

status "training ${ARM}F-s0/s1 (full recipe)"
smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of$K --split $SPLIT --steps 3 --batch 16 --eval-every 3 $J --tag smoke-jf"
for s in 0 1; do
  sub ppJF-t-$ARM-s$s $L/t-s$s --train --vram 40 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
      --data $DATA --split $SPLIT --steps $STEPS --batch $BATCH --warmup 300 --eval-every 1000 $J --tag ${ARM}F-F-s$s >/dev/null
done
waitdirs $L/t-s0 $L/t-s1
for s in 0 1; do $PY $S/pp_full_check.py train --tag ${ARM}F-F-s$s || die "training sanity ${ARM}F-F-s$s"; done
M="${ARM}F-F-s0 ${ARM}F-F-s1"
sub ppJF-probe $L/probe --vram 12 --cpu 4 --ram 40 -- $PY $S/pp_joint_probe.py --tags P2H10-F-s0 P2H10-F-s1 $M --out $OUT/probe_full.json >/dev/null

status "navtest"
"${B[@]}" run --model $M --bench navtest --wait || die "bench navtest"
"${B[@]}" report --bench navtest --arms J=${ARM}F-F-s0+${ARM}F-F-s1 --vs H=P2H10-F-s0+P2H10-F-s1 WA-JEPA --out $OUT/full || die "navtest report"
waitdirs $L/probe

status "HUGSIM 64 spec + spec_plan_smooth"
"${B[@]}" run --model $M --bench hugsim --preset spec spec_plan_smooth --scenarios all64 --wait || die "bench hugsim"
status "done (stage 2; report by main)"
date > "$D/DONE"
