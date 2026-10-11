#!/usr/bin/env bash
# Lane JOINT, phase 1 trainings (plans/2026-10-11-joint1-prereg.md): per new family a 300-step pilot (seed 0, all shards) and its 48-scene smoke
# (joint1_chain.sh smoke) and checklist (joint1_pilot_check.py); a family that passes fans out to its two full runs at once; when all six checkpoints
# exist, the bench reads: navtest and navhard (G frames) for every new checkpoint, HUGSIM 64 `spec` for JAPS. Every GPU job goes through the pool.
#   scripts/tmux_run.sh joint1-train bash experiments/alpasim/scripts/joint1_train.sh
# State: $DATA_DIR/runs/alpasim/joint1/train/{STATUS, DONE | ERROR, log.txt, pool/<job>/}. Rerunning resumes (finished jobs are not resubmitted).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/alpasim/joint1/train; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
BN=("$VPY" -m jevdrive.bench)
R=experiments/alpasim/results/joint1; mkdir -p "$R"
status() { echo "$(date '+%F %T') joint1 train: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 v=$2; shift 2; [[ -f $L/$n/DONE ]] && return
        $CL queue 2>/dev/null | awk -v n="j1-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
        $CL submit --owner alpasim-joint1 --name "j1-$n" --train --priority 15 --vram "$v" --cpu 6 --ram 60 --timeout-h 3 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 30; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }
S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape"
declare -A F=([JAPS]="--std alpasim --yr-mass 0.1 $S" [JAPSnoY]="--std alpasim $S" [JAPSnoA]="--yr-mass 0.1 $S")
declare -A DRV=([JAPS]=ap2 [JAPSnoY]=ap2 [JAPSnoA]=sh30)
declare -A EXP=([JAPS]=102,13,13 [JAPSnoY]=115,0,13 [JAPSnoA]=102,13,13)
declare -A VR=([JAPS]=44 [JAPSnoY]=40 [JAPSnoA]=40)
ARMS=(JAPS JAPSnoY JAPSnoA)
JT="$PY experiments/alpasim/scripts/joint_train.py train --label-bank"
status "code $(git rev-parse --short HEAD); pilots (300 steps, seed 0)"
for a in "${ARMS[@]}"; do sub "p-$a" "${VR[$a]}" $JT ${F[$a]} --seed 0 --steps 300 --eval-every 300 --tag "$a-P300"; done
waitdirs p-JAPS p-JAPSnoY p-JAPSnoA
status "pilot smokes (48 scenes each, at once)"
pids=()                                                              # wait on these pids only: a bare `wait` also waits on the log's tee
for a in "${ARMS[@]}"; do
  [[ -f $DATA_DIR/runs/alpasim/joint1/chain-smoke-$a-P300/DONE ]] && continue
  bash experiments/alpasim/scripts/joint1_chain.sh smoke "$a-P300" "${DRV[$a]}" > /dev/null 2>&1 & pids+=($!)
done
[[ ${#pids[@]} -eq 0 ]] || wait "${pids[@]}"
ok=()
for a in "${ARMS[@]}"; do
  [[ -f $DATA_DIR/runs/alpasim/joint1/chain-smoke-$a-P300/DONE ]] || { status "smoke of $a-P300 failed: $DATA_DIR/runs/alpasim/joint1/chain-smoke-$a-P300"; continue; }
  t=hinge,agent,agent_ho,road_ho
  if $PY experiments/alpasim/scripts/joint1_pilot_check.py --tag "$a-P300" --driver "${DRV[$a]}" --expect "${EXP[$a]}" --terms $t --out "$R/pilot_$a.json" > /dev/null; then
    ok+=("$a")
  else
    status "pilot checklist failed for $a: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fail'])" "$R/pilot_$a.json")"
  fi
done
[[ ${#ok[@]} -gt 0 ]] || die "no family passed its pilot"
status "pilots passed: ${ok[*]}; full runs (10 000 steps, seeds 0 / 1)"
for a in "${ok[@]}"; do for s in 0 1; do sub "f-$a-s$s" "${VR[$a]}" $JT ${F[$a]} --seed $s --tag "$a-s$s"; done; done
for a in "${ok[@]}"; do waitdirs "f-$a-s0" "f-$a-s1"; done
[[ ${#ok[@]} -eq 3 ]] || die "families that did not pass their pilot were not trained: see $R/pilot_*.json"
status "bench reads: navtest, navhard (G frames), HUGSIM 64 spec for JAPS"
NEW=$(for a in "${ARMS[@]}"; do echo -n "$a-s0 $a-s1 "; done)
"${BN[@]}" run --model $NEW --bench navtest --priority 12 || die "bench navtest"
"${BN[@]}" run --model $(for t in $NEW; do echo -n "$t@gimm "; done) --bench navhard --priority 12 || die "bench navhard"
"${BN[@]}" run --model JAPS-s0 JAPS-s1 --bench hugsim --preset spec --scenarios all64 --priority 12 || die "bench hugsim"
"${BN[@]}" status --model $NEW --bench navtest --wait || die "navtest"
"${BN[@]}" status --model $(for t in $NEW; do echo -n "$t@gimm "; done) --bench navhard --wait || die "navhard"
"${BN[@]}" status --model JAPS-s0 JAPS-s1 --bench hugsim --preset spec --wait || die "hugsim"
$CL usage --hours 24 > "$D/usage.txt" 2>&1 || true
status "all done"; date '+%F %T' > "$D/DONE"
