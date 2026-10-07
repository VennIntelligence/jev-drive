#!/usr/bin/env bash
# op_parity replay-hinge lane (plans/2026-10-07-replay-hinge-prereg.md), GPU stages after the offline gate (rh.py report: winner RM).
# One self-advancing chain in tmux jev (scripts/tmux_run.sh); every GPU job goes through the pool.
#   2. pilot: HP-F pilot recipe, seed 0, drivable hinge -> replay hinge + 0.5 m front-corner turn margin (RMP-F-s0); navtest (bench);
#      gate vs HP-F-s0: DAC failures -0.3 pp and EPDMS >= +0.2 (rh.py pilot --gate) -> else STOP (GATE_STOP)
#   3. full: P2H10 recipe x 2 seeds with the same hinge (RMH10-F-s0 / s1); navtest, navhard (GIMM frames), HUGSIM 64 spec_plan_smooth
# State: $DATA_DIR/runs/op_parity/replay_hinge/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/replay_hinge; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
VPY=$PWD/.venv/bin/python
L=$D/pool
RH="--hinge-lam 10 --hinge-replay --hinge-front-turn-margin 0.5"
SPLIT=navsim/op-parity-full
K=12
FULL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
LIST=experiments/hugsim/scripts/derot_all64.txt
status() { echo "$(date '+%F %T') op_parity replay_hinge: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 $RH --tag smoke-rh"

# ---------------------------------------------------------------- 2. pilot (HP-F recipe, seed 0)
status "pilot: train RMP-F-s0"
sub rh-t-RMP-s0 $L/t-RMP-s0 --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed 0 --frames warp --host \
    --data navtrain_full.s0of12 navtrain_full.s1of12 --split $SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 $RH --tag RMP-F-s0 >/dev/null
waitdirs $L/t-RMP-s0
$PY $S/pp_full_check.py train --tag RMP-F-s0 || die "training sanity RMP-F-s0"
status "pilot: navtest"
"${B[@]}" run --model RMP-F-s0 --bench navtest --wait || die "bench navtest RMP-F-s0"
$VPY $S/rh.py pilot --name pilot --arms RMP-F-s0 --refs HP-F-s0 --gate
rc=$?
if [[ $rc == 3 ]]; then status "STOP: pilot gate failed ($(tr -d '\n ' < experiments/op_parity/results/replay_hinge/pilot.json | cut -c1-300))"; touch "$D/GATE_STOP"; exit 0; fi
[[ $rc == 0 ]] || die "pilot readout"
status "pilot gate passed: $(tr -d '\n ' < experiments/op_parity/results/replay_hinge/pilot.json | cut -c1-300)"

# ---------------------------------------------------------------- 3. full P2H10 recipe x 2 seeds, navtest / navhard G / HUGSIM 64 spec_plan_smooth
status "full: train RMH10-F-s0 / s1"
for s in 0 1; do
  sub rh-t-RMH-s$s $L/t-RMH-s$s --train --vram 40 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
      --data $FULL --split $SPLIT --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $RH --tag RMH10-F-s$s >/dev/null
done
waitdirs $L/t-RMH-s0 $L/t-RMH-s1
for s in 0 1; do $PY $S/pp_full_check.py train --tag RMH10-F-s$s || die "training sanity RMH10-F-s$s"; done
status "full: navtest / navhard / HUGSIM"
"${B[@]}" run --model RMH10-F-s0 RMH10-F-s1 --bench navtest || die "bench navtest"
"${B[@]}" run --model RMH10-F-s0@gimm RMH10-F-s1@gimm --bench navhard || die "bench navhard"
"${B[@]}" run --model RMH10-F-s0 RMH10-F-s1 --bench hugsim --preset spec_plan_smooth --scenarios "$LIST" || die "bench hugsim"
"${B[@]}" status --model RMH10-F-s0 RMH10-F-s1 --bench navtest --wait || die "navtest"
"${B[@]}" status --model RMH10-F-s0@gimm RMH10-F-s1@gimm --bench navhard --wait || die "navhard"
"${B[@]}" status --model RMH10-F-s0 RMH10-F-s1 --bench hugsim --preset spec_plan_smooth --wait || die "hugsim"
$VPY $S/rh.py pilot --name full_seedpaired --arms RMH10-F-s0 --refs P2H10-F-s0 || die "full readout s0"
$VPY $S/rh.py pilot --name full_seedpaired_s1 --arms RMH10-F-s1 --refs P2H10-F-s1 || die "full readout s1"
O=experiments/op_parity/results/replay_hinge
"${B[@]}" report --bench navtest --arms RMH=RMH10-F-s0+RMH10-F-s1 --vs P2H=P2H10-F-s0+P2H10-F-s1 WA-JEPA --out $O || die "report navtest"
"${B[@]}" report --bench navhard --arms RMH=RMH10-F-s0@gimm+RMH10-F-s1@gimm --vs P2H=P2H10-F-s0@gimm+P2H10-F-s1@gimm WA-JEPA --out $O || die "report navhard"
"${B[@]}" report --bench hugsim --preset spec_plan_smooth --arms RMH=RMH10-F-s0+RMH10-F-s1 --vs P2H=P2H10-F-s0+P2H10-F-s1 WA-JEPA --scenarios all64 --out $O \
  || die "report hugsim"
status "done"; touch "$D/DONE"
