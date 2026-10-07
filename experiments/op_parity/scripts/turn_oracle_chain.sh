#!/usr/bin/env bash
# op_parity turn-oracle, step 1 (plans/2026-10-08-turn-oracle-prereg.md): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh turn-oracle experiments/op_parity/scripts/turn_oracle_chain.sh). Every GPU / CPU job through the pool, every navtest
# read through jevdrive.bench. Arms = the decision-160 H0 pilot recipe + pp_train --mem <kind>: OS sdf_shuf (matched control), OG sdf_gt
# (oracle, privileged: a probe only), PW sdf_wa / PV sdf_v (probe read-outs). H0 = RH0 and MW = RMW are reused, not retrained.
#   banks -> seed 0 (4 arms) -> navtest (+ OG memory off) -> four_dirs replay -> gate (clear negative: report seed 0 and stop)
#   -> seed 1 -> navtest -> replay -> report
# State: $DATA_DIR/runs/op_parity/turn_oracle/chain/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_oracle
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
T=$S/turn_oracle.py
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
L=$D/pool
DATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
SPLIT=navsim/op-parity-s234
declare -A TAG=([OS]=TOS [OG]=TOG [PW]=TOW [PV]=TOV) MEM=([OS]=sdf_shuf [OG]=sdf_gt [PW]=sdf_wa [PV]=sdf_v)
ARMS=(OS OG PW PV)
status() { echo "$(date '+%F %T') op_parity turn-oracle: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
train() {  # arm seed
  local t=${TAG[$1]}-F-s$2 m=${MEM[$1]}
  local smoke="$PY $S/pp_train.py --arm P2 --mem $m --frames warp --host --data navtrain_full.s2of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 --hinge-lam 10 --tag smoke-to-$m"
  sub to-t-$t $L/t-$t --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --mem $m --seed $2 --frames warp --host \
      --data $DATA --split $SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 --hinge-lam 10 --tag $t
}
seed() {  # seed: train the four arms, navtest reads, replay
  local s=$1 arm specs=()
  status "seed $s: train ${ARMS[*]}"
  for arm in "${ARMS[@]}"; do train $arm $s; done
  for arm in "${ARMS[@]}"; do
    waitdirs $L/t-${TAG[$arm]}-F-s$s
    $PY $S/pp_full_check.py train --tag ${TAG[$arm]}-F-s$s || die "training sanity ${TAG[$arm]}-F-s$s"
    specs+=(${TAG[$arm]}-F-s$s)
  done
  specs+=(TOG-F-s$s:noside)
  status "seed $s: navtest ${specs[*]}"
  "${B[@]}" run --model "${specs[@]}" --bench navtest || die "bench navtest seed $s"
  "${B[@]}" status --model "${specs[@]}" --bench navtest --wait || die "navtest seed $s"
  status "seed $s: four_dirs replay"
  sub to-replay-s$s $L/replay-s$s --vram 0.5 --cpu 48 --ram 64 -- $NAV $T replay --name s$s --models RH0-F-s$s RMW-F-s$s "${specs[@]}"
  waitdirs $L/replay-s$s
}

status "memory banks"
sub to-bank $L/bank --vram 24 --cpu 8 --ram 96 -- $PY $T bank
waitdirs $L/bank
seed 0
status "seed-0 gate"
$PY $T gate --replay s0 > "$D/gate-s0.txt" 2>&1; rc=$?
cat "$D/gate-s0.txt"
(( rc == 0 || rc == 2 )) || die "gate"
if (( rc == 2 )); then
  status "seed-0 gate: CLEAR NEGATIVE (seed 1 not run); report on seed 0"
  cp "$O/gate-s0.json" "$D/GATE_STOP"
  sub to-report-s0 $L/report-s0 --vram 0.5 --cpu 8 --ram 32 -- $PY $T report --name s0 --seeds 0 --replays s0
  waitdirs $L/report-s0
  status "done (clear negative on seed 0)"; date > "$D/DONE"; exit 0
fi
seed 1
status "report"
sub to-report-s0 $L/report-s0 --vram 0.5 --cpu 8 --ram 32 -- $PY $T report --name s0 --seeds 0 --replays s0
sub to-report $L/report --vram 0.5 --cpu 8 --ram 32 -- $PY $T report --seeds 0 1 --replays s0 s1
waitdirs $L/report-s0 $L/report
status "done"; date > "$D/DONE"
