#!/usr/bin/env bash
# op_parity geo-oracle (plans/2026-10-09-geo-oracle-prereg.md, decision 197): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh geo-oracle experiments/op_parity/scripts/geo_oracle_chain.sh). Every GPU / CPU job through the pool, every navtest read
# through jevdrive.bench. Arms = the SH30 pilot recipe (P2, hinge lambda 30 / margin 0.5 m, navsim/op-parity-s234, 3 000 steps x 64) + pp_train
# --mem <kind>: H0 none, GS geo_s (true drivable SDF), GA geo_a (true agent occupancy), GB geo_b (both), GX geo_x (geo_b shuffled across logs).
# The banks are PRIVILEGED oracle inputs: probes only, never a reportable driver.
#   split -> tokenizers s / a / b + H0-s0 (together) -> seed 0 (4 memory arms) -> navtest (+ memory-off reads) -> four_dirs replay
#   -> seed-0 gate (clear negative: report seed 0 and stop) -> seed 1 (5 arms) -> navtest -> replay -> report
# State: $DATA_DIR/runs/op_parity/geo_oracle/chain/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/geo_oracle
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
G=$S/geo_oracle.py
B=("$VPY" -m jevdrive.bench)
L=$D/pool
DATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
SPLIT=navsim/op-parity-s234
HF="--hinge-lam 30 --hinge-margin 0.5"
declare -A TAG=([H0]=GH0 [GS]=GOS [GA]=GOA [GB]=GOB [GX]=GOX) MEM=([H0]="" [GS]=geo_s [GA]=geo_a [GB]=geo_b [GX]=geo_x)
MARMS=(GS GA GB GX)
status() { echo "$(date '+%F %T') op_parity geo-oracle: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
memflag() { [[ -n ${MEM[$1]} ]] && echo "--mem ${MEM[$1]}"; }
train() {  # arm seed
  local t=${TAG[$1]}-F-s$2 mf; mf=$(memflag $1)
  local smoke="$PY $S/pp_train.py --arm P2 $mf --frames warp --host --data navtrain_full.s2of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 $HF --tag smoke-geo-${MEM[$1]:-h0}"
  sub geo-t-$t $L/t-$t --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 $mf --seed $2 --frames warp --host \
      --data $DATA --split $SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 $HF --tag $t
}
reads() {  # seed: navtest of the 5 arms + memory-off reads of GS / GA / GB, then the four_dirs replay of all of them
  local s=$1 arm specs=()
  for arm in H0 "${MARMS[@]}"; do
    waitdirs $L/t-${TAG[$arm]}-F-s$s
    $PY $S/pp_full_check.py train --tag ${TAG[$arm]}-F-s$s || die "training sanity ${TAG[$arm]}-F-s$s"
    specs+=(${TAG[$arm]}-F-s$s)
  done
  specs+=(GOS-F-s$s:noside GOA-F-s$s:noside GOB-F-s$s:noside)
  status "seed $s: navtest ${specs[*]}"
  "${B[@]}" run --model "${specs[@]}" --bench navtest || die "bench navtest seed $s"
  "${B[@]}" status --model "${specs[@]}" --bench navtest --wait || die "navtest seed $s"
  status "seed $s: four_dirs replay"
  sub geo-replay-s$s $L/replay-s$s --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/turn_oracle.py replay --name geo_s$s --models "${specs[@]}"
  waitdirs $L/replay-s$s
}

status "split + tokenizers (s, a, b) + H0 seed 0"
$PY $G split || die "split"
for k in s a b; do
  sub geo-tok-$k $L/tok-$k --vram 16 --cpu 6 --ram 48 --preflight "$PY $G tok --kind $k --smoke" -- $PY $G tok --kind $k
done
train H0 0
waitdirs $L/tok-s $L/tok-a $L/tok-b
status "seed 0: train ${MARMS[*]}"
for arm in "${MARMS[@]}"; do train $arm 0; done
reads 0
status "seed-0 gate"
$VPY $G gate > "$D/gate-s0.txt" 2>&1; rc=$?
cat "$D/gate-s0.txt"
(( rc == 0 || rc == 2 )) || die "gate"
if (( rc == 2 )); then
  status "seed-0 gate: CLEAR NEGATIVE (seed 1 not run); report on seed 0"
  cp "$O/gate-s0.json" "$D/GATE_STOP"
  sub geo-report-s0 $L/report-s0 --vram 0.5 --cpu 8 --ram 32 -- $PY $G report --name s0 --seeds 0 --replays geo_s0
  waitdirs $L/report-s0
  status "done (clear negative on seed 0)"; date > "$D/DONE"; exit 0
fi
status "seed 1: train H0 ${MARMS[*]}"
for arm in H0 "${MARMS[@]}"; do train $arm 1; done
reads 1
status "report"
sub geo-report-s0 $L/report-s0 --vram 0.5 --cpu 8 --ram 32 -- $PY $G report --name s0 --seeds 0 --replays geo_s0
sub geo-report $L/report --vram 0.5 --cpu 8 --ram 32 -- $PY $G report --seeds 0 1 --replays geo_s0 geo_s1
waitdirs $L/report-s0 $L/report
status "done"; date > "$D/DONE"
