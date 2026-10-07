#!/usr/bin/env bash
# op_parity turn-oracle, step 2, plan-head branch (plans/2026-10-08-turn-oracle-prereg.md, addendum): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh turn-oracle-b experiments/op_parity/scripts/turn_oracle_b_chain.sh). Every job through the pool / jevdrive.bench.
#   stage A  fresh-head ceiling: thin decoder with / without the true SDF -> bench score-poses + four_dirs replay -> areport
#   stage B  P2 pathway, seed 0, each oracle arm with its shuffled control: OG9 / OS9 (9 000 steps), OGh / OSh (hinge lambda 30, margin 0.5)
#            -> navtest (+ memory off) -> replay -> report b0 -> bgate
#   pilot    seed 1 of the best qualifying pair -> navtest -> replay -> report pilot (2-seed means); no pair qualifies: stop (GATE_STOP)
# State: $DATA_DIR/runs/op_parity/turn_oracle/chain-b/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_oracle
D=$O/chain-b; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
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
declare -A TAG=([OS9]=TOS9 [OG9]=TOG9 [OSh]=TOSH [OGh]=TOGH) MEM=([OS9]=sdf_shuf [OG9]=sdf_gt [OSh]=sdf_shuf [OGh]=sdf_gt) \
           STEPS=([OS9]=9000 [OG9]=9000 [OSh]=3000 [OGh]=3000) HINGE=([OS9]="--hinge-lam 10" [OG9]="--hinge-lam 10" [OSh]="--hinge-lam 30 --hinge-margin 0.5" [OGh]="--hinge-lam 30 --hinge-margin 0.5")
status() { echo "$(date '+%F %T') op_parity turn-oracle step 2: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
train() {  # arm seed
  local t=${TAG[$1]}-F-s$2
  sub to-t-$t $L/t-$t --train --vram 24 --cpu 6 --ram 40 -- $PY $S/pp_train.py --arm P2 --mem ${MEM[$1]} --seed $2 --frames warp --host \
      --data $DATA --split $SPLIT --steps ${STEPS[$1]} --batch 64 --warmup 100 --eval-every $(( ${STEPS[$1]} / 3 )) ${HINGE[$1]} --tag $t
}
read_arms() {  # seed name arms... : wait for the trainings, navtest reads (oracle arms also memory off), replay
  local s=$1 name=$2 arm specs=(); shift 2
  for arm in "$@"; do
    waitdirs $L/t-${TAG[$arm]}-F-s$s
    $PY $S/pp_full_check.py train --tag ${TAG[$arm]}-F-s$s || die "training sanity ${TAG[$arm]}-F-s$s"
    specs+=(${TAG[$arm]}-F-s$s); [[ ${MEM[$arm]} == sdf_gt ]] && specs+=(${TAG[$arm]}-F-s$s:noside)
  done
  status "navtest ${specs[*]}"
  "${B[@]}" run --model "${specs[@]}" --bench navtest || die "bench navtest $name"
  "${B[@]}" status --model "${specs[@]}" --bench navtest --wait || die "navtest $name"
  sub to-replay-$name $L/replay-$name --vram 0.5 --cpu 48 --ram 64 -- $NAV $T replay --name $name --models "${specs[@]}" $( (( s == 1 )) && echo RH0-F-s1 RMW-F-s1 )
  waitdirs $L/replay-$name
}

status "stage A decode + stage B training (seed 0)"
sub to-decode $L/decode --vram 24 --cpu 8 --ram 96 -- $PY $T decode
for arm in OS9 OG9 OSh OGh; do train $arm 0; done
waitdirs $L/decode
status "stage A: score-poses + replay"
"${B[@]}" score-poses --poses $O/decode/poses.npz --out $O/decode/score.csv --owner op_parity --wait || die "score-poses"
sub to-replay-decode $L/replay-decode --vram 0.5 --cpu 48 --ram 64 -- $NAV $T replay --name decode --poses $O/decode/poses.npz
waitdirs $L/replay-decode
$PY $T areport --score $O/decode/score.csv > "$D/areport.txt" 2>&1 || die "areport"
cat "$D/areport.txt"; date > "$D/STAGE_A_DONE"

status "stage B: seed 0 reads"
read_arms 0 b0 OS9 OG9 OSh OGh
sub to-report-b0 $L/report-b0 --vram 0.5 --cpu 8 --ram 32 -- $PY $T report --name b0 --seeds 0 --replays s0 b0 \
    --arms H0 OS OG OS9 OG9 OG9:off OSh OGh OGh:off --refs OS H0 --pairs OS9 OG9 --pairs OSh OGh --pairs OS9 OG9:off --pairs OSh OGh:off --bev H0 OS OG9 OGh
waitdirs $L/report-b0
BEST=$($PY $T bgate --replays b0 2> "$D/gate-b.txt"); rc=$?
cat "$D/gate-b.txt"
(( rc == 0 || rc == 2 )) || die "bgate"
if (( rc == 2 )); then
  status "stage B: no pair qualifies (closure >= 0.25 and T20 DAC drop >= 0.4 pp): no pilot"
  cp "$O/gate-b.json" "$D/GATE_STOP"; status "done (no pilot)"; date > "$D/DONE"; exit 0
fi
read -r CTRL ORC <<< "$BEST"
status "pilot: seed 1 of $ORC / $CTRL"
train $CTRL 1; train $ORC 1
read_arms 1 b1 $CTRL $ORC
sub to-report-pilot $L/report-pilot --vram 0.5 --cpu 8 --ram 32 -- $PY $T report --name pilot --seeds 0 1 --replays s0 b0 b1 \
    --arms H0 MW $CTRL $ORC $ORC:off --refs $CTRL H0 --pairs $CTRL $ORC --pairs $CTRL $ORC:off --bev H0 $CTRL $ORC
waitdirs $L/report-pilot
status "done"; date > "$D/DONE"
