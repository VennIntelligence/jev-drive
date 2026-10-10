#!/usr/bin/env bash
# FLOW1 (experiments/flowhead/plans/2026-10-10-flow-head-prereg.md), part 2, after flow_train_chain.sh is DONE: the turn-oracle replay of the
# DAC failures, the flow heads' samples for the 16 noise rows, their devkit scores on the > 20 deg tokens (score-poses, non-reactive), the
# equal-arc curve offsets, the bench reports and the lane's tables. Outputs stay under $DATA_DIR/runs/flowhead/flow1/ (nothing in the checkout).
#   scripts/tmux_run.sh flow1-read bash experiments/flowhead/scripts/flow_read_chain.sh
# State: $DATA_DIR/runs/flowhead/flow1/read/{STATUS, DONE | ERROR, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/flowhead/flow1
D=$O/read; L=$D/pool; mkdir -p "$L" "$O/report/bench"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
[[ -f $O/chain/DONE ]] || { echo "train chain not finished"; exit 1; }
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
SP=experiments/op_parity/scripts
F=experiments/flowhead/scripts
BN=("$VPY" -m jevdrive.bench)
status() { echo "$(date '+%F %T') flowhead flow1 read: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner flowhead --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
arm() { echo "$1=$2-F-s0${3:-}+$2-F-s1${3:-}"; }

status "replay, samples, curve offsets"
sub flow1-replay "$L/replay" --vram 0.5 --cpu 48 --ram 64 -- $NAV $SP/turn_oracle.py replay --name flow1 \
    --models FMH-F-s0 FMH-F-s1 RGH-F-s0 RGH-F-s1 SH30-F-s0 SH30-F-s1
sub flow1-div "$L/div" --vram 24 --cpu 8 --ram 24 -- $PY $F/flow1.py div
sub flow1-geom "$L/geom" --vram 0.5 --cpu 4 --ram 16 -- $VPY $F/flow1.py geom
waitdirs "$L/div"
status "score-poses of 8 noise rows x 2 seeds on the > 20 deg tokens"
"${BN[@]}" score-poses --poses "$O/div/poses.npz" --traffic non_reactive --out "$O/div/score.csv" --wait || die "score-poses"
waitdirs "$L/replay" "$L/geom"
status "tables"
"${BN[@]}" report --bench navtest --arms $(arm FM FMH) --vs $(arm RG RGH) $(arm SH30 SH30) WA-JEPA --out "$O/report/bench" || die "report navtest"
"${BN[@]}" report --bench navhard --arms $(arm FM FMH @gimm) --vs $(arm RG RGH @gimm) $(arm SH30 SH30 @gimm) WA-JEPA --out "$O/report/bench" || die "report navhard"
sub flow1-report "$L/report" --vram 0.5 --cpu 4 --ram 24 -- $PY $F/flow1.py report
waitdirs "$L/report"
$CL usage --hours 8 > "$D/usage.txt" 2>&1 || true
status "done"; date '+%F %T' > "$D/DONE"
