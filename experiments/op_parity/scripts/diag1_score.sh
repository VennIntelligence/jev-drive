#!/usr/bin/env bash
# Lane DIAG1, D6 on navtest: swap pose keys (diag1.py nav-swap) scored by `jevdrive.bench score-poses --traffic non_reactive` (CPU pool jobs).
# Run: scripts/tmux_run.sh diag1-score experiments/op_parity/scripts/diag1_score.sh
# State: $DATA_DIR/runs/op_parity/diag1/score/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (score-poses keeps finished chunks).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/diag1; ST=$D/score; mkdir -p "$ST"; rm -f "$ST/DONE" "$ST/ERROR"
exec > >(tee -a "$ST/log.txt") 2>&1
V=.venv/bin/python
status() { echo "$(date '+%F %T') diag1-score: $*" | tee "$ST/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$ST/ERROR"; exit 1; }
status "building swap poses"
[[ -f $D/swap_poses.npz ]] || $V experiments/op_parity/scripts/diag1.py nav-swap || die "nav-swap"
status "score-poses (8 jobs x 12 cores)"
$V -m jevdrive.bench score-poses --poses $D/swap_poses.npz --out $D/swap_score.csv --traffic non_reactive --cpu 12 --jobs 8 --wait || die "score-poses"
status "done"; date > "$ST/DONE"
