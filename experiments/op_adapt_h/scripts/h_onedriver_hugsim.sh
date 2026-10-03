#!/usr/bin/env bash
# One-driver table, HUGSIM 64 cells for it_dw3-s0: (1) no rule, (2) the sel3 selector (derot_sel 0.6, decision 96), both on all 64
# scenarios with the PR #57 controller. Usage (box, tmux): GPU=0 experiments/op_adapt_h/scripts/h_onedriver_hugsim.sh
# Files: $DATA_DIR/runs/op_adapt_H/chain/onedriver_hugsim/{STATUS,DONE,ERROR}; hugsim/it_dw3-s0_all64 and hugsim/it_dw3-s0_sel3.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}"
cd "$(dirname "$0")/../../.."
C=$DATA_DIR/runs/op_adapt_H/chain/onedriver_hugsim; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
S=experiments/op_adapt_h/scripts
export SCEN=experiments/hugsim/scripts/derot_all64.txt WORKERS=${WORKERS:-4} REPORT=$S/h_hugsim64_report.py
echo "$(date +%T) all64" > "$C/STATUS"
SUFFIX=_all64 $S/h_hugsim.sh it_dw3-s0 || echo all64 > "$C/ERROR"
echo "$(date +%T) sel3" > "$C/STATUS"
OPTS='{"derot_below": 3.0, "derot_sel": 0.6}' SUFFIX=_sel3 $S/h_hugsim.sh it_dw3-s0 || { echo sel3 > "$C/ERROR"; exit 1; }
echo "$(date +%T) done" > "$C/STATUS"; touch "$C/DONE"
