#!/usr/bin/env bash
# op-adapt H extra checks on one card (after chain it2 frees it): the adapted pilot-s0 on all 64 HUGSIM scenarios (no de-rotation),
# then pilot-s0 + the derot3 rule on the 10 spin scenarios. Usage (box, tmux): GPU=2 experiments/op_adapt_h/scripts/h_x64_chain.sh
# Files: $DATA_DIR/runs/op_adapt_H/chain/x64/{STATUS,DONE,ERROR}; hugsim/pilot-s0_all64 and hugsim/pilot-s0_derot3.
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
H=$DATA_DIR/runs/op_adapt_H
C=$H/chain/x64; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
S=experiments/op_adapt_h/scripts
until [[ -f $H/chain/it2/DONE ]]; do sleep 30; done
echo "$(date +%T) all64" > "$C/STATUS"
SCEN=experiments/hugsim/scripts/derot_all64.txt SUFFIX=_all64 WORKERS=${WORKERS:-4} REPORT=$S/h_hugsim64_report.py $S/h_hugsim.sh pilot-s0 || { echo all64 > "$C/ERROR"; exit 1; }
echo "$(date +%T) derot3" > "$C/STATUS"
OPTS='{"derot_below": 3.0}' SUFFIX=_derot3 $S/h_hugsim.sh pilot-s0 || { echo derot3 > "$C/ERROR"; exit 1; }
touch "$C/DONE"
