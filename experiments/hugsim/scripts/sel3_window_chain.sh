#!/usr/bin/env bash
# sel3 window check (results/sel3_window_check.md): unrotated replay vs native plan on 6 spin scenarios, it_dw3-s0, PR #57 controller.
#   A  original replay: 25 steps, first frame warmed, no desire fix        B  132-step window (derot_ctx 33) + replay-start desire
# The native plan is kept in both (derot_sel with derot_rotate false), so the dynamics equal the native run; derot.dpos in zs_steps.jsonl is
# the max |replay - native| plan position (m). Usage (box, tmux): GPU=0 experiments/hugsim/scripts/sel3_window_chain.sh
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
C=$DATA_DIR/runs/op_adapt_H/chain/sel3_window; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
H=experiments/op_adapt_h/scripts
export SCEN=experiments/hugsim/scripts/sel3_check6.txt WORKERS=${WORKERS:-3}
echo "$(date +%T) A" > "$C/STATUS"
OPTS='{"derot_below": 3.0, "derot_sel": 0.6, "derot_rotate": false}' SUFFIX=_chkA $H/h_hugsim.sh it_dw3-s0 || { echo A > "$C/ERROR"; exit 1; }
echo "$(date +%T) B" > "$C/STATUS"
OPTS='{"derot_below": 3.0, "derot_sel": 0.6, "derot_rotate": false, "derot_ctx": 33, "derot_prev_desire": true}' SUFFIX=_chkB $H/h_hugsim.sh it_dw3-s0 || { echo B > "$C/ERROR"; exit 1; }
echo "$(date +%T) done" > "$C/STATUS"; touch "$C/DONE"
