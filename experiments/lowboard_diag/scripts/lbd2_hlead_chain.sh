#!/usr/bin/env bash
# LOWDIAG2 (prereg amendment 3), HLEAD repeat via jevdrive.bench: op_lead on SH30-F-s1, op_lead on SH30-F-s0 once more, switch-off repeat on SH30-F-s1.
#   scripts/tmux_run.sh lbd2-hlead bash experiments/lowboard_diag/scripts/lbd2_hlead_chain.sh
# State: $DATA_DIR/runs/lowboard_diag/hlead2/{DONE | ERROR, *.log}; runs land under $DATA_DIR/runs/bench/hugsim/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/lowboard_diag/hlead2; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"; B=".venv/bin/python -m jevdrive.bench"
$B run --model SH30-F-s1 --bench hugsim --preset spec_plan_smooth --opts '{"op_lead": {}}' --scenarios all64 > "$O/s1_on.log" 2>&1 & p1=$!
$B run --model SH30-F-s0 --bench hugsim --preset spec_plan_smooth --opts '{"op_lead": {}}' --repeat hlead2 --scenarios all64 > "$O/s0_on_rep.log" 2>&1 & p2=$!
$B run --model SH30-F-s1 --bench hugsim --preset spec_plan_smooth --repeat hlead-off --scenarios all64 > "$O/s1_off_rep.log" 2>&1 & p3=$!
rc=0; for p in $p1 $p2 $p3; do wait $p || rc=1; done
(( rc == 0 )) && date > "$O/DONE" || echo "a bench run failed" > "$O/ERROR"
