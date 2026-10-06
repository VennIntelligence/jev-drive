#!/usr/bin/env bash
# Closed-loop rerun of six Cinque spin-out scenarios under three controllers (fixed = PR #57 as scored, fixed2 = PR #57 + tracker v2,
# ideal = no controller: the ego moves exactly along the plan). One resident Cinque server on one idle card.
# Usage (on the box, in tmux):  GPU=<idle card> [CTRLS="fixed fixed2 ideal"] [OPTS='{}'] [TAG=<suffix>] experiments/hugsim/scripts/spin_closed_loop.sh [out_dir]
# OPTS is the agent option JSON of zs_agent.py, e.g. '{"engage_s": 5, "oracle_vmax": 5}' (diagnostic: the privileged route follower
# drives the first 5 s up to 5 m/s, then the model takes over) with TAG=engage5.
# Output: <out_dir>/cinque-<controller>/..., <out_dir>/results.csv (same layout as the zero-shot exam).
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-spin}
L=experiments/hugsim/scripts/spin_scenarios.txt
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$OUT"
source scripts/bench_lane.sh
for c in ${CTRLS:-fixed fixed2 ideal}; do
    bench_hugsim cinque exam "cinque-$c${TAG:+-$TAG}" "$L" 2 "$c" "${OPTS:-\{\}}" || exit 1
done
echo "$(date +%T) done"
