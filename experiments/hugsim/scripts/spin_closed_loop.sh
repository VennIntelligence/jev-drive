#!/usr/bin/env bash
# Closed-loop rerun of six Cinque spin-out scenarios under three controllers (fixed = PR #57 as scored, fixed2 = PR #57 + tracker v2,
# ideal = no controller: the ego moves exactly along the plan). One resident Cinque server on one idle card.
# Usage (on the box, in tmux):  GPU=<idle card> [CTRLS="fixed fixed2 ideal"] [OPTS='{}'] [TAG=<suffix>] experiments/hugsim/scripts/spin_closed_loop.sh [out_dir]
# OPTS is the agent option JSON of zs_agent.py, e.g. '{"engage_s": 5, "oracle_vmax": 5}' (diagnostic: the privileged route follower
# drives the first 5 s up to 5 m/s, then the model takes over) with TAG=engage5.
# Output: <out_dir>/cinque-<controller>/..., <out_dir>/results.csv (same layout as the zero-shot exam).
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/hugsim-spin}
L=experiments/hugsim/scripts/spin_scenarios.txt
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$OUT/servers"
rm -f "$OUT/servers/cinque.ready"
CUDA_VISIBLE_DEVICES=$GPU setsid "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/archive/hugsim_zs_server.py cinque \
    --socket "$OUT/servers/cinque.sock" --ready-file "$OUT/servers/cinque.ready" > "$OUT/servers/cinque.log" 2>&1 &
srv=$!
trap 'kill -- -$srv 2>/dev/null' EXIT
until [[ -f $OUT/servers/cinque.ready ]]; do sleep 5; kill -0 $srv 2>/dev/null || { echo "server died"; exit 3; }; done
echo "$(date +%T) server ready"
$HPY experiments/hugsim/archive/zs_run.py setup-trees official fixed fixed2 ideal
for c in ${CTRLS:-fixed fixed2 ideal}; do
    $HPY experiments/hugsim/archive/zs_run.py run --preset exam --out "$OUT" --agent cinque --controller $c --gpu "$GPU" --workers 2 \
        --scenarios "$L" --socket "$OUT/servers/cinque.sock" --opts "${OPTS:-{\}}" ${TAG:+--tag cinque-$c-$TAG}
done
echo "$(date +%T) done"
