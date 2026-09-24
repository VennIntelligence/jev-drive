#!/usr/bin/env bash
# One dev L1 tuning round: ramp + profile on the dev routes at p01, then the selection rule.
# usage: b2d_controller_eval_tune_round.sh <round> <comma-separated arms> [workers]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/data/envs/tfv6/bin/python
DEV=/data/runs/b2d/controller-eval/v2/dev
IN=todos/2026-09-23-tfv6-controller/controller-eval
round=$1 arms=$2 workers=${3:-1}
for kind in ramp profile; do
  $PY scripts/b2d_controller_eval_l1_v2.py --kind $kind --refs $DEV/refs --routes $IN/l1-tune-dev.xml \
    --variants $IN/tune/$round-variants.json --arms $arms --seeds p01 --workers $workers \
    --server-base 140 --out $DEV/$round-$kind
done
PYTHONPATH=scripts $PY scripts/b2d_controller_eval_tune_score.py --round $round \
  --cruises $IN/l1-cruises.json --out $IN/tune/$round-result.json
