#!/usr/bin/env bash
# L1 for new controller candidates (ARMS, e.g. P4) under tag TAG: dev nominal + key interfaces, then
# held-out ramp/profile/crawl and all interface modes. References must already exist (v3 script).
# Score with b2d_controller_eval_l1_v3_score.py. Writes RUNNING/BLOCKED/DONE to $V2/SIGNAL.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PY=/data/envs/tfv6/bin/python
V2=/data/runs/b2d/controller-eval/v2
IN=todos/2026-09-23-tfv6-controller/controller-eval
VAR=$IN/l1-variants-v3.json
signal() { echo "$1 $(date -u +%FT%TZ) $2" > $V2/SIGNAL; echo "== $1 $2"; }
trap 'signal BLOCKED "candidate L1 failed (line $LINENO)"' ERR
run() { $PY scripts/b2d_controller_eval_l1_v2.py --variants $VAR "$@"; }

ARMS=${ARMS:?set ARMS, e.g. P4}; TAG=${TAG:?set TAG, e.g. v4}
signal RUNNING "$TAG dev"
for kind in ramp profile crawl; do
  run --kind $kind --refs $V2/dev/refs --routes $IN/l1-tune-dev.xml --arms $ARMS --seeds p01 \
    --server-base 140 --out $V2/dev/$TAG-$kind
done
for mode in short_2s model_noise stale_2hz stale_1hz; do
  run --kind profile --refs $V2/dev/refs --routes $IN/l1-tune-dev.xml --arms $ARMS --seeds p01 \
    --interface $mode --server-base 140 --out $V2/dev/$TAG-if-$mode
done
signal RUNNING "$TAG held-out"
for kind in ramp profile; do run --kind $kind --refs $V2/refs --arms $ARMS --out $V2/l1-$TAG-$kind; done
run --kind crawl --refs $V2/refs --arms $ARMS --seeds p01 --out $V2/l1-$TAG-crawl
for mode in short_2s model_noise stale_5hz stale_2hz stale_1hz pose_plan stop_jitter sparse_5s; do
  run --kind profile --refs $V2/refs --routes $V2/l1-interface-routes.xml --arms $ARMS --seeds p01 \
    --interface $mode --out $V2/l1-$TAG-if-$mode
done
signal DONE "$TAG L1 complete"
