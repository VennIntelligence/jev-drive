#!/usr/bin/env bash
# Task 10 L1 v3: crawl reference + low-rate/noisy plan interfaces; candidates P2 and P3
# (P3 = P2 + PCHIP plan timing, feedforward low-pass, adaptive stale timeout). Unattended:
# refs -> dev (C, D, P2, P3) -> held-out -> score. Writes RUNNING/BLOCKED/DONE to $V2/SIGNAL.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PY=/data/envs/tfv6/bin/python
V2=/data/runs/b2d/controller-eval/v2
IN=todos/2026-09-23-tfv6-controller/controller-eval
VAR=$IN/l1-variants-v3.json
signal() { echo "$1 $(date -u +%FT%TZ) $2" > $V2/SIGNAL; echo "== $1 $2"; }
trap 'signal BLOCKED "v3 L1 failed (line $LINENO)"' ERR
run() { $PY scripts/b2d_controller_eval_l1_v2.py --variants $VAR "$@"; }

signal RUNNING "v3 refs"
for refs in $V2/refs $V2/dev/refs; do
  probes=$(dirname $refs)/probe
  sha256sum $refs/ramp-*.json $refs/profile-*.json | sort > /tmp/refs-before
  $PY scripts/b2d_controller_eval_refs.py --probes $probes --cruises $IN/l1-cruises.json --out $refs > /dev/null
  sha256sum $refs/ramp-*.json $refs/profile-*.json | sort | diff -q - /tmp/refs-before   # ramp/profile must not change
done

signal RUNNING "v3 dev"
for kind in ramp profile crawl; do
  run --kind $kind --refs $V2/dev/refs --routes $IN/l1-tune-dev.xml --arms C,D,P2,P3 --seeds p01 \
    --server-base 140 --out $V2/dev/v3-$kind
done
for mode in short_2s model_noise stale_2hz stale_1hz; do
  run --kind profile --refs $V2/dev/refs --routes $IN/l1-tune-dev.xml --arms C,D,P2,P3 --seeds p01 \
    --interface $mode --server-base 140 --out $V2/dev/v3-if-$mode
done

signal RUNNING "v3 held-out"
run --kind crawl --refs $V2/refs --arms A,B,C,D,P2,P3 --seeds p01 --out $V2/l1-v3-crawl
for kind in ramp profile; do
  run --kind $kind --refs $V2/refs --arms P2,P3 --out $V2/l1-v3-$kind
done
for mode in model_noise stale_2hz stale_1hz; do
  run --kind profile --refs $V2/refs --routes $V2/l1-interface-routes.xml --arms A,B,C,D,P2,P3 --seeds p01 \
    --interface $mode --out $V2/l1-v3-if-$mode
done
for mode in short_2s sparse_5s stop_jitter stale_5hz pose_plan; do
  run --kind profile --refs $V2/refs --routes $V2/l1-interface-routes.xml --arms P2,P3 --seeds p01 \
    --interface $mode --out $V2/l1-v3-if-$mode
done

signal RUNNING "v3 score"
PYTHONPATH=scripts $PY scripts/b2d_controller_eval_l1_v3_score.py --v2 $V2 --cruises $IN/l1-cruises.json \
  --out $IN/results-v3
signal DONE "v3 L1 complete; results in $IN/results-v3"
