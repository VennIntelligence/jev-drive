#!/usr/bin/env bash
# Closed loop for one candidate config as arm P: TFv6 on the 16 held-out scenario routes x seeds 0,1
# and TCP on the 8-route seed-0 subset, two servers each. usage: CONFIG=<json> TAG=<name> $0
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PY=/data/envs/tfv6/bin/python
V2=/data/runs/b2d/controller-eval/v2
IN=todos/2026-09-23-tfv6-controller/controller-eval
signal() { echo "$1 $(date -u +%FT%TZ) $2" > $V2/SIGNAL; echo "== $1 $2"; }
trap 'signal BLOCKED "closed loop ${TAG:-?} failed (line $LINENO)"' ERR
export B2D_P_CONFIG=$(realpath ${CONFIG:?}); TAG=${TAG:?}
ids=$($PY -c "import xml.etree.ElementTree as E;print(','.join(r.get('id') for r in E.parse('$IN/l23-v2-heldout.xml').getroot().findall('route')))")
half1=$(echo $ids | tr ',' '\n' | awk 'NR%2==1' | paste -sd,); half2=$(echo $ids | tr ',' '\n' | awk 'NR%2==0' | paste -sd,)
signal RUNNING "$TAG TFv6"
$PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --route-ids $half1 --arms P --out $V2/l23-tfv6-$TAG --server-index 110 > $V2/$TAG-tfv6-a.log 2>&1 &
a=$!
$PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --route-ids $half2 --arms P --out $V2/l23-tfv6-$TAG --server-index 112 > $V2/$TAG-tfv6-b.log 2>&1 &
b=$!
wait $a; wait $b
signal RUNNING "$TAG TCP"
$PY scripts/b2d_controller_eval_campaign.py --planner tcp --route-ids 24816,24252,26944,25928 --arms P --seeds 0 --out $V2/l23-tcp-$TAG --server-index 110 > $V2/$TAG-tcp-a.log 2>&1 &
a=$!
$PY scripts/b2d_controller_eval_campaign.py --planner tcp --route-ids 25845,26990,25863,3364 --arms P --seeds 0 --out $V2/l23-tcp-$TAG --server-index 112 > $V2/$TAG-tcp-b.log 2>&1 &
b=$!
wait $a; wait $b
signal DONE "closed loop $TAG complete"
