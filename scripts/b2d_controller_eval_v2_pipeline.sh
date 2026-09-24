#!/usr/bin/env bash
# Task 10 v2, unattended: dev tuning (2 pre-registered rounds) -> freeze P -> held-out L1 for P
# -> L1 score -> TCP/TFv6 P smoke -> L2/L3 campaigns -> L2/L3 score.
# Every stage leaves $V2/stage-<name>.done and is skipped on rerun. Any failure writes
# "BLOCKED" to $V2/SIGNAL and stops; the end writes "DONE".
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PY=/data/envs/tfv6/bin/python
V2=/data/runs/b2d/controller-eval/v2
IN=todos/2026-09-23-tfv6-controller/controller-eval
RES=$IN/results-v2
mkdir -p $RES
signal() { echo "$1 $(date -u +%FT%TZ) $2" > $V2/SIGNAL; echo "== $1 $2"; }
trap 'signal BLOCKED "stage ${STAGE:-?} failed (line $LINENO)"' ERR
stage() { STAGE=$1; [ -f $V2/stage-$1.done ] && { echo "skip $1"; return 1; }; signal RUNNING "$1"; return 0; }
done_() { touch $V2/stage-$STAGE.done; }

STAGE=wait-dev-refs
until [ -f $V2/dev/refs/profile-traces.json ]; do sleep 60; done

if stage tune-round1; then
  scripts/b2d_controller_eval_tune_round.sh round1 E1,E2,E3,E4,E5,E6,B,C,D
  done_; fi
if stage tune-round2; then
  arms=$($PY scripts/b2d_controller_eval_tune_next.py --from-round round1 --to-round round2)
  scripts/b2d_controller_eval_tune_round.sh round2 "$arms"
  done_; fi
if stage freeze-p; then
  $PY scripts/b2d_controller_eval_tune_next.py --from-round round2 --freeze
  done_; fi
P_CONFIG=$(realpath $IN/P-final.json)
printf '{"P": {"controller_config": "%s", "presets": ["pursuit"]}}\n' "$P_CONFIG" > $V2/p-variants.json

STAGE=wait-l1-chain
until grep -q "L1 v2 chain complete" $V2/l1-chain.log; do sleep 120; done

if stage l1-p; then
  for kind in ramp profile; do
    $PY scripts/b2d_controller_eval_l1_v2.py --kind $kind --refs $V2/refs --variants $V2/p-variants.json \
      --arms P --out $V2/l1-P-$kind
  done
  for mode in short_2s sparse_5s stop_jitter stale_5hz pose_plan; do
    $PY scripts/b2d_controller_eval_l1_v2.py --kind profile --refs $V2/refs --variants $V2/p-variants.json \
      --arms P --interface $mode --routes $V2/l1-interface-routes.xml --seeds p01 --out $V2/l1-P-interface-$mode
  done
  done_; fi
if stage l1-score; then
  PYTHONPATH=scripts $PY scripts/b2d_controller_eval_l1_v2_score.py --v2 $V2 --cruises $IN/l1-cruises.json --out $RES
  done_; fi

export B2D_P_CONFIG=$P_CONFIG
if stage smoke; then
  $PY scripts/b2d_controller_eval_campaign.py --planner tcp --routes $V2/tcp-smoke-24240.xml \
    --arms NP --seeds 0 --out $V2/smoke-tcp --server-index 150
  $PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --routes $V2/tcp-smoke-24240.xml \
    --arms P --seeds 0 --out $V2/smoke-tfv6 --server-index 152
  done_; fi

ids=$($PY -c "import xml.etree.ElementTree as E;print(' '.join(r.get('id') for r in E.parse('$IN/l23-v2-heldout.xml').getroot().findall('route')))")
half1=$(echo $ids | tr ' ' '\n' | awk 'NR%2==1' | paste -sd,)
half2=$(echo $ids | tr ' ' '\n' | awk 'NR%2==0' | paste -sd,)
for planner in tfv6 tcp; do
  if stage l23-$planner; then
    $PY scripts/b2d_controller_eval_campaign.py --planner $planner --route-ids $half1 \
      --out $V2/l23-$planner --server-index 110 > $V2/l23-$planner-a.log 2>&1 &
    first=$!
    $PY scripts/b2d_controller_eval_campaign.py --planner $planner --route-ids $half2 \
      --out $V2/l23-$planner --server-index 112 > $V2/l23-$planner-b.log 2>&1 &
    second=$!
    wait $first; wait $second
    done_; fi
done
if stage l23-score; then
  $PY scripts/b2d_controller_eval_l23_score.py --tfv6 $V2/l23-tfv6 --tcp $V2/l23-tcp --out $RES
  done_; fi
signal DONE "pipeline complete; results in $RES"
