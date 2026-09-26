#!/usr/bin/env bash
# Night queue 2, N6: openpilot small `temporal` on the P5 v1 BehaviorAgent set (scripts/p5_openpilot.py unchanged,
# the Cinque / Lebowski recipe: 5 Hz frames, each held 4 steps of the queued model's 20 Hz clock), N shards on one card.
#   GPU=3 SHARDS=4 WORKERS=4 scripts/n6_op_small.sh
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
export P5_SET=carla_p5v1_ba CUDA_VISIBLE_DEVICES=${GPU:-3}
OP=$DATA_DIR/envs/openpilot/bin/python N=${SHARDS:-4}
echo "$(date '+%F %T') N6 op small: GPU $CUDA_VISIBLE_DEVICES, $N shards x ${WORKERS:-4} render workers"
pids=()
for i in $(seq 0 $((N - 1))); do
  nice -n 5 $OP scripts/p5_openpilot.py --models small --arrays temporal --out-sub op_streams_vis \
    --shard $i/$N --workers ${WORKERS:-4} > $DATA_DIR/tmp/n6-op-small-$i.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
(( rc == 0 )) || { echo "a shard failed"; exit 1; }
.venv/bin/python -m jevdrive.p5_openpilot finalize --models small --arrays temporal --sub op_streams_vis
echo "$(date '+%F %T') N6 op small done"
