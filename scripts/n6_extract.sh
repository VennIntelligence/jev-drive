#!/usr/bin/env bash
# Night queue 2, N6: V-JEPA 2 / DINOv2 / SigLIP2 features on the P5 v1 BA set, sharded over cards, then finalize.
#   SHARDS="3 3 4" WORKERS=16 scripts/n6_extract.sh      # one shard per listed GPU (a card may appear twice)
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
export P5_SET=carla_p5v1_ba
read -ra G <<< "${SHARDS:-3 3 4}"
N=${#G[@]}
echo "$(date '+%F %T') N6 extract: $N shards on GPUs ${G[*]}"
pids=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 5 .venv/bin/python -m jevdrive.n6_backbones extract --shard $i/$N \
    --batch ${BATCH:-64} --workers ${WORKERS:-16} > $DATA_DIR/tmp/n6-extract-$i.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
(( rc == 0 )) || { echo "a shard failed"; exit 1; }
.venv/bin/python -m jevdrive.n6_backbones finalize
echo "$(date '+%F %T') N6 extract done"
