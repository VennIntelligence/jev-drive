#!/usr/bin/env bash
# Cosmos pilot, CARLA side (todos/2026-09-28-cosmos-pilot.md): drive the selected P5 v1 worlds again with the P5
# recorder plus the 20 Hz openpilot-view cameras (scripts/cosmos_pair_agent.py). One CARLA server, scheduler row
# cosmos-pilot (GPU 1, server index 110, cores 24-47).
# Usage: scripts/cosmos_gen.sh <pilot1|all>
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
which=${1:-pilot1}
gpu=${COSMOS_GPU:-1} index=${COSMOS_CARLA_INDEX:-110} cpus=${COSMOS_CPUS:-24-47}
ids=$($DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cosmos_pilot ids --which "$which")
lead=$DATA_DIR/third_party/scout/lead-cvpr2026
export B2D_RESEED_AFTER_BUILD=1 LEAD_PROJECT_ROOT=$lead HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=3 \
  PYTHONPATH=$lead${PYTHONPATH:+:$PYTHONPATH} SAVE_PATH=$DATA_DIR/runs/cosmos/lead_save CUDA_VISIBLE_DEVICES=$gpu
echo "worlds: $ids"
exec taskset -c "$cpus" $DATA_DIR/envs/carla/bin/python scripts/b2d_run.py --routes $DATA_DIR/runs/p5v1/pairs.xml \
  --route-ids "$ids" --out $DATA_DIR/runs/cosmos/gen --workers 1 --server-index "$index" --gpu-rank "$gpu" \
  --tm-seed-from-id --agent scripts/cosmos_pair_agent.py --agent-config $DATA_DIR/runs/cosmos/agent.json \
  --python $DATA_DIR/envs/scout-tfv6/bin/python --fast-copy --no-spectator --max-attempts 2
