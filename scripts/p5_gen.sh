#!/usr/bin/env bash
# P5 pair generation (todos/2026-09-24-p5-carla-pairs-v0.md): drive route variants from runs/p5_pairs/pairs.xml
# with the recorder agent (BehaviorAgent + TFv6 shadow) in envs/scout-tfv6.
# Usage: scripts/p5_gen.sh <out-dir> <workers> <server-index> <route-ids (comma list)> [extra b2d_run args]
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1 workers=$2 index=$3 ids=$4; shift 4
lead=$DATA_DIR/third_party/scout/lead-cvpr2026
cfg=$DATA_DIR/runs/p5_pairs/agent_config.json
[[ -f $cfg ]] || { echo "missing $cfg (write it with the tfv6 model dir first)" >&2; exit 1; }
export B2D_RESEED_AFTER_BUILD=1 LEAD_PROJECT_ROOT=$lead HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-3} \
  PYTHONPATH=$lead${PYTHONPATH:+:$PYTHONPATH} SAVE_PATH=$DATA_DIR/runs/p5_pairs/lead_save
exec $DATA_DIR/envs/carla/bin/python scripts/b2d_run.py --routes $DATA_DIR/runs/p5_pairs/pairs.xml \
  --route-ids "$ids" --out "$out" --workers "$workers" --server-index "$index" --tm-seed-from-id \
  --agent scripts/p5_pair_agent.py --agent-config "$cfg" --python $DATA_DIR/envs/scout-tfv6/bin/python \
  --fast-copy --no-spectator --max-attempts 2 "$@"
