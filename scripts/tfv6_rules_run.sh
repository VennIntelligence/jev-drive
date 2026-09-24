#!/usr/bin/env bash
# TFv6 rules x interface runs (todos/2026-09-25-tfv6-rules-interface/README.md): the author's agent as shipped,
# driving Bench2Drive routes and P5 pair worlds from runs/tfv6_rules/routes.xml (python -m jevdrive.tfv6_rules build).
# Usage: scripts/tfv6_rules_run.sh <A|B> <on|off> <out-dir> <workers> <server-index> <gpu> <route-ids> [b2d_run args]
#   A = route + target speed (author default), B = waypoints; on = README reproduction heuristics
#   (Kalman, stop sign, creeping), off = LEAD defaults. <gpu> pins both torch (CUDA_VISIBLE_DEVICES) and CARLA
#   (--gpu-rank -> -graphicsadapter).
set -euo pipefail
cd "$(dirname "$0")/.."
arm=$1 rules=$2 out=$3 workers=$4 index=$5 gpu=$6 ids=$7; shift 7
case $rules in
  on)  rc="sensor_agent_creeping=True use_kalman_filter=True slower_for_stop_sign=True" ;;
  off) rc="sensor_agent_creeping=False use_kalman_filter=False slower_for_stop_sign=False" ;;
  *) echo "rules must be on|off" >&2; exit 2 ;;
esac
lead=$DATA_DIR/third_party/scout/lead-cvpr2026
export LEAD_CLOSED_LOOP_CONFIG="$rc" B2D_RESEED_AFTER_BUILD=1 B2D_CAPTURE_CRITERION_EVENTS=1 TFV6_AFTER_TRIGGER_S=${TFV6_AFTER_TRIGGER_S:-20} \
  CUDA_VISIBLE_DEVICES=$gpu LEAD_PROJECT_ROOT=$lead HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-3} OPENBLAS_CORETYPE=Haswell PYTHONPATH=$lead${PYTHONPATH:+:$PYTHONPATH}
exec $DATA_DIR/envs/carla/bin/python scripts/b2d_run.py --routes $DATA_DIR/runs/tfv6_rules/routes.xml \
  --route-ids "$ids" --out "$out" --workers "$workers" --server-index "$index" --gpu-rank "$gpu" \
  --agent scripts/tfv6_rules_agent.py --agent-config "$DATA_DIR/checkpoints/scout/tfv6/tfv6_resnet34+$arm" \
  --python $DATA_DIR/envs/scout-tfv6/bin/python --fast-copy --no-spectator --max-attempts 2 "$@"
