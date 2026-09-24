#!/usr/bin/env bash
# One arm x one TM seed of the SimLingo catalogue experiment (todos/2026-09-25-simlingo-catalogue).
# The same SimLingo release checkpoint and agent file drive under either
#   official : Bench2Drive 0.0.4 (7ec25d1) leaderboard + scenario_runner, as shipped
#   simlingo : the Bench2Drive copy vendored in RenzKa/simlingo (743b243), as shipped
# Everything else (runner, server lifecycle, ports, seeds, route XML content) is identical between arms.
# Usage (on the box): scripts/simlingo_catalogue_run.sh <arm> <tm_seed> <gpu> <server_index> <workers> [b2d_run args...]
#   e.g. scripts/tmux_run.sh slg-off-s1 scripts/simlingo_catalogue_run.sh official 1 0 0 6
set -euo pipefail
arm=$1 seed=$2 gpu=$3 sidx=$4 workers=$5; shift 5
S=$DATA_DIR/third_party/simlingo
SNAP=$(ls -d "$HF_HOME"/hub/models--RenzKa--simlingo/snapshots/26c7c89e797d4e25bbf640013317af8da26a5454)
CKPT=$SNAP/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
case $arm in
  official) export BENCH2DRIVE_ROOT=$DATA_DIR/third_party/Bench2Drive
            routes=$BENCH2DRIVE_ROOT/leaderboard/data/bench2drive220.xml ;;
  simlingo) export BENCH2DRIVE_ROOT=$S/Bench2Drive WORK_DIR=$S  # its evaluator reads $WORK_DIR/leaderboard/data/weather.xml
            routes=$S/leaderboard/data/bench2drive220.xml ;;
  *) echo "arm must be official or simlingo" >&2; exit 2 ;;
esac
out=${OUT_ROOT:-$DATA_DIR/runs/simlingo-catalogue}/$arm/seed$seed
mkdir -p "$out/viz"
# The agent imports simlingo_training.* and team_code.*; it writes its per-step metric dump under $SAVE_PATH.
export PYTHONPATH=$S:$S/team_code${PYTHONPATH:+:$PYTHONPATH}
export SAVE_PATH=$out/viz CUDA_VISIBLE_DEVICES=$gpu B2D_RC_TRACE=1 B2D_CAPTURE_CRITERION_EVENTS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --routes "$routes" --towns all --out "$out" \
  --workers "$workers" --gpu-rank "$gpu" --server-index "$sidx" --tm-seed "$seed" \
  --python "$DATA_DIR/envs/simlingo/bin/python" --agent "$S/team_code/agent_simlingo.py" --agent-config "$CKPT+/run" \
  --stall-s 480 --route-timeout-s 10800 "$@"
