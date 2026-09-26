#!/usr/bin/env bash
# Night queue 3, lane B, arm CL10 (todos/2026-09-26-night-queue-3.md, section CL): the four third-party Bench2Drive
# agents run AS SHIPPED - their own rig, preprocessing, model, controller / PIDs and post-processing heuristics, no
# tuning - on the official Bench2Drive 220 routes with the official evaluator ($DATA_DIR/third_party/Bench2Drive,
# 0.0.4 7ec25d1: 4000-tick cap, 99 % completion), through scripts/b2d_run.py (one CARLA server per worker, resume via
# done/<id>.json, only recorded PIDs are ever killed).
#
#   scripts/nq3_b_cl10.sh <tfv6|bridgedrive|simlingo|blue> <gpu> <workers> <server-index> <seed> <out> [ids,.. | all]
#   e.g. scripts/tmux_run.sh nq3-b-cl10-blue scripts/nq3_b_cl10.sh blue 2 1 385 0 $DATA_DIR/runs/nq3/b/cl10_smoke/blue 24211
#
# <gpu> pins both torch (CUDA_VISIBLE_DEVICES) and CARLA (--gpu-rank -> -graphicsadapter); <seed> is the TM seed.
# The server block is [server-index, server-index + $INDEX_SPAN) (default 2 x workers); keep it below the ephemeral
# port range (index <= 590). Overrides: B_CPUS (taskset, default lane B's 60-109), RUN_FLAGS (default
# "--fast-copy --cache-lights", both evaluator-side and checked equivalent), ROUTE_TIMEOUT_S (default 10800: the
# author agents run their networks every tick, a 4000-tick route at ~1 s / tick needs > 1 h).
#
# Per agent (where it came from in this repo):
#   tfv6        TransFuser v6 (LEAD cvpr2026 730bc1a, $DATA_DIR/third_party/scout/lead-cvpr2026), tfv6_resnet34 three-seed
#               ensemble ($DATA_DIR/checkpoints/scout/tfv6/tfv6_resnet34), envs/scout-tfv6. Author executor "A" of
#               decision 41 = route + target-speed PIDs, with the README's 95.28-DS reproduction heuristics on
#               (Kalman, stop sign, creeping) through the author's LEAD_CLOSED_LOOP_CONFIG: arm A1 of
#               todos/2026-09-25-tfv6-rules-interface, same agent file (scripts/tfv6_rules_agent.py, the author's
#               SensorAgent plus per-tick logging only) and env as scripts/tfv6_rules_run.sh. SAVE_PATH stays unset:
#               the agent points it at <attempt>/lead_save.
#   bridgedrive BridgeDrive 85aa089 on its pinned lead a41d116 ($DATA_DIR/third_party/bridgedrive/lead),
#               model_BridgeDrive_m2_k60_0030.pth ($DATA_DIR/models/bridgedrive), envs/bridgedrive. Closed-loop and
#               training overrides exactly as the author's scripts/eval_bench2drive_bridgedrive.sh (route + target
#               speed, 20 ODE steps, no diffusion speed), plus the anchor file as an absolute path (the author's is
#               relative to the lead root; b2d_route runs from the Bench2Drive root). Agent =
#               scripts/nq3_b_cl10_bridgedrive.py: the author's SensorAgent with the three start-up shims of the T3
#               smoke (GPU-name whitelist, undefined debug_mode, ffmpeg check); nothing touching control changes.
#               The author evaluates with LEAD's Bench2Drive copy + leaderboard_evaluator_v2.py, which differs from
#               the official evaluator only in launching CARLA itself and in restart counts, so the official tree.
#   simlingo    SimLingo (RenzKa/simlingo 743b243, $DATA_DIR/third_party/simlingo/team_code/agent_simlingo.py),
#               release checkpoint epoch=013 from the HF cache, envs/simlingo; the "official" arm of
#               scripts/simlingo_catalogue_run.sh (official tree, its team_code / simlingo_training prepended,
#               InternVL2-1B via $BENCH2DRIVE_ROOT/pretrained symlink). SAVE_PATH = <out>/viz (the agent requires it).
#   blue        BLUE (George-Ling3/BLUE 6970cb6, $DATA_DIR/third_party/blue/team_code/agent_simlingo.py) on the same
#               SimLingo checkpoint, gate gate/weights/blue_simlingo_gate.pt, trained_gate at 0.66, envs/blue; the
#               environment of the author's gate/evaluation/eval_blue_full.sh. The author's evaluator is its own
#               Bench2Drive copy (no 4000-tick cap, 90 % completion); here the official one, as the todo requires.
#               One env-only shim: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, because the gate .pt holds numpy objects that
#               torch >= 2.6's weights_only default refuses. The author's DEBUG = True stays: a viz image every 5 ticks
#               under <out>/viz.
# None of the rigs needs B2D_SENSOR_TICK: the only sensor_tick in them (SimLingo / BLUE IMU 0.05 s, GNSS 0.01 s) is
# <= the 0.05 s frame, i.e. every tick, which is what the official evaluator gives when it drops the attribute.
set -euo pipefail
(( $# >= 6 )) || { sed -n '6,8p' "$0"; exit 2; }
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
agent=$1 gpu=$2 workers=$3 sidx=$4 seed=$5 out=$6 ids=${7:-all}
B2D=$DATA_DIR/third_party/Bench2Drive
if [[ $ids == all ]]; then sel=(--towns all); else sel=(--route-ids "$ids"); fi
SL_CKPT=$HF_HOME/hub/models--RenzKa--simlingo/snapshots/26c7c89e797d4e25bbf640013317af8da26a5454/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
export BENCH2DRIVE_ROOT=$B2D CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
    B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset SAVE_PATH B2D_SENSOR_TICK B2D_PREPEND_PATH
case $agent in
  tfv6)
    L=$DATA_DIR/third_party/scout/lead-cvpr2026
    export LEAD_PROJECT_ROOT=$L PYTHONPATH=$L${PYTHONPATH:+:$PYTHONPATH} NUMBA_NUM_THREADS=3 OPENBLAS_CORETYPE=Haswell \
        LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=True use_kalman_filter=True slower_for_stop_sign=True"
    py=$DATA_DIR/envs/scout-tfv6/bin/python ag=scripts/tfv6_rules_agent.py cfg=$DATA_DIR/checkpoints/scout/tfv6/tfv6_resnet34+A ;;
  bridgedrive)
    L=$DATA_DIR/third_party/bridgedrive/lead
    export LEAD_PROJECT_ROOT=$L PYTHONPATH=$L${PYTHONPATH:+:$PYTHONPATH} NUMBA_NUM_THREADS=3 IS_BENCH2DRIVE=1 \
        PLANNER_TYPE=only_traj \
        LEAD_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 diffusion_speed=False" \
        LEAD_TRAINING_CONFIG="diffusion_speed=False plan_anchor_path=$L/anchor_utils/anchor_data/lead_cp_kmeans_60_10.npy"
    py=$DATA_DIR/envs/bridgedrive/bin/python ag=scripts/nq3_b_cl10_bridgedrive.py cfg=$DATA_DIR/models/bridgedrive ;;
  simlingo)
    S=$DATA_DIR/third_party/simlingo
    export PYTHONPATH=$S:$S/team_code${PYTHONPATH:+:$PYTHONPATH} B2D_PREPEND_PATH=$S SAVE_PATH=$out/viz TRANSFORMERS_OFFLINE=1
    py=$DATA_DIR/envs/simlingo/bin/python ag=$S/team_code/agent_simlingo.py cfg=$SL_CKPT ;;
  blue)
    S=$DATA_DIR/third_party/blue
    export PYTHONPATH=$S:$S/team_code${PYTHONPATH:+:$PYTHONPATH} B2D_PREPEND_PATH=$S SAVE_PATH=$out/viz TRANSFORMERS_OFFLINE=1 \
        BLUE_MODE=trained_gate BLUE_GATE_CKPT=$S/gate/weights/blue_simlingo_gate.pt BLUE_GATE_THRESHOLD=0.66 \
        BLUE_OUTPUT_DIR=$out/blue NCCL_NVLS_ENABLE=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    py=$DATA_DIR/envs/blue/bin/python ag=$S/team_code/agent_simlingo.py cfg=$SL_CKPT ;;
  *) echo "agent must be tfv6|bridgedrive|simlingo|blue" >&2; exit 2 ;;
esac
# SimLingo / BLUE load InternVL2-1B from ./pretrained relative to the evaluator's cwd (the Bench2Drive root)
[[ $agent != simlingo && $agent != blue ]] || [[ -e $B2D/pretrained/InternVL2-1B && -f $SL_CKPT ]] \
    || { echo "missing $B2D/pretrained/InternVL2-1B or $SL_CKPT" >&2; exit 1; }
mkdir -p "$out"
echo "$(date '+%F %T') cl10 $agent: gpu $gpu, $workers workers, server index $sidx, tm seed $seed, routes $ids -> $out" \
    | tee -a "$out/cl10.log"
# shellcheck disable=SC2086
exec taskset -c "${B_CPUS:-60-109}" "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py \
    --routes "$B2D/leaderboard/data/bench2drive220.xml" "${sel[@]}" --out "$out" --workers "$workers" \
    --server-index "$sidx" --index-span "${INDEX_SPAN:-$((2 * workers))}" --gpu-rank "$gpu" --tm-seed "$seed" \
    --python "$py" --agent "$ag" --agent-config "$cfg" --no-spectator --no-reap --client-threads 8 --max-attempts 3 \
    --stall-s 480 --route-timeout-s "${ROUTE_TIMEOUT_S:-10800}" ${RUN_FLAGS---fast-copy --cache-lights}
