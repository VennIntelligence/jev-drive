#!/usr/bin/env bash
# Top-10 T3 re-record of the P5 v1 BA worlds with BridgeDrive in shadow and BLUE's camera recorded
# (todos/2026-09-26-top10-intersection.md [T3]). One runner chain on one GPU; run several with disjoint server blocks.
#
#   scripts/top10_t3_gen.sh <gpu> <workers> <server-index> <cpu-list> <ids | @file> [out] [agent-config]
#
# Same b2d_run flags and environment as scripts/p5v1_gen.sh (the original BA generation), except the recorder
# (scripts/top10_t3_agent.py) and its environment (envs/bridgedrive, BridgeDrive's pinned lead a41d116).
# Routes are claimed with O_EXCL, so chains sharing one --out split the work; killed and re-run, it skips done/<id>.json.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/top10_t3
g=$1 w=$2 base=$3 cpus=$4 ids=$5 out=${6:-$R/gen} cfg=${7:-$R/agent.json}
[[ $ids == @* ]] && ids=$(cat "${ids#@}")
L=$DATA_DIR/third_party/bridgedrive/lead
export B2D_RESEED_AFTER_BUILD=1 LEAD_PROJECT_ROOT=$L HF_HUB_OFFLINE=1 SAVE_PATH=$R/lead_save \
    OMP_NUM_THREADS=${OMP_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_THREADS:-3} MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=$L${PYTHONPATH:+:$PYTHONPATH}
# BridgeDrive's closed-loop / training overrides, as in the author's eval_bench2drive_bridgedrive.sh (+ the smoke's
# anchor path and every produce_* output off)
export LEAD_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 diffusion_speed=False produce_demo_image=False produce_demo_video=False produce_debug_image=False produce_debug_video=False produce_input_image=False produce_input_video=False produce_grid_image=False produce_grid_video=False produce_input_log=False"
export LEAD_TRAINING_CONFIG="diffusion_speed=False plan_anchor_path=$L/anchor_utils/anchor_data/lead_cp_kmeans_60_10.npy"
mkdir -p "$out"
span=$(( 50 / w * w ))
echo "$(date '+%F %T') t3-gen gpu $g: $w CARLA instances, CPUs $cpus, server index $base-$(( base + span - 1 )), $(tr ',' '\n' <<< "$ids" | grep -c .) worlds -> $out"
CUDA_VISIBLE_DEVICES=$g BENCH2DRIVE_ROOT=$DATA_DIR/third_party/Bench2Drive WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "$cpus" \
    "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --routes "$DATA_DIR/runs/p5v1/pairs.xml" --route-ids "$ids" --out "$out" \
    --workers "$w" --server-index "$base" --index-span "$span" --gpu-rank "$g" --tm-seed-from-id \
    --agent scripts/top10_t3_agent.py --agent-config "$cfg" --python "$DATA_DIR/envs/bridgedrive/bin/python" \
    --fast-copy --no-spectator --no-reap --max-attempts 2 --stagger-s 20 --client-threads 8
