#!/usr/bin/env bash
# Night queue 3, lane A: one b2d_run chain of scripts/nq3_recorder.py on one GPU (todos/2026-09-26-night-queue-3.md).
#
#   scripts/nq3_gen.sh <gpu> <workers> <server-index> <cpu list> <ids | @file> <out> <agent-config> <routes.xml>
#
# Same b2d_run flags and environment as scripts/p6_gen.sh (P6 v0), except the recorder, the shadow processes it starts
# (their venvs and LEAD trees are in the agent config) and B2D_SENSOR_TICK (cameras may render at 5 Hz). Routes are
# claimed with O_EXCL, so chains sharing one out dir split the work; re-run, it skips done/<id>.json. Every PID it
# starts goes to <out>/pids.txt (scripts/p6_stop.sh <out> stops them, nothing is matched by name).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
g=$1 w=$2 base=$3 cpus=$4 ids=$5 out=$6 cfg=$7 routes=$8
[[ $ids == @* ]] && ids=$(cat "${ids#@}")
[[ -z $ids ]] && { echo "$(date '+%F %T') nq3-gen gpu $g: nothing to do"; exit 0; }
mkdir -p "$out"
L=$DATA_DIR/third_party/scout/lead-cvpr2026            # the recorder's base class (p5_pair_agent) lives there
export B2D_RESEED_AFTER_BUILD=1 B2D_SENSOR_TICK=1 LEAD_PROJECT_ROOT=$L HF_HUB_OFFLINE=1 SAVE_PATH=$DATA_DIR/runs/nq3/a/lead_save \
    OMP_NUM_THREADS=${OMP_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_THREADS:-3} MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    B2D_PIDS_WAIT=${B2D_PIDS_WAIT:-17000}
export PYTHONPATH=$L${PYTHONPATH:+:$PYTHONPATH}
span=$(( 30 / w * w ))
echo "$(date '+%F %T') nq3-gen gpu $g: $w CARLA instances, CPUs $cpus, server index $base-$(( base + span - 1 )), $(tr ',' '\n' <<< "$ids" | grep -c .) worlds -> $out"
CUDA_VISIBLE_DEVICES=$g BENCH2DRIVE_ROOT=$DATA_DIR/third_party/simlingo/Bench2Drive WORK_DIR=$DATA_DIR/third_party/simlingo \
    taskset -c "$cpus" "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --routes "$routes" --route-ids "$ids" --out "$out" \
    --workers "$w" --server-index "$base" --index-span "$span" --gpu-rank "$g" --tm-seed-from-id \
    --agent scripts/nq3_recorder.py --agent-config "$cfg" --python "$DATA_DIR/envs/p5v1-pdm/bin/python" \
    --fast-copy --no-spectator --no-reap --max-attempts 3 --stagger-s 20 --client-threads 8 --stall-s ${STALL_S:-600} &
echo "runner $! gpu $g" >> "$out/pids.txt"
wait $!
