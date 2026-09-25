#!/usr/bin/env bash
# CARLA harness scaling ladders of the closed-loop infrastructure acceptance
# (todos/2026-09-25-closed-loop-infra-acceptance/profiling.md). Each step is one scripts/b2d_scale.py invocation:
# the same route on every worker, fixed --max-ticks, servers per GPU climbing rung by rung.
#
#   scripts/infra_scale.sh <step> [extra b2d_scale args...]      run inside tmux (scripts/tmux_run.sh)
#
# Rigs: alp = the Alpamayo exam rig through the real exam agent (scripts/b2d_zeroshot_agent.py, "replay": "route",
# no model; 4 cameras at 10 Hz, --decimate 2 as in scripts/zeroshot_b2d_alp.sh); op = the openpilot + TCP partner
# rig through the cost stub (scripts/b2d_agent.py --rig op2tcp3: 2 x 1928x1208 + 3 x 1600x900, every tick).
# CPUs: $SCALE_CPUS (default 52-96, 45 CPUs on GPU 4's NUMA node). CARLA server indices 740-789.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
step=$1; shift
OUT=$DATA_DIR/runs/infra-acceptance/scale
CFG=$DATA_DIR/runs/infra-acceptance/cfg
CPUS=${SCALE_CPUS:-52-96}
G=${SCALE_GPU:-4}   # GPU of the single-GPU ladders
mkdir -p "$CFG"
cat > "$CFG/alp-route.json" <<EOF
{"model": "alpamayo", "replay": "route", "plan_every": 5, "controller": "fixed", "controller_preset": "carla",
 "controller_config": "$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json", "seed": 0,
 "dump_every": 0}
EOF
sed 's/"seed": 0,/"seed": 0, "cruise_mps": 1.5,/' "$CFG/alp-route.json" > "$CFG/alp-route-slow.json"
ALP=(--agent scripts/b2d_zeroshot_agent.py --agent-config "$CFG/alp-route.json" --decimate 2 --no-spectator)
# Bench2Drive routes are 100-220 m and the route oracle finishes most of them in 300-600 ticks at 8 m/s, which
# leaves no steady window; at 1.5 m/s a route lasts past the 1200-tick cap. Town12 route 1773 is blocked by its
# obstacle scenario and ticks to the cap at 8 m/s, so its ladder keeps the exam speed.
ALPS=(--agent scripts/b2d_zeroshot_agent.py --agent-config "$CFG/alp-route-slow.json" --decimate 2 --no-spectator)
OP=(--rig op2tcp3 --policy none --drive route --no-spectator)

scale() {  # scale <tag> <route> <gpus> <rungs> <max ticks> [b2d_scale args] -- <b2d_run args>
    local tag=$1 route=$2 gpus=$3 rungs=$4 ticks=$5; shift 5
    "$DATA_DIR/envs/carla/bin/python" scripts/b2d_scale.py --out "$OUT/$tag" --route-id "$route" --gpus "$gpus" \
        --rungs "$rungs" --max-ticks "$ticks" --cpus "$CPUS" "$@"
}

case $step in
    t12-alp)  scale t12-alp 1773 "$G" 1,2,4,6,8,10,12 1200 "$@" -- "${ALP[@]}" ;;
    t03-alp)  scale t03-alp 25378 "$G" 1,2,4,6,8,12,16 1200 "$@" -- "${ALPS[@]}" ;;
    t13-alp)  scale t13-alp 3561 "$G" 1,4,8 1200 "$@" -- "${ALPS[@]}" ;;
    t13-alp-lights) scale t13-alp-lights 3561 "$G" 1,4,8 1200 "$@" -- "${ALPS[@]}" --cache-lights ;;
    t12-op)   scale t12-op 1773 "$G" 1,2,4,6 800 "$@" -- "${OP[@]}" ;;
    t12-alp-res64)  # the off-screen spectator viewport shrunk to 64x64; the agent's cameras are untouched
        scale t12-alp-res64 1773 "$G" 1,6 1200 "$@" -- "${ALP[@]}" "--server-args=-ResX=64 -ResY=64" ;;
    # With --client-threads 8 the route client runs 8 CARLA worker threads instead of one per host hardware thread
    # (208): ~200 fewer threads per worker against the container's pids.max of 20480, which the default hit.
    t12-base6) scale t12-base6 1773 "$G" 6 1200 "$@" -- "${ALP[@]}" ;;   # same-session baseline for t12-t8
    t12-t8)   scale t12-t8 1773 "$G" 6 1200 "$@" -- "${ALP[@]}" --client-threads 8 ;;
    t13-t8)   scale t13-t8 3561 "$G" 1,8 1200 "$@" -- "${ALPS[@]}" --client-threads 8 ;;
    t13-lights-t8) scale t13-lights-t8 3561 "$G" 1,8 1200 "$@" -- "${ALPS[@]}" --client-threads 8 --cache-lights ;;
    t12-op-t8) scale t12-op-t8 1773 "$G" 1,4,6 800 "$@" -- "${OP[@]}" --client-threads 8 ;;
    t03-t8)   scale t03-t8 25378 "$G" 1,12,16 1200 "$@" -- "${ALPS[@]}" --client-threads 8 ;;
    x2)       # the same totals split over two cards: 4+4 and 6+6 (GPU list from $X2_GPUS, default 1,4)
        scale x2-alp 1773 "${X2_GPUS:-1,4}" 4,6 1200 "$@" -- "${ALP[@]}" ;;
    pyspy)    # where the route process spends its CPU, Town13 at 1 and 8 servers (sampling costs some CPU)
        scale pyspy-t13 3561 "$G" 1,8 1200 "$@" -- "${ALPS[@]}" --python scripts/pyspy_python.sh ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
esac
