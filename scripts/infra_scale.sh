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
mkdir -p "$CFG"
cat > "$CFG/alp-route.json" <<EOF
{"model": "alpamayo", "replay": "route", "plan_every": 5, "controller": "fixed", "controller_preset": "carla",
 "controller_config": "$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json", "seed": 0,
 "dump_every": 0}
EOF
ALP=(--agent scripts/b2d_zeroshot_agent.py --agent-config "$CFG/alp-route.json" --decimate 2 --no-spectator)
OP=(--rig op2tcp3 --policy none --drive route --no-spectator)

scale() {  # scale <tag> <route> <gpus> <rungs> <max ticks> [b2d_scale args] -- <b2d_run args>
    local tag=$1 route=$2 gpus=$3 rungs=$4 ticks=$5; shift 5
    "$DATA_DIR/envs/carla/bin/python" scripts/b2d_scale.py --out "$OUT/$tag" --route-id "$route" --gpus "$gpus" \
        --rungs "$rungs" --max-ticks "$ticks" --cpus "$CPUS" "$@"
}

case $step in
    t12-alp)  scale t12-alp 1773 4 1,2,4,6,8,10,12 1200 "$@" -- "${ALP[@]}" ;;
    t03-alp)  scale t03-alp 25378 4 1,2,4,8,12,16 1200 "$@" -- "${ALP[@]}" ;;
    t13-alp)  scale t13-alp 3561 4 1,4,8 1200 "$@" -- "${ALP[@]}" ;;
    t13-alp-lights) scale t13-alp-lights 3561 4 1,4,8 1200 "$@" -- "${ALP[@]}" --cache-lights ;;
    t12-op)   scale t12-op 1773 4 1,2,4,6 800 "$@" -- "${OP[@]}" ;;
    t12-alp-res64)  # the off-screen spectator viewport shrunk to 64x64; the agent's cameras are untouched
        scale t12-alp-res64 1773 4 1,6,10 1200 "$@" -- "${ALP[@]}" "--server-args=-ResX=64 -ResY=64" ;;
    x2)       # the same totals split over two cards: 4+4 and 6+6 (GPU list from $X2_GPUS, default 1,4)
        scale x2-alp 1773 "${X2_GPUS:-1,4}" 4,6 1200 "$@" -- "${ALP[@]}" ;;
    verify)   # behaviour equivalence of each candidate change at N=1: frame md5s + per-tick truth pose and control
        export B2D_FRAME_HASH=1
        g=${VERIFY_GPU:-0}
        scale verify-base-a 1773 "$g" 1 400 "$@" -- "${ALP[@]}"
        scale verify-base-b 1773 "$g" 1 400 "$@" -- "${ALP[@]}"
        scale verify-res64 1773 "$g" 1 400 "$@" -- "${ALP[@]}" "--server-args=-ResX=64 -ResY=64"
        scale verify-lights 1773 "$g" 1 400 "$@" -- "${ALP[@]}" --cache-lights
        scale verify-threads8 1773 "$g" 1 400 "$@" -- "${ALP[@]}" --client-threads 8 ;;
    pyspy)    # where the route process spends its CPU, Town13 at 1 and 8 servers (sampling costs some CPU)
        scale pyspy-t13 3561 4 1,8 1200 "$@" -- "${ALP[@]}" --python scripts/pyspy_python.sh ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
esac
