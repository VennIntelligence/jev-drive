#!/usr/bin/env bash
# Concurrent A/B of one harness option (todos/2026-09-25-closed-loop-infra-acceptance/profiling.md): two
# scripts/b2d_scale.py drivers at the same time on the same GPU, N servers each, same route, so both arms see the
# same background load (the box is shared and its load moves by tens of cores within minutes). Arm A is the default,
# arm B adds the option. CPUs 0-21 (A) and 22-44 (B), server indices 740-764 (A) and 765-789 (B).
#
#   [A_EXTRA="<b2d_run args for arm A only>"] scripts/infra_ab.sh <tag> <gpu> <n> <route> <ticks> <slow 0|1> -- <arm B args>
# Both arms run --client-threads 8 (the container's thread cap) unless an arm's own args override it.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
tag=$1 gpu=$2 n=$3 route=$4 ticks=$5 slow=$6; shift 7
OUT=$DATA_DIR/runs/infra-acceptance/scale
CFG=$DATA_DIR/runs/infra-acceptance/cfg/alp-route$([[ $slow == 1 ]] && echo -slow).json
ALP=(--agent scripts/b2d_zeroshot_agent.py --agent-config "$CFG" --decimate 2 --no-spectator --client-threads 8)
run() {  # run <arm> <cpus> <lo> <hi> [extra]
    local arm=$1 cpus=$2 lo=$3 hi=$4; shift 4
    "$DATA_DIR/envs/carla/bin/python" scripts/b2d_scale.py --out "$OUT/$tag-$arm" --route-id "$route" --gpus "$gpu" \
        --rungs "$n" --max-ticks "$ticks" --cpus "$cpus" --index-lo "$lo" --index-hi "$hi" --stagger-s 20 \
        -- "${ALP[@]}" "$@"
}
read -ra A_ARGS <<< "${A_EXTRA:-}"
run a 0-21 740 764 "${A_ARGS[@]}" & pa=$!
sleep 8
run b 22-44 765 789 "$@" & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
echo "arm a rc=$ra, arm b rc=$rb"
