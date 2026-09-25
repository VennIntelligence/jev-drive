#!/usr/bin/env bash
# Behaviour equivalence of the harness optimisations (todos/2026-09-25-closed-loop-infra-acceptance/profiling.md):
# the Alpamayo exam rig on Town12 route 1773 (a night route, so the street-light behaviour runs), 400 ticks, one
# server, once per variant plus two identical baselines that measure run-to-run spread. Every camera frame's md5 and
# grey thumbnail ($B2D_FRAME_HASH), the street-light rule ($B2D_LIGHTS_CHECK) and the per-tick truth pose / control
# are logged; scripts/infra_verify_compare.py compares them.
#
#   scripts/infra_verify.sh <tag> <gpu> <cpus> [variants...]     (run inside tmux; server indices 780-789)
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
tag=$1 gpu=$2 cpus=$3; shift 3
OUT=$DATA_DIR/runs/infra-acceptance/scale
CFG=$DATA_DIR/runs/infra-acceptance/cfg/alp-route.json
ALP=(--agent scripts/b2d_zeroshot_agent.py --agent-config "$CFG" --decimate 2 --no-spectator)
export B2D_FRAME_HASH=1 B2D_LIGHTS_CHECK=1
declare -A VAR=([base-a]="" [base-b]="" [res64]="--server-args=-ResX=64 -ResY=64" [lights]="--cache-lights"
                [threads8]="--client-threads 8")
for v in "${@:-base-a base-b res64 lights}"; do
    for w in $v; do
        extra=()
        [[ -n ${VAR[$w]} ]] && { [[ $w == res64 ]] && extra=("${VAR[$w]}") || read -ra extra <<< "${VAR[$w]}"; }
        "$DATA_DIR/envs/carla/bin/python" scripts/b2d_scale.py --out "$OUT/$tag-$w" --route-id 1773 --gpus "$gpu" \
            --rungs 1 --max-ticks 400 --cpus "$cpus" --index-lo 780 --index-hi 789 -- "${ALP[@]}" "${extra[@]}"
    done
done
python3 scripts/infra_verify_compare.py "$OUT" "$tag-base-a" "$tag-base-b" "$tag-res64" "$tag-lights"
