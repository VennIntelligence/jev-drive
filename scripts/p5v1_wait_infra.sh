#!/usr/bin/env bash
# Gate for the P5 v1 bulk generation (todos/2026-09-25-reactivity-program/i1-p5v1.md): return once the infra agent's
# CARLA profiling (todos/2026-09-25-closed-loop-infra-acceptance.md) has reported its worker layout, i.e. once
# gpu-plan.md has a line containing "[INFRA-PROFILE] layout ready". Machine-readable keys on that line
# (servers_per_gpu=N, cores_per_server=C) are copied to $DATA_DIR/runs/p5v1/layout.infra.env, which scripts/p5v1_gen.sh
# reads after layout.env; without keys the launcher keeps layout.env's defaults (our own profiling).
# Run it as slot p5v1-wait-infra. A script file on purpose: a grep pattern on a bash -c command line can match the
# waiter itself (docs/long-runs.md).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
OUT=$DATA_DIR/runs/p5v1/layout.infra.env
until line=$(grep -F '[INFRA-PROFILE] layout ready' "$PLAN" | tail -1) && [[ -n $line ]]; do sleep 60; done
echo "$(date +%T) infra layout reported: $line"
{
    echo "# from gpu-plan.md at $(date '+%F %T'): $line"
    s=$(grep -oE 'servers_per_gpu=[0-9]+' <<< "$line" | tail -1 | cut -d= -f2)
    c=$(grep -oE 'cores_per_server=[0-9.]+' <<< "$line" | tail -1 | cut -d= -f2)
    [[ -n $s ]] && echo "INFRA_SERVERS_PER_GPU=$s"
    [[ -n $c ]] && echo "INFRA_CORES_PER_SERVER=${c%%.*}"
} > "$OUT"
cat "$OUT"
