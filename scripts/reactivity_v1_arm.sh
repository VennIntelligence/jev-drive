#!/usr/bin/env bash
# Arm the P5 v1 re-check chain (todos/2026-09-25-reactivity-program.md, deviation 7) as gated slots in tmux `jev`.
# Everything waits for p5v1-index; PDM-Lite (the primary set) goes first on every card, BehaviorAgent follows.
#
#   p5v1-index -+- reactivity-v1-qwen-pdm -- reactivity-v1-qwen-ba --+
#               +- reactivity-v1-op-pdm ---- reactivity-v1-op-ba ----+
#   qwen-pdm + op-pdm -> reactivity-v1-exam-pdm, reactivity-v1-mc-pdm
#   qwen-ba  + op-ba  -> reactivity-v1-exam-ba,  reactivity-v1-mc-ba
#   all four          -> reactivity-v1-final
# Re-running this script re-arms every slot that has no .done sentinel (resumable: chunks, streams and sentinels skip
# finished work). Extra environment (V1_GPUS, QWEN_PROCS, ...) is passed through to scripts/reactivity.sh.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
S=$DATA_DIR/runs/sched
ENVS=()
for v in V1_GPUS QWEN_PROCS QWEN_CORES QWEN_CPUS OP_CPUS OP_CORES OP_GPU EXAM_CPUS MC_CPUS; do
    [[ -n ${!v:-} ]] && ENVS+=("$v=${!v}")
done
arm() {  # arm <slot> <after> <reactivity.sh args...>
    local slot=$1 after=$2; shift 2
    if [[ -f $S/$slot.done ]]; then echo "$slot: done, skipped"; return; fi
    if tmux list-windows -t jev -F '#W' | grep -qx "$slot"; then echo "$slot: window exists, skipped"; return; fi
    scripts/tmux_run.sh "$slot" scripts/slot_run.sh "$slot" --after "$after" -- env "${ENVS[@]}" scripts/reactivity.sh "$@"
}
arm reactivity-v1-qwen-pdm p5v1-index v1-qwen pdm
arm reactivity-v1-op-pdm p5v1-index v1-op pdm
arm reactivity-v1-qwen-ba reactivity-v1-qwen-pdm v1-qwen ba
arm reactivity-v1-op-ba reactivity-v1-op-pdm v1-op ba
arm reactivity-v1-exam-pdm reactivity-v1-qwen-pdm,reactivity-v1-op-pdm v1-exam pdm
arm reactivity-v1-mc-pdm reactivity-v1-qwen-pdm,reactivity-v1-op-pdm v1-mc pdm
arm reactivity-v1-exam-ba reactivity-v1-qwen-ba,reactivity-v1-op-ba v1-exam ba
arm reactivity-v1-mc-ba reactivity-v1-qwen-ba,reactivity-v1-op-ba v1-mc ba
arm reactivity-v1-final reactivity-v1-exam-pdm,reactivity-v1-mc-pdm,reactivity-v1-exam-ba,reactivity-v1-mc-ba v1-final
echo "$(date '+%Y-%m-%d %H:%M') [REACTIVITY/V1] chain armed after p5v1-index: reactivity-v1-{qwen,op,exam,mc}-{pdm,ba}, reactivity-v1-final (scripts/reactivity_v1_arm.sh)" \
    >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
