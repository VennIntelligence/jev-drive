#!/usr/bin/env bash
# Gate for the P5 v1 bulk generation (todos/2026-09-25-reactivity-program/i1-p5v1.md): return once the infra agent's
# CARLA harness profiling (todos/2026-09-25-closed-loop-infra-acceptance.md, slot line "[INFRA-PROFILE] slot start" in
# gpu-plan.md, windows jev:prof-*) has ended. It has no sentinel of its own, so any one of these counts as the end:
#   1. a sentinel $DATA_DIR/runs/sched/infra-prof*.done or infra-scale*.done (if the infra agent adds one);
#   2. an [INFRA-PROFILE] line in gpu-plan.md after the 15:25 start line that begins "[INFRA-PROFILE] (slot) end /
#      done / finished / complete / stopped" or says "profiling ended / done / finished / complete / stopped"
#      (a step that is done, e.g. "step t12 done", does not count);
#   3. no infra_scale.sh / b2d_scale.py process for 3 checks in a row, 5 minutes apart.
# Checks only start once the slot's --not-before (21:30) has passed, so a pause between two profiling steps in the
# afternoon cannot end the wait.
# Run it as slot p5v1-wait-infra (a script file, not bash -c: a pgrep pattern on our own command line would match us,
# docs/long-runs.md).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
S=$DATA_DIR/runs/sched
PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
quiet=0
while :; do
    if compgen -G "$S/infra-prof*.done" > /dev/null || compgen -G "$S/infra-scale*.done" > /dev/null; then
        echo "$(date +%T) infra profiling ended: sentinel $(ls "$S"/infra-prof*.done "$S"/infra-scale*.done 2>/dev/null | head -1)"
        exit 0
    fi
    if awk '/\[INFRA-PROFILE\] slot start/ {s = 1; next}
            s && tolower($0) ~ /\[infra-profile\] (slot )?(end|ended|done|finished|complete|stopped)|\[infra-profile\].*profiling (has )?(ended|done|finished|complete|stopped)/ {f = 1}
            END {exit !f}' "$PLAN"; then
        echo "$(date +%T) infra profiling ended: $(grep '\[INFRA-PROFILE\]' "$PLAN" | tail -1)"
        exit 0
    fi
    if pgrep -f 'infra_scale\.sh|b2d_scale\.py' > /dev/null; then quiet=0; else quiet=$(( quiet + 1 )); fi
    if (( quiet >= 3 )); then
        echo "$(date +%T) infra profiling ended: no infra_scale.sh / b2d_scale.py process for 3 checks (10 min)"
        exit 0
    fi
    sleep 300
done
