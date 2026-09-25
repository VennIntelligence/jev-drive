#!/usr/bin/env bash
# Wait until $DATA_DIR/runs/zeroshot-exam/gpu-plan.md gains, after its current last line, a line matching the
# extended regex $1. A script file on purpose: a `bash -c "... grep pattern ..."` waiter would match itself in pgrep-style
# checks (docs/long-runs.md). Usage: scripts/wait_plan_line.sh '\[CTL-P5\].*(end|GPU 2 released)'
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
from=$(( $(wc -l < "$PLAN") + 1 ))
echo "waiting for /$1/ in $PLAN from line $from"
until tail -n +"$from" "$PLAN" | grep -qE "$1"; do sleep 60; done
tail -n +"$from" "$PLAN" | grep -E "$1" | head -1
