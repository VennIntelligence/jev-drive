#!/usr/bin/env bash
# NAVSIM zero-shot: remaining Alpamayo phases, sequenced to share the GPU with the other exams
# (todos/2026-09-24-zeroshot-exam/navsim.md, box file runs/zeroshot-exam/gpu-plan.md):
#   1. when the first navtest-nav shard (tmux nzs-alp0 / nzs-alp1) exits: no-nav 3000-token subset, one process
#   2. when that and the WOD ADE-extra job (wod_zeroshot_alpamayo.py) are both done: navhard nav, two processes
#   3. then touch runs/zeroshot-exam/navsim-alpamayo.done (Bench2Drive's waiter needs it)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python
plan=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
note() { echo "$(date '+%Y-%m-%d %H:%M') [NAVSIM] $*" | tee -a "$plan"; }
running() { pgrep -u "$USER" -f "navsim_zs_alpamayo.py run $1" >/dev/null; }
alp() { HF_ENDPOINT=https://hf-mirror.com "$PY" scripts/navsim_zs_alpamayo.py run --batch 8 --workers 6 "$@"; }

while running "--split navtest --shard 0/2" && running "--split navtest --shard 1/2"; do sleep 30; done
note "a navtest-nav shard finished; start no-nav subset (1 proc, ~37 GB). Not starting a 2nd proc until WOD ADE-extra is done."
alp --split navtest --variants nonav --subset nonav3k --shard 0/1 --tag main || exit 1
# the WOD waiter's own command line names its script too, so this also covers "queued but not started yet"
while running "--split navtest --shard" || pgrep -u "$USER" -f wod_zeroshot_alpamayo >/dev/null; do sleep 60; done
note "no-nav subset done and WOD Alpamayo gone; start navhard two-stage nav (2 procs, ~37 GB each)."
alp --split navhard_two_stage --variants nav --shard 0/2 --tag main & p0=$!
sleep 90
alp --split navhard_two_stage --variants nav --shard 1/2 --tag main & p1=$!
wait $p0 && wait $p1 || { note "navhard Alpamayo FAILED, see tmux nzs-chain"; exit 1; }
touch "$DATA_DIR/runs/zeroshot-exam/navsim-alpamayo.done"
note "all NAVSIM Alpamayo phases done; touched navsim-alpamayo.done. NAVSIM holds no Alpamayo on the GPU now."
