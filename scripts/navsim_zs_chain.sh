#!/usr/bin/env bash
# NAVSIM zero-shot: all Alpamayo phases for one of two GPU processes (todos/2026-09-24-zeroshot-exam/navsim.md).
# Each phase resumes from its JSON lines. Usage: scripts/navsim_zs_chain.sh <shard 0|1>
#   navtest nav (shard k/2) -> navtest no-nav 3000-token subset (k/2) -> navhard two-stage nav (k/2)
# The process that finishes last touches runs/zeroshot-exam/navsim-alpamayo.done.
set -euo pipefail
cd "$(dirname "$0")/.."
k=$1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_ENDPOINT=https://hf-mirror.com
PY=$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python
plan=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
note() { echo "$(date '+%Y-%m-%d %H:%M') [NAVSIM] $*" | tee -a "$plan"; }
alp() { "$PY" scripts/navsim_zs_alpamayo.py run --batch 8 --workers 8 --shard "$k/2" --tag main "$@"; }
alp --split navtest --variants nav
alp --split navtest --variants nonav --subset nonav3k
alp --split navhard_two_stage --variants nav
touch "$DATA_DIR/runs/zeroshot-exam/navsim-alpamayo.shard$k.done"
note "Alpamayo process $k finished all phases (GPU 0 slot free)."
if [[ -e $DATA_DIR/runs/zeroshot-exam/navsim-alpamayo.shard$((1 - k)).done ]]; then
  touch "$DATA_DIR/runs/zeroshot-exam/navsim-alpamayo.done"
  note "all NAVSIM Alpamayo phases done; touched navsim-alpamayo.done."
fi
