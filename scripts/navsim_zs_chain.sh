#!/usr/bin/env bash
# NAVSIM zero-shot: the remaining Alpamayo phases for one GPU process (todos/2026-09-24-zeroshot-exam/navsim.md).
# Work is split dynamically (claimed 32-token chunks), so any number of processes on any GPU finish together.
# Usage: scripts/navsim_zs_chain.sh <name> <gpu> [wait-for-pgrep-pattern]
#   navtest no-nav 3000-token subset -> navhard two-stage nav; each process that ends checks whether it was the last.
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1 gpu=$2 waitfor=${3:-}
export CUDA_VISIBLE_DEVICES=$gpu HF_ENDPOINT=https://hf-mirror.com
PY=$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python
plan=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
note() { echo "$(date '+%Y-%m-%d %H:%M') [NAVSIM] $*" | tee -a "$plan"; }
[[ -n $waitfor ]] && while pgrep -u "$USER" -f "$waitfor" >/dev/null; do sleep 30; done
alp() { "$PY" scripts/navsim_zs_alpamayo.py run --batch 8 --workers 8 --tag main --claim "$name" "$@"; }
alp --split navtest --variants nonav --subset nonav3k || exit 1
alp --split navhard_two_stage --variants nav || exit 1
note "Alpamayo process $name (GPU $gpu) finished; its slot is free."
if ! pgrep -u "$USER" -f "navsim_zs_alpamayo.py run" >/dev/null; then
  touch "$DATA_DIR/runs/zeroshot-exam/navsim-alpamayo.done"
  note "all NAVSIM Alpamayo phases done; touched navsim-alpamayo.done."
fi
