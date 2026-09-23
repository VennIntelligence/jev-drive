#!/usr/bin/env bash
# P5 prototype: zero-shot VLM meta-action on Waymo val, judged by the logged future
# (todos/2026-09-23-p5-vlm-metaaction-proto.md).
# Usage: scripts/waymo_p5vlm.sh --steps select,labels | --steps gen --variant main [--out <run dir>] | --steps report --run <run dir>
# Batch 3 keeps Qwen3-VL-4B under the vlm stream's 15 GB (batch 4 peaks at 15.3 GB allocated).
# The 32B FP8 checkpoint needs the hub once per run to resolve the finegrained-fp8 kernel's version tag
# (the `kernels` package refuses to do it offline): run it with HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com.
# Start it with: scripts/tmux_run.sh p5vlm scripts/waymo_p5vlm.sh [...]. Run outputs: docs/long-runs.md.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$DATA_DIR/envs/jevdrive}"
export PYTHONWARNINGS=ignore::FutureWarning HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
uv sync -q
exec uv run python -u -m jevdrive.waymo_p5vlm "$@"
