#!/usr/bin/env bash
# Independent recheck of decisions 20's top s_ego decile claim (CPU only).
# Usage: scripts/tmux_run.sh topdec scripts/verify_top_decile.sh [--official /path/to/rater_feedback_utils.py]
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" PYTHONWARNINGS=ignore::FutureWarning
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=
uv sync -q
exec uv run python -u scripts/verify_top_decile.py "$@"
