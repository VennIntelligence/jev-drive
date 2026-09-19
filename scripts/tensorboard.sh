#!/usr/bin/env bash
# Start our TensorBoard on the box, reading every run under $DATA_DIR/runs (see docs/long-runs.md).
# AutoDL's own TensorBoard (root, port 6007, /root/tf-logs) cannot be stopped or retargeted without root,
# so ours listens on 6006, the AutoDL custom-service port. Runs in tmux window jev:tb; safe to re-run.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
if tmux list-windows -t jev -F '#W' 2>/dev/null | grep -qx tb; then echo "jev:tb already running"; exit 0; fi
exec "$(dirname "$0")/tmux_run.sh" tb env UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" \
  uv run tensorboard --logdir "$DATA_DIR/runs" --host 0.0.0.0 --port 6006 --reload_interval 15
