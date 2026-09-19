#!/usr/bin/env bash
# Start a long job in its own window of the box's tmux session `jev` (see docs/long-runs.md).
# Usage (on the box): scripts/tmux_run.sh <window-name> <command> [args ...]
#   e.g. scripts/tmux_run.sh probe scripts/probe_v0.sh --force
# The job runs in a bash login shell (so $DATA_DIR etc. are set) from the repo root.
# The window stays open after the job ends, so its output and exit code can be read.
set -euo pipefail
(( $# >= 2 )) || { sed -n '2,6p' "$0"; exit 1; }
name=$1; shift
repo=$(cd "$(dirname "$0")/.." && pwd)
tmux has-session -t jev 2>/dev/null || tmux new-session -d -s jev
if tmux list-windows -t jev -F '#W' | grep -qx "$name"; then
  echo "window jev:$name already exists; pick another name or close it first" >&2; exit 1
fi
cmd=$(printf '%q ' "$@")
tmux new-window -d -t jev -n "$name" \
  "bash -lc 'cd $repo && $cmd; echo; echo \"==> exited with \$? at \$(date +%H:%M:%S)\"; exec bash'"
echo "started jev:$name -> tmux attach -t jev, then select window $name"
