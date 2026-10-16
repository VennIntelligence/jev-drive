#!/usr/bin/env bash
# Start a long job in its own window of the box's tmux session `jev` (see docs/long-runs.md).
# Usage (on the box): scripts/tmux_run.sh <window-name> <command> [args ...]
#   e.g. scripts/tmux_run.sh probe experiments/probe_planner_v0/archive/probe_v0.sh --force
# The job runs in a bash login shell (so $DATA_DIR etc. are set) from the repo root.
# The window stays open after the job ends, so its output and exit code can be read.
# Temp files and compile caches go to the data disk (jevdrive/cl/disk.py, docs/remote-box.md "Disk").
set -euo pipefail
(( $# >= 2 )) || { sed -n '2,6p' "$0"; exit 1; }
name=$1; shift
repo=$(cd "$(dirname "$0")/.." && pwd)
tmux has-session -t jev 2>/dev/null || tmux new-session -d -s jev
if tmux list-windows -t jev -F '#W' | grep -qx "$name"; then
  echo "window jev:$name already exists; pick another name or close it first" >&2; exit 1
fi
cmd=$(printf '%q ' "$@")
# The job gets its own TMPDIR and shared compile caches on the data disk (jevdrive/cl/disk.py); the TMPDIR is removed
# when it ends and the system disk is pruned if low. If python3 cannot run that, the job still starts, with plain env.
tag="tmux-$name-$(date +%s)"
disk="python3 -m jevdrive.cl.disk"
tmux new-window -d -t jev -n "$name" \
  "bash -lc 'cd $repo && ( eval \"\$($disk env --tag $tag 2>/dev/null)\"; $cmd ); rc=\$?; $disk release --tag $tag --guard 2>/dev/null; echo; echo \"==> exited with \$rc at \$(date +%H:%M:%S)\"; exec bash'"
echo "started jev:$name -> tmux attach -t jev, then select window $name"
