#!/usr/bin/env bash
# Watch a running B2D window for completion, then append a finish line to gpu-plan.md
# and stop the paired policy-server window. Companion to scripts/b2d_wait_and_run.sh's
# own finish-tracking, for when the run was (re)started by hand rather than by the waiter.
#
# Usage: scripts/tmux_run.sh <name> scripts/b2d_finish_watch.sh <run-window> <server-window> <out-dir>
set -euo pipefail
run_win=$1; server_win=$2; out=$3
plan=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md

while pane_pid=$(tmux list-panes -t "jev:$run_win" -F '#{pane_pid}' 2>/dev/null); do
    [ -n "$pane_pid" ] || break
    pgrep -P "$pane_pid" >/dev/null 2>&1 || break
    sleep 60
done

tmux kill-window -t "jev:$server_win" 2>/dev/null || true

summary="$out/summary.json"
if [ -f "$summary" ]; then
    finished=$(python3 -c "import json;print(json.load(open('$summary')).get('routes_finished','?'))" 2>/dev/null || echo "?")
else
    finished="?"
fi
echo "## $(date +%Y-%m-%d\ %H:%M) [B2D] Alpamayo full run finished (routes_finished=${finished}); see $summary, results in $out" >> "$plan"
echo "done"
