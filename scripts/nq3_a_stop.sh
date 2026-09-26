#!/usr/bin/env bash
# Stop one lane-A generation dir (todos/2026-09-26-night-queue-3.md): scripts/p6_stop.sh on the PIDs recorded in
# <out>/pids.txt and their descendants, then every CARLA server b2d_run recorded in <out>/servers/carla-*.pid whose
# process is still a CarlaUE4 (its process group, by PID; nothing is matched by name). Run it BEFORE closing the tmux
# window: a runner killed first leaves its servers re-parented to init, out of reach of the descendant walk.
#   scripts/nq3_a_stop.sh <out>
set -uo pipefail
out=$1
cd "$(dirname "$0")/.."
[[ -f $out/pids.txt ]] && timeout 120 scripts/p6_stop.sh "$out" | tail -1
for f in "$out"/servers/carla-*.pid; do
    [[ -f $f ]] || continue
    p=$(cat "$f")
    if [[ -r /proc/$p/cmdline ]] && tr '\0' ' ' < /proc/$p/cmdline | grep -q CarlaUE4; then
        kill -TERM -- "-$p" 2>/dev/null; sleep 3; kill -KILL -- "-$p" 2>/dev/null
        echo "stopped server group $p ($f)"
    fi
done
