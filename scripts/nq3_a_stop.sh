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
# route clients b2d_run recorded (attempts/<id>/<n>/route.pid) that outlived their runner, and their shadow processes
for f in "$out"/attempts/*/*/route.pid; do
    p=$(cat "$f" 2>/dev/null) || continue
    if [[ -r /proc/$p/cmdline ]] && tr '\0' ' ' < /proc/$p/cmdline | grep -q b2d_route.py; then
        kids=$(awk -v pp="$p" '$4 == pp {print $1}' /proc/[0-9]*/stat 2>/dev/null)
        kill -KILL $kids "$p" 2>/dev/null
        echo "stopped route client $p and its children $kids ($f)"
    fi
done
for f in "$out"/servers/carla-*.pid; do
    [[ -f $f ]] || continue
    p=$(cat "$f")
    if [[ -r /proc/$p/cmdline ]] && tr '\0' ' ' < /proc/$p/cmdline | grep -q CarlaUE4; then
        kill -TERM -- "-$p" 2>/dev/null; sleep 3; kill -KILL -- "-$p" 2>/dev/null
        echo "stopped server group $p ($f)"
    fi
done
