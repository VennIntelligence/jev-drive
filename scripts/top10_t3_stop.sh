#!/usr/bin/env bash
# Stop the Top-10 T3 batch: only the chain PIDs recorded in runs/top10_t3/gen/pids.txt, their descendants, and the CARLA
# servers whose pid files are in runs/top10_t3/gen/servers/ (never by process name).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
R=$DATA_DIR/runs/top10_t3/gen
desc() { local p; for p in $(ps -o pid= --ppid "$1" 2>/dev/null); do desc "$p"; echo "$p"; done; }
all=()
for p in $(awk '{for(i=1;i<=NF;i++) if($i=="pid") print $(i+1)}' "$R/pids.txt" 2>/dev/null); do
    kill -0 "$p" 2>/dev/null && all+=($(desc "$p") "$p")
done
for f in "$R"/servers/*.pid; do
    [[ -f $f ]] || continue
    p=$(awk '{print $1}' "$f")
    # guard against a recycled pid: the recorded process must still be a CARLA server
    kill -0 "$p" 2>/dev/null && tr '\0' ' ' < "/proc/$p/cmdline" | grep -q CarlaUE4 && all+=($(desc "$p") "$p")
done
(( ${#all[@]} )) || { echo "nothing to stop"; exit 0; }
echo "stopping ${#all[@]} processes: ${all[*]}"
kill -TERM "${all[@]}" 2>/dev/null
sleep 10
kill -KILL "${all[@]}" 2>/dev/null
exit 0
