#!/usr/bin/env bash
# Stop P6 generation cleanly (todos/2026-09-26-night-queue-2.md N1), touching only the PIDs scripts/p6_gen.sh recorded in
# <OUT>/pids.txt and their descendants (found by walking /proc parent links, never by name): the chain shells, the
# b2d_run runners (SIGINT, so the route in flight is cancelled and never gets done/<id>.json; a re-run redoes it), and
# the CARLA servers and route clients under them.
#   scripts/p6_stop.sh [OUT]          default $DATA_DIR/runs/p6/gen
set -uo pipefail
OUT=${1:-$DATA_DIR/runs/p6/gen}
F=$OUT/pids.txt
[[ -f $F ]] || { echo "no $F"; exit 1; }
alive() { kill -0 "$1" 2>/dev/null; }
descendants() {  # all live descendants of the given PIDs
    python3 - "$@" <<'PY'
import os, sys
kids = {}
for d in os.listdir("/proc"):
    if d.isdigit():
        try:
            pp = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(pp, []).append(int(d))
out, todo = [], [int(x) for x in sys.argv[1:]]
while todo:
    p = todo.pop()
    for c in kids.get(p, []):
        out.append(c); todo.append(c)
print(" ".join(map(str, out)))
PY
}
gen=$(awk '$1 == "gen" {print $2}' "$F"); chains=$(awk '$1 == "chain" {print $2}' "$F"); runners=$(awk '$1 == "runner" {print $2}' "$F")
tree=$(descendants $gen $chains $runners)          # snapshot before anything dies and gets re-parented to init
for p in $gen $chains; do alive "$p" && kill -TERM "$p"; done
for p in $runners; do alive "$p" && kill -INT "$p"; done
for _ in $(seq 30); do n=0; for p in $runners; do alive "$p" && n=$((n + 1)); done; (( n == 0 )) && break; sleep 2; done
for p in $runners $tree; do alive "$p" && kill -TERM "$p"; done
sleep 8
for p in $runners $tree; do alive "$p" && kill -KILL "$p"; done
left=0; for p in $gen $chains $runners $tree; do alive "$p" && left=$((left + 1)); done
echo "stopped; processes left: $left"
