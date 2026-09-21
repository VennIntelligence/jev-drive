#!/usr/bin/env bash
# How many headless CARLA servers fit on the box? Starts N servers, runs one benchmark client
# against each at the same time, prints every client's JSON line. See docs/carla.md.
# Usage: scripts/carla_parallel.sh <n> [carla_bench.py args ...]
set -euo pipefail
n=${1:?usage: carla_parallel.sh <n> [bench args ...]}; shift
here=$(cd "$(dirname "$0")" && pwd)
out=${CARLA_RUN_DIR:-$DATA_DIR/runs/carla}/parallel-$n
mkdir -p "$out"

for i in $(seq 0 $((n - 1))); do "$here/carla_server.sh" start "$i"; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$out/vram_before_clients.txt"

# Stagger the clients: loading N worlds at once starves them all and they time out.
# Bench2Drive's own multi-task script staggers too (sleep 5 between tasks).
for i in $(seq 0 $((n - 1))); do
  "$DATA_DIR/envs/carla/bin/python" "$here/carla_bench.py" \
    --port $((2000 + 4 * i)) --tag "n$n-i$i" "$@" >"$out/client-$i.log" 2>&1 &
  sleep 20
done
sleep 45 && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >"$out/gpu_under_load.txt" &
wait

grep -h '^{' "$out"/client-*.log || true
echo "--- gpu under load: $(cat "$out/gpu_under_load.txt" 2>/dev/null)"
for i in $(seq 0 $((n - 1))); do "$here/carla_server.sh" stop "$i"; done
