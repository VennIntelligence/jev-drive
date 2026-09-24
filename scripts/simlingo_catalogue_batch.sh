#!/usr/bin/env bash
# Scored batch of the SimLingo catalogue experiment, design (a'): both arms, TM seed 1, all 220 routes
# (todos/2026-09-25-simlingo-catalogue). 10 workers, 5 per GPU, each arm split across both GPUs so neither arm
# gets a quieter card; runners of one arm share its out dir (b2d_run.py claims make that safe).
# Server indices 140-227 in four blocks of 22 (a worker that loses its slot moves up by the runner's worker count):
# RPC 9000-13350, TM 15000-19350. TM of index i sits on the RPC of index i+120, so staying inside 140-229 keeps clear
# of our own TM ports and of the TFv6 runner's 110-139.
# Usage (on the box, via slot_run.sh): scripts/simlingo_catalogue_batch.sh [seed]
set -uo pipefail
cd "$(dirname "$0")/.."
seed=${1:-1}
logs=${OUT_ROOT:-$DATA_DIR/runs/simlingo-catalogue}/batch-logs
mkdir -p "$logs"
# arm gpu server_index workers
plan=("official 0 140 3" "simlingo 0 162 2" "simlingo 1 184 3" "official 1 206 2")
pids=()
for p in "${plan[@]}"; do
  set -- $p
  scripts/simlingo_catalogue_run.sh "$1" "$seed" "$2" "$3" "$4" --stagger-s 30 \
    > "$logs/$1-g$2-seed$seed.log" 2>&1 &
  pids+=($!)
  sleep 90  # stagger the four runners' first server starts
done
rcs=()
for p in "${pids[@]}"; do wait "$p"; rcs+=($?); done
# b2d_run.py exits 1 when any route never finished (the known Town12/13 server segfaults do that), so judge the
# batch by coverage instead: fail only if an arm has fewer than 200 of 220 routes done.
root=${OUT_ROOT:-$DATA_DIR/runs/simlingo-catalogue}
ok=0
for arm in official simlingo; do
  n=$(ls "$root/$arm/seed$seed/done" 2>/dev/null | wc -l)
  echo "$arm seed$seed: $n/220 routes done; runner exit codes: ${rcs[*]}"
  (( n >= 200 )) || ok=1
done
exit $ok
