#!/usr/bin/env bash
# Scored batch of the SimLingo catalogue experiment, design (a'): both arms, TM seed 1, all 220 routes
# (todos/2026-09-25-simlingo-catalogue). 10 workers, 5 per GPU, each arm split across both GPUs so neither arm
# gets a quieter card; runners of one arm share its out dir (b2d_run.py claims make that safe).
# Server indices 140-227 in four blocks of 22 (a worker that loses its slot moves up by the runner's worker count):
# RPC 9000-13350, TM 15000-19350. TM of index i sits on the RPC of index i+120, so staying inside 140-229 keeps clear
# of our own TM ports and of the TFv6 runner's 110-139.
# Usage (on the box, via slot_run.sh): scripts/simlingo_catalogue_batch.sh <seed> <main|extra>
#   main  6 workers, 3 per GPU (each arm: 2 on one card, 1 on the other); judges coverage at the end
#   extra 4 more workers, 1 per arm per GPU, joining the same out dirs (claims keep routes from running twice)
set -uo pipefail
cd "$(dirname "$0")/.."
seed=${1:-1} stage=${2:-main}
logs=${OUT_ROOT:-$DATA_DIR/runs/simlingo-catalogue}/batch-logs
mkdir -p "$logs"
# arm gpu server_index workers
# Index blocks 140-259 (a failing worker moves up by its runner's worker count); TM of index i = RPC of i+120.
if [[ $stage == main ]]; then plan=("official 0 140 2" "simlingo 0 160 1" "simlingo 1 180 2" "official 1 200 1")
else plan=("official 0 220 1" "simlingo 0 230 1" "simlingo 1 240 1" "official 1 250 1"); fi
pids=()
for i in "${!plan[@]}"; do
  set -- ${plan[$i]}
  # Only each arm's first runner in the main stage may reap orphans from the out dir; any later runner would kill
  # its live sibling's servers and routes (seen on the first start: one attempt per arm lost, retried).
  reap=--no-reap; [[ $stage == main ]] && (( i < 2 )) && reap=
  scripts/simlingo_catalogue_run.sh "$1" "$seed" "$2" "$3" "$4" --stagger-s 30 $reap \
    > "$logs/$stage-$1-g$2-seed$seed.log" 2>&1 &
  pids+=($!)
  sleep 90  # stagger the four runners' first server starts
done
rcs=()
for p in "${pids[@]}"; do wait "$p"; rcs+=($?); done
[[ $stage == main ]] || exit 0
root=${OUT_ROOT:-$DATA_DIR/runs/simlingo-catalogue}
# routes still claimed by the extra runners finish before coverage is judged (claims are released at route end)
for i in $(seq 1 360); do
  ls "$root"/*/seed"$seed"/claims/*.lock >/dev/null 2>&1 || break
  sleep 30
done
# b2d_run.py exits 1 when any route never finished (the known Town12/13 server segfaults do that), so judge the
# batch by coverage instead: fail only if an arm has fewer than 200 of 220 routes done.
ok=0
for arm in official simlingo; do
  n=$(ls "$root/$arm/seed$seed/done" 2>/dev/null | wc -l)
  echo "$arm seed$seed: $n/220 routes done; runner exit codes: ${rcs[*]}"
  (( n >= 200 )) || ok=1
done
exit $ok
