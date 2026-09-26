#!/usr/bin/env bash
# Top-10 T3 batch: one scripts/top10_t3_gen.sh chain per GPU over runs/top10_t3/ids.txt, all sharing runs/top10_t3/gen
# (O_EXCL route claims split the work), server blocks 200 + 50 j. Touches runs/top10_t3/gen.finished at the end.
#   scripts/top10_t3_batch.sh <workers per gpu> <gpu>:<cpu list> [<gpu>:<cpu list> ...]
# Records every chain PID in runs/top10_t3/gen/pids.txt; stop with scripts/top10_t3_stop.sh (those PIDs, their
# descendants and the CARLA servers named in gen/servers/*.pid, nothing matched by name).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/top10_t3
w=$1; shift
mkdir -p "$R/gen"
rm -f "$R/gen.finished"
pids=()
j=0
for spec in "$@"; do
    g=${spec%%:*} cpus=${spec#*:}
    scripts/top10_t3_gen.sh "$g" "$w" $(( 200 + 50 * j )) "$cpus" "@$R/ids.txt" "$R/gen" "$R/agent.json" \
        > "$R/gen/chain-gpu$g-$(date +%m%d-%H%M).log" 2>&1 &
    pids+=($!)
    echo "$(date '+%F %T') chain gpu $g pid $! cpus $cpus block $(( 200 + 50 * j ))" | tee -a "$R/gen/pids.txt"
    j=$(( j + 1 ))
    sleep 30
done
for p in "${pids[@]}"; do wait "$p"; done
left=$(python3 -c "
import json,os
n=json.load(open('$R/need.json')); print(sum(1 for r in n if not os.path.exists('$R/gen/done/%s.json'%r)))")
echo "$(date '+%F %T') all chains ended; $left worlds not done"
touch "$R/gen.finished"
