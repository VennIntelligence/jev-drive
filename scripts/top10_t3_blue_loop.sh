#!/usr/bin/env bash
# Top-10 T3: offline BLUE over the re-recorded worlds as they finish (scripts/top10_t3_blue.py; todo [T3]).
#   scripts/top10_t3_blue_loop.sh "<gpu list>" <workers per gpu> <cpu list>
# Each pass: plan every world b2d_run has marked done, run the workers (sharded, 2 CPUs each), skip worlds already
# written. Stops after a pass that finds nothing new once runs/top10_t3/gen.finished exists.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/top10_t3
read -ra G <<< "$1"
per=$2
IFS=, read -ra C <<< "$(python3 -c "
import sys
out=[]
for p in sys.argv[1].split(','):
    a,_,b=p.partition('-'); out+=range(int(a),int(b or a)+1)
print(','.join(map(str,out)))" "$3")"
n=$(( ${#G[@]} * per ))
export CARLA_ROOT=$DATA_DIR/third_party/carla/CARLA_0.9.15 HF_HUB_OFFLINE=1 SAVE_PATH=$R/blue_save OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$R/blue" "$R/blue_logs"
while :; do
    before=$(ls "$R/blue" | grep -c '\.json$')
    .venv/bin/python -m jevdrive.top10_t3 blue-plan
    pids=()
    for ((s = 0; s < n; s++)); do
        g=${G[$(( s % ${#G[@]} ))]}
        CUDA_VISIBLE_DEVICES=$g taskset -c "${C[$(( 2 * s % ${#C[@]} ))]},${C[$(( (2 * s + 1) % ${#C[@]} ))]}" \
            "$DATA_DIR/envs/blue/bin/python" scripts/top10_t3_blue.py --plan "$R/blue_plan.json" --out "$R/blue" \
            --shard "$s/$n" >> "$R/blue_logs/shard$s.log" 2>&1 &
        pids+=($!)
    done
    echo "$(date '+%F %T') pass: $n workers, pids ${pids[*]}" | tee -a "$R/blue_logs/pids.txt"
    for p in "${pids[@]}"; do wait "$p"; done
    after=$(ls "$R/blue" | grep -c '\.json$')
    echo "$(date '+%F %T') pass done: $before -> $after worlds"
    if [[ -f $R/gen.finished ]] && (( after == before )); then break; fi
    (( after == before )) && sleep 120
done
