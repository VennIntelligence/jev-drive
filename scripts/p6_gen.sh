#!/usr/bin/env bash
# P6 v0 generation (todos/2026-09-26-night-queue-2.md N1): PDM-Lite drives every world of runs/p6/pairs.xml
# (jevdrive/p6.py build) under scripts/p5_pair_agent.py, one b2d_run chain per GPU, all chains sharing one --out.
#   scripts/tmux_run.sh p6-gen scripts/p6_gen.sh            (env: GPUS="0 1 2" WORKERS=6 CORES=3 ONLY=<id,id>)
# Chain j owns CARLA server indices BLOCK + 50j .. +49. Each chain takes WORKERS x CORES CPUs no live process is pinned
# to at its start. Resumable: re-run skips done/<id>.json and steals the claims of dead runners.
# Run dir runs/p6/gen: log.txt (this script), events.jsonl (b2d_run), chain-gpu<g>.log, gpus.txt.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/p6
OUT=${OUT:-$R/gen}
PY=$DATA_DIR/envs/carla/bin/python
read -ra G <<< "${GPUS:-0 1 2}"
W=${WORKERS:-6} CORES=${CORES:-3} BLOCK=${BLOCK:-800}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/log.txt") 2>&1
echo "$(date '+%F %T') p6-gen start: GPUs [${G[*]}], $W CARLA instances each, $CORES cores each, block $BLOCK, only=${ONLY:-all}"
echo "$(date '+%F %T') GPUs ${G[*]} x $W servers (server index $BLOCK..)" >> "$OUT/gpus.txt"

TFV6=$(python3 -c "import json;print(json.load(open('$DATA_DIR/runs/p5_pairs/agent_config.json'))['tfv6_model_dir'])")
# P5 v1's PDM-Lite recorder config (40 s after the trigger, 40 s standing, 70 s), plus props, PDM-Lite's registry and
# a stop 8 s after the ego has passed the scenario's actors.
echo "{\"tfv6_model_dir\": \"$TFV6\", \"save_threads\": 3, \"driver\": \"pdm_lite\", \"after_trigger_s\": 40.0, \"stuck_s\": 40.0, \"max_sim_s\": 70.0, \"record_props\": true, \"pass_stop_s\": 8.0}" > "$R/agent-p6.json"

free_cpus() {  # free_cpus <n>: n CPUs no live process is pinned to
    python3 - "$1" <<'EOF'
import os, sys
def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
n = int(sys.argv[1])
busy = set()
for d in os.listdir("/proc"):
    try:
        for line in open("/proc/%s/status" % d):
            if line.startswith("Cpus_allowed_list"):
                c = parse(line.split(":")[1].strip())
                if len(c) < 100:
                    busy |= set(c)
    except (OSError, ValueError):
        continue
mine = parse(open("/proc/self/status").read().split("Cpus_allowed_list:")[1].split()[0])
print(",".join(str(c) for c in [c for c in mine if c not in busy][:n]))
EOF
}

export B2D_RESEED_AFTER_BUILD=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=3 SAVE_PATH=$R/lead_save
export PYTHONPATH=$LEAD_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}

chain() {  # chain <j>
    local j=$1 g=${G[$1]} base=$(( BLOCK + $1 * 50 )) w=$W span cpus pass ids per=600 room
    while :; do       # container thread cap (pids.max 20480): ~600 threads per CARLA instance with its route client
        room=$(( (${PIDS_BUDGET:-19000} - $(cat /sys/fs/cgroup/pids.current)) / per ))
        (( room >= 1 )) && break
        echo "$(date +%T) gpu $g: waiting for thread room"; sleep 60
    done
    (( room < w )) && { echo "$(date +%T) gpu $g: thread cap allows $room of $w instances"; w=$room; }
    span=$(( 50 / w * w ))
    cpus=$(free_cpus $(( w * CORES )))
    echo "$(date +%T) chain gpu $g: $w instances, CPUs $cpus, server index $base-$(( base + span - 1 ))"
    for pass in 1 2; do
        ids=$(.venv/bin/python -m jevdrive.p6 ids --only "${ONLY:-}")
        [[ -z $ids ]] && break
        echo "$(date +%T) gpu $g pass $pass: $(tr ',' '\n' <<< "$ids" | wc -l) worlds left"
        CUDA_VISIBLE_DEVICES=$g BENCH2DRIVE_ROOT=$DATA_DIR/third_party/simlingo/Bench2Drive WORK_DIR=$DATA_DIR/third_party/simlingo \
            taskset -c "$cpus" "$PY" scripts/b2d_run.py --routes "$R/pairs.xml" --route-ids "$ids" --out "$OUT" \
            --workers "$w" --server-index "$base" --index-span "$span" --gpu-rank "$g" --tm-seed-from-id \
            --agent scripts/p5_pair_agent.py --agent-config "$R/agent-p6.json" --python "$DATA_DIR/envs/p5v1-pdm/bin/python" \
            --fast-copy --no-spectator --no-reap --max-attempts 2 --stagger-s 20 --client-threads 8
    done
    echo "$(date +%T) chain gpu $g end"
}

pids=()
for ((j = 0; j < ${#G[@]}; j++)); do
    chain "$j" > "$OUT/chain-gpu${G[$j]}.log" 2>&1 &
    pids+=($!)
    (( j + 1 < ${#G[@]} )) && sleep 45
done
for p in "${pids[@]}"; do wait "$p"; done
left=$(.venv/bin/python -m jevdrive.p6 ids --only "${ONLY:-}" | tr ',' '\n' | grep -c .)
echo "$(date '+%F %T') p6-gen end: $left worlds left"
(( left == 0 ))
