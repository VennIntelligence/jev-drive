#!/usr/bin/env bash
# WL fork generation (todos/2026-09-28-wm-loop.md): the fork runs of one stage under scripts/wl_fork_agent.py, one
# b2d_run chain per GPU, every chain working through the sets in turn (claims make chains share a set safely).
#   scripts/tmux_run.sh wl-gen scripts/wl_gen.sh      env: STAGE=pilot1|pilot10|full GPUS="1 2" WORKERS=6 CORES=3
#                                                          SETS="ba p6" TFV6=0|1 NORENDER=0|1 PRE_CAMS=11 BLOCK=1200
# Needs runs/wl/{forks.parquet, jobs.json, forks-<set>.xml} (python -m jevdrive.wl forks). Out: runs/wl/gen/<set>/
# (b2d_run layout), log.txt and chain-gpu<g>.log in runs/wl/gen/. Resumable: re-run skips done/<id>.json.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/wl
OUT=${OUT:-$R/gen}
PY=$DATA_DIR/envs/carla/bin/python
STAGE=${STAGE:-pilot1}
read -ra G <<< "${GPUS:-1}"
read -ra S <<< "${SETS:-ba p6}"
W=${WORKERS:-6} CORES=${CORES:-3} BLOCK=${BLOCK:-1200}
mkdir -p "$OUT"
echo "gen $$" >> "$OUT/pids.txt"
exec > >(tee -a "$OUT/log.txt") 2>&1
echo "$(date '+%F %T') wl-gen start: stage $STAGE, sets [${S[*]}], GPUs [${G[*]}] x $W, $CORES cores each, TFV6=${TFV6:-0} NORENDER=${NORENDER:-0} OUT=$OUT"

tree() { [[ $1 == p6 ]] && echo "$DATA_DIR/third_party/simlingo/Bench2Drive" || echo "$DATA_DIR/third_party/Bench2Drive"; }
pyenv() { [[ $1 == p6 ]] && echo "$DATA_DIR/envs/p5v1-pdm/bin/python" || echo "$DATA_DIR/envs/scout-tfv6/bin/python"; }
srcagent() { [[ $1 == p6 ]] && echo "$DATA_DIR/runs/p6/agent-p6.json" || echo "$DATA_DIR/runs/p5v1/agent-ba.json"; }
for s in "${S[@]}"; do   # the source recorder config + the WL job table (TFV6=0 drops the TFv6 shadow)
    python3 - "$(srcagent "$s")" "$R/agent-$s.json" "$R/jobs.json" "${TFV6:-0}" "${PRE_CAMS:-11}" "${NORENDER:-0}" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(wl_jobs=sys.argv[3], wl_pre_cams=int(sys.argv[5]), wl_norender=sys.argv[6] == "1")
if sys.argv[4] != "1":
    c["tfv6_model_dir"] = ""
json.dump(c, open(sys.argv[2], "w"))
PYEOF
done

free_cpus() {  # free_cpus <n>: n CPUs no live process is pinned to
    python3 - "$1" <<'PYEOF'
import os, sys
def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
n, busy = int(sys.argv[1]), set()
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
PYEOF
}

export B2D_RESEED_AFTER_BUILD=1 B2D_CAPTURE_CRITERION_EVENTS=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 \
    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=3 SAVE_PATH=$R/lead_save
export PYTHONPATH=$LEAD_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}

chain() {  # chain <j>
    local j=$1 g=${G[$1]} base=$(( BLOCK + $1 * 50 )) w=$W span cpus pass ids s per=${THREADS_PER_SERVER:-350} room
    while :; do       # container thread cap: thread-reduced CARLA servers + route clients
        room=$(( (${PIDS_BUDGET:-19000} - $(cat /sys/fs/cgroup/pids.current)) / per ))
        (( room >= 1 )) && break
        echo "$(date +%T) gpu $g: waiting for thread room"; sleep 60
    done
    (( room < w )) && { echo "$(date +%T) gpu $g: thread cap allows $room of $w instances"; w=$room; }
    span=$(( 50 / w * w ))
    cpus=$(free_cpus $(( w * CORES )))
    echo "$(date +%T) chain gpu $g: $w instances, CPUs $cpus, server index $base-$(( base + span - 1 ))"
    for pass in 1 2; do
        for s in "${S[@]}"; do
            ids=$(.venv/bin/python -m jevdrive.wl ids --stage "$STAGE" --set "$s" --out "$OUT")
            [[ -z $ids ]] && continue
            echo "$(date +%T) gpu $g pass $pass $s: $(tr ',' '\n' <<< "$ids" | wc -l) runs left"
            CUDA_VISIBLE_DEVICES=$g BENCH2DRIVE_ROOT=$(tree "$s") WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "$cpus" \
                "$PY" scripts/b2d_run.py --routes "$R/forks-$s.xml" --route-ids "$ids" --out "$OUT/$s" \
                --workers "$w" --server-index "$base" --index-span "$span" --gpu-rank "$g" --tm-seed-from-id \
                --agent scripts/wl_fork_agent.py --agent-config "$R/agent-$s.json" --python "$(pyenv "$s")" \
                --fast-copy --no-spectator --no-reap --max-attempts 2 --stagger-s 20 --client-threads 8 --stall-s ${STALL_S:-600} &
            echo "runner $! gpu $g $s" >> "$OUT/pids.txt"
            wait $!
        done
    done
    echo "$(date +%T) chain gpu $g end"
}

pids=()
for ((j = 0; j < ${#G[@]}; j++)); do
    chain "$j" > "$OUT/chain-gpu${G[$j]}.log" 2>&1 &
    pids+=($!)
    echo "chain $! gpu ${G[$j]}" >> "$OUT/pids.txt"
    (( j + 1 < ${#G[@]} )) && sleep 45
done
for p in "${pids[@]}"; do wait "$p"; done
echo "$(date '+%F %T') wl-gen end"
