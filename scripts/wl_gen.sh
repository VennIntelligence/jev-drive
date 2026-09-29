#!/usr/bin/env bash
# WL fork generation (todos/2026-09-28-wm-loop.md): the fork runs of one stage under scripts/wl_fork_agent.py, one
# b2d_run chain per GPU, every chain working through the sets in turn (claims make chains share a set safely).
#   scripts/tmux_run.sh wl-gen scripts/wl_gen.sh      env: STAGE=pilot1|pilot10|full SETS="ba p6 d2" TFV6=0|1 NORENDER=0|1
#                                                          PRE_CAMS=11 [ONE_GPU=g: only the row's GPU g (a GPU id), pilots]
# Cards, CARLA workers per card, server indices and cores come from the `wm-loop` row of $DATA_DIR/runs/sched/table.tsv
# (scripts/sch_table.py; GPU at position k uses indices idx0 + k*span ..); nothing here picks its own. The row must pass
# `sch_table.py check` before the chain starts. WORKERS=n caps the row's workers per card (pilots).
# Before each runner a chain waits until the card's CARLA servers (any lane, from the process table) + its workers
# <= CARD_CAP (6): the G lane drains a card only at route boundaries after runs/sched/demand/wm-loop.json appears.
# WL-2: WL_NAME=wl2 WL_LANE=wl2-gen ZYGOTE=1 (runs/wl2, the wl2-gen row, --zygote), forks from `python -m jevdrive.wl forks2`.
# Drain: touch $OUT/DRAIN -> runners take no new route (B2D_DRAIN_FILE), chains start nothing new.
# Needs runs/wl/{forks.parquet, jobs.json, forks-<set>.xml} (python -m jevdrive.wl forks). Out: runs/wl/gen/<set>/
# (b2d_run layout), log.txt and chain-gpu<g>.log in runs/wl/gen/. Resumable: re-run skips done/<id>.json.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
NAME=${WL_NAME:-wl} LANE=${WL_LANE:-wm-loop}
R=$DATA_DIR/runs/$NAME
export WL_NAME=$NAME
[[ $NAME != wl ]] && export WL_RESULTS=${WL_RESULTS:-$R/results}
OUT=${OUT:-$R/gen}
PY=$DATA_DIR/envs/carla/bin/python
STAGE=${STAGE:-pilot1}
read -ra S <<< "${SETS:-ba p6}"
eval "$(python3 - "$DATA_DIR/runs/sched/table.tsv" "$LANE" <<'PYEOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["lane"] == sys.argv[2]]
if not rows:
    print('echo "no %s row in table.tsv" >&2; exit 1' % sys.argv[2]); sys.exit()
r = rows[0]
g = [x for x in r["gpus"].split(",") if x not in ("", "-")]
if ":" in r["idx0"]:
    m = dict(x.split(":") for x in r["idx0"].split(","))
    idx = [int(m[x]) for x in g]
else:
    idx = [int(r["idx0"]) + k * int(r["idx_span"]) for k in range(len(g))]
print('G=(%s) IDX=(%s) ROW_W=%s SPAN=%s ROW_CPUS=%s' % (" ".join(g), " ".join(map(str, idx)), r["workers"], r["idx_span"], r["cpus"]))
PYEOF
)"
python3 scripts/sch_table.py check > /dev/null || { echo "sch_table.py check fails: not starting"; exit 1; }
NCH=${#G[@]}                              # the row's cores are split evenly over the row's cards
POS=($(seq 0 $(( NCH - 1 ))))
if [[ -n ${ONE_GPU:-} ]]; then
    k=-1; for i in "${!G[@]}"; do [[ ${G[$i]} == "$ONE_GPU" ]] && k=$i; done
    (( k >= 0 )) || { echo "GPU $ONE_GPU is not in the wm-loop row (${G[*]})"; exit 1; }
    G=("${G[$k]}"); IDX=("${IDX[$k]}"); POS=(0); NCH=1       # a single-card pilot gets all of the row's cores
fi
# IDX_OFFSET=n: this chain's servers are indices IDX + n .. (a second chain on the same card next to a running one; WORKERS=k)
for i in "${!IDX[@]}"; do IDX[$i]=$(( IDX[i] + ${IDX_OFFSET:-0} )); done
W=$(( ${WORKERS:-$ROW_W} < ROW_W ? ${WORKERS:-$ROW_W} : ROW_W ))
mkdir -p "$OUT"
echo "gen $$" >> "$OUT/pids.txt"
exec > >(tee -a "$OUT/log.txt") 2>&1
echo "$(date '+%F %T') wl-gen start: stage $STAGE, sets [${S[*]}], GPUs [${G[*]}] x $W (indices ${IDX[*]}, span $SPAN), cores $ROW_CPUS split over $NCH cards, TFV6=${TFV6:-0} NORENDER=${NORENDER:-0} OUT=$OUT"

tree() { [[ $1 != ba ]] && echo "$DATA_DIR/third_party/simlingo/Bench2Drive" || echo "$DATA_DIR/third_party/Bench2Drive"; }
pyenv() { [[ $1 != ba ]] && echo "$DATA_DIR/envs/p5v1-pdm/bin/python" || echo "$DATA_DIR/envs/scout-tfv6/bin/python"; }
srcagent() { [[ $1 != ba ]] && echo "$DATA_DIR/runs/p6/agent-p6.json" || echo "$DATA_DIR/runs/p5v1/agent-ba.json"; }
for s in "${S[@]}"; do   # the source recorder config + the WL job table (TFV6=0 drops the TFv6 shadow)
    python3 - "$(srcagent "$s")" "$OUT/agent-$s.json" "$R/jobs.json" "${TFV6:-0}" "${PRE_CAMS:-11}" "${NORENDER:-0}" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(wl_jobs=sys.argv[3], wl_pre_cams=int(sys.argv[5]), wl_norender=sys.argv[6] == "1")
if sys.argv[4] != "1":
    c["tfv6_model_dir"] = ""
json.dump(c, open(sys.argv[2], "w"))
PYEOF
done

row_cpus() {  # row_cpus <position k>: the k-th of NCH equal slices of the row's core list
    python3 - "$ROW_CPUS" "$1" "$NCH" <<'PYEOF'
import sys
out = []
for part in sys.argv[1].split(","):
    a, _, b = part.partition("-")
    out += range(int(a), int(b or a) + 1)
k, n = int(sys.argv[2]), int(sys.argv[3])
m = len(out) // n
print(",".join(map(str, out[k * m:(k + 1) * m])))
PYEOF
}

carla_on() {  # carla_on <gpu>: CARLA servers rendering on that card, any lane (-graphicsadapter = CUDA index here)
    python3 - "$1" <<'PYEOF'
import os, sys
flag, n = ("-graphicsadapter=" + sys.argv[1]).encode(), 0
for p in filter(str.isdigit, os.listdir("/proc")):
    try:
        a = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
    except OSError:
        continue
    n += b"CarlaUE4-Linux-Shipping" in a[0] and flag in a
print(n)
PYEOF
}

CARD_CAP=${CARD_CAP:-6}
export B2D_DRAIN_FILE=$OUT/DRAIN
export B2D_RESEED_AFTER_BUILD=1 B2D_CAPTURE_CRITERION_EVENTS=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 \
    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=3 SAVE_PATH=$R/lead_save
# THREAD_CAPS=1: BLAS / OpenMP pools of the route clients at 1 thread (the container default MKL_NUM_THREADS=175 gives a
# BehaviorAgent client ~215 threads, a PDM-Lite one 15-45); measured in todos/2026-09-29-wl2-prereg.md, "线程上限"
[[ ${THREAD_CAPS:-0} == 1 ]] && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=$LEAD_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}

chain() {  # chain <j>
    local j=$1 g=${G[$1]} base=${IDX[$1]} w=$W span=$SPAN cpus pass ids s per=${THREADS_PER_SERVER:-350} room
    while :; do       # container thread cap: thread-reduced CARLA servers + route clients
        room=$(( (${PIDS_BUDGET:-19000} - $(cat /sys/fs/cgroup/pids.current)) / per ))
        (( room >= 1 )) && break
        echo "$(date +%T) gpu $g: waiting for thread room"; sleep 60
    done
    (( room < w )) && { echo "$(date +%T) gpu $g: thread cap allows $room of $w instances"; w=$room; }
    cpus=$(row_cpus "${POS[$j]}")
    echo "$(date +%T) chain gpu $g: $w instances, CPUs $cpus, server index $base-$(( base + span - 1 ))"
    for pass in 1 2; do
        for s in "${S[@]}"; do
            ids=$(.venv/bin/python -m jevdrive.wl ids --stage "$STAGE" --set "$s" --out "$OUT")
            [[ -z $ids ]] && continue
            [[ -e $OUT/DRAIN ]] && { echo "$(date +%T) gpu $g: drained, starting nothing"; break 2; }
            local n0=-1 on
            while on=$(carla_on "$g"); (( on + w > CARD_CAP )); do
                (( on != n0 )) && echo "$(date +%T) gpu $g: $on CARLA on the card, waiting for room for $w (cap $CARD_CAP)"
                n0=$on; sleep 60
            done
            echo "$(date +%T) gpu $g pass $pass $s: $(tr ',' '\n' <<< "$ids" | wc -l) runs left"
            CUDA_VISIBLE_DEVICES=$g BENCH2DRIVE_ROOT=$(tree "$s") WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "$cpus" \
                "$PY" scripts/b2d_run.py --routes "$R/forks-$s.xml" --route-ids "$ids" --out "$OUT/$s" \
                --workers "$w" --server-index "$base" --index-span "$span" --gpu-rank "$g" --tm-seed-from-id \
                --agent scripts/wl_fork_agent.py --agent-config "$OUT/agent-$s.json" --python "$(pyenv "$s")" \
                --fast-copy --no-spectator --no-reap --max-attempts ${MAX_ATTEMPTS:-3} --stagger-s ${STAGGER_S:-30} --client-threads 8 --stall-s ${STALL_S:-600} $([[ ${ZYGOTE:-0} == 1 ]] && echo --zygote) &
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
