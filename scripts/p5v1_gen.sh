#!/usr/bin/env bash
# P5 v1 bulk generation (todos/2026-09-25-reactivity-program/i1-p5v1.md). Run it as slot p5v1-gen:
#   scripts/tmux_run.sh p5v1-gen scripts/slot_run.sh p5v1-gen --after p5v1-wait-infra --not-before 21:30 -- scripts/p5v1_gen.sh
#
# 1. Waits up to CONFIRM_WAIT_MIN (30) minutes for $R/layout.confirmed: the owner edits $R/layout.env after the infra
#    agent's CARLA profiling reports, then touches layout.confirmed. Without it the defaults in layout.env are used.
# 2. Reads $R/layout.env and starts one runner chain per GPU. A chain drives, for each pass and each expert in $EXPERTS,
#    that expert's remaining variants through scripts/b2d_run.py, pinned with taskset to the GPU's CPU list; CARLA uses
#    -graphicsadapter=<gpu>, TFv6 CUDA_VISIBLE_DEVICES=<gpu>. All chains share one --out per expert (claims split the
#    routes), each in its own block of CARLA server indices inside 800-899 (b2d_run --index-span).
# Resumable: killed and re-run, it skips done/<id>.json and steals the claims of dead runners.
# Exit 0 when each expert has >= MIN_DONE (0.95) of its worlds, so the index slot runs on what exists.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/p5v1
PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
PY=$DATA_DIR/envs/carla/bin/python
note() { echo "$(date '+%Y-%m-%d %H:%M') [REACTIVITY/I1] $*" | tee -a "$PLAN"; }

for ((i = 0; i < ${CONFIRM_WAIT_MIN:-30} * 2; i++)); do [[ -f $R/layout.confirmed ]] && break; sleep 30; done
confirmed=$([[ -f $R/layout.confirmed ]] && echo "confirmed by the owner" || echo "NOT confirmed within ${CONFIRM_WAIT_MIN:-30} min, defaults")
# shellcheck source=/dev/null
source "$R/layout.env"
stamp=$(date +%m%d-%H%M)
cp "$R/layout.env" "$R/layout.used-$stamp.env"
read -ra G <<< "$GPUS"
read -ra W <<< "$WORKERS"
n=${#G[@]}
block=$(( 100 / n ))
note "p5v1-gen start, layout $confirmed ($R/layout.used-$stamp.env): GPUs [${GPUS}] workers [${WORKERS}] experts [${EXPERTS}]"

# Orphans of an earlier, killed invocation hold ports in our blocks; every runner below runs --no-reap because the
# chains share an --out and one runner's reaper would kill the other's servers.
"$PY" - "$R/gen-ba" "$R/gen-pdm" <<'EOF'
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, "scripts")
import b2d_run as B
for o in sys.argv[1:]:
    if Path(o).exists():
        B.Runner.reap_orphans(SimpleNamespace(out=Path(o), _external_servers=None,
                                              event=lambda kind, **kw: print(kind, kw, flush=True)))
EOF

export B2D_RESEED_AFTER_BUILD=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=${OMP_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_THREADS:-3} SAVE_PATH=$R/lead_save
export PYTHONPATH=$LEAD_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}
# SimLingo's evaluator reads $WORK_DIR/leaderboard/data/weather.xml; the official one ignores WORK_DIR.
tree() { [[ $1 == pdm ]] && echo "$DATA_DIR/third_party/simlingo/Bench2Drive" || echo "$DATA_DIR/third_party/Bench2Drive"; }

chain() {  # chain <slot j> : every pass, every expert, on GPU G[j]
    local j=$1 g=${G[$1]} w=${W[$1]} cpus_var=CPUS_${G[$1]} base=$(( 800 + $1 * block )) span pass e ids
    span=$(( block / w * w ))
    for pass in 1 2; do
        for e in $EXPERTS; do
            ids=$(.venv/bin/python -m jevdrive.p5v1 ids --expert "$e")
            [[ -z $ids ]] && continue
            echo "$(date +%T) gpu $g pass $pass $e: $(tr ',' '\n' <<< "$ids" | wc -l) worlds left, cpus ${!cpus_var}, index $base+$span"
            CUDA_VISIBLE_DEVICES=$g BENCH2DRIVE_ROOT=$(tree "$e") WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "${!cpus_var}" "$PY" scripts/b2d_run.py \
                --routes "$R/pairs.xml" --route-ids "$ids" --out "$R/gen-$e" --workers "$w" --server-index "$base" \
                --index-span "$span" --gpu-rank "$g" --tm-seed-from-id --agent scripts/p5_pair_agent.py \
                --agent-config "$R/agent-$e.json" --python "$DATA_DIR/envs/scout-tfv6/bin/python" --fast-copy \
                --no-spectator --no-reap --max-attempts 2 --stagger-s 20
        done
    done
}

pids=()
for ((j = 0; j < n; j++)); do
    chain "$j" > "$R/chain-gpu${G[$j]}-$stamp.log" 2>&1 &
    pids+=($!)
    sleep 60            # the chains' first server starts do not collide
done
for p in "${pids[@]}"; do wait "$p"; done

rc=0
for e in $EXPERTS; do
    left=$(.venv/bin/python -m jevdrive.p5v1 ids --expert "$e" | tr ',' '\n' | grep -c .)
    total=$(.venv/bin/python -c "from jevdrive import p5v1; print(len(p5v1.variants(p5v1.cases())))")
    note "p5v1-gen $e: $(( total - left )) / $total worlds done"
    (( left * 100 > total * (100 - ${MIN_DONE_PCT:-95}) )) && rc=1
done
exit $rc
