#!/usr/bin/env bash
# P5 v1 bulk generation (todos/2026-09-25-reactivity-program/i1-p5v1.md). Run it as slot p5v1-gen:
#   scripts/tmux_run.sh p5v1-gen scripts/slot_run.sh p5v1-gen --after p5v1-wait-infra -- scripts/p5v1_gen.sh
#
# Layout: $R/layout.env (our defaults), then $R/layout.infra.env (servers_per_gpu / cores_per_server from the infra
# agent's "[INFRA-PROFILE] layout ready" line, written by scripts/p5v1_wait_infra.sh) overrides WORKERS and
# CORES_PER_SERVER. One runner chain per GPU in $GPUS; a chain whose GATE_<gpu> is "i3" first waits for the I3 rendering
# on that card to finish and VRAM_GB_PER_SERVER x WORKERS to be free (i3_done below). Each chain takes WORKERS x CORES_PER_SERVER CPUs from its CPUS_<gpu> list,
# minus every CPU a live process is pinned to at that moment (so chains and other jobs never overlap), and drives, for
# each pass and each expert in $EXPERTS, that expert's remaining worlds through scripts/b2d_run.py under taskset.
# CARLA uses -graphicsadapter=<gpu>, TFv6 CUDA_VISIBLE_DEVICES=<gpu>. All chains share one --out per expert (claims split
# the routes); chain j owns CARLA server indices 800 + 50j .. +49 (b2d_run --index-span).
# Resumable: killed and re-run, it skips done/<id>.json and steals the claims of dead runners.
# Exit 0 when each expert has >= MIN_DONE_PCT (95) % of its worlds, so the index slot runs on what exists.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/p5v1
S=$DATA_DIR/runs/sched
PLAN=$DATA_DIR/runs/zeroshot-exam/gpu-plan.md
PY=$DATA_DIR/envs/carla/bin/python
note() { echo "$(date '+%Y-%m-%d %H:%M') [REACTIVITY/I1] $*" | tee -a "$PLAN"; }

# shellcheck source=/dev/null
source "$R/layout.env"
INFRA_SERVERS_PER_GPU="" INFRA_CORES_PER_SERVER=""
# shellcheck source=/dev/null
[[ -f $R/layout.infra.env ]] && source "$R/layout.infra.env"
read -ra G <<< "$GPUS"
read -ra W <<< "$WORKERS"
[[ -n $INFRA_SERVERS_PER_GPU ]] && for j in "${!W[@]}"; do W[j]=$INFRA_SERVERS_PER_GPU; done
[[ -n $INFRA_CORES_PER_SERVER ]] && CORES_PER_SERVER=$INFRA_CORES_PER_SERVER
stamp=$(date +%m%d-%H%M)
{ cat "$R/layout.env"; [[ -f $R/layout.infra.env ]] && cat "$R/layout.infra.env"; } > "$R/layout.used-$stamp.env"
note "p5v1-gen start: GPUs [${G[*]}], CARLA instances [${W[*]}], $CORES_PER_SERVER cores each, experts [$EXPERTS] ($R/layout.used-$stamp.env)"

# Recorder configs. BehaviorAgent: v0's (plus parallel JPEG, same bytes). PDM-Lite: records longer (todo, deviation 3):
# in profiling it waited at a red light for the whole 20 s v0 allows after the trigger; the analysis also cuts its
# frames back to v0's window (jevdrive/p5v1.py v0_window).
TFV6=$(python3 -c "import json;print(json.load(open('$DATA_DIR/runs/p5_pairs/agent_config.json'))['tfv6_model_dir'])")
echo "{\"tfv6_model_dir\": \"$TFV6\", \"save_threads\": 3}" > "$R/agent-ba.json"
echo "{\"tfv6_model_dir\": \"$TFV6\", \"save_threads\": 3, \"driver\": \"pdm_lite\", \"after_trigger_s\": 40.0, \"stuck_s\": 40.0, \"max_sim_s\": 70.0}" > "$R/agent-pdm.json"

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

free_cpus() {  # free_cpus <cpu list> <n>: the first n CPUs of the list that no live process is pinned to
    python3 - "$1" "$2" <<'EOF'
import os, sys
def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
want, n = parse(sys.argv[1]), int(sys.argv[2])
busy = set()
for d in os.listdir("/proc"):
    try:
        for line in open("/proc/%s/status" % d):
            if line.startswith("Cpus_allowed_list"):
                c = parse(line.split(":")[1].strip())
                if len(c) < 100:          # pinned; an unpinned process is allowed on every CPU of the box
                    busy |= set(c)
    except (OSError, ValueError):
        continue
print(",".join(str(c) for c in [c for c in want if c not in busy][:n]))
EOF
}

i3_done() {  # i3_done <gpu> <GB>: the I3 rendering on this card has finished: its named final sentinel, or (fallback)
    # at least one i3-*.done, no jev:i3-* window with a live child, and >= <GB> free on the card, 3 checks 5 min apart
    local g=$1 need=$(( $2 * 1024 )) ok=0 w p free
    while :; do
        if [[ -n ${I3_FINAL_SLOT:-} ]] && [[ -f $S/$I3_FINAL_SLOT.done || -f $S/$I3_FINAL_SLOT.failed ]]; then
            return 0
        fi
        # fallback, in case the named sentinel never appears
        live=0
        while read -r w p; do [[ $w == i3-* ]] && pgrep -P "$p" > /dev/null && live=1; done \
            < <(tmux list-windows -t jev -F '#W #{pane_pid}' 2>/dev/null)
        free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
        if compgen -G "$S/i3-*.done" > /dev/null && (( live == 0 && free >= need )); then ok=$(( ok + 1 )); else ok=0; fi
        (( ok >= 3 )) && return 0
        sleep 300
    done
}

export B2D_RESEED_AFTER_BUILD=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=${OMP_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_THREADS:-3} SAVE_PATH=$R/lead_save
export PYTHONPATH=$LEAD_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}
# BehaviorAgent: official tree and v0's recorder env. PDM-Lite: SimLingo's tree (its evaluator reads
# $WORK_DIR/leaderboard/data/weather.xml) and envs/p5v1-pdm (SimLingo's numpy 1.23, which PDM-Lite's ragged-array code needs).
tree() { [[ $1 == pdm ]] && echo "$DATA_DIR/third_party/simlingo/Bench2Drive" || echo "$DATA_DIR/third_party/Bench2Drive"; }
pyenv() { [[ $1 == pdm ]] && echo "$DATA_DIR/envs/p5v1-pdm/bin/python" || echo "$DATA_DIR/envs/scout-tfv6/bin/python"; }

chain() {  # chain <j>: GPU G[j], every pass, every expert
    local j=$1 g=${G[$1]} w=${W[$1]} list_var=CPUS_${G[$1]} gate_var=GATE_${G[$1]} base=$(( 800 + $1 * 50 )) span cpus pass e ids
    span=$(( 50 / w * w ))
    if [[ ${!gate_var:-} == i3 ]]; then
        echo "$(date +%T) gpu $g: waiting for I3 to finish on this card"
        i3_done "$g" $(( w * ${VRAM_GB_PER_SERVER:-11} ))
    fi
    local need=$(( w * ${VRAM_GB_PER_SERVER:-11} * 1024 ))
    until (( $(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits) >= need )); do
        echo "$(date +%T) gpu $g: waiting for $(( need / 1024 )) GB free"; sleep 120
    done
    cpus=$(free_cpus "${!list_var}" $(( w * CORES_PER_SERVER )))
    note "p5v1-gen chain gpu $g start: $w CARLA instances, CPUs $cpus, server index $base-$(( base + span - 1 ))"
    for pass in 1 2; do
        for e in $EXPERTS; do
            ids=$(.venv/bin/python -m jevdrive.p5v1 ids --expert "$e")
            [[ -z $ids ]] && continue
            echo "$(date +%T) gpu $g pass $pass $e: $(tr ',' '\n' <<< "$ids" | wc -l) worlds left"
            CUDA_VISIBLE_DEVICES=$g BENCH2DRIVE_ROOT=$(tree "$e") WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "$cpus" \
                "$PY" scripts/b2d_run.py --routes "$R/pairs.xml" --route-ids "$ids" --out "$R/gen-$e" --workers "$w" \
                --server-index "$base" --index-span "$span" --gpu-rank "$g" --tm-seed-from-id \
                --agent scripts/p5_pair_agent.py --agent-config "$R/agent-$e.json" --python "$(pyenv "$e")" \
                --fast-copy --no-spectator --no-reap --max-attempts 2 --stagger-s 20
        done
    done
    note "p5v1-gen chain gpu $g end"
}

pids=()
for ((j = 0; j < ${#G[@]}; j++)); do
    chain "$j" > "$R/chain-gpu${G[$j]}-$stamp.log" 2>&1 &
    pids+=($!)
    sleep 60            # the chains' first server starts do not collide
done
for p in "${pids[@]}"; do wait "$p"; done

rc=0
total=$(.venv/bin/python -c "from jevdrive import p5v1; print(len(p5v1.variants(p5v1.cases())))")
for e in $EXPERTS; do
    left=$(.venv/bin/python -m jevdrive.p5v1 ids --expert "$e" | tr ',' '\n' | grep -c .)
    note "p5v1-gen $e: $(( total - left )) / $total worlds done"
    (( left * 100 > total * (100 - ${MIN_DONE_PCT:-95}) )) && rc=1
done
exit $rc
