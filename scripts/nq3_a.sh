#!/usr/bin/env bash
# Night queue 3, lane A: the whole lane as one unattended chain (todos/2026-09-26-night-queue-3.md, Q1 / Q3, [A] entries).
#   scripts/tmux_run.sh nq3-a scripts/nq3_a.sh            (re-run to resume: finished steps are skipped by their DONE)
#
#   prep      recorder configs, v0 re-record list (lane C's exam worlds), v1 route pool / worlds (jevdrive/nq3_a.py)
#   q3smoke   recovery smoke, 10 worlds on GPU 3 -> research/results/nq3/q3/recovery_smoke.csv, PASS or FAIL
#   v0rr      Q1 v0 re-record, 522 worlds (BridgeDrive shadow + BLUE camera), then E1 against P6 v0
#   offline   BLUE and SimLingo (scripts/top10_t3_blue.py) on the v0 re-record's exam frames, in the background from here
#   v1        Q3 P6 v1: main worlds, then the recovery worlds if q3smoke passed; then the E1 re-drive of 5 % of the main
#             worlds, the v1 frame index for lane C, expert statistics; writes v1/DONE (GPUs 3-5 go to lane B)
#   judge     Q1 CARLA-rig examinees (BridgeDrive, BLUE, SimLingo) with lane C's rule-7 judge
# Resources (main, fixed): GPUs 3, 4, 5, <= 6 CARLA servers each (index 600-689), cores 0-59, --client-threads 8, new
# servers wait while pids.current > 17 000. A world that crashes is retried by b2d_run (3 attempts); a step whose
# failure rate exceeds 10 % or whose wall clock exceeds twice its estimate stops the chain with runs/nq3/a/ERROR.
# STATUS.md is rewritten every 10 min; events.jsonl per docs/long-runs.md. Stop by PID only: scripts/nq3_a_stop.sh <out>.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/nq3/a
mkdir -p "$R"
exec > >(tee -a "$R/log.txt") 2>&1
PY=.venv/bin/python
GPUS=(3 4 5) IDX=(600 630 660)
CPUS=(${A_CPUS:-0-15 16-31 32-47})          # generation chains; 48-59 hold the offline BLUE / SimLingo workers
OFF_CPUS=${A_OFF_CPUS:-48-59}
W=${A_WORKERS:-6}
EST_V0RR=${EST_V0RR:-2.0} EST_V1=${EST_V1:-5.5} EST_SMOKE=0.5 EST_E1=0.6
RES=research/results/nq3
echo "gen $$" >> "$R/pids.txt"

ev() { printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N)" "$1" "${2:+, $2}" >> "$R/events.jsonl"; }
done_step() { printf 'finished %s\nwall_h %s\n%s\n' "$(date '+%F %T')" "$2" "$3" > "$R/$1/DONE"; ev step_end "\"step\": \"$1\", \"wall_h\": $2"; }

status() {  # STATUS.md every 10 min: step, progress, ETA, GPUs, cores, threads
    while :; do
        {
            echo "# lane A status ($(date '+%F %T') CST)"
            if [[ -f $R/CURRENT ]]; then
                read -r step out ids t0 est < "$R/CURRENT"
                echo "- step **$step** ($out)"
                echo "- $($PY -m jevdrive.nq3_a status --gen "$out" --file "$ids" --t0 "$t0" --est-h "$est" 2>/dev/null)"
            fi
            [[ -f $R/offline/CURRENT ]] && echo "- offline BLUE / SimLingo: $(ls $R/v0rr/blue 2>/dev/null | grep -c json) / $(ls $R/v0rr/simlingo 2>/dev/null | grep -c json) worlds of $(cat $R/offline/CURRENT)"
            echo "- steps done: $(cd $R && ls -d */DONE 2>/dev/null | tr '\n' ' ')"
            echo "- GPUs 3-5: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n 4,6p | tr '\n' ';')"
            echo "- load $(cut -d' ' -f1-3 /proc/loadavg); pids.current $(cat /sys/fs/cgroup/pids.current) / 20480"
            echo "- CARLA servers on lane A's ports: $(ps -eo args | grep -c 'carla-rpc-port=3[2-6][0-9][0-9][0-9] ')"
        } > "$R/STATUS.md.tmp" && mv "$R/STATUS.md.tmp" "$R/STATUS.md"
        sleep 600
    done
}

fail() {  # fail <reason>: ERROR file, stop the running step, end the chain
    local out=""
    [[ -f $R/CURRENT ]] && read -r _ out _ _ _ < "$R/CURRENT"
    [[ -n $out && -d $out ]] && scripts/nq3_a_stop.sh "$out"
    {
        echo "# lane A ERROR ($(date '+%F %T') CST)"
        echo "$1"
        [[ -f $R/CURRENT ]] && echo "step: $(cat $R/CURRENT)"
        [[ -n $out ]] && echo "progress: $($PY -m jevdrive.nq3_a status --gen "$out" --file "$(cut -d' ' -f3 $R/CURRENT)" --t0 0 --est-h 0 2>/dev/null)"
        echo; echo '## last 50 lines of log.txt'; tail -50 "$R/log.txt"
    } > "$R/ERROR"
    ev error "\"reason\": \"$1\""
    exit 1
}

gen() {  # gen <step> <id file> <agent config> <routes xml> <out> <est h> <n gpus> [fatal=1]
    local step=$1 ids=$2 cfg=$3 routes=$4 out=$5 est=$6 ng=$7 fatal=${8:-1} t0 j left req pids=()
    mkdir -p "$out"
    t0=$(date +%s)
    echo "$step $out $ids $t0 $est" > "$R/CURRENT"
    ev step_start "\"step\": \"$step\", \"worlds\": $(tr ',' '\n' < "$ids" | grep -c .), \"est_h\": $est"
    for ((j = 0; j < ng; j++)); do
        scripts/nq3_gen.sh "${GPUS[$j]}" "$W" "${IDX[$j]}" "${CPUS[$j]}" "@$ids" "$out" "$cfg" "$routes" \
            >> "$out/chain-gpu${GPUS[$j]}.log" 2>&1 &
        pids+=($!)
        echo "chain $! gpu ${GPUS[$j]}" >> "$out/pids.txt"
        sleep 30
    done
    while :; do
        local alive=0
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        (( alive )) || break
        if (( $(date +%s) - t0 > $(python3 -c "print(int(2 * $est * 3600))") )); then
            fail "$step: wall clock above twice the estimate ($est h)"
        fi
        sleep 60
    done
    req=$(tr ',' '\n' < "$ids" | grep -c .)
    left=$($PY -m jevdrive.nq3_a ids --file "$ids" --gen "$out" | tr ',' '\n' | grep -c .)
    $PY -m jevdrive.nq3_a ids --file "$ids" --gen "$out" | tr ',' '\n' | grep . > "$out/skipped.txt"
    echo "$(date '+%F %T') $step: $((req - left)) / $req worlds done, $left skipped after retries"
    if python3 -c "import sys; sys.exit(0 if $left / max($req, 1) > 0.10 else 1)"; then
        (( fatal )) && fail "$step: $left of $req worlds failed after retries (> 10 %)"
        echo "$(date '+%F %T') $step: $left of $req worlds failed after retries (> 10 %), not fatal for this step"
    fi
    rm -f "$R/CURRENT"
    echo "$(( $(date +%s) - t0 ))" > "$out/wall_s"
}

hours() { python3 -c "print(round(($(date +%s) - $1) / 3600, 3))"; }

status &
echo "status $!" >> "$R/pids.txt"
trap 'kill $(jobs -p) 2>/dev/null' EXIT
ev start "\"pid\": $$"

# ---------------------------------------------------------------- prep
mkdir -p "$R/prep"
if [[ ! -f $R/prep/DONE ]]; then
    t=$(date +%s)
    $PY -m jevdrive.nq3_a configs && $PY -m jevdrive.nq3_a need-v0 && $PY -m jevdrive.nq3_a build-v1 || fail "prep failed"
    $PY -m jevdrive.nq3_a smoke-ids | tr -d '\n' > "$R/v1/ids_smoke.txt"
    $PY -m jevdrive.nq3_a e1-ids | tr -d '\n' > "$R/v1/ids_e1.txt"
    done_step prep "$(hours $t)" "v0rr/ids.txt, v1/pairs.xml, v1/ids_{main,rec,smoke,e1}.txt"
fi

# ---------------------------------------------------------------- q3smoke: recovery gate (GPU 3 only)
mkdir -p "$R/q3smoke"
if [[ ! -f $R/q3smoke/DONE ]]; then
    t=$(date +%s)
    W0=$W; W=5
    gen q3smoke "$R/v1/ids_smoke.txt" "$R/agent_v1.json" "$R/v1/pairs.xml" "$R/v1/gen" "$EST_SMOKE" 1 0
    W=$W0
    $PY -m jevdrive.nq3_a recovery --gen "$R/v1/gen" --file "$(cat $R/v1/ids_smoke.txt)" --out "$RES/q3/recovery_smoke.csv" \
        | tee "$R/q3smoke/result.txt"
    if $PY -c "
import pandas as pd, sys
d = pd.read_csv('$RES/q3/recovery_smoke.csv')
sys.exit(0 if len(d) and d.back_by_3s.mean() >= 0.80 else 1)"; then
        touch "$R/q3smoke/PASS"
    else
        touch "$R/q3smoke/FAIL"
    fi
    done_step q3smoke "$(hours $t)" "$(ls $R/q3smoke | grep -E 'PASS|FAIL')"
fi

# ---------------------------------------------------------------- v0rr: Q1 v0 re-record + E1
mkdir -p "$R/v0rr"
if [[ ! -f $R/v0rr/DONE ]]; then
    t=$(date +%s)
    gen v0rr "$R/v0rr/ids.txt" "$R/agent_v0rr.json" "$DATA_DIR/runs/p6/pairs.xml" "$R/v0rr/gen" "$EST_V0RR" 3
    $PY -m jevdrive.nq3_a check-det --gen "$R/v0rr/gen" --need "$R/v0rr/need.json" --out "$RES/q1/v0rr_e1.csv" | tail -3 \
        | tee "$R/v0rr/e1.txt"
    $PY -c "
import pandas as pd, sys
d = pd.read_csv('$RES/q1/v0rr_e1.csv')
d = d[~d.flow.astype(bool)]            # flow worlds are not reproducible even under the v0 recorder ([A] 17:35)
bad = (d.first_div_k.notna() | ~d.cam_grid_same.astype(bool)).mean()
sys.exit(1 if bad > 0.10 else 0)" || fail "v0rr: more than 10 % of the re-recorded non-flow worlds diverge from P6 v0 (E1)"
    done_step v0rr "$(hours $t)" "v0rr/gen, $RES/q1/v0rr_e1.csv"
fi

# ---------------------------------------------------------------- offline BLUE / SimLingo on the v0 re-record (background)
offline() {
    mkdir -p "$R/offline" "$R/v0rr/blue" "$R/v0rr/simlingo" "$R/v0rr/offline_logs"
    $PY -m jevdrive.nq3_a blue-plan --gen "$R/v0rr/gen" --out "$R/v0rr/blue_plan.json" --frames exam
    python3 -c "import json;print(len(json.load(open('$R/v0rr/blue_plan.json'))))" > "$R/offline/CURRENT"
    IFS=, read -ra C <<< "$(python3 -c "
import sys
out = []
for p in '$OFF_CPUS'.split(','):
    a, _, b = p.partition('-'); out += range(int(a), int(b or a) + 1)
print(','.join(map(str, out)))")"
    local s=0 pids=() n=6
    export CARLA_ROOT=$DATA_DIR/third_party/carla/CARLA_0.9.15 HF_HUB_OFFLINE=1 SAVE_PATH=$R/blue_save OMP_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
    for model in blue simlingo; do
        for ((i = 0; i < 3; i++)); do
            env CUDA_VISIBLE_DEVICES=${GPUS[$i]} taskset -c "${C[$(( 2 * s % ${#C[@]} ))]},${C[$(( (2 * s + 1) % ${#C[@]} ))]}" \
                "$DATA_DIR/envs/$model/bin/python" scripts/top10_t3_blue.py --plan "$R/v0rr/blue_plan.json" \
                --out "$R/v0rr/$model" --shard "$i/3" --model "$model" >> "$R/v0rr/offline_logs/$model$i.log" 2>&1 &
            pids+=($!)
            echo "offline $! $model gpu ${GPUS[$i]}" >> "$R/pids.txt"
            s=$((s + 1))
        done
    done
    for p in "${pids[@]}"; do wait "$p"; done
    touch "$R/offline/DONE"
}
if [[ ! -f $R/offline/DONE ]]; then
    offline &
    OFF=$!
    echo "offline-loop $OFF" >> "$R/pids.txt"
fi

# ---------------------------------------------------------------- v1: P6 v1 + E1 re-drive + frame index + stats
mkdir -p "$R/v1"
if [[ ! -f $R/v1/DONE ]]; then
    t=$(date +%s)
    ids="$R/v1/ids_run.txt"
    if [[ -f $R/q3smoke/PASS ]]; then
        echo "$(cat $R/v1/ids_main.txt),$(cat $R/v1/ids_rec.txt)" > "$ids"
    else
        cat "$R/v1/ids_main.txt" > "$ids"
    fi
    gen v1 "$ids" "$R/agent_v1.json" "$R/v1/pairs.xml" "$R/v1/gen" "$EST_V1" 3
    gen v1e1 "$R/v1/ids_e1.txt" "$R/agent_e1.json" "$R/v1/pairs.xml" "$R/v1/e1" "$EST_E1" 3
    $PY -m jevdrive.nq3_a check-det --gen "$R/v1/e1" --ref "$R/v1/gen" --file "$(cat $R/v1/ids_e1.txt)" \
        --out "$RES/q3/v1_e1.csv" | tail -2 | tee "$R/v1/e1.txt"
    $PY -m jevdrive.nq3_a v1-post || fail "v1: expert statistics / frame index failed"
    done_step v1 "$(hours $t)" "v1/gen, processed/carla_p6_v1, $RES/q3"
fi

# ---------------------------------------------------------------- judge: Q1 CARLA-rig examinees on the v0 exam
mkdir -p "$R/judge"
if [[ ! -f $R/judge/DONE ]]; then
    t=$(date +%s)
    [[ -n ${OFF:-} ]] && wait "$OFF"
    [[ -f $R/offline/DONE ]] || fail "offline BLUE / SimLingo did not finish"
    $PY -m jevdrive.nq3_a judge-rig --gen "$R/v0rr/gen" --e1 "$RES/q1/v0rr_e1.csv" \
        --blue "BLUE speed waypoints=$R/v0rr/blue,SimLingo speed waypoints=$R/v0rr/simlingo" || fail "judge failed"
    done_step judge "$(hours $t)" "$RES/q1/carla_rig_*"
fi
touch "$R/DONE"
ev end "\"ok\": true"
echo "$(date '+%F %T') lane A chain finished"
