#!/usr/bin/env bash
# HUGSIM zero-shot exam driver (todos/2026-09-25-hugsim-exam): starts the resident policy servers, runs the
# closed-loop jobs of one phase through scripts/hugsim/zs_run.py, stops the servers.
# Usage (inside tmux, through scripts/slot_run.sh):  scripts/hugsim/zs_exam.sh <phase> [out_root]
#   check-op, check-alp       adapter acceptance runs (checklist scenarios + two derived ones), incl. shadow mode
#   scored-op, scored-alp, scored-base   pre-registered scored runs (todos/2026-09-25-hugsim-exam/scored.txt)
# Every phase writes into its own out dir (default $DATA_DIR/runs/hugsim-exam/<phase>).
# GPUs: ALP_GPU (Alpamayo server + its simulators), OP_GPU (openpilot servers + simulators), BASE_GPU (baselines).
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../.."
phase=$1
OUT=${2:-$DATA_DIR/runs/hugsim-exam/$phase}
ALP_GPU=${ALP_GPU:-1}; OP_GPU=${OP_GPU:-0}; BASE_GPU=${BASE_GPU:-0}
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$OUT/servers"
pids=()
cleanup() { for p in "${pids[@]}"; do kill -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT

server() {  # name python gpu model
    local sock=$OUT/servers/$1.sock ready=$OUT/servers/$1.ready
    rm -f "$ready"
    CUDA_VISIBLE_DEVICES=$3 setsid "$2" -u scripts/hugsim_zs_server.py "$4" --socket "$sock" --ready-file "$ready" \
        > "$OUT/servers/$1.log" 2>&1 &
    pids+=($!)
}
wait_ready() {
    local n
    for n in "$@"; do
        until [[ -f $OUT/servers/$n.ready ]]; do
            sleep 5
            kill -0 "${pids[@]}" 2>/dev/null || { echo "a server died, see $OUT/servers"; exit 3; }
        done
        echo "$(date +%T) server $n ready"
    done
}
rm -f "$OUT"/FAILED-*
run() {  # agent controller gpu workers scenarios [opts] [tag]
    local opts=${6:-}
    [[ -n $opts ]] || opts='{}'
    $HPY scripts/hugsim/zs_run.py run --out "$OUT" --agent "$1" --controller "$2" --gpu "$3" --workers "$4" \
        --scenarios "$5" --socket "$OUT/servers/$1.sock" --opts "$opts" ${7:+--tag "$7"} || touch "$OUT/FAILED-$1-$2${7:+-$7}"
}

derive_check() {  # checklist-only copies of scene-0071-easy-00 (not benchmark scenarios)
    mkdir -p "$OUT/scen/nuscenes"
    local s71=$DATA_DIR/datasets/hugsim/scenarios/nuscenes/scene-0071-easy-00.yaml
    # a standstill start, and a sign / frame check started 0.8 m right of the route and yawed 6 deg right
    $HPY scripts/hugsim/zs_run.py derive "$s71" "$OUT/scen/nuscenes/scene-0071-standstill-00.yaml" mode=standstill_00 start_velo=0
    $HPY scripts/hugsim/zs_run.py derive "$s71" "$OUT/scen/nuscenes/scene-0071-offset-00.yaml" mode=offset_00 \
        'start_ab=[0.8, 0.0]' 'start_euler=[0.0, 6.0, 0.0]'
    L=$OUT/checklist.txt; T=$OUT/turns.txt; S=$OUT/standstill.txt
    { cat todos/2026-09-25-hugsim-exam/checklist.txt; echo "$OUT/scen/nuscenes/scene-0071-standstill-00.yaml"
      echo "$OUT/scen/nuscenes/scene-0071-offset-00.yaml"; } > "$L"
    grep -E 'scene-0383-easy|scene-0920-easy|waymo/|kitti360/' todos/2026-09-25-hugsim-exam/checklist.txt > "$T"
    echo "$OUT/scen/nuscenes/scene-0071-standstill-00.yaml" > "$S"
}
SHADOW='{"dump_every": 4, "engage_s": 1000, "oracle_vmax": 5}'
D='{"dump_every": 4}'
opserv() { server cinque "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" cinque
           server lebowski "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" lebowski; wait_ready cinque lebowski; }
alpserv() { server alpamayo "$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python" "$ALP_GPU" alpamayo; wait_ready alpamayo; }

case $phase in
debug)   # MODELS="cinque alpamayo" SCEN=nuscenes/scene-0383-easy-00.yaml: one scenario per model, official controller
    for m in ${MODELS:-cinque}; do
        if [[ $m == alpamayo ]]; then server $m "$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python" "$ALP_GPU" $m; g=$ALP_GPU
        else server $m "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" $m; g=$OP_GPU; fi
        wait_ready $m
        run $m official $g 1 "${SCEN:-nuscenes/scene-0383-easy-00.yaml}" '{"dump_every": 4}'
    done
    ;;
check-op)   # adapter acceptance, openpilot + references; two lanes (one simulator each)
    derive_check; opserv
    ( for c in official fixed; do run cinque $c "$OP_GPU" 1 $L "$D"; done
      run cinque fixed "$OP_GPU" 1 $L "$SHADOW" cinque-shadow
      run cinque fixed "$OP_GPU" 1 $T '{"dump_every": 4, "desire": false}' cinque-fixed-nodesire
      run cinque fixed "$OP_GPU" 1 $S '{"dump_every": 4, "engage_s": 5}' cinque-fixed-engage ) & a=$!
    ( for c in official fixed; do run lebowski $c "$OP_GPU" 1 $L "$D"; done
      run lebowski fixed "$OP_GPU" 1 $L "$SHADOW" lebowski-shadow
      run route fixed "$OP_GPU" 1 $L; run cv fixed "$OP_GPU" 1 $L; run route official "$OP_GPU" 1 $L ) & b=$!
    wait $a $b
    ;;
check-op2)  # checklist rerun of the model-driven runs after the straight-stop fix (shadow runs are not affected)
    derive_check; opserv
    ( for c in official fixed; do run cinque $c "$OP_GPU" 1 $L "$D"; done
      run cinque fixed "$OP_GPU" 1 $T '{"dump_every": 4, "desire": false}' cinque-fixed-nodesire
      run cinque fixed "$OP_GPU" 1 $S '{"dump_every": 4, "engage_s": 5}' cinque-fixed-engage ) & a=$!
    ( for c in official fixed; do run lebowski $c "$OP_GPU" 1 $L "$D"; done
      run lebowski fixed "$OP_GPU" 1 $T '{"dump_every": 4, "desire": false}' lebowski-fixed-nodesire ) & b=$!
    wait $a $b
    ;;
check-alp)  # adapter acceptance, Alpamayo; two simulators against one server
    derive_check; alpserv
    ( run alpamayo official "$ALP_GPU" 1 $L "$D"
      run alpamayo fixed "$ALP_GPU" 1 $T '{"dump_every": 4, "nav": false}' alpamayo-fixed-nonav ) & a=$!
    run alpamayo fixed "$ALP_GPU" 1 $L "$D"
    run alpamayo fixed "$ALP_GPU" 1 $L "$SHADOW" alpamayo-shadow
    wait $a
    ;;
scored-op)   # pre-registered scored runs (todos/2026-09-25-hugsim-exam/scored.txt), openpilot, 2 lanes
    L=todos/2026-09-25-hugsim-exam/scored.txt; opserv
    OPTS=${OP_OPTS:-'{}'}
    ( run cinque official "$OP_GPU" 1 $L "$OPTS"; run lebowski official "$OP_GPU" 1 $L "$OPTS" ) & a=$!
    run cinque fixed "$OP_GPU" 1 $L "$OPTS"; run lebowski fixed "$OP_GPU" 1 $L "$OPTS"
    wait $a
    ;;
scored-alp)
    L=todos/2026-09-25-hugsim-exam/scored.txt; alpserv
    OPTS=${ALP_OPTS:-'{}'}
    ( run alpamayo official "$ALP_GPU" ${ALP_WORKERS:-1} $L "$OPTS" ) & a=$!
    run alpamayo fixed "$ALP_GPU" ${ALP_WORKERS:-1} $L "$OPTS"
    wait $a
    ;;
scored-base)  # official LTF client and constant velocity on the scored list
    L=todos/2026-09-25-hugsim-exam/scored.txt
    ( run ltf official "$BASE_GPU" 1 $L; run cv official "$BASE_GPU" 1 $L ) & a=$!
    run ltf fixed "$BASE_GPU" 1 $L; run cv fixed "$BASE_GPU" 1 $L
    wait $a
    ;;
*) echo "unknown phase $phase"; exit 2 ;;
esac
fails=$(ls "$OUT"/FAILED-* 2>/dev/null | wc -l)
echo "$(date +%T) phase $phase finished, $fails runner(s) failed"
exit $(( fails > 0 ))
