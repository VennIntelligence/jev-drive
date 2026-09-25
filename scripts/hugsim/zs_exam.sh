#!/usr/bin/env bash
# HUGSIM zero-shot exam driver (todos/2026-09-25-hugsim-exam): starts the resident policy servers, runs the
# closed-loop jobs of one phase through scripts/hugsim/zs_run.py, stops the servers.
# Usage (inside tmux, through scripts/slot_run.sh):  scripts/hugsim/zs_exam.sh <phase> [out_root]
#   checklist   adapter acceptance runs: every model x both controllers on the checklist scenarios + references
#   scored      the pre-registered scored runs (scenario list from todos/2026-09-25-hugsim-exam/scored.txt)
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

case $phase in
debug)   # MODELS="cinque alpamayo" SCEN=nuscenes/scene-0383-easy-00.yaml: one scenario per model, official controller
    for m in ${MODELS:-cinque}; do
        if [[ $m == alpamayo ]]; then server $m "$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python" "$ALP_GPU" $m; g=$ALP_GPU
        else server $m "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" $m; g=$OP_GPU; fi
        wait_ready $m
        run $m official $g 1 "${SCEN:-nuscenes/scene-0383-easy-00.yaml}" '{"dump_every": 4}'
    done
    ;;
checklist)
    # checklist scenarios + a standstill copy of scene-0071-easy-00 (start_velo 0), not a benchmark scenario
    mkdir -p "$OUT/scen/nuscenes"
    sed -e 's/^start_velo: .*/start_velo: 0/' -e 's/^mode: .*/mode: standstill_00/' \
        "$DATA_DIR/datasets/hugsim/scenarios/nuscenes/scene-0071-easy-00.yaml" > "$OUT/scen/nuscenes/scene-0071-standstill-00.yaml"
    L=$OUT/checklist.txt
    cat todos/2026-09-25-hugsim-exam/checklist.txt > "$L"
    echo "$OUT/scen/nuscenes/scene-0071-standstill-00.yaml" >> "$L"
    server alpamayo "$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python" "$ALP_GPU" alpamayo
    server cinque "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" cinque
    server lebowski "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" lebowski
    wait_ready cinque lebowski
    D='{"dump_every": 4}'
    ( for c in official fixed; do run cinque $c "$OP_GPU" 1 $L "$D"; run lebowski $c "$OP_GPU" 1 $L "$D"; done
      run cinque official "$OP_GPU" 1 $L '{"dump_every": 4, "desire": false}' cinque-official-nodesire
      for c in official fixed; do for b in cv route; do run $b $c "$BASE_GPU" 1 $L; done; done ) &
    op=$!
    wait_ready alpamayo
    for c in official fixed; do run alpamayo $c "$ALP_GPU" 1 $L "$D"; done
    run alpamayo official "$ALP_GPU" 1 $L '{"dump_every": 4, "rigid": false}' alpamayo-official-shift
    wait $op
    ;;
scored)
    L=todos/2026-09-25-hugsim-exam/scored.txt
    server alpamayo "$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python" "$ALP_GPU" alpamayo
    server cinque "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" cinque
    server lebowski "$DATA_DIR/envs/openpilot/bin/python" "$OP_GPU" lebowski
    wait_ready cinque lebowski
    ( for c in official fixed; do run cinque $c "$OP_GPU" 1 $L; done
      for c in official fixed; do run lebowski $c "$OP_GPU" 1 $L; done ) &
    op=$!
    ( for c in official fixed; do run ltf $c "$BASE_GPU" 1 $L; run cv $c "$BASE_GPU" 1 $L; done ) &
    base=$!
    wait_ready alpamayo
    ALP_OPTS=${ALP_OPTS:-'{}'}
    ( run alpamayo official "$ALP_GPU" 1 $L "$ALP_OPTS" ) & a1=$!
    run alpamayo fixed "$ALP_GPU" 1 $L "$ALP_OPTS"
    wait $a1 $op $base
    ;;
*) echo "unknown phase $phase"; exit 2 ;;
esac
fails=$(ls "$OUT"/FAILED-* 2>/dev/null | wc -l)
echo "$(date +%T) phase $phase finished, $fails runner(s) failed"
exit $(( fails > 0 ))
