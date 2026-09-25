#!/usr/bin/env bash
# Controller acceptance on Bench2Drive (todos/2026-09-25-closed-loop-infra-acceptance/b2d-controllers.md).
# The PDM-Lite expert (scripts/b2d_expert_agent.py) drives the pre-registered routes and logs its track; each controller
# arm then drives the same routes with the expert's own future track as its plan (scripts/b2d_zeroshot_agent.py
# "replay"), at the cadence and with the controller the exams used. Everything runs in SimLingo's Bench2Drive copy
# (PDM-Lite needs its CarlaDataProvider.active_scenarios bookkeeping), TM seed 0, one CARLA server per worker.
# Run under scripts/slot_run.sh or scripts/tmux_run.sh; CARLA server indices 700-739.
#
#   scripts/infra_ctl_accept.sh expert <gpu>    expert a (5 workers, index 700-709)
#   scripts/infra_ctl_accept.sh arms <gpu> [arm:index ...]   expert b ("eb") and the controller arms in parallel, 1 worker
#                                               each; default eb:715 z2:718 z5:721 f2:724 f5:727 p1:730 p2:733
#   scripts/infra_ctl_accept.sh all <gpu>       both
# Re-running resumes (b2d_run skips done/<id>.json). Exit non-zero on infrastructure failure only.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
mode=$1 gpu=$2
S=$DATA_DIR/third_party/simlingo
OUT=${ACCEPT_OUT:-$DATA_DIR/runs/infra-accept/b2d-ctl}
ROUTES=${ACCEPT_ROUTES:-2390,24211,1711,2373,3564,1833,1852,1956,2668,4183,11381,1825,2084,2086,2091,2115,2286,24330,17563,26458}
C=$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json
export BENCH2DRIVE_ROOT=$S/Bench2Drive WORK_DIR=$S OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
XML=$S/leaderboard/data/bench2drive220.xml
mkdir -p "$OUT/cfg" "$OUT/expert-logs"

run() {  # run <name> <server index> <workers> [b2d_run args...]
    local name=$1 idx=$2 w=$3; shift 3
    "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --routes "$XML" --route-ids "$ROUTES" --workers "$w" \
        --server-index "$idx" --gpu-rank "$gpu" --tm-seed 0 --no-spectator --no-reap --max-attempts 2 \
        --stall-s 480 --route-timeout-s 3600 --out "$OUT/$name" "$@"
}
expert() {  # expert <tag> <index> <workers>
    run "expert-$1" "$2" "$3" --python "$DATA_DIR/envs/simlingo/bin/python" \
        --agent scripts/b2d_expert_agent.py --agent-config "expert+$1"
}
arm_cfg() {  # arm_cfg <arm> <plan_every> <controller> <extra keys>
    cat > "$OUT/cfg/$1.json" <<EOF
{"model": "alpamayo", "replay": "$OUT/expert-logs/{route}.jsonl", "plan_every": $2, "controller": "$3",
 "controller_preset": "carla", "controller_config": "$C", "seed": 0, "dump_every": 0$4}
EOF
}
collect() {  # the expert-a log of each route's last finished attempt -> expert-logs/<id>.jsonl
    local d rid
    for d in "$OUT"/expert-a/done/*.json; do
        rid=$(basename "$d" .json)
        a=$(python3 -c "import json;print(json.load(open('$d'))['attempt'])")
        cp "$OUT/expert-a/attempts/$rid/$a/expert.jsonl" "$OUT/expert-logs/$rid.jsonl" || return 1
    done
}

if [[ $mode == expert || $mode == all ]]; then
    expert a 700 5; rc=$?
    collect || { echo "expert logs missing"; exit 1; }
    echo "$(date +%T) expert a: runner exit $rc, $(ls "$OUT"/expert-logs | wc -l) route logs"
fi
if [[ $mode == arms || $mode == all ]]; then
    F1=', "plan_forward_only": true, "zoo_cadence": "plan"'
    arm_cfg z2 5 zoo_pid "$F1"                                   # Alpamayo exam (full220-alpamayo-zoopid-f1)
    arm_cfg z5 2 zoo_pid "$F1"                                   # openpilot exam cadence (5 Hz plans)
    arm_cfg f2 5 fixed ""                                        # the pre-registered fixed controller (full220-alpamayo)
    arm_cfg f5 2 fixed ""
    arm_cfg p1 5 zoo_pid "$F1"', "zoo_lateral": "fixed"'         # lateral fix P1
    arm_cfg p2 5 zoo_pid "$F1"', "zoo_lateral": "time", "zoo_aim_s": 1.5'   # lateral fix P2
    specs=("${@:3}")
    [[ $mode == all || ${#specs[@]} == 0 ]] && specs=(eb:715 z2:718 z5:721 f2:724 f5:727 p1:730 p2:733)
    pids=()
    for spec in "${specs[@]}"; do    # arm:index, 3 indices apart so a moved server cannot land on a neighbour's
        arm=${spec%%:*} i=${spec##*:}
        if [[ $arm == eb ]]; then
            expert b "$i" 1 & pids+=($!)
        else
            run "arm-$arm" "$i" 1 --agent scripts/b2d_zeroshot_agent.py --agent-config "$OUT/cfg/$arm.json" & pids+=($!)
        fi
        sleep 25          # stagger the server starts across runners
    done
    bad=0
    for p in "${pids[@]}"; do wait "$p" || { echo "runner $p exit $?"; bad=1; }; done
    exit $bad
fi
