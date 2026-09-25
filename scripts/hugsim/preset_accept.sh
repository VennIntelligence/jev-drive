#!/usr/bin/env bash
# HUGSIM controller acceptance (todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md): the scene's logged
# trajectory (scripts/hugsim/preset_agent.py) as the plan, through each controller tree of scripts/hugsim/zs_run.py.
# Usage (inside tmux):  scripts/hugsim/preset_accept.sh <phase> [gpu]
#   smoke   plumbing check on one scene outside the acceptance set (ideal + official)
#   accept  the pre-registered set, one lane per controller (ideal, official, fixed), then preset_eval.py
#   validate  held-out scenes, controllers ideal, official, fixed and the candidate fixed2 (lqr-tracker-v2)
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../.."
phase=$1; GPU=${2:-0}
HPY=$DATA_DIR/envs/hugsim/bin/python
L=todos/2026-09-25-closed-loop-infra-acceptance
run() {  # controller scenarios out
    $HPY scripts/hugsim/zs_run.py run --out "$3" --agent preset --controller "$1" --gpu "$GPU" --workers 1 \
        --scenarios "$2" --tag "preset-$1" --timeout 1800 || touch "$3/FAILED-$1"
}
case $phase in
smoke)
    OUT=$DATA_DIR/runs/infra-accept/hugsim-smoke; mkdir -p "$OUT"; rm -f "$OUT"/FAILED-*
    for c in ideal official; do run $c nuscenes/scene-0051-easy-00.yaml "$OUT"; done
    $HPY scripts/hugsim/preset_eval.py "$OUT" --static <(echo nuscenes/scene-0051-easy-00.yaml)
    ;;
accept)
    OUT=$DATA_DIR/runs/infra-accept/hugsim; mkdir -p "$OUT"; rm -f "$OUT"/FAILED-*
    cat $L/hugsim-static.txt $L/hugsim-actor.txt > "$OUT/scenarios.txt"
    for c in ideal official fixed; do run $c "$OUT/scenarios.txt" "$OUT" & done
    wait
    $HPY scripts/hugsim/preset_eval.py "$OUT" --static $L/hugsim-static.txt --actor $L/hugsim-actor.txt > "$OUT/eval.log"
    ;;
validate)
    OUT=$DATA_DIR/runs/infra-accept/hugsim-val; mkdir -p "$OUT"; rm -f "$OUT"/FAILED-*
    cat $L/hugsim-val-static.txt $L/hugsim-val-actor.txt > "$OUT/scenarios.txt"
    for c in ideal official fixed fixed2; do run $c "$OUT/scenarios.txt" "$OUT" & done
    wait
    $HPY scripts/hugsim/preset_eval.py "$OUT" --static $L/hugsim-val-static.txt --actor $L/hugsim-val-actor.txt > "$OUT/eval.log"
    ;;
*) echo "unknown phase $phase"; exit 2 ;;
esac
fails=$(ls "$OUT"/FAILED-* 2>/dev/null | wc -l)
echo "$(date +%T) phase $phase finished, $fails lane(s) failed"
touch "$OUT/DONE"
exit $(( fails > 0 ))
