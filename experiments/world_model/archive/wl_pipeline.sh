#!/usr/bin/env bash
# WL features, training and readouts after the fork generation (fc65452:todos/2026-09-28-wm-loop.md). One-shot chain:
#   scripts/tmux_run.sh wl-pipe experiments/world_model/archive/wl_pipeline.sh     env: GPU=0 CPUS=150-165 STEPS="index opspec op vjepa z outcomes
#                                                               train report" (default: all, in this order), ARMS, SEEDS
# Each step appends to runs/wl/pipe/STATUS.md; the chain ends with DONE or ERROR (the failing step) in runs/wl/pipe/.
# Resumable: vjepa / op skip what exists; train re-runs an arm x seed only without a model.pt. NO_DONE=1: no DONE at the
# end (a caller such as experiments/world_model/archive/wl_full.sh runs more steps after these).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/../../.."
NAME=${WL_NAME:-wl}
export WL_NAME=$NAME
[[ $NAME != wl ]] && export WL_RESULTS=${WL_RESULTS:-$DATA_DIR/runs/$NAME/results}
P=$DATA_DIR/runs/$NAME/pipe
mkdir -p "$P"
rm -f "$P/DONE" "$P/ERROR"
exec > >(tee -a "$P/log.txt") 2>&1
GPU=${GPU:-0} CPUS=${CPUS:-150-165}
read -ra STEPS <<< "${STEPS:-index opspec op vjepa z outcomes train report}"
read -ra ARMS <<< "${ARMS:-main worig intonly holdout}"
read -ra SEEDS <<< "${SEEDS:-0 1 2}"
PY=".venv/bin/python"
OPPY=$DATA_DIR/envs/openpilot/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
status() { echo "$(date '+%F %T') $*" | tee -a "$P/STATUS.md"; }
run() {  # run <name> <cmd...>
    status "start $1"
    local t0=$SECONDS
    shift
    if ! CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" "$@"; then status "FAILED after $(( SECONDS - t0 )) s"; echo "$*" > "$P/ERROR"; exit 1; fi
    status "ok ($(( SECONDS - t0 )) s)"
}
for s in "${STEPS[@]}"; do
    case $s in
        index|opspec|vjepa|z|prune) run "$s" $PY -m experiments.world_model.archive.wl_data "$s" ;;
        op) run op env P5_SET=${NAME}_gen OMP_NUM_THREADS=2 $OPPY scripts/p5_openpilot.py --models cinque --arrays temporal \
                --out-sub op_streams_vis --workers 10 ;;
        outcomes) run outcomes $PY -m experiments.world_model.archive.wl_model outcomes ;;
        train)
            for a in "${ARMS[@]}"; do for sd in "${SEEDS[@]}"; do
                if compgen -G "$DATA_DIR/runs/$NAME/model/$a/seed$sd/*/model.pt" > /dev/null; then status "skip $a seed $sd (done)"; continue; fi
                run "train $a seed $sd" $PY -m experiments.world_model.archive.wl_model train --arm "$a" --seed "$sd"
            done; done ;;
        report) run report $PY -m experiments.world_model.archive.wl_model report ;;
        *) status "unknown step $s"; echo "$s" > "$P/ERROR"; exit 1 ;;
    esac
done
[[ -n ${NO_DONE:-} ]] && { status "steps ${STEPS[*]} ok (NO_DONE: the caller goes on)"; exit 0; }
touch "$P/DONE"
status "DONE"
