#!/usr/bin/env bash
# CPU scoring chain (tmux jev, scripts/tmux_run.sh): wait for the plans job, then v2 EPDMS of every model and of the shipped ONNX run.
#   experiments/op_parity/scripts/pp_score_chain.sh <plans pool log dir> <models ...>
set -uo pipefail
wait_dir=$1; shift
until [[ -e $wait_dir/DONE || -e $wait_dir/ERROR ]]; do sleep 15; done
[[ -e $wait_dir/ERROR ]] && { echo "plans job failed: $wait_dir/ERROR"; exit 1; }
export NAVSIM_THREADS=${NAVSIM_THREADS:-30} OPI_ROOT=op_lb
PY=$DATA_DIR/envs/op-train/bin/python
$PY experiments/op_parity/scripts/pp_eval.py score --models "$@" || exit 1
onnx=opi_lb_navtest_gimm-cinque__base
ls "$DATA_DIR"/runs/navsim/eval/v2_navtest_${onnx}/*/*.csv >/dev/null 2>&1 || \
  experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v2 navtest $onnx "$DATA_DIR/runs/op_lb/lb_navtest/preds/gimm-cinque__base.npz" \
    > "$DATA_DIR/runs/op_lb/lb_navtest/score_v2_$onnx.log" 2>&1
echo "scoring done $(date +%T)"
