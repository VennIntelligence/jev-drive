#!/usr/bin/env bash
# op_parity bias server for an arm ONNX (ego features -> intent_bias; default P2H10-F-s0) on card $MON_GPU, envs/op-train.
# usage: bias.sh [tag] -> $V/srv/bias.sock
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
rm -f $V/srv/bias.ready $V/srv/bias.sock
CUDA_VISIBLE_DEVICES=$MON_GPU PYTHONUNBUFFERED=1 exec $DATA_DIR/envs/op-train/bin/python -u experiments/op_parity/scripts/pp_hugsim.py serve \
  --tag ${1:-P2H10-F-s0} --socket $V/srv/bias.sock --ready-file $V/srv/bias.ready 2>&1 | tee -a $V/srv/bias.log
