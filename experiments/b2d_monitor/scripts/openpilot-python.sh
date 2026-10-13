#!/usr/bin/env bash
# envs/openpilot's python with tfv6's CUDA 12 / cuDNN 9 wheels on LD_LIBRARY_PATH (the op_arb_server ONNX CUDA backend). usage: openpilot-python.sh <args>
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export LD_LIBRARY_PATH=$(printf '%s:' $DATA_DIR/envs/tfv6/lib/python3.10/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}
exec $DATA_DIR/envs/openpilot/bin/python "$@"
