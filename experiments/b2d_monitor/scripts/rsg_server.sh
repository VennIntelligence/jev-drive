#!/usr/bin/env bash
# SH30 + ROUTE-MEM policy server for B2D (default RSG-L18-s0) on card $MON_GPU, envs/op-train; ready after ~30-80 s (compile warm-up).
# usage: rsg_server.sh [tag] [name] -> $V/srv/<name>.sock (name default rsg)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
T=${1:-RSG-L18-s0}; N=${2:-rsg}
rm -f $V/srv/$N.ready $V/srv/$N.sock
CUDA_VISIBLE_DEVICES=$MON_GPU PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 exec $DATA_DIR/envs/op-train/bin/python -u \
  experiments/b2d_monitor/scripts/rsg_server.py --tag $T --socket $V/srv/$N.sock --ready-file $V/srv/$N.ready 2>&1 | tee -a $V/srv/$N.log
