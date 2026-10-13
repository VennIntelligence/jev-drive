#!/usr/bin/env bash
# openpilot policy server (op_arb_server, shipped Cinque or an arm ONNX) on card $MON_GPU for the monitor loop, then one warm-up
# client (op_warm.py: 20 plans, the 8.8 s first-inference stall off the route clock).
# usage: server.sh [name] [onnx]   name op (default, shipped Cinque) -> $V/srv/op.sock;
#        server.sh op-p2h10 $DATA_DIR/runs/op_parity/hugsim/onnx/pp-P2H10-F-s0.onnx  (P2H10-F-s0, needs bias.sh too)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
N=${1:-op}
rm -f $V/srv/$N.ready $V/srv/$N.sock
CUDA_VISIBLE_DEVICES=$MON_GPU PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 bash experiments/b2d_monitor/scripts/openpilot-python.sh \
  experiments/op_closed_loop/archive/op_arb_server.py cinque --pool 1 --backend cuda-iob --socket $V/srv/$N.sock \
  --ready-file $V/srv/$N.ready ${2:+--onnx $2} 2>&1 | tee -a $V/srv/$N.log &
until [[ -e $V/srv/$N.ready ]]; do sleep 2; done
$PY experiments/b2d_monitor/scripts/op_warm.py $V/srv/$N.sock 20 && echo "server $N warm: $(date "+%F %T")"
wait
