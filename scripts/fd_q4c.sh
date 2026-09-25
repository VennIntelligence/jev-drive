#!/usr/bin/env bash
# Fusion diagnostics Q4c (todos/2026-09-25-fusion-diagnostics.md): openpilot lead outputs on the P5 streams, then the
# lead recall / distance-error score. GPU 3 only; 8 pinned cores (the fusion item's share before 19:00 CST).
# Usage (in tmux): scripts/fd_q4c.sh op | score
set -uo pipefail
cd ~/data/jev-drive
CORES=${FD_CORES:-198,199,202-207}
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PIN=(taskset -c "$CORES" nice -n 10)
OP=$DATA_DIR/envs/openpilot/bin/python
ARGS=(--arrays temporal lead lead_prob --out-sub op_streams_lead --key-prefix p5_ --workers 3)
case $1 in
    op)    "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 0/2 "${ARGS[@]}" & a=$!
           "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 1/2 "${ARGS[@]}" & b=$!
           wait $a; ra=$?; wait $b; rb=$?; (( ra == 0 && rb == 0 )) ;;
    score) "${PIN[@]}" .venv/bin/python -m jevdrive.fusion_q4c --workers 8 ;;
    *) echo "mode: op | score" >&2; exit 2 ;;
esac
