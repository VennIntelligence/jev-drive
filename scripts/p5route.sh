#!/usr/bin/env bash
# todos/2026-09-25-openpilot-temporal-p5-and-route.md on the shared box: 12 pinned cores (184-195), nice 10,
# GPU from slot_run (CUDA_VISIBLE_DEVICES). Launch each mode through scripts/slot_run.sh as slot p5route-<mode>.
set -uo pipefail
cd ~/data/jev-drive
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PIN=(taskset -c 184-195 nice -n 10)
OP=$DATA_DIR/envs/openpilot/bin/python
case $1 in
    check1)  "${PIN[@]}" $OP scripts/p5_openpilot.py --check 16 ;;
    op1)     "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 0/2 --workers 5 & a=$!
             "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 1/2 --workers 5 & b=$!
             wait $a; ra=$?; wait $b; rb=$?; (( ra == 0 && rb == 0 )) || exit 1
             "${PIN[@]}" .venv/bin/python -m jevdrive.p5_openpilot finalize ;;
    exam1)   "${PIN[@]}" .venv/bin/python -m jevdrive.p5_exam run --op cinque,lebowski ;;
    check2b) "${PIN[@]}" $OP scripts/drive_backbones_openpilot.py --split trainval --models cinque lebowski \
                 --workers 8 --limit 2 --out-sub op_check_nodesire ;;
    op2b)    pids=(); for i in 0 1 2; do
                 "${PIN[@]}" $OP scripts/drive_backbones_openpilot.py --split trainval --desire --models cinque lebowski \
                     --shard $i/3 --workers 4 & pids+=($!); done
             rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; (( rc == 0 )) || exit 1
             "${PIN[@]}" .venv/bin/python -m jevdrive.drive_backbones --steps finalize_op --models cinque,lebowski \
                 --split trainval_desire ;;
    heads2b) "${PIN[@]}" .venv/bin/python -m jevdrive.drive_backbones --steps heads_train --models cinque,lebowski \
                 --vram-gb 25 --feat-suffix _trainval_desire ;;
    *) echo "mode: check1 | op1 | exam1 | check2b | op2b | heads2b" >&2; exit 2 ;;
esac
