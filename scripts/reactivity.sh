#!/usr/bin/env bash
# todos/2026-09-25-reactivity-program.md on the shared box: 6 pinned cores (196-201; 12 = 196-207 until 16:10), nice 10, GPU from slot_run
# (CUDA_VISIBLE_DEVICES). Launch each mode through scripts/slot_run.sh as slot reactivity-<mode>.
set -uo pipefail
cd ~/data/jev-drive
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PIN=(taskset -c 196-201 nice -n 10)
OP=$DATA_DIR/envs/openpilot/bin/python
SUB=op_streams_vis
case $1 in
    # D0: `temporal`, `vision`, `hidden` on the P5 streams (same streams and renderer as experiment 1)
    d0-op)   "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 0/2 --workers 5 --arrays temporal vision hidden --out-sub $SUB & a=$!
             "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 1/2 --workers 5 --arrays temporal vision hidden --out-sub $SUB & b=$!
             wait $a; ra=$?; wait $b; rb=$?; (( ra == 0 && rb == 0 )) || exit 1
             "${PIN[@]}" .venv/bin/python -m jevdrive.p5_openpilot finalize --arrays temporal,vision,hidden --sub $SUB ;;
    d0-exam) "${PIN[@]}" .venv/bin/python -m jevdrive.p5_exam run --op cinque,lebowski --op-arrays temporal,vision,hidden --op-sub $SUB \
                 --heads-skip "op-cinque hidden" ;;   # deviation 6: a 16 384-d ridge is ~6 h of CPU eigh
    # M-C: dual-stream reaction head, pair / hard / uniform / single-stream arms (P5 v0 smoke)
    mc)      "${PIN[@]}" .venv/bin/python -m jevdrive.reactivity_mc ;;
    mc-wide) "${PIN[@]}" .venv/bin/python -m jevdrive.reactivity_mc --lams=-5,7 ;;   # post-hoc sensitivity
    *) echo "mode: d0-op | d0-exam | mc | mc-wide" >&2; exit 2 ;;
esac
