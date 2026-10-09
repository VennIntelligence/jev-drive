#!/usr/bin/env bash
# C1 offline replay, one pool job: both drivers over the scenes of <dir>/replay_spec.json (messages already extracted by
# c1_extract.py msgs into <dir>/msgs_{sh30,ap2}). Writes <dir>/replay_{sh30,ap2}.pkl and REPLAY_DONE / REPLAY_ERROR; removes the messages.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); d=$1
export ALPASIM_SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
rm -f "$d/REPLAY_DONE" "$d/REPLAY_ERROR"
for drv in sh30 ap2; do
  echo "replay $drv $(date +%H:%M:%S)" > "$d/STATUS"
  "$DATA_DIR/envs/op-train/bin/python" "$here/c1_replay.py" --driver $drv --msgs "$d/msgs_$drv" --spec "$d/replay_spec.json" --out "$d/replay_$drv.pkl" \
    || { echo "replay $drv failed" > "$d/REPLAY_ERROR"; echo error > "$d/STATUS"; exit 1; }
done
rm -rf "$d/msgs_sh30" "$d/msgs_ap2"
touch "$d/REPLAY_DONE"; echo done > "$d/STATUS"
