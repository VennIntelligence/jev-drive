#!/usr/bin/env bash
# M1 offline replay, one pool job: m1_replay.sh <dir> <stage>. Extracts the driver-side messages of <dir>/scenes_<stage>.txt from last
# night's 400-scene runs (c1_extract.py msgs), replays both drivers with the input swaps of m1_replay.py over <dir>/spec_<stage>.json,
# writes <dir>/replay_<stage>_{sh30,ap2}.pkl and <stage>_DONE / <stage>_ERROR, then removes the messages.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); d=$1; st=$2
export ALPASIM_SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
C0=$DATA_DIR/runs/alpasim/c0
rm -f "$d/${st}_DONE" "$d/${st}_ERROR"
fail() { echo "$1" > "$d/${st}_ERROR"; echo "error: $1" > "$d/STATUS"; exit 1; }
for drv in sh30 ap2; do
  run=$(ls -d $C0/$([[ $drv == sh30 ]] && echo sh30_400_rerun || echo ap2_400_r1)/*/ | tail -1)
  echo "$st extract $drv $(date +%H:%M:%S)" > "$d/STATUS"
  "$ALPASIM_SRC/.venv/bin/python" "$here/c1_extract.py" msgs --run "${run%/}" --scenes "$d/scenes_$st.txt" --out "$d/msgs_${st}_$drv" --jobs 6 || fail "extract $drv"
  echo "$st replay $drv $(date +%H:%M:%S)" > "$d/STATUS"
  "$DATA_DIR/envs/op-train/bin/python" "$here/m1_replay.py" --driver $drv --msgs "$d/msgs_${st}_$drv" --spec "$d/spec_$st.json" --out "$d/replay_${st}_$drv.pkl" || fail "replay $drv"
  rm -rf "$d/msgs_${st}_$drv"
done
touch "$d/${st}_DONE"; echo "$st done $(date +%H:%M:%S)" > "$d/STATUS"
