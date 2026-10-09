#!/usr/bin/env bash
# M1 showcase extraction, one pool job: m1_show.sh <out dir> <scene list> <name>=<run dir>:<SH30_TAG> ...
# Per run: the rollout records (c1_extract.py logs: scenes that still have their rollout.asl), the CAM_F0 frames as delivered and the
# model's input frames (c1_replay.py with the run's checkpoint) of the listed scenes -> <out>/<name>_{logs,frames,replay}.pkl, SHOW_DONE.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); d=$1; scenes=$2; shift 2
export ALPASIM_SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$d"; rm -f "$d/SHOW_DONE" "$d/SHOW_ERROR"
fail() { echo "$1" > "$d/SHOW_ERROR"; exit 1; }
python3 -c "import json,sys; json.dump({s: {'frames': True} for s in open(sys.argv[1]).read().split()}, open(sys.argv[2], 'w'))" "$scenes" "$d/spec.json"
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; run=${rest%%:*}; tag=${rest##*:}
  A="$ALPASIM_SRC/.venv/bin/python $here/c1_extract.py"
  [[ -f $d/${name}_logs.pkl ]] || $A logs --run "$run" --out "$d/${name}_logs.pkl" --jobs 6 || fail "logs $name"
  $A frames --run "$run" --scenes "$scenes" --out "$d/${name}_frames.pkl" --jobs 6 || fail "frames $name"
  $A msgs --run "$run" --scenes "$scenes" --out "$d/msgs_$name" --jobs 6 || fail "msgs $name"
  SH30_TAG=$tag "$DATA_DIR/envs/op-train/bin/python" "$here/c1_replay.py" --driver sh30 --msgs "$d/msgs_$name" --spec "$d/spec.json" --out "$d/${name}_replay.pkl" || fail "replay $name"
  rm -rf "$d/msgs_$name"
done
touch "$d/SHOW_DONE"
