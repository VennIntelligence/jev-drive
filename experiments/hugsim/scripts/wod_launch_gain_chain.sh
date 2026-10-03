#!/usr/bin/env bash
# CPU chain: Cinque on the first $1 events, then it_dw3-s0 on the first $2, 90 shards x 2 threads. Output $DATA_DIR/runs/hugsim-wodgain/. Resumable; DONE when finished.
set -u
N1=${1:-450}; N2=${2:-270}; SH=${3:-90}
O=$DATA_DIR/runs/hugsim-wodgain; mkdir -p $O; cd "$(dirname "$0")/../../.."
PY=$DATA_DIR/envs/openpilot/bin/python
for spec in "cinque $N1" "it_dw3-s0 $N2"; do
  set -- $spec
  pids=()
  for k in $(seq 0 $((SH-1))); do
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY experiments/hugsim/scripts/wod_launch_gain.py $1 $O/$1.s$k.jsonl --shard $k --nshard $SH --threads 2 --limit $2 > $O/$1.s$k.log 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "$1 finished $(date)" >> $O/STATUS
done
touch $O/DONE
