#!/usr/bin/env bash
# op-adapt r2 package D: YOLO26x-seg detections on every dataset (one process per GPU, part g/n), then PCA on WOD train
# and the token files. One-shot and resumable (finished chunks / token files are skipped). Signals in R2/det/lane/:
# STATUS (appended), DONE, ERROR.
#   scripts/tmux_run.sh d-det scripts/op_adapt_det_lane.sh [gpus=1,2,3,4] [cores=108,109] [datasets...]
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS=(${1:-1,2,3,4}); GPUS=(${GPUS[@]//,/ }); CORES=${2:-108,109}
DS=(${@:3}); [ ${#DS[@]} -gt 0 ] || DS=(wodtrain p5 wod nusc navtest navhard navtrain sim)   # wodtrain first: the PCA
R=$DATA_DIR/runs/op_adapt_r2/det; L=$R/lane; mkdir -p "$L"; rm -f "$L/DONE" "$L/ERROR"
UL=$DATA_DIR/envs/ultralytics/bin/python; OT=$DATA_DIR/envs/op-train/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
st() { echo "$(date '+%F %H:%M:%S') $*" | tee -a "$L/STATUS"; }
fail() { st "ERROR: $*"; echo "$*" > "$L/ERROR"; exit 1; }
for d in "${DS[@]}"; do
  until [ -f "$R/$d/cams.npz" ]; do sleep 30; done              # the list step writes cams.npz last
  st "detect $d on GPUs ${GPUS[*]}"
  pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} taskset -c "$CORES" "$UL" scripts/op_adapt_det.py detect "$d" --part "$i/${#GPUS[@]}" \
      --workers 1 > "$L/detect-$d-$i.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc = 0 ] || fail "detect $d (see $L/detect-$d-*.log)"
  st "detect $d done: $(ls "$R/$d/raw" | grep -vc tmp) chunks"
  if [ "$d" = wodtrain ] && [ ! -f "$R/pca.npz" ]; then
    taskset -c "$CORES" "$OT" scripts/op_adapt_det.py pca > "$L/pca.log" 2>&1 || fail pca; st "pca done"
  fi
  if [ -f "$R/pca.npz" ]; then
    for t in "${DS[@]}"; do                   # tokens of every detected dataset still without them
      [ -f "$L/tokens-$t.ok" ] || [ -z "$(ls "$R/$t/raw" 2>/dev/null)" ] || [ "$t" != "$d" -a ! -f "$L/detect-$t.ok" ] && continue
      taskset -c "$CORES" "$OT" scripts/op_adapt_det.py tokens "$t" > "$L/tokens-$t.log" 2>&1 || fail "tokens $t"
      touch "$L/tokens-$t.ok"; st "tokens $t done"
    done
  fi
  touch "$L/detect-$d.ok"
done
st "all done"; touch "$L/DONE"
