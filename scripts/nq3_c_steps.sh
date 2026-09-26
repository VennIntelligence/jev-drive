#!/usr/bin/env bash
# Night queue 3, lane C: the step bodies of scripts/nq3_c.sh (sourced; each step runs as `bash -c "source ...; <step>"`
# under the chain's timeout). Environment shared by every step.
: "${DATA_DIR:?}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
R=$DATA_DIR/runs/nq3/c
PY=$REPO/.venv/bin/python
OPPY=$DATA_DIR/envs/openpilot/bin/python
ULPY=$DATA_DIR/envs/ultralytics/bin/python
export CUDA_VISIBLE_DEVICES=6 P6=carla_p6

feats() {   # C1: Qwen (2 shards, cores 164-179) || V-JEPA 2 -> YOLO detect -> tokens (cores 172-179)
  local pids=() rc=0
  for i in 0 1; do     # CPU-bound (HF processor): both shards may use all 16 cores
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 164-179 \
      $PY -m jevdrive.nq3_feats qwen --shard $i/2 --batch 2 --workers 6 > "$R/feats/qwen$i.log" 2>&1 & pids+=($!)
  done
  (
    set -e
    [[ -f $DATA_DIR/processed/$P6/bb_vjepa2/mean.npy ]] || \
      OMP_NUM_THREADS=2 taskset -c 172-179 $PY -m jevdrive.nq3_feats vjepa --batch 64 --workers 6
    for i in 0 1; do
      OMP_NUM_THREADS=1 taskset -c $((172 + 4 * i))-$((175 + 4 * i)) $ULPY -m jevdrive.night2_n4 detect --part $i/2 \
        --images "$DATA_DIR/processed/$P6/nq3_images.parquet" --dst "$DATA_DIR/processed/$P6/nq3_dets" --workers 3 &
    done
    wait
    taskset -c 172-179 $PY -m jevdrive.nq3_feats tokens
  ) > "$R/feats/vjepa_yolo.log" 2>&1 & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -n 3 "$R"/feats/*.log
  return $rc
}

q1_op() {   # openpilot Cinque / Lebowski native plan on all 605 P6 streams (cores 164-179)
  P5_SET=$P6 OMP_NUM_THREADS=2 taskset -c 164-179 $OPPY scripts/p5_openpilot.py --arrays plan temporal \
    --out-sub op_streams_plan --workers 12
}

