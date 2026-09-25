#!/usr/bin/env bash
# Fusion diagnostics Q4: the SAM 3.1 batch over the three image lists, in the pre-registered order P5 -> nuScenes -> WOD
# (todos/2026-09-25-fusion-diagnostics.md). Data-parallel: PROCS processes per listed GPU claim 500-image shards
# (atomic mkdir), so the faster card takes more; every shard writes part-<k>.parquet once, and a rerun resumes after
# clearing stale claims (claims without a parquet are removed at start). Run inside tmux:
#   scripts/tmux_run.sh fd-sam-batch env GPUS="0 3" PROCS="3 2" scripts/fd_sam_chain.sh
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/.."
PY=$DATA_DIR/envs/sam3/bin/python
L=$DATA_DIR/processed/fusion_diag/lists
OUT=$DATA_DIR/processed/fusion_diag/sam
read -ra G <<< "${GPUS:-0}"
read -ra N <<< "${PROCS:-3}"
WORKERS=${WORKERS:-4}
for ds in ${DATASETS:-p5 nusc wod}; do
  for c in "$OUT/$ds"/part-*.claim; do [[ -e $c && ! -e ${c%.claim}.parquet ]] && rmdir "$c"; done
done
run() {  # one process: every dataset in order, claiming shards
  local gpu=$1 i=$2
  for ds in ${DATASETS:-p5 nusc wod}; do
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 PYTHONPATH=. "$PY" -m jevdrive.sam_detect detect --list "$L/$ds.parquet" \
      --out "$OUT/$ds" --part claim --shard-size 500 --workers "$WORKERS" --tag "batch-$ds-g$gpu-$i" || return 1
  done
}
pids=()
for k in "${!G[@]}"; do
  for ((i = 0; i < ${N[$k]}; i++)); do run "${G[$k]}" "$i" & pids+=($!); done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for ds in ${DATASETS:-p5 nusc wod}; do
  echo "$ds: $(ls "$OUT/$ds"/part-*.parquet 2>/dev/null | wc -l) shards written"
done
exit $rc
