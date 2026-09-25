#!/usr/bin/env bash
# Fusion diagnostics Q4: the SAM 3.1 batch over the three image lists, in the pre-registered order P5 -> nuScenes -> WOD
# (todos/2026-09-25-fusion-diagnostics.md). Each dataset writes part-*.parquet shards and skips finished ones, so a
# rerun resumes. Run inside tmux via slot_run.sh with --gpu; CORES pins the JPEG readers (default: 10 cores).
#   scripts/tmux_run.sh fd-sam-batch scripts/slot_run.sh fd-sam-batch --gpu 3 --vram-gb 30 -- scripts/fd_sam_chain.sh
set -euo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/.."
PY=$DATA_DIR/envs/sam3/bin/python
L=$DATA_DIR/processed/fusion_diag/lists
OUT=$DATA_DIR/processed/fusion_diag/sam
CORES=${CORES:-}
BATCH=${BATCH:-8} WORKERS=${WORKERS:-8}
for ds in ${DATASETS:-p5 nusc wod}; do
  echo "$(date '+%F %T') sam batch: $ds"
  ${CORES:+taskset -c $CORES} env OMP_NUM_THREADS=2 PYTHONPATH=. "$PY" -m jevdrive.sam_detect detect --list "$L/$ds.parquet" \
    --out "$OUT/$ds" --batch "$BATCH" --workers "$WORKERS" --tag "batch-$ds"
done
