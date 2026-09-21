#!/usr/bin/env bash
# A consistent read-only snapshot of the Waymo processed tree, for jobs that run while feature extraction
# is still landing shards. Usage (on the box): scripts/snapshot_processed.sh <name>  -> prints the DATA_DIR
#   DATA_DIR=$(scripts/snapshot_processed.sh l0) python -m jevdrive.waymo_l0 ...
#
# `jevdrive.waymo.reindex` rewrites index.parquet, past.npy, future.npy and rater.parquet together every time
# a shard lands, and the `row` column of rater.parquet is a position into the index as it stood at that
# moment. A job that reads the index before a rebuild and the rater file after it silently pairs rater
# trajectories with the wrong frames, or crashes on an out-of-range row. Copying the four files, then
# checking they agree, pins one version of all of them. Everything else (the per-shard feature dirs, which
# are append-only and immutable once written, and the raw shards) is symlinked, not copied.
set -euo pipefail
(( $# == 1 )) || { sed -n '2,4p' "$0"; exit 1; }
src=$DATA_DIR/processed/waymo_e2e
snap=$DATA_DIR/snapshots/$1-$(date +%Y%m%d-%H%M%S)
py=${PYTHON:-/root/autodl-tmp/ujs/envs/jevdrive/bin/python}
mkdir -p "$snap/processed/waymo_e2e"
ln -sfn "$DATA_DIR/datasets" "$snap/datasets"
ln -sfn "$DATA_DIR/runs" "$snap/runs"
ln -sfn "$src/features" "$snap/processed/waymo_e2e/features"
for try in 1 2 3; do
  cp "$src"/index.parquet "$src"/past.npy "$src"/future.npy "$src"/rater.parquet "$snap/processed/waymo_e2e/"
  if DATA_DIR=$snap "$py" -c '
import sys, numpy as np, pandas as pd
from jevdrive import waymo
df, (past, future) = waymo.load_index(), waymo.load_ego()
rows = waymo.load_rater()[0]
assert len(df) == len(past) == len(future), (len(df), len(past), len(future))
assert rows.max() < len(df) and (df.split.to_numpy()[rows] == "val").all()
print(f"snapshot: {len(df)} frames, {len(rows)} rater rows", file=sys.stderr)' ; then
    echo "$snap"; exit 0
  fi
  echo "snapshot torn by a concurrent reindex, retrying ($try)" >&2
  sleep 5
done
echo "could not take a consistent snapshot after 3 tries" >&2; exit 1
