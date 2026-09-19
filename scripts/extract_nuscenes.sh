#!/usr/bin/env bash
# Extract nuScenes from the AutoDL public archives into $DATA_DIR/datasets/nuscenes.
# Usage: scripts/extract_nuscenes.sh [mini]   (only mini for now)
set -euo pipefail

split="${1:-mini}"
src=/autodl-pub/data/nuScenes/Fulldatasetv1.0
dst="${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}/datasets/nuscenes"

case "$split" in
  mini) archives=("$src/Mini/v1.0-mini.tgz") ;;
  *) echo "unknown split: $split" >&2; exit 1 ;;
esac

mkdir -p "$dst"
for a in "${archives[@]}"; do
  echo "==> $a -> $dst"
  tar -xzf "$a" -C "$dst" --checkpoint=20000 --checkpoint-action=echo='%T'
done

echo "==> done"
ls "$dst"
du -sh "$dst"
