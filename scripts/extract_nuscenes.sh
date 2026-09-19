#!/usr/bin/env bash
# Extract nuScenes from the AutoDL public archives into $DATA_DIR/datasets/nuscenes.
# Usage:
#   scripts/extract_nuscenes.sh mini
#   scripts/extract_nuscenes.sh trainval [member ...]   (default member: samples/CAM_FRONT)
# e.g. scripts/extract_nuscenes.sh trainval samples/CAM_FRONT sweeps/CAM_FRONT
# gzip is single-threaded, so we run one tar per archive, all in parallel.
# Every blob is still read end to end, so expect this to be I/O bound. Run it in tmux.
set -euo pipefail

src=/autodl-pub/data/nuScenes/Fulldatasetv1.0
dst="${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}/datasets/nuscenes"
split="${1:-mini}"
members=("${@:2}")

extract() {  # <archive> [member ...]
  local t0=$SECONDS
  tar -xzf "$1" -C "$dst" "${@:2}"
  echo "==> $(basename "$1") done in $((SECONDS - t0))s"
}

mkdir -p "$dst"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT  # no orphan tars if one fails
case "$split" in
  mini)
    extract "$src/Mini/v1.0-mini.tgz" & pids+=($!) ;;
  trainval)
    (( ${#members[@]} )) || members=(samples/CAM_FRONT)
    echo "==> members: ${members[*]}"
    extract "$src/Trainval/v1.0-trainval_meta.tgz" & pids+=($!)
    for a in "$src"/Trainval/v1.0-trainval*_blobs.tgz; do
      extract "$a" "${members[@]}" & pids+=($!)
    done ;;
  *) echo "unknown split: $split" >&2; exit 1 ;;
esac

echo "==> ${#pids[@]} archives in parallel -> $dst"
for p in "${pids[@]}"; do wait "$p"; done  # set -e aborts on the first failed tar

echo "==> all done in ${SECONDS}s"
ls "$dst"
du -sh "$dst"
