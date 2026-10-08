#!/usr/bin/env bash
# Fetch the public AlpaSim nuPlan-track data needed for the dev smoke: trajdata cache, MTGS scene
# configs and asset shards (default: part001 only). Each tarball is streamed in N ranges from
# hf-mirror.com, extracted and deleted. Idempotent (.done markers); writes STATUS / DONE / ERROR.
# Usage (box): scripts/tmux_run.sh alpasim-fetch experiments/alpasim/scripts/fetch_data.sh [shard ...]
set -euo pipefail
ROOT=${ALPASIM_NUPLAN_ROOT:-$DATA_DIR/datasets/alpasim_nuplan}
OUT=$DATA_DIR/runs/alpasim/fetch; DL=$ROOT/_dl; N=${JOBS:-8}
EP=https://hf-mirror.com/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track/resolve/main
mkdir -p "$ROOT" "$DL" "$OUT"; rm -f "$OUT/DONE" "$OUT/ERROR"
exec > >(tee -a "$OUT/log.txt") 2>&1
trap 'echo "failed at line $LINENO" > "$OUT/ERROR"' ERR
sz() { stat -c%s "$1" 2>/dev/null || echo 0; }

fetch() {  # <path in the dataset repo>
  local rel=$1 tag size chunk t0=$SECONDS i pids=()
  tag=$(tr / _ <<<"$rel")
  [[ -e $ROOT/.done.$tag ]] && { echo "skip $rel"; return; }
  size=$(curl -sIL --retry 5 "$EP/$rel" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
  chunk=$(( (size + N - 1) / N ))
  echo "fetch $rel: $size bytes in $N ranges" | tee "$OUT/STATUS"
  for i in $(seq 0 $((N - 1))); do
    (
      a=$((i * chunk)); b=$((a + chunk - 1)); ((b >= size)) && b=$((size - 1))
      f=$DL/$tag.$i; want=$((b - a + 1))
      until (( $(sz "$f") == want )); do
        curl -sL --retry 3 --speed-limit 20000 --speed-time 60 -r $((a + $(sz "$f")))-$b "$EP/$rel" >> "$f" || sleep 5
      done
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"   # a bare `wait` would also wait for the tee of the log redirect
  local dl=$((SECONDS - t0)); t0=$SECONDS
  echo "extract $rel" > "$OUT/STATUS"
  for i in $(seq 0 $((N - 1))); do cat "$DL/$tag.$i"; done | pigz -dc | tar -x -C "$ROOT"
  rm -f "$DL/$tag".*; touch "$ROOT/.done.$tag"
  printf '%s\t%s bytes\tdownload %ss (%s MB/s)\textract %ss\n' "$rel" "$size" "$dl" \
    $((size / 1000000 / (dl > 0 ? dl : 1))) $((SECONDS - t0)) | tee -a "$OUT/sizes.tsv"
}

fetch trajdata_cache/nuplan_test.tar.gz
fetch MTGS_asset/navtest/configs.tar.gz
for s in "${@:-part001}"; do fetch "MTGS_asset/navtest/assets/$s.tar.gz"; done
rmdir "$DL" 2>/dev/null || true
du -sh --apparent-size "$ROOT"/* "$ROOT"/navtest/* | tee -a "$OUT/sizes.tsv"
echo "assets: $(ls "$ROOT/navtest/assets" | wc -l) folders" | tee -a "$OUT/sizes.tsv"
date > "$OUT/DONE"; echo done > "$OUT/STATUS"
