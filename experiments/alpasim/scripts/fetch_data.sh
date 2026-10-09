#!/usr/bin/env bash
# Fetch the public AlpaSim nuPlan-track data: trajdata cache, MTGS scene configs and asset shards (default:
# part001 only; `all` = part001..015). Each tarball is streamed in JOBS ranges from hf-mirror.com, sha256-checked,
# extracted and deleted; PAR shards run at once. Resumable and idempotent (.done markers, range files keep their
# bytes). Writes STATUS / DONE / ERROR for the run and per item under $OUT/<item>/, and stops with ERROR before
# free disk would drop below MIN_FREE_GB.
# Usage (box): JOBS=8 PAR=4 scripts/tmux_run.sh alpasim-dl experiments/alpasim/scripts/fetch_data.sh all
set -euo pipefail
ROOT=${ALPASIM_NUPLAN_ROOT:-$DATA_DIR/datasets/alpasim_nuplan}
OUT=$DATA_DIR/runs/alpasim/fetch; DL=$ROOT/_dl; N=${JOBS:-8}; PAR=${PAR:-1}; MIN_FREE=$(( ${MIN_FREE_GB:-100} * 1000000000 ))
REPO=datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track
EP=https://hf-mirror.com/$REPO/resolve/main; API=https://hf-mirror.com/api/$REPO/tree/main
mkdir -p "$ROOT" "$DL" "$OUT"; rm -f "$OUT/DONE" "$OUT/ERROR"
exec > >(tee -a "$OUT/log.txt") 2>&1
trap 'echo "failed at line $LINENO" > "$OUT/ERROR"' ERR
sz() { stat -c%s "$1" 2>/dev/null || echo 0; }
free() { df -B1 --output=avail "$ROOT" | tail -1; }

fetch() {  # <path in the dataset repo> [retry]
  local rel=$1 tag size chunk t0=$SECONDS i pids=() S
  tag=$(tr / _ <<<"$rel"); S=$OUT/$(basename "$rel" .tar.gz); mkdir -p "$S"; rm -f "$S/ERROR"
  [[ -e $ROOT/.done.$tag ]] && { echo "skip $rel"; date > "$S/DONE"; return; }
  # One writer per shard: two fetchers appending to the same range files interleave and corrupt them (2026-10-09:
  # two tmux windows ran the same shards, five tarballs failed sha256).
  exec {lk}>"$DL/$tag.lock"; flock -n "$lk" || { echo "$rel is being fetched by another run" | tee "$S/ERROR"; return 1; }
  trap 'echo "failed at line $LINENO" > "$S/ERROR"' ERR
  size=$(curl -sIL --retry 5 "$EP/$rel" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
  chunk=$(( (size + N - 1) / N ))
  echo "fetch $rel: $size bytes in $N ranges" | tee "$S/STATUS"
  for i in $(seq 0 $((N - 1))); do
    (
      a=$((i * chunk)); b=$((a + chunk - 1)); ((b >= size)) && b=$((size - 1))
      f=$DL/$tag.$i; want=$((b - a + 1))
      # hf-mirror throttles a connection after its first minute or two (7.6 MB/s fresh vs 2 MB/s after 30 min,
      # 2026-10-08), so every connection is cut after CONN_S seconds and resumed from the bytes on disk.
      # No curl --retry: a retried transfer restarts at the same offset and would append the bytes twice.
      # The body goes to a side file and is appended only for HTTP 206: an error page or a full 200 reply that ignores
      # the Range header would otherwise be appended as data, which keeps the size right and corrupts the bytes.
      until (( $(sz "$f") == want )); do
        (( $(free) > MIN_FREE )) || { echo "free disk below $((MIN_FREE / 1000000000)) GB while fetching $rel" > "$S/ERROR"; exit 1; }
        code=$(curl -sL -o "$f.new" -w '%{http_code}' --speed-limit 20000 --speed-time 60 --max-time "${CONN_S:-90}" \
          -r $((a + $(sz "$f")))-$b "$EP/$rel" || true)
        if [[ $code == 206 ]]; then
          # never append past the range end
          head -c $((want - $(sz "$f"))) "$f.new" >> "$f"
        else sleep 2; fi
        rm -f "$f.new"
      done
    ) &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i"; done   # a bare `wait` would also wait for the tee of the log redirect
  local dl=$((SECONDS - t0)); t0=$SECONDS
  echo "verify + extract $rel" > "$S/STATUS"
  parts() { for i in $(seq 0 $((N - 1))); do cat "$DL/$tag.$i"; done; }
  local want_sha got_sha
  # The tree API sometimes answers with an empty or non-JSON body (the 2026-10-09 exit): retry until it parses.
  want_sha=
  for i in $(seq 1 20); do
    want_sha=$(curl -sf --max-time 60 "$API/$(dirname "$rel")" | python3 -c "import json,sys; print(next(f['lfs']['oid'] for f in json.load(sys.stdin) if f['path']==sys.argv[1]))" "$rel" 2>/dev/null) && [[ -n $want_sha ]] && break
    want_sha=; sleep $((i * 5))
  done
  [[ -n $want_sha ]] || { echo "tree API gave no sha256 for $rel after 20 tries (chunks kept in $DL)" | tee "$S/ERROR"; return 1; }
  got_sha=$(parts | sha256sum | cut -d' ' -f1)
  if [[ $got_sha != "$want_sha" ]]; then
    # a corrupt tarball is deleted and fetched again once; a second failure stops this shard
    echo "sha256 mismatch for $rel: $got_sha != $want_sha; deleting its chunks"; rm -f "$DL/$tag".?  "$DL/$tag".??
    [[ ${2:-} == retry ]] && { echo "sha256 mismatch twice for $rel" | tee "$S/ERROR"; return 1; }
    exec {lk}>&-; fetch "$rel" retry; return
  fi
  # the extracted tree is ~1.2x the tarball and both exist until the archive is deleted
  (( $(free) - size * 12 / 10 > MIN_FREE )) || { echo "extracting $rel would leave under $((MIN_FREE / 1000000000)) GB free" | tee "$S/ERROR"; return 1; }
  parts | pigz -dc | tar -x -C "$ROOT"
  rm -f "$DL/$tag".?  "$DL/$tag".??; touch "$ROOT/.done.$tag"
  printf '%s\t%s bytes\tdownload %ss (%s MB/s)\textract %ss\n' "$rel" "$size" "$dl" \
    $((size / 1000000 / (dl > 0 ? dl : 1))) $((SECONDS - t0)) | tee -a "$OUT/sizes.tsv"
  date > "$S/DONE"; echo done > "$S/STATUS"
}

fetch trajdata_cache/nuplan_test.tar.gz
fetch MTGS_asset/navtest/configs.tar.gz
shards=("${@:-part001}"); [[ ${shards[0]} == all ]] && shards=($(printf 'part%03d ' $(seq 1 15)))
echo "shards ${shards[*]}: $PAR at once, $N ranges each" > "$OUT/STATUS"
pids=()
for s in "${shards[@]}"; do
  while (( $(jobs -rp | wc -l) >= PAR )); do sleep 5; done
  ( fetch "MTGS_asset/navtest/assets/$s.tar.gz" ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || { echo "a shard failed: $(ls "$OUT"/*/ERROR 2>/dev/null | tr '\n' ' ')" > "$OUT/ERROR"; exit 1; }
rmdir "$DL" 2>/dev/null || true
du -sh --apparent-size "$ROOT"/* "$ROOT"/navtest/* | tee -a "$OUT/sizes.tsv"
echo "assets: $(ls "$ROOT/navtest/assets" | wc -l) folders" | tee -a "$OUT/sizes.tsv"
date > "$OUT/DONE"; echo done > "$OUT/STATUS"
