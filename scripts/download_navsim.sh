#!/usr/bin/env bash
# Download NAVSIM (OpenScene + nuPlan maps) into $DATA_DIR/datasets/navsim, camera only (no LiDAR).
# See docs/navsim.md. Run on the box inside tmux:
#   scripts/tmux_run.sh navsim scripts/download_navsim.sh [part ...]
# Parts (default: all, in this order): maps logs navhard navtest navtrain
#   maps      nuPlan maps v1.1                         -> maps/
#   logs      OpenScene trainval + test log pickles    -> navsim_logs/{trainval,test}/
#   navhard   navhard_two_stage synthetic scenes       -> navhard_two_stage/
#   navtest   OpenScene test camera blobs              -> sensor_blobs/test/
#   navtrain  navtrain current + history camera blobs  -> sensor_blobs/trainval/
# Env: JOBS (concurrent archives, default 8), HF_ENDPOINT (default https://hf-mirror.com).
# Each archive runs download -> size + sha256 check -> extract (LiDAR skipped) -> delete, in its own slot,
# so extraction overlaps with the other slots' downloads. Safe to re-run: finished archives are skipped,
# partial downloads resume.
set -euo pipefail

: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export ROOT=$DATA_DIR/datasets/navsim ARC=$DATA_DIR/datasets/navsim/_archives
export STATE=$ARC/state HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1  # Xet storage breaks through proxies; plain HTTP resolve URLs work
JOBS=${JOBS:-8}
REPO=datasets/OpenDriveLab/OpenScene
MAPS_URL=https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
CLASH=http://127.0.0.1:7890
LIDAR=(--exclude='*/MergedPointCloud' --exclude='*/MergedPointCloud/*' --exclude='*.pcd')

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN/log.txt"; }
event() { printf '{"t": %s, "kind": "%s", %s}\n' "$(date +%s.%N)" "$1" "$2" >> "$RUN/events.jsonl"; }

# One archive: <name> <url> <size> <sha256|-> <dest> <strip> <route>. Runs in an xargs slot.
one() {
  local name=$1 url=$2 size=$3 sha=$4 dest=$5 strip=$6 route=$7
  local f=$ARC/$name part=$ARC/$name.part proxy=() t0 n
  [[ -e $STATE/$name.done ]] && { log "skip $name (done)"; return 0; }
  [[ $route == clash ]] && proxy=(-x "$CLASH")
  if [[ ! -e $f ]]; then
    t0=$SECONDS; local have0; have0=$(stat -c %s "$part" 2>/dev/null || echo 0)
    log "get  $name ($(numfmt --to=iec "$size"), resume at $(numfmt --to=iec "$have0"))"
    for try in $(seq 30); do
      # Re-resolve every try: the mirror redirects to signed URLs that expire. Abort stalls, then resume.
      curl -fsSL "${proxy[@]}" -C - --connect-timeout 30 --speed-limit 20480 --speed-time 120 \
        -o "$part" "$url" && break || true
      (( $(stat -c %s "$part" 2>/dev/null || echo 0) >= size )) && break
      log "retry $name (try $try, have $(numfmt --to=iec "$(stat -c %s "$part" 2>/dev/null || echo 0)"))"
      sleep 20
    done
    local got; got=$(stat -c %s "$part")
    [[ $got == "$size" ]] || { log "FAIL $name size $got != $size"; rm -f "$part"; return 1; }
    if [[ $sha != - ]] && ! echo "$sha  $part" | sha256sum -c --quiet - >/dev/null 2>&1; then
      log "FAIL $name sha256 mismatch, deleting"; rm -f "$part"; return 1
    fi
    mv "$part" "$f"
    local dt=$((SECONDS - t0)) mb; mb=$(( (size - have0) / 1000000 ))
    log "got  $name ${mb} MB in ${dt}s ($(awk -v m="$mb" -v t="$dt" 'BEGIN{printf "%.1f", m/(t>0?t:1)}') MB/s, verified)"
    event download_end "\"file\": \"$name\", \"bytes\": $((size - have0)), \"secs\": $dt"
  fi
  t0=$SECONDS; mkdir -p "$dest"
  if [[ $name == *.zip ]]; then
    rm -rf "$ARC/maps_tmp"; unzip -q -o "$f" -d "$ARC/maps_tmp"
    rm -rf "$dest/maps"; mv "$ARC/maps_tmp/nuplan-maps-v1.0" "$dest/maps"; rm -rf "$ARC/maps_tmp"
    n=$(find "$dest/maps" -type f | wc -l)
  else
    n=$(tar -I pigz -xvf "$f" -C "$dest" --strip-components="$strip" "${LIDAR[@]}" | wc -l)
  fi
  (( n > 0 )) || { log "FAIL $name extracted nothing"; return 1; }
  echo "$n" > "$STATE/$name.done"; rm -f "$f"
  log "done $name: $n entries extracted in $((SECONDS - t0))s, archive deleted"
  event extract_end "\"file\": \"$name\", \"entries\": $n, \"secs\": $((SECONDS - t0))"
}

# Job lines for one part, from the HF tree API (size + sha256 per file), camera archives only.
jobs_for() {
  local part=$1
  if [[ $part == maps ]]; then
    # S3 is unreachable direct from the box; this 1 GB file is the only one that needs Clash.
    local route=direct size
    size=$(curl -sfIL -m 20 "$MAPS_URL" | awk 'tolower($1)=="content-length:"{s=$2} END{print s+0}') || true
    if [[ -z $size || $size == 0 ]]; then
      clash-start >/dev/null && route=clash
      size=$(curl -sfIL -m 30 -x "$CLASH" "$MAPS_URL" | awk 'tolower($1)=="content-length:"{s=$2} END{print s+0}')
    fi
    printf '%s\t%s\t%s\t-\t%s\t0\t%s\n' nuplan-maps-v1.1.zip "$MAPS_URL" "$size" "$ROOT" "$route"
    return
  fi
  local dir pat dest strip
  case $part in
    logs)     dir=openscene-v1.1; pat='openscene_metadata_(trainval|test)\.tgz'; dest=$ROOT/navsim_logs; strip=2 ;;
    navhard)  dir=navsim-v2; pat='navsim_v2\.2_navhard_two_stage_.*'; dest=$ROOT; strip=0 ;;
    navtest)  dir=openscene-v1.1/openscene_sensor_test_camera; pat='.*'; dest=$ROOT/sensor_blobs; strip=2 ;;
    navtrain) dir=navsim; pat='navtrain_(current|history)_[0-9]+\.tgz'; dest=$ROOT/sensor_blobs/trainval; strip=1 ;;
    *) echo "unknown part: $part" >&2; exit 1 ;;
  esac
  curl -sf --retry 5 -m 60 "$HF_ENDPOINT/api/$REPO/tree/main/$dir" | python3 -c '
import json, re, sys
pat, base, dest, strip = sys.argv[1:]
for e in sorted(json.load(sys.stdin), key=lambda e: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", e["path"])]):
    name = e["path"].rsplit("/", 1)[-1]
    if e["type"] == "file" and re.fullmatch(pat, name) and "lfs" in e:
        print(name, base + "/" + e["path"], e["lfs"]["size"], e["lfs"]["oid"], dest, strip, "direct", sep="\t")
' "$pat" "$HF_ENDPOINT/$REPO/resolve/main" "$dest" "$strip"
}

if [[ ${1:-} == _one ]]; then IFS=$'\t' read -r -a a <<< "$2"; one "${a[@]}"; exit; fi

parts=("$@"); (( ${#parts[@]} )) || parts=(maps logs navhard navtest navtrain)
export RUN=$DATA_DIR/runs/download/navsim/$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN" "$STATE" "$ROOT"
log "parts: ${parts[*]}, jobs: $JOBS, endpoint: $HF_ENDPOINT, root: $ROOT, run: $RUN"
event start "\"parts\": \"${parts[*]}\", \"jobs\": $JOBS"
for p in "${parts[@]}"; do jobs_for "$p"; done > "$RUN/jobs.tsv"
todo=$(cut -f1 "$RUN/jobs.tsv" | while read -r n; do [[ -e $STATE/$n.done ]] || echo; done | wc -l)
log "$(wc -l < "$RUN/jobs.tsv") archives ($(awk -F'\t' '{s+=$3} END{printf "%.1f GB", s/1e9}' "$RUN/jobs.tsv")), $todo to do"

rc=0
xargs -P "$JOBS" -d '\n' -I{} "$(realpath "$0")" _one {} < "$RUN/jobs.tsv" || rc=$?
fails=$(cut -f1 "$RUN/jobs.tsv" | while read -r n; do [[ -e $STATE/$n.done ]] || echo "$n"; done)
if [[ -n $fails ]]; then
  log "FAILED ($(wc -l <<< "$fails")): $(paste -sd' ' <<< "$fails"); re-run the same command to resume"
  event end "\"ok\": false"; exit 1
fi
log "all $(wc -l < "$RUN/jobs.tsv") archives done in ${SECONDS}s (xargs rc $rc)"
du -sh "$ROOT"/{maps,navsim_logs/*,sensor_blobs/*,navhard_two_stage} 2>/dev/null | tee -a "$RUN/log.txt"
event end "\"ok\": true, \"secs\": $SECONDS"
