#!/usr/bin/env bash
# Resume <part> up to <size> bytes from <url> over <conns> parallel HTTP range requests (the mirrors cap each connection,
# not the host). Chunks of $CHUNK_MB (default 32) MB land in <part>.d/<offset> and are appended to <part> in order as soon
# as they are contiguous; a failed or stalled chunk is fetched again. Exit 0 when <part> has <size> bytes.
#   scripts/ranged_get.sh <url> <part> <size> <conns> [curl proxy args ...]
set -uo pipefail
url=$1 part=$2 size=$3 conns=$4; shift 4; proxy=("$@")
CH=$(( ${CHUNK_MB:-32} * 1000000 )); D=$part.d; mkdir -p "$D"; touch "$part"
have=$(stat -c %s "$part")
fetch() {  # <offset>: one chunk, retried until it is complete
  local o=$1 e=$(( $1 + CH - 1 )); (( e >= size )) && e=$(( size - 1 )); local n=$(( e - o + 1 ))
  for _ in $(seq 200); do
    curl -fsSL "${proxy[@]}" -r "$o-$e" --connect-timeout 30 --speed-limit 10240 --speed-time 60 -o "$D/$o.tmp" "$url" \
      && [[ $(stat -c %s "$D/$o.tmp") == "$n" ]] && { mv "$D/$o.tmp" "$D/$o"; return 0; }
    sleep 5
  done
  return 1
}
export -f fetch; export url size CH D; export proxy_s="${proxy[*]}"
# Appender: moves finished chunks into <part> in offset order while the fetchers run.
append() {
  local p=$(stat -c %s "$part")
  while [[ -e $D/$p ]]; do cat "$D/$p" >> "$part" && rm -f "$D/$p"; p=$(stat -c %s "$part"); done
}
seq "$have" "$CH" $(( size - 1 )) | xargs -P "$conns" -I{} bash -c 'proxy=($proxy_s); fetch {}' &
xp=$!
while kill -0 $xp 2>/dev/null; do append; sleep 5; done
wait $xp; append
[[ $(stat -c %s "$part") == "$size" ]] && rmdir "$D" 2>/dev/null; [[ $(stat -c %s "$part") == "$size" ]]
