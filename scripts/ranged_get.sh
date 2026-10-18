#!/usr/bin/env bash
# Resume <part> up to <size> bytes from <url> over <conns> parallel HTTP range requests (the mirrors cap each connection,
# not the host). Chunks of $CHUNK_MB (default 32) MB land in <part>.d/<offset> and are appended to <part> in order as soon
# as they are contiguous; a failed or stalled chunk is fetched again. Exit 0 when <part> has <size> bytes.
#   scripts/ranged_get.sh <url> <part> <size> <conns> [curl proxy args ...]
set -uo pipefail
url=$1 part=$2 size=$3 conns=$4; shift 4; proxy=("$@")
CH=$(( ${CHUNK_MB:-32} * 1000000 )); D=$part.d; mkdir -p "$D"; touch "$part"
have=$(stat -c %s "$part")
# Shared circuit breaker (all archives and chunks on this box share $GATE): a 429 / 5xx / reset halves the allowed number
# of concurrent range requests and sets a cool-down that doubles per consecutive failure (30 s .. 15 min); 15 clean chunks
# in a row raise the limit by one, up to $MAXCONN. State files: $GATE/{limit,until,fails,ok}, log: $GATE/events.log.
GATE=${GATE:-$(dirname "$part")/state/.gate}; MAXCONN=${MAXCONN:-16}; mkdir -p "$GATE"
gate_get() { cat "$GATE/$1" 2>/dev/null || echo "${2:-0}"; }
gate_fail() ( flock 9; (( $(gate_get until) > $(date +%s) )) && exit 0  # already tripped: in-flight failures do not stack
  f=$(( $(gate_get fails) + 1 )); l=$(( $(gate_get limit "$MAXCONN") / 2 )); (( l < 2 )) && l=2
  w=$(( 30 << (f > 5 ? 5 : f - 1) )); (( w > 900 )) && w=900
  echo "$f" > "$GATE/fails"; echo "$l" > "$GATE/limit"; echo 0 > "$GATE/ok"; echo $(( $(date +%s) + w )) > "$GATE/until"
  printf '%s fail #%s (%s): limit -> %s, pause %ss\n' "$(date '+%F %T')" "$f" "$1" "$l" "$w" >> "$GATE/events.log" ) 9>"$GATE/lock"
gate_ok() ( flock 9; echo 0 > "$GATE/fails"; k=$(( $(gate_get ok) + 1 )); l=$(gate_get limit "$MAXCONN")
  if (( k >= 15 && l < MAXCONN )); then l=$((l + 1)); k=0; echo "$l" > "$GATE/limit"
    printf '%s recovering: limit -> %s\n' "$(date '+%F %T')" "$l" >> "$GATE/events.log"; fi; echo "$k" > "$GATE/ok" ) 9>"$GATE/lock"
fetch() {  # <offset>: one chunk, retried (behind the gate) until it is complete
  local o=$1 e=$(( $1 + CH - 1 )) code fd i got u now; (( e >= size )) && e=$(( size - 1 )); local n=$(( e - o + 1 ))
  for _ in $(seq 400); do
    while :; do  # cool-down, then a free slot below the current limit
      u=$(gate_get until); now=$(date +%s); (( u > now )) && { sleep $(( u - now + RANDOM % 10 )); continue; }
      for i in $(seq "$(gate_get limit "$MAXCONN")"); do
        exec {fd}>"$GATE/slot.$i"; flock -n "$fd" && break 2; exec {fd}>&-
      done; sleep $(( 1 + RANDOM % 3 ))
    done
    code=$(curl -sSL "${proxy[@]}" -r "$o-$e" --connect-timeout 30 --speed-limit 10240 --speed-time 60 -o "$D/$o.tmp" \
      -w '%{http_code}' "$url" 2>/dev/null); exec {fd}>&-
    got=$(stat -c %s "$D/$o.tmp" 2>/dev/null || echo 0)
    if [[ $code == 206 && $got == "$n" ]]; then mv "$D/$o.tmp" "$D/$o"; gate_ok; return 0; fi
    rm -f "$D/$o.tmp"; gate_fail "http $code"
  done
  return 1
}
export -f fetch gate_get gate_fail gate_ok; export url size CH D GATE MAXCONN; export proxy_s="${proxy[*]}"
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
