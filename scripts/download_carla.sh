#!/usr/bin/env bash
# Fetch the CARLA 0.9.15 Linux package into $DATA_DIR/third_party/carla. Resumable.
# Usage: scripts/download_carla.sh [base|maps|all]     (default: all)
#
# Two byte-range streams per file, each curl capped at 120 s. That cap is the point: long-lived
# connections to the Backblaze mirror decay and never recover (measured 0.30 MB/s after an hour
# versus 1.45 MB/s on a fresh connection), and AdditionalMaps failed outright with
# "curl: (18) transfer closed with N bytes remaining". Cycling the connection fixed both:
# 0.30 -> 2.95 MB/s. Do not "optimise" the cap away.
#
# Direct, never through a proxy: measured per stream at a deep offset,
# direct 1.45 MB/s, AutoDL turbo 0.20-0.28, Clash 0.08-0.47.
# Keep it at two streams while anything else is downloading; see docs/network-proxy.md.
set -uo pipefail
cd "$DATA_DIR/third_party/carla" 2>/dev/null || { mkdir -p "$DATA_DIR/third_party/carla" && cd "$_"; }
unset http_proxy https_proxy all_proxy
BASE=https://carla-releases.s3.us-east-005.backblazeb2.com/Linux

fetch() {  # fetch <file> <total-bytes> <part-prefix>
  local file=$1 total=$2 pre=$3 head mid
  [[ -f $file ]] || : >"$file"
  head=$(stat -c%s "$file")
  if (( head >= total )); then echo "$file already complete"; return 0; fi
  mid=$(( head + (total - head) / 2 ))
  echo "== $file: have $head, split at $mid, total $total"

  pull() {  # pull <partfile> <abs-start> <abs-end-inclusive>
    local f=$1 s=$2 e=$3 have
    while :; do
      have=0; [[ -f $f ]] && have=$(stat -c%s "$f")
      (( s + have > e )) && return 0
      curl -sS --max-time 120 -r "$((s + have))-$e" "$BASE/$file" >>"$f"
      sleep 2
    done
  }
  pull "${pre}1" "$head" "$((mid - 1))" &
  pull "${pre}2" "$mid" "$((total - 1))" &
  wait
  cat "${pre}1" "${pre}2" >>"$file" && rm -f "${pre}1" "${pre}2"
  local got; got=$(stat -c%s "$file")
  (( got == total )) || { echo "FAIL: $file is $got, expected $total"; return 1; }
  echo "== $file complete ($got bytes)"
}

case ${1:-all} in
  base) fetch CARLA_0.9.15.tar.gz 8386636048 part ;;
  maps) fetch AdditionalMaps_0.9.15.tar.gz 7375946087 amap ;;
  all)  fetch CARLA_0.9.15.tar.gz 8386636048 part && fetch AdditionalMaps_0.9.15.tar.gz 7375946087 amap ;;
  *) sed -n '2,4p' "$0"; exit 1 ;;
esac
