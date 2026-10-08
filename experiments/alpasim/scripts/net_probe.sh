#!/usr/bin/env bash
# Measure total download throughput of one asset shard per route and stream count, to size fetch_data.sh.
# Each config opens S ranged connections for T seconds (the lifetime fetch_data.sh gives a connection) and
# reports total MB/s. Routes: mirror (hf-mirror.com, direct), hf (huggingface.co through the box proxy).
# Usage (box): scripts/tmux_run.sh alpasim-probe experiments/alpasim/scripts/net_probe.sh [streams ...]
set -uo pipefail
OUT=$DATA_DIR/runs/alpasim/fetch; TMP=$OUT/_probe; T=${T:-90}; mkdir -p "$TMP"
REL=datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track/resolve/main/MTGS_asset/navtest/assets/part015.tar.gz
probe() {  # <route> <streams>
  local route=$1 s=$2 url px=() i
  if [[ $route == hf ]]; then url=https://huggingface.co/$REL; px=(-x "${https_proxy:-http://127.0.0.1:7890}")   # clash, docs/network-proxy.md
  else url=https://hf-mirror.com/$REL; px=(--noproxy '*'); fi
  rm -f "$TMP"/*
  for i in $(seq 0 $((s - 1))); do
    curl -sL "${px[@]}" --max-time "$T" -r $((i * 400000000))- "$url" -o "$TMP/$i" &
  done
  wait
  printf '%s\tstreams %s\t%s MB/s\n' "$route" "$s" "$(du -sb "$TMP" | awk -v t="$T" '{printf "%.1f", $1 / 1e6 / t}')" | tee -a "$OUT/net_probe.tsv"
}
rm -f "$OUT/net_probe.DONE"; date | tee -a "$OUT/net_probe.tsv"
for s in "${@:-8 16 32 64}"; do for r in ${ROUTES:-mirror hf}; do probe $r "$s"; done; done
rm -rf "$TMP"; date > "$OUT/net_probe.DONE"
