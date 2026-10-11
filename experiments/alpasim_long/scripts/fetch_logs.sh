#!/usr/bin/env bash
# Fetch the raw nuPlan v1.1 test-split sqlite logs behind the 1491 public AlpaSim nuPlan scenes (136 logs, 7.7 GB compressed)
# by HTTP range reads from the public nuPlan archive, into $DATA_DIR/datasets/nuplan_raw/test/. Resumable (complete logs are skipped).
# Usage (box): scripts/tmux_run.sh al-fetch bash experiments/alpasim_long/scripts/fetch_logs.sh [logs.txt] [streams]
set -euo pipefail
cd "$(dirname "$0")/../../.."
logs=${1:-experiments/alpasim_long/results/logs_all.txt}; streams=${2:-8}
out=$DATA_DIR/datasets/nuplan_raw/test
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890  # docs/network-proxy.md: S3 is not reachable direct
clash-start >/dev/null
mkdir -p "$out"
python3 -u experiments/alpasim_long/scripts/nuplan_zip.py fetch --logs "$logs" --out "$out" --streams "$streams" 2>&1 | tee -a "$out/../fetch_log.txt"
