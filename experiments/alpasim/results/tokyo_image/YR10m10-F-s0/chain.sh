#!/usr/bin/env bash
# YR10m10-F-s0 submission image: waits until both cards are idle (the ONPOL-PAI 281 queue holds them), then new_tag.sh
# (cached build -> test.sh -> 48-scene smoke at 8 concurrent, native ref) and one more smoke with the driver at 2 concurrent rollouts.
set -uo pipefail
C=/data/runs/alpasim/tokyo_image/yr10m10_chain; cd /data/runs/alpasim/tokyo_image/code-yr10m10
st() { echo "$(date "+%F %T") $*" | tee "$C/STATUS"; }
rm -f $C/{DONE,ERROR}; n=0
st "waiting for both cards idle (< 1000 MiB, 3 checks a minute apart)"
until (( n >= 3 )); do
  if (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) < 1000 )); then n=$((n+1)); else n=0; fi; sleep 60
done
st "cards idle: new_tag.sh"
export DATA_DIR=/data WHEELS=/data/cache/jev_wheels
O=/data/runs/alpasim/tokyo_image/YR10m10-F-s0-$(date +%Y%m%d-%H%M%S); echo $O > $C/OUT
OUT=$O bash experiments/alpasim/docker/new_tag.sh YR10m10-F-s0 || { st "ERROR new_tag.sh: $(cat $O/STATUS)"; echo new_tag > $C/ERROR; exit 1; }
st "smoke with the driver at 2 concurrent rollouts"
WIZ_EXTRA="runtime.endpoints.driver.n_concurrent_rollouts=2" bash experiments/alpasim/docker/smoke.sh "$(cat $O/IMAGE)" $O/smoke_c2 "" /data/runs/alpasim/native_ref/YR10m10-F-s0/aggregate/results-summary.json \
  || { st "ERROR smoke_c2: $(cat $O/smoke_c2/STATUS)"; echo smoke_c2 > $C/ERROR; exit 1; }
date > $C/DONE; st "done: $O"
