#!/usr/bin/env bash
# SCH GPU helper: stop a helper (SIGINT, so it releases its claim) when its card's free VRAM drops below a floor,
# so a lane step on that card never OOMs because of the helper. Exits when the helper is gone.
# Usage: vram_guard.sh <pid> <card> <min_free_MiB>
set -uo pipefail
pid=$1 card=$2 floor=$3
log() { echo "$(date '+%F %T') [vram_guard $pid gpu$card] $*"; }
log "start: SIGINT when free < $floor MiB"
while kill -0 "$pid" 2> /dev/null; do
  free=$(nvidia-smi -i "$card" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if (( free < floor )); then log "free $free MiB < $floor: kill -INT $pid"; kill -INT "$pid"; break; fi
  sleep 10
done
while kill -0 "$pid" 2> /dev/null; do sleep 2; done
log "helper $pid gone"
