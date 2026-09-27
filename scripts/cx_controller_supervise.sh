#!/usr/bin/env bash
# Run once inside tmux jev. Repeated failures back off to 60s; never kill children.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is required}"
cd "$(dirname "$0")/.."
out=$DATA_DIR/runs/sched/controller
mkdir -p "$out"
exec 9>"$out/supervisor.lock"
flock -n 9 || { echo 'controller supervisor already owns the lock'; exit 1; }
backoff=5
while true; do
    start=$SECONDS
    python3 scripts/cx_controller.py run >> "$out/supervisor.log" 2>&1
    rc=$?
    printf '%s controller exited rc=%s; restart in %ss\n' "$(date -u +%FT%TZ)" "$rc" "$backoff" >> "$out/supervisor.log"
    (( SECONDS - start > 300 )) && backoff=5
    sleep "$backoff"
    (( backoff < 60 )) && backoff=$((backoff * 2))
    (( backoff > 60 )) && backoff=60
done
