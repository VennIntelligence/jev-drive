#!/usr/bin/env bash
# P5 generation, second runner (todos/2026-09-24-p5-carla-pairs-v0.md): once jev:qv-train has exited (the GPU is
# no longer shared with it; jev:after-train then takes <= 40 GB and 8 threads), add a runner with more workers on
# the same --out. b2d_run's claims split the remaining routes between the two runners.
# Usage: scripts/p5_gen_more.sh <workers> <server-index> <route-ids>
set -uo pipefail
cd "$(dirname "$0")/.."
done_window() { ! tmux list-windows -t jev -F '#W' | grep -qx "$1" || tmux capture-pane -p -t "jev:$1" | grep -q "==> exited with"; }
until done_window qv-train; do sleep 60; done
echo "$(date +%T) qv-train exited; adding $1 workers from server index $2"
echo "$(date '+%F %H:%M') | p5 | jev:p5-gen2 | +$1 x ~11 GB (CARLA index $2+ with TFv6) | +$1 x ~2.5 cores | until the batch is done | P5 generation second runner (after qv-train)" >> $DATA_DIR/runs/RESOURCE_LEDGER.md
exec scripts/p5_gen.sh $DATA_DIR/runs/p5_pairs/gen "$1" "$2" "$3" --stagger-s 20
