#!/usr/bin/env bash
# P4 wave 2 (todos/2026-09-23-p4-carla-feature-gap.md): once wave 1's runner exits, drive the frozen wave-2 route
# list on server index 70; once the vlm stream's 32B jobs have exited (RESOURCE_LEDGER: <= 1 CARLA server while
# they run), add a second runner on index 73. Both share one --out, so b2d_run's claims split the routes.
set -uo pipefail
cd "$(dirname "$0")/.."
out=$DATA_DIR/runs/p4_carla/gen
ids=$(tail -n +2 research/results/p4-carla-gap/routes_wave2.csv | cut -d, -f1 | paste -sd,)
run() { $DATA_DIR/envs/carla/bin/python scripts/b2d_run.py --out "$out" --workers 1 --server-index "$1" \
  --route-ids "$ids" --agent scripts/p4_carla_agent.py --agent-config research/results/p4-carla-gap/agent_config.json \
  --fast-copy --no-spectator --max-attempts 2 --stagger-s 0; }
done_window() { ! tmux list-windows -t jev -F '#W' | grep -qx "$1" || tmux capture-pane -p -t "jev:$1" | grep -q "==> exited with"; }
until done_window p4-gen; do sleep 60; done
echo "$(date +%T) wave 1 runner exited; starting wave 2 on index 70"
run 70 &
until done_window p5vlm-32b && done_window p5vlm-nr; do sleep 60; done
echo "$(date +%T) vlm 32B jobs exited; adding a second runner on index 73"
echo "$(date '+%F %H:%M') | p4 | jev:p4-wave2 | +7-9 GB (2nd CARLA server, index 73) | +4 cores | until wave 2 is done | P4 wave 2 second runner" >> $DATA_DIR/runs/RESOURCE_LEDGER.md
run 73 &
wait
echo "$(date +%T) wave 2 done"
