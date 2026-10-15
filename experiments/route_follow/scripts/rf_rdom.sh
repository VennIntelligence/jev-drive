#!/usr/bin/env bash
# ROUTE-DOMINANCE (plans/2026-10-16-route-dominance-prereg.md): rf_rdom.py stages on the box, repo root.
#   rf_rdom.sh cpu      CPU (run in tmux): rf_appzeros extract on RSG-L18-s0's held / train pool runs, states, lane-graph swaps
#   rf_rdom.sh submit   two GPU pool jobs (held, train): served / zero / zeroS / swap plans for RSG-L18-s0 and APP1X-L18-s0
#   rf_rdom.sh report   CPU: contacts, registered lines -> $O/results, $O/figs
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/route_dominance; TR=$DATA_DIR/envs/op-train/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; S=experiments/route_follow/scripts; W=${RDOM_WORKERS:-16}
runs() { ls -d $DATA_DIR/runs/onpol_long/roll/$1/c*/run | paste -sd, -; }
case $1 in
cpu)
  mkdir -p $O/zeros/held $O/zeros/train
  [[ -s $O/zeros/held/rollouts.pkl ]] || $AP $S/rf_appzeros.py extract --runs RSG-L18-s0=$(runs RSG-L18-s0-held) --out $O/zeros/held --workers $W || exit 1
  [[ -s $O/zeros/train/rollouts.pkl ]] || $AP $S/rf_appzeros.py extract --runs RSG-L18-s0=$(runs RSG-L18-s0) --out $O/zeros/train --workers $W || exit 1
  $TR $S/rf_rdom.py states --zeros $O/zeros --out $O || exit 1
  $NAV $S/rf_rdom.py swap --states $O --out $O --workers $W ;;
submit)
  mkdir -p $O/pool
  for s in held train; do
    id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name rdom-$s --tries 1 --priority 14 --no-check --log-dir $O/pool/$s \
         --vram 12 --cpu 6 --ram 60 --timeout-h 1 -- $TR $S/rf_rdom.py run --set $s --states $O --out $O | awk '{print $NF}') && [[ -n $id ]] || { echo "submit failed" >&2; exit 1; }
    echo "$id rdom-$s" | tee -a $O/jobs.txt
  done ;;
report)
  $TR $S/rf_rdom.py report --states $O --out $O/results --fig $O/figs/route_dominance.png ;;
*) echo "rf_rdom.sh cpu | submit | report" >&2; exit 2 ;;
esac
