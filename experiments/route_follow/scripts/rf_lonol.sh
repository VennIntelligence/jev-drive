#!/usr/bin/env bash
# LON-DIAG reading 3 (plans/2026-10-15-longitudinal-diag-prereg.md, 读数 3; Amendment 1): open-loop counterfactuals of RSG-L18-s0 on the
# navtrain_full offline rows (rf_lonol.py). Box, repo root.
#   rf_lonol.sh cells     CPU (tmux): cell flags of the 12 shards + the training store's speed distribution
#   rf_lonol.sh submit    one GPU pool job: store + gate + 6 fed rows on every cell row; then rf_lonol.sh report (CPU)
#   rf_lonol.sh report    -> $O/results (openloop*.md / .json) and $O/figs
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/lon_diag/openloop; TR=$DATA_DIR/envs/op-train/bin/python; S="0 1 2 3 4 5 6 7 8 9 10 11"
case $1 in
cells)
  mkdir -p $O
  $TR experiments/route_follow/scripts/rf_lonol.py cells --shards $S --out $O/cells || exit 1
  $TR experiments/route_follow/scripts/rf_lonol.py speeds --out $O/cells/speeds.json ;;
submit)
  mkdir -p $O/pool
  id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name lonol-RSG-L18-s0 --tries 1 --priority 14 --no-check --log-dir $O/pool/run \
       --vram 32 --cpu 6 --ram 90 --timeout-h 2 -- $TR experiments/route_follow/scripts/rf_lonol.py run --shards $S --tag RSG-L18-s0 --cells $O/cells \
       --out $O/run | awk '{print $NF}') && [[ -n $id ]] || { echo "submit failed" >&2; exit 1; }
  echo "$id lonol-RSG-L18-s0" | tee -a $O/jobs.txt ;;
report)
  for x in "" --shard0; do
    $TR experiments/route_follow/scripts/rf_lonol.py report --cells $O/cells --run $O/run --out $O/results --figs $O/figs --speeds $O/cells/speeds.json $x > /dev/null || exit 1
  done
  ls -la $O/results $O/figs ;;
*) echo "rf_lonol.sh cells | submit | report" >&2; exit 2 ;;
esac
