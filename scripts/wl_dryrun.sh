#!/usr/bin/env bash
# WL-2 representation dry run: train arms x 3 seeds one after another (skips finished ones), then write DONE.
# Usage (on the box, in tmux via scripts/tmux_run.sh): scripts/wl_dryrun.sh <chain-name> "<arm> <arm> ..."
# GPU comes from CUDA_VISIBLE_DEVICES; cores are pinned to the lane's 48-71 (runs/sched/table.tsv, lane wl-dryrun).
set -u
name=$1; arms=$2
D=${DATA_DIR:-$HOME/data}
out=$D/runs/wl/dryrun/chain_$name
mkdir -p "$out"; rm -f "$out/DONE" "$out/ERROR"
for a in $arms; do
  for s in 0 1 2; do
    if ls "$D"/runs/wl/dryrun/"$a"/seed"$s"/*/model.pt >/dev/null 2>&1; then continue; fi
    echo "$(date +%H:%M:%S) start $a seed $s" >> "$out/STATUS"
    if taskset -c 48-71 .venv/bin/python -m jevdrive.wl_dryrun train --arm "$a" --seed "$s"; then
      echo "$(date +%H:%M:%S) done $a seed $s" >> "$out/STATUS"
    else
      echo "$(date +%H:%M:%S) ERROR $a seed $s" | tee -a "$out/STATUS" > "$out/ERROR"
      exit 1
    fi
  done
done
touch "$out/DONE"
