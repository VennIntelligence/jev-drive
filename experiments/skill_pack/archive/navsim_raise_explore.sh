#!/usr/bin/env bash
# NAVSIM raise held-out explorations (fc65452:todos/2026-09-30-navsim-raise.md): navtrain held-out logs only, no test data.
#   scripts/tmux_run.sh rexp env CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_ORDER=PCI_BUS_ID experiments/skill_pack/archive/navsim_raise_explore.sh rules curve ...
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/skill_pack/raise/explore; mkdir -p "$R"; rm -f "$R/ERROR" "$R/DONE"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for w in "$@"; do
  echo "$(date '+%F %T') $w" >> "$R/STATUS"
  taskset -c "${RAISE_CPUS:-155-183}" nice -n 19 "$DATA_DIR/envs/jevdrive/bin/python" -m experiments.skill_pack.archive.navsim_raise explore "$w" \
    >> "$R/log.txt" 2>&1 || { echo "$w" > "$R/ERROR"; exit 1; }
done
echo "$(date '+%F %T') DONE $*" >> "$R/STATUS"; touch "$R/DONE"
