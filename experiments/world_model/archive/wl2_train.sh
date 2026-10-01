#!/usr/bin/env bash
# WL-2 training lane (fc65452:todos/2026-09-29-wl2-prereg.md): one self-advancing chain of "<arms>|<seeds>" phases on one GPU.
# Usage (on the box, in tmux via scripts/tmux_run.sh): experiments/world_model/archive/wl2_train.sh <lane> <gpu> <cpus> "<arms>|<seeds>" ...
#   a phase whose arms are "vrep" runs WL-1's frozen predictors (`wl2_model vrep`); otherwise `wl2_model train`.
# Resumable: finished arm x seed (a model.pt under runs/wl2/model) are skipped, T resumes from its checkpoint.
# Writes runs/wl2/train_chain/<lane>/{STATUS,log.txt} and DONE or ERROR.
set -u
lane=$1; gpu=$2; cpus=$3; shift 3
D=${DATA_DIR:?DATA_DIR is not set}
cd "$(dirname "$0")/../../.."
out=$D/runs/wl2/train_chain/$lane
mkdir -p "$out"; rm -f "$out/DONE" "$out/ERROR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for ph in "$@"; do
  arms=${ph%%|*}; seeds=${ph##*|}
  echo "$(date '+%F %T') start $arms | $seeds" >> "$out/STATUS"
  if [[ $arms == vrep ]]; then step=vrep; else step=train; fi
  if CUDA_VISIBLE_DEVICES=$gpu taskset -c "$cpus" .venv/bin/python -m experiments.world_model.archive.wl2_model $step --arms $arms --seeds $seeds >> "$out/log.txt" 2>&1; then
    echo "$(date '+%F %T') done $arms | $seeds" >> "$out/STATUS"
  else
    echo "$(date '+%F %T') ERROR $arms | $seeds" | tee -a "$out/STATUS" > "$out/ERROR"
    exit 1
  fi
done
date '+%F %T' > "$out/DONE"
