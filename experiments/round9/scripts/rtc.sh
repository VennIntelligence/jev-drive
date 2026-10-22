#!/usr/bin/env bash
# ROUND9 ROUTE-CHECK chain (box, CPU only, repo root): scripts/tmux_run.sh rtc experiments/round9/scripts/rtc.sh [stage ...]
# stages (default: extract events): extract (.venv) | events (AlpaSim venv) | band (op-train venv); writes $O/DONE at the end.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=
O=$DATA_DIR/runs/round9/route_check; mkdir -p $O; rm -f $O/DONE
PY=experiments/round9/scripts/rtc.py; AS=$DATA_DIR/third_party/alpasim
for st in ${@:-extract events}; do
  echo "$(date '+%F %T') $st start" | tee -a $O/stages.txt
  case $st in
    extract) .venv/bin/python $PY extract ;;
    events) env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python $PY events ;;
    band*) $DATA_DIR/envs/op-train/bin/python $PY band --ckpt ${st#band:} ;;
  esac > $O/$st.log 2>&1 || { echo "$(date '+%F %T') $st FAILED" | tee -a $O/stages.txt; }
  echo "$(date '+%F %T') $st end" | tee -a $O/stages.txt
done
date > $O/DONE
