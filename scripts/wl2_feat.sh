#!/usr/bin/env bash
# WL-2 feature extraction loop (todos/2026-09-30-wl2-feat.md): scripts/tmux_run.sh wl2-vjepa scripts/wl2_feat.sh vjepa
#   vjepa: GPU 2, cores 0-7      op: GPU 3, cores 8-15      (override GPU / CPUS / WL2_WORKERS in the env)
# Loops over the finished runs until the generation ends (pipe/STATUS.md "generation end"), then one last pass.
# feat/DONE_<mode> or feat/ERROR_<mode> in runs/wl2/feat/; resumable (skips what is stored).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
mode=${1:?vjepa|op}
F=$DATA_DIR/runs/wl2/feat
mkdir -p "$F"
rm -f "$F/DONE_$mode" "$F/ERROR_$mode"
case $mode in vjepa) GPU=${GPU:-2} CPUS=${CPUS:-0-7} ;; op) GPU=${GPU:-3} CPUS=${CPUS:-8-15} ;; esac
export PYTHONPATH=. WL_RESULTS=$DATA_DIR/runs/wl2/results CUDA_VISIBLE_DEVICES=$GPU
if taskset -c "$CPUS" .venv/bin/python -m jevdrive.wl2_feat "$mode" --workers "${WL2_WORKERS:-8}" 2>&1 | tee -a "$F/log_$mode.txt"; [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    date '+%F %T' > "$F/DONE_$mode"
else
    date '+%F %T' > "$F/ERROR_$mode"
fi
