#!/usr/bin/env bash
# WL-2 feature extraction loop (fc65452:todos/2026-09-30-wl2-feat.md): scripts/tmux_run.sh wl2-vjepa experiments/world_model/archive/wl2_feat.sh vjepa
#   vjepa: GPU 2, cores 0-7      op: GPU 3, cores 8-15      (override GPU / CPUS / WL2_WORKERS in the env)
# Loops over the finished runs until the generation ends (pipe/STATUS.md "generation end"), then one last pass.
# feat/DONE_<mode> or feat/ERROR_<mode> in runs/wl2/feat/; resumable (skips what is stored).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/../../.."
mode=${1:?vjepa|op}
SHARD=${WL2_SHARD:-0/1}                      # op only: runs with route_id % n == i
tag=$mode; [[ $SHARD != 0/1 ]] && tag=${mode}_${SHARD/\//of}
F=$DATA_DIR/runs/wl2/feat
mkdir -p "$F"
rm -f "$F/DONE_$tag" "$F/ERROR_$tag"
case $mode in vjepa) GPU=${GPU:-2} CPUS=${CPUS:-0-7} ;; op) GPU=${GPU:-3} CPUS=${CPUS:-8-15} ;; esac
export PYTHONPATH=. WL_RESULTS=$DATA_DIR/runs/wl2/results CUDA_VISIBLE_DEVICES=$GPU
if taskset -c "$CPUS" .venv/bin/python -m experiments.world_model.archive.wl2_feat "$mode" --workers "${WL2_WORKERS:-8}" --shard "$SHARD" 2>&1 | tee -a "$F/log_$tag.txt"; [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    date '+%F %T' > "$F/DONE_$tag"
else
    date '+%F %T' > "$F/ERROR_$tag"
fi
