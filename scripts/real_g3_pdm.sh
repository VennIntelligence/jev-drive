#!/usr/bin/env bash
# G3b (1), todos/2026-09-26-real-data-transfer.md [G3] 08:36: the official v1.1 PDMS of E2's "continue" and "brake"
# proposals on the 911 navtrain edit-pair tokens (metric cache for these tokens only; devkit as shipped, replayed
# through jevdrive.navsim_agent.PrecomputedAgent), then label (c) into processed/elicit_e2/navtrain/main/pdm_scores.csv.
#   scripts/real_g3_pdm.sh [threads]
set -euo pipefail
threads=${1:-48}
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
.venv/bin/python -m jevdrive.real_g3 pdm-prep
d=$(ls -td "$DATA_DIR"/runs/real-data-transfer/g3pdmprep/*/ | head -1); d=${d%/}
dk=$DATA_DIR/third_party/navsim-v1.1 py=$DATA_DIR/envs/navsim1/bin/python
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
export OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim NAVSIM_EXP_ROOT=$d/eval NAVSIM_DEVKIT_ROOT=$dk
export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$py" -c "import numpy as np; A=np.random.default_rng(0).normal(size=(40,40)); e=abs(A@np.linalg.inv(A)-np.eye(40)).max(); assert e<1e-8, f'BLAS broken: {e}'"
toks="[$(sed "s/.*/'&'/" "$d/tokens.txt" | paste -sd,)]"
worker=(worker=ray_distributed_no_torch worker.threads_per_node=$threads)
filt="train_test_split.scene_filter.tokens=$toks"
t0=$(date +%s)
"$py" "$dk/navsim/planning/script/run_metric_caching.py" train_test_split=navtrain cache.cache_path=$d/metric_cache "${worker[@]}" "$filt"
echo "cache: $(( $(date +%s) - t0 )) s, $(find "$d/metric_cache" -name metric_cache.pkl | wc -l) tokens"
for p in continue brake; do
  "$py" "$dk/navsim/planning/script/run_pdm_score.py" train_test_split=navtrain metric_cache_path=$d/metric_cache \
    experiment_name=g3_$p "${worker[@]}" "$filt" \
    agent._target_=jevdrive.navsim_agent.PrecomputedAgent "+agent.predictions=$d/$p.npz"
done
echo "total: $(( $(date +%s) - t0 )) s"
c=$(ls -t "$d"/eval/g3_continue/*/*.csv | head -1) b=$(ls -t "$d"/eval/g3_brake/*/*.csv | head -1)
.venv/bin/python -m jevdrive.real_g3 pdm-read "$d" "$c" "$b"
