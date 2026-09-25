#!/usr/bin/env bash
# E3 PDM-scorer sign agreement (todos/2026-09-26-elicitation-program.md, [E3] item 11): metric cache for the sampled
# navtrain tokens only, then the official v1.1 PDMS of the "continue" and "brake" proposals (devkit as shipped,
# replayed through jevdrive.navsim_agent.PrecomputedAgent).
#   scripts/elicit_e3_scorer.sh <e3 run dir> [threads]
set -euo pipefail
run=$1 threads=${2:-16}
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
.venv/bin/python -m jevdrive.elicit_e3 scorer-prep "$run"
d=$run/scorer
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
[[ -e $d/cache.done ]] || { "$py" "$dk/navsim/planning/script/run_metric_caching.py" train_test_split=navtrain \
  cache.cache_path=$d/metric_cache "${worker[@]}" "$filt" && touch "$d/cache.done"; }
echo "cache: $(( $(date +%s) - t0 )) s"
for p in continue brake; do
  "$py" "$dk/navsim/planning/script/run_pdm_score.py" train_test_split=navtrain metric_cache_path=$d/metric_cache \
    experiment_name=e3_$p "${worker[@]}" "$filt" \
    agent._target_=jevdrive.navsim_agent.PrecomputedAgent "+agent.predictions=$d/$p.npz"
done
echo "total: $(( $(date +%s) - t0 )) s"
c=$(ls -t "$d"/eval/e3_continue/*/*.csv | head -1) b=$(ls -t "$d"/eval/e3_brake/*/*.csv | head -1)
.venv/bin/python -m jevdrive.elicit_e3 scorer-read "$run" "$c" "$b"
