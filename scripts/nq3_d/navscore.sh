#!/usr/bin/env bash
# Official NAVSIM v1.1 PDMS for lane D of night queue 3 (todos/2026-09-26-night-queue-3.md, [D] entries).
# Each job line: "<split> <name> <predictions.npz>". split navtest -> scripts/navsim_zs_score.sh (devkit as shipped,
# replay agent); split nq3hold -> the same run_pdm_score.py on the navtrain held-out scene filter written by
# `jevdrive.nq3_q4a prep` with E6's v1_e6sub metric cache. Results: runs/navsim/eval/v1_<split>_<name>/<ts>/.
#   scripts/nq3_d/navscore.sh <jobs.txt> [PAR=2] [THREADS=8]
set -euo pipefail
jobs=$1 par=${2:-2} threads=${3:-8}
repo=$(cd "$(dirname "$0")/../.." && pwd)
dir=$(dirname "$jobs")
export NAVSIM_THREADS=$threads REPO=$repo
one() {
  local split=$1 name=$2 npz=$3 log=$dir/score_${1}_${2}.log
  if ls "$DATA_DIR/runs/navsim/eval/v1_${split}_${name}"/*/*.csv >/dev/null 2>&1; then echo "skip $name (scored)"; return 0; fi
  if [[ $split == navtest ]]; then
    "$REPO/scripts/navsim_zs_score.sh" score v1 navtest "$name" "$npz" > "$log" 2>&1
  else
    local dk=$DATA_DIR/third_party/navsim-v1.1 py=$DATA_DIR/envs/navsim1/bin/python
    ( export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps \
        OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim NAVSIM_EXP_ROOT=$DATA_DIR/runs/navsim/eval NAVSIM_DEVKIT_ROOT=$dk \
        PYTHONPATH=$REPO OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      "$py" -c "import numpy as np; A=np.random.default_rng(0).normal(size=(40,40)); e=abs(A@np.linalg.inv(A)-np.eye(40)).max(); assert e<1e-8, f'BLAS broken: {e}'"
      "$py" "$dk/navsim/planning/script/run_pdm_score.py" train_test_split=nq3hold \
        "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$DATA_DIR/runs/nq3/q4a/hydra]" \
        metric_cache_path=$DATA_DIR/runs/navsim/metric_cache/v1_e6sub experiment_name=v1_${split}_$name \
        worker=ray_distributed_no_torch worker.threads_per_node=$NAVSIM_THREADS \
        agent._target_=jevdrive.navsim_agent.PrecomputedAgent "+agent.predictions=$npz" ) > "$log" 2>&1
  fi
  ls "$DATA_DIR/runs/navsim/eval/v1_${split}_${name}"/*/*.csv >/dev/null 2>&1 || { echo "FAILED $name (see $log)"; return 1; }
  echo "done $name"
}
export -f one
export dir
grep -v '^\s*$' "$jobs" | xargs -P "$par" -L 1 bash -c 'one "$0" "$1" "$2"'
