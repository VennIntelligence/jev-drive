#!/usr/bin/env bash
# Official NAVSIM scoring for the zero-shot exam (todos/2026-09-24-zeroshot-exam/navsim.md). Devkits run as shipped
# (scripts/setup_navsim_devkit.sh); our only addition is the replay agent jevdrive.navsim_agent.PrecomputedAgent.
#
#   scripts/navsim_zs_score.sh cache v1|v2 <split>                 metric cache (CPU, ray)
#   scripts/navsim_zs_score.sh score v1|v2 <split> <name> <agent>  agent = cv | human | <predictions.npz>
#
#   v1 = navsim v1.1 run_pdm_score.py -> PDMS;  v2 = navsim main @ 0a380a9 -> EPDMS: navtest via
#   run_pdm_score_one_stage.py (non-reactive log replay, the devkit default), navhard_two_stage via run_pdm_score.py.
# Caches: $DATA_DIR/runs/navsim/metric_cache/<v>_<split>; results: $DATA_DIR/runs/navsim/eval/<v>_<split>_<name>/<ts>/
# NAVSIM_THREADS (default 16) caps ray workers: the box has 25 cores shared with two other exams.
set -euo pipefail
cmd=$1 ver=$2 split=$3
repo=$(cd "$(dirname "$0")/.." && pwd)
if [[ $ver == v1 ]]; then dk=$DATA_DIR/third_party/navsim-v1.1; py=$DATA_DIR/envs/navsim1/bin/python
else dk=$DATA_DIR/third_party/navsim; py=$DATA_DIR/envs/navsim2/bin/python; fi
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
export OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim NAVSIM_EXP_ROOT=$DATA_DIR/runs/navsim/eval
export NAVSIM_DEVKIT_ROOT=$dk PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
# numpy 1.23.4 (pinned by navsim) bundles an OpenBLAS that picks a broken kernel on this box's Sapphire Rapids CPU:
# pinv / inv come back silently wrong (errors ~1e3), the PDM LQR simulator blows up and every score is garbage.
# Forcing the Haswell kernel fixes it; refuse to run if linear algebra is still wrong.
export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-Haswell}
"$py" -c "import numpy as np; A=np.random.default_rng(0).normal(size=(40,40)); e=abs(A@np.linalg.inv(A)-np.eye(40)).max(); assert e<1e-8, f'BLAS broken: {e}'"
cache=$DATA_DIR/runs/navsim/metric_cache/${ver}_$split
threads=${NAVSIM_THREADS:-16}
worker=(worker=ray_distributed_no_torch worker.threads_per_node=$threads)
syn=(synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
     synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles)
[[ $ver == v1 ]] && syn=()

if [[ $cmd == cache ]]; then
  key=metric_cache_path; [[ $ver == v1 ]] && key=cache.cache_path   # v1.1 names it differently
  exec "$py" "$dk/navsim/planning/script/run_metric_caching.py" train_test_split=$split \
    $key=$cache "${worker[@]}" "${syn[@]}"
fi

name=$4 agent=$5
case $agent in
  cv) a=(agent=constant_velocity_agent) ;;
  human) a=(agent=human_agent) ;;
  *.npz) a=(agent._target_=jevdrive.navsim_agent.PrecomputedAgent "+agent.predictions=$agent") ;;
  *) echo "unknown agent $agent" >&2; exit 1 ;;
esac
script=run_pdm_score.py
[[ $ver == v2 && $split != *two_stage* ]] && script=run_pdm_score_one_stage.py
"$py" "$dk/navsim/planning/script/$script" train_test_split=$split metric_cache_path=$cache \
  experiment_name=${ver}_${split}_$name "${worker[@]}" "${a[@]}" "${syn[@]}"
