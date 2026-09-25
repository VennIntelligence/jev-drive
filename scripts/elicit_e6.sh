#!/usr/bin/env bash
# Elicitation E6 (b) chain on the box (todos/2026-09-26-elicitation-program.md, [E6]); resumable, every step skips
# what is already done. Writes <prep>/chain.done or <prep>/chain.failed.
#   scripts/elicit_e6.sh <prep run dir> [model]
# CPU steps pinned to E6_CPUS (default 136-159, 24 cores); the fit uses GPU E6_GPU (default 2) for minutes.
set -uo pipefail
cd ~/data/jev-drive
P=$1 M=${2:-cinque}
CPUS=${E6_CPUS:-136-159}; N=$(python3 -c "a,b='$CPUS'.split('-');print(int(b)-int(a)+1)")
GPU=${E6_GPU:-2}
CACHE=$DATA_DIR/runs/navsim/metric_cache/v1_e6sub
fail() { echo "FAILED: $*"; touch "$P/chain.failed"; exit 1; }
rm -f "$P/chain.failed"

# 1. v1.1 metric cache of the 20 000-token navtrain subset (devkit as shipped)
if [[ ! -f $P/cache.done ]]; then
  dk=$DATA_DIR/third_party/navsim-v1.1
  env NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim \
      NAVSIM_EXP_ROOT=$DATA_DIR/runs/navsim/eval NAVSIM_DEVKIT_ROOT=$dk OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 taskset -c "$CPUS" "$DATA_DIR/envs/navsim1/bin/python" \
      "$dk/navsim/planning/script/run_metric_caching.py" train_test_split=e6sub \
      "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$P/hydra]" cache.cache_path=$CACHE \
      worker=ray_distributed_no_torch worker.threads_per_node=$N 2>&1 | grep -av "Processing scenario" | tail -20 \
      || fail cache
  n=$(find "$CACHE" -name metric_cache.pkl | wc -l); echo "cache: $n files"
  (( n >= 19000 )) || fail "cache has only $n files"
  touch "$P/cache.done"
fi

# 2. per-anchor sub-scores (devkit simulator + scorer as shipped)
if [[ ! -f $P/score.done ]]; then
  env NAVSIM_DEVKIT_ROOT=$DATA_DIR/third_party/navsim-v1.1 NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps \
      OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 taskset -c "$CPUS" \
      "$DATA_DIR/envs/navsim1/bin/python" scripts/elicit_e6_score.py run "$CACHE" "$P/anchors.npz" "$P/tokens.txt" "$P/score" \
      --procs "$N" 2>&1 | grep -av -i warn || fail score
  touch "$P/score.done"
fi

# 3. heads, weights and selections (GPU, minutes)
if [[ ! -f $P/fit.done ]]; then
  echo "$(date '+%F %H:%M') [ELICIT/E6] fit on GPU $GPU (minutes)" >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
  env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 nice -n 10 taskset -c "$CPUS" .venv/bin/python -m jevdrive.elicit_e6 fit "$P" "$P/score" --model "$M" \
      || fail fit
  d=$(ls -td "$DATA_DIR"/runs/elicitation/e6-fit/* | head -1); echo "$d" > "$P/fit.done"
  echo "$(date '+%F %H:%M') [ELICIT/E6] fit done, GPU $GPU released" >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
fi
F=$(cat "$P/fit.done")

# 4. official scoring: PDMS (v1.1) and EPDMS (main @ 0a380a9) on navtest, EPDMS on navhard two-stage
for arm in hydra clsref; do
  for vs in "v1 navtest" "v2 navtest" "v2 navhard_two_stage"; do
    set -- $vs; ver=$1 split=$2 name=e6_${arm}_$M
    [[ -f $P/scored_${ver}_${split}_${arm} ]] && continue
    env NAVSIM_THREADS=$N taskset -c "$CPUS" nice -n 10 scripts/navsim_zs_score.sh score $ver $split $name "$F/${split}_${arm}_${M}_temporal.npz" \
        2>&1 | grep -av "Processing\|WARN\|Warn" | tail -5 || fail "score $ver $split $arm"
    touch "$P/scored_${ver}_${split}_${arm}"
  done
done
touch "$P/chain.done"
echo "chain done"
