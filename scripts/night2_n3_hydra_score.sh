#!/usr/bin/env bash
# Night queue 2, N3 [B]: Hydra seeds 1 and 2 -- a new CPU k-means vocabulary each, then E6's per-anchor sub-scores
# (scripts/elicit_e6_score.py as shipped) on E6's fixed 20 000-token subset and v1.1 metric cache. Resumable.
#   scripts/night2_n3_hydra_score.sh [seeds="1 2"]
# CPUs: N3_CPUS (default 64-111, 48 cores), niced.
set -uo pipefail
cd ~/data/jev-drive
CPUS=${N3_CPUS:-64-111}; N=$(python3 -c "a,b='$CPUS'.split('-');print(int(b)-int(a)+1)")
CACHE=$DATA_DIR/runs/navsim/metric_cache/v1_e6sub
for s in ${1:-1 2}; do
  P=$(ls -td "$DATA_DIR"/runs/night2/n3-prep-s$s/* 2>/dev/null | head -1)
  if [[ -z $P || ! -f $P/anchors.npz ]]; then
    OMP_NUM_THREADS=16 taskset -c "$CPUS" nice -n 10 .venv/bin/python -m jevdrive.night2_n3 prep --seed "$s" || { echo "FAILED prep $s"; exit 1; }
    P=$(ls -td "$DATA_DIR"/runs/night2/n3-prep-s$s/* | head -1)
  fi
  [[ -f $P/score.done ]] && continue
  echo "$(date '+%F %H:%M') seed $s: scoring into $P/score on $N cores"
  env NAVSIM_DEVKIT_ROOT=$DATA_DIR/third_party/navsim-v1.1 NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps \
      OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 taskset -c "$CPUS" \
      "$DATA_DIR/envs/navsim1/bin/python" scripts/elicit_e6_score.py run "$CACHE" "$P/anchors.npz" "$P/tokens.txt" "$P/score" \
      --procs "$N" 2>&1 | grep -av -i warn | tee -a "$P/log.txt" || { echo "FAILED score $s"; exit 1; }
  touch "$P/score.done"
done
echo "all seeds scored"
