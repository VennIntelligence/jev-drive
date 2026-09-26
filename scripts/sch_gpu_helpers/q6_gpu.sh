#!/usr/bin/env bash
# SCH pull-forward of lane D Q6's cacheable GPU steps, with the exact commands of scripts/nq3_d/q6.sh, on a borrowed card
# and core range. Only steps whose outputs the chain reuses unchanged:
#   sf-check        the chain's gate, run first; its two extractions are cached by done.json (nq3_q6._run_vjepa)
#   sf-extract      runs/nq3/q6/sf_units/done.json -> the chain's re-run skips the GPU part, redoes only the row gather
#   rt-nav-extract  processed/navsim_vjepa2/navtest/units/done.json, {mean,last_mean,tokens}.npy -> skipped the same way
#   rt-heads        runs/nq3/q6/heads/vjmc_<m>_s<s>.pt, skipped per file by nq3_q6.rt_heads
# No runs/nq3/q6/done/ marker is written: the chain re-runs each step and finds the caches. The fits (sf-fit*,
# sf-refit-check) are not cached and stay with the chain.
#   scripts/tmux_run.sh sch-fwd-q6 scripts/sch_gpu_helpers/q6_gpu.sh <gpu> <cpus>
# Stop: kill -TERM -- -$(cat $DATA_DIR/runs/sched/pull_forward/q6/pgid)
set -euo pipefail
: "${DATA_DIR:?}"
[[ $(ps -o pgid= $$ | tr -d ' ') == "$$" ]] || exec setsid --wait "$0" "$@"
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
GPU=${1:?gpu} CPUS=${2:?cpus} STEPS=${3:-sf-check sf-extract rt-nav-extract rt-heads}
J=$DATA_DIR/runs/sched/pull_forward/q6
mkdir -p "$J"
echo $$ > "$J/pgid"
{ echo "commit $(git rev-parse HEAD)"; md5sum scripts/nq3_d/q6.sh jevdrive/nq3_q6.py jevdrive/features.py jevdrive/n6_backbones.py \
    jevdrive/elicit_e1.py jevdrive/reactivity_mc.py; } > "$J/provenance.txt"
echo "[$(date '+%F %T')] SCH pull-forward Q6 ($STEPS) on GPU $GPU, cores $CPUS"
[[ -e $DATA_DIR/runs/nq3/d/q6 ]] && { echo "lane D chain already in q6: leave it to the chain"; exit 3; }
# q6.sh's environment, cores swapped
export P5_SET=carla_p5v1_ba CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
PY=(taskset -c "$CPUS" .venv/bin/python -m jevdrive.nq3_q6)
for s in $STEPS; do
  echo "[$(date +%T)] start $s"
  case $s in
    sf-check)       "${PY[@]}" sf-check --n 256 --batch 64 --workers 6 ;;
    sf-extract)     "${PY[@]}" sf-extract --batch 64 --workers 6 ;;
    rt-nav-extract) "${PY[@]}" rt-nav-extract --batch 64 --workers 6 ;;
    rt-heads)       "${PY[@]}" rt-heads ;;
    *) echo "not a cacheable step: $s"; exit 2 ;;
  esac
  echo "{\"step\": \"$s\", \"end\": \"$(date '+%F %T')\"}" >> "$J/steps.jsonl"
done
date '+%F %T' > "$J/DONE"
echo "[$(date '+%F %T')] done"
