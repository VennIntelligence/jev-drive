#!/usr/bin/env bash
# Night queue 3, lane D, Q6 (todos/2026-09-26-night-queue-3.md, Q6 and the [D] 16:50 entry): V-JEPA 2 single-frame
# control, V-JEPA 2 stream on real data, main table. Called by scripts/nq3_d.sh with no arguments; resumable (a step
# with a marker in $DATA_DIR/runs/nq3/q6/done/ is skipped); exits non-zero on the first failure.
# Resources (lane D): GPU 6 (<= 20 GB), cores 180-199; the devkit scoring overlaps the GPU steps on 180-191.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/../.."
GPU=${Q6_GPU:-6}
CPUS=${Q6_CPUS:-180-199}
SCORE_CPUS=${Q6_SCORE_CPUS:-180-191}
GPU_CPUS=${Q6_GPU_CPUS:-192-199}
OUT=$DATA_DIR/runs/nq3/q6
mkdir -p "$OUT/done"
export P5_SET=carla_p5v1_ba CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
PY=(taskset -c "$GPU_CPUS" .venv/bin/python -m jevdrive.nq3_q6)
echo "[$(date '+%F %T')] Q6 start. Estimate: ~1.3 h wall, ~0.6 GPU h (extraction 25 min, fits 30 min, devkit 6 jobs 15 min overlapped, table 5 min)"

step() {  # step <name> <cmd ...>
  local name=$1; shift
  if [[ -f $OUT/done/$name ]]; then echo "[$(date +%T)] skip $name (done)"; return; fi
  echo "[$(date +%T)] start $name"
  local t0=$SECONDS
  "$@"
  echo "{\"step\": \"$name\", \"wall_s\": $((SECONDS - t0)), \"end\": \"$(date '+%F %T')\"}" > "$OUT/done/$name"
  echo "[$(date +%T)] end $name ($((SECONDS - t0)) s)"
}

step sf-check        "${PY[@]}" sf-check --n 256 --batch 64 --workers 6
step rt-nav-extract  "${PY[@]}" rt-nav-extract --batch 64 --workers 6
step rt-heads        "${PY[@]}" rt-heads
step rt-wod          "${PY[@]}" rt-wod
step rt-nav          "${PY[@]}" rt-nav
# devkit (v1.1 PDMS) of the 6 prior + Delta arms, in the background on its own cores while the GPU steps run
score_pid=
if [[ ! -f $OUT/done/rt-score ]]; then
  ( OMP_NUM_THREADS=1 NAVSIM_THREADS=6 taskset -c "$SCORE_CPUS" scripts/real_g1_score.sh "$OUT/nav/jobs.txt" 2 \
      > "$OUT/nav/score.log" 2>&1 && ! grep -q FAILED "$OUT/nav/score.log" && touch "$OUT/done/rt-score" ) &
  score_pid=$!
  echo "[$(date +%T)] devkit scoring started (pid $score_pid)"
fi
step sf-extract      "${PY[@]}" sf-extract --batch 64 --workers 6
refit() {
  "${PY[@]}" sf-fit --backbone vjepa2 --seed 0
  local d; d=$(ls -d "$OUT"/fits/refit-vjepa2-seed0/*/ | tail -1)
  "${PY[@]}" sf-refit-check --fit-dir "$d"
}
step sf-refit-check  refit
for s in 0 1 2; do step "sf-fit-s$s" "${PY[@]}" sf-fit --seed "$s"; done
step sf-report       "${PY[@]}" sf-report
if [[ -n $score_pid ]]; then wait "$score_pid" || true; fi
[[ -f $OUT/done/rt-score ]] || { echo "devkit scoring failed, see $OUT/nav/score.log"; exit 1; }
step rt-verdict      "${PY[@]}" rt-verdict
step table           taskset -c "$CPUS" .venv/bin/python -m jevdrive.nq3_q6 table
echo "[$(date '+%F %T')] Q6 done"
