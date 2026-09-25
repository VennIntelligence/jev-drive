#!/usr/bin/env bash
# Elicitation E1, NAVSIM column (todos/2026-09-26-elicitation-program.md, deviation [E1] (5)): Qwen features on
# navtest (already running in window e1-navqwen) and navhard two-stage -> prior + Delta predictions -> devkit scores.
# Launch on the box in tmux: scripts/tmux_run.sh e1-navsim env GPU=1 scripts/elicit_e1_navsim.sh
# Re-running skips what exists (feature chunks, scored names); writes <run>/scored.done at the end.
set -uo pipefail
cd ~/data/jev-drive
PY=.venv/bin/python
GPU=${GPU:-1}
until $PY -m jevdrive.navsim_qwen check navtest 2>/dev/null; do sleep 60; done
env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 nice -n 10 $PY -m jevdrive.navsim_qwen work navhard_two_stage --workers 4
until $PY -m jevdrive.navsim_qwen check navhard_two_stage 2>/dev/null; do sleep 60; done   # other cards may still hold chunks
run=$(ls -td $DATA_DIR/runs/elicitation/e1-navsim/*/ 2>/dev/null | head -1)
if [[ -z $run || ! -f $run/navsim_activation.csv ]]; then
  env CUDA_VISIBLE_DEVICES=$GPU P5_SET=carla_p5v1_ba OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 nice -n 10 \
    $PY -m jevdrive.elicit_e1 navsim || exit 1
  run=$(ls -td $DATA_DIR/runs/elicitation/e1-navsim/*/ | head -1)
fi
S() {  # S v1|v2 split name npz: score unless a csv for that name exists
  ls $DATA_DIR/runs/navsim/eval/${1}_${2}_$3/*/*.csv >/dev/null 2>&1 && return 0
  NAVSIM_THREADS=12 nice -n 10 scripts/navsim_zs_score.sh score "$1" "$2" "$3" "$4" > $run/score_${1}_${2}_$3.log 2>&1
}
for m in cinque lebowski; do
  for pr in ridge_late cls_late; do
    n=e1_${pr}_${m}_plus_mc
    S v1 navtest $n $run/navtest_${pr}_${m}_plus_mc.npz &
    S v2 navtest $n $run/navtest_${pr}_${m}_plus_mc.npz
    wait
  done
  S v2 navhard_two_stage e1_ridge_late_${m}_plus_mc $run/navhard_two_stage_ridge_late_${m}_plus_mc.npz
done
touch $run/scored.done
echo "E1 NAVSIM scored -> $run"
