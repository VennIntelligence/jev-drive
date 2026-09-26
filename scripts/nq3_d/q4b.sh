#!/usr/bin/env bash
# Night queue 3, lane D, Q4b (todos/2026-09-26-night-queue-3.md, [D] Q4b entry): NAVSIM-protocol openpilot features on
# the P5 v1 BA obs rows (GPU 6, envs/openpilot), Hydra refits and the compatibility check, the P5 exam if comparable.
# Called by scripts/nq3_d.sh with CPUS / GPU exported; estimate ~90 min (features ~40 min, Hydra 6 refits ~15 min).
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
CPUS=${CPUS:-180-199} GPU=${GPU:-6}
export P5_SET=carla_p5v1_ba
echo "[$(date +%T)] Q4b: estimate ~90 min"
f=$DATA_DIR/processed/carla_p5v1_ba/nq3_q4b_navsim_protocol
if [[ ! -f $f/lebowski.npz ]]; then
  mkdir -p "$f"; .venv/bin/python -c "import pandas as pd, json, os; t = pd.read_parquet(os.environ['DATA_DIR'] + '/processed/carla_p5v1_ba/index.parquet'); json.dump(sorted(t.frame_name[t.role == 'obs']), open('$f/obs_names.json', 'w'))"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 taskset -c "$CPUS" "$DATA_DIR/envs/openpilot/bin/python" scripts/nq3_d/q4b_openpilot.py --check 8
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 taskset -c "$CPUS" "$DATA_DIR/envs/openpilot/bin/python" scripts/nq3_d/q4b_openpilot.py --workers 12
fi
py=(taskset -c "$CPUS" .venv/bin/python -m jevdrive.nq3_q4b)
[[ -f $DATA_DIR/runs/nq3/q4b/navridge.npz ]] || CUDA_VISIBLE_DEVICES=$GPU "${py[@]}" hydra
"${py[@]}" compat
CUDA_VISIBLE_DEVICES=$GPU "${py[@]}" exam
echo "[$(date +%T)] Q4b done"
