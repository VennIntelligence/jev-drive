#!/usr/bin/env bash
# SCH pull-forward of lane D Q4b's feature block: scripts/nq3_d/q4b.sh lines 12-16, verbatim, on a borrowed card and
# core range. q4b.sh skips that block once processed/carla_p5v1_ba/nq3_q4b_navsim_protocol/lebowski.npz exists
# (q4b_openpilot.py saves cinque.npz, then lebowski.npz, only after the whole batch). Hydra / compat / exam stay with
# the chain. Refuses to start if the chain has already entered q4b (no concurrent writers).
#   scripts/tmux_run.sh sch-fwd-q4b scripts/sch_gpu_helpers/q4b_features.sh <gpu> <cpus>
# Stop: kill -TERM -- -$(cat $DATA_DIR/runs/sched/pull_forward/q4b/pgid)
set -euo pipefail
: "${DATA_DIR:?}"
[[ $(ps -o pgid= $$ | tr -d ' ') == "$$" ]] || exec setsid --wait "$0" "$@"   # own process group, stoppable by PGID
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
GPU=${1:?gpu} CPUS=${2:?cpus}
J=$DATA_DIR/runs/sched/pull_forward/q4b
mkdir -p "$J"
echo $$ > "$J/pgid"
{ echo "commit $(git rev-parse HEAD)"; md5sum scripts/nq3_d/q4b.sh scripts/nq3_d/q4b_openpilot.py scripts/p5_openpilot.py \
    scripts/navsim_zs_openpilot.py scripts/wod_zeroshot_openpilot.py jevdrive/p5_openpilot.py jevdrive/openpilot/model.py; } > "$J/provenance.txt"
echo "[$(date '+%F %T')] SCH pull-forward Q4b features on GPU $GPU, cores $CPUS"
[[ -e $DATA_DIR/runs/nq3/d/q4b ]] && { echo "lane D chain already in q4b: leave it to the chain"; exit 3; }
export P5_SET=carla_p5v1_ba
f=$DATA_DIR/processed/carla_p5v1_ba/nq3_q4b_navsim_protocol
if [[ ! -f $f/lebowski.npz ]]; then
  mkdir -p "$f"; .venv/bin/python -c "import pandas as pd, json, os; t = pd.read_parquet(os.environ['DATA_DIR'] + '/processed/carla_p5v1_ba/index.parquet'); json.dump(sorted(t.frame_name[t.role == 'obs']), open('$f/obs_names.json', 'w'))"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 taskset -c "$CPUS" "$DATA_DIR/envs/openpilot/bin/python" scripts/nq3_d/q4b_openpilot.py --check 8
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 taskset -c "$CPUS" "$DATA_DIR/envs/openpilot/bin/python" scripts/nq3_d/q4b_openpilot.py --workers 12
fi
# the chain's consumer (nq3_q4b.p5_sets) must find every P5 row; shapes and finiteness
taskset -c "$CPUS" .venv/bin/python - <<'EOF'
import json, os, numpy as np
from jevdrive import nq3_q4b as Q, night2_n3 as N3
d = os.environ["DATA_DIR"] + "/processed/carla_p5v1_ba/nq3_q4b_navsim_protocol"
want = set(json.load(open(d + "/obs_names.json")))
p5 = N3.p5_data()
for m in Q.MODELS:
    z = np.load(f"{d}/{m}.npz")
    assert set(z["name"]) == want and len(z["name"]) == len(want) and np.isfinite(z["temporal"]).all()
    e, x, n = Q.p5_sets(m, p5)["p5"]
    print(m, "rows", len(z["name"]), "temporal", z["temporal"].shape, z["temporal"].dtype, "clamped", float(z["clamped"].mean()), "p5 rows", x.shape)
EOF
date '+%F %T' > "$J/DONE"
echo "[$(date '+%F %T')] done; the chain's q4b step will skip the feature block"
