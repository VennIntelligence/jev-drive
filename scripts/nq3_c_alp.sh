#!/usr/bin/env bash
# Night queue 3, lane C, Q1: Alpamayo 1.5 (nav, E[1 sample]) on the P6 v0 exam frames (scripts/nq3_c_alpamayo.py).
# Idempotent: finished frames are skipped, and a DONE file ends it at once. A second invocation waits on the lock.
# Pins GPU 6 and cores 158-163 itself; logs and DONE under $DATA_DIR/runs/nq3/c/q1_alp/.
#   scripts/tmux_run.sh nq3-c-alp scripts/nq3_c_alp.sh          (ALP_DEADLINE=HH:MM overrides 23:30, box clock;
#   ALP_BATCH=B > 1 batches B frames per call: faster on a shared card, but not bit-identical to batch 1)
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
D=$DATA_DIR/runs/nq3/c/q1_alp
mkdir -p "$D"
exec 9>"$D/lock"
flock 9
if [[ -f $D/DONE ]]; then
    echo "q1_alp: DONE already there: $(cat "$D/DONE")"
    exit 0
fi
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python
t0=$(date +%s)
echo "$(date '+%F %T') q1_alp start (pid $$)" | tee -a "$D/log.txt"
taskset -c 158-163 "$PY" scripts/nq3_c_alpamayo.py run --workers 4 --threads 1 --batch "${ALP_BATCH:-1}" \
    --deadline "${ALP_DEADLINE:-23:30}" --summary "$D/summary.json" 2>&1 | tee -a "$D/run.log"
"$PY" - "$D" "$t0" <<'EOF'
import json, sys, time
d, t0 = sys.argv[1], float(sys.argv[2])
s = json.load(open(f"{d}/summary.json"))
json.dump({"wall_s": round(time.time() - t0), "frames_per_priority_done_of_all": s["per_priority"],
           "stopped_before_unit": s["stopped_before_unit"], "s_per_frame": s["s_per_frame"],
           "outputs": s["outputs"]}, open(f"{d}/DONE", "w"))
EOF
echo "$(date '+%F %T') q1_alp done: $(cat "$D/DONE")" | tee -a "$D/log.txt"
