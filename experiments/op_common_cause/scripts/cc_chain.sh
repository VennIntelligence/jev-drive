#!/usr/bin/env bash
# Lane B chain (night 2026-10-03): every domain's shards in parallel on one card, then the report.
# Resumable: a shard whose raw npz exists is skipped. STATUS / DONE / ERROR in $DATA_DIR/runs/op_common_cause/chain/.
#   scripts/tmux_run.sh ccB experiments/op_common_cause/scripts/cc_chain.sh [gpu] [shards_per_domain] [domains...]
set -uo pipefail
GPU=${1:-1}; N=${2:-2}; shift 2 || true
DOMS=${*:-nav carla wod}
cd "$(dirname "$0")/../../.."
L=$DATA_DIR/runs/op_common_cause/chain; mkdir -p "$L"; rm -f "$L/DONE" "$L/ERROR"
PY=$DATA_DIR/envs/openpilot/bin/python
st() { echo "$(date '+%F %T') ccB: $*" > "$L/STATUS"; echo "$(date '+%T') $*"; }
pids=()
for d in $DOMS; do
  for ((i = 0; i < N; i++)); do
    if [[ -f $DATA_DIR/runs/op_common_cause/raw/$d/$d-${i}of$N.npz ]]; then echo "skip $d $i/$N"; continue; fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY experiments/op_common_cause/scripts/cc_run.py --domain "$d" --shard "$i/$N" \
      --workers 4 > "$L/$d-$i.log" 2>&1 &
    pids+=($!)
  done
done
st "running ${#pids[@]} shards on GPU $GPU: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for d in $DOMS; do for ((i = 0; i < N; i++)); do
  [[ -f $DATA_DIR/runs/op_common_cause/raw/$d/$d-${i}of$N.npz ]] || { fail=1; echo "missing $d $i/$N"; }
done; done
if (( fail )); then st "a shard failed, see $L/*.log"; echo "shard failure" > "$L/ERROR"; exit 1; fi
st "shards done; report"
if .venv/bin/python experiments/op_common_cause/scripts/cc_report.py > "$L/report.log" 2>&1; then
  st "done"; date > "$L/DONE"
else
  st "report failed"; tail -20 "$L/report.log" > "$L/ERROR"; exit 1
fi
