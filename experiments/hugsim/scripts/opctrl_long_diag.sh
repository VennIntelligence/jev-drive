#!/usr/bin/env bash
# CPU replay (steps 0-100) of the stuck opctrl runs and of the base3 0418 run, recording the action head's acceleration (decision 118 follow-up).
# Usage (box, tmux): [PARTS=8] [THREADS=4] experiments/hugsim/scripts/opctrl_long_diag.sh   -> $DATA_DIR/runs/opctrl/replay_stuck/{DONE,ERROR}
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=$DATA_DIR/runs/opctrl/replay_stuck
CSV=$DATA_DIR/runs/opctrl/closed/results.csv
mkdir -p "$OUT/op" "$OUT/base"; rm -f "$OUT/DONE" "$OUT/ERROR"
ONLY=$($DATA_DIR/envs/hugsim/bin/python -c "
import csv
R={}
for r in csv.DictReader(open('$CSV')):
    R.setdefault(r['scenario'], {})[r['tag']] = r['end']
print(','.join(s for s, d in R.items() if d.get('cinque-opctrl') == 'max_steps'))")
N=${PARTS:-8}; pids=()
for i in $(seq 0 $((N - 1))); do
    "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/scripts/opctrl_replay.py "$CSV" "$OUT/op" --tag cinque-opctrl --only "$ONLY" \
        --part "$i/$N" --steps 100 --threads "${THREADS:-4}" > "$OUT/op$i.log" 2>&1 &
    pids+=($!)
done
"$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/scripts/opctrl_replay.py "$CSV" "$OUT/base" --tag cinque-fixed-base3 \
    --only scene-0418-hard-00,scene-3000_3200-medium-00 --steps 100 --threads "${THREADS:-4}" > "$OUT/base.log" 2>&1 &
pids+=($!)
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if (( rc == 0 )); then touch "$OUT/DONE"; else echo "rc $rc" > "$OUT/ERROR"; fi
