#!/usr/bin/env bash
# Cosmos G4 full run (todos/2026-09-28-cosmos-pilot.md, "full run"): the one-shot lane driver jevdrive/cosmos_full.py on the
# cores of the `cosmos-full` row of runs/sched/table.tsv. Resumable: re-running skips everything already on disk.
#   scripts/tmux_run.sh cosmos-full scripts/cosmos_full.sh [--target 2000] [--insts 0,5 --keep-npy]
# env: COSMOS_FULL_DIR (runs/<dir>, default cosmos_full; staged pilots use cosmos_full/stage1 ...), CARLA_W="g:n,..."
#      (CARLA servers per card, default the row's workers), COSMOS_SLOTS="0,1,1-b" (one Cosmos worker per entry, on the
#      card before the dash), COSMOS_LATE="2,3" (slots that start when CARLA is done), CONTROLS_PROCS, DISK_FLOOR_GB,
#      COSMOS_RGB_ATTRS='{"exposure_mode": "manual"}' (render fix for the RGB camera)
# Signals: $DATA_DIR/runs/$COSMOS_FULL_DIR/lane/{STATUS,DONE,ERROR}; touch lane/DRAIN to stop at the next boundary.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
L=$DATA_DIR/runs/${COSMOS_FULL_DIR:-cosmos_full}/lane
mkdir -p "$L"
python3 scripts/sch_table.py check > /dev/null || { echo "sch_table.py check fails: not starting" | tee -a "$L/STATUS"; echo "sch check" > "$L/ERROR"; exit 1; }
cpus=$(python3 - "$DATA_DIR/runs/sched/table.tsv" <<'PYEOF'
import csv, sys
print(next(r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["lane"] == "cosmos-full")["cpus"])
PYEOF
)
exec taskset -c "$cpus" "$DATA_DIR/envs/jevdrive/bin/python" -m jevdrive.cosmos_full run "$@"
