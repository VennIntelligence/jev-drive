#!/usr/bin/env bash
# INERTIA (plans/2026-10-20-inertia-prereg.md): pool jobs on the box, repo root. Ids in $O/jobs.txt.
#   inertia.sh recede        supplement 1 (CPU job): A0's stored 10 Hz plans on the overrun / control approaches
#   inertia.sh smoke         cf on 3 scenes (GPU job) -> $O/smoke
#   inertia.sh all           cf (GPU) -> report (CPU, --after)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/inertia; PY=experiments/defect_fix/scripts/inertia.py; TR=$DATA_DIR/envs/op-train/bin/python; mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries 1 --priority ${PRIO:-10} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
recede) sub inertia-recede --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR $PY recede ;;
smoke)  sub inertia-smoke --vram 24 --cpu 8 --ram 48 --timeout-h 1 -- bash -c "O_SMOKE=1 $TR $PY cf --limit 3 --workers 3" ;;
all)    c=$(sub inertia-cf --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- $TR $PY cf) || exit 1
        sub inertia-report --after $c --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY report ;;
*) sed -n 2,5p $0; exit 2 ;;
esac
