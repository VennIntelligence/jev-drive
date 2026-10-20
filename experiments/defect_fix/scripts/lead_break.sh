#!/usr/bin/env bash
# LEAD-BREAK (plans/2026-10-20-lead-break-prereg.md): pool jobs on the box, repo root. Ids in $O/jobs.txt.
#   lead_break.sh all     select -> rows (CPU: onp_rows rows + cold from RB0-NAT's stored rollout.asl) -> cf (GPU) ; nav (CPU) ; report --after both
#   job body: rows
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/lead_break; PY=experiments/defect_fix/scripts/lead_break.py; TR=$DATA_DIR/envs/op-train/bin/python; mkdir -p $O/pool
AS=$DATA_DIR/third_party/alpasim; RB=$DATA_DIR/runs/defect_fix/lineage/read/RB0-NAT
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries 1 --priority ${PRIO:-10} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
all)
  git rev-parse --short HEAD > $O/code.txt; $TR $PY select || exit 1
  r=$(sub lead-break-rows --vram 0.5 --cpu 24 --ram 96 --timeout-h 2 -- bash $0 rows) || exit 1
  c=$(sub lead-break-cf --after $r --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- $TR $PY cf) || exit 1
  n=$(sub lead-break-nav --vram 0.1 --cpu 12 --ram 96 --timeout-h 1 -- $TR $PY nav) || exit 1
  sub lead-break-report --after $c,$n --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY report > /dev/null || exit 1
  echo "rows $r cf $c nav $n" ;;
rows)
  STG=$DATA_DIR/tmp/lead_break_rows; mkdir -p $STG $O/rows
  env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python experiments/onpol_pai/scripts/onp_rows.py rows --pass RB0 \
      --runs "$RB/runs/*" --out $O/rows --stage $STG --tag RB0-F-s0 --scenes $(cat $O/scenes.txt) --jobs 24 || exit 1
  env JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass RB0 --out $O/rows --stage $STG --jobs 24 || exit 1
  env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass RB0 --out $O/rows > $O/rows/report_RB0.log 2>&1
  tail -3 $O/rows/report_RB0.log; rm -rf $STG ;;
*) sed -n 2,4p $0; exit 2 ;;
esac
