#!/usr/bin/env bash
# PAI-STOP-PROBE chain (plans/2026-10-18-pai-stop-probe-prereg.md). Box, repo root; every job through the GPU pool, priority $PRIO (11).
#   rf_stopprobe.sh submit [smoke]   rows -> cache -> feats -> probe (smoke: probe --smoke only, on existing feats), job ids -> $O/jobs.txt
#   job bodies: rows | cache | feats | probe [--smoke]
# rows: onp_rows rows + cold on the 18-scene camera rerun (runs/route_follow/pai_stop_cues/read, rollout.asl kept; no PAI_DUMP, so G-in checks
# the drive.jsonl half only) -> $O/rows/c18; cache: onp_cache build -> $O/cache/c18 (tokens + teacher); feats / probe: rf_stopprobe.py.
set -o pipefail
D=${DATA_DIR:-/root/autodl-tmp/ujs}; O=$D/runs/route_follow/stop_probe; RD=$D/runs/route_follow/pai_stop_cues/read
AS=$D/third_party/alpasim; TR=$D/envs/op-train/bin/python; V=.venv/bin/python; ME=experiments/route_follow/scripts/rf_stopprobe.sh
PY=experiments/route_follow/scripts/rf_stopprobe.py; PRIO=${PRIO:-11}; STG=$D/tmp/stopprobe_c18
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner stop-probe --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case $1 in
submit)
  if [[ $2 == smoke ]]; then
    sub sp-probe-smoke --vram 16 --cpu 16 --ram 96 --timeout-h 1 -- bash $ME probe --smoke; exit; fi
  r=$(sub sp-rows --vram 0.5 --cpu 18 --ram 64 --timeout-h 1 -- bash $ME rows) || exit 1
  c=$(sub sp-cache --after $r --vram 20 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME cache) || exit 1
  f=$(sub sp-feats --after $c --vram 20 --cpu 8 --ram 96 --timeout-h 1 -- bash $ME feats) || exit 1
  q=$(sub sp-probe --after $f --vram 16 --cpu 16 --ram 128 --timeout-h 2 -- bash $ME probe) || exit 1
  echo "rows $r cache $c feats $f probe $q" ;;
rows)
  mkdir -p $STG
  env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python experiments/onpol_pai/scripts/onp_rows.py rows --pass c18 --runs "$RD/runs/*" \
      --out $O/rows --stage $STG --tag RSG-L18-s0 --jobs 16 || exit 1
  env JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass c18 --out $O/rows --stage $STG --jobs 16 || exit 1
  env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass c18 --out $O/rows > $O/rows/report_c18.log 2>&1
  tail -3 $O/rows/report_c18.log; rm -rf $STG ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass c18 --rows $O/rows/c18 --out $O/cache ;;
feats) $TR $PY feats ;;
probe) shift; $TR $PY probe "$@" ;;
*) echo "usage: $0 submit [smoke] | rows | cache | feats | probe [--smoke]" >&2; exit 2 ;;
esac
