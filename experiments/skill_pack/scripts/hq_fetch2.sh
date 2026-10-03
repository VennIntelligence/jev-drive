#!/usr/bin/env bash
# Addendum 1 of plans/2026-10-04-history-quality-prereg.md: the same draw extended to 1 500 navtest tokens; pooled run
# dirs lb_hq_navtestX / lb_hq_navhard1X. STATUS / DONE2 / ERROR2 in $DATA_DIR/runs/op_lb/hq.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_lb/hq; rm -f "$R/ERROR2" "$R/DONE2"
J=$DATA_DIR/envs/jevdrive/bin/python; H=experiments/skill_pack/scripts/hq_real.py
st() { echo "$(date '+%F %T') [x] $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR2"; exit 1; }
clash-start > /dev/null 2>&1 || true
$J $H select --n 1500 >> "$R/log.txt" 2>&1 || die select
st "dbs + locate"
$J $H dbs --threads 8 >> "$R/dbs.log" 2>&1 & p1=$!
$J $H locate --threads 32 >> "$R/locate.log" 2>&1 || die locate
wait $p1 || die dbs
st "index"; $J $H index --threads 40 --logs 8 >> "$R/index.log" 2>&1 || die index
st "fetch"; $J $H fetch --threads 32 >> "$R/fetch.log" 2>&1 || die fetch
st "render"
for d in lb_hq_navtestX lb_hq_navhard1X; do $J $H render --data $d >> "$R/render.log" 2>&1 || die "render $d"; done
st done; touch "$R/DONE2"
