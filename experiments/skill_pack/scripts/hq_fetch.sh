#!/usr/bin/env bash
# Lane C (history quality), data stage: real 10 Hz CAM_F0 history of the selected tokens from the nuPlan archives,
# rendered as openpilot context frames (hq_real.py). Resumes from disk. STATUS / DONE / ERROR in $DATA_DIR/runs/op_lb/hq.
#   scripts/tmux_run.sh hqfetch experiments/skill_pack/scripts/hq_fetch.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_lb/hq; mkdir -p "$R"; rm -f "$R/ERROR" "$R/DONE"
J=$DATA_DIR/envs/jevdrive/bin/python; H=experiments/skill_pack/scripts/hq_real.py
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
clash-start > /dev/null 2>&1 || true
[[ -f $R/sel.json ]] || $J $H select >> "$R/log.txt" 2>&1 || die select
st "dbs + locate"
$J $H dbs --threads 8 >> "$R/dbs.log" 2>&1 & p1=$!
[[ -f $R/locate.json ]] || $J $H locate --threads 64 >> "$R/locate.log" 2>&1 || die locate
wait $p1 || die dbs
st "index"; $J $H index --threads 48 --logs 6 >> "$R/index.log" 2>&1 || die index
st "fetch"; $J $H fetch --threads 32 >> "$R/fetch.log" 2>&1 || die fetch
$J $H check-keys --n 40 >> "$R/fetch.log" 2>&1 || die check-keys
st "render"
for d in lb_hq_navtest lb_hq_navhard1; do $J $H render --data $d >> "$R/render.log" 2>&1 || die "render $d"; done
st done; touch "$R/DONE"
