#!/usr/bin/env bash
# op_parity turn-ceiling de-water (plans/2026-10-08-turn-ceiling-dewater-prereg.md): de-watered ceilings and cross-fitted selectors from the
# per-token candidate scores of turn_ceiling (no new scoring). CPU only, at most half of the cgroup cores.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh turn-dewater experiments/op_parity/scripts/turn_dewater_chain.sh):
#   ceiling -> select (smoke: one repeat, nothing reported) -> select -> selreport.
# State: $DATA_DIR/runs/op_parity/turn_dewater/chain/{STATUS, DONE, ERROR, log.txt}; tables and figures in $DATA_DIR/runs/op_parity/turn_dewater/report.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_dewater
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
S=experiments/op_parity/scripts/turn_dewater.py
A=(--out "$O/report" --figs "$O/report/figs")
status() { echo "$(date '+%F %T') op_parity turn-dewater: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
for st in ${STAGES:-ceiling smoke select selreport}; do
  status "$st"
  if [[ $st == smoke ]]; then $PY $S select --smoke "${A[@]}" || die smoke
  else $PY $S "$st" "${A[@]}" || die "$st"; fi
done
status "done"; date > "$D/DONE"
