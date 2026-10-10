#!/usr/bin/env bash
# CORR0 (plans/2026-10-10-corr0-prereg.md): navtest map labels -> arms -> identity gate on 10 tokens -> all turn tokens -> report.
# CPU only. One self-advancing chain in tmux jev (scripts/tmux_run.sh corr0 experiments/corridor/scripts/corr_chain.sh);
# STOP_AFTER=poses | s10 | score ends early. State: $DATA_DIR/runs/corridor/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes the scoring.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/corridor
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
NPY=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
S=experiments/corridor/scripts
status() { echo "$(date '+%F %T') corridor corr0: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
stop() { [[ ${STOP_AFTER:-} == "$1" ]] && { status "stopped after $1 (STOP_AFTER)"; date > "$D/DONE"; exit 0; }; }
score() {   # score <tokens file> <tag>
  read -ra K < "$O/keys.txt"
  "$VPY" -m jevdrive.bench score-poses --poses "$O/poses.npz" --keys "${K[@]}" --tokens "$1" --out "$O/score$2.csv" \
      --traffic non_reactive --owner corridor --wait || die "score-poses $2"
  [[ -f $O/score$2.csv ]] || die "score-poses $2 wrote no CSV"
  "$VPY" $S/corr.py gate --score "$O/score$2.csv" --tag "$2" || die "identity gate $2"
}
if [[ ! -f $O/geom.pkl || -n ${REBUILD:-} ]]; then status "geom"; "$NPY" $S/corr_geom.py build || die "geom"; fi
status "poses"; "$VPY" $S/corr.py poses || die "poses"
stop poses
status "score: 10 tokens"; score "$O/tokens_s10.txt" _s10
stop s10
status "score: turn tokens"; score "$O/tokens.txt" _turn
stop score
status "report"; "$VPY" $S/corr.py report --score "$O/score_turn.csv" || die "report"
status "done"; date > "$D/DONE"
