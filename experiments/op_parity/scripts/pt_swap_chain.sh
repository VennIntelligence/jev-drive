#!/usr/bin/env bash
# op_parity path-timing-swap (plans/2026-10-09-path-timing-swap-prereg.md, lane SW1): the 2 x 2 swap of {path shape, speed profile} between
# every stored navtest plan and the log. CPU only; cells are scored by `python -m jevdrive.bench score-poses --traffic non_reactive` on the
# pool, capped at JOBS pool jobs so the other lanes keep their cores.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh pt-swap experiments/op_parity/scripts/pt_swap_chain.sh):
#   build -> stage 0: 48 tokens x all keys (identity gate) -> all tokens (identity gate) -> select + replay of the DAC-failing rows -> report.
#   STOP_AFTER=build | s0 | score ends early (the report is only run once the pre-registered rules are fixed).
# State: $DATA_DIR/runs/op_parity/pt_swap/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes the scoring (finished chunks stay).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/pt_swap
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
NPY=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
S=experiments/op_parity/scripts/pt_swap.py
JOBS=${JOBS:-3}
status() { echo "$(date '+%F %T') op_parity pt-swap: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
stop() { [[ ${STOP_AFTER:-} == "$1" ]] && { status "stopped after $1 (STOP_AFTER)"; date > "$D/DONE"; exit 0; }; }
score() {   # score <stage> -> $O/score_<stage>.csv
  read -ra K < "$O/keys.txt"
  "$VPY" -m jevdrive.bench score-poses --poses "$O/poses.npz" --keys "${K[@]}" --tokens "$O/tokens_$1.txt" --out "$O/score_$1.csv" \
      --traffic non_reactive --jobs "$JOBS" --owner op_parity --wait || die "score-poses $1"
  [[ -f $O/score_$1.csv ]] || die "score-poses $1 wrote no CSV"
  "$VPY" $S gate --stage "$1" --score "$O/score_$1.csv" || die "identity gate $1"
}

if [[ ! -f $O/poses.npz || -n ${REBUILD:-} ]]; then status "build"; "$VPY" $S build || die "build"; fi
stop build
status "stage 0: 48 tokens"; score s0
stop s0
status "stage 1: all tokens"; score all
stop score
status "select"; "$VPY" $S select --score "$O/score_all.csv" || die "select"
status "replay"; "$NPY" $S replay || die "replay"
status "report"; "$VPY" $S report --score "$O/score_all.csv" || die "report"
status "done"; date > "$D/DONE"
