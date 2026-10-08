#!/usr/bin/env bash
# op_parity nc-taxonomy (plans/2026-10-09-nc-taxonomy-prereg.md, overnight N1): SH30's NC / TTC failure taxonomy on navtest and navhard
# stage 1, and the same-path longitudinal scaling oracle. CPU only; the family is scored by `python -m jevdrive.bench score-poses
# --traffic non_reactive` on the pool, capped at JOBS pool jobs so the other lanes keep their cores.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh nc-tax experiments/op_parity/scripts/nc_tax_chain.sh):
#   select + replay (navtest, navhard) -> family -> stage 0: failing tokens x 14 keys (identity gate) -> report
#   -> stage 1: all tokens x 14 keys (identity gate) -> report. STOP_AFTER=stage0 ends before stage 1.
# State: $DATA_DIR/runs/op_parity/nc_tax/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes the scoring (finished chunks stay).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/nc_tax
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
NPY=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
S=experiments/op_parity/scripts/nc_tax.py
JOBS=${JOBS:-3}
status() { echo "$(date '+%F %T') op_parity nc-tax: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
score() {   # score <stage> -> $O/score_<stage>.csv
  read -ra K < "$O/keys.txt"
  "$VPY" -m jevdrive.bench score-poses --poses "$O/poses.npz" --keys "${K[@]}" --tokens "$O/tokens_$1.txt" --out "$O/score_$1.csv" \
      --traffic non_reactive --jobs "$JOBS" --owner op_parity --wait || die "score-poses $1"
  [[ -f $O/score_$1.csv ]] || die "score-poses $1 wrote no CSV"
  "$VPY" $S gate --stage "$1" --score "$O/score_$1.csv" || die "identity gate $1"
}

for b in navtest navhard; do
  status "select $b"; "$VPY" $S select --bench $b || die "select $b"
  status "replay $b"; "$NPY" $S replay --bench $b || die "replay $b"
done
status "family"; "$VPY" $S family || die "family"
status "stage 0: failing tokens"; score fail
status "report (stage 0)"; "$VPY" $S report --score-fail "$O/score_fail.csv" || die "report stage 0"
[[ ${STOP_AFTER:-} == stage0 ]] && { status "stopped after stage 0 (STOP_AFTER)"; date > "$D/DONE"; exit 0; }
status "stage 1: all tokens"; score all
status "report (stage 1)"; "$VPY" $S report --score-fail "$O/score_fail.csv" --score-all "$O/score_all.csv" || die "report stage 1"
status "done"; date > "$D/DONE"
