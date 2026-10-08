#!/usr/bin/env bash
# op_parity turn-ceiling (plans/2026-10-08-turn-ceiling-prereg.md): privileged best-of-K ceiling of a small trajectory family around SH30's plan
# on the navtest > 20 deg tokens. CPU only; scoring is `python -m jevdrive.bench score-poses` on the pool.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh turn-ceiling experiments/op_parity/scripts/turn_ceiling_chain.sh):
#   family -> 24 tokens (identity gate) -> 300 tokens (identity gate, measured cost, stage-1 key set) -> all 3 154 tokens -> report -> navtrain counts.
# A failed identity gate stops the chain before any other score is read. STOP_AFTER=t300 ends after stage 0.
# State: $DATA_DIR/runs/op_parity/turn_ceiling/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (score-poses keeps finished chunks).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_ceiling
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
S=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
status() { echo "$(date '+%F %T') op_parity turn-ceiling: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
score() {   # score <stage> <keys ...> -> $O/score_<stage>.csv
  local st=$1; shift
  "${B[@]}" score-poses --poses "$O/poses.npz" --keys "$@" --tokens "$O/tokens_$st.txt" --out "$O/score_$st.csv" --owner op_parity --wait \
      || die "score-poses $st"
  [[ -f $O/score_$st.csv ]] || die "score-poses $st wrote no CSV"
}

status "family"
"$VPY" $S/turn_ceiling.py family || die "family"
read -ra K33 < "$O/keys_F33.txt"

status "stage 0a: 24 tokens x ${#K33[@]} keys"
score t24 "${K33[@]}"
"$VPY" $S/turn_ceiling.py gate --stage t24 --score "$O/score_t24.csv" || die "identity gate t24"

status "stage 0: 300 tokens x ${#K33[@]} keys"
score t300 "${K33[@]}"
"$VPY" $S/turn_ceiling.py gate --stage t300 --score "$O/score_t300.csv" || die "identity gate / cost t300"
[[ ${STOP_AFTER:-} == t300 ]] && { status "stopped after stage 0 (STOP_AFTER)"; date > "$D/DONE"; exit 0; }

read -ra K1 < "$O/stage1_keys.txt"
status "stage 1: all tokens x ${#K1[@]} keys"
score all "${K1[@]}"

status "report"
$PY $S/turn_ceiling.py report --score "$O/score_all.csv" --out "$O/report" || die "report"
status "navtrain counts (read-only)"
"$VPY" $S/turn_ceiling.py navtrain || die "navtrain"
status "done"; date > "$D/DONE"
