#!/usr/bin/env bash
# The ONPOL-PAI row pipeline end to end on the Tokyo box: rows (alpasim-base) -> cold (driver image) -> report. Re-runnable: scenes
# that already have a file are skipped, so run it again whenever new run dirs have finished.
#   onp_rows.sh <pass> '<run dir glob>' ['<glob>' ...]        e.g.  onp_rows.sh pol '/data/runs/onpol_pai/pass1/*/runs/*'
# Env: JOBS (workers per stage; default = threads - 8, at most 16), plus onp_env.sh's. Extra `rows` flags: ROWS_ARGS (e.g. "--limit 4").
# Writes <ONP_OUT>/<pass>/{<scene>.npz, STATUS, DONE | ERROR, log.txt, times.jsonl, failed/} and <ONP_OUT>/rows_<pass>.md.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); P=$1; shift
OUT=${ONP_OUT:-${DATA_DIR:-/data}/runs/onpol_pai/rows}; mkdir -p "$OUT/$P"
j=${JOBS:+--jobs $JOBS}
fail() { echo "$(date '+%F %T') ERROR $*" | tee "$OUT/$P/STATUS"; echo "$*" > "$OUT/$P/ERROR"; exit 1; }
# shellcheck disable=SC2086
bash "$here/onp_env.sh" scripts/onp_rows.py rows --pass "$P" --runs "$@" $j ${ROWS_ARGS:-} 2> >(grep -v "^Creating\|it/s\]" >&2) || fail "stage rows crashed"
# shellcheck disable=SC2086
bash "$here/onp_env.sh" --drv scripts/onp_rows.py cold --pass "$P" $j || fail "stage cold crashed"
bash "$here/onp_env.sh" --drv scripts/onp_rows.py report --pass "$P" > "$OUT/$P/report.log" 2>&1 || fail "report crashed, see report.log"
tail -1 "$OUT/$P/report.log"
