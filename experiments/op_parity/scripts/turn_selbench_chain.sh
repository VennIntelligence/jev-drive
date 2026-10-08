#!/usr/bin/env bash
# op_parity turn selector bench (plans/2026-10-08-turn-selector-bench-prereg.md). One self-advancing chain in tmux jev:
#   scripts/tmux_run.sh turn-selbench experiments/op_parity/scripts/turn_selbench_chain.sh        (STOP_AFTER=smoke stops after the identity gate)
# refit N7 -> G-repro -> smoke (ts0 + tsB on 6 whole logs) -> G-id -> [full: navtest A / B x 2 seeds, navhard A / B x 2 seeds] -> report.
# State: $DATA_DIR/runs/op_parity/turn_selbench/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (finished jobs and stages are skipped).
# Waits only on the pool jobs' DONE / ERROR files and on `bench --wait` (no bare `wait`, no pgrep).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_selbench
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
S=experiments/op_parity/scripts
L=$O/pool
SMOKE=navsim/op-parity-tsbench-smoke
status() { echo "$(date '+%F %T') op_parity turn-selbench: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; local ld=$L/$n; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="tsb-$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -rf "$ld"
        $CL submit --owner op_parity --name "tsb-$n" --log-dir "$ld" "$@" || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 30; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n/ERROR"; done; return 0; }

status "refit N7 (decision 191 protocol) + G-repro"
sub refit --vram 16 --cpu 4 --ram 40 -- $PY $S/turn_selbench.py refit
waitdirs refit
sub repro --vram 16 --cpu 4 --ram 40 -- $PY $S/turn_selbench.py repro
waitdirs repro
grep -q '"ok": true' $O/gate_repro.json || die "G-repro failed"

status "smoke: ts0 + tsB on $SMOKE (6 whole logs)"
"${B[@]}" run --model SH30-F-s0@warp:ts0 SH30-F-s1@warp:ts0 SH30-F-s0@warp:tsB --bench navtest --subset $SMOKE --wait || die "smoke bench"
sub idcheck --vram 0.5 --cpu 2 --ram 8 -- $PY $S/turn_selbench_report.py idcheck --subset $SMOKE
waitdirs idcheck
grep -q '"ok": true' $O/gate_id.json || die "G-id failed"
[[ ${STOP_AFTER:-} == smoke ]] && { status "smoke done, stopped (STOP_AFTER=smoke)"; date > "$D/DONE"; exit 0; }

status "full navtest: A / B x 2 seeds"
"${B[@]}" run --model SH30-F-s0@warp:tsB SH30-F-s1@warp:tsB SH30-F-s0@warp:tsA SH30-F-s1@warp:tsA --bench navtest --wait || die "bench navtest"
status "full navhard: B / A x 2 seeds"
"${B[@]}" run --model SH30-F-s0@gimm:tsB SH30-F-s1@gimm:tsB SH30-F-s0@gimm:tsA SH30-F-s1@gimm:tsA --bench navhard --wait || die "bench navhard"

status "report"
sub report --vram 0.5 --cpu 8 --ram 40 -- $PY $S/turn_selbench.py report
waitdirs report
status "done"; date > "$D/DONE"
