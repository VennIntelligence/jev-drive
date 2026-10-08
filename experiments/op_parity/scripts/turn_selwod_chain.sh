#!/usr/bin/env bash
# op_parity turn selector on WOD val (plans/2026-10-08-turn-selwod-prereg.md). One self-advancing chain in tmux jev:
#   scripts/tmux_run.sh turn-selwod experiments/op_parity/scripts/turn_selwod_chain.sh      (STOP_AFTER=smoke | select | report)
# smoke (6 frames, s0: extract -> select, plan-level gates) -> full extract SH30 s0 / s1 (+ WLG-full s0 / s1) -> select -> report.
# No RFS is read before the `report` stage (the prereg is committed in between: STOP_AFTER=select).
# State: $DATA_DIR/runs/op_parity/turn_selwod/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_selwod
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$O/pool
status() { echo "$(date '+%F %T') op_parity turn-selwod: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; local ld=$L/$n; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="tsw-$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -rf "$ld"
        $CL submit --owner op_parity --name "tsw-$n" --log-dir "$ld" "$@" || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 30; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n/ERROR"; done; return 0; }

sub spans --vram 0.5 --cpu 2 --ram 16 -- $VPY $S/turn_selwod.py spans
waitdirs spans
status "smoke: extract 6 frames of SH30-F-s0 (builds the tapped TensorRT engine)"
sub x-smoke --vram 8 --cpu 8 --ram 32 -- $OP $S/turn_selwod.py extract --tag SH30-F-s0 --suffix _smoke --limit 6 --workers 6
waitdirs x-smoke
sub s-smoke --vram 4 --cpu 4 --ram 16 -- $PY $S/turn_selwod.py select --tag SH30-F-s0 --suffix _smoke
waitdirs s-smoke
[[ ${STOP_AFTER:-} == smoke ]] && { status "smoke done, stopped (STOP_AFTER=smoke)"; date > "$D/DONE"; exit 0; }

status "full extract: SH30-F-s0 / s1, WLG-full-s0 / s1 (479 rater frames each)"
for t in SH30-F-s0 SH30-F-s1 WLG-full-s0 WLG-full-s1; do
  sub x-$t --vram 8 --cpu 14 --ram 40 -- $OP $S/turn_selwod.py extract --tag $t --workers 12
done
for t in SH30-F-s0 SH30-F-s1 WLG-full-s0 WLG-full-s1; do
  waitdirs x-$t
  sub s-$t --vram 4 --cpu 4 --ram 16 -- $PY $S/turn_selwod.py select --tag $t
done
waitdirs s-SH30-F-s0 s-SH30-F-s1 s-WLG-full-s0 s-WLG-full-s1
[[ ${STOP_AFTER:-} == select ]] && { status "select done, stopped (STOP_AFTER=select)"; date > "$D/DONE"; exit 0; }

status "report"
sub report --vram 0.5 --cpu 8 --ram 32 -- $VPY $S/turn_selwod.py report
waitdirs report
status "done"; date > "$D/DONE"
