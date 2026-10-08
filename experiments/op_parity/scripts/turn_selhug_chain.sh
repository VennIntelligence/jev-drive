#!/usr/bin/env bash
# op_parity turn selector in HUGSIM (plans/2026-10-08-turn-selector-hugsim-prereg.md). One self-advancing chain in tmux jev:
#   scripts/tmux_run.sh turn-selhug experiments/op_parity/scripts/turn_selhug_chain.sh      (STOP_AFTER=id|pilot stops after that gate)
# G-eqv -> TRT engines -> G-id (ts0, 11 scenarios) -> G-pilot (tsB, 11 scenarios) -> full (tsB x 2 seeds, all64) -> report.
# State: $DATA_DIR/runs/op_parity/turn_selhug/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes. Waits only on `bench --wait` and pool job files.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_selhug
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JV=$DATA_DIR/envs/jevdrive/bin/python
CL="$JV -m jevdrive.cl"
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
S=experiments/op_parity/scripts
L=$O/pool
P=spec_plan_smooth
status() { echo "$(date '+%F %T') op_parity turn-selhug: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; local ld=$L/$n; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="tsh-$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -rf "$ld"
        $CL submit --owner op_parity --name "tsh-$n" --log-dir "$ld" "$@" || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 30; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n/ERROR"; done; return 0; }
S11=$S/turn_selhug_s11.txt

status "G-eqv"
E=$DATA_DIR/runs/op_parity/turn_selbench/gate_eqv_s0.json
[[ -f $E ]] && grep -q '"ok": true' $E || { sub eqv --vram 24 --cpu 4 --ram 40 -- $PY $S/turn_selhug.py check --seed 0; waitdirs eqv; }
grep -q '"ok": true' $E || die "G-eqv failed"
status "TensorRT engines of the tapped ONNX"
sub trt --vram 20 --cpu 6 --ram 60 -- $DATA_DIR/envs/openpilot/bin/python $S/turn_selhug.py trt; waitdirs trt

status "G-id: ts0 on the 11 scenarios (seed 0)"
"${B[@]}" run --model SH30-F-s0:ts0 --bench hugsim --preset $P --scenarios $S11 --wait || die "bench ts0 s11"
$JV $S/turn_selhug.py idcheck --seed 0 || die "idcheck"
grep -q '"ok": true' $O/gate_id_s0.json || die "G-id failed (see $O/gate_id_s0.json)"
[[ ${STOP_AFTER:-} == id ]] && { status "G-id passed, stopped"; date > "$D/DONE"; exit 0; }

status "G-pilot: tsB on the 11 scenarios (seed 0)"
"${B[@]}" run --model SH30-F-s0:tsB --bench hugsim --preset $P --scenarios $S11 --wait || die "bench tsB s11"
$JV $S/turn_selhug.py pilot --seed 0 || die "pilot"
grep -q '"ok": true' $O/gate_pilot_s0.json || die "G-pilot failed (see $O/gate_pilot_s0.json)"
[[ ${STOP_AFTER:-} == pilot ]] && { status "G-pilot passed, stopped"; date > "$D/DONE"; exit 0; }

status "full: tsB x 2 seeds, all64"
"${B[@]}" run --model SH30-F-s0:tsB SH30-F-s1:tsB --bench hugsim --preset $P --scenarios all64 --wait || die "bench tsB all64"
status "report"
$JV $S/turn_selhug.py report || die "report"
status "done"; date > "$D/DONE"
