#!/usr/bin/env bash
# op_parity navhard readout under protocol G (GIMM frames; results/navhard.md "Protocol G"): P0 (harness check vs op_guard `shipped`)
# and the full-run P2 checkpoints. The G front tokens of lb_navhard are already cached by pp_navhard_chain.sh's first prep call.
# State: $DATA_DIR/runs/op_parity/navhard_gimm/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/navhard_gimm; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
S=experiments/op_parity/scripts
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
status() { echo "$(date '+%F %T') op_parity navhard_gimm: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
status "bench plans + scoring"
"${B[@]}" run --model P0@gimm P2-F-s0@gimm P2-F-s1@gimm --bench navhard --wait || die "bench navhard"
$PY $S/pp_navhard.py report-gimm || die "report"
status "done"
date > "$D/DONE"
