#!/usr/bin/env bash
# Readout (d) of op-adapt H: an adapted checkpoint in HUGSIM closed loop on decision 96's 10 spin scenarios, PR #57 controller,
# no de-rotation rule (plans/2026-10-04-op-adapt-H-prereg.md). The checkpoint becomes a serving ONNX (op_adapt_l's op_l_onnx.py,
# no adapter), one resident Cinque server serves it, zs_run drives the scenarios; then h_hugsim_report.py counts spins.
# Usage (box, tmux):  [WORKERS=3] experiments/op_adapt_h/scripts/h_hugsim.sh <run tag | O>
# Optional env: SCEN=<scenario list file> (default derot_spin10.txt), OPTS=<json controller opts> (default {}), SUFFIX=<out dir / tag suffix>, REPORT=<report script>.
# Files: $DATA_DIR/runs/op_adapt_H/hugsim/<tag>/{STATUS,DONE,ERROR,results.csv,bench-runs.json}; servers/traces live in the canonical bench directory.
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
TAG=$1
H=$DATA_DIR/runs/op_adapt_H
OUT=$H/hugsim/$TAG${SUFFIX:-}
SCEN=${SCEN:-experiments/hugsim/scripts/derot_spin10.txt}
OPTS=${OPTS-'{}'}
W=${WORKERS:-3}
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/ERROR"
source scripts/bench_lane.sh
fail() { echo "$(date +%T) $*" | tee "$OUT/ERROR"; exit 1; }
st() { echo "$(date +%T) $*" | tee "$OUT/STATUS"; }
MODEL=H-$TAG; [[ $TAG == O ]] && MODEL=cinque
st "bench scenarios from $SCEN"
bench_hugsim "$MODEL" exam "cinque-fixed-H$TAG${SUFFIX:-}" "$SCEN" "$W" fixed "$OPTS" || fail "bench hugsim"
$HPY ${REPORT:-experiments/op_adapt_h/scripts/h_hugsim_report.py} "$OUT" "cinque-fixed-H$TAG${SUFFIX:-}" || fail "report"
st "done"
touch "$OUT/DONE"
