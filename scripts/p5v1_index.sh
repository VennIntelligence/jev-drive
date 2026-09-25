#!/usr/bin/env bash
# P5 v1 index and label validity (todos/2026-09-25-reactivity-program/i1-p5v1.md), after p5v1-gen. Run it as slot
# p5v1-index. The v0 pipeline per expert (jevdrive/p5v1.py index -> processed/carla_p5v1_<expert>), then the
# label-validity tables and the two experts on the same pairs (research/results/p5-v1/).
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
PIN=(taskset -c "${INDEX_CPUS:-100-149}")
"${PIN[@]}" .venv/bin/python -m jevdrive.p5v1 index
"${PIN[@]}" .venv/bin/python -m jevdrive.p5v1 validity
echo "$(date '+%Y-%m-%d %H:%M') [REACTIVITY/I1] p5v1-index done: research/results/p5-v1/ (label_validity_*.csv, experts_compare.csv)" \
    >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
