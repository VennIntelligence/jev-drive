#!/usr/bin/env bash
# P5 v1 index and label validity (todos/2026-09-25-reactivity-program/i1-p5v1.md), after p5v1-gen. Run it as slot
# p5v1-index. The v0 pipeline per expert (jevdrive/p5v1.py index -> processed/carla_p5v1_<expert>), then the
# label-validity tables and the two experts on the same pairs (research/results/p5-v1/). Pinned to 16 CPUs of the
# generation's lists (INDEX_CPUS) that no live process is pinned to.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
cpus=$(python3 - "${INDEX_CPUS:-52-103,156-189}" 16 <<'EOF'
import os, sys
def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
busy = set()
for d in os.listdir("/proc"):
    try:
        for line in open("/proc/%s/status" % d):
            if line.startswith("Cpus_allowed_list"):
                c = parse(line.split(":")[1].strip())
                if len(c) < 100:
                    busy |= set(c)
    except (OSError, ValueError):
        continue
print(",".join(str(c) for c in [c for c in parse(sys.argv[1]) if c not in busy][:int(sys.argv[2])]))
EOF
)
echo "index on CPUs $cpus"
taskset -c "$cpus" .venv/bin/python -m jevdrive.p5v1 index
taskset -c "$cpus" .venv/bin/python -m jevdrive.p5v1 validity
echo "$(date '+%Y-%m-%d %H:%M') [REACTIVITY/I1] p5v1-index done: research/results/p5-v1/ (label_validity_*.csv, experts_compare.csv) on the box" \
    >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"
