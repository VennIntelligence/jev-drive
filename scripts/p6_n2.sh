#!/usr/bin/env bash
# Night queue 2, N2 on N1's P6 frames (todos/2026-09-26-night-queue-2.md, [A-N2] 12:31): frame index, openpilot
# temporal / vision streams, labels, probes a / b / c, desire targets on x10 obstacle frames and the desire runner.
# GPU $GPU only (default 3), pinned to $NCPU cores no live process is pinned to (default 24). Run dir runs/p6/n2.
# Resumable: every step skips what exists (the openpilot runner skips finished streams).
set -euo pipefail
cd ~/data/jev-drive
R=$DATA_DIR/runs/p6/n2; mkdir -p "$R"; exec > >(tee -a "$R/log.txt") 2>&1
GPU=${GPU:-3} NCPU=${NCPU:-24} SET=carla_p6
CPUS=$(python3 - "$NCPU" <<'PY'
import os, sys
def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-"); out += range(int(a), int(b or a) + 1)
    return out
busy = set()
for d in os.listdir("/proc"):
    try:
        for line in open(f"/proc/{d}/status"):
            if line.startswith("Cpus_allowed_list"):
                c = parse(line.split(":")[1].strip())
                if len(c) < 100: busy |= set(c)
    except (OSError, ValueError): pass
mine = parse(open("/proc/self/status").read().split("Cpus_allowed_list:")[1].split()[0])
print(",".join(map(str, [c for c in mine if c not in busy][: int(sys.argv[1])])))
PY
)
echo "$(date '+%F %T') p6-n2 start: GPU $GPU, CPUs $CPUS" | tee -a "$R/gpus.txt"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 P5_SET=$SET
PY=".venv/bin/python" OP="$DATA_DIR/envs/openpilot/bin/python"
T="taskset -c $CPUS"
[[ -f $DATA_DIR/processed/$SET/index.parquet ]] || $T $PY -m jevdrive.p6 index
[[ -f $DATA_DIR/processed/$SET/op_plan.json ]] || $T $PY -c "from jevdrive import p5_openpilot as P; print(P.prepare())"
$T $OP scripts/p5_openpilot.py --arrays temporal vision hidden --out-sub op_streams_vis --workers ${OP_WORKERS:-8}
for m in cinque lebowski; do $T $PY -c "from jevdrive import p5_openpilot as P; print(P.finalize('$m', ('temporal', 'vision', 'hidden'), 'op_streams_vis'))"; done
[[ -f $DATA_DIR/processed/$SET/night2_labels.parquet ]] || $T $PY -m jevdrive.night2_n2 labels --set $SET --source p6 --workers $NCPU
$T $PY -m jevdrive.night2_n2 targets --set $SET --query "world == 'x10' and a"
$T $OP scripts/night2_desire.py --set $SET --workers 6
$T $PY -m jevdrive.night2_n2 desire --sets $SET
echo "$(date '+%F %T') p6-n2 done (probes: jevdrive.p6 n2probe)"
