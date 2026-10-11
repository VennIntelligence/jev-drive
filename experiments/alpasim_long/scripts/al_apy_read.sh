#!/usr/bin/env bash
# ALPASIM-LONG amendment 2 read-out: APY10m10-AB-s0 (submission #1) at 18 / 10 decisions next to the held read's six checkpoints.
# Pool CPU job, after the two APY jobs:  -> $DATA_DIR/runs/alpasim_long/held/read_apy/{per_scene.csv, report.md, report.json}
set -euo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/alpasim_long/held; P=$DATA_DIR/third_party/alpasim/.venv/bin/python
R=$(python3 - "$O/read/per_scene.csv" <<'PY'
import csv, sys
runs = {}
for r in csv.DictReader(open(sys.argv[1])):
    runs.setdefault(r["label"], set()).add(r["run"])
print(" ".join(f"{k}={','.join(sorted(v))}" for k, v in sorted(runs.items())))
PY
)
$P experiments/alpasim_long/scripts/long_read.py collect --runs $R "APY10m10-AB-s0@18=$O/a18/APY10m10-AB-s0" "APY10m10-AB-s0@10=$O/a10/APY10m10-AB-s0" \
  --out "$O/read_apy" --jobs 8
$P experiments/alpasim_long/scripts/long_read.py report --in "$O/read_apy" --out "$O/read_apy/report.md" --json "$O/read_apy/report.json"
python3 - "$O/read_apy/per_scene.csv" "$O" <<'PY'
import csv, glob, os, sys
for r in csv.DictReader(open(sys.argv[1])):
    if "/a1" in r["run"] and r["run"].startswith(sys.argv[2]) and float(r["score"] or 0) > 0:
        for f in glob.glob(f"{r['run']}/rollouts/{r['scene']}/*/rollout.asl"): os.remove(f)
PY
