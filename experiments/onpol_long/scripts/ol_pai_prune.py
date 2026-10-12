#!/usr/bin/env python3
"""ONPOL-LONG PAI read (Tokyo): drop rollout.asl of scored, unfailed rollouts in finished chunk dirs (ot2_loop / read1 rule; metrics.parquet,
the mp4 and the summary stay). Walks /data/runs/onpol_long/pai281/<run>/runs/<chunk>/ with a DONE file; one pass per call."""
import glob
import json
import os

n = b = 0
for d in glob.glob("/data/runs/onpol_long/pai281/*/runs/*/"):
    f = d + "sim/aggregate/results-summary.json"
    if not (os.path.exists(d + "DONE") and os.path.exists(f)):
        continue
    for r in json.load(open(f))["rollouts"]:
        if not r.get("failure_reason") and (r.get("score") or 0) > 0:
            for a in glob.glob(f"{d}sim/rollouts/{r['clipgt_id']}/*/rollout.asl"):
                b += os.path.getsize(a)
                os.remove(a)
                n += 1
print(f"pruned {n} rollout.asl, {b / 2**30:.1f} GiB")
