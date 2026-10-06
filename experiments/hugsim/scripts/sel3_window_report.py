#!/usr/bin/env python
"""Summarise derot.dpos (replay vs native plan, unrotated control) per run directory: python sel3_window_report.py <dir>..."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from jevdrive.bench.compat import read_rows, trace_dir

for d in map(pathlib.Path, sys.argv[1:]):
    dp, per = [], {}
    if (d / "results.csv").exists():
        last = {(r["tag"], r["scenario"]): r for r in read_rows(d / "results.csv") if r["end"] != "crash"}
        files = {trace_dir(r, d) / "zs_steps.jsonl" for r in last.values()}
    else:
        files = set(d.rglob("zs_steps.jsonl"))
    for f in sorted(files):
        v = [json.loads(l).get("derot", {}).get("dpos") for l in open(f)]
        v = [x for x in v if x is not None]
        per[f.parent.name] = v
        dp += v
    a = np.array(dp) if dp else np.zeros(1)
    print(f"{d.name}: replays {len(dp)} in {sum(bool(v) for v in per.values())} scenes; dpos m: median {np.median(a):.4f} p95 {np.percentile(a, 95):.3f} max {a.max():.3f}; "
          f"share > 0.05 m {np.mean(a > 0.05):.3f}")
    for k, v in per.items():
        if v:
            print(f"  {k}: n {len(v)} median {np.median(v):.4f} max {max(v):.3f}")
