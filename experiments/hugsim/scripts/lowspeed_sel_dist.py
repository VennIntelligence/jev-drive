#!/usr/bin/env python
"""Plan 1 s direction (phi, car frame) distributions in the exam baseline logs (cinque-fixed, no rule), used to choose the
selective low-speed rule parameters offline (plans/2026-10-04-lowspeed-ctrl-selective-prereg.md). Baseline logs only; no
closed-loop result of any rule arm is read.   box: $DATA_DIR/envs/hugsim/bin/python <this> <lowspeed_runs.csv> <out.json>"""
import csv, json, math, os, sys
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
norm = lambda n: n.removeprefix("scene").replace("-", "_").strip("_")
base = {norm(r["scenario"]): r for r in csv.DictReader(open(sys.argv[1])) if r["arm"] == "base"}
cls = {"ev14_spin": [], "ev14_non": [], "spin_pre": [], "spin_launch": [], "nonspin_launch": [], "nonspin_all": [], "nonspin_big": []}
for d in sorted((D / "runs/hugsim-exam/scored-op/cinque-fixed/zs").iterdir()):
    f = d / "zs_steps.jsonl"
    if norm(d.name) not in base or not f.exists():
        continue
    spin, start = base[norm(d.name)]["spin"] == "True", int(base[norm(d.name)]["start"])
    recs = [json.loads(x) for x in open(f)][1:]
    vs = [r["v"] for r in recs]
    ev = [0] + [k for k in range(1, len(vs)) if vs[k] >= 1 and max(vs[max(0, k - 8):k]) < 0.4]
    for k, r in enumerate(recs):
        p = np.array(r["plan"], float)
        if len(p) < 2 or np.linalg.norm(p[1]) < 0.3 or r["v"] >= 3.5:
            continue
        phi = math.degrees(math.atan2(p[1, 0], p[1, 1]))
        row = (abs(phi), r["v"])
        ks = [k - e for e in ev if 1 <= k - e <= 4]
        if ks:   # steps 1-4 after a launch event, labelled by whether that event spins (divergence 0-12 steps after it)
            cls["ev14_spin" if spin and any(0 <= start - e <= 12 for e in ev if 1 <= k - e <= 4) else "ev14_non"].append(row)
        in_launch = any(0 <= k - e <= 12 for e in ev)
        if spin:
            if k <= start:
                cls["spin_pre"].append(row)
                if in_launch:
                    cls["spin_launch"].append(row)
        else:
            cls["nonspin_all"].append(row)
            if in_launch:
                cls["nonspin_launch"].append(row)
out = {}
Q = [10, 25, 50, 75, 90, 95]
for k, v in cls.items():
    if not v:
        continue
    a = np.array([x[0] for x in v])
    out[k] = dict(n=len(a), q={q: round(float(np.percentile(a, q)), 2) for q in Q},
                  frac_le={d: round(float((a <= d).mean()), 3) for d in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)})
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 3.5)):
        b = np.array([x[0] for x in v if lo <= x[1] < hi])
        if len(b):
            out[k][f"v{lo}-{hi}"] = dict(n=len(b), med=round(float(np.median(b)), 2), p75=round(float(np.percentile(b, 75)), 2),
                                         p90=round(float(np.percentile(b, 90)), 2), frac_le_3=round(float((b <= 3).mean()), 3))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text("class,abs_phi_deg,v\n" + "\n".join(f"{k},{a:.3f},{v:.3f}" for k, rows in cls.items() for a, v in rows) + "\n")
print(json.dumps(out, indent=1))
