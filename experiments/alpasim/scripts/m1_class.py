"""M1: mechanism class of the zero-score rollouts of any run, by decision 202's rules (c1_review.mechanism, unchanged), for the registered
"M-class zeros halved" line (plans/2026-10-09-m1-yaw-damping-prereg.md). Mac side.

  m1_class.py --map <map pkl> [--noshift <dec npz>:<driver>] NAME=<logs pkl> [NAME=<logs pkl> ...] [--scenes <txt>] [--json out]

logs pkl = c1_extract.py logs of a run (only scenes that kept their rollout.asl are in it: the failed ones). Two counts per run:
M (C1's rule) and M' (the same without the route clause: log heading change < 8 deg and the ego >= 0.4 m off the logged path at the event,
after S / H / F). --noshift: zeros among the scenes whose first route waypoint stays within 2 m laterally (rig frame) in that driver's run.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_lib as L  # noqa: E402
import c1_review as RV  # noqa: E402


def classify(o, name, s, rd):
    r = RV.base(o, name, s)
    f = r["flags"][0]
    e = RV.collision(o) | {"flag": "collision"} if f == "collision_at_fault" else RV.lateral(o, f, rd)
    m = RV.mechanism(r, e)
    m2 = m if m in "SHF" else ("M" if abs(r["turn"]) < 8 and 0.4 <= abs(e["lat"]) < 4.9 else m)
    return dict(scene=s, flag=e["flag"], mech=m, mech_noroute=m2, turn=r["turn"], bend=r["bend"], lat=e["lat"], t=e["t"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"), ap.add_argument("--map", required=True), ap.add_argument("--noshift"), ap.add_argument("--scenes"), ap.add_argument("--json")
    a = ap.parse_args()
    M = pickle.load(open(a.map, "rb"))
    keep = set(Path(a.scenes).read_text().split()) if a.scenes else None
    flat = None
    if a.noshift:
        f, drv = a.noshift.rsplit(":", 1)
        Z = np.load(f)
        ry = {}
        for s, num in zip(Z[f"{drv}|scene"], Z[f"{drv}|num"]):
            ry.setdefault(s, []).append(num[6])
        flat = {s for s, y in ry.items() if np.nanmax(np.abs(y)) <= 2.0 and (keep is None or s in keep)}
    out = {}
    print("| run | zeros classified | S | H | F | M | T | R | M' (no route clause) | zeros on scenes without route shift" + (f" ({len(flat)})" if flat else "") + " |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for sp in a.runs:
        name, f = sp.split("=", 1)
        rows = [classify(o, name, s, L.road(M[s])) for s, o in sorted(pickle.load(open(f, "rb")).items())
                if o["summary"]["score"] == 0 and (keep is None or s in keep) and s in M]
        c = {k: sum(r["mech"] == k for r in rows) for k in "SHFMTR"}
        nf = sum(r["scene"] in flat for r in rows) if flat is not None else None
        out[name] = dict(rows=rows, counts=c, m_noroute=sum(r["mech_noroute"] == "M" for r in rows), zeros_noshift=nf)
        print(f"| {name} | {len(rows)} | " + " | ".join(str(c[k]) for k in "SHFMTR") + f" | {out[name]['m_noroute']} | {'' if nf is None else nf} |")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
