#!/usr/bin/env python3
"""op-drive dlon question, part 2: per arm (drive / dlon) over the same routes, how often does openpilot's lead head fire without a
ground-truth vehicle on the route (phantom lead), how far is openpilot's own 2 s path from the car's heading on lane-follow steps,
and the lane-line confidence. Same logs as op_drive_dlon_diag.py.
    python3 scripts/op_drive_dlon_pairs.py [--root DIR] [--tags dv d1]"""
import argparse
import glob
import json
import os

import numpy as np

from op_drive_dlon_diag import CAM_TO_BUMPER, load


def stats(rs):
    v = np.array([r["v"] for r in rs if not r["warm"]])
    mv = [r for r in rs if not r["warm"] and r["v"] > 1.0]
    st = [r for r in rs if not r["warm"] and r["v"] <= 1.0]

    def phantom(r):
        c = r.get("ctx", {})
        if r["lp"][0] <= 0.5:
            return False
        head = r["lead"][0][0] - CAM_TO_BUMPER
        return "lead_gap" not in c or abs(c["lead_gap"] - head) > 5

    lf = [r for r in mv if r["v"] > 2.0 and not r["zone"] and r.get("lat_why") in (None, "div")]
    return dict(n=len(v), v=v.mean() if len(v) else float("nan"), still=len(st) / max(len(v), 1),
                ph_mv=np.mean([phantom(r) for r in mv]) if mv else float("nan"),
                ph_all=np.mean([phantom(r) for r in rs if not r["warm"]]) if len(v) else float("nan"),
                y2=np.median([abs(r["op_xy"][1][1]) for r in lf]) if lf else float("nan"),
                lane=np.median([min(r["lane"][1], r["lane"][2]) for r in lf]) if lf else float("nan"),
                plan5=np.median([r["vplan"][4] for r in lf]) if lf else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ["DATA_DIR"], "runs", "op_drive", "arms"))
    ap.add_argument("--tags", nargs="+", default=["dv", "d1"])
    a = ap.parse_args()
    print("| run | route | steps | mean v | standstill share | phantom lead (moving) | phantom lead (all) | op y@2s med (lane-follow, m) | inner lane prob min med | plan v5 med (lane-follow) |")
    print("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    agg = {}
    for tag in a.tags:
        for arm in ("drive", "dlon"):
            for d in sorted(glob.glob(os.path.join(a.root, f"{tag}-{arm}-s*"))):
                for rid, rs in load(d).items():
                    s = stats(rs)
                    agg.setdefault(arm, []).append(s)
                    print(f"| {os.path.basename(d)} | {rid} | {s['n']} | {s['v']:.2f} | {s['still']:.2f} | {s['ph_mv']:.3f} | {s['ph_all']:.3f} | {s['y2']:.2f} | {s['lane']:.2f} | {s['plan5']:.2f} |")
    print()
    print("| arm | runs | mean v | standstill share | phantom lead (moving) | op y@2s med | inner lane prob min | plan v5 med |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for arm, L in agg.items():
        m = lambda k: np.nanmean([x[k] for x in L])  # noqa: E731
        print(f"| {arm} | {len(L)} | {m('v'):.2f} | {m('still'):.2f} | {m('ph_mv'):.3f} | {m('y2'):.2f} | {m('lane'):.2f} | {m('plan5'):.2f} |")


if __name__ == "__main__":
    main()
