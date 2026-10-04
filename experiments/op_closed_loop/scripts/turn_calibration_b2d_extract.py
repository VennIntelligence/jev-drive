"""B2D tick table for the turn-calibration analysis: model curvature (action head, plan-derived) vs the curvature the dense route requires.

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/turn_calibration_b2d_extract.py [--out PATH]

Reads $DATA_DIR/runs/vlm_arb/arms/<arm>/attempts/<route>/1/{plans,ticks}.jsonl + route.json of every arm (CPU only); one row per non-warm plan tick
with v >= 1 m/s. Output: $DATA_DIR/runs/op_closed_loop/turn_calibration/b2d_ticks.npz.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
import turn_calibration_lib as L  # noqa: E402
from jevdrive.common import data_dir, n_cpus  # noqa: E402

ARMS = data_dir() / "runs/vlm_arb/arms"
A_FIX = (5.0, 10.0)


def one(d):
    d = Path(d)
    arm, route = d.parents[2].name, d.parent.name
    try:
        R = json.load(open(d / "route.json"))
        plans = [json.loads(x) for x in open(d / "plans.jsonl")]
        ticks = {t["frame"]: t for t in map(json.loads, open(d / "ticks.jsonl"))}
    except (OSError, ValueError):
        return []
    xy, cmd = np.array(R["xy"]), np.array(R["cmd"])
    if len(xy) < 10:
        return []
    P, g = L.resample(xy)
    kr = L.curvature(P)
    mans = L.maneuvers(P)
    # raw route index -> grid index (arc length of the raw point)
    s_raw = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    rows = []
    gi_prev = 0
    for p in plans:
        t = ticks.get(p["frame"])
        if t is None or "truth" not in t or p.get("warm") or p["v"] < 1.0:
            continue
        v = p["v"]
        tx, ty, tyaw = t["truth"]
        # nearest grid point in a window ahead of the previous projection (monotone progress)
        lo, hi = max(gi_prev - 10, 0), min(gi_prev + int(30 / 0.5) + 60, len(P))
        j = lo + int(np.argmin(np.linalg.norm(P[lo:hi] - [tx, ty], axis=1)))
        gi_prev = max(gi_prev, j)
        a_v = float(np.clip(v, 4.0, 20.0))
        # maneuver overlapping the lookahead window [j, j + a_v / 0.5]
        mt, ang, rmin, dnext = 0, np.nan, np.nan, np.nan
        for m in mans:
            if m["i0"] <= j + a_v / 0.5 and m["i1"] >= j - 1:
                mt, ang, rmin = L.turn_type(abs(m["angle"])), m["angle"], m["rmin"]
                break
            if m["i0"] > j and (np.isnan(dnext) or (m["i0"] - j) * 0.5 < dnext):
                dnext = (m["i0"] - j) * 0.5
        ox = p["op_xy"]
        plan_k = [2 * o[1] / max(o[0] ** 2 + o[1] ** 2, 1e-6) for o in ox[:2]]       # chord curvature at the 1 s / 2 s plan point (sign fixed in analysis)
        plan_a = [float(np.hypot(*o)) for o in ox[:2]]
        rows.append(dict(
            arm=arm, route=route, frame=p["frame"], v=v, act_k=p["act_k"], plan_k1=plan_k[0], plan_k2=plan_k[1], plan_a1=plan_a[0], plan_a2=plan_a[1],
            req_v=L.chord_kappa(P, j, a_v), req5=L.chord_kappa(P, j, 5.0), req10=L.chord_kappa(P, j, 10.0),
            req_p1=L.chord_kappa(P, j, max(plan_a[0], 2.0)), req_p2=L.chord_kappa(P, j, max(plan_a[1], 2.0)),
            kloc=float(kr[min(j + int(a_v / 0.5 / 2), len(P) - 1)]), mtype=mt, mang=ang, mrmin=rmin, dnext=dnext,
            cmd=int(cmd[min(int(p["ri"]), len(cmd) - 1)]),
            lat=p["lat"], why=str(p["lat_why"]), zone=bool(p["zone"]), desire=int(p["desire"]), intent=int(p.get("intent", 0)),
            img=str(p.get("img")), opc=bool(t.get("opc") is not None), steer=float(t["steer"]), s_route=float(g[j])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(data_dir() / "runs/op_closed_loop/turn_calibration/b2d_ticks.npz"))
    a = ap.parse_args()
    dirs = sorted(str(p.parent) for p in ARMS.glob("*/attempts/*/1/plans.jsonl") if re.match(r"(v2|eval|shadow|pilot)", p.parts[-5]))
    print(len(dirs), "runs")
    with ProcessPoolExecutor(min(n_cpus(), 64)) as ex:
        rows = [r for rs in ex.map(one, dirs, chunksize=4) for r in rs]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()
