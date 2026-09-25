#!/usr/bin/env python3
"""Adapter acceptance of the Alpamayo B2D re-smoke (smoke2) and the pre-registered choice of the full-run config.
Criteria: todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md, section 6 (frozen before the runs).

    python zeroshot_b2d_alp_smoke2_check.py f1=<arm dir> f1f2b=<arm dir> --out <choice.json>

Per arm, from plans.jsonl / ticks.jsonl / results.json of the last attempt of every route:
  A0  infrastructure: every requested route finished (status "finished"), no CARLA restart, plans logged
  A1  reverse plans at free standstill (v < 0.3 m/s, >= 2 s before the route's first collision): share of throttle
      ticks in the 0.5 s after the plan <= 5 %
  A2  no collision at v < 3 m/s preceded within 2 s by a standstill reverse plan that got throttle
Reported, not gating: over-throttle share (ticks with throttle > 0 while v > 1.1 x the desired speed of the plan in
force, i.e. where the per-tick Zoo PID would already brake); start ratio v(+1 s) / (x@3s / 3 s) over free-standstill
go plans; all low-speed collisions after any standstill throttle; stall episodes >= 10 s starting at a collision; DS/RC.
Choice: "f1f2b" if it passes A0-A2 (the UniAD / VAD per-tick cadence, the default); else "f1" if it passes; else none
(exit 1). Driving score never decides.
NumPy + standard library, Python 3.8.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

REAR_TO_CENTER = 1.388633


def load(p):
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def arm_metrics(run):
    summ = json.load(open(os.path.join(run, "summary.json"))) if os.path.exists(os.path.join(run, "summary.json")) else {}
    a0 = []
    if not summ:
        a0.append("no summary.json")
    if summ.get("routes_never_finished"):
        a0.append("never finished: %s" % summ["routes_never_finished"])
    if summ.get("restarts"):
        a0.append("server restarts: %s" % summ["restarts"])
    rev, go, lowcol, revcol, pinned, ds, rc, over = [], [], 0, 0, 0, [], [], []
    for rdir in sorted(glob.glob(os.path.join(run, "attempts", "*"))):
        atts = sorted([a for a in glob.glob(rdir + "/*") if os.path.exists(a + "/results.json")],
                      key=lambda a: int(a.rsplit("/", 1)[1]))
        if not atts:
            a0.append("%s: no results" % rdir)
            continue
        A = atts[-1]
        rr = os.path.join(A, "route_result.json")
        if os.path.exists(rr) and json.load(open(rr)).get("status") != "finished":
            a0.append("%s: %s" % (A, json.load(open(rr)).get("status")))
        T, P = load(A + "/ticks.jsonl"), load(A + "/plans.jsonl")
        try:
            rec = json.load(open(A + "/results.json"))["_checkpoint"]["records"][0]
        except (KeyError, IndexError, ValueError):
            a0.append("%s: no record" % A)
            continue
        if not T or not P:
            a0.append("%s: no plans/ticks" % A)
            continue
        ds.append(rec["scores"]["score_composed"])
        rc.append(rec["scores"]["score_route"])
        tt = np.array([q["t"] for q in T])
        v = np.array([q["v"] for q in T])
        thr = np.array([q["throttle"] for q in T])
        tr = np.array([q["truth"] for q in T])
        pt = np.array([p["t"] for p in P])
        pdes = np.array([(p.get("zoo_pid") or {}).get("desired_speed", np.nan) for p in P])
        for q in T:
            d = q.get("zoo_desired")
            if d is None:
                j = int(np.searchsorted(pt, q["t"] + 1e-6)) - 1
                d = pdes[j] if j >= 0 else np.nan
            if np.isfinite(d):
                over.append(q["throttle"] > 0 and q["v"] > 1.1 * d)
        ctr = tr[:, :2] + REAR_TO_CENTER * np.c_[np.cos(tr[:, 2]), np.sin(tr[:, 2])]
        tcol = []
        for k in ("collisions_layout", "collisions_vehicle", "collisions_pedestrian"):
            for s in rec["infractions"].get(k, []):
                m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", s)
                tcol.append(float(tt[int(np.argmin(np.hypot(*(ctr - [float(m.group(1)), float(m.group(2))]).T)))]))
        t_first = min(tcol) if tcol else 1e9
        stand = []
        for p in P:
            if p["speed"] >= 0.3:
                continue
            x = np.array([q[0] for q in p["path"]])
            i = int(np.searchsorted(tt, p["t"] - 1e-6))
            kind = "rev" if x[:12].min() < -0.3 else ("stay" if x[11] < 1.2 else "go")
            stand.append((p["t"], i, kind, bool((thr[i:i + 10] > 0).any()) if i < len(thr) else False))
            if p["t"] >= t_first - 2.0:
                continue
            if kind == "rev":
                rev.append(float((thr[i:i + 10] > 0).mean()))
            elif kind == "go" and i + 20 < len(v):
                go.append(float(v[i + 20] / max(x[11] / 3.0, 1e-3)))
        for tc in tcol:
            j = int(np.searchsorted(tt, tc))
            if v[max(j - 1, 0)] < 3 and any(tc - 2.0 <= s[0] <= tc and s[3] for s in stand):
                lowcol += 1
            if v[max(j - 1, 0)] < 3 and any(tc - 2.0 <= s[0] <= tc and s[3] and s[2] == "rev" for s in stand):
                revcol += 1
        st = v < 0.5
        edges = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if (b - a) * 0.05 >= 10 and any(tt[a] - 3 <= tc <= tt[b - 1] for tc in tcol):
                pinned += 1
    a1 = float(np.mean(rev)) if rev else 0.0
    a3 = float(np.median(go)) if go else None
    return {"A0_ok": not a0, "A0_issues": a0[:20], "A1_rev_throttle_share": round(a1, 4), "A1_n": len(rev),
            "A1_ok": a1 <= 0.05, "start_ratio_median": None if a3 is None else round(a3, 3), "start_n": len(go),
            "A2_rev_throttle_collisions": revcol, "A2_ok": revcol == 0,
            "over_throttle_share": round(float(np.mean(over)), 4) if over else None,
            "lowspeed_collisions_after_standstill_throttle": lowcol,
            "pinned_stalls": pinned, "n_routes": len(ds), "DS_mean": round(float(np.mean(ds)), 2) if ds else None,
            "RC_mean": round(float(np.mean(rc)), 2) if rc else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=run dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for spec in a.arms:
        name, _, path = spec.partition("=")
        res[name] = arm_metrics(path)
        res[name]["pass"] = res[name]["A0_ok"] and res[name]["A1_ok"] and res[name]["A2_ok"]
    choice = next((k for k in ("f1f2b", "f1") if k in res and res[k]["pass"]), None)
    res["choice"] = choice
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if choice else 1)


if __name__ == "__main__":
    main()
