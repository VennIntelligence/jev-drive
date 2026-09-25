"""Stall episodes (v < 0.5 m/s for >= 10 s) per route: duration, displacement, collisions within 7 m, plan classes, CoC.
Part of the Alpamayo B2D stall diagnosis (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md).
Box: $DATA_DIR/envs/carla/bin/python - <run dir> ... < this file; prints JSON / text to stdout."""
import json, re, sys, glob, os
import numpy as np
D = sys.argv[1]
rows = []
for rdir in sorted(glob.glob(D + "/attempts/*")):
    atts = sorted([a for a in glob.glob(rdir + "/*") if os.path.exists(a + "/results.json")], key=lambda a: int(a.rsplit("/", 1)[1]))
    if not atts: continue
    A = atts[-1]
    try:
        T = [json.loads(l) for l in open(A + "/ticks.jsonl")]
        P = [json.loads(l) for l in open(A + "/plans.jsonl")]
        rec = json.load(open(A + "/results.json"))["_checkpoint"]["records"][0]
    except Exception as e:
        continue
    if not T or not P: continue
    v = np.array([t["v"] for t in T]); tt = np.array([t["t"] for t in T])
    thr = np.array([t["throttle"] for t in T])
    tr = np.array([t["truth"][:2] if "truth" in t else [np.nan, np.nan] for t in T])
    st = v < 0.5
    edges = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
    eps = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if (b - a) * .05 >= 10]
    cols = []
    for k in ("collisions_layout", "collisions_vehicle", "collisions_pedestrian"):
        for s in rec["infractions"].get(k, []):
            m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", s)
            typ = re.search(r"type=(\S+)", s)
            cols.append((k, typ.group(1) if typ else "?", float(m.group(1)), float(m.group(2))))
    for a, b in eps:
        p0 = tr[a]
        # displacement during the episode (does the car creep / push?)
        disp = float(np.nanmax(np.linalg.norm(tr[a:b] - p0, axis=1)))
        near = [c for c in cols if np.hypot(c[2] - p0[0], c[3] - p0[1]) < 7.0]
        pl = [p for p in P if tt[a] <= p["t"] < tt[b - 1]]
        x3 = np.array([p["path"][11][0] for p in pl]); xmin = np.array([min(q[0] for q in p["path"][:12]) for p in pl])
        thrp = np.array([p["zoo_pid"]["throttle"] for p in pl]) if pl else np.zeros(0)
        cots = [p["cot"] for p in pl]
        rows.append(dict(route=rec["route_id"].split("_")[1], scen=rec["scenario_name"].rsplit("_", 1)[0], status=rec["status"][:22],
                         t0=round(float(tt[a]), 1), dur=round((b - a) * .05, 1), final=bool(b == len(st)), disp=round(disp, 2),
                         collided=";".join("%s:%s" % (c[0].split("_")[1], c[1]) for c in near),
                         frac_thr=round(float((thr[a:b] > 0).mean()), 2), n_plans=len(pl),
                         stay=round(float(((x3 < 1.2) & (xmin > -0.3)).mean()), 2) if pl else None,
                         rev=round(float((xmin < -0.3).mean()), 2) if pl else None,
                         go=round(float(((x3 >= 1.2) & (xmin > -0.3)).mean()), 2) if pl else None,
                         nav=sum(1 for p in pl if p.get("nav_text")) / max(len(pl), 1),
                         top_cot=max(set(cots), key=cots.count)[:80] if cots else ""))
json.dump(rows, sys.stdout)
