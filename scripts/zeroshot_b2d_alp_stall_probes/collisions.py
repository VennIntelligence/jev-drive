"""Every official collision: its tick (closest approach of the ego centre to the reported location), speed, and the last four plans with the Zoo PID output.
Part of the Alpamayo B2D stall diagnosis (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md).
Box: $DATA_DIR/envs/carla/bin/python - <run dir> ... < this file; prints JSON / text to stdout."""
import json, re, sys, glob, os
import numpy as np
D = sys.argv[1]
out = []
for rdir in sorted(glob.glob(D + "/attempts/*")):
    atts = sorted([a for a in glob.glob(rdir + "/*") if os.path.exists(a + "/results.json")], key=lambda a: int(a.rsplit("/", 1)[1]))
    if not atts: continue
    A = atts[-1]
    try:
        T = [json.loads(l) for l in open(A + "/ticks.jsonl")]
        P = [json.loads(l) for l in open(A + "/plans.jsonl")]
        rec = json.load(open(A + "/results.json"))["_checkpoint"]["records"][0]
    except Exception:
        continue
    if not T or not P: continue
    tt = np.array([t["t"] for t in T]); v = np.array([t["v"] for t in T])
    tr = np.array([t["truth"] for t in T])
    ctr = tr[:, :2] + 1.388633 * np.c_[np.cos(tr[:, 2]), np.sin(tr[:, 2])]
    pt = np.array([p["t"] for p in P])
    for k in ("collisions_layout", "collisions_vehicle", "collisions_pedestrian"):
        for s in rec["infractions"].get(k, []):
            m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", s); typ = re.search(r"type=(\S+)", s)
            d = np.hypot(*(ctr - [float(m.group(1)), float(m.group(2))]).T)
            i = int(np.argmin(d))
            j = int(np.searchsorted(pt, tt[i], side="right")) - 1   # plan in force at the collision
            prev = [P[q] for q in range(max(j - 3, 0), j + 1)]        # the plans of the last ~2 s
            def cls(p):
                x = [q[0] for q in p["path"]]
                if min(x[:12]) < -0.3: return "rev"
                if x[11] < 1.2: return "stay"
                return "go"
            out.append(dict(route=rec["route_id"].split("_")[1], kind=k.split("_")[1], typ=typ.group(1) if typ else "?",
                            t=round(float(tt[i]), 2), dmin=round(float(d[i]), 2), v=round(float(v[max(i - 1, 0)]), 2),
                            vmax2s=round(float(v[max(i - 40, 0):i + 1].max()), 2),
                            plans=[(round(p["t"], 1), round(p["speed"], 2), cls(p), round(p["path"][11][0], 2),
                                    round(p["zoo_pid"]["desired_speed"], 2), p["zoo_pid"]["throttle"], p["zoo_pid"]["brake"]) for p in prev],
                            cot=P[j]["cot"][:90] if j >= 0 else ""))
json.dump(out, sys.stdout)
