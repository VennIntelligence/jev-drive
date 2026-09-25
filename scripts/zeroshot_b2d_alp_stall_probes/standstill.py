"""Controller response to standstill plans (v < 0.3 m/s) by plan class; with a 2nd arg only plans >= 2 s before the first collision.
Part of the Alpamayo B2D stall diagnosis (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md).
Box: $DATA_DIR/envs/carla/bin/python - <run dir> ... < this file; prints JSON / text to stdout."""
import json, sys, glob, os, re
import numpy as np
D = sys.argv[1]; FREE = len(sys.argv) > 2
agg = {}
lunge = []
for rdir in sorted(glob.glob(D + "/attempts/*")):
    atts = sorted([a for a in glob.glob(rdir + "/*") if os.path.exists(a + "/results.json")], key=lambda a: int(a.rsplit("/", 1)[1]))
    if not atts: continue
    A = atts[-1]
    try:
        T = [json.loads(l) for l in open(A + "/ticks.jsonl")]; P = [json.loads(l) for l in open(A + "/plans.jsonl")]
    except Exception: continue
    if not T or not P: continue
    try: rec = json.load(open(A + "/results.json"))["_checkpoint"]["records"][0]
    except Exception: continue
    tcol = 1e9
    if "truth" in T[0]:
        tr = np.array([t["truth"] for t in T]); ctr = tr[:, :2] + 1.388633 * np.c_[np.cos(tr[:, 2]), np.sin(tr[:, 2])]
        for k in ("collisions_layout", "collisions_vehicle", "collisions_pedestrian"):
            for s_ in rec["infractions"].get(k, []):
                m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", s_)
                tcol = min(tcol, T[int(np.argmin(np.hypot(*(ctr - [float(m.group(1)), float(m.group(2))]).T)))]["t"])
    elif rec["infractions"].get("collisions_vehicle") or rec["infractions"].get("collisions_layout"):
        tcol = -1.0 if FREE else 1e9
    tt = np.array([t["t"] for t in T]); v = np.array([t["v"] for t in T]); thr = np.array([t["throttle"] for t in T]); br = np.array([t["brake"] for t in T])
    for p in P:
        if p["speed"] >= 0.3 or (FREE and p["t"] >= tcol - 2.0): continue
        x = [q[0] for q in p["path"]]
        c = "rev" if min(x[:12]) < -0.3 else ("stay" if x[11] < 1.2 else "go")
        i = int(np.searchsorted(tt, p["t"] - 1e-6))
        w = slice(i, min(i + 10, len(T)))
        a = agg.setdefault(c, [0, 0.0, 0.0, 0.0])
        a[0] += 1; a[1] += float((thr[w] > 0).mean()); a[2] += float(br[w].mean()); a[3] += float(v[min(i + 10, len(T) - 1)])
        if c == "go":
            # plan speed over 0.5-3 s vs achieved speed 0.5 s and 1.0 s later
            lunge.append((x[11] / 3.0, float(v[min(i + 10, len(T) - 1)]), float(v[min(i + 20, len(T) - 1)]), float(thr[w].mean())))
res = {c: dict(n=a[0], frac_ticks_throttle=a[1] / a[0], mean_brake=a[2] / a[0], v_after_0p5s=a[3] / a[0]) for c, a in agg.items()}
L = np.array(lunge) if lunge else np.zeros((0, 4))
if len(L):
    res["go_lunge"] = dict(n=len(L), plan_v3_median=float(np.median(L[:, 0])), v_0p5_median=float(np.median(L[:, 1])),
                           v_1p0_median=float(np.median(L[:, 2])), ratio_0p5_median=float(np.median(L[:, 1] / np.maximum(L[:, 0], 1e-3))),
                           frac_overshoot_1p1=float((L[:, 1] > 1.1 * L[:, 0]).mean()))
print(json.dumps(res, indent=1))
