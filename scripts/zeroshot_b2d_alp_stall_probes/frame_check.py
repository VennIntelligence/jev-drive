"""Plan frame / time-base check: plan x, y and heading at 1 s against the simulator-truth motion 1 s later (speed >= 3 m/s).
Part of the Alpamayo B2D stall diagnosis (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md).
Box: $DATA_DIR/envs/carla/bin/python - <run dir> ... < this file; prints JSON / text to stdout."""
import json, sys, glob, os
import numpy as np
D = sys.argv[1]
rows = []
for rdir in sorted(glob.glob(D + "/attempts/*"))[:60]:
    atts = sorted([a for a in glob.glob(rdir + "/*") if os.path.exists(a + "/results.json")], key=lambda a: int(a.rsplit("/", 1)[1]))
    if not atts: continue
    A = atts[-1]
    T = [json.loads(l) for l in open(A + "/ticks.jsonl")]; P = [json.loads(l) for l in open(A + "/plans.jsonl")]
    if not T or not P or "truth" not in T[0]: continue
    tt = np.array([t["t"] for t in T]); tr = np.array([t["truth"] for t in T]); v = np.array([t["v"] for t in T])
    for p in P:
        i = int(np.searchsorted(tt, p["t"] - 1e-6)); j = i + 20
        if j >= len(T) or p["speed"] < 3: continue
        x0, y0, h0 = tr[i]; dx, dy = tr[j, 0] - x0, tr[j, 1] - y0
        fwd = np.cos(h0) * dx + np.sin(h0) * dy; left = np.sin(h0) * dx - np.cos(h0) * dy   # CARLA y right -> left positive
        dyaw = -(np.unwrap([h0, tr[j, 2]])[1] - h0)   # CCW positive
        # plan heading at 1 s from the path tangent
        path = np.array(p["path"]); tan = path[4] - path[2]
        rows.append((p["speed"], path[3, 0], path[3, 1], fwd, left, np.arctan2(tan[1], tan[0]), dyaw))
R = np.array(rows)
print("n", len(R))
print("plan x@1s / speed median %.3f ; realized fwd@1s / speed median %.3f" % (np.median(R[:, 1] / R[:, 0]), np.median(R[:, 3] / R[:, 0])))
m = np.abs(R[:, 2]) > 0.3
print("lateral sign agreement plan y@1s vs realized left@1s: %.3f (n=%d), corr %.3f" % ((np.sign(R[m, 2]) == np.sign(R[m, 4])).mean(), m.sum(), np.corrcoef(R[:, 2], R[:, 4])[0, 1]))
m = np.abs(R[:, 5]) > 0.05
print("heading sign agreement plan vs realized yaw change: %.3f (n=%d), corr %.3f" % ((np.sign(R[m, 5]) == np.sign(R[m, 6])).mean(), m.sum(), np.corrcoef(R[:, 5], R[:, 6])[0, 1]))
