"""Offline replay of the vendored Zoo PID on logged standstill plans, as logged and with the forward-only (v >= 0) plan clamp.
Part of the Alpamayo B2D stall diagnosis (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md).
Box: $DATA_DIR/envs/carla/bin/python - <run dir> ... < this file; prints JSON / text to stdout."""
import json, sys, glob, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/data/jev-drive/scripts"))
from b2d_zoo_pid import PIDController
T6 = np.arange(1, 7) * 0.5
TIMES = np.arange(1, 21) * 0.25

def forward_only(path):
    p = np.vstack([[0.0, 0.0], path]); d = np.diff(p, axis=0)
    keep = d[:, 0] > 0
    return np.cumsum(d * keep[:, None], axis=0)

def zoo(path, speed):
    q = np.stack([np.interp(T6, TIMES, path[:, k]) for k in range(2)], -1)
    wp = np.stack([-q[:, 1], q[:, 0]], -1)
    pid = PIDController()
    s, t, b, m = pid.control_pid(wp, np.float64(speed), np.array([0.0, 30.0]))
    return m["desired_speed"], bool(b), float(t)

out = {}
for run in sys.argv[1:]:
    agg = {}
    for rdir in sorted(glob.glob(run + "/attempts/*")):
        for A in glob.glob(rdir + "/*"):
            f = A + "/plans.jsonl"
            if not os.path.exists(f): continue
            for line in open(f):
                try: p = json.loads(line)
                except ValueError: continue
                if p["speed"] >= 0.3: continue
                path = np.array(p["path"]); x = path[:, 0]
                c = "rev" if x[:12].min() < -0.3 else ("stay" if x[11] < 1.2 else "go")
                d0, b0, t0 = zoo(path, p["speed"]); d1, b1, t1 = zoo(forward_only(path), p["speed"])
                a = agg.setdefault(c, np.zeros(5)); a += [1, b0, b1, d0, d1]
    out[os.path.basename(run)] = {c: dict(n=int(a[0]), brake_now=round(a[1] / a[0], 3), brake_clamped=round(a[2] / a[0], 3),
                                          desired_now=round(a[3] / a[0], 3), desired_clamped=round(a[4] / a[0], 3)) for c, a in agg.items()}
print(json.dumps(out, indent=1))
