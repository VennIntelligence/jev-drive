"""SPIN-FIX spin detector on controller rows (one rollout: arrays t, vx, vy, wz, delta), the lines of plans/2026-10-13-spin-fix-prereg.md.
  handoff  |vx wz| > 6 m/s^2 anywhere in (1.7, 3.2] s (steer-long amendment 4, the gains pilot's rule)
  fast     after 1.7 s: vx >= 5 m/s and |vx wz| >= 6 m/s^2 for >= 0.3 s (3 rows), or sideslip |vy| / vx >= 0.15 at vx >= 5 m/s
  slow     after 1.7 s: vx < 5 m/s and |steering state| >= 0.7 rad (full lock is 0.785) for >= 1.0 s (10 rows): circling at low speed
spin = any of the three; -> dict(spin, kind, t_first)."""
import numpy as np


def _run(m, n):
    c = 0
    for i, x in enumerate(m):
        c = c + 1 if x else 0
        if c >= n:
            return i - n + 1
    return -1


def detect(t, vx, vy, wz, delta):
    t, vx, vy, wz, delta = map(np.asarray, (t, vx, vy, wz, delta))
    A, post = np.abs(vx * wz), t > 1.7
    out = {}
    h = np.where(post & (t <= 3.2) & (A > 6))[0]
    out["handoff"] = float(t[h[0]]) if len(h) else None
    i = _run(post & (vx >= 5) & (A >= 6), 3)
    j = np.where(post & (vx >= 5) & (np.abs(vy) >= 0.15 * np.maximum(vx, 1e-3)))[0]
    tf = [float(t[i])] if i >= 0 else []
    tf += [float(t[j[0]])] if len(j) else []
    out["fast"] = min(tf) if tf else None
    k = _run(post & (vx < 5) & (np.abs(delta) >= 0.7), 10)
    out["slow"] = float(t[k]) if k >= 0 else None
    ks = [(v, n) for n, v in out.items() if v is not None]
    return dict(spin=bool(ks), kind="+".join(n for _, n in sorted(ks)), t_first=min(v for v, _ in ks) if ks else None, **out)
