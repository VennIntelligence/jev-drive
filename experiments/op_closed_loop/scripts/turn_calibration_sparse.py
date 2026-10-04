"""Dense vs sparse route: what a pure-pursuit steering would do through the B2D junction turns when it is given a real-car-like sparse route.

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/turn_calibration_sparse.py

Per route maneuver (|total turn| >= 25 deg, from the dense route) a kinematic-bicycle pure-pursuit run (constant speed, wheelbase 2.86 m, |steer| <= 0.6 rad) starts 25 m before the turn on the
dense centreline, aligned, and drives until 15 m after it. Paths given to the controller:
  dense      the leaderboard's dense route (what the shipped zones steer by)
  lb50       the leaderboard's own downsample_route(.., 50 m): points at command changes / lane changes / every 50 m, linear polyline between them
  road_<s>   a road-level polyline from map matching: the dense route decimated to 10 m with correlated lateral noise (sd s metres, 30 m correlation length), 20 draws; road_0 / road5_0 =
             no noise, decimated to 10 m / 5 m (the decimation alone)
Metrics versus the dense centreline over the turn window: peak cross-track error, peak commanded curvature, integral of |kappa| (the heading change commanded), heading change delivered.
Output: $DATA_DIR/runs/op_closed_loop/turn_calibration/sparse.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
import turn_calibration_lib as L  # noqa: E402
sys.path.insert(0, str(REPO / "lib"))
from route_poly import noisy_road, poly_resample  # noqa: E402,F401  (shared with experiments/op_route_cmd: one navigation-noise definition)
from jevdrive.common import data_dir  # noqa: E402

ARMS = data_dir() / "runs/vlm_arb/arms"
WB, DELTA_MAX, DT = 2.8605, 0.6, 0.05
SPEEDS = (3.0, 5.0, 8.0)
SIGMAS = (1.0, 2.0, 4.0)
N_DRAW = 20


def pursue(path, start_xy, start_psi, v, end_xy_idx_s, ld):
    """Pure pursuit on a polyline (resampled 0.25 m); returns the trajectory (x, y, psi, kappa_cmd)."""
    Pp, gp = poly_resample(path, 0.25)
    x, y, psi = start_xy[0], start_xy[1], start_psi
    prog = int(np.argmin(np.linalg.norm(Pp - start_xy, axis=1)))
    out = []
    for _ in range(int(120.0 / DT)):
        lo, hi = max(prog - 4, 0), min(prog + 80, len(Pp))
        prog = lo + int(np.argmin(np.linalg.norm(Pp[lo:hi] - [x, y], axis=1)))
        tgt = min(prog + int(ld / 0.25), len(Pp) - 1)
        d = Pp[tgt] - [x, y]
        dist = max(float(np.hypot(*d)), 1e-3)
        alpha = np.arctan2(d[1], d[0]) - psi
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        k = 2 * np.sin(alpha) / dist
        delta = np.clip(np.arctan(WB * k), -DELTA_MAX, DELTA_MAX)
        k = np.tan(delta) / WB
        x += v * DT * np.cos(psi)
        y += v * DT * np.sin(psi)
        psi += v * DT * k
        out.append((x, y, psi, k))
        if prog >= len(Pp) - 3:
            break
    return np.array(out)


def metrics(traj, D, it0, it1, v):
    """Cross-track error to the dense centreline, peak curvature and integral of |kappa| over the turn window (dense index it0 - 20 ... it1)."""
    xy = traj[:, :2]
    d = np.linalg.norm(xy[:, None, :] - D[None, :, :], axis=2)
    near, dev = d.argmin(1), d.min(1)
    w = (near >= it0 - 20) & (near <= it1)
    if not w.any():
        w[:] = True
    return dict(peak=float(dev[w].max()), mean=float(dev[w].mean()), kpeak=float(np.abs(traj[w, 3]).max()),
                kint=float(np.abs(traj[w, 3]).sum() * v * DT), dpsi=float(np.degrees(traj[-1, 2] - traj[0, 2])))


def main():
    rng = np.random.default_rng(0)
    routes = {}
    for f in sorted(ARMS.glob("*/attempts/*/1/route.json")):
        rid = f.parts[-3]
        if rid not in routes:
            R = json.load(open(f))
            if len(R["xy"]) > 10:
                routes[rid] = (np.array(R["xy"]), np.array(R["cmd"]))
    rows = []
    for rid, (xy, cmd) in routes.items():
        P, g = L.resample(xy)
        mans = [m for m in L.maneuvers(P) if abs(m["angle"]) >= 25]
        if not mans:
            continue
        D, gd = poly_resample(xy, 0.25)
        ds = L.leaderboard_downsample(cmd, 50.0, xy)
        variants = {"dense": xy, "lb50": xy[ds]}
        for mi, m in enumerate(mans):
            s0 = max(g[m["i0"]] - 25.0, 0.0)
            s1 = g[m["i1"]] + 15.0
            i0 = int(np.searchsorted(gd, s0))
            i1 = min(int(np.searchsorted(gd, s1)), len(D) - 1)
            psi0 = np.arctan2(*(D[min(i0 + 4, len(D) - 1)] - D[i0])[::-1])
            seg_xy = xy  # the controller gets the whole route; the run stops at the turn's exit point
            for v in SPEEDS:
                ld = max(3.0, v + 1.5)
                for name, path in list(variants.items()) + [("road_%g" % sg, None) for sg in SIGMAS] + [("road_0", None), ("road5_0", None)]:
                    draws = [path] if path is not None else [noisy_road(xy, float(name.split("_")[1]), rng, decim=5.0 if name.startswith("road5") else 10.0)
                                                             for _ in range(1 if name.endswith("_0") else N_DRAW)]
                    for dr, pth in enumerate(draws):
                        tr = pursue(pth, D[i0], psi0, v, None, ld)
                        # keep the part up to the exit point of the turn
                        j = int(np.argmin(np.linalg.norm(tr[:, :2] - D[i1], axis=1)))
                        tr = tr[:j + 1]
                        r = metrics(tr, D, int(np.searchsorted(gd, g[m["i0"]])), i1, v)
                        rows.append(dict(route=rid, man=mi, angle=m["angle"], rmin=m["rmin"], v=v, variant=name, draw=dr, **r,
                                         turn_deg=float(np.degrees(L.heading(P)[min(m["i1"] + 2, len(P) - 1)] - L.heading(P)[max(m["i0"] - 2, 0)]))))
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    p = data_dir() / "runs/op_closed_loop/turn_calibration/sparse.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    print(len(rows), "rows,", len(set(out["route"])), "routes ->", p)


if __name__ == "__main__":
    main()
