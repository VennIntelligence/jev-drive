"""Q3: does the native model already know where the road edge is? (plan: experiments/skill_pack/plans/2026-10-03-...)

CPU, navsim2 env. For every stage-2 token: openpilot's predicted road edges (cached `heads` of the native plan file,
road_edges (2, 33, 2) = [y, z] at X_IDXS, [left, right]) in the ego frame, the map's drivable-area boundary on the same
cross-sections (x = 5 ... 30 m, ego frame, the contiguous drivable interval around the centre line), and the native plan's own
path (33 knots, converted with the lever-arm adapter) -> roadedge.pkl.

Edge points into the ego frame: x_ego = x_op + d_x, y_ego = d_y - y_op (openpilot y is to the right; d = camera on the vehicle).
"""
import argparse
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402

X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
XS = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
_G = {}


def _init():
    z = np.load(L.NATIVE_PLANS)
    info = __import__("json").loads(str(z["info"]))
    s = info["heads_slices"]["road_edges"]
    _G.update(names=z["names"].tolist(), heads=z["heads"], plan_pos=z["plan_pos"], plan_yaw=z["plan_yaw"], sl=s,
              cp=L.cache_paths(), idx={e["token"]: e for e in L.index()})
    _G["row"] = {t: i for i, t in enumerate(_G["names"])}


def map_boundary(mc, xs):
    """Left / right boundary y (ego frame, left +) of the contiguous drivable-area interval around y = 0 at each x."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    am = mc.drivable_area_map
    area = am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.DRIVABLE_AREA, S.CARPARK_AREA])
    u = unary_union([am._geometries[k] for k in area])
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    g = lambda x, y: np.array([o[0] + c * x - s * y, o[1] + s * x + c * y])  # noqa: E731
    out = np.full((len(xs), 3), np.nan)     # [left, right, 1 if the centre line is inside the drivable area at this x]
    for k, x in enumerate(xs):
        inter = u.intersection(LineString([g(x, -40), g(x, 40)]))
        if inter.is_empty:
            continue
        segs = [inter] if inter.geom_type == "LineString" else [q for q in getattr(inter, "geoms", []) if q.geom_type == "LineString"]
        iv = []
        for q in segs:
            p = np.asarray(q.coords)
            yl = [(-s * (px - o[0]) + c * (py - o[1])) for px, py in p]   # lateral in ego frame (x fixed)
            iv.append((min(yl), max(yl)))
        iv.sort()
        hit = [v for v in iv if v[0] - 1e-6 <= 0 <= v[1] + 1e-6]
        v = hit[0] if hit else min(iv, key=lambda v: min(abs(v[0]), abs(v[1])))
        out[k] = (v[1], v[0], float(bool(hit)))
    return out


def work(t):
    i = _G["row"][t]
    mc = L.load_cache(_G["cp"][t])
    cam = np.asarray(_G["idx"][t]["cams"][-1]["CAM_F0"]["t"], float)
    h = _G["heads"][i]
    sl = _G["sl"]
    # heads layout: slices are offsets into the concatenated non-hidden heads; road_edges = mu (264/2) then log-std
    base = sl if isinstance(sl, int) else sl[0]
    re = h[base:base + 132].reshape(2, 33, 2)
    std = np.exp(np.minimum(h[base + 132:base + 264], 11)).reshape(2, 33, 2)
    ex = X_IDXS + cam[0]                              # ego-frame x of the edge points
    ey = cam[1] - re[..., 0]                          # (2, 33): [left, right] in ego frame, left +
    pos, yaw = _G["plan_pos"][i], _G["plan_yaw"][i]
    plan = I.to_rear(pos, yaw, I.T_IDXS, cam[:2], I.T_IDXS, "lever", "linear")     # (33, 3) x, y, yaw at the 33 knots
    return dict(token=t, ex=ex, ey=ey, estd=std[..., 0], plan=plan, map=map_boundary(mc, XS), xs=XS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    toks = [e["token"] for e in L.index() if e["stage"] == "two"]
    if a.limit:
        toks = toks[:a.limit]
    with mp.get_context("fork").Pool(a.procs, initializer=_init) as pool:
        rows = pool.map(work, toks, chunksize=8)
    pickle.dump({r["token"]: r for r in rows}, open(L.OUT / ("roadedge_debug.pkl" if a.limit else "roadedge.pkl"), "wb"))
    print(len(rows))
