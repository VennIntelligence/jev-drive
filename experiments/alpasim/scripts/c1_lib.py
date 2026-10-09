"""C1 shared helpers (Mac side): load the pickles written by c1_extract.py and compute per-scene geometry.

Frames: `actors` and `logged` are AABB-centre poses in the rollout's local frame; driver trajectories, routes and the controller trace are
rear-axle (rig) poses. CENTER is the rear axle -> AABB centre offset along the heading, measured from the logs.
Privileged data (map, other actors, the logged future) is used here for labelling only.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[3]
TMP = ROOT / "tmp/c1"
EGO_L, EGO_W, CENTER = 5.176, 2.297, 1.461
FLAGS = ("collision_at_fault", "offroad", "left_corridor_laterally")
SHORT = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor"}


@lru_cache(None)
def load(name: str):
    return pickle.load(open(TMP / f"{name}.pkl", "rb"))


def runs():
    return {"SH30": load("sh30_logs"), "AP2": load("ap2_logs")}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def to_center(xyh):
    """Rear-axle poses (n, 3) -> AABB-centre poses."""
    xyh = np.atleast_2d(np.asarray(xyh, float))
    return np.c_[xyh[:, 0] + CENTER * np.cos(xyh[:, 2]), xyh[:, 1] + CENTER * np.sin(xyh[:, 2]), xyh[:, 2]]


def box(x, y, h, L=EGO_L, W=EGO_W) -> Polygon:
    c = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]]) @ rot(h).T + [x, y]
    return Polygon(c)


def interp_pose(tr, t):
    """tr (n, 4) t_us, x, y, yaw -> (x, y, yaw) at t (us), clamped."""
    t = np.clip(t, tr[0, 0], tr[-1, 0])
    h = np.unwrap(tr[:, 3])
    return np.array([np.interp(t, tr[:, 0], tr[:, 1]), np.interp(t, tr[:, 0], tr[:, 2]), np.interp(t, tr[:, 0], h)])


def speed_at(tr, t, dt=0.2e6):
    a, b = interp_pose(tr, t - dt), interp_pose(tr, t + dt)
    t0, t1 = np.clip([t - dt, t + dt], tr[0, 0], tr[-1, 0])
    return float(np.hypot(*(b[:2] - a[:2])) / max((t1 - t0) * 1e-6, 1e-3))


def arc(xy):
    return float(np.hypot(*np.diff(np.asarray(xy)[:, :2], axis=0).T).sum()) if len(xy) > 1 else 0.0


def signed_lat(line_xy, p):
    """Signed lateral offset of point p from a polyline (left of the direction of travel positive) and the arc position."""
    ls = LineString(line_xy)
    s = ls.project(Point(p))
    q, q2 = np.array(ls.interpolate(max(s - 0.5, 0)).coords[0]), np.array(ls.interpolate(min(s + 0.5, ls.length)).coords[0])
    d = q2 - q
    n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
    return float((np.asarray(p) - np.array(ls.interpolate(s).coords[0])) @ n), float(s)


def road(m) -> dict:
    """Map record of c1_extract.py -> road-area polygons with their vertices, lane polygons with their centre lines, and the unions."""
    ar, av, ln, lc = [], [], [], []
    for a in m["areas"]:
        try:
            ar.append(shapely.make_valid(Polygon(a["ext"], [h for h in a["holes"] if len(h) >= 3])))
            av.append(np.concatenate([a["ext"]] + list(a["holes"])))
        except Exception:
            pass
    for l in m["lanes"]:
        try:
            ok = l["left"] is not None and l["right"] is not None and len(l["left"]) > 1 and len(l["right"]) > 1
            ln.append(shapely.make_valid(Polygon(np.r_[l["left"], l["right"][::-1]])) if ok else LineString(l["center"]).buffer(1.85))
            lc.append(LineString(l["center"]))
        except Exception:
            pass
    return dict(areas=ar, verts=av, lanes=ln, centers=lc, area=unary_union(ar).buffer(0.01).buffer(-0.01), lane=unary_union(ln))


def onroad(rd, pose_c) -> bool:
    """Is the ego footprint inside the mapped drivable area (all road areas and lanes within the extraction radius)."""
    p = box(*pose_c)
    return bool(rd["area"].covers(p) or rd["lane"].covers(p))


def onroad_scorer(rd, pose_c) -> bool:
    """Replica of eval.scorers.offroad on nuPlan maps (no road edges): a lane within 6 m containing the footprint, else the union of
    those lanes, else the union of the road areas that have a vertex within 25 m of the ego (ROAD_AREA_QUERY_DIST_M)."""
    p, c = box(*pose_c), Point(pose_c[:2])
    near = [g for g, cl in zip(rd["lanes"], rd["centers"]) if cl.distance(c) <= 6.0]
    if any(g.contains(p) for g in near) or (len(near) > 1 and unary_union(near).contains(p)):
        return True
    ar = [g for g, v in zip(rd["areas"], rd["verts"]) if (np.hypot(*(v - pose_c[:2]).T) <= 25.0).any()]
    return bool(ar) and bool(unary_union(ar).buffer(0.001).buffer(-0.001).covers(p))


def metric(o, name):
    m = o["metrics"].get(name)
    return None if m is None else m[np.argsort(m[:, 0])]


def first_event(o, flag):
    """First scored sim time (us) at which a zero flag fires, None if never."""
    if flag == "collision_at_fault":
        a, b = metric(o, "collision_front"), metric(o, "collision_lateral")
        v = np.maximum(a[:, 1], b[:, 1])
        t = a[:, 0]
    else:
        m = metric(o, flag)
        t, v = m[:, 0], m[:, 1]
    rel = metric(o, "eval_relevant")
    ok = (v > 0) & (rel[:, 1] > 0)
    return int(t[ok][0]) if ok.any() else None


def zero_flags(o):
    return [f for f in FLAGS if o["summary"]["score_metrics"].get(f)]


def gt(o):
    return o["logged"][0]["traj"]


def ego(o):
    return o["actors"]["EGO"]


def turn_deg(o) -> float:
    g = gt(o)
    return float(np.degrees(np.unwrap(g[:, 3])[-1] - g[0, 3]))


def turn_bucket(d) -> str:
    d = abs(d)
    return "< 5" if d < 5 else "5-20" if d < 20 else "20-45" if d < 45 else "> 45"


def plan_c(o, k):
    """Decision k's returned trajectory as AABB-centre poses (n, 4) t, x, y, yaw."""
    tr = o["drive"][k]["traj"]
    return np.c_[tr[:, 0], to_center(tr[:, 1:4])]
