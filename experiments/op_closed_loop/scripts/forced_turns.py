"""Classify every route turn >= 25 deg of the Bench2Drive val routes as FORCED (the map offers no way to go straight) or CHOICE.

  $DATA_DIR/envs/carla/bin/python experiments/op_closed_loop/scripts/forced_turns.py [--out experiments/op_closed_loop/results/junction_forced_turns.csv]

CPU only, no CARLA server: carla.Map is built from the OpenDrive file. Dense route = GlobalRoutePlanner (1 m) between the XML key points, as the
leaderboard does (cross-checked against the route.json of earlier runs). Turns = route_poly.maneuvers (|kappa| > 0.02, R < 50 m, merged over
gaps < 8 m) with |angle| >= 25 deg, junction turns and plain road curves alike; mi is the index among the route's >= 25 deg maneuvers, the
numbering junction_cl_report.py uses.
Per turn: a decision waypoint 25 m before the turn start on the route lane; every drivable continuation of that lane (waypoint.next, 1 m steps,
all branches) out to the turn end + 15 m; each leaf is an exit with its heading change vs the decision waypoint.
  n_exits   distinct leaves (clustered within 3 m)
  straight  some exit within 30 deg of straight
  forced    no exit within 30 deg of straight (includes the single-exit curves / L-bends, and T-stems with only left + right)
  kind      junction (the turn window touches a junction road) | curve
Lane level: only the route's own lane is followed (a lane-change to a lane that may go straight is not counted).
"""
import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "lib"), str(Path(__file__).resolve().parent)]
from route_poly import maneuvers  # noqa: E402
import turn_calibration_lib as L  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
XML = DATA / "third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml"
XODR = "/home/ujs/carlaCache/0.9.15/Carla/Maps/{t}/OpenDrive/{t}.xodr"
XODR_FLAT = "/home/ujs/carlaCache/0.9.15/Carla/Maps/OpenDrive/{t}.xodr"
CARLA_PY = Path(os.path.expanduser("~/data/third_party/carla/CARLA_0.9.15/PythonAPI/carla"))
BEFORE_M, AFTER_M, STRAIGHT_DEG, MIN_DEG = 25.0, 15.0, 30.0, 25.0
_maps, _grp = {}, {}


def get_map(town):
    import carla
    if town not in _maps:
        p = XODR.format(t=town)
        p = p if os.path.exists(p) else XODR_FLAT.format(t=town)
        _maps[town] = carla.Map(town, open(p).read())
    return _maps[town]


def get_grp(town):
    sys.path.insert(0, str(CARLA_PY))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    if town not in _grp:
        _grp[town] = GlobalRoutePlanner(get_map(town), 1.0)
    return _grp[town]


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def dense_route(town, pts):
    import carla
    grp = get_grp(town)
    locs = [carla.Location(*p) for p in pts]
    wps = []
    for a, b in zip(locs[:-1], locs[1:]):
        wps += [w for w, _ in grp.trace_route(a, b)]
    return wps


def exits(w0, horizon):
    """All leaves of the lane-following tree of w0, `horizon` m ahead: [(x, y, yaw change deg)]."""
    y0 = w0.transform.rotation.yaw
    out, stack = [], [(w0, 0.0)]
    n = 0
    while stack and n < 20000:
        w, d = stack.pop()
        n += 1
        if d >= horizon:
            out.append((w.transform.location.x, w.transform.location.y, wrap(w.transform.rotation.yaw - y0)))
            continue
        nx = w.next(1.0)
        if not nx:
            out.append((w.transform.location.x, w.transform.location.y, wrap(w.transform.rotation.yaw - y0)))   # dead end: the lane ends
            continue
        stack += [(c, d + 1.0) for c in nx]
    return out


def cluster(leaves, tol=3.0):
    reps = []
    for x, y, a in leaves:
        if not any(math.hypot(x - r[0], y - r[1]) < tol for r in reps):
            reps.append((x, y, a))
    return reps


def route_turns(rid, town, pts, rj):
    wps = dense_route(town, pts)
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    isj = np.array([rj.get(w.road_id, -1) != -1 for w in wps])
    P, g = L.resample(xy)
    rows = []
    for mi, m in enumerate(x for x in maneuvers(P) if abs(x["angle"]) >= MIN_DEG):
        g0, g1 = g[m["i0"]], g[m["i1"]]
        k0 = int(np.searchsorted(cum, max(g0 - BEFORE_M, 0.0)))
        k1 = int(np.searchsorted(cum, g1 + AFTER_M))
        w0 = wps[min(k0, len(wps) - 1)]
        horizon = float(cum[min(k1, len(cum) - 1)] - cum[min(k0, len(cum) - 1)])
        ex = cluster(exits(w0, horizon))
        a_ex = [a for _, _, a in ex]
        a_taken = wrap(wps[min(k1, len(wps) - 1)].transform.rotation.yaw - w0.transform.rotation.yaw)
        straight = any(abs(a) <= STRAIGHT_DEG for a in a_ex)
        kc = int(np.searchsorted(cum, g0)), int(np.searchsorted(cum, g1))
        rows.append(dict(route=rid, town=town, mi=mi, angle=round(m["angle"], 1), rmin=round(m["rmin"], 1), g0=round(float(g0), 1), g1=round(float(g1), 1),
                         kind="junction" if isj[max(kc[0] - 10, 0):kc[1] + 10].any() else "curve", n_exits=len(ex),
                         exit_angles=" ".join("%d" % round(a) for a in sorted(a_ex)), taken_deg=round(a_taken, 1),
                         straight=int(straight), forced=int(not straight), single_exit=int(len(ex) == 1)))
    return rows, xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/op_closed_loop/results/junction_forced_turns.csv"))
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / "experiments/op_common_cause/scripts"))
    import pair_inv_carla as PI
    rows = []
    for e in ET.parse(XML).getroot().iter("route"):
        rid, town = e.get("id"), e.get("town")
        if town not in PI.TAB:
            PI.TAB[town] = PI.xodr_tables(town)
        pts = [(float(p.get("x")), float(p.get("y")), float(p.get("z"))) for p in e.find("waypoints")]
        try:
            r, _ = route_turns(rid, town, pts, PI.TAB[town][0])
        except Exception as ex:   # noqa: BLE001
            print("route", rid, "failed:", ex, file=sys.stderr)
            continue
        rows += r
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_f = sum(r["forced"] for r in rows)
    print("turns >= 25 deg: %d on %d routes; forced %d, choice %d" % (len(rows), len({r["route"] for r in rows}), n_f, len(rows) - n_f))


if __name__ == "__main__":
    main()
