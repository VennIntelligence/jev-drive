#!/usr/bin/env python
"""Night queue 3 Q3 (todos/2026-09-26-night-queue-3.md, [A] entries): Bench2Drive-style single-scenario clips cut from
the Leaderboard 2.0 long routes (routes_training.xml, Town12; routes_validation.xml, Town13), for the obstacle classes
of P6. Bench2Drive's own clips (bench2drive220 / 0.0.4 val) start 12 m before the scenario's trigger point and end
~122 m after it (medians over their 114 obstacle clips; 10-90 % 10-14 m and 95-127 m), with keypoints every 2 m;
these clips are cut the same way from the long route's dense plan, which is traced offline exactly as the leaderboard
traces it (GlobalRoutePlanner at 1 m between consecutive keypoints) on the town's OpenDRIVE map. The scenario element
is copied verbatim; the weather is the long route's, interpolated at the trigger's share of the route and written as a
constant (Bench2Drive's clips carry one weather at 0 and 100 %).

  $DATA_DIR/envs/carla/bin/python scripts/nq3_clips.py --out <clips.xml> [--classes a,b]
Clip id = 9 000 000 + 10 000 * source index (0 training, 1 validation) ... see _cid; one clip per scenario instance.
Python 3.8 (envs/carla).
"""
import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

DATA = os.environ["DATA_DIR"]
CARLA_ROOT = os.environ.get("CARLA_ROOT", DATA + "/third_party/carla/CARLA_0.9.15")
sys.path.insert(0, CARLA_ROOT + "/PythonAPI/carla")
import carla  # noqa: E402
from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402

B2D = DATA + "/third_party/Bench2Drive/leaderboard/data/"
SOURCES = (("routes_training.xml", "Town12"), ("routes_validation.xml", "Town13"))
CLASSES = ("Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane", "AccidentTwoWays",
           "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays", "HazardAtSideLaneTwoWays", "VehicleOpensDoorTwoWays")
PRE_M, POST_M, MIN_POST_M, STEP_M = 12.0, 122.0, 90.0, 2.0
WEATHER_KEYS = ("cloudiness", "precipitation", "precipitation_deposits", "wetness", "wind_intensity",
                "sun_azimuth_angle", "sun_altitude_angle", "fog_density")


def _cid(src, route_idx, sc_idx):
    return 9000000 + 100000 * src + 1000 * route_idx + sc_idx


def dense(grp, route):
    """The leaderboard's interpolate_trajectory, locations only: (n, 3) at 1 m."""
    kp = [carla.Location(float(p.get("x")), float(p.get("y")), float(p.get("z"))) for p in route.find("waypoints")]
    pts = []
    for a, b in zip(kp[:-1], kp[1:]):
        for wp, _ in grp.trace_route(a, b):
            loc = wp.transform.location
            pts.append((loc.x, loc.y, loc.z))
    return np.array(pts)


def weather_at(route, frac):
    ws = sorted(route.find("weathers").findall("weather"), key=lambda w: float(w.get("route_percentage")))
    p = 100.0 * frac
    lo = [w for w in ws if float(w.get("route_percentage")) <= p][-1]
    hi = next((w for w in ws if float(w.get("route_percentage")) >= p), ws[-1])
    pl, ph = float(lo.get("route_percentage")), float(hi.get("route_percentage"))
    u = 0.0 if ph == pl else (p - pl) / (ph - pl)
    return {k: round((1 - u) * float(lo.get(k)) + u * float(hi.get(k)), 2) for k in WEATHER_KEYS}


def clips(src_idx, fname, town, classes):
    root = ET.parse(B2D + fname).getroot()
    xodr = CARLA_ROOT + "/CarlaUE4/Content/Carla/Maps/%s/OpenDrive/%s.xodr" % (town, town)
    grp = GlobalRoutePlanner(carla.Map(town, open(xodr).read()), 1.0)
    out, skipped = [], {"pre": 0, "post": 0, "far": 0}
    for ri, route in enumerate(root.findall("route")):
        scs = list(route.iter("scenario"))
        if not any(s.get("type") in classes for s in scs):
            continue
        P = dense(grp, route)
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
        for si, sc in enumerate(scs):
            if sc.get("type") not in classes:
                continue
            tp = sc.find("trigger_point")
            T = np.array([float(tp.get("x")), float(tp.get("y"))])
            dist = np.hypot(*(P[:, :2] - T).T)
            i = int(np.argmin(dist))
            if dist[i] > 3.0:
                skipped["far"] += 1
                continue
            if s[i] < PRE_M:
                skipped["pre"] += 1
                continue
            end = min(s[i] + POST_M, s[-1])
            if end - s[i] < MIN_POST_M:
                skipped["post"] += 1
                continue
            grid = np.arange(s[i] - PRE_M, end + 1e-6, STEP_M)
            pts = np.stack([np.interp(grid, s, P[:, j]) for j in range(3)], 1)
            r = ET.Element("route", {"id": str(_cid(src_idx, ri, si)), "town": town,
                                     "source": "%s:%s:%s" % (fname, route.get("id"), sc.get("name"))})
            wps = ET.SubElement(r, "waypoints")
            for x, y, z in pts:
                ET.SubElement(wps, "position", {"x": "%.1f" % x, "y": "%.1f" % y, "z": "%.1f" % z})
            ET.SubElement(r, "scenarios").append(copy.deepcopy(sc))
            w = weather_at(route, s[i] / s[-1])
            ws = ET.SubElement(r, "weathers")
            for pct in ("0", "100"):
                ET.SubElement(ws, "weather", dict({k: str(v) for k, v in w.items()}, route_percentage=pct))
            out.append(r)
    print("%s: %d clips, skipped %s" % (fname, len(out), skipped), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default=",".join(CLASSES))
    a = ap.parse_args()
    classes = set(a.classes.split(","))
    root = ET.Element("routes")
    for si, (f, town) in enumerate(SOURCES):
        for r in clips(si, f, town, classes):
            root.append(r)
    ET.indent(root) if hasattr(ET, "indent") else None
    ET.ElementTree(root).write(a.out)
    print("-> %s (%d clips)" % (a.out, len(root)))


if __name__ == "__main__":
    main()
