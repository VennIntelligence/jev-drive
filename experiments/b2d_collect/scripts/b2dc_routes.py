#!/usr/bin/env python
"""B2D collector route set (plans/2026-10-06-b2d-collect-prereg.md, section "Routes"). CPU only, no CARLA server: carla.Map is built from the
OpenDrive files and routes are densified with CARLA's GlobalRoutePlanner at 1 m, the way the leaderboard interpolates a route.

  $DATA_DIR/envs/simlingo/bin/python experiments/b2d_collect/scripts/b2dc_routes.py build [--n 1000] [--seed 0] [--workers 24]
  -> $DATA_DIR/runs/b2d_collect/routes/<version>/{routes.xml, manifest.csv, candidates.csv, overlap.json, summary.json}
     and the split b2d/b2dc-train (unit route) in jevdrive/data/splits

Candidates, one scenario per route (Bench2Drive's clip unit):
  LB  every scenario of the CARLA Leaderboard 2.0 long routes (routes_training.xml, Town12, 38 types; routes_validation.xml, Town13), cut
      from `before` m ahead of its trigger to `after` m past it (SimLingo's dataset_generation/split_route_files.py distances), both ends
      moved out of any junction. Scenario element (with its parameters) copied verbatim.
  JT  junction manoeuvres in all twelve Bench2Drive towns: every (junction, entry road, exit road) connector of a Driving lane, approach
      D_BEFORE (15 / 30 / 50 m, random) and 35 m past the exit, densified by the planner and kept only when the planned path takes that
      connector. Each candidate gets one of the junction scenario types that fits it (signalised / stop sign / T junction / turn
      direction): the eight Bench2Drive-only types (Vanilla*, T_Junction, *LeftTurnEnterFlow) plus the LB junction types; parameters are
      drawn from the LB instances of that type (Bench2Drive-only types: the parameter values of the bench2drive220 instances, never their
      positions).
  SLC SequentialLaneChange: 125 m on a multi-lane road, lane i -> adjacent lane -> lane i.

Hold-out (Bench2Drive evaluation): a candidate is dropped if, in the same town as any route of bench2drive220.xml or
bench2drive_0.0.4_val.xml, (a) its path crosses a junction that evaluation route crosses, (b) its trigger is within TRIG_M of an evaluation
trigger, or (c) more than OVERLAP_M of its path lies within NEAR_M of the evaluation route's path. overlap.json counts each rule.

Selection (n routes): every Bench2Drive type gets min(MIN_PER_TYPE, available); then junction turns (a LEFT / RIGHT RoadOption inside a
junction) are filled to TURN_SHARE of n with left and right alternating, round robin over towns weighted by the town mix of
bench2drive220; the rest round robin over scenario types. A junction connector is used at most once. Weather: SimLingo's random ranges,
seeded per route.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", DATA / "third_party/carla/CARLA_0.9.15"))
sys.path.insert(0, str(CARLA_ROOT / "PythonAPI/carla"))
B2D = DATA / "third_party/Bench2Drive/leaderboard/data"
SIM = DATA / "third_party/simlingo/leaderboard/data"
EVAL_XMLS = [B2D / "bench2drive220.xml", B2D / "bench2drive_0.0.4_val.xml"]
LB_XMLS = [SIM / "routes_training.xml", SIM / "routes_validation.xml"]
TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD", "Town11", "Town12", "Town13", "Town15"]
VERSION = "v1"

TRIG_M, NEAR_M, OVERLAP_M = 50.0, 3.0, 20.0
MIN_PER_TYPE, TURN_SHARE = 12, 0.55
D_BEFORE, D_AFTER = (15.0, 30.0, 50.0), 35.0
JT_CAP = 1500                       # connectors kept per town (junctions in random order)
TURN_DEG = 25.0
B2D_ONLY = ["NonSignalizedJunctionLeftTurnEnterFlow", "SequentialLaneChange", "SignalizedJunctionLeftTurnEnterFlow", "T_Junction",
            "VanillaNonSignalizedTurn", "VanillaNonSignalizedTurnEncounterStopsign", "VanillaSignalizedTurnEncounterGreenLight",
            "VanillaSignalizedTurnEncounterRedLight"]
# SimLingo dataset_generation/split_route_files.py: metres before the first trigger and after the last ("10 m more than the scenario lasts")
BEFORE = {"HardBreakRoute": 50.0, "HighwayExit": 50.0}
AFTER = {"Accident": 86, "AccidentTwoWays": 86, "BlockedIntersection": 20, "ConstructionObstacle": 70, "ConstructionObstacleTwoWays": 70,
         "ControlLoss": 130, "CrossingBicycleFlow": 25, "DynamicObjectCrossing": 70, "EnterActorFlow": 110, "EnterActorFlowV2": 110,
         "HardBreakRoute": 65, "HazardAtSideLane": 150, "HazardAtSideLaneTwoWays": 150, "HighwayCutIn": 200, "HighwayExit": 30,
         "InterurbanActorFlow": 30, "InterurbanAdvancedActorFlow": 50, "InvadingTurn": 50, "MergerIntoSlowTraffic": 250,
         "MergerIntoSlowTrafficV2": 250, "NonSignalizedJunctionLeftTurn": 30, "NonSignalizedJunctionRightTurn": 30,
         "OppositeVehicleRunningRedLight": 30, "OppositeVehicleTakingPriority": 30, "ParkedObstacle": 70, "ParkedObstacleTwoWays": 70,
         "ParkingCrossingPedestrian": 60, "ParkingCutIn": 85, "ParkingExit": 50, "PedestrianCrossing": 30, "PriorityAtJunction": 30,
         "SignalizedJunctionLeftTurn": 30, "SignalizedJunctionRightTurn": 30, "StaticCutIn": 80, "VehicleOpensDoorTwoWays": 40,
         "VehicleTurningRoute": 70, "VehicleTurningRoutePedestrian": 70, "YieldToEmergencyVehicle": 260}
WEATHER_RANGES = {  # SimLingo split_route_files.py, random_weather (not easy)
    "cloudiness": [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0, 50.0, 60.0, 80.0, 100.0],
    "precipitation": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
    "precipitation_deposits": [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
    "wetness": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0],
    "wind_intensity": [5.0, 10.0, 25.0, 30.0, 50.0, 60.0, 80.0, 100.0],
    "sun_azimuth_angle": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
    "sun_altitude_angle": [-90.0, -45.0, -30.0, -10, -15.0, 5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0, 90.0],
    "fog_density": [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0, 70.0, 100.0]}
OPT = {"VOID": -1, "LEFT": 1, "RIGHT": 2, "STRAIGHT": 3, "LANEFOLLOW": 4, "CHANGELANELEFT": 5, "CHANGELANERIGHT": 6}

_maps, _grp, _tab = {}, {}, {}


def xodr_path(town):
    p = CARLA_ROOT / f"CarlaUE4/Content/Carla/Maps/{town}/OpenDrive/{town}.xodr"
    return p if p.exists() else CARLA_ROOT / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"


def get_map(town):
    import carla
    if town not in _maps:
        _maps[town] = carla.Map(town, xodr_path(town).read_text())
    return _maps[town]


def get_grp(town):
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    if town not in _grp:
        _grp[town] = GlobalRoutePlanner(get_map(town), 1.0)
    return _grp[town]


def incoming_roads(town):
    """junction id -> set of incoming road ids (OpenDrive connection table)."""
    if town not in _tab:
        inc = defaultdict(set)
        for _, el in ET.iterparse(str(xodr_path(town)), events=("end",)):
            if el.tag == "junction":
                for c in el.findall("connection"):
                    inc[int(el.get("id"))].add(int(c.get("incomingRoad")))
                el.clear()
            elif el.tag == "road":
                el.clear()
        _tab[town] = dict(inc)
    return _tab[town]


def cls_of(dyaw):
    """CARLA yaw is clockwise-positive: a positive heading change is a right turn."""
    d = (dyaw + 180.0) % 360.0 - 180.0
    return "uturn" if abs(d) > 135 else "right" if d > TURN_DEG else "left" if d < -TURN_DEG else "straight"


# ---------------------------------------------------------------- dense paths
class Path_:
    """Dense planner path: xyz (n, 3), opt (n,) RoadOption ints, jid (n,) junction id or -1, s (n,) arc length."""

    def __init__(self, xyz, opt, jid):
        self.xyz, self.opt, self.jid = np.asarray(xyz, float), np.asarray(opt, int), np.asarray(jid, int)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xyz[:, :2], axis=0), axis=1))] if len(self.xyz) else np.zeros(0)

    def cut(self, i0, i1):
        return Path_(self.xyz[i0:i1 + 1], self.opt[i0:i1 + 1], self.jid[i0:i1 + 1])

    def junctions(self):
        return sorted(set(self.jid[self.jid >= 0].tolist()))

    def turn(self):
        """'left' / 'right' if a LEFT / RIGHT RoadOption occurs inside a junction, else ''."""
        o = self.opt[self.jid >= 0]
        return "left" if (o == 1).any() else "right" if (o == 2).any() else ""

    def keypoints(self, step=2.0):
        """Positions every `step` m (B2D route style) plus the last point."""
        if len(self.s) == 0:
            return np.zeros((0, 3))
        idx = np.unique(np.r_[np.searchsorted(self.s, np.arange(0, self.s[-1], step)), len(self.s) - 1])
        return self.xyz[idx]


def dense(town, pts) -> Path_:
    import carla
    m, grp = get_map(town), get_grp(town)
    xyz, opt, jid = [], [], []
    for a, b in zip(pts[:-1], pts[1:]):
        for wp, o in grp.trace_route(carla.Location(*map(float, a)), carla.Location(*map(float, b))):
            loc = wp.transform.location
            if xyz and abs(xyz[-1][0] - loc.x) < 1e-3 and abs(xyz[-1][1] - loc.y) < 1e-3:
                continue
            xyz.append((loc.x, loc.y, loc.z))
            opt.append(OPT.get(o.name, -1))
            jid.append(wp.get_junction().id if wp.is_junction else -1)
    return Path_(xyz, opt, jid)


def parse_routes(xml):
    out = []
    for r in ET.parse(str(xml)).getroot().iter("route"):
        pts = [(float(p.get("x")), float(p.get("y")), float(p.get("z"))) for p in r.find("waypoints").iter("position")]
        scen = list(r.find("scenarios").iter("scenario")) if r.find("scenarios") is not None else []
        for s in scen:
            s.tail = None
        out.append(dict(id=r.get("id"), town=r.get("town"), pts=pts, scen=[ET.tostring(s).decode().strip() for s in scen]))
    return out


def trig_xyz(s_xml):
    p = ET.fromstring(s_xml).find("trigger_point")
    return np.array([float(p.get("x")), float(p.get("y")), float(p.get("z"))])


# ---------------------------------------------------------------- evaluation routes (hold-out)
def _eval_job(r):
    P = dense(r["town"], r["pts"])
    trig = [trig_xyz(s).tolist() for s in r["scen"]]
    return dict(id=r["id"], town=r["town"], junctions=P.junctions(), xy=P.xyz[:, :2].tolist(), trig=trig)


class Holdout:
    def __init__(self, evals):
        self.j = defaultdict(set)
        self.trig = defaultdict(list)
        self.xy = defaultdict(list)
        for e in evals:
            self.j[e["town"]].update(e["junctions"])
            self.trig[e["town"]] += [t[:2] for t in e["trig"]]
            self.xy[e["town"]].append(np.asarray(e["xy"]))
        self.xy = {t: np.concatenate(v) for t, v in self.xy.items()}
        self.trig = {t: np.asarray(v) for t, v in self.trig.items()}
        from scipy.spatial import cKDTree
        self.tree = {t: cKDTree(v) for t, v in self.xy.items()}

    def why(self, town, P: Path_, trig) -> str:
        """'' if the candidate is clean, else the first rule it breaks."""
        if town not in self.tree:
            return ""
        if set(P.junctions()) & self.j[town]:
            return "junction"
        if trig is not None and len(self.trig[town]) and np.linalg.norm(self.trig[town] - np.asarray(trig)[:2], axis=1).min() < TRIG_M:
            return "trigger"
        d, _ = self.tree[town].query(P.xyz[:, :2])
        ds = np.diff(P.s, prepend=0.0)
        if ds[d < NEAR_M].sum() > OVERLAP_M:
            return "path"
        return ""


# ---------------------------------------------------------------- LB candidates
def _lb_job(r):
    town = r["town"]
    P = dense(town, r["pts"])
    out = []
    for sx in r["scen"]:
        el = ET.fromstring(sx)
        typ = el.get("type")
        t = trig_xyz(sx)
        i = int(np.argmin(np.linalg.norm(P.xyz - t, axis=1)))
        s0, s1 = P.s[i] - max(BEFORE.get(typ, 20.0), 30.0), P.s[i] + AFTER.get(typ, 60) + 10.0
        i0 = max(0, int(np.searchsorted(P.s, s0)))
        if P.jid[i0] >= 0:                                    # start outside a junction, 5 m before it (SimLingo's junction cooldown)
            while i0 > 0 and P.jid[i0] >= 0:
                i0 -= 1
            i0 = max(0, int(np.searchsorted(P.s, P.s[i0] - 5.0)))
        i1 = min(len(P.s) - 1, int(np.searchsorted(P.s, s1)))
        while i1 < len(P.s) - 1 and P.jid[i1] >= 0:           # end outside a junction, 10 m past it
            i1 += 1
        if P.jid[max(i1 - 1, 0)] >= 0:
            i1 = min(len(P.s) - 1, int(np.searchsorted(P.s, P.s[i1] + 10.0)))
        C = P.cut(i0, i1)
        jd = np.nan
        after = np.where((P.jid[i:] >= 0))[0]
        if len(after):
            jd = float(P.s[i + after[0]] - P.s[i])
        out.append(dict(src="LB", town=town, type=typ, scen=sx, path=C, trig=t.tolist(), base=r["id"], trig_to_junction=jd,
                        key=f"LB:{town}:{r['id']}:{el.get('name')}"))
    return out


# ---------------------------------------------------------------- JT candidates
def _walk(wp, dist, back):
    """Walk `dist` m along the lane (back = towards predecessors), stopping before any junction; returns the last waypoint and metres walked."""
    cur, d = wp, 0.0
    while d < dist:
        nx = cur.previous(1.0) if back else cur.next(1.0)
        if not nx or nx[0].is_junction:
            break
        cur, d = nx[0], d + 1.0
    return cur, d


def _jt_town(args):
    import carla
    town, seed = args
    rng = random.Random(f"{seed}:{town}")
    m = get_map(town)
    inc = incoming_roads(town)
    tl = np.array([[l.transform.location.x, l.transform.location.y] for l in m.get_all_landmarks_of_type("1000001")]).reshape(-1, 2)
    stop = np.array([[l.transform.location.x, l.transform.location.y] for l in m.get_all_landmarks_of_type("206")]).reshape(-1, 2)
    juncs, seen, out = {}, set(), []
    for wp in m.generate_waypoints(4.0):
        if wp.is_junction:
            j = wp.get_junction()
            juncs.setdefault(j.id, j)
    order = sorted(juncs.items())
    rng.shuffle(order)
    for jid, J in order:
        if len(out) >= JT_CAP:
            break
        bb = J.bounding_box
        c = np.array([bb.location.x, bb.location.y])
        r = math.hypot(bb.extent.x, bb.extent.y) + 15.0
        signal = bool(len(tl)) and bool((np.linalg.norm(tl - c, axis=1) < r).any())
        n_in = len(inc.get(jid, ()))
        for a, b in J.get_waypoints(carla.LaneType.Driving):
            cls = cls_of(b.transform.rotation.yaw - a.transform.rotation.yaw)
            if cls == "uturn":
                continue
            pre, nxt = a.previous(1.5), b.next(1.5)
            if not pre or not nxt or pre[0].is_junction or nxt[0].is_junction:
                continue
            key = (jid, pre[0].road_id, nxt[0].road_id)
            if key in seen:
                continue
            seen.add(key)
            dbefore = rng.choice(D_BEFORE)
            st, walked = _walk(pre[0], dbefore, back=True)
            en, _ = _walk(nxt[0], D_AFTER, back=False)
            if walked < 10.0:
                continue
            L = a.transform.location.distance(b.transform.location)
            mid = a.next(max(0.5, L / 2))
            mid = mid[0] if mid else a
            pts = [(st.transform.location.x, st.transform.location.y, st.transform.location.z),
                   (mid.transform.location.x, mid.transform.location.y, mid.transform.location.z),
                   (en.transform.location.x, en.transform.location.y, en.transform.location.z)]
            try:
                P = dense(town, pts)
            except Exception:
                continue
            if len(P.s) < 10 or P.junctions() != [jid]:
                continue
            want = {"left": 1, "right": 2, "straight": 3}[cls]
            if want not in set(P.opt[P.jid >= 0].tolist()):
                continue
            d_mid = np.linalg.norm(P.xyz[:, :2] - pts[1][:2], axis=1).min()
            if d_mid > 1.5:
                continue
            ient = int(np.argmax(P.jid >= 0))
            near_stop = bool(len(stop)) and bool((np.linalg.norm(stop - P.xyz[ient, :2], axis=1) < 20.0).any())
            out.append(dict(src="JT", town=town, junction=jid, cls=cls, signal=signal, stop=near_stop, tjunc=n_in == 3,
                            path=P, i_entry=ient, key=f"JT:{town}:{jid}:{key[1]}:{key[2]}"))
    return out


def jt_types(c):
    t, sig = c["cls"], c["signal"]
    if t == "straight":
        return ["OppositeVehicleRunningRedLight"] if sig else ["OppositeVehicleTakingPriority", "PriorityAtJunction"]
    ty = ["VehicleTurningRoute", "VehicleTurningRoutePedestrian"]
    if sig:
        ty += ["VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight", "OppositeVehicleRunningRedLight"]
        ty += ["SignalizedJunctionLeftTurn", "SignalizedJunctionLeftTurnEnterFlow"] if t == "left" else ["SignalizedJunctionRightTurn"]
    else:
        ty += ["VanillaNonSignalizedTurn"] + (["VanillaNonSignalizedTurnEncounterStopsign"] if c["stop"] else [])
        ty += ["NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionLeftTurnEnterFlow"] if t == "left" else ["NonSignalizedJunctionRightTurn"]
        ty += ["BlockedIntersection"]
    if c["tjunc"]:
        ty.append("T_Junction")
    return ty


# ---------------------------------------------------------------- SequentialLaneChange candidates
def _slc_town(args):
    import carla
    town, seed = args
    rng = random.Random(f"slc:{seed}:{town}")
    m = get_map(town)
    out, used = [], set()
    wps = [w for w in m.generate_waypoints(25.0) if not w.is_junction and w.lane_type == carla.LaneType.Driving]
    rng.shuffle(wps)
    for w in wps:
        if (w.road_id, w.section_id) in used or len(out) >= 40:
            continue
        side = w.get_left_lane() if rng.random() < 0.5 else w.get_right_lane()
        if side is None or side.lane_type != carla.LaneType.Driving or side.lane_id * w.lane_id < 0:
            continue
        seq = [w]
        ok = True
        for d, lane in ((45.0, "side"), (90.0, "back"), (125.0, "back")):
            nx = w.next(d)
            if not nx or nx[0].is_junction:
                ok = False
                break
            n0 = nx[0]
            if lane == "side":
                n0 = n0.get_left_lane() if side.lane_id == (w.get_left_lane().lane_id if w.get_left_lane() else None) else n0.get_right_lane()
                if n0 is None or n0.lane_type != carla.LaneType.Driving:
                    ok = False
                    break
            seq.append(n0)
        if not ok:
            continue
        pts = [(x.transform.location.x, x.transform.location.y, x.transform.location.z) for x in seq]
        try:
            P = dense(town, pts)
        except Exception:
            continue
        if P.junctions() or not ((P.opt == 5) | (P.opt == 6)).any() or P.s[-1] > 200:
            continue
        used.add((w.road_id, w.section_id))
        out.append(dict(src="SLC", town=town, cls="", path=P, key=f"SLC:{town}:{w.road_id}:{w.section_id}:{w.lane_id}"))
    return out


# ---------------------------------------------------------------- XML
def weather_xml(rng):
    vals = {k: rng.choice(v) for k, v in WEATHER_RANGES.items()}
    ws = ET.Element("weathers")
    for pct in ("0", "100"):
        ET.SubElement(ws, "weather", dict(route_percentage=pct, **{k: str(float(v)) for k, v in vals.items()}))
    return ws, vals


def scenario_xml(typ, name, trig_xyz_yaw, params):
    s = ET.Element("scenario", name=name, type=typ)
    x, y, z, yaw = trig_xyz_yaw
    ET.SubElement(s, "trigger_point", x=f"{x:.1f}", y=f"{y:.1f}", z=f"{z:.1f}", yaw=f"{yaw:.1f}")
    for p in params:
        s.append(copy.deepcopy(p))
    return s


def route_xml(rid, c, rng):
    r = ET.Element("route", id=str(rid), town=c["town"])
    wp = ET.SubElement(r, "waypoints")
    for x, y, z in c["path"].keypoints(2.0):
        ET.SubElement(wp, "position", x=f"{x:.1f}", y=f"{y:.1f}", z=f"{z:.1f}")
    sc = ET.SubElement(r, "scenarios")
    sc.append(c["scen_el"])
    ws, vals = weather_xml(rng)
    r.append(ws)
    return r, vals


def heading_at(P: Path_, i):
    j = min(i + 2, len(P.s) - 1)
    k = max(j - 4, 0)
    d = P.xyz[j, :2] - P.xyz[k, :2]
    return math.degrees(math.atan2(d[1], d[0]))


def main_build(a):
    rng = random.Random(a.seed)
    out = DATA / "runs" / "b2d_collect" / "routes" / VERSION
    out.mkdir(parents=True, exist_ok=True)
    from jevdrive.run import Run
    from jevdrive.data import splits
    with Run("b2d_collect", "routes", seed=a.seed, config=vars(a)) as run:
        run.use_split(splits.load("b2d/bench2drive220"))
        run.use_split(splits.load("b2d/bench2drive-0.0.4-val"))
        evals = [r for x in EVAL_XMLS for r in parse_routes(x)]
        lbs = [r for x in LB_XMLS for r in parse_routes(x)]
        with Pool(a.workers) as pool:
            ev = pool.map(_eval_job, evals, chunksize=4)
            run.info(f"eval routes densified: {len(ev)}")
            H = Holdout(ev)
            lb = [c for x in pool.map(_lb_job, lbs, chunksize=1) for c in x]
            run.info(f"LB candidates: {len(lb)}")
            jt = [c for x in pool.map(_jt_town, [(t, a.seed) for t in TOWNS], chunksize=1) for c in x]
            run.info(f"JT candidates: {len(jt)}")
            slc = [c for x in pool.map(_slc_town, [(t, a.seed) for t in TOWNS], chunksize=1) for c in x]
            run.info(f"SLC candidates: {len(slc)}")

        # parameter templates: LB instances per type; Bench2Drive-only types from bench2drive220 (values only, never positions)
        tmpl = defaultdict(list)
        for r in lbs + evals:
            for sx in r["scen"]:
                el = ET.fromstring(sx)
                ps = [p for p in el if p.tag != "trigger_point"]
                if not any({"x", "y", "z"} & set(p.attrib) for p in ps):
                    tmpl[el.get("type")].append(ps)
        tjd = defaultdict(list)                                # trigger -> junction entry distance per LB type
        for c in lb:
            if np.isfinite(c["trig_to_junction"]) and c["trig_to_junction"] < 60:
                tjd[c["type"]].append(c["trig_to_junction"])
        tjd = {k: float(np.median(v)) for k, v in tjd.items()}

        overlap, cands = Counter(), []
        for c in lb:
            why = H.why(c["town"], c["path"], c["trig"])
            overlap[f"LB:{why or 'kept'}"] += 1
            if not why:
                c["scen_el"] = ET.fromstring(c["scen"])
                c["turn"] = c["path"].turn()
                cands.append(c)
        for c in jt:
            ty = jt_types(c)
            c["types"] = ty
            P, ie = c["path"], c["i_entry"]
            why = H.why(c["town"], P, None)
            overlap[f"JT:{why or 'kept'}"] += 1
            if not why:
                c["turn"] = P.turn()
                cands.append(c)
        for c in slc:
            why = H.why(c["town"], c["path"], c["path"].xyz[3])
            overlap[f"SLC:{why or 'kept'}"] += 1
            if not why:
                c["types"], c["turn"] = ["SequentialLaneChange"], ""
                cands.append(c)
        run.info(f"hold-out: {dict(overlap)}")

        def realize(c, typ):
            """Give a JT / SLC candidate its scenario element (trigger on the path before the junction entry, parameters from a template)."""
            P = c["path"]
            if c["src"] == "JT":
                d = tjd.get(typ, 8.0)
                s_t = max(P.s[c["i_entry"]] - d, min(3.0, P.s[c["i_entry"]] - 1.0))
            else:
                s_t = 3.0
            i = int(np.searchsorted(P.s, s_t))
            x, y, z = P.xyz[i]
            ps = rng.choice(tmpl[typ]) if tmpl.get(typ) else []
            c = dict(c, type=typ, trig=[x, y, z], scen_el=scenario_xml(typ, f"{typ}_1", (x, y, z + 0.0, heading_at(P, i)), ps))
            return c

        # ---- selection
        n = a.n
        by_type = defaultdict(list)
        for c in cands:
            for t in ([c["type"]] if c["src"] == "LB" else c["types"]):
                by_type[t].append(c)
        for v in by_type.values():
            rng.shuffle(v)
        all_types = sorted(set(by_type) | set(B2D_ONLY))
        used, chosen = set(), []

        def take(c, typ):
            if c["key"] in used:
                return False
            used.add(c["key"])
            chosen.append(c if c["src"] == "LB" else realize(c, typ))
            return True

        for t in all_types:                                   # 1. every type
            k = 0
            for c in by_type.get(t, []):
                if k >= MIN_PER_TYPE:
                    break
                k += take(c, t)
        ev_town = Counter(e["town"] for e in ev if e["id"] in {r["id"] for r in parse_routes(EVAL_XMLS[0])})
        n_turn = int(round(TURN_SHARE * n))
        turns = defaultdict(lambda: defaultdict(list))        # town -> side -> candidates
        for c in cands:
            if c["turn"]:
                turns[c["town"]][c["turn"]].append(c)
        for t in turns.values():
            for v in t.values():
                rng.shuffle(v)
        wt = {t: ev_town.get(t, 0) + 1 for t in turns}
        side, guard = "left", 0
        while sum(1 for c in chosen if c["turn"]) < n_turn and len(chosen) < n and guard < 100000:   # 2. junction turns
            guard += 1
            town = rng.choices(list(wt), weights=list(wt.values()))[0]
            pool_ = turns[town][side]
            while pool_ and pool_[-1]["key"] in used:
                pool_.pop()
            if not pool_:
                if not any(turns[t][s] for t in turns for s in ("left", "right")):
                    break
                continue
            c = pool_.pop()
            typ = c["type"] if c["src"] == "LB" else rng.choice([t for t in c["types"] if t not in ("OppositeVehicleTakingPriority",)])
            if take(c, typ):
                side = "right" if side == "left" else "left"
        order, guard = [t for t in all_types if by_type.get(t)], 0
        while len(chosen) < n and guard < 100000:             # 3. the rest, round robin over types
            guard += 1
            t = order[guard % len(order)]
            v = by_type[t]
            while v and v[-1]["key"] in used:
                v.pop()
            if v:
                take(v.pop(), t)
            elif not any(by_type[x] for x in order):
                break

        # ---- write
        root = ET.Element("routes")
        rows = []
        for i, c in enumerate(chosen):
            rid = 900000 + i
            r, w = route_xml(rid, c, random.Random(f"{a.seed}:w:{rid}"))
            root.append(r)
            rows.append(dict(route_id=rid, town=c["town"], type=c["type"], src=c["src"], turn=c["turn"],
                             junctions=" ".join(map(str, c["path"].junctions())), length_m=round(float(c["path"].s[-1]), 1),
                             key=c["key"], sun_alt=w["sun_altitude_angle"], precip=w["precipitation"], fog=w["fog_density"]))
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(out / "routes.xml", encoding="utf-8", xml_declaration=True)
        with open(out / "manifest.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        with open(out / "candidates.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["key", "src", "town", "types", "turn", "length_m"])
            for c in cands:
                wr.writerow([c["key"], c["src"], c["town"], c.get("type") or "|".join(c["types"]), c["turn"], round(float(c["path"].s[-1]), 1)])
        tc = Counter(r["type"] for r in rows)
        summ = {"n": len(rows), "types": len(tc), "per_type": dict(sorted(tc.items())), "missing_types": sorted(set(all_types) - set(tc)),
                "turn_share": float(np.mean([bool(r["turn"]) for r in rows])), "turn_side": dict(Counter(r["turn"] for r in rows)),
                "towns": dict(Counter(r["town"] for r in rows)), "src": dict(Counter(r["src"] for r in rows)),
                "length_m": {"mean": float(np.mean([r["length_m"] for r in rows])), "p10": float(np.percentile([r["length_m"] for r in rows], 10)),
                             "p90": float(np.percentile([r["length_m"] for r in rows], 90)), "sum_km": float(np.sum([r["length_m"] for r in rows]) / 1e3)},
                "night_share": float(np.mean([r["sun_alt"] < 0 for r in rows])),
                "candidates": {"LB": len(lb), "JT": len(jt), "SLC": len(slc), "kept": len(cands)}, "trigger_to_junction_m": tjd}
        (out / "overlap.json").write_text(json.dumps(dict(overlap), indent=1))
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        sha = hashlib.sha256((out / "routes.xml").read_bytes()).hexdigest()[:16]
        sp = splits.define("b2d", "b2dc-train", [r["route_id"] for r in rows], unit="route",
                           origin=f"experiments/b2d_collect/scripts/b2dc_routes.py build --seed {a.seed} --n {n}: $DATA_DIR/runs/b2d_collect/routes/"
                                  f"{VERSION}/routes.xml (sha256 {sha}); one scenario per route, positional hold-out from bench2drive220 and "
                                  f"bench2drive_0.0.4_val (junction / trigger {TRIG_M:g} m / path {OVERLAP_M:g} m within {NEAR_M:g} m)",
                           status="frozen", used_by=["b2d_collect"])
        run.use_split(sp)
        run.summary.update(summ, split=sp.id)
        run.info(json.dumps(summ))


def main_subset(a):
    """routes.xml subset (ids) -> a new xml (the staged launch's 1 / ~10 route files)."""
    src = DATA / "runs" / "b2d_collect" / "routes" / VERSION / "routes.xml"
    root = ET.parse(str(src)).getroot()
    keep = set(a.ids.split(","))
    new = ET.Element("routes")
    for r in root.iter("route"):
        if r.get("id") in keep:
            new.append(r)
    ET.ElementTree(new).write(a.out, encoding="utf-8", xml_declaration=True)
    print(len(new), "routes ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("subset")
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": main_build, "subset": main_subset}[a.cmd](a)
