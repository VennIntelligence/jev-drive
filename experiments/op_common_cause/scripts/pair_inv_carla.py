"""Pair inventory, CARLA / Bench2Drive part. CPU only; run with envs/carla (carla client, no server needed:
carla.Map is built straight from the OpenDrive file).
  python pair_inv_carla.py --out <dir> --workers 16   -> <dir>/carla_summary.json, carla_traversals.json
Phase 1: for each route of the two B2D route files (bench2drive220.xml, bench2drive_0.0.4_val.xml), find the junction traversals on the route path (waypoint.is_junction),
         the branches the map offers from the same entry lane (junction.get_waypoints), and group traversals by junction.
Phase 2: for every existing closed-loop recording on the box (ticks.jsonl, 20 Hz ego pose, no camera frames) count the
         ticks within 30 m before a junction entry along the route (sampled at 2 Hz), by branch class and speed.
CARLA yaw is clockwise-positive (left-handed): a positive yaw change is a RIGHT turn.
"""
import argparse, glob, json, math, os, re, sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

XODR = "/home/ujs/carlaCache/0.9.15/Carla/Maps/{t}/OpenDrive/{t}.xodr"
XODR_FLAT = "/home/ujs/carlaCache/0.9.15/Carla/Maps/OpenDrive/{t}.xodr"
ROUTE_XMLS = ["bench2drive220.xml", "bench2drive_0.0.4_val.xml"]   # both sets appear in the recordings; ids are disjoint
ROUTE_DIR = os.path.expanduser("~/data/third_party/Bench2Drive/leaderboard/data/")
WINDOW_M, TURN_DEG = 30.0, 25.0
SPEED_BINS = [0, 1, 3, 6, 10, 15, 1e9]
SPEED_LAB = ["<1", "1-3", "3-6", "6-10", "10-15", ">=15"]
_maps = {}


def cls_of(dyaw_deg):
    d = (dyaw_deg + 180) % 360 - 180
    if abs(d) > 135:
        return "uturn"
    return "right" if d > TURN_DEG else "left" if d < -TURN_DEG else "straight"


def get_map(town):
    import carla
    if town not in _maps:
        p = XODR.format(t=town)
        if not os.path.exists(p):
            p = XODR_FLAT.format(t=town)
        _maps[town] = carla.Map(town, open(p).read())
    return _maps[town]


DENSE = {}   # route id -> dense (x, y) path
TAB = {}   # town -> (road_id -> junction id, junction id -> [(incoming, connecting, [(from_lane, to_lane)])], road_id -> length)


def xodr_tables(town):
    p = XODR.format(t=town)
    if not os.path.exists(p):
        p = XODR_FLAT.format(t=town)
    rj, length, juncs = {}, {}, {}
    for ev, el in ET.iterparse(p, events=("end",)):
        if el.tag == "road":
            rj[int(el.get("id"))] = int(el.get("junction"))
            length[int(el.get("id"))] = float(el.get("length"))
            el.clear()
        elif el.tag == "junction":
            conns = []
            for c in el.findall("connection"):
                conns.append((int(c.get("incomingRoad")), int(c.get("connectingRoad")),
                              [(int(l.get("from")), int(l.get("to"))) for l in c.findall("laneLink")]))
            juncs[int(el.get("id"))] = conns
            el.clear()
    return rj, juncs, length


_bcls = {}
_grp = {}


def get_grp(town):
    sys.path.insert(0, os.path.expanduser("~/data/third_party/carla/CARLA_0.9.15/PythonAPI/carla"))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    if town not in _grp:
        _grp[town] = GlobalRoutePlanner(get_map(town), 1.0)
    return _grp[town]


def conn_class(town, road, lane):
    """Net heading change of a connecting-road lane in its driving direction -> class."""
    import carla
    k = (town, road, lane)
    if k not in _bcls:
        m = get_map(town)
        L = TAB[town][2][road]
        s0, s1 = (0.05, L - 0.05) if lane < 0 else (L - 0.05, 0.05)
        try:
            y0 = m.get_waypoint_xodr(road, lane, s0).transform.rotation.yaw
            y1 = m.get_waypoint_xodr(road, lane, s1).transform.rotation.yaw
            _bcls[k] = cls_of(y1 - y0)
        except Exception:
            _bcls[k] = None
    return _bcls[k]


def analyse_route(args):
    import carla
    rid, town, pts = args
    m = get_map(town)
    rj, juncs, _ = TAB[town]
    # The XML keypoints are sparse (a junction can sit between two of them): densify like the leaderboard does, with
    # GlobalRoutePlanner at 1 m between consecutive keypoints.
    grp = get_grp(town)
    locs = [carla.Location(*p) for p in pts]
    wps = []
    for a, b in zip(locs[:-1], locs[1:]):
        seg = [w for w, _ in grp.trace_route(a, b)]
        wps += seg if not wps else [w for w in seg]
    P = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    isj = np.array([rj.get(w.road_id, -1) != -1 for w in wps])
    out, i = [], 0
    while i < len(wps):
        if not isj[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(wps) and isj[j + 1]:
            j += 1
        if i == 0 or j + 1 >= len(wps):      # route starts or ends inside the junction: not a clean approach
            i = j + 1
            continue
        en, ex = wps[i - 1], wps[j + 1]
        jid = rj[wps[i].road_id]
        taken = cls_of(ex.transform.rotation.yaw - en.transform.rotation.yaw)
        alts = set()
        for inc, con, links in juncs[jid]:
            if inc != en.road_id:
                continue
            for fl, tl in links:
                if fl == en.lane_id:
                    c = conn_class(town, con, tl)
                    if c:
                        alts.add((c, con, tl))
        out.append(dict(route=rid, town=town, junction=jid, entry=[en.road_id, en.lane_id], exit=[ex.road_id, ex.lane_id],
                        taken=taken, branches=sorted(alts), s_entry=float(cum[i - 1]), i_entry=i - 1))
        i = j + 1
    return rid, out, {"n_pts": len(pts), "len_m": float(cum[-1]), "dense": np.round(P, 2).tolist()}


def load_routes():
    r, seen = [], set()
    for x in ROUTE_XMLS:
        for e in ET.parse(ROUTE_DIR + x).getroot().iter("route"):
            if e.get("id") in seen:
                continue
            seen.add(e.get("id"))
            pts = [(float(p.get("x")), float(p.get("y")), float(p.get("z"))) for p in e.find("waypoints")]
            r.append((e.get("id"), e.get("town"), pts))
    return r


def phase1(workers, out):
    R = sorted(load_routes(), key=lambda r: r[1])
    for t in {t for _, t, _ in R}:
        TAB[t] = xodr_tables(t)
    with Pool(workers) as pool:
        res = pool.map(analyse_route, R, chunksize=4)
    trav = [t for _, o, _ in res for t in o]
    DENSE.update({rid: info.pop("dense") for rid, _, info in res})
    json.dump(trav, open(out / "carla_traversals.json", "w"))
    S = {"routes": len(R), "routes_by_town": dict(Counter(t for _, t, _ in R)),
         "route_len_km": round(sum(i["len_m"] for _, _, i in res) / 1000, 1), "dense_pts_per_route_median": float(np.median([len(DENSE[r]) for r in DENSE])),
         "traversals": len(trav), "routes_with_traversal": len({t["route"] for t in trav})}
    S["traversals_by_taken"] = dict(Counter(t["taken"] for t in trav))
    alt = [t for t in trav if len({b[0] for b in t["branches"]} - {t["taken"]}) >= 1]
    S["traversals_with_map_alternative_class"] = len(alt)
    S["traversals_with_map_alternative_by_taken_alt"] = dict(Counter(f"{t['taken']}->{c}" for t in alt for c in {b[0] for b in t["branches"]} - {t["taken"]}))
    S["unique_junctions_visited"] = len({(t["town"], t["junction"]) for t in trav})
    S["unique_junction_entry_lanes_visited"] = len({(t["town"], t["junction"], *t["entry"]) for t in trav})
    # same junction, different routes
    g = defaultdict(list)
    for t in trav:
        g[(t["town"], t["junction"])].append(t)
    S["junctions_visited_by_ge2_routes"] = sum(len({t["route"] for t in v}) >= 2 for v in g.values())
    ge = defaultdict(list)
    for t in trav:
        ge[(t["town"], t["junction"], *t["entry"])].append(t)
    multi_exit = {k: v for k, v in ge.items() if len({tuple(t["exit"]) for t in v}) >= 2}
    multi_cls = {k: v for k, v in ge.items() if len({t["taken"] for t in v}) >= 2}
    S["entry_lanes_with_ge2_routes_different_exit_road_lane"] = len(multi_exit)
    S["entry_lanes_with_ge2_routes_different_exit_class"] = len(multi_cls)
    S["traversals_in_such_class_groups"] = sum(len(v) for v in multi_cls.values())
    S["routes_in_such_class_groups"] = len({t["route"] for v in multi_cls.values() for t in v})
    S["class_group_pairs_by_taken"] = dict(Counter("/".join(sorted({t["taken"] for t in v})) for v in multi_cls.values()))
    return S, trav


def tick_file(args):
    path, rid, route_pts, travs = args
    P = np.array(route_pts)[:, :2]   # dense route path
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    n_lines, rows = 0, []
    with open(path) as f:
        for k, line in enumerate(f):
            n_lines += 1
            if k % 10:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "truth" not in d:
                return path, rid, n_lines, rows, True
            x, y = d["truth"][0], d["truth"][1]
            dd = np.hypot(P[:, 0] - x, P[:, 1] - y)
            ix = int(dd.argmin())
            if dd[ix] > 5.0:
                continue
            s = cum[ix]
            for ti, t in enumerate(travs):
                gap = t["s_entry"] - s
                if 0 <= gap <= WINDOW_M:
                    rows.append((ti, gap, d["v"], d["t"]))
    return path, rid, n_lines, rows, False


def phase2(workers, R, trav, out):
    rp = dict(DENSE)
    tr = defaultdict(list)
    for t in trav:
        tr[t["route"]].append(t)
    files = []
    root = os.path.expanduser("~/data/runs")
    for p in glob.glob(root + "/**/ticks.jsonl", recursive=True):
        rid = next((c for c in reversed(p.split("/")) if c in rp), None)
        if rid:
            files.append((p, rid, rp[rid], tr.get(rid, [])))
    print("tick files", len(files), flush=True)
    S = {"tick_files": len(files)}
    allrows = []
    nl = Counter(); nskip = 0
    with Pool(workers) as pool:
        for k, (p, rid, n, rows, notruth) in enumerate(pool.imap_unordered(tick_file, files, chunksize=4)):
            nskip += notruth
            if notruth:
                continue
            nl[rid] += n
            for r in rows:
                allrows.append((rid, *r))
            if k % 500 == 0:
                print(k, len(files), len(allrows), flush=True)
    S["tick_files_without_truth_skipped"] = nskip
    S["routes_with_recording"] = len(nl)
    S["recordings_per_route_median"] = float(np.median([sum(1 for f in files if f[1] == r) for r in nl])) if nl else 0
    S["ticks_total_20hz"] = int(sum(nl.values()))
    S["approach_ticks_2hz_sampled"] = len(allrows)
    if allrows:
        A = [dict(route=r, ti=ti, gap=g, v=v, t=t) for r, ti, g, v, t in allrows]
        for r in A:
            tv = tr[r["route"]][r["ti"]]
            r["taken"] = tv["taken"]
            r["n_alt"] = len({b[0] for b in tv["branches"]} - {tv["taken"]})
            r["trav"] = f"{r['route']}:{r['ti']}"
        def sp(v):
            v = abs(v)
            return SPEED_LAB[next(i for i in range(6) if SPEED_BINS[i] <= v < SPEED_BINS[i + 1])]
        S["approach_unique_routes"] = len({r["route"] for r in A})
        S["approach_unique_traversals"] = len({r["trav"] for r in A})
        S["approach_by_taken"] = dict(Counter(r["taken"] for r in A))
        B = [r for r in A if r["n_alt"] >= 1]
        S["approach_with_map_alt_ticks"] = len(B)
        S["approach_with_map_alt_unique_traversals"] = len({r["trav"] for r in B})
        S["approach_with_map_alt_by_taken"] = dict(Counter(r["taken"] for r in B))
        S["approach_by_speed"] = {k: Counter(sp(r["v"]) for r in A).get(k, 0) for k in SPEED_LAB}
        S["approach_with_map_alt_by_speed"] = {k: Counter(sp(r["v"]) for r in B).get(k, 0) for k in SPEED_LAB}
        S["approach_hist_ge_2s"] = sum(r["t"] >= 2.0 for r in A)
        S["approach_with_alt_hist_ge_2s"] = sum(r["t"] >= 2.0 for r in B)
        S["approach_with_alt_hist_ge_4s"] = sum(r["t"] >= 4.0 for r in B)
        S["approach_with_alt_by_route_recordings"] = dict(sorted(Counter(r["route"] for r in B).items(), key=lambda x: -x[1])[:10])
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S1, trav = phase1(a.workers, out)
    print(json.dumps(S1, indent=1), flush=True)
    S2 = phase2(a.workers, load_routes(), trav, out)
    S = {"phase1_routes": S1, "phase2_recordings": S2}
    json.dump(S, open(out / "carla_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S2, indent=1, default=str))


if __name__ == "__main__":
    main()
