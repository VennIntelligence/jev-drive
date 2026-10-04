"""CARLA counterfactual route pairs, stage 3 step 1 (envs/carla, CPU, carla.Map from the xodr, no server): junction
topology census of the Bench2Drive towns, the B2D hold-out, and the route-polyline builder the later steps import.

Census, per town: every junction's driving connectors (carla Junction.get_waypoints), each walked out of the junction
on both sides to its incoming lane and its outgoing lane. Units:
  approach lane   (junction, incoming road, incoming lane)                  legal exits = its connectors' outgoing roads
  approach road   (junction, incoming road, driving direction)              nav exits = union over its lanes
  junction        distinct outgoing (road, direction) over all connectors
Turn angle = heading change from the incoming lane (at the junction entry) to the outgoing lane (at the exit),
right-handed (left positive); class left / straight / right / uturn (|a| > 135 or same road reversed), +-25 deg.
Exit index = rank among the approach road's exits counted from the left (u-turn leftmost, right-hand traffic).
Free approach = metres of the incoming lane upstream of the entry before the previous junction (capped at FREE_CAP_M):
a pose at distance d with H s of history at speed v needs d + v * H of it.

Hold-out: junctions (town, junction id) traversed by the img_carla test / dev routes (runs/op_img_cmd/carla/split.json,
img2_bank.carla_split: decision-95 routes + random to 60 test, 18 dev) and, stricter, by any Bench2Drive route of the two
route files (op_common_cause carla_traversals.json, 226 traversals) -- a fine-tune on those junctions leaks into any
later B2D closed-loop score there.

  $DATA_DIR/envs/carla/bin/python experiments/op_route_cmd/scripts/carla_topo.py [--workers 4]
  -> $DATA_DIR/runs/op_route_cmd/carla_topo/<ts>/{topo_<town>.json, holdout.json, summary.json}; the summary is copied
     by hand to experiments/op_route_cmd/results/carla_topo_summary.json
"""
import argparse, json, math, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments" / "op_common_cause" / "scripts")]
import pair_inv_carla as PI  # noqa: E402

TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD", "Town11", "Town12",
         "Town13", "Town15")
TURN_DEG, UTURN_DEG, STEP_M, FREE_CAP_M, OUT_MAX_M = 25.0, 135.0, 1.0, 80.0, 120.0
POLY_N, POLY_STEP = 16, 10.0          # 16 points, 0 ... 150 m along the path from the ego


# ---------------------------------------------------------------- geometry helpers (right-handed: x east, y north)
def xy(w):
    l = w.transform.location
    return np.array([l.x, -l.y])


def hdg(w):
    return -math.radians(w.transform.rotation.yaw)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def cls_of(a, uturn=False):
    if uturn or abs(a) > UTURN_DEG:
        return "uturn"
    return "left" if a > TURN_DEG else "right" if a < -TURN_DEG else "straight"


def straightest(cands, ref):
    return min(cands, key=lambda w: abs(wrap_deg(w.transform.rotation.yaw - ref.transform.rotation.yaw)))


def step(w, back=False, keep_lane=True):
    c = w.previous(STEP_M) if back else w.next(STEP_M)
    if not c:
        return None
    same = [v for v in c if keep_lane and v.road_id == w.road_id and v.lane_id == w.lane_id]
    return same[0] if same else straightest(c, w)


def leave(w, back):
    """Walk out of the junction from connector waypoint w; returns the first non-junction waypoint and the metres walked."""
    L = 0.0
    while w is not None and w.is_junction and L < OUT_MAX_M:
        w, L = step(w, back), L + STEP_M
    return (w, L) if w is not None and not w.is_junction else (None, L)


def free_len(w):
    """Upstream metres of w's lane before the previous junction (capped)."""
    L = 0.0
    while L < FREE_CAP_M:
        w = step(w, back=True)
        if w is None or w.is_junction:
            break
        L += STEP_M
    return L


# ---------------------------------------------------------------- census
def connectors(town):
    import carla
    m = PI.get_map(town)
    J = {}
    for a, b in m.get_topology():
        for w in (a, b):
            if w.is_junction:
                j = w.get_junction()
                J.setdefault(j.id, j)
    rows, bad = [], Counter()
    for jid, j in J.items():
        for s, e in j.get_waypoints(carla.LaneType.Driving):
            inc, _ = leave(s, back=True)
            out, _ = leave(e, back=False)
            if inc is None or out is None:
                bad["no incoming / outgoing lane"] += 1
                continue
            uturn = inc.road_id == out.road_id and (inc.lane_id > 0) != (out.lane_id > 0)
            a = wrap_deg(-(out.transform.rotation.yaw - inc.transform.rotation.yaw))
            rows.append(dict(j=jid, conn=[s.road_id, s.lane_id], inc=[inc.road_id, inc.lane_id],
                             out=[out.road_id, out.lane_id], dir_out=int(out.lane_id > 0), angle=round(a, 1),
                             cls=cls_of(a, uturn), entry=[round(float(v), 2) for v in xy(inc)],
                             free=free_len(inc)))
    return rows, dict(bad), len(J)


def exits_of(rows):
    """Approach road -> its exits sorted from the left: [(out road, dir), angle, cls, [incoming lanes that have it]]."""
    by = defaultdict(dict)
    for r in rows:
        k = (r["j"], r["inc"][0], int(r["inc"][1] > 0))
        e = (r["out"][0], r["dir_out"])
        d = by[k].setdefault(e, dict(exit=list(e), angles=[], cls=Counter(), lanes=set(), conns=[]))
        d["angles"].append(r["angle"])
        d["cls"][r["cls"]] += 1
        d["lanes"].add(r["inc"][1])
        d["conns"].append(dict(conn=r["conn"], inc_lane=r["inc"][1], out_lane=r["out"][1]))
    out = {}
    for k, ex in by.items():
        L = []
        for d in ex.values():
            c = d["cls"].most_common(1)[0][0]
            a = float(np.median(d["angles"]))
            L.append(dict(exit=d["exit"], angle=a, cls=c, lanes=sorted(d["lanes"]), conns=d["conns"],
                          key=180.0 if c == "uturn" else a))
        L.sort(key=lambda d: -d["key"])
        for i, d in enumerate(L):
            d["index_from_left"] = i
            d.pop("key")
        out[k] = L
    return out


def town_job(town):
    rows, bad, nj = connectors(town)
    ex = exits_of(rows)
    free = {}
    for r in rows:
        free[(r["j"], *r["inc"])] = r["free"]
    return town, rows, bad, nj, {"%d:%d:%d" % k: v for k, v in ex.items()}, {"%d:%d:%d" % k: v for k, v in free.items()}


def hist(vals, edges):
    """Counts with the last bin open: edges (2, 3, 4, 5) -> {'2', '3', '4', '5+'}; values below edges[0] -> '<x'."""
    c = Counter()
    for v in vals:
        if v < edges[0]:
            c["<%d" % edges[0]] += 1
        else:
            k = min(v, edges[-1])
            c["%d%s" % (k, "+" if k == edges[-1] else "")] += 1
    return dict(sorted(c.items()))


def holdout():
    from jevdrive.common import data_dir
    T = json.load(open(REPO / "experiments" / "op_common_cause" / "results" / "carla_traversals.json"))
    sp = json.loads((data_dir() / "runs" / "op_img_cmd" / "carla" / "split.json").read_text())
    role = {}
    for tok, v in sp.items():
        r = tok.rsplit("-", 1)[0]
        if v in ("test", "dev"):
            role[r] = v
    g = defaultdict(set)
    for t in T:
        g["b2d_any"].add((t["town"], t["junction"]))
        if role.get(t["route"]) == "test":
            g["img_test"].add((t["town"], t["junction"]))
        if role.get(t["route"]) in ("test", "dev"):
            g["img_test_dev"].add((t["town"], t["junction"]))
        if t["route"] in ("27297", "27043", "9196", "24944", "15102", "27870", "22535", "37969", "24497", "28147",
                          "16390", "15612", "15483", "17280", "16529", "16508", "19324", "2520", "19832"):
            g["d95"].add((t["town"], t["junction"]))
    return {k: sorted(v) for k, v in g.items()}


def summarize(res, H):
    S, tot = {"per_town": {}}, Counter()
    held = {k: set(map(tuple, v)) for k, v in H.items()}
    for town, rows, bad, nj, ex, free in res:
        A = {k: v for k, v in ex.items()}
        lane_exits = defaultdict(set)
        for r in rows:
            lane_exits[(r["j"], *r["inc"])].add((r["out"][0], r["dir_out"]))
        jex = defaultdict(set)
        for r in rows:
            jex[r["j"]].add((r["out"][0], r["dir_out"]))
        multi = {k: v for k, v in A.items() if len(v) >= 2}           # approach roads with a choice
        jm = {int(k.split(":")[0]) for k in multi}
        t = dict(junctions=nj, junctions_with_connectors=len(jex), junctions_with_choice=len(jm),
                 junction_distinct_exits=hist([len(v) for v in jex.values()], (2, 3, 4, 5)),
                 approach_roads=len(A), approach_road_exits=hist([len(v) for v in A.values()], (2, 3, 4, 5)),
                 approach_lanes=len(lane_exits), approach_lane_legal_exits=hist([len(v) for v in lane_exits.values()], (2, 3, 4)),
                 lane_exit_polylines=sum(len(v) for v in lane_exits.values()),
                 road_exit_polylines=sum(len(v) for v in A.values()),
                 road_exit_polylines_choice=sum(len(v) for v in multi.values()),
                 approach_roads_3plus_exits=sum(len(v) >= 3 for v in A.values()),
                 exit_cls=dict(Counter(e["cls"] for v in multi.values() for e in v)),
                 connectors_dropped=bad)
        # free approach length of approach lanes that belong to a choice road
        fl = [f for k, f in free.items() if "%s:%s:%d" % (k.split(":")[0], k.split(":")[1], int(int(k.split(":")[2]) > 0)) in multi]
        t["choice_lanes_free_ge"] = {str(d): int(sum(f >= d for f in fl)) for d in (20, 30, 45, 60, 80)}
        for name, hs in held.items():
            keep = {k: v for k, v in multi.items() if (town, int(k.split(":")[0])) not in hs}
            t["held_" + name] = dict(junctions=len({j for (tw, j) in hs if tw == town}),
                                    train_junctions=len({int(k.split(":")[0]) for k in keep}),
                                    train_approach_roads=len(keep), train_roads_3plus=sum(len(v) >= 3 for v in keep.values()),
                                    train_polylines=sum(len(v) for v in keep.values()))
        S["per_town"][town] = t
        for k, v in t.items():
            if isinstance(v, int):
                tot[k] += v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, int):
                        tot[k + "." + kk] += vv
                    elif isinstance(vv, dict):
                        for k3, v3 in vv.items():
                            tot[k + "." + kk + "." + k3] += v3
    S["total"] = dict(sorted(tot.items()))
    S["holdout_junctions"] = {k: len(v) for k, v in held.items()}
    return S


def main():
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--towns", default=",".join(TOWNS))
    a = ap.parse_args()
    towns = a.towns.split(",")
    with Run("op_route_cmd", "carla_topo", config=vars(a)) as run:
        with Pool(a.workers) as pool:
            res = []
            for r in pool.imap_unordered(town_job, sorted(towns, key=lambda t: t not in ("Town12", "Town13"))):
                res.append(r)
                run.info("%s: %d junctions, %d connectors", r[0], r[3], len(r[1]))
                json.dump(dict(rows=r[1], dropped=r[2], junctions=r[3], exits=r[4], free=r[5]),
                          open(run.path("topo_%s.json" % r[0]), "w"))
        H = holdout()
        json.dump(H, open(run.path("holdout.json"), "w"))
        S = summarize(sorted(res), H)
        json.dump(S, open(run.path("summary.json"), "w"), indent=1)
        run.summary.update(S["total"])
        print(json.dumps(S["total"], indent=1))


# ---------------------------------------------------------------- route polylines (imported by carla_pilot.py)
def walk_path(m, town, conn, back_m, ahead_m):
    """World (right-handed) centreline points of: the incoming lane from back_m before connector `conn` (road, lane),
    the connector, then the outgoing lane continuing (same lane, else the straightest successor at later junctions)
    to ahead_m past the ego point. Returns (pts (n, 2), arc of the connector start)."""
    road, lane = conn
    if town not in PI.TAB:
        PI.TAB[town] = PI.xodr_tables(town)
    w0 = m.get_waypoint_xodr(road, lane, 0.05 if lane < 0 else PI.TAB[town][2][road] - 0.05)
    back, w, s = [], w0, 0.0
    while s < back_m:
        w = step(w, back=True)
        if w is None:
            break
        back.append(w)
        s += STEP_M
    fw, w, s = [w0], w0, 0.0
    while s < ahead_m:
        w = step(w)
        if w is None:
            break
        fw.append(w)
        s += STEP_M
    P = np.array([xy(v) for v in back[::-1] + fw])
    return P, len(back) * STEP_M


def poly_ego(P, ego_xy, ego_yaw):
    """Resample world path P from its point nearest the ego at POLY_STEP over POLY_N points; ego frame (x fwd, y left)."""
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s0 = cum[int(np.argmin(np.linalg.norm(P - ego_xy, axis=1)))]
    s = s0 + POLY_STEP * np.arange(POLY_N)
    mask = s <= cum[-1]
    q = np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], -1) - ego_xy
    c, sn = math.cos(ego_yaw), math.sin(ego_yaw)
    loc = np.stack([c * q[:, 0] + sn * q[:, 1], -sn * q[:, 0] + c * q[:, 1]], -1)
    return loc.astype(np.float32), mask


if __name__ == "__main__":
    main()
