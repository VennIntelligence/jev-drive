"""CARLA counterfactual route pairs, stage 3 step 1: the pose plan (envs/carla, CPU only, carla.Map from the xodr, no server).

For a sample of training approach roads (strict hold-out: no junction that any Bench2Drive route crosses) draw approach poses
(distance d to the connector start, speed profile, weather), build the synthetic 5 Hz history along the ego lane, and for EVERY exit of
the approach road the clean hindsight polyline (lib/route_poly.hindsight, same convention as the real-data sidecars: ego rear-axle frame,
x forward, y left, 16 vertices at 10 m, vertex 0 = origin). The image is exit-independent: one pose = one frame stack, k exits = k rows.

Ego lane = the approach lane with the most legal exits. An exit that the ego lane does not have is routed from the nearest lane that has it
and blended onto it laterally over the approach (`lane_change` = True), so a 3-exit road always gives 3 polylines from one pose.
Beyond the first junction the path keeps its lane, else the straightest successor; where the straightest successor is a > 25 deg turn (a T
junction) the path ends there (pmask False beyond).

  $DATA_DIR/envs/carla/bin/python experiments/op_route_cmd/scripts/carla_pairs_plan.py --n-poses 2000 --tag s2000
  -> $DATA_DIR/runs/op_route_cmd/carla_plan/<ts>/{poses.pkl, summary.json}; poses.pkl = list of pose dicts (see build_pose)
"""
import argparse, hashlib, json, math, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments" / "op_common_cause" / "scripts"),
                str(Path(__file__).resolve().parent)]
import carla_topo as CT  # noqa: E402
import pair_inv_carla as PI  # noqa: E402
import route_poly as RP  # noqa: E402

DT, K_HIST = 0.2, 10                       # 5 Hz lattice, frames t0 - 1.8 s ... t0
DISTS = (10.0, 20.0, 30.0)                 # rear axle to the connector start, m
PROFILES = ("stopped", "creep", "brake", "cruise")
SPEED = {"stopped": 0.0, "creep": 3.0, "cruise": 8.0}   # brake: v0 = min(sqrt(2 a d), 9), constant decel to the connector start
WEATHER = (("ClearNoon", 4), ("CloudyNoon", 3), ("WetNoon", 2), ("WetCloudyNoon", 2), ("SoftRainNoon", 1), ("MidRainyNoon", 1),
           ("ClearSunset", 2), ("CloudySunset", 1), ("WetSunset", 1))
AHEAD_M = 185.0                            # path beyond the pose (150 m horizon + smoothing margin)
HIST_MARGIN_M = 6.0
TURN_STOP_DEG = 25.0


# ---------------------------------------------------------------- map walking
def conn_wp(m, town, conn):
    road, lane = conn
    if town not in PI.TAB:
        PI.TAB[town] = PI.xodr_tables(town)
    return m.get_waypoint_xodr(road, lane, 0.05 if lane < 0 else PI.TAB[town][2][road] - 0.05)


def walk_back(w0, back_m):
    out, w, s = [w0], w0, 0.0
    while s < back_m:
        w = CT.step(w, back=True)
        if w is None or w.is_junction:
            break
        out.append(w)
        s += CT.STEP_M
    return out                                  # out[k] is k m before the connector start (k = 0 is the connector start)


def walk_ahead(w0, ahead_m):
    """Forward from the connector start: connector, outgoing lane, keep lane else straightest; stop at an ambiguous successor."""
    out, w, s = [w0], w0, 0.0
    while s < ahead_m:
        c = w.next(CT.STEP_M)
        if not c:
            break
        same = [v for v in c if v.road_id == w.road_id and v.lane_id == w.lane_id]
        if same:
            w = same[0]
        else:
            w = CT.straightest(c, w)
            if len(c) > 1 and abs(CT.wrap_deg(w.transform.rotation.yaw - out[-1].transform.rotation.yaw)) > TURN_STOP_DEG:
                break
        out.append(w)
        s += CT.STEP_M
    return out


def lane_offset(w):
    """Distance from the road's centre line (inner edge of the nearest same-direction lane) to w's lane centre, and the number of
    same-direction lanes of the road: how far a road-level navigation polyline can sit from this lane-level one."""
    import carla
    off, ww, n_in = w.lane_width / 2.0, w, 0
    while True:
        ww = ww.get_left_lane()
        if ww is None or ww.lane_id * w.lane_id <= 0:
            break
        off += ww.lane_width
        n_in += 1
    ww, n_out = w, 0
    while True:
        ww = ww.get_right_lane()
        if ww is None or ww.lane_id * w.lane_id <= 0 or ww.lane_type != carla.LaneType.Driving:
            break
        n_out += 1
    return float(off), n_in + n_out + 1


def wp_arrays(wps):
    xs = np.array([[w.transform.location.x, w.transform.location.y, w.transform.location.z, w.transform.rotation.yaw,
                    w.transform.rotation.pitch] for w in wps])
    xs[:, 3] = np.degrees(np.unwrap(np.radians(xs[:, 3])))
    return xs                                   # (n, 5): x, y, z, yaw deg, pitch deg in CARLA coordinates


def at_dist(arr, D):
    """Interpolate the walk-back array at distances D (m before the connector start); beyond the array -> nan."""
    k = np.arange(len(arr), dtype=float)
    out = np.stack([np.interp(D, k, arr[:, c], right=np.nan) for c in range(5)], -1)
    out[np.asarray(D) > k[-1]] = np.nan
    return out


# ---------------------------------------------------------------- polylines
def rh(P):
    """CARLA (y right) xy -> right-handed (y left)."""
    return np.stack([P[:, 0], -P[:, 1]], -1)


def to_ego(P, ego, yaw):
    q = P - ego
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1]], -1)


def exit_path(back_xy, fwd_xy, d, lane_gap=None):
    """World (RH) dense path of one exit, from the pose point (d m before the connector start) on: back_xy[k] = k m before the start,
    fwd_xy[k] = k m after it. lane_gap = (vector from the other lane's point at the pose to the ego lane's point) blends the path onto
    the ego lane at the pose and releases it at the connector start."""
    i0 = int(math.floor(d))
    P = np.vstack([back_xy[i0:0:-1], fwd_xy]) if i0 >= 1 else fwd_xy
    if lane_gap is not None:
        s = np.arange(len(P), dtype=float)
        w = np.clip(s / max(d, 1.0), 0, 1)
        w = w * w * (3 - 2 * w)                                     # smoothstep: 0 at the pose, 1 at the connector start
        P = P + lane_gap[None] * (1 - w)[:, None]
    return P


def polyline(path_xy, ego, yaw):
    """Clean hindsight label of one exit in the (noised) t0 ego frame; vertex 0 = ego origin. Returns hindsight dict."""
    q = to_ego(path_xy, ego, yaw)
    d0 = np.linalg.norm(q, axis=1)
    j = int(np.argmin(d0))
    q = q[j:]
    q = q[np.r_[False, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1)) > 0.3]] if len(q) > 1 else q
    return RP.hindsight(np.vstack([[0.0, 0.0], q]))


# ---------------------------------------------------------------- one pose
def hist_noise(rng):
    d0, e0 = np.clip(rng.normal(0, 0.25), -0.5, 0.5), np.clip(rng.normal(0, 0.8), -2.0, 2.0)
    return d0 + 0.04 * rng.normal(size=K_HIST), e0 + 0.15 * rng.normal(size=K_HIST)   # lateral m (left +), yaw deg


def build_pose(m, town, jid, road, direction, exits, lane, d, profile, weather, rng, split, pid):
    ex_lane = [e for e in exits if lane in e["lanes"]]
    conn_of = {}
    for e in exits:
        legal = lane in e["lanes"]
        lanes = sorted(e["lanes"], key=lambda l: (abs(l - lane), l))
        c = min((c for c in e["conns"] if c["inc_lane"] == (lane if legal else lanes[0])), key=lambda c: c["out_lane"])
        conn_of[tuple(e["exit"])] = (c, legal)
    ego_conn = conn_of[tuple(ex_lane[0]["exit"])][0]["conn"]
    w_ego = conn_wp(m, town, ego_conn)
    v0 = {"brake": min(math.sqrt(2 * 2.0 * d), 9.0)}.get(profile, SPEED.get(profile))
    a = v0 * v0 / (2 * d) if profile == "brake" else 0.0
    tau = DT * np.arange(K_HIST - 1, -1, -1)                      # time before t0 of each history frame, oldest first
    trav = v0 * tau + 0.5 * a * tau ** 2 if profile != "stopped" else 0.0 * tau
    need = d + float(trav.max()) + HIST_MARGIN_M
    back = walk_back(w_ego, need)
    if len(back) - 1 < need:
        return None
    arr = wp_arrays(back)
    pos = at_dist(arr, d + trav)                                    # (K, 5) lane-centre points of the rear axle
    if np.isnan(pos).any():
        return None
    lat, dyaw = hist_noise(rng)
    yaw_c = pos[:, 3] + dyaw                                        # CARLA yaw deg (clockwise positive)
    # CARLA: forward = (cos yaw, sin yaw), right = (-sin yaw, cos yaw); a left offset `lat` moves by -lat * right
    fwd = np.stack([np.cos(np.radians(pos[:, 3])), np.sin(np.radians(pos[:, 3]))], -1)
    rgt = np.stack([-fwd[:, 1], fwd[:, 0]], -1)
    xy = pos[:, :2] - lat[:, None] * rgt
    hist = dict(x=xy[:, 0], y=xy[:, 1], z=pos[:, 2], yaw=yaw_c, pitch=pos[:, 4], t=-tau)
    ego = rh(xy[-1:])[0]
    yaw_rh = -math.radians(yaw_c[-1])
    off, n_lanes = lane_offset(w_ego)
    back_xy = rh(arr[:, :2])
    pnt0 = back_xy[int(math.floor(d))]
    rows = []
    for e in exits:
        c, legal = conn_of[tuple(e["exit"])]
        w0 = conn_wp(m, town, c["conn"])
        fw = walk_ahead(w0, d + AHEAD_M)
        fwd_xy = rh(wp_arrays(fw)[:, :2])
        if legal:
            bxy = back_xy
            gap = None
        else:
            bxy = rh(wp_arrays(walk_back(w0, d + 2.0))[:, :2])
            if len(bxy) <= int(math.floor(d)):
                return None
            gap = pnt0 - bxy[int(math.floor(d))]
        P = exit_path(bxy, fwd_xy, d, gap)
        h = polyline(P, ego, yaw_rh)
        rows.append(dict(exit=list(e["exit"]), cls=e["cls"], angle=float(e["angle"]), index_from_left=int(e["index_from_left"]),
                         legal=bool(legal), lane_change=not legal, conn=list(c["conn"]), poly=h["poly"], pmask=h["pmask"], plen=h["plen"],
                         turn_deg=h["turn_deg"], turn_s=h["turn_s"], turn_end_s=h["turn_end_s"], turn_rmin=h["turn_rmin"],
                         n_turn=h["n_turn"], in_turn=h["in_turn"], max_turn_deg=h["max_turn_deg"]))
    cluster = "%s:J%d" % (town, jid)
    return dict(id=pid, town=town, junction=jid, road=road, dir=direction, lane=lane, d=d, profile=profile, v0=v0, a=a, weather=weather,
                split=split, cluster=cluster, n_exits=len(exits), hist=hist, lane_off_m=off, n_lanes=n_lanes, exits=rows,
                wp_ground_z=float(pos[-1, 2]))


# ---------------------------------------------------------------- selection
def h01(*a):
    return int(hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def candidates(topo_dir, towns, held):
    out = defaultdict(list)
    for town in towns:
        T = json.load(open(Path(topo_dir) / f"topo_{town}.json"))
        ex = {}
        for k, v in T["exits"].items():
            j, road, dr = map(int, k.split(":"))
            if len(v) < 2 or (town, j) in held:
                continue
            ex[k] = v
        for k, v in ex.items():
            j, road, dr = map(int, k.split(":"))
            lanes = sorted({l for e in v for l in e["lanes"]})
            nleg = {l: sum(l in e["lanes"] for e in v) for l in lanes}
            lane = max(lanes, key=lambda l: (nleg[l], T["free"].get(f"{j}:{road}:{l}", 0.0), -abs(l)))
            out[town].append(dict(j=j, road=road, dir=dr, exits=v, lane=lane, nleg=nleg[lane],
                                  free=T["free"].get(f"{j}:{road}:{lane}", 0.0)))
    return out


def allocate(cands, n_roads, p3):
    """Towns weighted by sqrt(#roads); 3+-exit roads fill p3 of every town's quota where they exist."""
    w = {t: math.sqrt(len(c)) for t, c in cands.items()}
    tot = sum(w.values())
    q = {t: min(len(c), max(1, round(n_roads * w[t] / tot))) for t, c in cands.items()}
    pick = []
    for t, c in cands.items():
        three = sorted([x for x in c if len(x["exits"]) >= 3], key=lambda x: h01(t, x["j"], x["road"], x["dir"], "s"))
        two = sorted([x for x in c if len(x["exits"]) < 3], key=lambda x: h01(t, x["j"], x["road"], x["dir"], "s"))
        n3 = min(len(three), round(q[t] * p3))
        sel = three[:n3] + two[:q[t] - n3]
        if len(sel) < q[t]:
            sel += three[n3:n3 + q[t] - len(sel)]
        pick += [(t, x) for x in sel]
    return pick


def tour(poses):
    """Order by town, then a greedy nearest-neighbour tour over the junctions (poses of one junction stay together): teleports between
    consecutive poses are short, which keeps Large-Map tile streaming cheap."""
    out = []
    for town in sorted({p["town"] for p in poses}):
        J = defaultdict(list)
        for p in poses:
            if p["town"] == town:
                J[p["junction"]].append(p)
        cen = {j: np.array([ps[0]["hist"]["x"][-1], ps[0]["hist"]["y"][-1]]) for j, ps in J.items()}
        cur = min(cen, key=lambda j: tuple(cen[j]))
        todo = set(cen)
        while todo:
            todo.discard(cur)
            out += sorted(J[cur], key=lambda p: p["id"])
            if todo:
                cur = min(todo, key=lambda j: np.linalg.norm(cen[j] - cen[cur]))
    return out


def main():
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", required=True)
    ap.add_argument("--n-poses", type=int, default=2000)
    ap.add_argument("--poses-per-road", type=int, default=3)
    ap.add_argument("--p3", type=float, default=0.5, help="share of 3+-exit roads")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--towns", default=",".join(CT.TOWNS))
    ap.add_argument("--tag", default="plan")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with Run("op_route_cmd", "carla_plan_" + a.tag, config=vars(a)) as run:
        held = {tuple(x) for x in json.load(open(Path(a.topo) / "holdout.json"))["b2d_any"]}
        run.info("hold-out: %d junctions (any Bench2Drive route)", len(held))
        cands = candidates(a.topo, a.towns.split(","), held)
        pick = allocate(cands, int(a.n_poses / a.poses_per_road), a.p3)
        run.info("roads picked: %d of %d candidates", len(pick), sum(map(len, cands.values())))
        rng = np.random.default_rng(a.seed)
        wnames, wp = zip(*WEATHER)
        poses, drop = [], Counter()
        for town in sorted({t for t, _ in pick}):
            m = PI.get_map(town)
            for t, x in sorted((p for p in pick if p[0] == town), key=lambda p: (p[1]["j"], p[1]["road"], p[1]["dir"])):
                dev = h01(town, x["j"], "dev") < a.dev_frac
                for k in range(a.poses_per_road):
                    d = DISTS[(k + int(h01(town, x["j"], x["road"], "d") * 3)) % 3]
                    prof = PROFILES[int(rng.integers(len(PROFILES)))]
                    if d == 10.0 and prof == "cruise":
                        prof = "brake"
                    if d == 30.0 and prof == "stopped":
                        prof = "creep"
                    wth = wnames[int(rng.choice(len(wnames), p=np.array(wp) / sum(wp)))]
                    pid = "%s-j%d-r%dd%d-l%d-d%d-%s-%d" % (town, x["j"], x["road"], x["dir"], x["lane"], int(d), prof, k)
                    try:
                        p = build_pose(m, town, x["j"], x["road"], x["dir"], x["exits"], x["lane"], d, prof, wth, rng,
                                       "dev" if dev else "train", pid)
                    except Exception as e:  # noqa: BLE001
                        drop["error " + type(e).__name__] += 1
                        continue
                    if p is None:
                        drop["approach too short"] += 1
                        continue
                    poses.append(p)
            run.info("%s: %d poses so far, dropped %s", town, len(poses), dict(drop))
        poses = tour(poses)
        pickle.dump(poses, open(run.path("poses.pkl"), "wb"))
        rows = [r for p in poses for r in p["exits"]]
        S = dict(poses=len(poses), rows=len(rows), dropped=dict(drop), dev_poses=sum(p["split"] == "dev" for p in poses),
                 by_town=dict(Counter(p["town"] for p in poses)), by_profile=dict(Counter(p["profile"] for p in poses)),
                 by_d=dict(Counter(int(p["d"]) for p in poses)), by_n_exits=dict(Counter(p["n_exits"] for p in poses)),
                 lane_change_rows=sum(r["lane_change"] for r in rows), by_cls=dict(Counter(r["cls"] for r in rows)),
                 mean_valid_vertices=float(np.mean([r["pmask"].sum() for r in rows])),
                 held_out_junctions=len(held), poses_in_held_out=sum((p["town"], p["junction"]) in held for p in poses),
                 lane_off_m=dict(zip(("p50", "p90", "max"), np.percentile([p["lane_off_m"] for p in poses], [50, 90, 100]).round(2).tolist())))
        json.dump(S, open(run.path("summary.json"), "w"), indent=1)
        run.summary.update({k: v for k, v in S.items() if isinstance(v, (int, float))})
        print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
