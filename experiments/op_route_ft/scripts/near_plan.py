"""rc-*-near: pose plan of NEAR, LOW-SPEED CARLA junction poses (decision 129 follow-up; envs/carla, CPU only, carla.Map from the xodr, no server).

Same map walking, hold-out, dev hash, weather and polyline convention as experiments/op_route_cmd/scripts/carla_pairs_plan.py; different poses:
  * approach poses: rear axle 0-10 m before the connector start (d ~ U(0, 10)), on the ego lane (the approach lane with the most legal exits);
    rows = every exit LEGAL from that lane (no lane-change blends: a car < 10 m before the mouth does not change lanes), so 3-exit lanes give 3 rows;
  * in-turn poses: rear axle 0.5-6 m along the connector of one legal turning exit (|angle| >= 25 deg); rows = that exit plus any other legal exit
    whose lane-centre path still passes within 0.75 m and 10 deg of the pose (undiverged), usually none;
  * speed 0-3 m/s with a history consistent with it: kind `stopped` (v 0 throughout, one render repeated, 10%), `halted` (v0 0, was decelerating at
    0.5-1.5 m/s^2, 10%), `rolling` (v0 ~ U(0.3, 3), constant acceleration ~ U(-1, 1) m/s^2 over the 1.8 s history, speed clipped to [0, 5]; 80%).
    `profile` = kind (the renderer renders one frame only for `stopped`).
Every row also carries `dense`: the lane-centre path (after the same smooth convergence of the pose offset as the polyline) in the t0 ego frame at
1 m up to 80 m; the trainer derives near targets from it, not from the 10 m vertices (they cut every 90-degree corner).

  $DATA_DIR/envs/carla/bin/python experiments/op_route_ft/scripts/near_plan.py --topo <carla_topo run> --n-poses 4000 --tag near4k
  -> $DATA_DIR/runs/op_route_ft/near_plan_<tag>/<ts>/{poses.pkl, summary.json}
"""
import argparse, json, math, pickle, sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CPD = REPO / "experiments" / "op_route_cmd" / "scripts"
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments" / "op_common_cause" / "scripts"), str(CPD)]
import carla_pairs_plan as CP  # noqa: E402
import carla_topo as CT  # noqa: E402
import route_poly as RP  # noqa: E402

D_APP = (0.0, 10.0)                        # approach: rear axle this far before the connector start, m
S_IN = (0.5, 6.0)                          # in-turn: rear axle this far along the taken connector, m
V_ROLL, A_ROLL, A_HALT, V_CAP = (0.3, 3.0), (-1.0, 1.0), (0.5, 1.5), 5.0
KINDS = (("stopped", 0.1), ("halted", 0.1), ("rolling", 0.8))
TURN_MIN_DEG = 25.0
UNDIV_M, UNDIV_DEG = 0.75, 10.0
NDENSE = 80                                # dense path: 1 m spacing up to 80 m (= rft.NPATH)


def kinematics(kind, v0, acc):
    """-> (tau (K,) s before t0 oldest first, trav (K,) m driven from that frame to t0)."""
    tau = CP.DT * np.arange(CP.K_HIST - 1, -1, -1)
    if kind == "stopped":
        return tau, np.zeros_like(tau)
    t = np.linspace(-tau.max(), 0.0, 1801)
    v = np.clip(v0 + acc * t, 0.0, V_CAP)
    cum = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))]
    return tau, cum[-1] - np.interp(-tau, t, cum)


def track(arr_back, arr_fwd):
    """Back walk (k m before the connector start) + forward walk from it -> (T (n, 5), u (n,)): u = arc from the connector start (negative before)."""
    T = np.vstack([arr_back[::-1], arr_fwd[1:]])
    T[:, 3] = np.degrees(np.unwrap(np.radians(T[:, 3])))
    return T, np.arange(len(T), dtype=float) - (len(arr_back) - 1)


def at_u(T, u, U):
    out = np.stack([np.interp(U, u, T[:, c]) for c in range(5)], -1)
    out[(np.asarray(U) < u[0]) | (np.asarray(U) > u[-1])] = np.nan
    return out


def polyline_dense(path_xy, ego, yaw):
    """carla_pairs_plan.polyline + the converged dense path (ego frame, 1 m, NDENSE) it is built from."""
    q = CP.to_ego(path_xy, ego, yaw)
    j = int(np.argmin(np.linalg.norm(q, axis=1)))
    q = q[j:]
    if len(q) > 1:
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        w = np.clip(s / CP.CONVERGE_M, 0.0, 1.0)
        q = q - (1 - w * w * (3 - 2 * w))[:, None] * q[0]
        q = q[np.r_[False, np.diff(s) > 0.3]] if len(q) > 1 else q
    full = np.vstack([[0.0, 0.0], q])
    dense = np.zeros((NDENSE, 2), np.float32)
    dm = np.zeros(NDENSE, bool)
    if len(full) >= 2:
        P, _ = RP.poly_resample(full, 1.0)
        n = min(len(P), NDENSE)
        dense[:n], dm[:n] = P[:n], True
    return RP.hindsight(full), dense, dm


def build_near(m, town, x, kind, where, pos, v0, acc, weather, rng, split, pid, track_exit=None):
    """where = 'app' (pos = d m before the connector start) or 'in' (pos = s m along track_exit's connector)."""
    lane, exits = x["lane"], x["exits"]
    legal = [e for e in exits if lane in e["lanes"]]
    if not legal:
        return None
    conn = {tuple(e["exit"]): min((c for c in e["conns"] if c["inc_lane"] == lane), key=lambda c: c["out_lane"]) for e in legal}
    tau, trav = kinematics(kind, v0, acc)
    u0 = -pos if where == "app" else pos
    need = max(0.0, -u0) + float(trav.max()) + CP.HIST_MARGIN_M
    w_first = CP.conn_wp(m, town, conn[tuple(legal[0]["exit"])]["conn"])
    back = CP.walk_back(w_first, need)
    if len(back) - 1 < need:
        return None
    arr_b = CP.wp_arrays(back)
    tracks = {}
    for e in legal:
        fw = CP.walk_ahead(CP.conn_wp(m, town, conn[tuple(e["exit"])]["conn"]), CP.AHEAD_M + 10.0)
        tracks[tuple(e["exit"])] = track(arr_b, CP.wp_arrays(fw))
    T, u = tracks[tuple((track_exit or legal[0])["exit"])]
    H = at_u(T, u, u0 - trav)                                       # (K, 5) lane-centre rear-axle points, oldest first
    if np.isnan(H).any():
        return None
    lat, dyaw = CP.hist_noise(rng)
    yaw_c = H[:, 3] + dyaw
    fwd = np.stack([np.cos(np.radians(H[:, 3])), np.sin(np.radians(H[:, 3]))], -1)
    rgt = np.stack([-fwd[:, 1], fwd[:, 0]], -1)
    xy = H[:, :2] - lat[:, None] * rgt
    hist = dict(x=xy[:, 0], y=xy[:, 1], z=H[:, 2], yaw=yaw_c, pitch=H[:, 4], t=-tau)
    ego, yaw_rh = CP.rh(xy[-1:])[0], -math.radians(yaw_c[-1])
    c0, yaw0 = CP.rh(H[-1:, :2])[0], -math.radians(H[-1, 3])         # un-noised lane-centre pose (divergence test)
    off, n_lanes = CP.lane_offset(w_first)
    rows = []
    for e in legal:
        Te, ue = tracks[tuple(e["exit"])]
        P = CP.rh(Te[ue >= u0 - 2.0, :2])
        if where == "in" and e is not track_exit:
            dd = np.linalg.norm(P - c0, axis=1)
            j = int(np.argmin(dd))
            hd = -Te[ue >= u0 - 2.0, 3][j]
            if dd[j] > UNDIV_M or abs(CT.wrap_deg(hd - math.degrees(yaw0))) > UNDIV_DEG:
                continue
        h, dense, dm = polyline_dense(P, ego, yaw_rh)
        rows.append(dict(exit=list(e["exit"]), cls=e["cls"], angle=float(e["angle"]), index_from_left=int(e["index_from_left"]), legal=True,
                         lane_change=False, conn=list(conn[tuple(e["exit"])]["conn"]), poly=h["poly"], pmask=h["pmask"], plen=h["plen"],
                         turn_deg=h["turn_deg"], turn_s=h["turn_s"], turn_end_s=h["turn_end_s"], turn_rmin=h["turn_rmin"], n_turn=h["n_turn"],
                         in_turn=h["in_turn"], max_turn_deg=h["max_turn_deg"], dense=dense, dmask=dm, taken=e is (track_exit or e)))
    if not rows:
        return None
    return dict(id=pid, town=town, junction=x["j"], road=x["road"], dir=x["dir"], lane=lane, d=float(-u0), where=where, profile=kind, v0=float(v0),
                a=float(-acc), weather=weather, split=split, cluster="%s:J%d" % (town, x["j"]), n_exits=len(rows), n_exits_road=len(exits), hist=hist,
                lane_off_m=off, n_lanes=n_lanes, exits=rows, wp_ground_z=float(H[-1, 2]))


def draw_kind(rng):
    k = str(rng.choice([k for k, _ in KINDS], p=[p for _, p in KINDS]))
    if k == "stopped":
        return k, 0.0, 0.0
    if k == "halted":
        return k, 0.0, -float(rng.uniform(*A_HALT))
    return k, float(rng.uniform(*V_ROLL)), float(rng.uniform(*A_ROLL))


def main():
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", required=True)
    ap.add_argument("--n-poses", type=int, default=4000)
    ap.add_argument("--poses-per-road", type=int, default=3, help="2 approach + 1 in-turn (3 approach when the lane has no turning exit)")
    ap.add_argument("--p3", type=float, default=0.5)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--towns", default=",".join(CT.TOWNS))
    ap.add_argument("--tag", default="near")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--id-prefix", default="N")
    a = ap.parse_args()
    with Run("op_route_ft", "near_plan_" + a.tag, seed=a.seed, config=vars(a)) as run:
        held = {tuple(x) for x in json.load(open(Path(a.topo) / "holdout.json"))["b2d_any"]}
        cands = CP.candidates(a.topo, a.towns.split(","), held)
        pick = CP.allocate(cands, int(1.25 * a.n_poses / a.poses_per_road), a.p3)
        run.info("hold-out %d junctions; roads picked %d of %d", len(held), len(pick), sum(map(len, cands.values())))
        rng = np.random.default_rng(a.seed)
        wn, wp = zip(*CP.WEATHER)
        wp = np.array(wp, float) / sum(wp)
        poses, drop = [], Counter()
        for town in sorted({t for t, _ in pick}):
            m = CP.PI.get_map(town)
            for _, x in sorted((p for p in pick if p[0] == town), key=lambda p: (p[1]["j"], p[1]["road"], p[1]["dir"])):
                split = "dev" if CP.h01(town, x["j"], "dev") < a.dev_frac else "train"
                turning = [e for e in x["exits"] if x["lane"] in e["lanes"] and abs(e["angle"]) >= TURN_MIN_DEG]
                plan = [("app", None)] * (a.poses_per_road - 1) + [("in", turning[int(rng.integers(len(turning)))]) if turning else ("app", None)]
                for k, (where, te) in enumerate(plan):
                    kind, v0, acc = draw_kind(rng)
                    pos = float(rng.uniform(*D_APP)) if where == "app" else float(rng.uniform(*S_IN))
                    wth = wn[int(rng.choice(len(wn), p=wp))]
                    pid = a.id_prefix + "%s-j%d-r%dd%d-l%d-%s%d-%s-%d" % (town, x["j"], x["road"], x["dir"], x["lane"], where, int(10 * pos), kind, k)
                    try:
                        p = build_near(m, town, x, kind, where, pos, v0, acc, wth, rng, split, pid, te)
                    except Exception as e:  # noqa: BLE001
                        drop["error " + type(e).__name__] += 1
                        continue
                    if p is None:
                        drop["infeasible"] += 1
                        continue
                    poses.append(p)
            run.info("%s: %d poses so far, dropped %s", town, len(poses), dict(drop))
        if len(poses) > a.n_poses:
            poses = [poses[i] for i in np.sort(rng.permutation(len(poses))[: a.n_poses])]
        poses = CP.tour(poses)
        assert not any((p["town"], p["junction"]) in held for p in poses)
        pickle.dump(poses, open(run.path("poses.pkl"), "wb"))
        rows = [r for p in poses for r in p["exits"]]
        v0 = np.array([p["v0"] for p in poses])
        S = dict(poses=len(poses), rows=len(rows), dropped=dict(drop), dev_poses=sum(p["split"] == "dev" for p in poses),
                 by_where=dict(Counter(p["where"] for p in poses)), by_kind=dict(Counter(p["profile"] for p in poses)),
                 by_town=dict(Counter(p["town"] for p in poses)), by_n_exits=dict(Counter(p["n_exits"] for p in poses)),
                 by_cls=dict(Counter(r["cls"] for r in rows)), three_exit_app_poses=sum(p["n_exits"] >= 3 and p["where"] == "app" for p in poses),
                 v0_pct=np.percentile(v0, [10, 50, 90]).round(2).tolist(), d_app_pct=np.percentile([p["d"] for p in poses if p["where"] == "app"], [10, 50, 90]).round(2).tolist(),
                 junctions=len({(p["town"], p["junction"]) for p in poses}), poses_in_held_out=0)
        json.dump(S, open(run.path("summary.json"), "w"), indent=1)
        run.summary.update({k: v for k, v in S.items() if isinstance(v, (int, float))})
        print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
