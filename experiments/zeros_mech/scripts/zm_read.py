#!/usr/bin/env python3
"""ZEROS-MECH (plans/2026-10-12-zeros-mech-prereg.md): what the zeros of the on-policy S-hinge checkpoints come from, read on the stored
18-decision held rollouts. CPU only; run dirs, pools and maps are read, never written.

  collect (box, AlpaSim venv: $DATA_DIR/third_party/alpasim/.venv/bin/python)
      zm_read.py collect --runs <label>=<run dir> ... [--workers N] [--limit N]
      Per scene, through the run's own scene loader: the logged rig path, the logged traffic the simulator replays (physics NONE), the
      vector map (lanes in travel direction), AlpaSim's route; per label: the driver's records (anchor pose, served plan, command), the
      evaluator's 2 Hz series and, where the rollout.asl was kept (failed rollouts), its actor poses for gate G-act.
      -> $DATA_DIR/runs/zeros_mech/collect/<ts>/ per_decision.parquet, per_rollout.parquet, gates.json, errors.csv
  rows    (box, envs/op-train, CPU torch)   zm_read.py rows [--pool on1_L18Y] [--rows-dir L18Y]
      the agent hinge of the round-1 pool rows recomputed per object on the served plan (lib/agent_hinge.py, 0.3 m in the ego's
      corridor / 0 m beside it, as S trains it): the object that binds, classified -> $DATA_DIR/runs/zeros_mech/rows/<ts>/rows.parquet
  bev     (box, AlpaSim venv)   zm_read.py bev --collect DIR --picks picks.json
      lanes with their direction class, actor boxes, ego box, served plan and logged path around chosen decisions -> <collect dir>/bev.pkl
  report  (Mac, .venv)   zm_read.py report --collect DIR --rows DIR [--bev FILE] [--out DIR] [--figs DIR]
      the registered tables, lines and verdicts, the 43-event table of YS-L18-s1, figures

Frames: rollout local frame (the scene loader's), ego = rear axle, x forward, y left. Decision k at t_k = 17 000 + 500 000 k us.
CIs: log-clustered bootstrap in jevdrive.stats' convention (default_rng(0), 10 000 draws of logs, ratio of sums). Two labels are compared
under one draw of logs; their visited decisions differ, so it is the difference of two ratios, not a paired difference (the reason this
read does not call stats.paired).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("collect", "bev"):        # one thread per worker process: the box sets OMP_NUM_THREADS to the
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):  # quota, and 64
        os.environ[_v] = "1"                                         # workers x (BLAS + polars + rayon pools) exhaust the thread limit
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/anchor_offset/scripts"), str(REPO / "experiments/alpasim/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
T0, DT, NK = 17_000, 500_000, 18
EXT, START_M, LANE_R = 40.0, 40.0, 10.0          # straight extension of the logged path; route start offset; lane search radius (m)
LEAD_GAP, LEAD_Y, LEAD_PSI = 15.0, 1.5, np.radians(30.0)
SAME, ONC, JUNC = np.radians(60.0), np.radians(120.0), np.radians(30.0)
LANE = {0: "none", 1: "same", 2: "onc", 3: "cross"}
SER = ("dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory", "offroad", "collision_any", "collision_front", "collision_lateral",
       "collision_rear", "left_corridor_laterally", "wrong_lane")
KIND = (("collision_at_fault", "collision_any", "collision"), ("offroad", "offroad", "offroad"),
        ("left_corridor_laterally", "left_corridor_laterally", "corridor"))         # ol_clips.py table: flag, timing series, type
FLAGS = ("collision_at_fault", "offroad", "left_corridor_laterally")
LHT = ("singapore",)


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def dedupe(P):
    """Drop consecutive repeated points of a polyline (stations must increase)."""
    P = np.asarray(P, float)
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-6]
    return P[keep]


def at_station(R, S, s):
    s = np.atleast_1d(s)
    return np.c_[np.interp(s, S, R[:, 0]), np.interp(s, S, R[:, 1])]


def in_ego(a, X):
    """Points X (n, 2) of the local frame -> the frame of pose a = (x, y, yaw)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, float).reshape(-1, 2) - np.asarray(a[:2], float)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def to_local(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.asarray(a[:2], float) + np.c_[c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]]


def corners(xc, yc, psi, L, W):
    """Box centres / headings / sizes (n,) -> corners (n, 4, 2)."""
    c, s = np.cos(psi)[:, None], np.sin(psi)[:, None]
    u = np.array([1.0, 1.0, -1.0, -1.0])[None] * (np.asarray(L)[:, None] / 2)
    v = np.array([1.0, -1.0, -1.0, 1.0])[None] * (np.asarray(W)[:, None] / 2)
    return np.stack([xc[:, None] + c * u - s * v, yc[:, None] + s * u + c * v], -1)


# ---------------------------------------------------------------- per-scene pieces (AlpaSim venv)
class Lanes:
    """Lane class of a point and heading against the scene's vector map: the evaluator's own lane polygon and centre-line direction
    (eval.scorers.offroad). same = some containing lane within 60 deg; else onc = some beyond 120 deg; else cross; none = no lane."""

    def __init__(self, vm):
        import shapely
        from eval.scorers.offroad import _get_center_line_yaw_at_projection, _get_lane_polygon
        self.vm, self.c, self.sh, self.poly, self.yaw = vm, {}, shapely, _get_lane_polygon, _get_center_line_yaw_at_projection

    def lane(self, ln):
        if ln.id not in self.c:
            p = self.poly(ln)
            self.sh.prepare(p)
            self.c[ln.id] = (p, self.sh.LineString(np.asarray(ln.center.points)[:, :2]))
        return self.c[ln.id]

    def cls(self, x, y, z, psi):
        """-> (class code, junction flag): junction = two containing lanes more than 30 deg apart (overlapping connectors)."""
        pt = self.sh.Point(float(x), float(y))
        yaws = []
        for ln in self.vm.get_lanes_within(np.array([x, y, z], float), LANE_R):
            poly, cl = self.lane(ln)
            if poly.contains(pt):
                yaws.append(float(self.yaw(cl, pt)))
        if not yaws:
            return 0, False
        yaws = np.array(yaws)
        d = np.abs(wrap(yaws - psi))
        junc = bool(len(yaws) > 1 and np.abs(wrap(yaws[:, None] - yaws[None])).max() > JUNC)
        return (1 if (d < SAME).any() else 2 if (d > ONC).any() else 3), junc


def objects(ds):
    out = []
    for oid in ds.traffic_objects.keys():
        o = ds.traffic_objects[oid]
        t = o.trajectory
        T = np.asarray(t.timestamps_us, float)
        if not len(T):
            continue
        out.append(dict(id=str(oid), T=T, P=np.asarray(t.positions, float)[:, :2], Y=np.unwrap(np.asarray(t.yaws, float)), L=float(o.aabb.x),
                        W=float(o.aabb.y), static=bool(o.is_static), cls=str(o.label_class)))
    return out


def obj_state(objs, t):
    """Logged traffic at time t (us; the simulator replays it) -> dict of arrays: id, x, y, yaw, L, W, v (0.5 s backward difference,
    forward at a track's start), static. Only tracks whose logged time range covers t."""
    rows = []
    for o in objs:
        T, P, Y = o["T"], o["P"], o["Y"]
        if t < T[0] - 1 or t > T[-1] + 1:
            continue
        p = np.array([np.interp(t, T, P[:, 0]), np.interp(t, T, P[:, 1])])
        if t - 5e5 >= T[0] - 1:
            q, h = np.array([np.interp(t - 5e5, T, P[:, 0]), np.interp(t - 5e5, T, P[:, 1])]), 0.5
            v = np.hypot(*(p - q)) / h
        elif t + 5e5 <= T[-1] + 1:
            q = np.array([np.interp(t + 5e5, T, P[:, 0]), np.interp(t + 5e5, T, P[:, 1])])
            v = np.hypot(*(q - p)) / 0.5
        else:
            v = np.hypot(*(P[-1] - P[0])) / max((T[-1] - T[0]) * 1e-6, 0.1) if len(T) > 1 else 0.0
        rows.append((o["id"], p[0], p[1], float(np.interp(t, T, Y)), o["L"], o["W"], v, o["static"]))
    if not rows:
        return dict(id=np.array([], str), x=np.zeros(0), y=np.zeros(0), yaw=np.zeros(0), L=np.zeros(0), W=np.zeros(0), v=np.zeros(0),
                    static=np.zeros(0, bool))
    c = list(zip(*rows))
    return dict(id=np.array(c[0]), x=np.array(c[1]), y=np.array(c[2]), yaw=np.array(c[3]), L=np.array(c[4]), W=np.array(c[5]),
                v=np.array(c[6]), static=np.array(c[7], bool))


def rel(st, a, front):
    """Traffic state in the ego frame of pose a: centre (n, 2), heading difference, gap from the ego's front bumper to the object's
    rearmost corner along x."""
    E = in_ego(a, np.c_[st["x"], st["y"]])
    dpsi = wrap(st["yaw"] - a[2])
    C = corners(E[:, 0], E[:, 1], dpsi, st["L"], st["W"])
    return E, dpsi, C[..., 0].min(1) - front


def ego_poly(a, rear, front, hw):
    import shapely
    C = to_local(a, np.array([[front, hw], [front, -hw], [rear, -hw], [rear, hw]]))
    return shapely.Polygon(C)


def obj_poly(st, i):
    import shapely
    return shapely.Polygon(corners(st["x"][i:i + 1], st["y"][i:i + 1], st["yaw"][i:i + 1], st["L"][i:i + 1], st["W"][i:i + 1])[0])


def series(run, scene, rid):
    import glob
    import polars as pl
    f = glob.glob(f"{run}/rollouts/{scene}/{rid or '*'}/metrics.parquet") or glob.glob(f"{run}/rollouts/{scene}/*/metrics.parquet")
    if not f:
        return None, None
    m = pl.read_parquet(f[0]).filter(pl.col("name").is_in(list(SER))).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    ser = {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}
    asl = glob.glob(str(Path(f[0]).parent / "rollout.asl"))
    return ser, (asl[0] if asl else None)


def asl_poses(f):
    """actor_poses of a rollout.asl -> {actor id: (n, 4) t_us, x, y, yaw}."""
    import asyncio
    from c1_extract import _read, p3

    async def go():
        A = {}
        async for _, m in _read(f, {"actor_poses"}):
            t = int(m.timestamp_us)
            for x in m.actor_poses:
                A.setdefault(x.actor_id, []).append([t, *p3(x.actor_pose)])
        return {k: np.array(v, float) for k, v in A.items()}
    return asyncio.run(go())


def ev_type(score, flags, ser, now):
    """ol_clips.py `table`: for a zero-score rollout, the earliest flagged failure -> (type, decision, kinds)."""
    if score is None or float(score) != 0:
        return "", -1, ""
    first = {}
    for flag, m, kind in KIND:
        v = ser.get(m, {})
        if float(flags.get(flag) or 0) > 0:
            ts = sorted(t for t, x in v.items() if (x or 0) > 0)
            if ts:
                first[kind] = (max(int(np.searchsorted(now, ts[0], side="right")) - 1, 0), ts[0])
    if not first:
        return "other", -1, "-"
    typ = min(first, key=lambda q: first[q][1])
    return typ, first[typ][0], "+".join(first)


def scene_job(job):
    """(scene, scene-loader config, {label: dict(run, rid, score, metrics, recs)}) -> dict(dec=[...], roll=[...], gate={...})."""
    scene, cfg, labs = job
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    from ao_e1 import loader, project, stations
    ds = loader(cfg).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg, Yg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.unwrap(np.asarray(tr.yaws, float))
    vc = ds.rig.vehicle_config
    rear, front, hw = float(vc.aabb_x_offset_m), float(vc.aabb_x_offset_m + vc.aabb_x_m), float(vc.aabb_y_m) / 2
    loc = str(getattr(ds.scene, "location", "") or "")
    s_onc = -1 if any(x in loc.lower() for x in LHT) else 1
    Lp = dedupe(np.vstack([Pg[:, :2], Pg[-1, :2] + EXT * np.array([np.cos(Yg[-1]), np.sin(Yg[-1])])]))
    SL = stations(Lp)
    Sg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Pg[:, :2], axis=0).T))]       # the log's own station at each GT sample
    zt = lambda t: float(np.interp(t, Tg, Pg[:, 2]))                           # noqa: E731
    # AlpaSim's route, extended by every decision as the rollouts extended it (ao_e1); descriptive only (route60), so a failure is counted
    try:
        g = RouteGenerator.create(Pg, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=START_M)
        for d in labs.values():
            for r in d["recs"]:
                x, y, yaw = r["anchor"]
                g.generate_route(int(r["t0"]), Pose(np.array([x, y, zt(r["t0"])]), np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])))
        Rr = dedupe(np.asarray(g.route_polyline_in_local.points, float)[:, :2])
        route_err = 0
    except Exception:  # noqa: BLE001
        Rr, route_err = Lp.copy(), 1
    SR = stations(Rr)
    objs = objects(ds)
    tk = T0 + DT * np.arange(NK)
    states = [obj_state(objs, t) for t in tk]
    lanes = Lanes(ds.map)
    lane_log = [lanes.cls(np.interp(t, Tg, Pg[:, 0]), np.interp(t, Tg, Pg[:, 1]), zt(t), np.interp(t, Tg, Yg)) for t in tk]
    dec, roll = [], []
    gate = dict(act_pairs=0, act_ok=0, act_max=0.0, act_missing=0, ego_pairs=0, ego_ok=0, ego_max=0.0, rollouts_asl=0, route_err=route_err)
    for lab, d in labs.items():
        recs = sorted(d["recs"], key=lambda r: r["t0"])
        ser, asl = series(d["run"], scene, d["rid"])
        base = dict(label=lab, scene=scene, loc=loc, s_onc=s_onc)
        if ser is None:
            roll.append(dict(base, has_rollout=False))
            continue
        at = lambda n, t: ser.get(n, {}).get(int(t))                             # noqa: E731
        anch = {int((r["t0"] - T0) // DT): np.asarray(r["anchor"], float) for r in recs}
        hint_l = hint_r = None
        for r in recs:
            k = int((r["t0"] - T0) // DT)
            a = anch[k]
            t = T0 + DT * k
            if k - 1 in anch:
                v_e = np.hypot(*(a[:2] - anch[k - 1][:2])) / 0.5
            else:
                h = np.asarray(r["hist"], float)
                v_e = np.hypot(*(h[-1, :2] - h[-2, :2])) / 0.5
            lo, hi = (-np.inf, np.inf) if hint_l is None else (hint_l - 30, hint_l + 60)
            s_e, e = project(Lp, SL, a[:2], lo, hi)
            hint_l = s_e
            lo, hi = (-np.inf, np.inf) if hint_r is None else (hint_r - 30, hint_r + 60)
            s_r, _ = project(Rr, SR, a[:2], lo, hi)
            hint_r = s_r
            ch = at_station(Rr, SR, [s_r, s_r + 5, s_r + 55, s_r + 60])
            route60 = np.nan if route_err else np.degrees(wrap(np.arctan2(*(ch[3] - ch[2])[::-1]) - np.arctan2(*(ch[1] - ch[0])[::-1])))
            P = np.asarray(r["poses"], float)
            Pm = to_local(a, P[:, :2])
            yl = [project(Lp, SL, Pm[i], s_e - 10, s_e + 80)[1] for i in (3, 5, 7)]
            xl, ylg, yawl = np.interp(t, Tg, Pg[:, 0]), np.interp(t, Tg, Pg[:, 1]), np.interp(t, Tg, Yg)
            st = states[k]
            E, dpsi, gap = rel(st, a, front)
            box = (gap > 0) & (gap <= LEAD_GAP) & (np.abs(E[:, 1]) < LEAD_Y) & (np.abs(dpsi) < LEAD_PSI)
            close = (st["v"] < v_e - 1.0) | ((st["v"] < 0.5) & (v_e > 1.0))
            lead = box & close
            ib = np.flatnonzero(box)
            ib = ib[np.argmin(gap[ib])] if len(ib) else -1
            il = np.flatnonzero(lead)
            il = il[np.argmin(gap[il])] if len(il) else -1
            ahead = (E[:, 0] > 0) & (E[:, 0] < 30) & (np.abs(E[:, 1]) < 3.0)
            front_pt = to_local(a, [[front, 0.0]])[0]
            pend_yaw = a[2] + P[7, 2]
            lr, jr = lanes.cls(a[0], a[1], zt(t), a[2])
            lf, _ = lanes.cls(front_pt[0], front_pt[1], zt(t), a[2])
            lpl, _ = lanes.cls(Pm[7, 0], Pm[7, 1], zt(t), pend_yaw)
            t4 = t + 4_000_000
            dec.append(dict(base, k=k, t_us=t, n_keys=r.get("n_keys"), x=a[0], y=a[1], yaw=a[2], v_e=v_e, s_e=s_e, e=e,
                            x_log=xl, y_log=ylg, yaw_log=yawl, s_log_t=float(np.interp(t, Tg, Sg)),
                            log_dpsi4=float(np.interp(t4, Tg, Yg) - yawl) if t4 <= Tg[-1] else np.nan, plan_dpsi4=float(P[7, 2]),
                            cmd=int(r["cmd"]), route0_x=(r.get("route0") or [np.nan, np.nan])[0], route0_y=(r.get("route0") or [np.nan, np.nan])[1],
                            route60=route60, yh2=P[3, 1], yh3=P[5, 1], yh4=P[7, 1], yl2=yl[0], yl3=yl[1], yl4=yl[2], px4=Pm[7, 0], py4=Pm[7, 1],
                            box=bool(ib >= 0), box_gap=gap[ib] if ib >= 0 else np.nan, box_v=st["v"][ib] if ib >= 0 else np.nan,
                            box_id=st["id"][ib] if ib >= 0 else "", lead=bool(il >= 0), lead_gap=gap[il] if il >= 0 else np.nan,
                            lead_v=st["v"][il] if il >= 0 else np.nan, lead_y=E[il, 1] if il >= 0 else np.nan, lead_id=st["id"][il] if il >= 0 else "",
                            n_ahead=int(ahead.sum()), lane_rear=LANE[lr], junc_rear=jr, lane_front=LANE[lf], lane_plan=LANE[lpl],
                            lane_log=LANE[lane_log[k][0]], junc_log=lane_log[k][1],
                            **{n: at(n, t) for n in SER}))
        # rollout level: events (ol_clips rule), the evaluator's truncation step, the at-fault contact and the object hit
        now = np.array(sorted(r["now"] for r in recs), float)
        typ, kev, kinds = ev_type(d["score"], d["metrics"], ser, now)
        ks = sorted(anch)
        trunc = [k for k in ks if (at("dist_to_gt_trajectory", T0 + DT * k) or 0) >= 4.0 or (at("collision_any", T0 + DT * k) or 0) > 0
                 or (at("offroad", T0 + DT * k) or 0) > 0]
        kc = next((k for k in ks if (at("collision_front", T0 + DT * k) or 0) > 0 or (at("collision_lateral", T0 + DT * k) or 0) > 0), -1)
        hit = {}
        if kc >= 0:
            st, ep = states[kc], ego_poly(anch[kc], rear, front, hw)
            if len(st["id"]):
                ov = np.array([ep.intersection(obj_poly(st, i)).area for i in range(len(st["id"]))])
                dist = np.array([ep.distance(obj_poly(st, i)) for i in range(len(st["id"]))])
                i = int(ov.argmax()) if ov.max() > 0 else int(dist.argmin())
                kb = max(kc - 2, 0)
                sb = states[kb]
                j = np.flatnonzero(sb["id"] == st["id"][i])
                hb = dict(hit_x=np.nan, hit_y=np.nan, hit_dpsi=np.nan, hit_v=np.nan)
                if len(j) and kb in anch:
                    Eb, db, _ = rel(sb, anch[kb], front)
                    hb = dict(hit_x=Eb[j[0], 0], hit_y=Eb[j[0], 1], hit_dpsi=float(np.degrees(db[j[0]])), hit_v=sb["v"][j[0]])
                hit = dict(hit_id=st["id"][i], hit_area=float(ov[i]), hit_dist=float(dist[i]), hit_static=bool(st["static"][i]), **hb)
        if asl:                                                                        # gate G-act
            A = asl_poses(asl)
            gate["rollouts_asl"] += 1
            byid = {o["id"]: o for o in objs}
            for aid, X in A.items():
                if aid == "EGO":                                                   # the asl keeps EGO at its box centre, the driver the rear axle
                    for t, x, y, _ in X:
                        k = int((t - T0) // DT)
                        if t >= T0 and (t - T0) % DT == 0 and k in anch:
                            c = to_local(anch[k], [[(rear + front) / 2, 0.0]])[0]
                            dd = float(np.hypot(x - c[0], y - c[1]))
                            gate["ego_pairs"] += 1
                            gate["ego_ok"] += dd <= 0.05
                            gate["ego_max"] = max(gate["ego_max"], dd)
                    continue
                o = byid.get(aid)
                if o is None:
                    gate["act_missing"] += len(X)
                    continue
                inr = (X[:, 0] >= o["T"][0] - 1) & (X[:, 0] <= o["T"][-1] + 1)
                gate["act_missing"] += int((~inr).sum())
                X = X[inr]
                dd = np.hypot(X[:, 1] - np.interp(X[:, 0], o["T"], o["P"][:, 0]), X[:, 2] - np.interp(X[:, 0], o["T"], o["P"][:, 1]))
                gate["act_pairs"] += len(dd)
                gate["act_ok"] += int((dd <= 0.05).sum())
                gate["act_max"] = max(gate["act_max"], float(dd.max()) if len(dd) else 0.0)
        roll.append(dict(base, has_rollout=True, run=d["run"], rid=d["rid"], score=d["score"], n_dec=len(recs), has_asl=bool(asl),
                         **{f: float(d["metrics"].get(f) or 0) for f in FLAGS}, dist_to_gt=d["metrics"].get("dist_to_gt_trajectory"),
                         type=typ, k_ev=kev, kinds=kinds, k_end=trunc[0] if trunc else (ks[-1] if ks else -1), k_c=kc,
                         dpsi_log=float(np.degrees(np.interp(tk[-1], Tg, Yg) - np.interp(tk[0], Tg, Yg))), **hit))
    lg = np.array([c for c, _ in lane_log])
    gate.update(log_lane_n=int((lg > 0).sum()), log_same=int((lg == 1).sum()), log_onc=int((lg == 2).sum()), log_cross=int((lg == 3).sum()),
                log_none=int((lg == 0).sum()), ego_dims=[rear, front, hw])
    return dict(dec=dec, roll=roll, gate=gate)


def collect(a):
    import pandas as pd
    from ao_e1 import drive_records
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    held = splits.load("alpasim_nuplan/onpol-held")
    runs = dict(s.split("=", 1) for s in a.runs)
    with Run("zeros_mech", "collect", config=vars(a)) as run:
        run.use_split(held)
        keep = set(held.members)
        jobs, cfg = {}, None
        for lab, D in runs.items():
            cfg = cfg or str(Path(D) / "generated-user-config-0.yaml")
            recs = drive_records(D)
            for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]:
                s = r["clipgt_id"]
                if s in keep:
                    jobs.setdefault(s, {})[lab] = dict(run=D, rid=r.get("rollout_id"), score=r.get("score"), metrics=r.get("metrics") or {},
                                                       recs=recs.get(s, []))
        norec = sorted(s for s, v in jobs.items() if not any(x["recs"] for x in v.values()))   # route-fold scenes: no rollout in any label
        units = [(s, cfg, v) for s, v in sorted(jobs.items()) if s not in norec][: a.limit or None]
        workers = a.workers or min(64, n_cpus())
        run.info(f"{len(units)} scenes x {len(runs)} labels ({len(norec)} without any driver record skipped), {workers} workers, scene config {cfg}")
        res = par.pmap(scene_job, units, workers=workers, run=run, desc="scenes")
        out = Path(run.dir)
        err = [dict(scene=units[i][0], err=res.errors[i].strip().splitlines()[-1][:300]) for i in sorted(res.errors)]
        pd.DataFrame(err, columns=["scene", "err"]).to_csv(out / "errors.csv", index=False)
        vals = [v for v in res.values if v]
        D = pd.DataFrame([x for v in vals for x in v["dec"]])
        R = pd.DataFrame([x for v in vals for x in v["roll"]])
        D.to_parquet(out / "per_decision.parquet", index=False)
        R.to_parquet(out / "per_rollout.parquet", index=False)
        G = {k: sum(v["gate"][k] for v in vals) for k in vals[0]["gate"] if k not in ("act_max", "ego_max", "ego_dims")}
        G.update(act_max=max(v["gate"]["act_max"] for v in vals), ego_max=max(v["gate"]["ego_max"] for v in vals),
                 ego_dims=vals[0]["gate"]["ego_dims"], scenes=len(units), scene_errors=len(err), no_record_scenes=norec)
        G["G_act"] = dict(share=G["act_ok"] / max(G["act_pairs"], 1), missing=G["act_missing"], ego_share=G["ego_ok"] / max(G["ego_pairs"], 1))
        G["G_act"]["pass"] = bool(G["G_act"]["share"] >= 0.99 and G["act_missing"] == 0 and G["G_act"]["ego_share"] >= 0.99)
        G["G_dir"] = dict(same=G["log_same"] / max(G["log_lane_n"], 1), onc=G["log_onc"] / max(G["log_lane_n"], 1))
        G["G_dir"]["pass"] = bool(G["G_dir"]["same"] >= 0.95 and G["G_dir"]["onc"] <= 0.01)
        full = R[R.has_rollout].groupby(["label", "scene"]).n_dec.first()
        G["G_cov"] = dict(rollouts=int(len(full)), full18=float((full == NK).mean()), scene_error_share=len(err) / max(len(units), 1))
        G["G_cov"]["pass"] = bool(G["G_cov"]["full18"] >= 0.99 and G["G_cov"]["scene_error_share"] <= 0.01)
        zf = DATA / "runs/onpol_long/zeros/YS-L18-s1/zeros.csv"
        if zf.exists() and "YS-L18-s1" in runs:
            Z = pd.read_csv(zf)
            mine = R[(R.label == "YS-L18-s1") & (R.type != "")].set_index("scene")
            Zs = Z.set_index("scene")
            diff = [dict(scene=s, csv=f"{Zs.type.get(s, '-')}@{Zs.decision.get(s, '-')}", ours=f"{mine.type.get(s, '-')}@{mine.k_ev.get(s, '-')}")
                    for s in sorted(set(Zs.index) | set(mine.index))
                    if Zs.type.get(s) != mine.type.get(s) or int(Zs.decision.get(s, -9)) != int(mine.k_ev.get(s, -9))]
            G["G_zero"] = dict(csv=len(Z), ours=int(len(mine)), mismatches=diff)
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        run.summary.update(scenes=len(units), errors=len(err), decisions=len(D), rollouts=len(R), G_act=G["G_act"], G_dir=G["G_dir"],
                           G_cov=G["G_cov"], G_zero=len(G.get("G_zero", {}).get("mismatches", [])))
        run.info(json.dumps({k: G[k] for k in ("G_act", "G_dir", "G_cov")}, default=float))


# ---------------------------------------------------------------- rows (envs/op-train)
def rows(a):
    import pandas as pd
    import torch
    from agent_hinge import AgentHinge
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    torch.set_num_threads(max(1, min(32, n_cpus())))
    P, Rd = DATA / "runs/op_parity/cache" / a.pool, DATA / "runs/onpol_nuplan/rows" / a.rows_dir
    with Run("zeros_mech", "rows", config=vars(a)) as run:
        tab, act = np.load(P / "tab.npz"), np.load(Rd / "active.npz")
        names = tab["names"]
        assert np.array_equal(act["names"], names), "active.npz rows differ from the pool's"
        meta = np.load(Rd / "meta.npz")
        poses, kk, speed = meta["poses"].astype(np.float32), meta["k"], tab["speed"].astype(np.float64)
        H = AgentHinge(P / "agent.npz", names, torch.device("cpu"), 0.3, 0.0)
        idx = np.flatnonzero(act["agent"] > 0)
        out, gate = [], []
        with torch.no_grad():
            for i in range(0, len(names), 2048):
                r = torch.arange(i, min(i + 2048, len(names)))
                X = torch.from_numpy(poses[r.numpy()])
                dist, cnt, cor = H.distances(X[..., 0], X[..., 1], X[..., 2], r)
                h = (torch.where(cor, 0.3, 0.0) - dist).clamp_min(0).masked_fill(~cnt, 0.0)            # (B, 41, K)
                gate.append(h.amax(-1).mean(1).numpy())
                flat = h.flatten(1).argmax(1)
                out.append(np.c_[(flat // h.shape[2]).numpy(), (flat % h.shape[2]).numpy(),
                                 cor.flatten(1).gather(1, flat[:, None])[:, 0].numpy()])
        hv, ob = np.concatenate(gate), np.concatenate(out)
        ok_gate = float((np.abs(hv - act["agent"]) <= 1e-4).mean())
        z = np.load(P / "agent.npz")
        zbox, zval = z["box"], z["valid"]                                   # once: every key access re-reads the npz
        pos = {t: j for j, t in enumerate(z["tokens"].tolist())}
        rows_ = []
        for i in idx:
            j, s_, o, cor_ = pos[names[i]], int(ob[i, 0]), int(ob[i, 1]), bool(ob[i, 2])
            val = zval[j, :, o]
            t0 = int(np.flatnonzero(val)[0]) if val.any() else -1
            b = zbox[j, max(t0, 0), o]
            dpsi = float(wrap(b[2]))
            C = corners(np.array([b[0]]), np.array([b[1]]), np.array([dpsi]), np.array([b[3]]), np.array([b[4]]))[0]
            v_o = float(np.hypot(*(zbox[j, 1, o, :2] - zbox[j, 0, o, :2])) / 0.5) if val[0] and val[1] else np.nan
            rows_.append(dict(name=names[i], k=int(kk[i]), agent=float(act["agent"][i]), agent_recomputed=float(hv[i]), step=s_, obj=o,
                              corridor=cor_, t_first=t0, x=float(b[0]), y=float(b[1]), dpsi=float(np.degrees(dpsi)), L=float(b[3]),
                              W=float(b[4]), gap=float(C[:, 0].min() - 4.049), v_obj=v_o, v_ego=float(speed[i])))
        Df = pd.DataFrame(rows_)
        Df.to_parquet(Path(run.dir) / "rows.parquet", index=False)
        G = dict(rows=int(len(names)), active=int(len(idx)), active_share=float(len(idx) / len(names)), G_row_share=ok_gate,
                 G_row_pass=bool(ok_gate >= 0.999), max_abs_diff=float(np.abs(hv - act["agent"]).max()))
        (Path(run.dir) / "gates.json").write_text(json.dumps(G, indent=1))
        run.summary.update(G)
        run.info(json.dumps(G))


# ---------------------------------------------------------------- bev (AlpaSim venv)
def bev(a):
    """Map, traffic and plan around chosen decisions, in the ego frame of that decision, for the example strip."""
    import pickle
    import pandas as pd
    from ao_e1 import drive_records, loader
    C = Path(a.collect)
    picks = json.loads(Path(a.picks).read_text())
    R = pd.read_parquet(C / "per_rollout.parquet")
    cfg = str(Path(R[R.has_rollout].run.iloc[0]) / "generated-user-config-0.yaml")
    out, recs_of = {}, {}
    for p in picks:
        lab, scene, k = p["label"], p["scene"], int(p["k"])
        runs = R[(R.label == lab) & (R.scene == scene)].run
        if runs.empty:
            continue
        if lab not in recs_of:
            recs_of[lab] = drive_records(runs.iloc[0])
        r = next(x for x in recs_of[lab][scene] if int((x["t0"] - T0) // DT) == k)
        ds = loader(cfg).get_data_source(scene)
        tr = ds.rig.trajectory
        Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
        vc = ds.rig.vehicle_config
        a_ = np.asarray(r["anchor"], float)
        t = T0 + DT * k
        lanes = Lanes(ds.map)
        L = []
        for ln in ds.map.get_lanes_within(np.array([a_[0], a_[1], float(np.interp(t, Tg, Pg[:, 2]))]), 45.0):
            poly, cl = lanes.lane(ln)
            geoms = getattr(poly, "geoms", [poly])
            ext = [in_ego(a_, np.asarray(gm.exterior.coords)[:, :2]) for gm in geoms if hasattr(gm, "exterior")]
            ctr = np.asarray(ln.center.points)[:, :2]
            hd = np.arctan2(*(ctr[-1] - ctr[0])[::-1]) if len(ctr) > 1 else 0.0
            L.append(dict(ext=ext, center=in_ego(a_, ctr), dpsi=float(np.degrees(wrap(hd - a_[2])))))
        st = obj_state(objects(ds), t)
        E, dpsi, gap = rel(st, a_, float(vc.aabb_x_offset_m + vc.aabb_x_m))
        out[f"{lab}|{scene}|{k}"] = dict(pick=p, lanes=L, objs=dict(id=st["id"], x=E[:, 0], y=E[:, 1], dpsi=dpsi, L=st["L"], W=st["W"], v=st["v"]),
                                         plan=np.asarray(r["poses"], float), log=in_ego(a_, Pg[:, :2]), cmd=int(r["cmd"]),
                                         ego=(float(vc.aabb_x_offset_m), float(vc.aabb_x_offset_m + vc.aabb_x_m), float(vc.aabb_y_m) / 2))
    pickle.dump(out, open(C / "bev.pkl", "wb"), protocol=4)
    print(len(out), "BEV examples ->", C / "bev.pkl")


# ---------------------------------------------------------------- report (Mac .venv)
GROUPS = {"base": ("YR10m10-F-s0",), "YS": ("YS-L18-s0", "YS-L18-s1"), "YS2G": ("YS2G-L18-s0", "YS2G-L18-s1"),
          "YNONE": ("YNONE-L18-s0", "YNONE-L18-s1"), "YR-s1": ("YR10m10-F-s1",)}
COLOR = {"base": "#777777", "YS": "#0072B2", "YS2G": "#009E73", "YNONE": "#E69F00", "YR-s1": "#56B4E9"}
LEFT, STRAIGHT, RIGHT = 0, 1, 2


class Boot:
    """Log-clustered bootstrap in jevdrive.stats' convention: one draw of logs (default_rng(0), 10 000), statistics as ratios of
    per-log sums; every statistic of the report uses the same draws, so differences between labels are under one resample."""

    def __init__(self, logs, n=10_000, seed=0):
        self.logs = np.array(sorted(set(logs)))
        self.ix = {g: i for i, g in enumerate(self.logs)}
        idx = np.random.default_rng(seed).integers(len(self.logs), size=(n, len(self.logs)))
        self.M = np.stack([np.bincount(r, minlength=len(self.logs)) for r in idx]).astype(float)

    def sums(self, x, logs):
        x, i = np.asarray(x, float), np.array([self.ix[g] for g in logs], int)
        ok = np.isfinite(x)
        return np.bincount(i[ok], x[ok], len(self.logs)), np.bincount(i[ok], minlength=len(self.logs)).astype(float)

    def boot(self, s, c):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.M @ s) / (self.M @ c)

    @staticmethod
    def ci(b):
        b = np.asarray(b, float)
        b = b[np.isfinite(b)]
        return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))] if len(b) else [np.nan, np.nan]

    def mean(self, x, logs):
        s, c = self.sums(x, logs)
        return dict(n=int(c.sum()), mean=float(s.sum() / c.sum()) if c.sum() else np.nan, ci=self.ci(self.boot(s, c)))

    def diff(self, xa, la, xb, lb):
        (sa, ca), (sb, cb) = self.sums(xa, la), self.sums(xb, lb)
        pt = sa.sum() / ca.sum() - sb.sum() / cb.sum() if ca.sum() and cb.sum() else np.nan
        return dict(n_a=int(ca.sum()), n_b=int(cb.sum()), mean=float(pt), ci=self.ci(self.boot(sa, ca) - self.boot(sb, cb)))

    def ratio(self, num, den, logs):
        """1 - sum(num) / sum(den) with its CI (the shortfall of a peak against the log's)."""
        (sn, _), (sd, _) = self.sums(num, logs), self.sums(den, logs)
        return float(1 - sn.sum() / sd.sum()), 1 - self.boot(sn, sd)

    def wmedian(self, x, logs):
        """Lower median (inverted CDF; +inf allowed) under every draw -> (B,)."""
        x, i = np.asarray(x, float), np.array([self.ix[g] for g in logs], int)
        o = np.argsort(x, kind="stable")
        cw = np.cumsum(self.M[:, i[o]], 1)
        return x[o][(cw >= cw[:, -1:] / 2).argmax(1)]


def f3(r, k="mean", f=".3f"):
    v = r[k] if isinstance(r, dict) else r
    if not isinstance(r, dict) or "ci" not in r:
        return "-" if v is None or not np.isfinite(v) else f"{v:{f}}"
    lo, hi = r["ci"]
    return f"{v:{f}} [{lo:{f}}, {hi:{f}}]" if np.isfinite(v) else "-"


def md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "|".join([":--"] + ["--:"] * (len(head) - 1)) + "|"] + \
        ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def cross(c, t, lvl=0.5):
    """First time a cumulative fraction c (over times t) reaches lvl, linear between samples; NaN if never."""
    i = np.flatnonzero(c >= lvl)
    if not len(i):
        return np.nan
    i = i[0]
    return float(t[0]) if i == 0 else float(t[i - 1] + (lvl - c[i - 1]) / (c[i] - c[i - 1]) * (t[i] - t[i - 1]))


def drift_onset(e, k, kev):
    """First decision <= kev with |e| > 1 m after which e keeps its sign and |e| does not fall by more than 0.1 m per decision."""
    m = k <= kev
    e, k = e[m], k[m]
    for j in range(len(k)):
        if abs(e[j]) > 1.0:
            seg = e[j:]
            if (np.sign(seg) == np.sign(seg[0])).all() and (np.diff(np.abs(seg)) >= -0.1).all():
                return int(k[j])
    return int(kev)


def obj_class(x, y, dpsi_deg, ahead_x=1.461, lat=(1.5, 3.0)):
    """In the ego frame: crossing / oncoming by heading first, then in-lane lead (ahead, |y| < 1.5 m, < 30 deg), adjacent, other."""
    d = abs(dpsi_deg)
    if not np.isfinite(d):
        return "-"
    if d > 120:
        return "oncoming"
    if d > 60:
        return "crossing"
    if x > ahead_x and abs(y) < lat[0] and d < 30:
        return "in-lane lead"
    if lat[0] <= abs(y) < lat[1]:
        return "adjacent"
    return "other"


def report(a):
    import pandas as pd
    C, Rw, out, figs = Path(a.collect), Path(a.rows), Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True), figs.mkdir(parents=True, exist_ok=True)
    D, R = pd.read_parquet(C / "per_decision.parquet"), pd.read_parquet(C / "per_rollout.parquet")
    G, GR = json.loads((C / "gates.json").read_text()), json.loads((Rw / "gates.json").read_text())
    RW = pd.read_parquet(Rw / "rows.parquet")
    labels = sorted(R.label.unique())
    ok = R[R.has_rollout & R.score.notna()].groupby("scene").label.nunique()
    scenes = set(ok[ok == len(labels)].index)
    R = R[R.scene.isin(scenes) & R.has_rollout].copy()
    D = D[D.scene.isin(scenes)].copy()
    for X in (R, D):
        X["log"] = X.scene.str.rsplit("-", n=1).str[0]
    B = Boot(R.log)
    grp = {lab: g for g, ls in GROUPS.items() for lab in ls}
    R["grp"], D["grp"] = R.label.map(grp), D.label.map(grp)
    D = D.merge(R[["label", "scene", "k_end", "type", "k_ev", "k_c", "dpsi_log"]], on=["label", "scene"], how="left")
    D["in_play"] = D.k <= D.k_end
    for t in (2, 3, 4):
        D[f"d{t}"] = D[f"yl{t}"] - D.e
    D["ad3"], D["sd3"] = D.d3.abs(), D.s_onc * D.d3
    D["cond"] = np.where(D.lead, "lead", np.where(D.box, "follow", "no-lead"))
    D["onc_rear"], D["onc_front"], D["wl"] = (D.lane_rear == "onc").astype(float), (D.lane_front == "onc").astype(float), (D.wrong_lane > 0).astype(float)
    R["zero"] = R.score == 0
    R["scls"] = np.where(R.dpsi_log.abs() >= 45, "turn", np.where(R.dpsi_log.abs() < 15, "straight", "bend"))
    R["clean"] = (R.collision_at_fault == 0) & (R.offroad == 0) & (R.left_corridor_laterally == 0)
    J = dict(labels=labels, scenes=len(scenes), logs=len(B.logs), collect=str(C), rows=str(Rw), gates=G, gates_rows=GR)
    Dk = {key: x.sort_values("k") for key, x in D.groupby(["label", "scene"])}
    L = []

    def sel(df, g):
        return df[df.label.isin(GROUPS[g] if g in GROUPS else (g,))]

    # ------------------------------------------------ per rollout: R1 timing, events, collisions
    rr = []
    for r in R.itertuples():
        x = Dk.get((r.label, r.scene))
        if x is None or not len(x) or x.k.iloc[0] != 0:
            continue
        k, t = x.k.to_numpy(), 0.017 + 0.5 * x.k.to_numpy()
        pe, pl_ = np.unwrap(x.yaw.to_numpy()), np.unwrap(x.yaw_log.to_numpy())
        e, cmd = x.e.to_numpy(), x.cmd.to_numpy()
        sg, tot = np.sign(r.dpsi_log), np.radians(abs(r.dpsi_log))
        o = dict(label=r.label, grp=r.grp, scene=r.scene, log=r.log, scls=r.scls, dpsi_log=r.dpsi_log, type=r.type, k_ev=r.k_ev, k_c=r.k_c,
                 zero=r.zero, clean=r.clean, s_onc=float(x.s_onc.iloc[0]))
        if tot > 0:
            cl, ce = sg * (pl_ - pl_[0]) / tot, sg * (pe - pe[0]) / tot
            tl, te = cross(cl, t), cross(ce, t)
            o.update(t_log=tl, t_e=te, delta=(te - tl) if np.isfinite(te) else np.inf,
                     ds=(np.interp(te, t, x.s_e) - np.interp(tl, t, x.s_log_t)) if np.isfinite(te) else np.inf)
            nz = np.flatnonzero(np.isin(cmd, (LEFT, RIGHT)))
            o.update(t_cmd=float(t[nz[0]]) if len(nz) else np.nan, cmd_dir_ok=float(np.mean((cmd[nz] == LEFT) == (sg > 0))) if len(nz) else np.nan)
        dt = np.diff(t)
        o.update(w_e=float(np.max(np.abs(np.diff(pe)) / dt)), w_log=float(np.max(np.abs(np.diff(pl_)) / dt)))
        if len(k) >= 3:
            arc = lambda X, Y: np.hypot(np.diff(X), np.diff(Y))                      # noqa: E731
            ae, al = arc(x.x.to_numpy(), x.y.to_numpy()), arc(x.x_log.to_numpy(), x.y_log.to_numpy())
            o.update(kap_e=float(np.max(np.abs(pe[2:] - pe[:-2]) / np.maximum(ae[1:] + ae[:-1], 2.0))),
                     kap_log=float(np.max(np.abs(pl_[2:] - pl_[:-2]) / np.maximum(al[1:] + al[:-1], 2.0))))
        if r.zero and r.type in ("corridor", "offroad", "collision"):
            kev = int(r.k_ev)
            at = x[x.k <= kev]
            e_ev = float(at.e.iloc[-1]) if len(at) else np.nan
            o.update(e_ev=e_ev, inside=bool(sg * e_ev > 0) if r.scls == "turn" else None)
            if r.scls == "turn":
                late = not np.isfinite(o["ds"]) or o["ds"] > 0
                o["cls"] = ("late" if late else "early") + "-" + ("inside" if o["inside"] else "outside")
                late_t = not np.isfinite(o["delta"]) or o["delta"] > 0
                o["cls_t"] = ("late" if late_t else "early") + "-" + ("inside" if o["inside"] else "outside")
            else:
                kd = drift_onset(e, k, kev)
                w = (k >= kd - 3) & (k <= kd)
                nzw = np.flatnonzero(w & np.isin(cmd, (LEFT, RIGHT)))
                flip = bool(any(j >= 1 and w[j] and cmd[j] in (LEFT, RIGHT) and cmd[j - 1] == STRAIGHT for j in range(len(k))))
                o.update(k_d=kd, cmd_win=bool(len(nzw)), cmd_flip=flip, cmd_persist=bool(np.isin(cmd[k <= kd], (LEFT, RIGHT)).all()),
                         cmd_same=bool((cmd[nzw[-1]] == LEFT) == (e_ev > 0)) if len(nzw) else None,
                         route_turns=bool(abs(x.route60.to_numpy()[nzw[-1]]) >= 20) if len(nzw) and np.isfinite(x.route60.to_numpy()[nzw[-1]]) else None)
            kw = int(r.k_c) if r.type == "collision" and r.k_c >= 0 else kev
            win = x[(x.k >= kw - 6) & (x.k <= kw)]
            o.update(lead_win=bool(win.lead.any()), defl_win=bool((win.ad3 >= 0.5).any()),
                     onc_pre=bool((x[(x.k >= kev - 4) & (x.k <= kev)].lane_rear == "onc").any()),
                     lane_ev=str(at.lane_rear.iloc[-1]) if len(at) else "-")
        if r.collision_at_fault > 0:
            kc = int(r.k_c) if r.k_c >= 0 else int(r.k_ev)
            win = x[(x.k >= kc - 6) & (x.k <= kc)]
            lead_ids = set(win.lead_id[win.lead])
            lw, dw = bool(win.lead.any()), bool((win.ad3 >= 0.5).any())
            o.update(coll=True, kc=kc, c_lead=lw, c_defl=dw, c_cls="cut-around" if lw and dw else "straight-in" if lw else "swerve, no lead" if dw else "drift",
                     hit_cls=obj_class(r.hit_x, r.hit_y, r.hit_dpsi) if pd.notna(getattr(r, "hit_x", np.nan)) else "-",
                     hit_static=bool(r.hit_v < 0.5) if pd.notna(getattr(r, "hit_v", np.nan)) else None, hit_is_lead=bool(r.hit_id in lead_ids) if isinstance(r.hit_id, str) else None)
        rr.append(o)
    RR = pd.DataFrame(rr)
    RR.drop(columns=["log"]).round(4).to_csv(out / "per_rollout_reads.csv.gz", index=False)

    # ------------------------------------------------ context: zeros by label
    L += [f"# ZEROS-MECH: what the S-family zeros come from (held {len(scenes)} scenes, {len(B.logs)} logs, {len(labels)} labels, 18 decisions)", "",
          "Registered in [plans/2026-10-12-zeros-mech-prereg.md](../plans/2026-10-12-zeros-mech-prereg.md) (pushed before any read); script "
          "`scripts/zm_read.py`; box run dirs `" + str(C.name) + "` (collect) and `" + str(Rw.name) + "` (rows) under `$DATA_DIR/runs/zeros_mech/`. "
          "Groups: base = `YR10m10-F-s0` (YS's start), YS = `YS-L18-s0/s1`, YS2G = `YS2G-L18-s0/s1`, YNONE = `YNONE-L18-s0/s1`, YR-s1 = `YR10m10-F-s1`. "
          "CIs: log-clustered bootstrap (39 logs, 10 000 draws, one draw for all statistics).", ""]
    ga, gd, gc = G["G_act"], G["G_dir"], G["G_cov"]
    L += ["## Gates", ""] + md_table(["gate", "line", "read", "verdict"], [
        ["G-cov", ">= 99% of rollouts with 18 decisions; scene errors <= 1%", f"{gc['full18']:.4f} of {gc['rollouts']} rollouts; {G['scene_errors']} scene errors", "pass" if gc["pass"] else "FAIL"],
        ["G-act", "asl actor poses vs replayed log <= 0.05 m on >= 99%; EGO vs anchor", f"{G['act_ok']} / {G['act_pairs']} pairs ({G['rollouts_asl']} rollouts, max {G['act_max']:.1e} m), "
         f"missing {G['act_missing']}; EGO {G['ego_ok']} / {G['ego_pairs']} (max {G['ego_max']:.1e} m)", "pass" if ga["pass"] else "FAIL"],
        ["G-dir", "logged poses in a lane: same >= 95%, oncoming <= 1%", f"same {gd['same']:.4f}, oncoming {gd['onc']:.4f} of {G['log_lane_n']} ({G['log_none']} in no lane)", "pass" if gd["pass"] else "FAIL"],
        ["G-row", "recomputed agent hinge = active.npz (1e-4) on >= 99.9% of rows", f"{GR['G_row_share']:.4f} of {GR['rows']} rows (max diff {GR['max_abs_diff']:.1e}); {GR['active']} active ({GR['active_share']:.3f})", "pass" if GR["G_row_pass"] else "FAIL"],
        ["G-zero", "YS-L18-s1 zeros = zeros.csv", f"{G['G_zero']['ours']} / {G['G_zero']['csv']}, {len(G['G_zero']['mismatches'])} mismatches", "pass" if not G["G_zero"]["mismatches"] else "see list"]]) + [
        "", "The first full collect (`20261012-154957`) failed G-cov (215 scene errors: 64 workers each started BLAS / polars / rayon thread pools "
        "sized to the box and hit the thread limit; 8 more were the route-fold scenes without rollouts) and was voided; the code was fixed (one "
        "thread per worker, those 8 scenes skipped) and the whole collect rerun once. The asl stores EGO at its box centre (1.461 m ahead of the "
        "rear axle); G-act compares it with the anchor moved there.", ""]
    zt = R[R.zero].groupby("label").type.value_counts().unstack(fill_value=0)
    atf = R.groupby("label").collision_at_fault.apply(lambda v: int((v > 0).sum()))
    L += ["## Zeros by label (context)", ""] + md_table(["label", "zeros", "corridor", "offroad", "collision", "other", "at-fault collisions"],
                                                      [[lab, int(zt.loc[lab].sum()) if lab in zt.index else 0] + [int(zt.loc[lab].get(c, 0)) if lab in zt.index else 0 for c in ("corridor", "offroad", "collision", "other")] + [int(atf.get(lab, 0))] for lab in labels]) + [""]
    J["zeros"] = {lab: {c: int(zt.loc[lab].get(c, 0)) for c in zt.columns} for lab in zt.index}

    # ------------------------------------------------ R1
    T = RR[RR.scls == "turn"]
    r1 = {}
    rows_ = []
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1"):
        x = sel(T, g)
        r1[g] = dict(n=int(len(x)), censored=float((~np.isfinite(x.delta)).mean()),
                     delta=[float(np.quantile(x.delta, q, method="inverted_cdf")) for q in (0.25, 0.5, 0.75)],
                     ds=[float(np.quantile(x.ds, q, method="inverted_cdf")) for q in (0.25, 0.5, 0.75)],
                     t_log_minus_cmd=float(np.nanmedian(x.t_log - x.t_cmd)), t_e_minus_cmd=float(np.nanmedian(x.t_e - x.t_cmd)),
                     cmd_dir_ok=float(np.nanmean(x.cmd_dir_ok)))
        rows_.append([g, r1[g]["n"], f"{r1[g]['censored']:.3f}", "{1:+.2f} [{0:+.2f}, {2:+.2f}]".format(*r1[g]["delta"]),
                      "{1:+.1f} [{0:+.1f}, {2:+.1f}]".format(*r1[g]["ds"]), f"{r1[g]['t_log_minus_cmd']:.2f}", f"{r1[g]['t_e_minus_cmd']:.2f}", f"{r1[g]['cmd_dir_ok']:.3f}"])
    for col in ("delta", "ds"):
        ya, yb = sel(T, "YS"), sel(T, "base")
        bd = B.wmedian(ya[col], ya.log) - B.wmedian(yb[col], yb.log)
        r1[f"YS-base_{col}"] = dict(mean=float(np.quantile(ya[col], 0.5, method="inverted_cdf") - np.quantile(yb[col], 0.5, method="inverted_cdf")), ci=B.ci(bd))
    L += ["## R1 turn timing and command (H1)", "",
          f"Turn scenes: the log turns >= 45 deg over the 18 decisions ({T.scene.nunique()} scenes); straight: < 15 deg "
          f"({RR[RR.scls == 'straight'].scene.nunique()}); bend: the rest. delta = t_ego - t_log at half the log's heading change "
          "(s; never reached = +inf, `censored`); ds = the same in station along the logged path (m; + = turns further along = late).", ""]
    L += md_table(["group", "turn rollouts", "censored", "delta median [IQR] (s)", "ds median [IQR] (m)", "t_log - t_cmd (s, median)", "t_ego - t_cmd", "command side = turn side"], rows_)
    L += ["", f"YS - base, median delta: {f3(r1['YS-base_delta'], f='+.2f')} s; median ds: {f3(r1['YS-base_ds'], f='+.2f')} m.", ""]
    # corridor zeros on turn scenes: classes
    Z = RR[RR.zero & (RR.type == "corridor")]
    ct = []
    cls_n = {}
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1") + tuple(labels):
        x = sel(Z[Z.scls == "turn"], g)
        c = x.cls.value_counts()
        c_t = x.cls_t.value_counts()
        cls_n[g] = dict(n=int(len(x)), **{k_: int(c.get(k_, 0)) for k_ in ("early-inside", "late-outside", "early-outside", "late-inside")},
                        t={k_: int(c_t.get(k_, 0)) for k_ in ("early-inside", "late-outside", "early-outside", "late-inside")})
        if g in GROUPS:
            ct.append([g, len(x)] + [cls_n[g][k_] for k_ in ("early-inside", "late-outside", "early-outside", "late-inside")] +
                      [" / ".join(str(cls_n[g]["t"][k_]) for k_ in ("early-inside", "late-outside", "early-outside", "late-inside"))])
    L += ["Corridor zeros on turn scenes, by where the ego reaches half the log's heading change along the road (ds) and its side of the log's arc at the event:", ""]
    L += md_table(["group", "events", "early-inside", "late-outside", "early-outside", "late-inside", "same by the sign of delta (EI / LO / EO / LI)"], ct) + [""]
    # straight scenes
    S_ = D[D.in_play & D.scene.isin(set(R[R.scls == "straight"].scene))]
    st_rows, cmd_share = [], {}
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1"):
        x = sel(S_, g)
        cmd_share[g] = B.mean(x.cmd.isin((LEFT, RIGHT)).astype(float), x.log)
    cmd_share["YS-base"] = B.diff(sel(S_, "YS").cmd.isin((LEFT, RIGHT)).astype(float), sel(S_, "YS").log,
                                  sel(S_, "base").cmd.isin((LEFT, RIGHT)).astype(float), sel(S_, "base").log)
    Zs = Z[Z.scls == "straight"]
    base_rate = {}
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1"):
        x = sel(Zs, g)
        n = len(x)
        cw = x.cmd_win.astype(bool) if n else pd.Series(dtype=bool)
        cl_ = sel(RR[RR.clean & (RR.scls == "straight")], g)
        hits = []
        for q in cl_.itertuples():
            c = Dk[(q.label, q.scene)].cmd.to_numpy()
            nz = np.isin(c, (LEFT, RIGHT))
            hits += [bool(nz[max(0, j - 3): j + 1].any()) for j in range(3, len(c))]
        base_rate[g] = float(np.mean(hits)) if hits else np.nan
        st_rows.append([g, f3(cmd_share[g]), n, int(cw.sum()), f"{cw.mean():.2f}" if n else "-", int(x.cmd_flip.astype(bool).sum()) if n else 0,
                        int(x.cmd_persist.astype(bool).sum()) if n else 0, int((x.cmd_same == True).sum()) if n else 0,  # noqa: E712
                        int((x.route_turns == True).sum()) if n else 0, f"{base_rate[g]:.2f}"])  # noqa: E712
    L += ["Straight scenes (< 15 deg): share of in-play decisions with a LEFT / RIGHT command, and the corridor zeros' command around the drift onset "
          "(k_d = first decision with |e| > 1 m after which the offset keeps its sign and grows; window k_d - 3 .. k_d):", ""]
    L += md_table(["group", "non-straight command share of decisions", "corridor zeros", "command in window", "share", "flipped in window", "non-straight from the start",
                   "command toward the drift side", "route turns within 60 m", "base rate (clean straight rollouts, any 4-decision window)"], st_rows)
    L += ["", f"YS - base, non-straight command share on straight scenes: {f3(cmd_share['YS-base'], f='+.3f')}.", ""]
    # H1-smooth
    sm = {}
    srows = []
    for g in tuple(labels) + ("base", "YS", "YS2G", "YNONE"):
        x = sel(T, g)
        pw, bw = B.ratio(x.w_e, x.w_log, x.log)
        pk, bk = B.ratio(x.kap_e, x.kap_log, x.log)
        sm[g] = dict(n=int(len(x)), w=dict(mean=pw, ci=B.ci(bw)), kap=dict(mean=pk, ci=B.ci(bk)), _bw=bw, _bk=bk)
        if g in labels:
            srows.append([g, len(x), f3(sm[g]["w"]), f3(sm[g]["kap"])])
    for k_ in ("w", "kap"):
        sm[f"YS-base_{k_}"] = dict(mean=sm["YS"][k_]["mean"] - sm["base"][k_]["mean"], ci=B.ci(sm["YS"][f"_b{k_[0]}"] - sm["base"][f"_b{k_[0]}"]))
    TP = D[D.scene.isin(set(T.scene)) & (D.log_dpsi4.abs() >= np.radians(30))]
    ptr = {g: float((np.sign(sel(TP, g).log_dpsi4) * sel(TP, g).plan_dpsi4).sum() / sel(TP, g).log_dpsi4.abs().sum()) for g in ("base", "YS", "YS2G", "YNONE")}
    L += ["Peak heading rate on turn scenes: shortfall = 1 - sum(ego peak) / sum(log peak) over the label's turn rollouts (yaw rate over 0.5 s; "
          "curvature = heading change over 1 s / max(arc, 2 m), which takes speed out):", ""]
    L += md_table(["label", "turn rollouts", "yaw-rate shortfall", "curvature shortfall"], srows)
    L += ["", f"YS - base: yaw rate {f3(sm['YS-base_w'], f='+.3f')}, curvature {f3(sm['YS-base_kap'], f='+.3f')}. Descriptive: the served plan's 4 s heading "
          "change over the log's from the same moment, on decisions where the log turns >= 30 deg in the next 4 s: " +
          ", ".join(f"{g} {v:.2f}" for g, v in ptr.items()) + ".", ""]
    # H1 lines
    za, zb = sel(Zs, "YS"), cls_n["YS"]
    na, nb = len(za), zb["n"]
    sa = float(za.cmd_win.astype(bool).mean()) if na else np.nan
    pa = None if na < 4 else bool(sa >= 0.5)
    pb = None if nb < 4 else bool(zb["early-inside"] >= zb["late-outside"])
    if pa and pb:
        h1c = "holds"
    elif pa and not pb:
        h1c = "partial: straight-road part only" + (" ((b) unreadable)" if pb is None else " ((b) fails)")
    elif pb and not pa:
        h1c = "partial: turn part only" + (" ((a) unreadable)" if pa is None else " ((a) fails)")
    else:
        h1c = "not supported"
    same_share = float((za.cmd_same == True).sum() / max(za.cmd_win.astype(bool).sum(), 1)) if na else np.nan  # noqa: E712
    cooc = bool(pa) and (same_share < 0.5 or base_rate["YS"] >= sa)
    smooth_all = all(sm[lab]["w"]["mean"] >= 0.20 and sm[lab]["w"]["ci"][0] > 0 for lab in labels)
    not_s = bool(sm["YS-base_w"]["ci"][0] <= 0 <= sm["YS-base_w"]["ci"][1] or sm["YS-base_w"]["mean"] <= 0)
    kap_all = all(sm[lab]["kap"]["mean"] >= 0.20 and sm[lab]["kap"]["ci"][0] > 0 for lab in labels)
    J["R1"] = dict(turn=r1, classes=cls_n, cmd_share=cmd_share, base_rate=base_rate, smooth={k_: {kk: v for kk, v in d_.items() if not kk.startswith("_")} if isinstance(d_, dict) else d_ for k_, d_ in sm.items()},
                   plan_turn_ratio=ptr, line=dict(a_n=na, a_share=sa, a_pass=pa, a_same_dir_share=same_share, b_n=nb, b_EI=zb["early-inside"], b_LO=zb["late-outside"], b_pass=pb,
                                                  H1_command=h1c, cooccurrence_only=cooc, H1_smooth_supported=smooth_all, curvature_all=kap_all, not_the_S_lesson=not_s))
    L += ["**Lines (YS pooled).** H1-command: (a) straight-scene corridor zeros with a non-straight command at or within 1.5 s before the drift onset: "
          f"{int(za.cmd_win.astype(bool).sum()) if na else 0} / {na} = {sa:.2f} (line >= 0.5{'' if na >= 4 else ', unreadable: < 4 events'}; same side as the drift "
          f"{same_share:.2f}; base rate {base_rate['YS']:.2f}); (b) turn-scene corridor zeros early-inside {zb['early-inside']} vs late-outside {zb['late-outside']} "
          f"(line EI >= LO{'' if nb >= 4 else ', unreadable: < 4 events'}) -> **H1-command {h1c}**" + ("; the command and the drift only co-occur (registered caveat)" if cooc else "") + ". "
          f"H1-smooth: yaw-rate shortfall >= 0.20 with CI above 0 in all 8 labels: **{'supported' if smooth_all else 'not supported'}**; "
          f"YS - base {f3(sm['YS-base_w'], f='+.3f')}: **{'not the S lesson' if not_s else 'larger under S'}**; curvature version all >= 0.20: {kap_all}.", ""]

    # ------------------------------------------------ R2
    P = D[D.in_play]
    r2, r2rows = {}, []
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1"):
        for c in ("lead", "no-lead", "follow"):
            x = sel(P[P.cond == c], g)
            r2[g, c] = dict(ad3=B.mean(x.ad3, x.log), sd3=B.mean(x.sd3, x.log), ayh3=float(x.yh3.abs().mean()), ayl3=float(x.yl3.abs().mean()),
                            ad2=float(x.d2.abs().mean()), ad4=float(x.d4.abs().mean()), defl=float((x.ad3 >= 0.5).mean()), n=int(len(x)))
            r2rows.append([g, c, len(x), f3(r2[g, c]["ad3"]), f3(r2[g, c]["sd3"], f="+.3f"), f"{r2[g, c]['ad2']:.3f} / {r2[g, c]['ad4']:.3f}",
                           f"{r2[g, c]['ayh3']:.3f}", f"{r2[g, c]['ayl3']:.3f}", f"{r2[g, c]['defl']:.3f}"])
    dd = {}
    for a_, b_ in (("YS", "base"), ("YS", "YNONE"), ("YS2G", "base"), ("YNONE", "base")):
        for c in ("lead", "no-lead"):
            xa, xb = sel(P[P.cond == c], a_), sel(P[P.cond == c], b_)
            dd[f"{a_}-{b_}", c, "abs"] = B.diff(xa.ad3, xa.log, xb.ad3, xb.log)
            dd[f"{a_}-{b_}", c, "onc"] = B.diff(xa.sd3, xa.log, xb.sd3, xb.log)
    did = {}
    for a_ in ("YS", "YS2G"):
        parts = [B.sums(sel(P[P.cond == c], g).ad3, sel(P[P.cond == c], g).log) for g in (a_, "base") for c in ("lead", "no-lead")]
        bt = (B.boot(*parts[0]) - B.boot(*parts[1])) - (B.boot(*parts[2]) - B.boot(*parts[3]))
        pt = (r2[a_, "lead"]["ad3"]["mean"] - r2[a_, "no-lead"]["ad3"]["mean"]) - (r2["base", "lead"]["ad3"]["mean"] - r2["base", "no-lead"]["ad3"]["mean"])
        did[a_] = dict(mean=pt, ci=B.ci(bt))
    L += ["## R2 cut-around the lead (H2)", "",
          "In-play decisions. lead = a vehicle 0-15 m ahead of the front bumper, centre |y| < 1.5 m, heading within 30 deg, slower than the ego by > 1 m/s "
          "(or stopped while the ego moves); no-lead = no vehicle in that box; follow = in the box but not slower. d(t) = the served plan's lateral offset from "
          "the logged path at t minus the ego's own offset (left +); s_onc d(3) = toward the oncoming side (+).", ""]
    L += md_table(["group", "condition", "decisions", "mean |d(3)| (m)", "mean s_onc d(3) (m)", "|d(2)| / |d(4)|", "|y_h(3)|", "|y_log(3)|", "share |d(3)| >= 0.5 m"], r2rows)
    L += ["", "Differences (the line uses YS - base, |d(3)|):", ""] + md_table(
        ["a - b", "lead: |d(3)|", "lead: s_onc d(3)", "no-lead: |d(3)|", "no-lead: s_onc d(3)"],
        [[k_, f3(dd[k_, "lead", "abs"], f="+.3f"), f3(dd[k_, "lead", "onc"], f="+.3f"), f3(dd[k_, "no-lead", "abs"], f="+.3f"), f3(dd[k_, "no-lead", "onc"], f="+.3f")]
         for k_ in ("YS-base", "YS-YNONE", "YS2G-base", "YNONE-base")])
    L += ["", f"Difference in differences of |d(3)| ((lead - no-lead) - same for base): YS {f3(did['YS'], f='+.3f')}, YS2G {f3(did['YS2G'], f='+.3f')}.", ""]
    gaps = {}
    for g in ("base", "YS", "YS2G", "YNONE"):
        x = sel(P[P.cond == "lead"], g)
        gaps[g] = [B.mean(x[(x.lead_gap > lo) & (x.lead_gap <= hi)].ad3, x[(x.lead_gap > lo) & (x.lead_gap <= hi)].log) for lo, hi in ((0, 5), (5, 10), (10, 15))]
    L += ["Mean |d(3)| under the lead condition by gap (0-5 / 5-10 / 10-15 m): " + "; ".join(
        f"{g} " + " / ".join(f"{r_['mean']:.3f} (n {r_['n']})" for r_ in v) for g, v in gaps.items()) + ".", ""]
    # collisions
    CO = RR[RR.get("coll", pd.Series(False, index=RR.index)).fillna(False).astype(bool)]
    crow, cshare = [], {}
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1"):
        x = sel(CO, g)
        c = x.c_cls.value_counts()
        h = x.hit_cls.value_counts()
        cshare[g] = dict(n=int(len(x)), cut=float(c.get("cut-around", 0) / len(x)) if len(x) else np.nan, counts={k_: int(v) for k_, v in c.items()},
                         hit={k_: int(v) for k_, v in h.items()}, hit_static=int((x.hit_static == True).sum()), hit_is_lead=int((x.hit_is_lead == True).sum()))  # noqa: E712
        crow.append([g, len(x)] + [int(c.get(k_, 0)) for k_ in ("cut-around", "straight-in", "swerve, no lead", "drift")] +
                    [f"{cshare[g]['cut']:.2f}" if len(x) else "-", ", ".join(f"{k_} {v}" for k_, v in h.items()), cshare[g]["hit_static"], cshare[g]["hit_is_lead"]])
    L += ["At-fault collisions (window: contact decision k_c and the 6 before = 3 s; lead = the lead condition at any of them; deflection = |d(3)| >= 0.5 m at any of them; "
          "object hit classified where it was 1 s before contact, in the ego frame):", ""]
    L += md_table(["group", "collisions", "cut-around", "straight-in", "swerve, no lead", "drift", "cut-around share", "object hit", "hit a static object", "hit = the lead"], crow) + [""]
    # rows
    RW["cls"] = [obj_class(r.x, r.y, r.dpsi) for r in RW.itertuples()]
    RW["lead_full"] = (RW.gap > 0) & (RW.gap <= LEAD_GAP) & (RW.y.abs() < LEAD_Y) & (RW.dpsi.abs() < 30) & ((RW.v_obj < RW.v_ego - 1) | ((RW.v_obj < 0.5) & (RW.v_ego > 1)))
    top = RW[RW.agent >= RW.agent.quantile(0.9)]
    rws = {}
    for nm, x in (("all active", RW), ("top decile", top)):
        c = x.cls.value_counts(normalize=True)
        rws[nm] = dict(n=int(len(x)), **{k_: float(c.get(k_, 0)) for k_ in ("in-lane lead", "adjacent", "crossing", "oncoming", "other")},
                       lead_full=float(x.lead_full.mean()), corridor=float(x.corridor.mean()), static=float((x.v_obj < 0.5).mean()))
    L += [f"Row side (pool `on1_L18Y`, {GR['rows']} rows; agent hinge active on {GR['active']}): the object that binds the hinge on the served plan, "
          "classified by its position at t0 in the row's own frame (descriptive, not in the line):", ""]
    L += md_table(["rows", "n", "in-lane same-direction lead", "adjacent (1.5-3 m)", "crossing (60-120 deg)", "oncoming (> 120 deg)", "other",
                   "full lead condition (closing)", "binding step inside the ego's corridor (0.3 m margin)", "object static (< 0.5 m/s)"],
                  [[nm, v["n"]] + [f"{v[k_]:.3f}" for k_ in ("in-lane lead", "adjacent", "crossing", "oncoming", "other", "lead_full", "corridor", "static")] for nm, v in rws.items()]) + [""]
    d_lead, d_nol = dd["YS-base", "lead", "abs"], dd["YS-base", "no-lead", "abs"]
    i_ok = bool(d_lead["mean"] >= 0.20 and d_lead["ci"][0] > 0 and d_nol["mean"] < 0.10)
    ii_read = cshare["YS"]["n"] >= 4
    ii_ok = bool(ii_read and cshare["YS"]["cut"] >= 0.5 and cshare["YS"]["cut"] > cshare["base"]["cut"])
    refuted = bool(d_lead["ci"][0] <= 0 <= d_lead["ci"][1] and (cshare["YS"]["cut"] < 0.25 if ii_read else True))
    h2 = "holds" if (i_ok or ii_ok) else "refuted" if refuted else "not established"
    J["R2"] = dict(table={f"{g}|{c}": v for (g, c), v in r2.items()}, diffs={f"{k_[0]}|{k_[1]}|{k_[2]}": v for k_, v in dd.items()}, did=did, gaps=gaps,
                   collisions=cshare, rows=rws, line=dict(i=i_ok, ii_readable=ii_read, ii=ii_ok, refuted=refuted, H2=h2))
    L += [f"**Lines (YS pooled).** (i) lead: YS - base |d(3)| {f3(d_lead, f='+.3f')} m (line >= +0.20 with CI above 0) and no-lead {f3(d_nol, f='+.3f')} m (line < +0.10): "
          f"{'met' if i_ok else 'not met'}; (ii) cut-around share of YS's at-fault collisions {cshare['YS']['cut']:.2f} of {cshare['YS']['n']} vs base {cshare['base']['cut']:.2f} of "
          f"{cshare['base']['n']} (line >= 0.5 and above base): {'met' if ii_ok else 'not met'}. Refutation line (lead CI includes 0 and cut-around < 1/4): "
          f"{'met' if refuted else 'not met'} -> **H2 {h2}**.", ""]

    # ------------------------------------------------ R3
    r3, r3rows = {}, []
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1"):
        x = sel(P, g)
        r3[g] = dict(rear=B.mean(x.onc_rear, x.log), front=B.mean(x.onc_front, x.log), wrong_lane=B.mean(x.wl, x.log),
                     none=float((x.lane_rear == "none").mean()), cross=float((x.lane_rear == "cross").mean()))
        r3rows.append([g, len(x), f3(r3[g]["rear"], f=".4f"), f3(r3[g]["front"], f=".4f"), f"{r3[g]['cross']:.4f}", f"{r3[g]['none']:.4f}", f3(r3[g]["wrong_lane"], f=".4f")])
    for a_ in ("YS", "YS2G", "YNONE"):
        for col in ("onc_rear", "onc_front"):
            xa, xb = sel(P, a_), sel(P, "base")
            r3[f"{a_}-base", col] = B.diff(xa[col], xa.log, xb[col], xb.log)
    L += ["## R3 lane direction (H3)", "",
          "Lane class of the rear axle (primary) and of the front bumper centre against the scene's lanes (evaluator's lane polygons and centre-line direction): "
          "same if any containing lane is within 60 deg of the ego heading, else oncoming if any is beyond 120 deg, else crossing; none = in no lane. "
          "Shares of in-play decisions; `wrong_lane` = the evaluator's own flag (unwrapped angle, known false positives).", ""]
    L += md_table(["group", "decisions", "rear axle in an oncoming lane", "front in an oncoming lane", "rear crossing", "rear in no lane", "evaluator wrong_lane"], r3rows)
    L += ["", "Differences: " + "; ".join(f"{a_} - base rear {f3(r3[f'{a_}-base', 'onc_rear'], f='+.4f')}, front {f3(r3[f'{a_}-base', 'onc_front'], f='+.4f')}"
                                         for a_ in ("YS", "YS2G", "YNONE")) + ".", ""]
    zr = []
    zon = {}
    ZZ = RR[RR.zero & RR.type.isin(("corridor", "offroad", "collision"))]
    for g in ("base", "YS", "YS2G", "YNONE", "YR-s1"):
        x = sel(ZZ, g)
        row = [g]
        for ty in ("corridor", "collision", "offroad"):
            y = x[x.type == ty]
            row.append(f"{int(y.onc_pre.sum())} / {len(y)}")
        y = x[x.type.isin(("corridor", "collision"))]
        zon[g] = dict(n=int(len(y)), onc=int(y.onc_pre.sum()), share=float(y.onc_pre.mean()) if len(y) else np.nan)
        zr.append(row + [f"{zon[g]['share']:.2f}" if len(y) else "-"])
    L += ["Zeros with the rear axle in an oncoming lane at any decision in the 2 s before the event (k_ev - 4 .. k_ev):", ""]
    L += md_table(["group", "corridor", "collision", "offroad", "corridor + collision share"], zr) + [""]
    ld = P[(P.cond == "lead") & (P.ad3 >= 0.5)]
    ldp = {g: dict(n=int(len(sel(ld, g))), onc=float((sel(ld, g).lane_plan == "onc").mean()) if len(sel(ld, g)) else np.nan) for g in ("base", "YS", "YS2G", "YNONE", "YR-s1")}
    L += ["Lead-conditioned deflections (lead and |d(3)| >= 0.5 m) whose plan end point (4 s) lies in an oncoming lane: " +
          "; ".join(f"{g} {v['onc']:.2f} of {v['n']}" for g, v in ldp.items()) + ".", ""]
    n3 = zon["YS"]["n"]
    arm3 = None if n3 < 8 else bool(zon["YS"]["share"] >= 0.25)
    worse = bool(r3["YS-base", "onc_rear"]["ci"][0] > 0)
    J["R3"] = dict(table=r3 | {}, zeros=zon, lead_defl_plan_onc=ldp, line=dict(n=n3, share=zon["YS"]["share"], arm=arm3, worse=worse))
    J["R3"]["table"] = {(k_ if isinstance(k_, str) else "|".join(k_)): v for k_, v in r3.items()}
    L += [f"**Lines (YS pooled).** Corridor + collision zeros with oncoming occupancy in the 2 s before the event: {zon['YS']['onc']} / {n3} = {zon['YS']['share']:.2f} "
          f"(line >= 0.25{'' if n3 >= 8 else ', unreadable: < 8 events'}) -> **H3 {'gets an arm' if arm3 else 'does not get an arm' if arm3 is False else 'unreadable'}**; "
          f"oncoming decision share YS - base {f3(r3['YS-base', 'onc_rear'], f='+.4f')} (line: CI above 0) -> **the on-policy lesson {'made it worse' if worse else 'did not make it worse'}**.", ""]

    # ------------------------------------------------ the 43 zeros of YS-L18-s1
    E = RR[(RR.label == "YS-L18-s1") & RR.zero].sort_values(["type", "k_ev", "scene"]).copy()
    E["sid"] = E.scene.str.rsplit("-", n=1).str[1].str[:8]

    def flags(r):
        f = []
        if r.scls == "turn":
            if r.cls == "early-inside":
                f.append("H1-cmd")
            if r.cls == "late-outside":
                f.append("H1-wide")
        elif r.cmd_win == True:  # noqa: E712  (object columns: True / False / NaN)
            f.append("H1-cmd")
        if r.lead_win == True and r.defl_win == True:  # noqa: E712
            f.append("H2")
        if r.onc_pre == True:  # noqa: E712
            f.append("H3")
        if r.type == "offroad" and r.k_ev == 0:
            f.append("start")
        return ", ".join(f) or "none (drift)"
    E["fits"] = [flags(r) for r in E.itertuples()]
    erows = []
    for r in E.itertuples():
        sc_ = f"{r.scls} {'L' if r.dpsi_log > 0 else 'R'} {r.dpsi_log:+.0f}" if r.scls != "straight" else f"straight {r.dpsi_log:+.0f}"
        if r.scls == "turn":
            r1s = f"ds {r.ds:+.1f} m, delta {r.delta:+.2f} s, {r.cls}" if np.isfinite(r.ds) else f"never half-turned, {r.cls}"
        else:
            r1s = (f"onset k{int(r.k_d)}, cmd {'yes' if r.cmd_win else 'no'}" + (" (flip)" if r.cmd_flip else "") + (" (from start)" if r.cmd_persist else "") +
                   ("" if r.cmd_same is None or (isinstance(r.cmd_same, float) and np.isnan(r.cmd_same)) else (", toward drift" if r.cmd_same else ", away from drift")))
        r2s = ("lead" if r.lead_win else "no lead") + (", deflect" if r.defl_win else "") + (f"; {r.c_cls}, hit {r.hit_cls}" if getattr(r, "coll", False) == True else "")  # noqa: E712
        erows.append([r.sid, r.type, int(r.k_ev), sc_, f"{r.e_ev:+.1f}", r1s, r2s, ("oncoming" if r.onc_pre else "-") + f" / {r.lane_ev}", r.fits])
    E.to_csv(out / "events_YS-L18-s1.csv", index=False)
    fit_counts = {k_: int(sum(k_ in f for f in E.fits)) for k_ in ("H1-cmd", "H1-wide", "H2", "H3", "start", "none")}
    J["events_YS_s1"] = dict(n=int(len(E)), fits=fit_counts)
    L += [f"## YS-L18-s1: the {len(E)} zeros, one line each", "",
          "Columns: scene token (first 8 characters), type, event decision, scene class (log heading change over the rollout, deg), ego offset from the logged "
          "path at the event (m, left +), R1 reading, R2 reading (3 s before the event; collisions: class and object hit), R3 (oncoming in the 2 s before / "
          "rear-axle lane class at the event), and the rule-based fit (H1-cmd: straight / bend with a non-straight command in the onset window, or turn early-inside; "
          "H1-wide: turn late-outside; H2: lead and deflection; H3: oncoming; start: offroad from decision 0).", ""]
    L += md_table(["scene", "type", "k_ev", "scene class", "e at event", "R1", "R2", "R3", "fits"], erows)
    L += ["", "Fit counts: " + ", ".join(f"{k_} {v}" for k_, v in fit_counts.items()) + " (an event can fit several).", ""]

    # ------------------------------------------------ exploratory (after the registered read; not in the verdicts)
    X_ = {}
    L17 = D.groupby("scene").s_log_t.max()                                          # the log's own station at t_17
    ZE = RR[RR.zero & RR.type.isin(("corridor", "offroad", "collision"))].copy()
    ZE["past_end"] = [bool(Dk[(r.label, r.scene)].pipe(lambda x: x[x.k <= r.k_ev]).s_e.max() > L17[r.scene]) for r in ZE.itertuples()]
    pe_rows = []
    for g in ("base", "YS", "YS2G", "YNONE"):
        x = sel(ZE, g)
        X_[f"past_end|{g}"] = {f"{sc_}|{ty}": [int(y.past_end.sum()), int(len(y))] for (sc_, ty), y in x.groupby(["scls", "type"])}
        y = x[(x.scls == "straight") & (x.type == "corridor")]
        z = x[(x.scls == "turn") & (x.type == "corridor")]
        pe_rows.append([g, f"{int(y.past_end.sum())} / {len(y)}", f"{int(z.past_end.sum())} / {len(z)}", f"{int(x.past_end.sum())} / {len(x)}"])
    # lead decisions on a straight road ahead (the log turns < 10 deg in the next 4 s): the plan's own lateral motion y_h(3)
    PS = P[(P.cond.isin(("lead", "no-lead"))) & (P.log_dpsi4.abs() < np.radians(10))].copy()
    PS["ayh3"], PS["syh3"] = PS.yh3.abs(), PS.s_onc * PS.yh3
    yh_rows = []
    for c in ("lead", "no-lead"):
        xa, xb = sel(PS[PS.cond == c], "YS"), sel(PS[PS.cond == c], "base")
        da, ds_ = B.diff(xa.ayh3, xa.log, xb.ayh3, xb.log), B.diff(xa.syh3, xa.log, xb.syh3, xb.log)
        X_[f"yh3_straight|{c}"] = dict(abs=da, onc=ds_, YS=B.mean(xa.ayh3, xa.log), base=B.mean(xb.ayh3, xb.log))
        yh_rows.append([c, f"{len(xa)} / {len(xb)}", f3(X_[f'yh3_straight|{c}']['YS']), f3(X_[f'yh3_straight|{c}']['base']), f3(da, f="+.3f"), f3(ds_, f="+.3f")])
    # the registered cut-around collisions of YS: does the plan itself move sideways (max |y_h(3)| in the window), and is the log bending?
    cu = []
    for r in CO[CO.label.isin(GROUPS["YS"]) & (CO.c_cls == "cut-around")].itertuples():
        x = Dk[(r.label, r.scene)]
        w = x[(x.k >= r.kc - 6) & (x.k <= r.kc)]
        cu.append([r.label, r.scene.rsplit("-", 1)[1][:8], r.scls, int(r.kc), f"{w.ad3.max():.2f}", f"{w.yh3.abs().max():.2f}",
                   f"{(w.yl3 - w.yh3).abs().max():.2f}", r.hit_cls])
    # rows: what "other" is
    oth = RW[RW.cls == "other"]
    X_["rows_other"] = dict(n=int(len(oth)), lat_ge3=float((oth.y.abs() >= 3).mean()), behind=float((oth.x <= 1.461).mean()),
                            psi30_60=float(((oth.dpsi.abs() >= 30) & (oth.dpsi.abs() <= 60)).mean()))
    # oncoming decisions inside junction overlaps; turn direction of the early-inside / late-outside turn zeros
    jr = {g: float(sel(P[P.onc_rear > 0], g).junc_rear.mean()) for g in ("base", "YS", "YS2G", "YNONE")}
    side = lambda x: x.assign(dirn=np.where(np.sign(x.dpsi_log) * x.s_onc > 0, "across", "kerb"))       # noqa: E731  across = toward the oncoming side
    tz, ta, tr_ = side(Z[Z.scls == "turn"]), side(ZE[ZE.scls == "turn"]), side(T)
    td = {g: {f"{w_}|{c_}|{d_}": int(((sel(x, g).cls == c_) & (sel(x, g).dirn == d_)).sum()) for w_, x in (("corridor", tz), ("all", ta))
              for c_ in ("early-inside", "late-outside") for d_ in ("across", "kerb")} |
          {f"ds|{d_}": [float(np.quantile(sel(tr_, g)[sel(tr_, g).dirn == d_].ds, q, method="inverted_cdf")) for q in (0.25, 0.5, 0.75)] for d_ in ("across", "kerb")}
          for g in ("base", "YS", "YS2G", "YNONE")}
    zo = {}
    for g in ("YS", "YS2G", "YNONE"):
        xa, xb = sel(ZZ[ZZ.type.isin(("corridor", "collision"))], g), sel(ZZ[ZZ.type.isin(("corridor", "collision"))], "base")
        zo[g] = B.diff(xa.onc_pre.astype(float), xa.log, xb.onc_pre.astype(float), xb.log)
    X_.update(oncoming_in_junction=jr, turn_dir=td, zeros_onc_diff=zo, cut_around_YS=cu)
    J["exploratory"] = X_
    L += ["## Exploratory (after the registered read; not in the verdicts)", "",
          "1. Events after the ego has passed the end of the human's logged path (its station beyond the log's own station at t_17); the evaluator then "
          "measures the corridor against the straight extension of the log's last heading:", ""]
    L += md_table(["group", "straight-scene corridor zeros past the log's end", "turn-scene corridor zeros", "all zeros"], pe_rows) + [""]
    L += ["2. The plan's own lateral motion y_h(3) (ego heading frame) on decisions where the log turns < 10 deg in the next 4 s (curvature small), "
          "lead vs no-lead: d(3) is measured against the logged path, which itself moves sideways when the human passes a slow lead or bears off "
          "(8051c60f: the plan goes straight, |y_h(3)| <= 0.1 m, while the log bears right, d(3) 1.1-2.5 m):", ""]
    L += md_table(["condition", "decisions YS / base", "YS mean |y_h(3)|", "base", "YS - base |y_h(3)|", "YS - base s_onc y_h(3)"], yh_rows) + [""]
    L += ["3. YS's registered cut-around collisions: max |d(3)|, max |y_h(3)| (the plan's own sideways motion; large on turns by curvature) and max "
          "|y_log(3) - y_h(3)| (how far the logged path is from the ego's heading line at the plan's 3 s point) in the window:", ""]
    L += md_table(["label", "scene", "scene class", "k_c", "max |d(3)|", "max |y_h(3)|", "max |y_log - y_h|", "object hit"], cu) + [""]
    L += [f"4. Row side, the {X_['rows_other']['n']} binding objects classed `other`: {X_['rows_other']['lat_ge3']:.2f} at |y| >= 3 m at t0 (reached by a "
          f"curving plan), {X_['rows_other']['behind']:.2f} not ahead of the ego box centre, {X_['rows_other']['psi30_60']:.2f} at 30-60 deg.",
          "5. Oncoming decisions inside a junction overlap (two containing lanes > 30 deg apart): " + ", ".join(f"{g} {v:.2f}" for g, v in jr.items()) + ".",
          "6. Turn zeros by turn side (across = turning across the oncoming lanes: left under right-hand traffic, right in Singapore, 113 of the 452 "
          "scenes; kerb = the other way); early-inside / late-outside counts, corridor zeros and all zero types on turn scenes, and ds over all turn rollouts:", ""]
    L += md_table(["group", "corridor, across: EI / LO", "corridor, kerb: EI / LO", "all zeros, across: EI / LO", "all zeros, kerb: EI / LO",
                   "ds median [IQR], across (m)", "ds, kerb (m)"],
                  [[g, f"{v['corridor|early-inside|across']} / {v['corridor|late-outside|across']}", f"{v['corridor|early-inside|kerb']} / {v['corridor|late-outside|kerb']}",
                    f"{v['all|early-inside|across']} / {v['all|late-outside|across']}", f"{v['all|early-inside|kerb']} / {v['all|late-outside|kerb']}",
                    "{1:+.2f} [{0:+.2f}, {2:+.2f}]".format(*v["ds|across"]), "{1:+.2f} [{0:+.2f}, {2:+.2f}]".format(*v["ds|kerb"])] for g, v in td.items()])
    L += ["",
          "7. Share of corridor + collision zeros with oncoming occupancy, difference to base (event-level, log-clustered): " +
          "; ".join(f"{g} - base {f3(v, f='+.2f')}" for g, v in zo.items()) + ".", ""]

    # ------------------------------------------------ verdicts and arms
    arm_a = h2 == "holds"
    arm_b = bool(arm3)
    arm_c = "yes" if h1c == "holds" else ("straight-road half only" if h1c.startswith("partial: straight") else "no")
    J["arms"] = dict(A=arm_a, B=arm_b, C=arm_c)
    L += ["## Verdicts by the registered lines", ""] + md_table(["hypothesis", "verdict", "arm (prereg section 8)"], [
        ["H1-command", h1c + (" (co-occurrence caveat)" if cooc else ""), f"C (route waypoints into the adapter): {arm_c}"],
        ["H1-smooth", ("supported" if smooth_all else "not supported") + f"; {'not the S lesson' if not_s else 'larger under S'}", "-"],
        ["H2", h2, f"A (in-lane same-direction objects out of the lateral agent hinge): {'yes' if arm_a else 'no'}"],
        ["H3", ("gets an arm" if arm3 else "no arm" if arm3 is False else "unreadable") + f"; lesson {'made it worse' if worse else 'did not make it worse'}", f"B (direction-aware road raster): {'yes' if arm_b else 'no'}"]]) + [""]
    (out / "zm_report.md").write_text("\n".join(L) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {str(k_): clean(v) for k_, v in o.items() if not str(k_).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o
    (out / "zm.json").write_text(json.dumps(clean(J), indent=1))
    picks = bev_picks(RR, D)
    (out / "picks.json").write_text(json.dumps(picks, indent=1))
    figures(a, D, RR, P, T, gaps, r2, figs, picks)
    print("\n".join(L))


def bev_picks(RR, D):
    """One example decision per class for the BEV strip, deterministic: YS-L18-s1 before YS-L18-s0, scenes with a published clip
    (experiments/onpol_long/figs/zeros/) first, then by scene id; each scene used once."""
    picks, used = [], set()
    clips = {f.stem for f in (REPO / "experiments/onpol_long/figs/zeros").glob("*.gif")}
    Y = RR[RR.label.isin(("YS-L18-s1", "YS-L18-s0")) & RR.zero].copy()
    Y["noclip"] = ~Y.scene.isin(clips)
    Y = Y.sort_values(["label", "noclip", "scene"], ascending=[False, True, True])
    Dk = {k_: x.sort_values("k") for k_, x in D[D.label.isin(("YS-L18-s1", "YS-L18-s0"))].groupby(["label", "scene"])}

    def first(m):
        m = m & ~Y.scene.isin(used)
        return Y[m].iloc[0] if m.any() else None

    def add(cls, r, k):
        used.add(r.scene)
        picks.append(dict(cls=cls, label=r.label, scene=r.scene, k=int(max(k, 0)), type=r.type, k_ev=int(r.k_ev)))
    for cls, m in (("early-inside", (Y.scls == "turn") & (Y.cls == "early-inside") & (Y.type == "corridor")),
                   ("late-outside", (Y.scls == "turn") & (Y.cls == "late-outside") & (Y.type == "corridor"))):
        r = first(m)
        if r is not None:
            add(cls, r, r.k_ev - 2)
    r = first((Y.scls != "turn") & (Y.cmd_win == True) & (Y.type == "corridor"))  # noqa: E712
    if r is not None:
        x = Dk[(r.label, r.scene)]
        w = x[(x.k >= r.k_d - 3) & (x.k <= r.k_d) & x.cmd.isin((LEFT, RIGHT))]
        add("command on a straight road", r, int(w.k.iloc[0]) if len(w) else r.k_d)
    r = first((Y.lead_win == True) & (Y.defl_win == True))  # noqa: E712
    if r is not None:
        x = Dk[(r.label, r.scene)]
        kw = int(r.k_c) if r.type == "collision" and r.k_c >= 0 else int(r.k_ev)
        w = x[(x.k >= kw - 6) & (x.k <= kw) & x.lead]
        add("lead and deflection", r, int(w.k.iloc[-1]) if len(w) else kw)
    r = first(Y.onc_pre == True)  # noqa: E712
    if r is not None:
        x = Dk[(r.label, r.scene)]
        w = x[(x.k >= r.k_ev - 4) & (x.k <= r.k_ev) & (x.lane_rear == "onc")]
        add("oncoming lane", r, int(w.k.iloc[0]) if len(w) else r.k_ev)
    r = first(Y.type.isin(("corridor", "collision")) & ~(Y.cls.isin(("early-inside", "late-outside")) | (Y.cmd_win == True) |  # noqa: E712
                                                        ((Y.lead_win == True) & (Y.defl_win == True)) | (Y.onc_pre == True)))  # noqa: E712
    if r is not None:
        add("none of the three (drift)", r, r.k_ev - 2)
    return picks


def figures(a, D, RR, P, T, gaps, r2, figs, picks):
    import pickle
    sys.path.insert(0, str(REPO / "research"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    show = ("base", "YS", "YS2G", "YNONE")
    sel = lambda df, g: df[df.label.isin(GROUPS[g])]                              # noqa: E731
    # 1. delta and ds on turn scenes
    fig, ax = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.1))
    for j, (col, bins, cap, lab) in enumerate((("delta", np.arange(-4, 4.01, 0.5), 4.25, "t_ego - t_log at half the log's turn (s)"),
                                               ("ds", np.arange(-20, 20.1, 2.5), 21.25, "station difference at half the turn (m)"))):
        for g in show:
            v = sel(T, g)[col].to_numpy(float)
            v = np.where(np.isfinite(v), np.clip(v, bins[0], bins[-1] - 1e-6), cap)
            h, _ = np.histogram(v, np.r_[bins, cap + (bins[1] - bins[0]) / 2])
            ax[j].step(np.r_[bins, cap + (bins[1] - bins[0]) / 2][:-1], h / max(len(v), 1), where="post", color=COLOR[g], label=g, lw=1.0)
        ax[j].axvline(0, color="#999999", lw=0.5)
        ax[j].set_xlabel(lab)
        ax[j].set_ylabel("share of turn rollouts")
        ax[j].text(cap, ax[j].get_ylim()[1] * 0.92, "never", ha="center", fontsize=7)
        ps.panel(ax[j], "(a)" if j == 0 else "(b)")
    ax[0].legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, figs / "delta_hist")
    plt.close(fig)
    # 2. |d(3)| vs gap under the lead condition, no-lead level dashed
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.3))
    xs = np.array([2.5, 7.5, 12.5])
    for i, g in enumerate(show):
        m = np.array([r_["mean"] for r_ in gaps[g]])
        lo, hi = np.array([r_["ci"][0] for r_ in gaps[g]]), np.array([r_["ci"][1] for r_ in gaps[g]])
        ax.errorbar(xs + (i - 1.5) * 0.35, m, yerr=[m - lo, hi - m], color=COLOR[g], marker="o", ms=3, lw=1, capsize=1.5, label=g)
        ax.axhline(r2[g, "no-lead"]["ad3"]["mean"], color=COLOR[g], ls="--", lw=0.7)
    ax.set_xticks(xs, ["0-5", "5-10", "10-15"])
    ax.set_xlabel("gap to the slower lead (m)")
    ax.set_ylabel("mean |d(3 s)| (m)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    ps.save(fig, figs / "lead_offset_gap")
    plt.close(fig)
    # 3. oncoming share by decision index
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.1))
    for g in show:
        x = sel(P, g).groupby("k").onc_rear.mean()
        ax.plot(x.index, 100 * x.values, color=COLOR[g], label=g, marker="o", ms=2)
    ax.set_xlabel("decision index k (2 Hz)")
    ax.set_ylabel("rear axle in an oncoming lane (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, figs / "oncoming_by_k")
    plt.close(fig)
    # 4. BEV strip
    if a.bev and Path(a.bev).exists():
        bv = pickle.load(open(a.bev, "rb"))
        ex = [bv[f"{p['label']}|{p['scene']}|{p['k']}"] for p in picks if f"{p['label']}|{p['scene']}|{p['k']}" in bv]
        if ex:
            fig, axs = plt.subplots(1, len(ex), figsize=(ps.DOUBLE_COLUMN_IN, 3.0))
            axs = np.atleast_1d(axs)
            fc = lambda d: "#F7C9B0" if abs(d) > 120 else "#F3E7A6" if abs(d) > 60 else "#E4E4E4"   # noqa: E731
            for j, (ax, b) in enumerate(zip(axs, ex)):
                for ln in b["lanes"]:
                    for P_ in ln["ext"]:
                        ax.fill(-P_[:, 1], P_[:, 0], color=fc(ln["dpsi"]), lw=0.3, ec="#BBBBBB", zorder=0)
                    ax.plot(-ln["center"][:, 1], ln["center"][:, 0], color="#AAAAAA", lw=0.3, zorder=1)
                ob = b["objs"]
                for i in range(len(ob["id"])):
                    Cc = corners(np.array([ob["x"][i]]), np.array([ob["y"][i]]), np.array([ob["dpsi"][i]]), np.array([ob["L"][i]]), np.array([ob["W"][i]]))[0]
                    ax.fill(-Cc[:, 1], Cc[:, 0], color="#555555" if ob["v"][i] < 0.5 else "#999999", zorder=2)
                r_, f_, hw_ = b["ego"]
                ax.fill([-hw_, hw_, hw_, -hw_], [r_, r_, f_, f_], color=COLOR["YS"], zorder=3)
                ax.plot(-b["log"][:, 1], b["log"][:, 0], color="#000000", ls="--", lw=0.7, zorder=4)
                pl = np.r_[[[0, 0, 0]], b["plan"]]
                ax.plot(-pl[:, 1], pl[:, 0], color=ps.PALETTE["vermillion"], lw=1.1, marker="o", ms=1.5, zorder=5)
                ax.set_xlim(-14, 14)
                ax.set_ylim(-8, 42)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                p = b["pick"]
                ax.set_xlabel(f"{p['cls']}\n{p['scene'].rsplit('-', 1)[1][:8]} k{p['k']} ({p['type']} k{p['k_ev']}), cmd {'LSRU'[b['cmd']]}", fontsize=6.5)
                ps.panel(ax, f"({'abcdefgh'[j]})")
            fig.tight_layout()
            ps.save(fig, figs / "bev_examples")
            plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("collect")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=0, help="0 = min(64, jevdrive.common.n_cpus())")
    p.add_argument("--limit", type=int, default=0, help="first N scenes only (code check)")
    p = sp.add_parser("rows")
    p.add_argument("--pool", default="on1_L18Y")
    p.add_argument("--rows-dir", default="L18Y")
    p = sp.add_parser("bev")
    p.add_argument("--collect", required=True)
    p.add_argument("--picks", required=True)
    p = sp.add_parser("report")
    p.add_argument("--collect", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--bev")
    p.add_argument("--out", default=str(REPO / "experiments/zeros_mech/results"))
    p.add_argument("--figs", default=str(REPO / "experiments/zeros_mech/figs"))
    a = ap.parse_args()
    dict(collect=collect, rows=rows, bev=bev, report=lambda a: report(a))[a.cmd](a)
