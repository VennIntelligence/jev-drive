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

import numpy as np

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
    # AlpaSim's route, extended by every decision as the rollouts extended it (ao_e1)
    g = RouteGenerator.create(Pg, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=START_M)
    for d in labs.values():
        for r in d["recs"]:
            x, y, yaw = r["anchor"]
            g.generate_route(int(r["t0"]), Pose(np.array([x, y, zt(r["t0"])]), np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])))
    Rr = dedupe(np.asarray(g.route_polyline_in_local.points, float)[:, :2])
    SR = stations(Rr)
    objs = objects(ds)
    tk = T0 + DT * np.arange(NK)
    states = [obj_state(objs, t) for t in tk]
    lanes = Lanes(ds.map)
    lane_log = [lanes.cls(np.interp(t, Tg, Pg[:, 0]), np.interp(t, Tg, Pg[:, 1]), zt(t), np.interp(t, Tg, Yg)) for t in tk]
    dec, roll = [], []
    gate = dict(act_pairs=0, act_ok=0, act_max=0.0, act_missing=0, ego_pairs=0, ego_ok=0, ego_max=0.0, rollouts_asl=0)
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
            route60 = np.degrees(wrap(np.arctan2(*(ch[3] - ch[2])[::-1]) - np.arctan2(*(ch[1] - ch[0])[::-1])))
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
        units = [(s, cfg, v) for s, v in sorted(jobs.items())][: a.limit or None]
        workers = a.workers or min(64, n_cpus())
        run.info(f"{len(units)} scenes x {len(runs)} labels, {workers} workers, scene config {cfg}")
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
                 ego_dims=vals[0]["gate"]["ego_dims"], scenes=len(units), scene_errors=len(err))
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
def report(a):
    raise NotImplementedError("report: written after the collect code check")


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
