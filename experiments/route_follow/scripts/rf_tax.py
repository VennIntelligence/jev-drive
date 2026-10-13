#!/usr/bin/env python3
"""ZERO-TAXONOMY (research/zero-taxonomy.md): per-rollout facts of stored AlpaSim reads (nuPlan held 460 at 18 decisions, PAI 281), for a
taxonomy of every zero and low score. CPU only; run dirs, maps and logs are read, never written. Self-contained (copied to the boxes as is).

  extract  (box: AlpaSim venv; Tokyo: inside alpasim-base:0.89.0 with /mnt/nre-data and /repo/src mounted)
      rf_tax.py extract --runs <run dir> [<run dir> ...] --out <pkl> [--workers N] [--limit N] [--asl-actors]
      Per rollout: summary row, the evaluator's metric series, the controller trace (ego, 10 Hz), the logged rig path, AlpaSim's route rebuilt
      at the ego's 0.5 s poses (RouteGenerator, MAP), traffic at 0.5 s (nuPlan: the logged traffic the simulator replays, physics NONE,
      zeros_mech gate G-act; PAI --asl-actors: the rollout.asl actor poses and labels), per 0.5 s step the ego's lane class / junction flag /
      drivable margin, and for rollouts that need a figure (score < 0.9) the road polygons and lane centre lines around the ego path, and
      the served plans (nuPlan drive.jsonl; PAI asl driver_return).
  waitlines (Tokyo, same image)   rf_tax.py waitlines --runs <run dir> ... --out <pkl>     PAI wait lines (STOP / YIELD) near the log path
  report   (Mac, .venv)   rf_tax.py report --set NAME=<pkl>[,<ref pkl>] ... --out DIR
      classification of every rollout (one primary label per zero / low score, secondary labels), pedestrian and stopped-obstacle
      encounters, per-scene CSVs <DIR>/<name>.csv and <DIR>/summary.json; module-level rules below (RULES). A third pkl (waitlines)
      adds the stop-line read: NAME=<pkl>,<ref pkl>,<waitlines pkl>.
  figs     (Mac, .venv)   rf_tax.py figs --set NAME=<pkl> ... --picks picks.json --figs DIR      BEV at the event (+ front frame) per pick

Frames: rollout local frame; ego = the controller's state (rear axle), x forward, y left. Times in us on the evaluator's clock.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

LANE_R, JUNC, SAME, ONC = 10.0, np.radians(30.0), np.radians(60.0), np.radians(120.0)
FIG_R = 60.0
_L = {}


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def loader(cfg):
    if cfg not in _L:
        from alpasim_runtime.config import UserSimulatorConfig
        from alpasim_runtime.scene_loader import build_scene_loader
        from omegaconf import OmegaConf
        _L[cfg] = build_scene_loader(OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(cfg)))
    return _L[cfg]


def yawq(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def ctrl(run, rid):
    """Controller trace -> dict t (us), x, y, yaw, vx, vy, wz (10 Hz)."""
    f = Path(run) / f"controller/alpasim_controller_{rid}.csv"
    if not f.exists():
        return None
    import csv
    rows = list(csv.reader(open(f)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    if not len(A):
        return None
    c = lambda k: A[:, h[k]]  # noqa: E731
    return dict(t=c("timestamp_us"), x=c("x"), y=c("y"), yaw=np.unwrap(yawq(c("qx"), c("qy"), c("qz"), c("qw"))), vx=c("vx"), vy=c("vy"),
                wz=c("wz"))


def series(run, scene, rid):
    import polars as pl
    f = glob.glob(f"{run}/rollouts/{scene}/{rid}/metrics.parquet")
    if not f:
        return None, None
    m = pl.read_parquet(f[0]).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    ser = {n: np.array(sorted(zip(x["timestamps_us"].to_list(), [np.nan if v is None else v for v in x["values"].to_list()])), float).reshape(-1, 2)
           for (n,), x in m.group_by("name")}
    asl = glob.glob(str(Path(f[0]).parent / "rollout.asl"))
    mp4 = glob.glob(str(Path(f[0]).parent / "*camera_front*.mp4"))
    return ser, dict(asl=asl[0] if asl else None, mp4=mp4[0] if mp4 else None)


def read_asl(f, want_actors):
    import asyncio
    from alpasim_utils.logs import async_read_pb_log

    def p3(p):
        q = p.quat
        return [p.vec.x, p.vec.y, float(yawq(q.x, q.y, q.z, q.w))]

    async def go():
        o = dict(size={}, actors={}, plans=[])
        async for e in async_read_pb_log(f):
            k = e.WhichOneof("log_entry")
            if k == "rollout_metadata":
                o["size"] = {x.actor_id: (x.aabb.size_x, x.aabb.size_y, x.actor_label) for x in e.rollout_metadata.actor_definitions.actor_aabb}
            elif k == "actor_poses" and want_actors:
                m = e.actor_poses
                for x in m.actor_poses:
                    o["actors"].setdefault(x.actor_id, []).append([int(m.timestamp_us), *p3(x.actor_pose)])
            elif k == "driver_request":
                o["plans"].append(dict(now=int(e.driver_request.time_now_us)))
            elif k == "driver_return" and o["plans"]:
                o["plans"][-1]["traj"] = np.array([[q.timestamp_us, *p3(q.pose)] for q in e.driver_return.trajectory.poses], float)
        o["actors"] = {k: np.array(v, float) for k, v in o["actors"].items()}
        return o
    return asyncio.run(go())


def logged_objects(ds):
    out = {}
    for oid in ds.traffic_objects.keys():
        o = ds.traffic_objects[oid]
        t = o.trajectory
        T = np.asarray(t.timestamps_us, float)
        if len(T):
            out[str(oid)] = dict(T=T, P=np.asarray(t.positions, float)[:, :2], Y=np.unwrap(np.asarray(t.yaws, float)), L=float(o.aabb.x),
                                 W=float(o.aabb.y), static=bool(o.is_static), cls=str(o.label_class))
    return out


def traffic_at(objs, tg):
    """objs {id: dict T, P, Y, L, W, static, cls} -> per grid time a dict of arrays (tracks covering t only); v by 0.5 s difference."""
    out = []
    for t in tg:
        rows = []
        for i, o in objs.items():
            T, P, Y = o["T"], o["P"], o["Y"]
            if t < T[0] - 1 or t > T[-1] + 1:
                continue
            p = np.array([np.interp(t, T, P[:, 0]), np.interp(t, T, P[:, 1])])
            ta, tb = (t - 5e5, t) if t - 5e5 >= T[0] - 1 else (t, min(t + 5e5, T[-1]))
            v = (np.hypot(np.interp(tb, T, P[:, 0]) - np.interp(ta, T, P[:, 0]), np.interp(tb, T, P[:, 1]) - np.interp(ta, T, P[:, 1]))
                 / max((tb - ta) * 1e-6, 1e-3)) if tb > ta else 0.0
            rows.append((i, o["cls"], p[0], p[1], float(np.interp(t, T, Y)), o["L"], o["W"], v, o["static"]))
        c = list(zip(*rows)) if rows else [[]] * 9
        out.append(dict(id=np.array(c[0], str), cls=np.array(c[1], str), x=np.array(c[2], float), y=np.array(c[3], float),
                        yaw=np.array(c[4], float), L=np.array(c[5], float), W=np.array(c[6], float), v=np.array(c[7], float),
                        static=np.array(c[8], bool)))
    return out


def asl_objects(A, size):
    out = {}
    for i, X in A.items():
        if i == "EGO" or len(X) < 1:
            continue
        L, W, lab = size.get(i, (4.5, 2.0, "?"))
        out[i] = dict(T=X[:, 0], P=X[:, 1:3], Y=np.unwrap(X[:, 3]), L=float(L), W=float(W), static=bool(np.ptp(X[:, 1:3], 0).max() < 0.3),
                      cls=str(lab))
    return out


class Map:
    """Lane direction class, junction flag and drivable margin (lane polygons + road areas: the evaluator's offroad surface)."""

    def __init__(self, vm, pts, rad=80.0):
        import shapely
        import shapely.ops
        from eval.scorers.offroad import _get_lane_polygon, _repair_polygonal_geometry, _get_center_line_yaw_at_projection
        self.vm, self.yawp, self.sh = vm, _get_center_line_yaw_at_projection, shapely
        pts = np.asarray(pts, float)
        lo, hi = pts.min(0) - rad, pts.max(0) + rad
        box = shapely.box(lo[0], lo[1], hi[0], hi[1])
        self.lanes, polys = [], []
        for ln in vm.lanes:
            c = np.asarray(ln.center.points)[:, :2]
            if ((c >= lo) & (c <= hi)).all(1).any():
                g = _get_lane_polygon(ln)
                polys.append(g)
                self.lanes.append((ln, g, c, shapely.LineString(c)))
        cen = (lo + hi) / 2
        try:
            for ra in vm.get_road_areas_within(np.r_[cen, 0.0], float(np.hypot(*(hi - lo)) / 2)):
                polys.append(_repair_polygonal_geometry(vm.get_road_area_polygon_2d(ra.id)))
        except Exception:  # noqa: BLE001  maps without road areas
            pass
        U = _repair_polygonal_geometry(shapely.ops.unary_union(polys)).intersection(box) if polys else shapely.Polygon()
        self.U = U.buffer(0.001).buffer(-0.001)
        self.B = self.U.boundary
        shapely.prepare(self.U)
        self.byid = {ln.id: (ln, g, c) for ln, g, c, _ in self.lanes}
        self.tree = shapely.STRtree([g for _, g, _, _ in self.lanes]) if self.lanes else None

    def margin(self, P):
        sh = self.sh
        P = np.asarray(P, float).reshape(-1, 2)
        if self.U.is_empty:
            return np.full(len(P), np.nan)
        d = sh.distance(sh.points(P), self.B)
        return np.where(sh.contains_xy(self.U, P[:, 0], P[:, 1]), d, -d)

    def at(self, p, psi):
        """-> lane class (0 none, 1 same, 2 onc, 3 cross), junction flag (rf_offroad.Road.junction's rule)."""
        if self.tree is None:
            return 0, False
        pt = self.sh.Point(float(p[0]), float(p[1]))
        yaws, junc = [], False
        for i in self.tree.query(pt, predicate="within"):
            ln, g, c, ls = self.lanes[i]
            yaws.append(float(self.yawp(ls, pt)))
            turn = abs(np.degrees(wrap(np.arctan2(*(c[-1] - c[-2])[::-1]) - np.arctan2(*(c[1] - c[0])[::-1])))) if len(c) > 2 else 0.0
            if turn >= 20 and any(len(self.byid[q][0].next_lanes) >= 2 for q in ln.prev_lanes if q in self.byid):
                junc = True
        if not yaws:
            return 0, False
        y = np.array(yaws)
        junc = junc or bool(len(y) >= 2 and np.abs(wrap(y[:, None] - y[None])).max() >= np.radians(25))
        d = np.abs(wrap(y - psi))
        return (1 if (d < SAME).any() else 2 if (d > ONC).any() else 3), junc

    def fig(self, c, r=FIG_R):
        sh = self.sh
        clip = self.U.intersection(sh.Point(*c).buffer(r))
        gs = [g for g in getattr(clip, "geoms", [clip]) if g.geom_type == "Polygon"]
        lanes = [c_ for _, _, c_, _ in self.lanes if (np.hypot(*(c_ - c).T) < r).any()]
        return dict(road=[np.asarray(g.exterior.coords, np.float32) for g in gs],
                    holes=[np.asarray(h.coords, np.float32) for g in gs for h in g.interiors], lanes=[x.astype(np.float32) for x in lanes])


def drive_recs(run):
    """nuPlan driver log: {session: [records sorted by t0]} (anchor, poses, cmd)."""
    f = Path(run) / "driver-logs/drive.jsonl"
    S = {}
    if not f.exists():
        return S
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") == "drive":
            S.setdefault(r["session"], []).append(dict(t0=int(r["t0"]), anchor=r["anchor"], poses=r["poses"], cmd=r.get("cmd")))
    return {k: sorted(v, key=lambda r: r["t0"]) for k, v in S.items()}


def scene_job(job):
    scene, cfg, rolls, asl_actors = job
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    ds = loader(cfg).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg, Yg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.unwrap(np.asarray(tr.yaws, float))
    vc = ds.rig.vehicle_config
    dims = (float(vc.aabb_x_offset_m), float(vc.aabb_x_offset_m + vc.aabb_x_m), float(vc.aabb_y_m) / 2)
    loc = str(getattr(getattr(ds, "scene", None), "location", "") or "")
    objs = None if asl_actors else logged_objects(ds)
    zt = lambda t: float(np.interp(t, Tg, Pg[:, 2]))  # noqa: E731
    out = []
    mp = None
    for r in rolls:
        rid, run = r["rid"], r["run"]
        C, (ser, files) = ctrl(run, rid), series(run, scene, rid)
        o = dict(scene=scene, rid=rid, run=run, score=r["score"], metrics=r["metrics"], loc=loc, dims=dims, files=files)
        if C is None or ser is None:
            out.append(o)
            continue
        sel = np.arange(min(4, len(C["t"]) - 1), len(C["t"]), 5)          # 0.5 s grid on the controller clock (nuPlan: the decision times)
        tg = C["t"][sel]
        g = RouteGenerator.create(Pg, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
        route_err = 0
        try:
            for i in sel:
                yaw = C["yaw"][i]
                g.generate_route(int(C["t"][i]), Pose(np.array([C["x"][i], C["y"][i], zt(C["t"][i])]),
                                                      np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])))
            route = np.asarray(g.route_polyline_in_local.points, float)[:, :2]
        except Exception:  # noqa: BLE001
            route, route_err = Pg[:, :2].copy(), 1
        a = None
        if files["asl"]:
            try:
                a = read_asl(files["asl"], asl_actors)
            except Exception:  # noqa: BLE001
                a = None
        ob = asl_objects(a["actors"], a["size"]) if asl_actors and a else objs
        if ob is None:
            ob = {}
        traffic = traffic_at(ob, tg)
        if mp is None:
            mp = Map(ds.map, np.r_[Pg[:, :2], np.c_[C["x"], C["y"]]])
        lane = [mp.at((C["x"][i], C["y"][i]), C["yaw"][i]) for i in sel]
        rear, front, hw = dims
        fp = lambda i: np.array([[C["x"][i] + np.cos(C["yaw"][i]) * u - np.sin(C["yaw"][i]) * v, C["y"][i] + np.sin(C["yaw"][i]) * u + np.cos(C["yaw"][i]) * v]  # noqa: E731
                                 for u, v in ((front, hw), (front, -hw), (rear, -hw), (rear, hw))])
        o.update(ctrl=C, sel=sel, tg=tg, ser=ser, route=route.astype(np.float32), route_err=route_err, log=dict(T=Tg, P=Pg[:, :2], Y=Yg),
                 traffic=traffic, lane=np.array([x[0] for x in lane]), junc=np.array([x[1] for x in lane]),
                 margin=np.array([mp.margin(fp(i)).min() for i in sel]), margin_c=mp.margin(np.c_[C["x"][sel], C["y"][sel]]))
        if r["score"] is None or float(r["score"]) < 0.9 or r.get("fig"):
            pts = np.c_[C["x"], C["y"]]
            o["fig"] = mp.fig(pts[len(pts) // 2], r=FIG_R + float(np.ptp(pts, 0).max()) / 2)
            if a is not None and a["plans"]:
                o["plans"] = [(p["now"], p["traj"].astype(np.float32)) for p in a["plans"][::5] if "traj" in p]
            elif r.get("recs"):
                o["plans"] = [(x["t0"], np.asarray(x["anchor"], np.float32), np.asarray(x["poses"], np.float32)) for x in r["recs"]]
        out.append(o)
    return out


def extract(a):
    from concurrent.futures import ProcessPoolExecutor
    jobs = {}
    for run in a.runs:
        recs = drive_recs(run)
        S = json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]
        cfg = str(Path(run) / "generated-user-config-0.yaml")
        for r in S:
            jobs.setdefault((r["clipgt_id"], cfg), []).append(dict(rid=r["rollout_id"], run=run, score=r.get("score"), metrics=r.get("metrics") or {},
                                                                   recs=recs.get(r["rollout_id"], [])))
    units = [(s, c, v, a.asl_actors) for (s, c), v in sorted(jobs.items())][: a.limit or None]
    print(len(units), "scenes", flush=True)
    res, err = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(scene_job, u): u[0] for u in units}
        for i, f in enumerate(futs):
            try:
                res += f.result()
            except Exception as e:  # noqa: BLE001
                err.append((futs[f], repr(e)[:300]))
            if i % 50 == 0:
                print(i, "done", len(err), "errors", flush=True)
    pickle.dump(dict(rolls=res, errors=err), open(a.out, "wb"), protocol=4)
    print(len(res), "rollouts,", len(err), "errors ->", a.out, flush=True)
    for e in err[:5]:
        print(e)


def wl_job(job):
    """(scene, cfg) -> wait lines (STOP / YIELD / UNKNOWN; trajdata WAIT_LINE elements) within 30 m of the logged path."""
    scene, cfg = job
    from trajdata.maps.vec_map_elements import MapElementType as M
    ds = loader(cfg).get_data_source(scene)
    P = np.asarray(ds.rig.trajectory.positions, float)[:, :2]
    out = []
    for w in ds.map.elements[M.WAIT_LINE].values():
        q = np.asarray(w.polyline.points, float)[:, :2]
        if len(q) >= 2 and np.hypot(*(P[:, None] - q[None]).transpose(2, 0, 1)).min() < 30:
            out.append(dict(id=str(w.id), kind=str(w.wait_line_type), pts=q.astype(np.float32)))
    return scene, out


def waitlines(a):
    from concurrent.futures import ProcessPoolExecutor
    jobs = {}
    for run in a.runs:
        for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]:
            jobs.setdefault(r["clipgt_id"], str(Path(run) / "generated-user-config-0.yaml"))
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(wl_job, sorted(jobs.items())))
    pickle.dump(res, open(a.out, "wb"), protocol=4)
    print(len(res), "scenes,", sum(len(v) for v in res.values()), "wait lines ->", a.out)


# ---------------------------------------------------------------- report (Mac)
RULES = """
zero type   earliest flagged failure on the 2 Hz series (ol_clips / zeros_mech rule): at-fault collision (collision_front | collision_lateral),
            offroad, left_corridor_laterally. Rear contact (collision_rear) is never at fault: no zero, but it truncates progress.
collision   object hit = largest overlap (else nearest) with the ego box at the first contact, traffic interpolated by id between 0.5 s steps.
            ped / cyclist by class; stationary = object speed < 0.5 m/s or static (ego hit a stopped vehicle / obstacle); moving vehicles split
            by heading difference: same direction ahead (ego ran into it), crossing (60-120 deg), oncoming (> 120 deg), side (same direction,
            beside: sideswipe while turning or changing lane).
offroad     junction = the evaluator lane map says intersection at the ego (or 0.5 s around); inside / outside of the turn from AlpaSim's
            route turn over +-20 m and the ego's side of the route (rf_offroad.py rule); else straight / curve / lane edge.
corridor    beyond the log's end (ego station >= log length - 2 m: the evaluator measures a straight extension) -> route-end artefact; else the
            log's heading change over the stretch the ego covered (3 s before the event to 15 m past it) against the ego's own yaw change:
            log turns >= 30 deg and the ego < 10 deg (or the other way) -> went straight at the turn; same sign but less -> turn failed
            (under-turned / wide), more -> over-turned (cut inside); log < 30 deg and ego >= 30 deg -> turned off where the log went
            straight (wrong exit); log lateral jog >= 2.5 m with < 20 deg -> missed lane change; else drift.
low score   score = min(progress / 0.8, 1) without a hard failure (no comfort or speed term exists). Stop = v < 1 m/s for >= 1 s before
            any truncation; its reason at the stop's start: stopped object (moves < 1 m before the log reaches it) in the ego's path within
            15 m -> waiting behind a stopped lead the log passed; a stopped object that leaves (moves > 4 m) before the log arrives -> queue, or
            late start when the ego moves off > 2 s after it left; pedestrian in path within 10 m -> yield; moving object in path -> following; else
            the log's speed at the same station >= 2 m/s -> unexplained stop, else stopped where the log was slow. Primary = root cause:
            late start / stopped lead / pedestrian / unexplained stop (>= 1.5 s) > slow (median position-aligned speed ratio < 0.85; split
            by a moving lead within 20 m on >= half the steps) > other stops > rear contact alone > other. A rear contact (logged follower,
            physics NONE, drives into a slower ego) is secondary "rear_ended": it truncates progress.
in path     object centre within hw + W/2 + 0.3 m of the ego's actual path of the next 5 s (extended 20 m along its last heading) or of the
            straight heading line, its near edge 0 to G m ahead of the front bumper (G = 10 m pedestrians, 15 m stopped objects).
stop line   (PAI) the human stopped (log v < 0.5 m/s for >= 1 s) with a map wait line (STOP / YIELD / UNKNOWN) whose centre lies
            within 4 m of the log path, 3 m behind to 12 m ahead of the stop; the ego "ran" it when it reached the line's station - 1 m (within
            4 m of the log path) at > 2 m/s before the human moved off.
spin        |vx wz| >= 6 m/s^2 for >= 0.3 s, or |vy / vx| >= 0.2 at vx >= 3 m/s, in the 3 s before the event (steer-long 3.3).
"""
PED = {"pedestrian", "person", "PEDESTRIAN"}
CYC = {"bicycle", "BICYCLE", "cyclist", "rider", "motorcycle", "MOTORCYCLE"}
VEH = {"vehicle", "VEHICLE", "automobile", "heavy_truck", "bus", "trailer", "other_vehicle", "EGO"}


def cgroup(c):
    return "ped" if c in PED else "cyclist" if c in CYC else "vehicle" if c in VEH else "obstacle"


def poly_proj(R, P):
    """Point(s) P -> (distance, station, signed lateral (+ left), tangent heading) on polyline R."""
    P = np.asarray(P, float).reshape(-1, 2)
    a, b = R[:-1], R[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    S = np.r_[0.0, np.cumsum(np.sqrt(L2))]
    t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
    Q = a[None] + t[..., None] * ab[None]
    d = np.hypot(*(P[:, None] - Q).transpose(2, 0, 1))
    i = d.argmin(1)
    n = np.arange(len(P))
    h = np.arctan2(ab[i, 1], ab[i, 0])
    w = P - Q[n, i]
    lat = np.cos(h) * w[:, 1] - np.sin(h) * w[:, 0]
    return d[n, i], S[i] + t[n, i] * np.sqrt(L2[i]), lat, h


def dedupe(P):
    P = np.asarray(P, float)
    return P[np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-3]]


def first_pos(ser, name, t_max=np.inf):
    x = ser.get(name)
    if x is None or not len(x):
        return np.nan
    m = (np.nan_to_num(x[:, 1]) > 0) & (x[:, 0] <= t_max)
    return float(x[m, 0].min()) if m.any() else np.nan


def box_pts(x, y, yaw, L, W, n=6):
    u = np.r_[np.linspace(-L / 2, L / 2, n), np.full(n, L / 2), np.linspace(-L / 2, L / 2, n), np.full(n, -L / 2)]
    v = np.r_[np.full(n, W / 2), np.linspace(-W / 2, W / 2, n), np.full(n, -W / 2), np.linspace(-W / 2, W / 2, n)]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.c_[x + c * u - s * v, y + s * u + c * v]


def box_gap(b1, b2):
    """Approximate distance between two boxes (x, y, yaw, L, W); <= 0 when they overlap (a sample point of one inside the other)."""
    p1, p2 = box_pts(*b1), box_pts(*b2)

    def inside(P, b):
        c, s = np.cos(b[2]), np.sin(b[2])
        d = P - np.array(b[:2])
        u, v = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        return ((np.abs(u) <= b[3] / 2) & (np.abs(v) <= b[4] / 2)).any()
    if inside(p1, b2) or inside(p2, b1):
        return 0.0
    return float(np.hypot(*(p1[:, None] - p2[None]).transpose(2, 0, 1)).min())


def stop_lines(r, wl):
    """Where the human stopped (log v < 0.5 m/s for >= 1 s) at a wait line 0-12 m ahead of the stop, and what the ego did there.
    -> list of dict(kind, s_line, log_wait_s, ego_ran, ego_stopped, t_cross)."""
    T, S, v = r.LT, r.LS[:len(r.LT)], r.Lv
    if len(T) < 3:
        return []
    h = np.arctan2(*(r.LP[-1] - r.LP[-2])[::-1])
    Lx = np.vstack([r.LP, r.LP[-1] + 15.0 * np.array([np.cos(h), np.sin(h)])])
    out, seen = [], set()
    st = v < 0.5
    i = 0
    E = np.c_[r.x, r.y]
    while i < len(st):
        if st[i]:
            j = i
            while j < len(st) and st[j]:
                j += 1
            if (T[min(j, len(T) - 1)] - T[i]) * 1e-6 >= 1.0:
                s0 = S[i]
                t_end = T[min(j, len(T) - 1)]
                for w in wl:
                    if w["id"] in seen:
                        continue
                    _, sl, lat, _ = poly_proj(Lx, w["pts"].mean(0)[None])
                    if abs(lat[0]) <= 4.0 and -3.0 <= sl[0] - s0 <= 12.0:
                        seen.add(w["id"])
                        _, se, le, _ = poly_proj(Lx, E)
                        k = np.flatnonzero((se > sl[0] - 1.0) & (np.abs(le) < 4.0) & (r.v > 2.0))      # at the line, still moving
                        tc = float(r.t[k[0]]) if len(k) else np.nan
                        near = (se < sl[0]) & (se > sl[0] - 15)
                        out.append(dict(kind=w["kind"], s_line=float(sl[0]), log_wait_s=float((t_end - T[i]) * 1e-6),
                                        ego_ran=bool(np.isfinite(tc) and tc < t_end), t_cross=(tc - r.t0) / 1e6 if np.isfinite(tc) else np.nan,
                                        ego_stopped=bool((r.v[near] < 0.5).any()) if near.any() else False))
            i = j
        else:
            i += 1
    return out


class Roll:
    """Derived quantities of one extracted rollout."""

    def __init__(self, o):
        self.o, C = o, o["ctrl"]
        self.t, self.x, self.y, self.yaw = C["t"], C["x"], C["y"], C["yaw"]
        self.v = np.hypot(C["vx"], C["vy"])
        self.t0 = self.t[0] - 100_000
        self.rear, self.front, self.hw = o["dims"]
        L = o["log"]
        self.LP = dedupe(L["P"])
        if len(self.LP) < 2:                                                # the logged ego never moved
            self.LP = np.r_[L["P"][:1], L["P"][:1] + 0.01 * np.array([[np.cos(L["Y"][0]), np.sin(L["Y"][0])]])]
        self.LS = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.LP, axis=0).T))]
        keep = np.r_[True, np.hypot(*np.diff(L["P"], axis=0).T) > 1e-3]
        T = L["T"][keep] if keep.sum() >= 2 else L["T"][[0, -1]]
        self.LT = T
        self.Lv = np.gradient(self.LS, T * 1e-6) if len(T) > 2 else np.zeros(len(T))
        self.route = dedupe(o["route"]) if len(o["route"]) > 1 else self.LP
        if len(self.route) < 2:
            self.route = self.LP
        self.tg, self.tr = o["tg"], o["traffic"]

    def k(self, t):
        return int((t - self.t0) // 500_000)

    def pose(self, t):
        return np.array([np.interp(t, self.t, self.x), np.interp(t, self.t, self.y), np.interp(t, self.t, self.yaw)])

    def speed(self, t):
        return float(np.interp(t, self.t, self.v))

    def objs_at(self, t):
        """Traffic at time t, interpolated by id between the two neighbouring 0.5 s steps."""
        tg = self.tg
        j = int(np.clip(np.searchsorted(tg, t) - 1, 0, len(tg) - 1))
        j2 = min(j + 1, len(tg) - 1)
        A, B = self.tr[j], self.tr[j2]
        w = 0.0 if j2 == j else float(np.clip((t - tg[j]) / (tg[j2] - tg[j]), 0, 1))
        out = {k: A[k].copy() for k in A}
        if w > 0 and len(A["id"]) and len(B["id"]):
            ib = {i: n for n, i in enumerate(B["id"])}
            for n, i in enumerate(A["id"]):
                m = ib.get(i)
                if m is not None:
                    for k in ("x", "y"):
                        out[k][n] = (1 - w) * A[k][n] + w * B[k][m]
                    out["yaw"][n] = A["yaw"][n] + w * wrap(B["yaw"][m] - A["yaw"][n])
        return out

    def ego_box(self, t):
        a = self.pose(t)
        c = (self.rear + self.front) / 2
        return (a[0] + c * np.cos(a[2]), a[1] + c * np.sin(a[2]), a[2], self.front - self.rear, 2 * self.hw)

    def in_path(self, j, G):
        """Objects of grid step j in the ego's path within G m: list of dict(i, gap, lat, cls, grp, v, static, id)."""
        st = self.tr[j]
        if not len(st["id"]):
            return []
        i0 = self.o["sel"][j]
        a = np.array([self.x[i0], self.y[i0], self.yaw[i0]])
        P = np.c_[self.x[i0:i0 + 51], self.y[i0:i0 + 51]]
        h = self.yaw[min(i0 + 50, len(self.yaw) - 1)]
        P = dedupe(np.vstack([P, P[-1] + 20.0 * np.array([np.cos(h), np.sin(h)])]))
        X = np.c_[st["x"], st["y"]]
        d, s, lat, _ = poly_proj(P, X)
        c, sn = np.cos(a[2]), np.sin(a[2])
        D = X - a[:2]
        ex, ey = c * D[:, 0] + sn * D[:, 1], -sn * D[:, 0] + c * D[:, 1]
        out = []
        for n in range(len(X)):
            lim = self.hw + st["W"][n] / 2 + 0.3
            g1, g2 = s[n] - self.front - st["L"][n] / 2, ex[n] - self.front - st["L"][n] / 2
            hit = (-1.0 <= g1 <= G and abs(lat[n]) <= lim) or (-1.0 <= g2 <= G and abs(ey[n]) <= lim)
            if hit:
                out.append(dict(i=n, gap=float(min(g for g, l_ in ((g1, lat[n]), (g2, ey[n])) if -1 <= g <= G and abs(l_) <= lim)),
                                lat=float(ey[n]), cls=str(st["cls"][n]), grp=cgroup(str(st["cls"][n])), v=float(st["v"][n]), static=bool(st["static"][n]),
                                id=str(st["id"][n]), x=float(st["x"][n]), y=float(st["y"][n])))
        return out

    def log_station(self, P):
        return poly_proj(self.LP, P)[1]

    def vlog_at(self, s):
        return float(np.interp(s, self.LS, self.Lv))


def classify(o, ref=None, wl=None):
    """One extracted rollout -> row dict (primary / secondary labels and the facts behind them) and encounter lists."""
    row = dict(scene=o["scene"], score=o["score"], rid=o["rid"])
    if "ctrl" not in o or o.get("ser") is None:
        row.update(primary="infra", zero_type="infra" if not o["score"] else "", notes="no controller trace / metrics")
        return row, [], []
    r, ser, m = Roll(o), o["ser"], o["metrics"]
    sc = float(o["score"] or 0)
    tf = {"collision": np.nanmin([first_pos(ser, "collision_front"), first_pos(ser, "collision_lateral")]),
          "offroad": first_pos(ser, "offroad"), "corridor": first_pos(ser, "left_corridor_laterally")}
    flag = {"collision": m.get("collision_at_fault"), "offroad": m.get("offroad"), "corridor": m.get("left_corridor_laterally")}
    t_rear = first_pos(ser, "collision_rear")
    t_any = first_pos(ser, "collision_any")
    zt = ""
    t_ev = np.nan
    if sc == 0:
        c = {k: v for k, v in tf.items() if float(flag[k] or 0) > 0 and np.isfinite(v)}
        zt = min(c, key=c.get) if c else "other"
        t_ev = c[zt] if c else np.nan
    row.update(zero_type=zt, progress=m.get("progress_clipped_rel"), gt_dist=m.get("gt_dist_traveled_m"), dist=m.get("dist_traveled_m"),
               t_ev_s=(t_ev - r.t0) / 1e6 if np.isfinite(t_ev) else np.nan, k_ev=r.k(t_ev) if np.isfinite(t_ev) else -1,
               v_ev=r.speed(t_ev) if np.isfinite(t_ev) else np.nan, loc=o.get("loc", ""), black=bool(np.nan_to_num(ser.get("img_is_black", np.zeros((1, 2)))[:, 1]).max() > 0))
    sec, notes = [], []
    # spin (steer-long 3.3), on the 10 Hz trace
    if np.isfinite(t_ev):
        w = (r.t >= t_ev - 3e6) & (r.t <= t_ev)
        A = np.abs(o["ctrl"]["vx"] * o["ctrl"]["wz"])[w]
        run_ = np.convolve((A >= 6).astype(int), np.ones(3, int), "valid").max() >= 3 if w.sum() >= 3 else False
        vx = o["ctrl"]["vx"][w]
        slip = ((np.abs(o["ctrl"]["vy"][w]) / np.maximum(np.abs(vx), 1e-3) >= 0.2) & (np.abs(vx) >= 3)).any()
        if run_ or slip:
            sec.append("spin")
    jg = lambda t: int(np.clip(np.searchsorted(r.tg, t) - 1, 0, len(r.tg) - 1))  # noqa: E731
    obj = {}
    if zt == "collision":
        e = r.ego_box(t_ev)
        st = r.objs_at(t_ev)
        if len(st["id"]):
            g = np.array([box_gap(e, (st["x"][n], st["y"][n], st["yaw"][n], st["L"][n], st["W"][n])) for n in range(len(st["id"]))])
            n = int(g.argmin())
            a = r.pose(t_ev)
            dpsi = float(np.degrees(abs(wrap(st["yaw"][n] - a[2]))))
            c_, s_ = np.cos(a[2]), np.sin(a[2])
            ex, ey = c_ * (st["x"][n] - a[0]) + s_ * (st["y"][n] - a[1]), -s_ * (st["x"][n] - a[0]) + c_ * (st["y"][n] - a[1])
            obj = dict(cls=str(st["cls"][n]), grp=cgroup(str(st["cls"][n])), v=float(st["v"][n]), static=bool(st["static"][n]), gap=float(g[n]),
                       dpsi=dpsi, ex=float(ex), ey=float(ey), id=str(st["id"][n]))
            side = "front" if first_pos(ser, "collision_front") <= np.nan_to_num(first_pos(ser, "collision_lateral"), nan=np.inf) else "lateral"
            obj["side"] = side
            if obj["grp"] in ("ped", "cyclist"):
                p = f"coll_{obj['grp']}"
            elif obj["static"] or obj["v"] < 0.5:
                p = "coll_stationary"
            elif dpsi > 120:
                p = "coll_veh_oncoming"
            elif dpsi >= 60:
                p = "coll_veh_crossing"
            elif ex > 0 and abs(ey) < r.hw + 1.0 and side == "front":
                p = "coll_veh_ahead"
            else:
                p = "coll_veh_side"
        else:
            p = "coll_unknown"
        row.update(hit_id=obj.get("id", ""), hit_cls=obj.get("cls", ""), hit_v=obj.get("v", np.nan), hit_dpsi=obj.get("dpsi", np.nan), hit_side=obj.get("side", ""),
                   hit_gap=obj.get("gap", np.nan))
        lane = o["lane"][max(jg(t_ev) - 4, 0):jg(t_ev) + 1]
        if (lane == 2).any():
            sec.append("oncoming_lane")
        if o["junc"][max(jg(t_ev) - 1, 0):jg(t_ev) + 2].any():
            sec.append("junction")
    elif zt == "offroad":
        j = jg(t_ev)
        a = r.pose(t_ev)
        junc = bool(o["junc"][max(j - 1, 0):j + 2].any())
        d, s0, lat, _ = poly_proj(r.route, a[None, :2])
        S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(r.route, axis=0).T))]
        hd = lambda s: float(np.arctan2(*np.diff(np.c_[np.interp([s - 2, s + 2], S, r.route[:, 0]), np.interp([s - 2, s + 2], S, r.route[:, 1])], axis=0)[0][::-1]))  # noqa: E731
        turn = float(np.degrees(wrap(hd(min(s0[0] + 20, S[-1] - 2)) - hd(max(s0[0] - 20, 2)))))
        inside = abs(turn) >= 20 and np.sign(lat[0]) == np.sign(turn)
        p = ("off_junc_inside" if inside else "off_junc_outside") if junc and abs(turn) >= 20 else "off_junc_straight" if junc else \
            ("off_curve_inside" if inside else "off_curve_outside") if abs(turn) >= 20 else "off_straight"
        if t_ev <= r.t[0]:                                                  # offroad from the evaluator's first step: the start pose
            p = "off_at_start"
        row.update(route_turn=turn, route_lat=float(lat[0]), margin_ev=float(o["margin"][j]))
    elif zt == "corridor":
        a = r.pose(t_ev)
        s_ev = float(r.log_station(a[None, :2])[0])
        tb = max(t_ev - 3e6, r.t[0])
        s_b = float(r.log_station(r.pose(tb)[None, :2])[0])
        s_hi = min(s_ev + 15, r.LS[-1])
        hl = lambda s: float(np.interp(s, r.LS[:len(r.LT)], np.unwrap(np.interp(r.LT, o["log"]["T"], o["log"]["Y"]))))  # noqa: E731
        dl = float(np.degrees(hl(s_hi) - hl(s_b)))
        de = float(np.degrees(r.pose(t_ev)[2] - r.pose(tb)[2]))
        W = np.c_[np.interp(np.linspace(s_b, s_hi, 20), r.LS, r.LP[:, 0]), np.interp(np.linspace(s_b, s_hi, 20), r.LS, r.LP[:, 1])]
        ch = W[-1] - W[0]
        nrm = np.array([-ch[1], ch[0]]) / max(np.hypot(*ch), 1e-6)
        jog = float(np.ptp((W - W[0]) @ nrm))
        beyond = s_ev >= r.LS[-1] - 2.0
        if beyond:
            p = "cor_route_end"
        elif abs(dl) >= 30 and (abs(de) < 10 or np.sign(de) != np.sign(dl)):
            p = "cor_straight_at_turn"
        elif abs(dl) >= 30:
            p = "cor_turn_under" if abs(de) < abs(dl) else "cor_turn_over"
        elif abs(de) >= 30:
            p = "cor_wrong_exit"
        elif jog >= 2.5:
            p = "cor_lane_change"
        else:
            p = "cor_drift"
        tl_ev = float(np.interp(s_ev, r.LS[:len(r.LT)], r.LT))               # when the log was where the ego is now
        row.update(log_dpsi=dl, ego_dpsi=de, log_jog=jog, beyond_log=bool(beyond), log_len=float(r.LS[-1]), log_v_end=float(r.Lv[-1]),
                   log_stop_s=float(np.sum(np.diff(o["log"]["T"]) * (np.hypot(*np.diff(o["log"]["P"], axis=0).T) / np.maximum(np.diff(o["log"]["T"]) * 1e-6, 1e-3) < 0.5)) * 1e-6),
                   ahead_of_log_s=float((tl_ev - t_ev) / 1e6))
        if o["junc"][max(jg(t_ev) - 4, 0):jg(t_ev) + 1].any():
            sec.append("junction")
    elif zt == "other":
        p = "zero_other"
    else:
        p = ""
    # stops and slowness (all rollouts, before the end of valid driving)
    t_tr = np.nanmin([t_ev, t_rear, t_any]) if np.isfinite(np.nanmin([t_ev, t_rear, t_any, np.inf])) else r.t[-1]
    t_tr = min(t_tr, r.t[-1])
    vv = r.v[r.t <= t_tr]
    stops, stop_t = [], {}
    i = 0
    while i < len(vv):
        if vv[i] < 1.0:
            j = i
            while j < len(vv) and vv[j] < 1.0:
                j += 1
            if (j - i) >= 10:
                t_s = r.t[i]
                jj = jg(t_s)
                P_ = r.in_path(jj, 15.0)
                a = r.pose(t_s)
                s_e = float(r.log_station(a[None, :2])[0])
                vl = r.vlog_at(s_e)
                peds = [q for q in P_ if q["grp"] == "ped" and q["gap"] <= 10]
                stp = [q for q in P_ if q["grp"] != "ped" and (q["v"] < 0.5 or q["static"])]
                mov = [q for q in P_ if q["grp"] != "ped" and q["v"] >= 0.5]
                passed, go = False, np.nan
                if stp:                                                     # the nearest stopped object: when does it leave, and did the
                    q = min(stp, key=lambda q: q["gap"])                    # log reach its station while it was still there
                    _, so, lo, _ = poly_proj(r.LP, np.array([[q["x"], q["y"]]]))
                    for j2 in range(jj, len(r.tg)):
                        k2 = np.flatnonzero(r.tr[j2]["id"] == q["id"])
                        if len(k2) and np.hypot(r.tr[j2]["x"][k2[0]] - q["x"], r.tr[j2]["y"][k2[0]] - q["y"]) > 4.0:   # left, not creeping
                            go = (r.tg[j2] - r.t0) / 1e6
                            break
                    t_reach = np.interp(so[0], r.LS[:len(r.LT)], r.LT) if r.LS[-1] > so[0] + 3 else np.inf
                    passed = bool(np.isfinite(t_reach) and (np.isnan(go) or (t_reach - r.t0) / 1e6 < go))
                why = "ped_yield" if peds else "stopped_lead" if stp else "follow_lead" if mov else "unexplained" if vl >= 2 else "with_log"
                dur = (j - i) * 0.1
                if why == "stopped_lead" and not passed:                    # a queue: the lead left; late start if the ego follows > 2 s later
                    why = "late_start" if np.isfinite(go) and (t_s - r.t0) / 1e6 + dur > go + 2.0 else "queue"
                stops.append(dict(t_s=(t_s - r.t0) / 1e6, dur=dur, why=why, log_passed=passed, vlog=vl, lead_go=go,
                                  obj=(min(stp or mov or peds, key=lambda q: q["gap"])["cls"] if (stp or mov or peds) else "")))
                stop_t[why] = stop_t.get(why, 0) + dur
            i = j
        else:
            i += 1
    # position-aligned speed ratio while moving
    rat, lead_steps = [], 0
    for jj, tt in enumerate(r.tg):
        if tt > t_tr:
            break
        a = r.pose(tt)
        vl = r.vlog_at(float(r.log_station(a[None, :2])[0]))
        if vl >= 2 and r.speed(tt) >= 1:
            rat.append(r.speed(tt) / vl)
            lead_steps += any(q["grp"] != "ped" and q["v"] >= 0.5 for q in r.in_path(jj, 20.0))
    ratio = float(np.median(rat)) if rat else np.nan
    row.update(n_stops=len(stops), stop_s=sum(s["dur"] for s in stops), speed_ratio=ratio,
               stops=";".join(f"{s['why']}@{s['t_s']:.1f}s/{s['dur']:.1f}s{'/logpassed' if s['log_passed'] else ''}{'/go%.1fs' % s['lead_go'] if np.isfinite(s['lead_go']) else ''}{'/' + s['obj'] if s['obj'] else ''}" for s in stops),
               rear_hit=bool(np.isfinite(t_rear)), max_lat=float(np.nanmax(ser["lateral_dist_to_gt_trajectory"][:, 1])) if "lateral_dist_to_gt_trajectory" in ser else np.nan)
    if sc > 0 and sc < 0.9:
        rear = bool(np.isfinite(t_rear) or (np.isfinite(t_any) and t_any < r.t[-1] - 1e5))
        if rear:
            sec.append("rear_ended")
            tr_ = np.nanmin([t_rear, t_any])
            e = r.ego_box(tr_)
            st = r.objs_at(tr_)
            a = r.pose(tr_)
            lag = float(np.interp(tr_, r.LT, r.LS[:len(r.LT)]) - r.log_station(a[None, :2])[0])
            row.update(rear_t_s=(tr_ - r.t0) / 1e6, rear_v_ego=r.speed(tr_), rear_lag=lag,
                       rear_vlog=r.vlog_at(float(np.interp(tr_, r.LT, r.LS[:len(r.LT)]))))
            if len(st["id"]):
                g = np.array([box_gap(e, (st["x"][n], st["y"][n], st["yaw"][n], st["L"][n], st["W"][n])) for n in range(len(st["id"]))])
                n = int(g.argmin())
                c_, s_ = np.cos(a[2]), np.sin(a[2])
                row.update(rear_cls=str(st["cls"][n]), rear_v_obj=float(st["v"][n]), rear_gap=float(g[n]),
                           rear_ex=float(c_ * (st["x"][n] - a[0]) + s_ * (st["y"][n] - a[1])))
        bad = {k: v for k, v in stop_t.items() if k in ("late_start", "stopped_lead", "ped_yield", "unexplained") and v >= 1.5}
        if bad:                                                             # root cause first; the rear contact is its consequence
            p = "low_" + max(bad, key=bad.get)
        elif np.isfinite(ratio) and ratio < 0.85:
            p = "low_slow_behind_lead" if rat and lead_steps >= len(rat) / 2 else "low_slow_free"
        elif stop_t and max(stop_t.values()) >= 1.5:
            p = "low_" + max(stop_t, key=stop_t.get)
        elif rear:
            p = "low_rear_ended"
        else:
            p = "low_other"
        if row["max_lat"] >= 2.0:
            sec.append("partial_corridor")
        if any(s["why"] == "stopped_lead" and s["log_passed"] for s in stops):
            sec.append("stopped_lead_log_passed")
        if any(s["why"] == "late_start" for s in stops):
            sec.append("late_start")
    elif sc >= 0.9:
        p = "ok"
    if ref is not None:
        rs = float(ref["score"] or 0)
        row["ref_score"] = rs
        if sc == 0 and rs == 0:
            sec.append("ref_zero")
    # pedestrian and stopped-object encounters (before the event / truncation)
    peds, blocks = [], []
    seen_p, seen_b = set(), set()
    for jj, tt in enumerate(r.tg):
        if tt > t_tr:
            break
        P_ = r.in_path(jj, 15.0)
        for q in P_:
            if q["grp"] == "ped" and q["gap"] <= 10 and q["id"] not in seen_p:
                seen_p.add(q["id"])
                v0 = r.speed(tt)
                w = (r.t >= tt) & (r.t <= tt + 3e6)
                vmin = float(r.v[w].min()) if w.any() else v0
                peds.append(dict(scene=o["scene"], id=q["id"], k=r.k(tt), t_s=(tt - r.t0) / 1e6, gap=q["gap"], v0=v0, vmin=vmin, ped_v=q["v"],
                                 braked=bool(v0 - vmin >= max(1.0, 0.3 * v0)), slow=bool(v0 < 1.5)))
            if q["grp"] != "ped" and (q["v"] < 0.5 or q["static"]) and q["id"] not in seen_b:
                seen_b.add(q["id"])
                _, so, lo, _ = poly_proj(r.LP, np.array([[q["x"], q["y"]]]))
                go = np.inf
                for j2 in range(jj, len(r.tg)):
                    k2 = np.flatnonzero(r.tr[j2]["id"] == q["id"])
                    if len(k2) and np.hypot(r.tr[j2]["x"][k2[0]] - q["x"], r.tr[j2]["y"][k2[0]] - q["y"]) > 1.0:
                        go = r.tg[j2]
                        break
                reach = np.interp(so[0], r.LS[:len(r.LT)], r.LT) if r.LS[-1] > so[0] + 3 else np.inf
                blocks.append(dict(scene=o["scene"], id=q["id"], cls=q["cls"], k=r.k(tt), t_s=(tt - r.t0) / 1e6, gap=q["gap"], v0=r.speed(tt),
                                   log_passed=bool(np.isfinite(reach) and reach < go), left_s=(go - r.t0) / 1e6 if np.isfinite(go) else np.nan,
                                   log_lat=float(lo[0]), x=q["x"], y=q["y"]))
    for b in blocks:                                                        # outcome: hit / passed / waited behind / other
        hit = zt == "collision" and obj.get("id") == b["id"]
        tl = r.t[r.t <= t_tr]
        X = np.c_[r.x[r.t <= t_tr], r.y[r.t <= t_tr]]
        Y = r.yaw[r.t <= t_tr]
        ex = np.cos(Y) * (b["x"] - X[:, 0]) + np.sin(Y) * (b["y"] - X[:, 1])
        passed = bool((ex < r.rear - 1.0).any() & (tl >= b["t_s"] * 1e6 + r.t0).any() and (ex[tl >= b["t_s"] * 1e6 + r.t0] < r.rear - 1).any())
        waited = any(s["why"] in ("stopped_lead", "queue", "late_start") and s["t_s"] >= b["t_s"] - 0.5 for s in stops)
        b["outcome"] = "hit" if hit else "passed" if passed else "waited" if waited else "other"
    for pdx in peds:
        pdx["hit"] = zt == "collision" and obj.get("grp") == "ped" and obj.get("id") == pdx["id"]
        pdx["score"] = sc
        pdx["zero_type"] = zt
    if wl is not None:
        SL = stop_lines(r, wl)
        ran = [x for x in SL if x["ego_ran"]]
        row.update(n_stopline=len(SL), stopline_ran=len(ran), stopline_kinds="+".join(sorted({x["kind"] for x in SL})),
                   stopline_ran_t=";".join(f"{x['kind']}@{x['t_cross']:.1f}s" for x in ran))
        if ran:
            sec.append("ran_stop_line")
    row.update(primary=p, secondary="+".join(sec), n_ped=len(peds), n_block=len(blocks),
               ped_braked=sum(q["braked"] for q in peds), block_out="+".join(sorted({b["outcome"] for b in blocks})),
               notes=";".join(notes))
    return row, peds, blocks


def load_set(f):
    d = pickle.load(open(f, "rb"))
    return {o["scene"]: o for o in d["rolls"]}


def report(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for spec in a.set:
        name, files = spec.split("=", 1)
        fs = files.split(",")
        D = load_set(fs[0])
        Rf = load_set(fs[1]) if len(fs) > 1 and fs[1] else {}
        WL = pickle.load(open(fs[2], "rb")) if len(fs) > 2 else None
        rows, P, B = [], [], []
        for s, o in sorted(D.items()):
            row, peds, blocks = classify(o, Rf.get(s), None if WL is None else WL.get(s, []))
            rows.append(row)
            P += peds
            B += blocks
        df = pd.DataFrame(rows)
        act = df.get("hit_cls", pd.Series("", index=df.index)).replace("", np.nan)
        for c in ("rear_cls",):
            if c in df:
                act = act.fillna(df[c])
        df["actor_cls"] = act.fillna(df.stops.fillna("").str.extract(r"/([A-Za-z_]+)(?:;|$)")[0])
        df["v_ev"] = df.v_ev.fillna(df.get("rear_v_ego"))
        key = ["scene", "score", "zero_type", "primary", "secondary", "k_ev", "t_ev_s", "v_ev", "actor_cls", "progress", "speed_ratio", "stops", "notes"]
        df = df[key + [c for c in df.columns if c not in key]]
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.4g")
        pd.DataFrame(P).to_csv(out / f"{name}_peds.csv", index=False, float_format="%.4g")
        pd.DataFrame(B).to_csv(out / f"{name}_blocks.csv", index=False, float_format="%.4g")
        sc = df.score.fillna(0)
        band = pd.cut(sc, [-0.01, 0, 0.6, 0.9, 1.0], labels=["zero", "low<0.6", "mid0.6-0.9", "ok>=0.9"])
        tab = df.assign(band=band, deficit=(1 - sc) / len(df)).groupby(["band", "primary"], observed=True).agg(n=("score", "size"),
                                                                                                             deficit=("deficit", "sum"))
        summ[name] = dict(n=len(df), mean=float(sc.mean()), zeros=int((sc == 0).sum()), gap094=float(0.94 - sc.mean()),
                          deficit_total=float((1 - sc).mean()),
                          table=[dict(band=b, primary=q, n=int(v.n), deficit=round(float(v.deficit), 4)) for (b, q), v in tab.iterrows()],
                          secondary=df.secondary.fillna("").str.split("+").explode().value_counts().drop("", errors="ignore").to_dict(),
                          peds=dict(encounters=len(P), scenes=len({x["scene"] for x in P}), braked=sum(x["braked"] for x in P),
                                    hit=sum(x["hit"] for x in P), coll_ped_zero=int((df.primary == "coll_ped").sum())),
                          blocks=pd.DataFrame(B).groupby(["outcome", "log_passed"]).size().rename("n").reset_index().to_dict("records") if B else [])
        print(name, summ[name]["n"], round(summ[name]["mean"], 4), summ[name]["zeros"])
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))


COL = dict(vehicle="#8c8c8c", ped="#e8590c", cyclist="#9c36b5", obstacle="#2f2f2f")


def draw_bev(ax, o, t, title, hit_id=None, R=45.0):
    """Road, lanes, route, log path, ego path and box, traffic boxes and the served plan at time t, centred on the ego."""
    import matplotlib.patches as mp
    r = Roll(o)
    a = r.pose(t)
    F = o.get("fig") or {}
    for g in F.get("road", []):
        ax.add_patch(mp.Polygon(g, closed=True, fc="#eef1f4", ec="#b7c0c8", lw=0.6, zorder=0))
    for h in F.get("holes", []):
        ax.add_patch(mp.Polygon(h, closed=True, fc="white", ec="#b7c0c8", lw=0.6, zorder=0))
    for c in F.get("lanes", []):
        ax.plot(c[:, 0], c[:, 1], color="#c9d3dc", lw=0.5, zorder=1)
        if len(c) > 3:
            m = len(c) // 2
            ax.annotate("", c[m + 1], c[m], arrowprops=dict(arrowstyle="-|>", color="#aab6c2", lw=0.5, mutation_scale=6), zorder=1)
    ax.plot(r.route[:, 0], r.route[:, 1], color="#1c7ed6", lw=1.4, ls="--", zorder=2, label="route")
    ax.plot(r.LP[:, 0], r.LP[:, 1], color="black", lw=1.0, zorder=3, label="log path")
    k = r.t <= t
    ax.plot(r.x[k], r.y[k], color="#e03131", lw=1.8, zorder=4, label="ego path")
    ax.plot(r.x[~k], r.y[~k], color="#e03131", lw=0.8, ls=":", zorder=4)
    lt = np.interp(t, r.LT, r.LS[:len(r.LT)])
    lp = np.c_[np.interp(lt, r.LS, r.LP[:, 0]), np.interp(lt, r.LS, r.LP[:, 1])][0]
    ax.plot(*lp, marker="o", ms=5, mfc="white", mec="black", zorder=6, label="log at same time")
    pl = o.get("plans") or []
    if pl:
        q = min(pl, key=lambda z: abs(z[0] - t) if z[0] <= t else np.inf)
        if len(q) == 2:
            P = q[1][:, 1:3]
        else:
            an, P0 = np.asarray(q[1], float), np.asarray(q[2], float)
            c_, s_ = np.cos(an[2]), np.sin(an[2])
            P = np.c_[an[0] + c_ * P0[:, 0] - s_ * P0[:, 1], an[1] + s_ * P0[:, 0] + c_ * P0[:, 1]]
        ax.plot(P[:, 0], P[:, 1], color="#f08c00", lw=1.6, marker=".", ms=3, zorder=5, label="served plan")
    st = r.objs_at(t)
    for n in range(len(st["id"])):
        if np.hypot(st["x"][n] - a[0], st["y"][n] - a[1]) > R * 1.5:
            continue
        gp = cgroup(str(st["cls"][n]))
        C = box_pts(st["x"][n], st["y"][n], st["yaw"][n], st["L"][n], st["W"][n], n=2)[[1, 2, 4, 7]]
        hl = hit_id is not None and str(st["id"][n]) == str(hit_id)
        ax.add_patch(mp.Polygon(C, closed=True, fc=COL[gp] if not hl else "#fa5252", ec="black" if hl else "none", alpha=0.85, lw=1.2, zorder=5))
        if gp in ("ped", "cyclist"):
            ax.plot(st["x"][n], st["y"][n], marker="o", ms=6, mfc="none", mec=COL[gp], zorder=6)
    E = box_pts(*r.ego_box(t), n=2)[[1, 2, 4, 7]]
    ax.add_patch(mp.Polygon(E, closed=True, fc="#e03131", ec="black", lw=0.8, zorder=7))
    ax.set_xlim(a[0] - R, a[0] + R)
    ax.set_ylim(a[1] - R, a[1] + R)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def figs(a):
    """picks.json: [{name, set, scene, t_s, title, caption, hit_id?, frame?}] -> <figs>/tax_<name>.png (BEV, plus the front frame if given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = {}
    for spec in a.set:
        k, f = spec.split("=", 1)
        sets[k] = f
    cache = {}
    for pk in json.loads(Path(a.picks).read_text()):
        if pk["set"] not in cache:
            cache[pk["set"]] = load_set(sets[pk["set"]])
        o = cache[pk["set"]][pk["scene"]]
        r = Roll(o)
        t = r.t0 + pk["t_s"] * 1e6
        fr = pk.get("frame")
        fig, axs = plt.subplots(1, 2 if fr else 1, figsize=(13 if fr else 6.5, 6.2), gridspec_kw=dict(width_ratios=[1, 1.35]) if fr else None)
        ax = axs[0] if fr else axs
        draw_bev(ax, o, t, f"{pk['scene'][:24]}  t = {pk['t_s']:.1f} s  score {float(o['score'] or 0):.2f}", pk.get("hit_id"), pk.get("R", 45.0))
        ax.legend(loc="lower left", fontsize=6.5, framealpha=0.8)
        if fr and Path(fr).exists():
            axs[1].imshow(plt.imread(fr))
            axs[1].set_axis_off()
            axs[1].set_title(f"front camera at {pk.get('frame_t', pk['t_s']):.1f} s", fontsize=9)
        fig.suptitle(pk["title"], fontsize=11)
        fig.text(0.01, 0.01, pk["caption"], fontsize=7.5, wrap=True, va="bottom")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(Path(a.figs) / f"tax_{pk['name']}.png", dpi=110)
        plt.close(fig)
        print("tax_" + pk["name"])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--asl-actors", action="store_true")
    w = sp.add_parser("waitlines")
    w.add_argument("--runs", nargs="+", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--workers", type=int, default=8)
    r = sp.add_parser("report")
    r.add_argument("--set", nargs="+", required=True)
    r.add_argument("--out", required=True)
    f = sp.add_parser("figs")
    f.add_argument("--set", nargs="+", required=True)
    f.add_argument("--picks", required=True)
    f.add_argument("--figs", required=True)
    a = ap.parse_args()
    {"extract": extract, "waitlines": waitlines, "report": report, "figs": figs}[a.cmd](a)


if __name__ == "__main__":
    main()
