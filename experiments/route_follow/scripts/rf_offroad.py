#!/usr/bin/env python3
"""ROUTE-FOLLOW offroad diagnosis (research/route-follow-offroad.md): why arms trained with the route hinge (on_route.py) on non-log rows
gain offroad zeros on held 460. CPU only, read-only on run dirs and pools. Box, AlpaSim's venv.

  rf_offroad.py held --runs LABEL=<run dir>[,<run dir>..] ... --out DIR      -> DIR/held.parquet (one row per label x offroad-zero scene),
                                                                              DIR/held_dec.parquet (all rollouts: scene, k_z, lat to route),
                                                                              DIR/fig/<label>__<scene>.pkl (BEV data)
  rf_offroad.py pool --name L18RB0 --out DIR                                  -> DIR/pool.parquet (one row per pool row)

Geometry: everything in the rollout's local frame (AlpaSim data source; the driver's anchors are in it). Drivable surface = union of the
vector map's lane polygons (eval.scorers.offroad._get_lane_polygon, the evaluator's own) and road-area polygons near the rollout, i.e. what the
evaluator's offroad scorer accepts (nuPlan maps carry no road edges). margin(p) = signed distance of point p to the drivable boundary (> 0
inside); a 2.3 m wide car centred on p fits only when margin >= FIT = 1.0 m.
Route = AlpaSim's RouteGenerator rebuilt at every logged anchor (rm_build.regen, full=): the label of decision k is rm_build.label of the full
polyline right after decision k (exactly the pool's route.npz label). Zero = first evaluator offroad step (metrics parquet, any timestamp);
k_z = the decision at or before it; the zero point = ego anchor interpolated to that time.

held, per offroad zero:
  H1  share of label samples (decision k_z) within 30 m of the zero point that are off the drivable surface (lab_off) / with margin < FIT
      (lab_nofit); the label point nearest the zero point: its margin
  H2  decisions k_z - 4 .. k_z: served plan poses at 1.0 / 1.5 / 2.0 s -> distance to that decision's label (hinge satisfied: all <= DZ)
      and margin; h2 = some decision has the hinge satisfied while a pose does not fit
  H3  served memory (rmem) points at +10 / +20 m vs the decision's rebuilt full route: point-to-polyline distance
  H4  class of the zero point: junction (inside an intersection: >= 2 lanes with headings >= 25 deg apart contain the route point, or a turning
      connector out of a branching lane), approach (junction within 30 m ahead on the route), jog (route lateral step >= 1.5 m within +-15 m,
      lane change), curve (route heading change >= 20 deg over +-20 m), straight
  lat_z = distance of the ego (zero point) to the route
pool, per row: label samples 0 .. 40 m ahead off / no-fit shares; the row's served plan (meta poses, the rollout policy's) at 1 - 2 s: beyond
the dead zone of the label (hinge active) and the label projection of an active pose does not fit while the pose does (the hinge pulls a
fitting pose toward a non-fitting target = "pull_off").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/route_mem/scripts"),
                str(R_ / "experiments/alpasim/scripts"), str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/anchor_offset/scripts"),
                str(R_ / "experiments/eyes_vs_head/scripts"), str(R_ / "experiments/alpasim_long/scripts")]
import rmem as RM  # noqa: E402

T0, DT = 17_000, 500_000
FIT, DZ, NEAR, PRE = 1.0, 1.0, 30.0, 4
CORN = [(4.049, 1.1485), (4.049, -1.1485), (-1.127, -1.1485), (-1.127, 1.1485)]   # Pacifica footprint around the rear axle (5.176 x 2.297 m)
IPL = [1, 2, 3]                                   # plan poses at 1.0 / 1.5 / 2.0 s (on_route W_T = 2)
I10, I20 = int(np.flatnonzero(RM.OFFS == 10.0)[0]), RM.I20


class Road:
    """Drivable surface of one scene near a set of points (local frame)."""

    def __init__(self, vm, pts, rad=120.0):
        import shapely
        import shapely.ops
        from eval.scorers.offroad import _get_lane_polygon, _repair_polygonal_geometry
        self.vm = vm
        pts = np.asarray(pts, float)
        lo, hi = pts.min(0) - rad, pts.max(0) + rad
        box = shapely.box(lo[0], lo[1], hi[0], hi[1])
        polys, self.lanes = [], []
        for ln in vm.lanes:
            c = np.asarray(ln.center.points)[:, :2]
            if ((c >= lo) & (c <= hi)).all(1).any():
                g = _get_lane_polygon(ln)
                polys.append(g)
                self.lanes.append((ln, g, c))
        cen = (lo + hi) / 2
        for ra in vm.get_road_areas_within(np.r_[cen, 0.0], float(np.hypot(*(hi - lo)) / 2)):
            polys.append(_repair_polygonal_geometry(vm.get_road_area_polygon_2d(ra.id)))
        U = _repair_polygonal_geometry(shapely.ops.unary_union(polys)).intersection(box)
        self.U = U.buffer(0.001).buffer(-0.001)
        self.B = self.U.boundary
        shapely.prepare(self.U)
        self.byid = {ln.id: (ln, g, c) for ln, g, c in self.lanes}
        self.tree = shapely.STRtree([g for _, g, _ in self.lanes])

    def margin(self, P):
        import shapely
        P = np.asarray(P, float).reshape(-1, 2)
        out = np.full(len(P), np.nan)
        ok = np.isfinite(P).all(1)
        if ok.any():
            pt = shapely.points(P[ok])
            d = shapely.distance(pt, self.B)
            out[ok] = np.where(shapely.contains_xy(self.U, P[ok, 0], P[ok, 1]), d, -d)
        return out

    def junction(self, p):
        """1 when p lies inside an intersection (module docstring)."""
        import shapely
        idx = self.tree.query(shapely.Point(p), predicate="within")
        hs = []
        for i in idx:
            ln, g, c = self.lanes[i]
            j = int(np.argmin(np.hypot(*(c - p).T)))
            a, b = c[max(j - 1, 0)], c[min(j + 1, len(c) - 1)]
            hs.append(np.arctan2(b[1] - a[1], b[0] - a[0]))
            turn = abs(np.degrees(_wrap(np.arctan2(*(c[-1] - c[-2])[::-1]) - np.arctan2(*(c[1] - c[0])[::-1])))) if len(c) > 2 else 0.0
            if turn >= 20 and any(len(self.byid[q][0].next_lanes) >= 2 for q in ln.prev_lanes if q in self.byid):
                return True
        hs = np.array(hs)
        return bool(len(hs) >= 2 and np.degrees(np.abs(_wrap(hs[:, None] - hs[None])).max()) >= 25)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def head(at, S, s):
    """Heading (rad) of the chord over +-2 m at station s, s clamped into the polyline (no zero-length chord at the ends)."""
    s = float(np.clip(s, 2.0, max(S[-1] - 2.0, 2.0)))
    return float(np.arctan2(*np.diff(at(np.array([s - 2, s + 2])), axis=0)[0][::-1]))


def stations(R):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]


def poly_dist(R, P):
    """Point-to-polyline distance of points P (n, 2) to R (m, 2) and the projection points."""
    P = np.asarray(P, float).reshape(-1, 2)
    R = R[np.isfinite(R).all(1)]
    a, b = R[:-1], R[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
    Q = a[None] + t[..., None] * ab[None]
    d = np.hypot(*(P[:, None] - Q).transpose(2, 0, 1))
    i = d.argmin(1)
    return d[np.arange(len(P)), i], Q[np.arange(len(P)), i]


def rclass(road, R, p):
    """Class of point p against route R (local): junction / approach / jog / curve / straight."""
    S = stations(R)
    _, q = poly_dist(R, p[None])
    s0 = S[int(np.argmin(np.hypot(*(R - q[0]).T)))]
    at = lambda s: np.c_[np.interp(s, S, R[:, 0]), np.interp(s, S, R[:, 1])]  # noqa: E731
    if road.junction(q[0]):
        return "junction"
    if any(road.junction(x) for x in at(np.arange(s0 + 2, s0 + 31, 2.0))):
        return "approach"
    W = at(np.arange(s0 - 15, s0 + 15.1, 1.0))
    c = W[-1] - W[0]
    n = np.array([-c[1], c[0]]) / max(np.hypot(*c), 1e-9)
    h = lambda s: head(at, S, s)  # noqa: E731
    dh = abs(np.degrees(_wrap(h(s0 + 20) - h(s0 - 20))))
    if np.ptp((W - W[0]) @ n) >= 1.5 and dh < 20:
        return "jog"
    return "curve" if dh >= 20 else "straight"


def first_offroad(D, scene):
    import polars as pl
    f = sorted(glob.glob(f"{D}/rollouts/{scene}/*/metrics.parquet"))
    if not f:
        return None, None
    m = pl.read_parquet(f[0]).filter(pl.col("name").is_in(["offroad", "collision_front", "collision_lateral", "dist_to_gt_trajectory"]))
    ser = {n: sorted(zip(x["timestamps_us"].cast(pl.Int64).to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}
    t_off = next((t for t, v in ser.get("offroad", []) if (v or 0) > 0), None)
    ev = [t for n in ("collision_front", "collision_lateral") for t, v in ser.get(n, []) if (v or 0) > 0]
    ev += [t for t, v in ser.get("dist_to_gt_trajectory", []) if (v or 0) >= 4.0]
    return t_off, (min(ev) if ev else None)


def held_one(job):
    label, D, scene, recs, need_zero = job
    from rm_build import regen, label as rlabel
    from ao_e1 import loader
    import ev as EV
    Fu = {}
    W, A = regen(scene, recs, EV.CFG, Fu)
    ks = [r["k"] for r in recs]
    tk = np.array([r["t0"] for r in recs], float)                           # t0 is already on the evaluator's clock (k = (t0 - T0) // DT)
    Rfin = Fu[ks[-1]]
    t_off, t_ev = first_offroad(D, scene) if need_zero else (None, None)
    out = dict(label=label, scene=scene, n_dec=len(recs))
    if t_off is None:
        return out, None
    i = int(np.clip(np.searchsorted(tk, t_off, side="right") - 1, 0, len(recs) - 1))
    kz = ks[i]
    anc = np.array([r["anchor"] for r in recs], float)
    z = np.array([np.interp(t_off, tk, anc[:, 0]), np.interp(t_off, tk, anc[:, 1])])
    zyaw = float(np.interp(t_off, tk, np.unwrap(anc[:, 2])))
    ds = loader(str(EV.CFG)).get_data_source(scene)
    road = Road(ds.map, np.r_[anc[:, :2], z[None]])
    Rk = Fu[kz]
    lab = RM.to_local(A[kz], rlabel(Rk, A[kz]))
    lab = lab[np.isfinite(lab).all(1)]
    sel = lab[np.hypot(*(lab - z).T) <= NEAR]
    mg = road.margin(sel)
    lat_z = float(poly_dist(Rk, z[None])[0][0])
    S_ = stations(Rk)
    q = poly_dist(Rk, z[None])[1][0]
    s0 = S_[int(np.argmin(np.hypot(*(Rk - q).T)))]
    at = lambda s: np.c_[np.interp(s, S_, Rk[:, 0]), np.interp(s, S_, Rk[:, 1])]  # noqa: E731
    hd = lambda s: head(at, S_, s)  # noqa: E731
    tz = np.array([np.cos(hd(s0)), np.sin(hd(s0))])
    side = float(np.sign(tz[0] * (z - q)[1] - tz[1] * (z - q)[0]))           # +1 = ego left of the route
    turn = float(np.degrees(_wrap(hd(s0 + 20) - hd(s0 - 20))))               # > 0 = the route turns left here
    fp = RM.to_local(np.r_[z, zyaw], np.array(CORN))
    import shapely
    out.update(side=side, turn20=turn, inside=bool(abs(turn) >= 20 and side * np.sign(turn) > 0), fp_off=bool(not road.U.covers(shapely.Polygon(fp))),
               fp_min_margin=float(road.margin(fp).min()))
    out.update(k_z=int(kz), t_off=int(t_off), off_first=bool(t_ev is None or t_off <= t_ev), z_margin=float(road.margin(z[None])[0]), lat_z=lat_z,
               lat_anchor=float(poly_dist(Rk, anc[i, None, :2])[0][0]), lab_n=int(len(sel)),
               lab_off=float((mg < 0).mean()) if len(sel) else np.nan, lab_nofit=float((mg < FIT).mean()) if len(sel) else np.nan,
               lab_min_margin=float(mg.min()) if len(sel) else np.nan,
               lab_near_margin=float(road.margin(poly_dist(Rk, z[None])[1])[0]))
    h2, plans = [], []
    for j in range(max(0, i - PRE), i + 1):
        r = recs[j]
        a = np.asarray(r["anchor"], float)
        P = RM.to_local(a, np.asarray(r["poses"], float)[:, :2])
        Lj = RM.to_local(a, rlabel(Fu[ks[j]], a))
        Lj = Lj[np.isfinite(Lj).all(1)]
        d = poly_dist(Lj, P[IPL])[0] if len(Lj) >= 2 else np.full(3, np.nan)
        mp = road.margin(P[IPL])
        h2.append(dict(k=int(ks[j]), dmax=float(np.nanmax(d)), fit_min=float(mp.min()), sat=bool(np.nanmax(d) <= DZ), nofit=bool(mp.min() < FIT),
                       lab_proj_margin=float(road.margin(poly_dist(Lj, P[IPL])[1]).min()) if len(Lj) >= 2 else np.nan))
        plans.append(P)
    out.update(h2_sat_nofit=any(x["sat"] and x["nofit"] for x in h2), h2_any_nofit=any(x["nofit"] for x in h2),
               h2_unsat_nofit=any((not x["sat"]) and x["nofit"] for x in h2), h2_dmax_med=float(np.median([x["dmax"] for x in h2])),
               h2_labproj_nofit=any(x["lab_proj_margin"] < FIT for x in h2), h2=json.dumps(h2))
    mem = None
    rm = recs[i].get("rmem")
    if rm is not None:
        Fm = np.asarray(rm, float).reshape(RM.N_PT, RM.NF)
        v = Fm[:, 0] > 0.5
        mem = RM.to_local(A[kz], Fm[v, 1:3] * 20.0)
        for nm, ii in (("m10", I10), ("m20", I20)):
            out[nm] = float(poly_dist(Rk, RM.to_local(A[kz], Fm[ii, 1:3] * 20.0))[0][0]) if v[ii] else np.nan
        out["mem_n"] = int(v.sum())
        out["mem_first"] = float(RM.OFFS[v][0]) if v.any() else np.nan     # nearest covered offset (m ahead of the ego's projection)
        out["mem_maxgap"] = float(poly_dist(Rk, mem)[0].max()) if v.any() else np.nan
    for nm, j in (("z", i), ("pre", max(i - PRE, 0))):                     # H5: memory coverage at the zero decision and 2 s before
        rj = recs[j].get("rmem")
        if rj is not None:
            vj = np.asarray(rj, float).reshape(RM.N_PT, RM.NF)[:, 0] > 0.5
            out[f"first_{nm}"] = float(RM.OFFS[vj][0]) if vj.any() else np.inf
            out[f"c20_{nm}"] = bool(vj[I20])
    out["cls"] = rclass(road, Rk, z)
    out["cls_ego"] = "junction" if road.junction(z) else out["cls"]
    import shapely
    clip = road.U.intersection(shapely.Point(z).buffer(70))
    fig = dict(z=z, anc=anc, kz=int(kz), i=i, route=Rfin, route_k=Rk, label=lab, mem=mem, plans=plans,
               road=[np.asarray(g.exterior.coords) for g in getattr(clip, "geoms", [clip]) if g.geom_type == "Polygon"],
               holes=[np.asarray(h.coords) for g in getattr(clip, "geoms", [clip]) if g.geom_type == "Polygon" for h in g.interiors], info=dict(out))
    return out, fig


def lat_one(job):
    """Decision-level lateral distance of the ego to the route (all decisions) for the pairing."""
    label, D, scene, recs, _ = job
    from rm_build import regen
    import ev as EV
    Fu = {}
    W, A = regen(scene, recs, EV.CFG, Fu)
    t_off, _ = first_offroad(D, scene)
    return [dict(label=label, scene=scene, k=int(r["k"]), lat=float(poly_dist(Fu[r["k"]], np.asarray(r["anchor"], float)[None, :2])[0][0]),
                 t_off=t_off if t_off is not None else -1) for r in recs]


def run_held(a):
    import pandas as pd
    from ao_e1 import drive_records
    out = Path(a.out)
    (out / "fig").mkdir(parents=True, exist_ok=True)
    Z, L = [], []
    for spec in a.runs:
        label, dirs = spec.split("=", 1)
        seen = set()
        for D in dirs.split(","):
            summ = {r["clipgt_id"]: r for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]}
            dr = drive_records(D)
            for sc, r in sorted(summ.items()):
                if sc in seen or sc not in dr:
                    continue
                seen.add(sc)
                m = r.get("metrics") or {}
                zero = r.get("score") == 0
                Z.append(dict(label=label, scene=sc, run=D, score=r.get("score"), offroad=float(m.get("offroad") or 0),
                              corridor=float(m.get("left_corridor_laterally") or 0), atf=float(m.get("collision_at_fault") or 0), zero=zero))
                L.append((label, D, sc, [{k: v for k, v in x.items() if k not in ("ms", "hist")} for x in dr[sc]], zero and (m.get("offroad") or 0) > 0))
    S = pd.DataFrame(Z)
    S.to_csv(out / "scenes.csv", index=False)
    sc_off = set(S[S.zero & (S.offroad > 0)].scene)
    jobs = [j for j in L if j[4]]
    lj = [j for j in L if j[2] in sc_off]
    print(f"{len(S)} rollouts, {len(jobs)} offroad zeros, {len(sc_off)} scenes, {len(lj)} lat jobs", flush=True)
    rows, lat = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for j, res in zip(jobs, ex.map(_safe, [(held_one, j) for j in jobs])):
            if isinstance(res, str):
                print("ERR", j[0], j[2], res[-300:], flush=True)
                continue
            r, fig = res
            rows.append(r)
            if fig is not None:
                pickle.dump(fig, open(out / "fig" / f"{j[0]}__{j[2]}.pkl", "wb"))
        for res in ex.map(_safe, [(lat_one, j) for j in lj]):
            if not isinstance(res, str):
                lat += res
    pd.DataFrame(rows).to_parquet(out / "held.parquet", index=False)
    pd.DataFrame(lat).to_parquet(out / "held_dec.parquet", index=False)
    print(pd.DataFrame(rows).groupby("label").agg(n=("scene", "size"), lab_off=("lab_off", "mean"), lat_z=("lat_z", "median")).to_string())


def _safe(fj):
    f, j = fj
    try:
        return f(j)
    except Exception as e:  # noqa: BLE001
        import traceback
        return f"{e!r} {traceback.format_exc()[-800:]}"


def pool_one(job):
    run, scene, idx, anchors, poses, polys = job
    from ao_e1 import loader
    import ev as EV
    from rm_build import LAB
    ds = loader(str(EV.CFG)).get_data_source(scene)
    pts = np.concatenate([np.asarray(anchors)[:, :2]] + [RM.to_local(a, p[np.isfinite(p).all(1)]) for a, p in zip(anchors, polys)])
    road = Road(ds.map, pts, rad=40.0)
    ahead = (LAB >= 0) & (LAB <= 40)
    out = []
    for i, a, P, Lr in zip(idx, anchors, poses, polys):
        ok = np.isfinite(Lr).all(1)
        r = dict(row=int(i), scene=scene, ok=bool(ok.sum() >= 2))
        if ok.sum() < 2:
            out.append(r)
            continue
        Lw = RM.to_local(a, Lr[ok])
        mg = road.margin(RM.to_local(a, Lr[ahead & ok]))
        Pp = RM.to_local(a, np.asarray(P, float)[IPL, :2])
        d, Q = poly_dist(Lw, Pp)
        mp, mq = road.margin(Pp), road.margin(Q)
        act = d > DZ
        Lr_ok = Lr[ok]                                                     # row frame: turn of the label over 0 .. 30 m, side of the active poses
        S_ = stations(Lr_ok)
        at = lambda q: np.c_[np.interp(q, S_, Lr_ok[:, 0]), np.interp(q, S_, Lr_ok[:, 1])]  # noqa: E731
        s_e = float(np.interp(0.0, LAB[ok], S_))
        hh = [head(at, S_, q) for q in (s_e, s_e + 30)]
        turn = float(np.degrees(_wrap(hh[1] - hh[0])))
        Pr = np.asarray(P, float)[IPL, :2]
        _, Qr = poly_dist(Lr_ok, Pr)
        tg = np.array([np.cos(hh[0]), np.sin(hh[0])])
        side = np.sign(tg[0] * (Pr - Qr)[:, 1] - tg[1] * (Pr - Qr)[:, 0])  # +1 = pose left of the label
        r.update(turn30=turn, outside=bool(abs(turn) >= 20 and act.any() and (side[act] * np.sign(turn) < 0).all()),
                 inside_act=bool(abs(turn) >= 20 and act.any() and (side[act] * np.sign(turn) > 0).all()))
        r.update(lab_off=float((mg < 0).mean()) if len(mg) else np.nan, lab_nofit=float((mg < FIT).mean()) if len(mg) else np.nan,
                 active=bool(act.any()), pull_off=bool((act & (mp >= FIT) & (mq < FIT)).any()), plan_nofit=bool((mp < FIT).any()),
                 dmax=float(d.max()), cls=rclass(road, Lw, np.asarray(a, float)[:2]) if a_cls(i) else "")
        out.append(r)
    return out


def a_cls(i):
    return i % 4 == 0                                  # class for a quarter of the rows (cost)


def run_pool(a):
    import pandas as pd
    D = Path(os.environ["DATA_DIR"])
    m = np.load(D / "runs/onpol_nuplan/rows" / a.name / "meta.npz", allow_pickle=True)
    z = np.load(D / "runs/op_parity/cache" / f"on1_{a.name}" / "route.npz")
    assert (z["tokens"] == m["names"]).all()
    by = {}
    for i, (rn, sc) in enumerate(zip(m["run"].tolist(), m["scene"].tolist())):
        by.setdefault((rn, sc), []).append(i)
    keys = sorted(by)[: a.limit or None]
    jobs = [(rn, sc, by[rn, sc], m["anchor"][by[rn, sc]], m["poses"][by[rn, sc]], z["poly"][by[rn, sc]]) for rn, sc in keys]
    print(f"{len(jobs)} rollouts, poses {m['poses'].shape}", flush=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(_safe, [(pool_one, j) for j in jobs]):
            if isinstance(res, str):
                print("ERR", res[-300:], flush=True)
            else:
                rows += res
    P = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    P.to_parquet(Path(a.out) / "pool.parquet", index=False)
    q = P[P.ok]
    print(f"rows {len(q)}: lab_off>0 {(q.lab_off > 0).mean():.3f}, lab_nofit>0 {(q.lab_nofit > 0).mean():.3f}, active {q.active.mean():.3f}, "
          f"pull_off {q.pull_off.mean():.3f}, plan_nofit {q.plan_nofit.mean():.3f}")


def run_fig(a):
    """BEV of chosen zeros: --pick LABEL__SCENE ... (fig/<pick>.pkl) -> --png-dir/offroad_<n>.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    od = Path(a.png_dir)
    od.mkdir(parents=True, exist_ok=True)
    for n, pk in enumerate(a.pick, 1):
        f = pickle.load(open(Path(a.out) / "fig" / f"{pk}.pkl", "rb"))
        z, I = f["z"], f["info"]
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        for g in f["road"]:
            ax.fill(g[:, 0], g[:, 1], color="0.88", lw=0)
        for h in f["holes"]:
            ax.fill(h[:, 0], h[:, 1], color="white", lw=0)
        ax.plot(*f["route"].T, color="#2b6cb0", lw=1.0, alpha=0.6, label="full route (end of rollout)")
        ax.plot(*f["label"].T, color="#2b6cb0", lw=2.2, ls="--", label=f"label at k_z = {f['kz']}")
        if f["mem"] is not None and len(f["mem"]):
            ax.plot(*f["mem"].T, "o", ms=3, color="#2f855a", label=f"memory at k_z (first +{I.get('mem_first', np.nan):.0f} m)")
        for j, P in enumerate(f["plans"]):
            last = j == len(f["plans"]) - 1
            ax.plot(*P.T, color="#c05621" if last else "#dd6b20", lw=2.0 if last else 0.8, alpha=1 if last else 0.5,
                    label="served plans k_z-4 .. k_z" if j == 0 else None)
        A = f["anc"]
        ax.plot(A[:, 0], A[:, 1], "-", color="k", lw=1.2, marker=".", ms=4, label="ego path (anchors, 2 Hz)")
        ax.plot(*z, "x", color="#c53030", ms=12, mew=3, label="first offroad")
        ax.set_xlim(z[0] - 40, z[0] + 40), ax.set_ylim(z[1] - 40, z[1] + 40), ax.set_aspect("equal")
        ax.set_title(f"{pk.split('__')[0]}  {pk.split('__')[1][:38]}\n{I['cls']}, route turn {I['turn20']:+.0f} deg, ego "
                     f"{'inside' if I['inside'] else 'outside / straight'}, lat to route {I['lat_z']:.2f} m", fontsize=8)
        ax.legend(fontsize=6.5, loc="lower left")
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(od / f"offroad_{n}.png", dpi=130)
        plt.close(fig)
        print(od / f"offroad_{n}.png", pk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("held")
    h.add_argument("--runs", nargs="+", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--workers", type=int, default=24)
    p = sub.add_parser("pool")
    p.add_argument("--name", default="L18RB0")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=24)
    g = sub.add_parser("fig")
    g.add_argument("--out", required=True)
    g.add_argument("--pick", nargs="+", required=True)
    g.add_argument("--png-dir", required=True)
    a = ap.parse_args()
    {"held": run_held, "pool": run_pool, "fig": run_fig}[a.cmd](a)
