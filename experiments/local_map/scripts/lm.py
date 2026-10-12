#!/usr/bin/env python3
"""LOCAL-MAP (plans/2026-10-12-local-map-prereg.md): would a rule-based local map have saved the frozen-token road probe of decision 275
at the road-exit zeros? Carry what was seen (the probe's ego-lane boundaries) or given (the route waypoints) 1-4 s earlier into the zero
window's frame with the egomotion the driver receives, and measure it against the map. No training, no GPU.

  read    (box, AlpaSim venv, CPU)  lm.py read [--workers N] [--limit N]
      per decision of the zero group (98 road-exit zero rollouts, decision 275's tok/zero rows) and the held group (decision 275's held
      re-run): the carried estimates L1 (probe), L2 (route), L3 (fused) for d in {0, 2, 4, 6, 8} and the newest-covering route, at
      D = 10 / 20 / 40 m; gates G-frame, G-carry, G-route; BEV payloads of the window decisions
      -> $DATA_DIR/runs/local_map/read/<ts>/ est_zero.parquet, est_held.parquet, gates.json, bev.pkl (+ copies of decision 275's labels)
  report  (Mac, .venv)  lm_report.py --read DIR   (gates G-sign, G-d0, the registered tables, verdict, figures)

Frame: rollout local frame; ego = rig = rear axle, x forward, y LEFT (ev.py / rp.py). A carried estimate at D is the first crossing (in point
order, linear, no extrapolation) of the carried polyline with the line x = cam_x + D of the current decision.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "read":                        # one BLAS / polars thread per worker
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/road_probe/scripts")]
import rp as RP  # noqa: E402  (imports ev.py and sets the AlpaSim / driver paths)

EV = RP.EV
DATA = EV.DATA
OUT = DATA / "runs/local_map"
RPW = RP.W
T0, DT = 17_000, 500_000
DS = (10, 20, 40)
DLIST = (0, 2, 4, 6, 8)
RIG_TO_BOX = 1.461                                                      # rollout_spec.vehicle.rig_to_bounding_box.x


def yaw_q(q):
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def first_cross(Pe, xs):
    """First crossing (in point order) of the ego-frame polyline Pe (m, 2) with x = xs -> y; nan if none (no extrapolation)."""
    if Pe is None or len(Pe) < 2:
        return np.nan
    a, b = Pe[:-1, 0] - xs, Pe[1:, 0] - xs
    m = np.flatnonzero((a * b <= 0) & (np.abs(Pe[1:, 0] - Pe[:-1, 0]) > 1e-9))
    if not len(m):
        return np.nan
    i = m[0]
    return float(Pe[i, 1] + (xs - Pe[i, 0]) / (Pe[i + 1, 0] - Pe[i, 0]) * (Pe[i + 1, 1] - Pe[i, 1]))


def seg_dist(P, q):
    """Distance of point q to polyline P and whether the nearest point is interior (not clamped at an end)."""
    d = P[1:] - P[:-1]
    t = ((q - P[:-1]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12)
    tc = np.clip(t, 0, 1)
    D = np.hypot(*(P[:-1] + tc[:, None] * d - q).T)
    i = int(np.argmin(D))
    inner = not ((i == 0 and t[i] < 0) or (i == len(d) - 1 and t[i] > 1))
    return float(D[i]), inner


def carry(a_from, a_to, X):
    """Ego-frame points of pose a_from -> ego frame of a_to (rigid, through the local frame)."""
    return EV.in_ego(a_to, EV.to_local(a_from, X))


def estimates(rows, routes_local):
    """rows: dicts (kt, anchor, cam, P (13,), poses) of one rollout; routes_local: {kt: (m, 2) local-frame waypoints}.
    -> per-row dicts of carried estimates (columns described in the module docstring)."""
    by = {r["kt"]: r for r in rows}
    out = []
    for r in rows:
        k, a, cam = r["kt"], r["anchor"], r["cam"]
        rec = dict(row=r["row"], kt=k)
        xsD = {D: cam[0] + D for D in DS}
        # plan (served, ego frame of k) from the rear axle
        pl = np.r_[[[0.0, 0.0]], np.asarray(r["poses"], float)[:, :2]]
        for D in DS:
            rec[f"plan{D}"] = first_cross(pl, xsD[D])
        # L1: carried probe polylines (any d on demand; columns for DLIST)
        l1 = {}

        def probe(d, s, D, k=k, a=a):
            if (d, s, D) not in l1:
                src, j0 = by.get(k - d), "LR".index(s)
                pts = None if src is None else carry(src["anchor"], a, np.c_[src["cam"][0] + np.array(DS, float), src["P"][[j0, 4 + j0, 8 + j0]]])
                l1[d, s, D] = first_cross(pts, xsD[D])
            return l1[d, s, D]
        for d in DLIST:
            for s in "LR":
                for D in DS:
                    rec[f"L1_{d}_{s}{D}"] = probe(d, s, D)
        # L2: carried routes (each d; and the newest covering one, any d >= 0)
        rc = {}
        for d in range(0, k + 1):
            if (k - d) in routes_local and len(routes_local[k - d]) >= 2:
                rc[d] = EV.in_ego(a, routes_local[k - d])
        for d in DLIST:
            for D in DS:
                P = rc.get(d)
                rec[f"L2_{d}_c{D}"] = first_cross(P, xsD[D])
                rec[f"L2_{d}_P{D}"] = np.nan                       # filled by the reader with the point-to-polyline distance
        for D in DS:
            v, dn = np.nan, np.nan
            for d in sorted(rc):
                v = first_cross(rc[d], xsD[D])
                if np.isfinite(v):
                    dn = d
                    break
            rec[f"L2_n_c{D}"], rec[f"L2_n_d{D}"] = v, dn
            v, dn = np.nan, np.nan                                     # exploratory (post-read): newest covering route that starts at or
            for d in sorted(rc):                                       # behind the line, so a crossing on a returning leg is not taken
                if rc[d][0, 0] <= xsD[D]:
                    v = first_cross(rc[d], xsD[D])
                    if np.isfinite(v):
                        dn = d
                        break
            rec[f"L2_b_c{D}"], rec[f"L2_b_d{D}"] = v, dn
        for d in DLIST:
            rec[f"x0_{d}"] = rc[d][0, 0] if d in rc else np.nan        # carried route start (ego x) and distance travelled since k - d
            rec[f"trav_{d}"] = float(np.hypot(*(by[k - d]["anchor"][:2] - a[:2]))) if (k - d) in by else np.nan
        # L3: route centre + probe width (same d, else the current probe)
        for d in (*DLIST, "n", "b"):
            for D in DS:
                c = rec[f"L2_{d}_c{D}"]
                dw = rec[f"L2_{d}_d{D}"] if d in ("n", "b") else d
                w = probe(int(dw), "L", D) - probe(int(dw), "R", D) if np.isfinite(dw) else np.nan
                if not np.isfinite(w):
                    w = probe(0, "L", D) - probe(0, "R", D)
                rec[f"L3_{d}_L{D}"], rec[f"L3_{d}_R{D}"] = c + w / 2, c - w / 2
        rec["_rc"] = rc
        out.append(rec)
    return out


def route_regen(ds, rows):
    """AlpaSim's RouteGenerator (MAP, start offset 40 m) at each decision's anchor -> {kt: (m, 2) local waypoints} (ao_e1.py's way)."""
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    tr = ds.rig.trajectory
    P, Tt = np.asarray(tr.positions, float), np.asarray(tr.timestamps_us, float)
    g = RouteGenerator.create(P, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
    out, rig = {}, {}
    for r in sorted(rows, key=lambda r: r["kt"]):
        x, y, yaw = r["anchor"]
        pose = Pose(np.array([x, y, np.interp(r["t0"], Tt, P[:, 2])]), np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]))
        wp = np.asarray(g.generate_route(int(r["t0"]), pose).waypoints, float)[:, :2]
        wp = wp[np.isfinite(wp).all(1)]
        rig[r["kt"]] = wp
        out[r["kt"]] = EV.to_local(r["anchor"], wp) if len(wp) else np.zeros((0, 2))
    return out, rig, np.asarray(g.route_polyline_in_local.points, float)[:, :2]


def msgs_of(label, scene):
    """Stored driver messages -> ({ts: (x, y, yaw) local rig pose}, {route ts: (20, 2) rig-frame waypoints with NaN})."""
    import sh30_driver as Dv
    pb = Dv.egodriver_pb2
    ego, routes = {}, {}
    for kind, raw in pickle.load(open(EV.OUT / "msgs" / label / f"{scene}.pkl", "rb")):
        if kind == "driver_ego_trajectory":
            for p in pb.RolloutEgoTrajectory.FromString(raw).trajectory.poses:
                ego[int(p.timestamp_us)] = (p.pose.vec.x, p.pose.vec.y, yaw_q(p.pose.quat))
        elif kind == "route_request":
            r = pb.RouteRequest.FromString(raw).route
            routes[int(r.timestamp_us)] = np.array([[w.x, w.y] for w in r.waypoints], float).reshape(-1, 2)
    return ego, routes


def unit(job):
    """(group, label, scene, rows, window ks, route0 by kt) -> dict(est, gates, bev)."""
    g, label, scene, rows, win, r0 = job
    from ao_e1 import loader
    ds = loader(str(EV.CFG)).get_data_source(scene)
    gates = dict(frame_anchor=[], frame_sim=[], carry=[], carry_mirror=[], carry_chord=[], route_wp=[], route0=[])
    regen, regen_rig, dense = route_regen(ds, rows)
    if g == "zero":
        import c1_extract as X
        ego, msg_routes = msgs_of(label, scene)
        routes = {}
        for ts, wp in msg_routes.items():
            kt = int(round((ts - T0) / DT))
            wp = wp[np.isfinite(wp).all(1)]
            routes[kt] = EV.to_local(np.asarray(ego[ts], float), wp) if len(wp) else np.zeros((0, 2))
            if kt in regen_rig and len(wp) and len(regen_rig[kt]) == len(wp):       # G-route on the zero group: regenerated vs received
                gates["route_wp"] += np.hypot(*(regen_rig[kt] - wp).T).tolist()
            elif kt in regen_rig:
                gates["route_wp"] += [np.inf] * max(len(wp), 1)
        _, o = X.one_log((str(EV.RUNS[label]), scene))
        E = o["actors"]["EGO"]
        for r in rows:
            p = np.asarray(ego[r["t0"]], float)
            gates["frame_anchor"].append(float(np.hypot(*(p[:2] - r["anchor"][:2]))))
            i = int(np.argmin(np.abs(E[:, 0] - r["t0"])))
            if abs(E[i, 0] - r["t0"]) <= 1:
                q = E[i, 1:3] - RIG_TO_BOX * np.array([np.cos(E[i, 3]), np.sin(E[i, 3])])
                gates["frame_sim"].append(float(np.hypot(*(q - p[:2]))))
            else:
                gates["frame_sim"].append(np.inf)
        # G-carry (amended before the read, prereg section 7): the route received at k-1 carried into k against the map route polyline
        # (AlpaSim's dense lane-centre route) in the frame of k, interior points only; mirrored = headings negated. The registered first
        # form (against the 4.2 m chords of the route received at k) is kept as carry_chord.
        by = {r["kt"]: r for r in rows}
        for r in rows:
            k = r["kt"]
            if k - 1 in by and k - 1 in routes and len(routes[k - 1]):
                Dn = EV.in_ego(r["anchor"], dense)
                for mir, key in ((False, "carry"), (True, "carry_mirror")):
                    a1, a0 = np.array(by[k - 1]["anchor"], float), np.array(r["anchor"], float)
                    if mir:
                        a1[2], a0[2] = -a1[2], -a0[2]
                    C = carry(a1, a0, EV.in_ego(by[k - 1]["anchor"], routes[k - 1]))
                    for q in C:
                        dd, inner = seg_dist(Dn, q)
                        if inner:
                            gates[key].append(dd)
                if k in routes and len(routes[k]) >= 2:
                    Pk = EV.in_ego(r["anchor"], routes[k])
                    for q in EV.in_ego(r["anchor"], routes[k - 1]):
                        dd, inner = seg_dist(Pk, q)
                        if inner:
                            gates["carry_chord"].append(dd)
    else:
        routes = regen
        for r in rows:
            if r["kt"] in r0 and r0[r["kt"]] is not None:
                wp = regen_rig[r["kt"]]
                gates["route0"].append(float(np.hypot(*(wp[0] - np.asarray(r0[r["kt"]], float)))) if len(wp) else np.inf)
    est = estimates(rows, routes)
    by = {r["kt"]: r for r in rows}
    for e in est:                                                      # L2 secondary: point-to-polyline distance from the map centre point
        r = by[e["kt"]]
        for D in DS:
            c = r["c"][D]
            if np.isfinite(c):
                for d in DLIST:
                    P = e["_rc"].get(d)
                    if P is not None and len(P) >= 2:
                        e[f"L2_{d}_P{D}"] = seg_dist(P, np.array([r["cam"][0] + D, c]))[0]
    bev = {}
    if g == "zero" and win:
        tr = ds.rig.trajectory
        Pg, Yg = np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
        LP = np.vstack([Pg, Pg[-1] + np.outer(np.linspace(2.0, 60.0, 30), [np.cos(Yg[-1]), np.sin(Yg[-1])])])
        A = np.array([r["anchor"] for r in rows])
        G = EV.Geo(ds, (np.vstack([Pg, A[:, :2]]).min(0) - 90, np.vstack([Pg, A[:, :2]]).max(0) + 90))
        box = lambda p: ((np.abs(p[:, 1]) < 30) & (p[:, 0] > -15) & (p[:, 0] < 70)).any()  # noqa: E731
        for e in est:
            r = by[e["kt"]]
            if r["k"] not in win:
                continue
            a = r["anchor"]
            Lg = EV.in_ego(a, LP)
            probe = {}
            for d in DLIST:
                src = by.get(e["kt"] - d)
                if src is not None:
                    probe[d] = {s: carry(src["anchor"], a, np.c_[src["cam"][0] + np.array(DS, float), [src["P"][j], src["P"][4 + j], src["P"][8 + j]]])
                                for s, j in (("L", 0), ("R", 1))}
            bev[(label, scene, r["k"])] = dict(edges=[p.astype(np.float32) for p in (EV.in_ego(a, q) for q in G.edges) if box(p)],
                                                road=[p.astype(np.float32) for p in (EV.in_ego(a, q) for q in G.road) if box(p)],
                                                log=Lg[(Lg[:, 0] > -15) & (Lg[:, 0] < 70) & (np.abs(Lg[:, 1]) < 30)].astype(np.float32),
                                                plan=np.asarray(r["poses"], np.float32), cam=r["cam"], probe=probe,
                                                route={d: e["_rc"][d].astype(np.float32) for d in e["_rc"] if d in DLIST or d == e.get("L2_n_d20")})
    for e in est:
        e.pop("_rc")
        e.update(group=g, tag=label, scene=scene, k=by[e["kt"]]["k"])
    return dict(est=est, gates=gates, bev=bev)


def read(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    held = splits.load("alpasim_nuplan/onpol-held")
    with Run("local_map", "read", config=vars(a)) as run:
        run.use_split(held)
        P = {g: 0.5 * (np.load(RPW / "pred/P-p1000-s0.npz")[g] + np.load(RPW / "pred/P-p1000-s1.npz")[g]) for g in ("zero", "held")}
        jobs = []
        ev = RP.zero_events()
        win = {}
        for e in ev:
            win.setdefault((e["group"], e["scene"]), set()).update(range(max(0, int(e["k_ev"]) - 4), int(e["k_ev"]) + 1))
        r0 = {}
        for rd in sorted(RPW.glob("roll/c*/run")):
            for scene, (n, rr) in RP.sessions(rd).items():
                r0[scene] = {int(round((int(x["t0"]) - T0) / DT)): x.get("route0") for x in rr}
        for g in ("zero", "held"):
            tb = dict(np.load(RPW / f"tok/{g}/tab.npz"))
            L = pd.read_parquet(RPW / f"labels/{g}.parquet")
            assert (L.row.values == np.arange(len(tb["k"]))).all()
            lab = tb["label"] if "label" in tb else np.full(len(tb["k"]), RP.HELD_TAG)
            by = {}
            for i in range(len(tb["k"])):
                kt = int(round((int(tb["t0"][i]) - T0) / DT))
                c = {D: 0.5 * (L[f"L{D}"].iloc[i] + L[f"R{D}"].iloc[i]) for D in DS}
                by.setdefault((str(lab[i]), str(tb["scene"][i])), []).append(dict(row=i, k=int(tb["k"][i]), kt=kt, t0=int(tb["t0"][i]), anchor=tb["anchor"][i].astype(float),
                                                                               cam=tb["cam"][i].astype(float), poses=tb["poses"][i], P=P[g][i, :12].astype(float), c=c))
            keys = sorted(by)[: a.limit or None]
            jobs += [(g, lb, sc, by[lb, sc], win.get((lb, sc), set()) if g == "zero" else set(), r0.get(sc, {}) if g == "held" else {}) for lb, sc in keys]
        run.info(f"{len(jobs)} rollouts")
        res = par.pmap(unit, jobs, workers=a.workers or n_cpus(), run=run, desc="rollouts")
        out = Path(run.dir)
        vals = [v for v in res.values if v]
        for g in ("zero", "held"):
            D = pd.DataFrame([e for v, j in zip(res.values, jobs) if v and j[0] == g for e in v["est"]]).sort_values("row").reset_index(drop=True)
            D.to_parquet(out / f"est_{g}.parquet", index=False)
            shutil.copyfile(RPW / f"labels/{g}.parquet", out / f"labels_{g}.parquet")
        gl = {k: np.array([x for v in vals for x in v["gates"][k]], float) for k in vals[0]["gates"]}
        G = dict(rollouts=len(jobs), errors=len(res.errors), err_first=[res.errors[i].strip().splitlines()[-1][:300] for i in sorted(res.errors)[:5]])
        fa, fs = gl["frame_anchor"], gl["frame_sim"]
        G["G_frame"] = dict(decisions=len(fa), anchor_share=float((fa < 0.01).mean()), anchor_max=float(fa.max()), sim_share=float((fs < 0.05).mean()),
                            sim_median=float(np.median(fs)), sim_max=float(fs.max()))
        G["G_frame"]["pass"] = bool(G["G_frame"]["anchor_share"] >= 0.99 and G["G_frame"]["sim_share"] >= 0.99)
        c, cm = gl["carry"], gl["carry_mirror"]
        cc = gl["carry_chord"]
        G["G_carry"] = dict(points=len(c), median=float(np.median(c)), p95=float(np.quantile(c, 0.95)), mirror_median=float(np.median(cm)),
                            chord_median=float(np.median(cc)), chord_p95=float(np.quantile(cc, 0.95)))
        G["G_carry"]["pass"] = bool(G["G_carry"]["median"] < 0.05 and G["G_carry"]["p95"] < 0.20)
        G["G_sign_mirror"] = dict(ratio=float(np.median(cm) / max(np.median(c), 1e-9)))
        G["G_sign_mirror"]["pass"] = bool(G["G_sign_mirror"]["ratio"] > 10)
        rw, r0g = gl["route_wp"], gl["route0"]
        G["G_route"] = dict(zero_points=len(rw), zero_share=float((rw <= 0.05).mean()), held_decisions=len(r0g), held_share=float((r0g <= 0.05).mean()),
                            held_median=float(np.median(r0g)))
        G["G_route"]["pass"] = bool(G["G_route"]["zero_share"] >= 0.99 and G["G_route"]["held_share"] >= 0.99)
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        bev = {}
        for v in vals:
            bev.update(v["bev"])
        pickle.dump(bev, open(out / "bev.pkl", "wb"), protocol=4)
        run.summary.update({k: v for k, v in G.items() if k.startswith("G_")})
        run.info(json.dumps(G, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("read")
    p.add_argument("--workers", type=int, default=0), p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dict(read=read)[a.cmd](a)
