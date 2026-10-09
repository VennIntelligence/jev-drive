#!/usr/bin/env python3
"""BODY1 diagnosis of the baseline zeros, analysis side (results/zeros_diagnosis.md). Input: bd1_diag.py's extraction and replay under
$DATA_DIR/runs/body1/diag/. Privileged truth, for the diagnosis only: the simulator's object boxes at matching times (swv1_lib.Rollout.sweep),
the simulator's own map (road areas + lanes of c1_extract.py map, the polygons the offroad scorer reads) and the logged ego path (the corridor).

  truth   per decision and per candidate (bd1_diag.CAND): agent contact of the 4 s sweep, footprint leaving the road union, leaving the 4 m
          corridor; the served plan's margins, the ego state against the log, the camera's view of the first contact, the NAVSIM drivable
          raster of the scene's navtest token registered to the rollout frame -> dec.csv, cand.npz, scenes.csv
  tables  the read: classes per zero, head recall and AUC on these decisions, ceilings per candidate family, states against the row set
          -> results/diag/*.csv, summary.json
  figs    figs/diag/*.png (class matrix, states, logit against truth, BEV strips)

Truth conventions. Agent: any box intersection at matching times over the plan's 41 samples; rear = the object's centre is behind the ego's
rear bumper at first contact (excluded, as the row labels do). Road: the scorer's rule for nuPlan maps (no road-edge layer): the ego box must
be covered by the union of the road areas and lanes; depth = the largest distance of a box corner from that union (0 when covered).
Corridor: the scorer's lateral distance of the box centre from the logged path >= 4 m. Head thresholds: serve_body.THR_RP (Amendment 3).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402

import bd1_diag as G  # noqa: E402
import col1_lib as CL  # noqa: E402
import swv1_lib as S  # noqa: E402

DATA, OUT = G.DATA, G.OUT
RES = _H.parents[1] / "results/diag"
FIG = _H.parents[1] / "figs/diag"
FA, FB, CA, CB = 0.3999, 1.8308, -1.7170, -0.5145          # serve_body.THR_RP
R_CAP = 0.10                                                # serve_body.R_CAP: a ramp is eligible while |a| <= R_CAP x the plan's 4 s arc
CORR = 4.0                                                  # eval config max_dist_to_gt_trajectory
BAND = 0.20                                                 # Amendment 1: the head's boundary positive needs depth > 0.20 m
CAM_X, CAM_NEAR, CAM_HALF = 1.79, 4.1, np.radians(31.0)     # CAM_F0 ahead of the rear axle; nearest ground point in the image; half horizontal FOV
REASON = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor"}
NC = len(G.CAND)
LAT, SLW, BOTH = slice(1, 11), slice(11, 13), slice(13, 33)


# ---------------------------------------------------------------- geometry
def road_union(mp):
    P = [shapely.Polygon(a["ext"], [h for h in a["holes"] if len(h) >= 3]) for a in mp["areas"] if len(a["ext"]) >= 3]
    P += [shapely.Polygon(np.r_[ln["left"], ln["right"][::-1]]) for ln in mp["lanes"] if ln["left"] is not None and ln["right"] is not None and len(ln["left"]) + len(ln["right"]) >= 3]
    U = shapely.unary_union(shapely.make_valid(np.array(P, object))).buffer(0.02)      # 2 cm: closes the slivers between adjacent map polygons
    shapely.prepare(U)
    return U


def corner_pts(c, L, W):
    """Centre poses (..., 3) -> corner coordinates (..., 4, 2)."""
    c = np.asarray(c, float)
    lx, ly = np.array([1, 1, -1, -1.0]) * L / 2, np.array([1, -1, -1, 1.0]) * W / 2
    cs, sn = np.cos(c[..., 2:3]), np.sin(c[..., 2:3])
    return np.stack([c[..., 0:1] + lx * cs - ly * sn, c[..., 1:2] + lx * sn + ly * cs], -1)


def road_depth(U, c, L, W):
    """Centre poses (..., 3) -> depth outside the road union (..., ) in m, 0 when the box is covered."""
    pts = corner_pts(c, L, W)
    poly = shapely.polygons(pts)
    out = ~shapely.covers(U, poly)
    d = np.zeros(poly.shape)
    if out.any():
        d[out] = np.maximum(shapely.distance(shapely.points(pts[out]), U).max(-1), 1e-4)
    return d


class Corridor:
    """The scorer's `_lateral_distance_to_gt` on the logged ego centre path: the distance to the path, and past its end the distance to its
    straight continuation (the scorer measures the component perpendicular to the final heading there)."""

    def __init__(self, gt_c):
        xy = gt_c[:, 1:3]
        seg = np.diff(xy, axis=0)
        j = np.flatnonzero(np.hypot(*seg.T) > 1e-3)
        self.len = float(np.hypot(*seg.T).sum())
        ext = [xy[-1] + 80.0 * seg[j[-1]] / np.hypot(*seg[j[-1]])] if len(j) else []
        self.ls = shapely.LineString(np.r_[xy, ext]) if len(j) else shapely.Point(xy[0])

    def __call__(self, xy):
        return shapely.distance(shapely.points(np.atleast_2d(xy)), self.ls)


def path_state(r, rig):
    """Rig pose (3,) -> signed lateral offset from the logged path (left +), arc position, heading minus the path's heading (rad)."""
    lat, s = r.lat(rig[None, :2])[0]
    seg = np.diff(r.path, axis=0)
    cs = np.r_[0, np.cumsum(np.hypot(*seg.T))]
    i = int(np.clip(np.searchsorted(cs, s, "right") - 1, 0, len(seg) - 1))
    return float(lat), float(s), float(CL.wrap(rig[2] - np.arctan2(seg[i, 1], seg[i, 0])))


def visible(anchor, p):
    """World point p seen from the front camera at rig pose anchor: in the horizontal FOV and beyond the nearest ground row."""
    q = CL.into(np.asarray(anchor, float), np.asarray(p, float)[None])[0]
    xc = q[0] - CAM_X
    return bool(xc > CAM_NEAR and abs(np.arctan2(q[1], xc)) < CAM_HALF)


def register(r, fut):
    """The navtest token's rear-axle frame at t0 in the rollout frame: the logged pose whose following path matches the token's logged future.
    -> (rig pose (3,), mean residual m, seconds after the scene start) or None."""
    g, best = r.gt_rig, None
    for i in range(len(g)):
        t = g[i, 0] + np.arange(1, 9) * 0.5e6
        ok = t <= g[-1, 0]
        if ok.sum() < 3:
            break
        e = float(np.hypot(*(CL.into(g[i, 1:4], CL.interp(g, t[ok])[:, :2]) - fut[ok, :2]).T).mean())
        if best is None or e < best[1]:
            best = (g[i, 1:4].copy(), e, (g[i, 0] - g[0, 0]) * 1e-6)
    return best


def raster_margin(sdf, frame, rig, x0=-8.0, y0=-24.0, res=0.5):
    """Rig poses (n, 3) in the rollout frame -> min footprint-corner SDF of the token's raster (n,), nan where a corner is off the raster."""
    c = S.centre(rig, 1.461)
    pts = corner_pts(c, 5.176, 2.297).reshape(-1, 2)
    q = CL.into(frame, pts)
    fx, fy = (q[:, 0] - x0) / res, (q[:, 1] - y0) / res
    ins = (fx >= 0) & (fx <= sdf.shape[0] - 1) & (fy >= 0) & (fy <= sdf.shape[1] - 1)
    i0, j0 = np.clip(np.floor(fx).astype(int), 0, sdf.shape[0] - 2), np.clip(np.floor(fy).astype(int), 0, sdf.shape[1] - 2)
    a, b = np.clip(fx - i0, 0, 1), np.clip(fy - j0, 0, 1)
    v = (sdf[i0, j0] * (1 - a) * (1 - b) + sdf[i0 + 1, j0] * a * (1 - b) + sdf[i0, j0 + 1] * (1 - a) * b + sdf[i0 + 1, j0 + 1] * a * b).astype(float)
    v[~ins] = np.nan
    return v.reshape(-1, 4).min(1)


# ---------------------------------------------------------------- truth
def one_scene(st, kind, row, o, mp, Z, tok):
    scene = row["scene"] if kind == "zero" else row["clean"]
    why = REASON[row["reason"]] if kind == "zero" else "clean"
    r = S.Rollout(o, "nuplan", st, scene, "C" if why == "collision" else "N")
    U = road_union(mp)
    cor = Corridor(o["logged"][0]["traj"])
    flag = {"collision": "collision_at_fault", "offroad": "offroad", "corridor": "left_corridor_laterally"}.get(why)
    t_ev = CL.first_event(o, flag) if flag else None
    m = Z["scene"] == scene
    order = np.argsort(Z["k"][m])
    P8, za, zb, ego_v = Z["poses"][m][order], Z["za"][m][order], Z["zb"][m][order], Z["ego"][m][order]
    rec = o["rec"]
    assert len(rec) == len(P8) == len(r.dec), (scene, len(rec), len(P8), len(r.dec))
    d_rep = float(np.abs(P8 - np.array([x["poses"] for x in rec])).max())
    struck = S.struck(r) if why == "collision" else None
    # the scorer's series on the executed ego against this file's truth (2 Hz)
    ego = r.ego
    ex_depth = road_depth(U, ego[:, 1:4], r.Le, r.We)
    ex_lat = cor(ego[:, 1:3])
    mo, ml = CL.metric(o, "offroad"), CL.metric(o, "lateral_dist_to_gt_trajectory")
    chk = dict(off_n=len(mo), off_agree=int(((ex_depth > 0) == (np.interp(ego[:, 0], mo[:, 0], mo[:, 1]) > 0.5)).sum()), lat_err=float(np.abs(ex_lat[2:] - np.interp(ego[2:, 0], ml[:, 0], ml[:, 1])).max()))
    reg = register(r, tok["fut"]) if tok is not None else None
    sdf = tok["sdf"].astype(np.float32) if (reg is not None and reg[1] < 0.5 and tok["ok"]) else None
    D, C = [], dict(a_hit=[], b_depth=[], c_out=[], clear=[])
    for k, x in enumerate(rec):
        now, anchor = r.now[k], np.array(x["anchor"], float)
        Q = G.candidates(P8[k])
        W = np.stack([S.dense(anchor, q) for q in Q])                               # (33, 41, 3) rig, world
        cen = np.stack([S.centre(w, r.off) for w in W])
        depth = road_depth(U, cen, r.Le, r.We)                                      # (33, 41)
        clat = cor(cen[..., :2].reshape(-1, 2)).reshape(NC, 41)
        a_hit, a_first, a_obj, a_clr, hit_st, st_first = np.zeros(NC, bool), np.full(NC, -1), [""] * NC, np.full(NC, 99.0), np.zeros(NC, bool), np.full(NC, -1)
        for c in range(NC):
            for oid, (clr, j) in r.sweep(k, W[c]).items():
                a_clr[c] = min(a_clr[c], clr)
                if j < 0:
                    continue
                p = r.obj_pose(oid, now + j * 1e5)[0][0]
                if CL.into(cen[c, j], p[None, :2])[0, 0] < -r.Le / 2:                 # behind the rear bumper at first contact: rear-end
                    continue
                if oid == struck:
                    hit_st[c], st_first[c] = True, j
                if not a_hit[c] or j < a_first[c]:
                    a_hit[c], a_first[c], a_obj[c] = True, j, oid
        b_out, c_out = depth.max(1) > 0, clat.max(1) >= CORR
        clear = ~a_hit & ~b_out & ~c_out
        s4 = float(np.hypot(*np.diff(np.r_[np.zeros((1, 2)), P8[k][:, :2]], axis=0).T).sum())
        elig = np.abs(G.A_LAT) <= R_CAP * s4
        head_clear = (za[k, LAT] < CA) & (zb[k, LAT] < CB) & elig
        lat, s_arc, herr = path_state(r, anchor)
        gt_now = CL.interp(r.gt_rig, now)[0]
        jb = int(np.argmax(depth[0] > 0)) if b_out[0] else -1
        b_pt = None
        if jb >= 0:
            pts = corner_pts(cen[0, jb], r.Le, r.We)
            b_pt = pts[int(np.argmax(shapely.distance(shapely.points(pts), U)))]
        st_pt = r.obj_pose(struck, now)[0][0, :2] if struck else None
        rm = raster_margin(sdf, reg[0], W[0]) if sdf is not None else np.full(41, np.nan)
        trk = float(np.hypot(*(anchor[:2] - r.plan[k - 1][5, :2]))) if k else np.nan
        D.append(dict(set=st, seed=int(st[-1]), kind=kind, reason=why, scene=scene, k=k, t=round((now - r.T0) * 1e-6, 3), t_ev=None if t_ev is None else round((t_ev - r.T0) * 1e-6, 3),
                      pre=int(t_ev is None or now < t_ev), v0=round(float(r.ego_v(now)[0]), 3), lat_off=round(lat, 3), head_err_deg=round(float(np.degrees(herr)), 2),
                      ahead_m=round(s_arc - r.lat(gt_now[None, :2])[0, 1], 2), plan_arc=round(s4, 2), n_slots=int(Z["n_slots"][m][order][k]), cmd=int(Z["cmd"][m][order][k]),
                      za=round(float(za[k, 0]), 3), zb=round(float(zb[k, 0]), 3), flag_a=int(za[k, 0] >= FA), flag_b=int(zb[k, 0] >= FB),
                      a_hit=int(a_hit[0]), a_first_s=round(0.1 * a_first[0], 1) if a_hit[0] else "", a_clr=round(float(a_clr[0]), 2), a_obj=a_obj[0],
                      hit_struck=int(hit_st[0]), struck_first_s=round(0.1 * st_first[0], 1) if hit_st[0] else "",
                      struck_vis=int(visible(anchor, st_pt)) if struck else "", struck_v=round(float(r.obj_v(struck, now)[0]), 2) if struck else "",
                      b_out=int(b_out[0]), b_first_s=round(0.1 * jb, 1) if jb >= 0 else "", b_depth=round(float(depth[0].max()), 3), b_out_scene=int((depth[0][now + S.TAU * 1e6 <= r.T1 + 1e3] > 0).any()),
                      b_vis=int(visible(anchor, b_pt)) if b_pt is not None else "", r_margin=round(float(np.nanmin(rm)), 3) if np.isfinite(rm).any() else "",
                      r_margin_at_exit=round(float(rm[jb]), 3) if jb >= 0 and np.isfinite(rm[jb]) else "",
                      c_out=int(c_out[0]), c_first_s=round(0.1 * int(np.argmax(clat[0] >= CORR)), 1) if c_out[0] else "", c_max=round(float(clat[0].max()), 2),
                      track_err=round(trk, 3) if k else "", replay_diff=round(d_rep, 5),
                      n_elig=int(elig.sum()), clear_plan=int(clear[0]), clear_lat=int(clear[LAT].any()), clear_lat_elig=int((clear[LAT] & elig).any()),
                      min_clear_lat=round(float(np.abs(G.A_LAT)[clear[LAT]].min()), 1) if clear[LAT].any() else "", clear_slow=int(clear[SLW].any()), clear_both=int(clear[BOTH].any()),
                      a_clear_lat=int((~a_hit[LAT]).any()), a_clear_slow=int((~a_hit[SLW]).any()), a_clear_both=int((~a_hit[BOTH]).any()),
                      head_clear=int(head_clear.any()), head_pick_true_clear=int(clear[LAT][np.flatnonzero(head_clear)[np.argmin(np.abs(G.A_LAT)[head_clear])]]) if head_clear.any() else ""))
        C["a_hit"].append(a_hit), C["b_depth"].append(depth.max(1)), C["c_out"].append(c_out), C["clear"].append(clear)
    sc = dict(set=st, seed=int(st[-1]), kind=kind, reason=why, scene=scene, t_ev=D[0]["t_ev"], struck=struck or "", n_dec=len(D), replay_diff=d_rep,
              ex_depth_max=float(ex_depth.max()), ex_depth_at_ev=float(ex_depth[np.searchsorted(ego[:, 0], t_ev)]) if t_ev else "", ex_lat_max=float(ex_lat.max()),
              ex_min_obst=float(np.nanmin(CL.metric(o, "min_distance_to_obstacle_m")[:, 1])), reg_res=reg[1] if reg else "", reg_t=reg[2] if reg else "",
              ex_raster_at_ev=float(raster_margin(sdf, reg[0], CL.to_rig(ego[np.searchsorted(ego[:, 0], t_ev)][None], r.off))[0]) if (t_ev and sdf is not None) else "",
              log_turn_deg=float(np.degrees(CL.wrap(r.gt_rig[-1, 3] - r.gt_rig[0, 3]))), tok_turn_deg=float(np.degrees(tok["fut"][-1, 2])) if tok is not None else "",
              v_start=float(r.ego_v(r.T0 + 2e5)[0]), log_arc=float(cor.len), **chk)
    return D, sc, {k: np.array(v) for k, v in C.items()}, dict(r=r, U=U, P8=P8, za=za, zb=zb, reg=reg)


def tokens():
    z = np.load(DATA / "runs/op_probe/labels/navtest.npz")
    t = np.load(DATA / "runs/op_parity/cache/lb_navtest/tab.npz")
    pos, fut = {n: i for i, n in enumerate(z["tokens"].tolist())}, {n: i for i, n in enumerate(t["names"].tolist())}
    return lambda tk: (dict(sdf=z["sdf"][pos[tk]], ok=bool(z["ok"][pos[tk]]), fut=t["fut"][fut[tk]].astype(float)) if tk in pos and tk in fut else None)


def load_set(st):
    return CL.load(OUT / "x" / st / "logs.pkl"), CL.load(OUT / "x" / st / "map.pkl"), dict(np.load(OUT / "replay" / f"{st}.npz"))


def cmd_truth(a, run):
    rows = list(csv.DictReader(open(OUT / "zeros.csv")))
    tok = tokens()
    D, SC, C = [], [], {}
    for st in G.SETS:
        L, M, Z = load_set(st)
        kind = st.split("-")[0]
        for row in [x for x in rows if x["seed"] == st[-1]]:
            scene = row["scene"] if kind == "zero" else row["clean"]
            if scene not in L:
                run.info(f"{st}: {scene} missing from the extraction")
                continue
            d, sc, c, _ = one_scene(st, kind, row, L[scene], M[scene], Z, tok(scene.rsplit("-", 1)[1]))
            if kind == "clean":
                sc["pair"] = row["scene"]
            D += d
            SC.append(sc)
            for k, v in c.items():
                C.setdefault(k, []).append(v)
        run.info(f"{st}: {sum(x['set'] == st for x in SC)} scenes")
    for name, R in (("dec", D), ("scenes", SC)):
        keys = list(dict.fromkeys(k for x in R for k in x))
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader(), w.writerows(R)
    np.savez_compressed(OUT / "cand.npz", **{k: np.concatenate(v) for k, v in C.items()}, cand=np.array(G.CAND))
    run.summary.update(decisions=len(D), scenes=len(SC), off_agree=sum(x["off_agree"] for x in SC), off_n=sum(x["off_n"] for x in SC),
                       lat_err_max=max(x["lat_err"] for x in SC), replay_diff_max=max(x["replay_diff"] for x in SC))
    run.info(str(run.summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("truth", "tables", "figs"))
    a = ap.parse_args()
    from jevdrive.run import Run
    with Run("body1", f"diag-{a.cmd}", config=vars(a)) as run:
        if a.cmd == "truth":
            cmd_truth(a, run)
        else:
            import bd1_diag_report as R
            (R.cmd_tables if a.cmd == "tables" else R.cmd_figs)(a, run)


if __name__ == "__main__":
    main()
