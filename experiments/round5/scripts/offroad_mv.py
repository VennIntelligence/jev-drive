#!/usr/bin/env python3
"""ROUND5 OFFROAD-MV: what the PAI-human-row recipe's extra offroad zeros are (R3MV s0 / s1, MV-LC s0 / s1 on PAI 281 ACC + GAP, scenes where
NTV0-L18-s0 is not offroad) and what in the training rows could cause them. Diagnosis, stored reads only, no GPU. Lane script: offroad_mv.sh.

  events  (box, repo .venv)  every offroad zero of the four reads on a scene NTV0 is not offroad: D1's lateral anatomy (d1_mac.lateral_one:
          geometry class, speed, human turn, context) + the plan at the decisions 3 / 2 / 1 s before the event against the human log and against
          NTV0's plan at the same log station (lateral offset, heading, curvature vs the log's over the plan's arc, plan speed vs the human's, plan
          footprint margin to the extract's road surface) -> results/offroad_mv_events.csv, offroad_mv_plan.csv
  data    (box, op-train env, CPU)  the drawn training rows by state class (R3MV draw for the PAI parts ph / ph05 / ph15, navtrain uniform):
          label lateral acceleration and curvature at speed, 4 s lateral offset on straight rows, the label footprint's margin to the road raster
          (round 5 road_pai rasters; navtrain: the nuPlan road raster the S rows' hinge uses) and, on turning rows, which edge the label nears
          (inside / outside clearance at t0 and at the label's tightest pose) -> $O/data.json
  report  (box, repo .venv)  events + data + the R5-R (road hinge) PAI-FIX check -> results/offroad_mv.md

Inputs on the box: $DATA_DIR/runs/pai_fix/{facts,tax} (pai_fix tax extracts), $DATA_DIR/runs/round5/offroad_mv/dec (d1.py extract of the four
reads, R3MV-vl, R5R-vl), $DATA_DIR/runs/round5/d1/dec_NTV0-281ag.pkl.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round5/offroad_mv"
RES = REPO / "experiments/round5/results"
READS = {"R3MV-s0": "R3MV-L18-s0-281ag", "R3MV-s1": "R3MV-L18-s1-281ag", "MVLC-s0": "R5LC-L18-s0-281ag", "MVLC-s1": "R5LC-L18-s1-281ag"}
DECS = {"R3MV-s0": "R3MV-s0-281ag", "R3MV-s1": "R3MV-s1-281ag", "MVLC-s0": "R5LC-s0-281ag", "MVLC-s1": "R5LC-s1-281ag"}
NTV0 = "NTV0-L18-s0-281ag"
LEADS = (3.0, 2.0, 1.0)                    # plan read-outs this many seconds before the event


def _d1():
    """d1_mac with its stores pointed at the box (facts / tax of pai_fix, decisions of this lane)."""
    sys.path[:0] = [str(HERE)]
    import d1_mac as D
    D.T = O / "stage"
    return D


def stage():
    """symlink the inputs into the layout d1_mac reads (stage/{facts,tax,dec})."""
    S = O / "stage"
    for sub in ("facts", "tax", "dec"):
        (S / sub).mkdir(parents=True, exist_ok=True)
    for n in list(READS.values()) + [NTV0, "R2C-L18-s0-281ag", "R2J-L18-s0-281ag", "R3MV-L18-s0-vl", "R5R-L18-s0-vl", "NTV0-L18-s0-vl"]:
        for sub, ext in (("facts", "csv"), ("tax", "pkl")):
            src, dst = DATA / f"runs/pai_fix/{sub}/{n}.{ext}", S / sub / f"{n}.{ext}"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
    for f in list((O / "dec").glob("dec_*.pkl")) + [DATA / f"runs/round5/d1/dec_{x}-281ag.pkl" for x in ("NTV0", "R2C", "R2J")]:
        dst = S / "dec" / f.name
        if not dst.exists():
            dst.symlink_to(f)


# ---------------------------------------------------------------- events
def road_of(o):
    """the extract's road surface (shapely polygon) around the rollout (offroad / failing rolls carry `fig`), else None."""
    import shapely
    fg = o.get("fig") or {}
    if not fg.get("road"):
        return None
    polys = [shapely.Polygon(r) for r in fg["road"] if len(r) >= 3]
    U = shapely.union_all([p.buffer(0) for p in polys])
    for h in fg.get("holes", []):
        if len(h) >= 3:
            U = U.difference(shapely.Polygon(h))
    shapely.prepare(U)
    return U


def foot_margin(U, P, Y, dims):
    """min signed distance (+ inside) of the footprint corners at poses P (n, 2), yaw Y (n,) to the surface U."""
    import shapely
    if U is None:
        return np.nan
    rear, front, hw = dims
    c, s = np.cos(Y), np.sin(Y)
    pts = np.concatenate([np.c_[P[:, 0] + c * u - s * v, P[:, 1] + s * u + c * v] for u, v in ((front, hw), (front, -hw), (rear, -hw), (rear, hw))])
    d = shapely.distance(shapely.points(pts), U.boundary)
    return float(np.where(shapely.contains_xy(U, pts[:, 0], pts[:, 1]), d, -d).min())


def plan_geom(D, C, Lg, d, k, U=None, dims=None):
    """the model plan of decision k against the human log: ego / plan lateral, heading, curvature over the plan's arc vs the log's over the same
    stations, plan speed vs the human's at that station."""
    Pl, Yl, Tl = np.asarray(Lg["P"], float), np.asarray(Lg["Y"], float), np.asarray(Lg["T"], float)
    Sl = D.arclen(Pl)
    tk = float(d["t0"][k])
    a = D.ego_at(C, tk)
    W = D.plan_world(d, k, a)
    pm = d["pm"][k].astype(float)
    s0, lat0 = D.project(Pl, Sl, Yl, a[0], a[1])
    s2, lat2 = D.project(Pl, Sl, Yl, *W[3])
    s4, lat4 = D.project(Pl, Sl, Yl, *W[7])
    arc = float(D.arclen(np.r_[[[0, 0]], pm[:, :2]])[-1])
    yl0 = float(np.interp(s0, Sl, Yl))
    yp4 = a[2] + pm[7, 2]
    dlog = float(np.interp(s0 + arc, Sl, Yl) - yl0)                     # the log's heading change over the plan's arc from the ego's station
    th = float(np.interp(s0, Sl, Tl))
    vh = (float(np.interp(th + 0.25e6, Tl, Sl)) - float(np.interp(th - 0.25e6, Tl, Sl))) / 0.5
    hdg = lambda x: float(np.degrees((x + np.pi) % (2 * np.pi) - np.pi))  # noqa: E731
    r = dict(t=tk, st=s0, v0=float(d["v0"][k]), v_h=vh, lat0=lat0, lat2=lat2, lat4=lat4, hdg0=hdg(a[2] - yl0),
             hdg4=hdg(yp4 - float(np.interp(s4, Sl, Yl))), turn_plan=float(np.degrees(pm[7, 2])), turn_log=float(np.degrees(dlog)),
             arc=arc, v_plan=arc / 4.0, ds4=s4 - s0, v_set=float(d["v_set"][k]), v_model=float(d["v_model"][k]), won=str(d["won"][k]),
             lead_d=float(d["lead_d"][k]), lead_vl=float(d["lead_vl"][k]))
    if U is not None:
        r["m_plan"] = foot_margin(U, W, a[2] + pm[:, 2], dims)
        r["m_ego"] = foot_margin(U, a[None, :2], np.array([a[2]]), dims)
    return r


def pull_scan(D, o, d, step=5):
    """decisions every 0.5 s with v0 >= 4 m/s: the model plan slows (end speed < 0.6 v0), moves right (4 s point y < -0.8 m, |yaw| < 20 deg),
    both, while the human at the ego's log station keeps >= 0.8 of its speed over the next 4 s (no human slow-down there)."""
    C, Lg = o["ctrl"], o["log"]
    Pl, Yl, Tl = np.asarray(Lg["P"], float), np.asarray(Lg["Y"], float), np.asarray(Lg["T"], float)
    Sl = D.arclen(Pl)
    vl = np.r_[0, np.diff(Sl) / np.maximum(np.diff(Tl) * 1e-6, 1e-3)]
    n = slow = right = both = 0
    for k in range(0, len(d["t0"]), step):
        tk = float(d["t0"][k])
        if tk > C["t"][-1] or d["v0"][k] < 4:
            continue
        a = D.ego_at(C, tk)
        st = D.project(Pl, Sl, Yl, a[0], a[1])[0]
        th = float(np.interp(st, Sl, Tl))
        vh0, vh4 = float(np.interp(th, Tl, vl)), float(np.interp(th + 4e6, Tl, vl))
        if vh0 < 3 or vh4 < 0.8 * vh0:
            continue
        pm = d["pm"][k]
        ve = float(np.hypot(*(pm[7, :2] - pm[6, :2])) / 0.5)
        sl, rt = ve < 0.6 * d["v0"][k], (pm[7, 1] < -0.8) and abs(np.degrees(pm[7, 2])) < 20
        n += 1
        slow += sl
        right += rt
        both += sl and rt
    return dict(n_dec=n, slow=slow, right=right, both=both)


def cmd_events(a):
    import pandas as pd
    stage()
    D = _d1()
    FN, AN, DN = D.facts(NTV0), D.load_tax(NTV0), D.load_dec("NTV0-281ag")
    rows, prow = [], []
    for k, n in READS.items():
        F, A, Dc = D.facts(n), D.load_tax(n), D.load_dec(DECS[k])
        for s in F.index[F.fclass == "offroad"]:
            if s not in A or (s in FN.index and FN.fclass[s] == "offroad"):
                continue
            o, d = A[s], Dc.get(s)
            r = D.lateral_one(o, d, F.fclass[s], F.primary[s], F.t_ev_s[s])
            C, Lg = o["ctrl"], o["log"]
            t0 = float(C["t"][0])
            te = t0 + r["t_ev"] * 1e6
            U = road_of(o)
            e = D.ego_at(C, te)
            Pl, Yl = np.asarray(Lg["P"], float), np.asarray(Lg["Y"], float)
            Sl = D.arclen(Pl)
            st_e = D.project(Pl, Sl, Yl, e[0], e[1])[0]
            r |= dict(read=k, scene=s, ntv0=FN.fclass.get(s, "rpc"), ntv0_score=FN.score.get(s, np.nan), v_hand=F.v_hand[s], st_ev=st_e,
                      m_ev=foot_margin(U, e[None, :2], np.array([e[2]]), o["dims"]),
                      turn_prev40=D._turn_deg(Yl, Sl, st_e - 40, st_e), turn_next20=D._turn_deg(Yl, Sl, st_e, st_e + 20))
            rows.append(r)
            if d is None or not len(d["t0"]):
                continue
            on, dn = AN.get(s), DN.get(s)
            for L in LEADS:
                kk = min(max(int(np.searchsorted(d["t0"], te - L * 1e6)), 0), len(d["t0"]) - 1)
                g = plan_geom(D, C, Lg, d, kk, U, o["dims"])
                prow.append(dict(read=k, scene=s, who="read", lead=L, **g))
                if on is None or dn is None or not len(dn["t0"]):
                    continue
                # NTV0's decision whose ego station is closest to this decision's (NTV0's own log = the same human log)
                Pl, Yl = np.asarray(on["log"]["P"], float), np.asarray(on["log"]["Y"], float)
                Sl = D.arclen(Pl)
                tt = dn["t0"][dn["t0"] <= on["ctrl"]["t"][-1]]
                if not len(tt):
                    continue
                sn = np.array([D.project(Pl, Sl, Yl, *D.ego_at(on["ctrl"], t)[:2])[0] for t in tt])
                j = int(np.argmin(np.abs(sn - g["st"])))
                if abs(sn[j] - g["st"]) < 3.0:
                    prow.append(dict(read=k, scene=s, who="NTV0", lead=L, **plan_geom(D, on["ctrl"], on["log"], dn, j, U, o["dims"])))
    E, P = pd.DataFrame(rows), pd.DataFrame(prow)
    # read-wide scan: model plans that slow down and pull right while the human at that station keeps its speed
    scan = []
    for k, n in [("NTV0", NTV0), ("R2C", "R2C-L18-s0-281ag"), ("R2J", "R2J-L18-s0-281ag")] + list(READS.items()):
        F, A, Dc = D.facts(n), D.load_tax(n), D.load_dec(DECS.get(k, f"{k}-281ag"))
        for s in F.index:
            if s not in A or s not in Dc or not len(Dc[s]["t0"]):
                continue
            scan.append(dict(read=k, scene=s, fclass=F.fclass[s], **pull_scan(D, A[s], Dc[s])))
    S = pd.DataFrame(scan)
    S.to_csv(RES / "offroad_mv_pullscan.csv", index=False, float_format="%.3f")
    E.to_csv(RES / "offroad_mv_events.csv", index=False, float_format="%.3f")
    P.to_csv(RES / "offroad_mv_plan.csv", index=False, float_format="%.3f")
    print(E[["read", "scene", "cls", "v_ego", "turn_deg", "lat_ev", "ntv0"]].to_string())


# ---------------------------------------------------------------- data (training rows)
VB = (2, 5, 10, 15, 20, 25, 30, 60)
OFFS = np.arange(0.0, 8.01, 0.25)
CTR = 1.45                                  # car centre ahead of the rear axle (PAI rig / Pacifica, about)


def label_feats(v0, fut):
    """per-row label geometry: peak lateral acceleration from the positions (v^2 x Menger curvature of consecutive label points, origin
    included), max |heading|, 4 s lateral y4 (signed, left +), end speed (last 0.5 s), arc."""
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[:, :, :2]], 1)                 # (n, 9, 2)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)                                  # (n, 8)
    a, b, c = P[:, :-2], P[:, 1:-1], P[:, 2:]
    cr = np.abs((b - a)[..., 0] * (c - a)[..., 1] - (b - a)[..., 1] * (c - a)[..., 0])
    den = seg[:, :-1] * seg[:, 1:] * np.linalg.norm(c - a, axis=-1)
    kap = np.where(den > 1e-3, 2 * cr / np.maximum(den, 1e-9), 0.0)                    # (n, 7) at points 1..7
    kap = np.where((seg[:, :-1] > 0.25) & (seg[:, 1:] > 0.25), kap, 0.0)               # no curvature from creeping points
    v = (seg[:, :-1] + seg[:, 1:]) / 1.0                                                 # m/s around the point
    yaw = np.unwrap(fut[:, :, 2], axis=1)
    return dict(alat=(v ** 2 * kap).max(1), kmax=kap.max(1), hmax=np.degrees(np.abs(yaw).max(1)), y4=fut[:, 7, 1], arc=seg.sum(1),
                vend=seg[:, -1] / 0.5, v0=v0, h4=np.degrees(yaw[:, -1]))


def sdf_points(H, rows, pts):
    """SDF at points pts (B, P, 2) (rig frame) of the Hinge rows."""
    import torch
    from drivable_hinge import X0, Y0
    g = torch.stack([(pts[..., 1] - Y0) / 24.0 - 1.0, (pts[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
    return torch.nn.functional.grid_sample(H.sdf[rows].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]


def clearances(H, rows, fut, bs=2048):
    """left / right clearance (m) of the car centre to the first SDF < 0 along the pose's normal (capped 8 m) at t0 and the 8 label poses
    -> left (n, 9), right (n, 9); the label footprint's min margin (n,) and the t0 footprint's (n,) (road_pai._margins, Pacifica)."""
    import torch
    sys.path[:0] = [str(HERE)]
    from road_pai import _margins
    L, R = [], []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs])
            f = np.concatenate([np.zeros((len(r), 1, 3)), fut[i:i + bs]], 1)
            c, s = np.cos(f[..., 2]), np.sin(f[..., 2])
            cx, cy = f[..., 0] + CTR * c, f[..., 1] + CTR * s
            out = []
            for side in (1.0, -1.0):
                px, py = cx[..., None] - side * s[..., None] * OFFS, cy[..., None] + side * c[..., None] * OFFS
                v = sdf_points(H, r, torch.as_tensor(np.stack([px, py], -1).reshape(len(r), -1, 2), dtype=torch.float32)).numpy()
                neg = v.reshape(len(r), 9, len(OFFS)) < 0
                out.append(np.where(neg.any(-1), OFFS[np.argmax(neg, -1)], 8.0))
            L.append(out[0]), R.append(out[1])
    m0, ml, _ = _margins(H, rows, fut.astype(np.float32))
    return np.concatenate(L), np.concatenate(R), ml, m0


def all_rows_index(pool):
    """a copy of road_pai's index with every raster row usable (ok = True: no fallback; the analysis wants the labels the hinge skips too)."""
    z = dict(np.load(DATA / f"runs/round5/road_pai/road_{pool}.npz"))
    z["ok"] = np.ones(len(z["tokens"]), bool)
    z["sdf_file"] = np.array(str(DATA / "runs/round5/road_pai" / str(z["sdf_file"])))   # absolute: Path(parent) / abs = abs
    f = O / "idx" / f"road_all_{pool}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **z)
    return f


def row_table(v0, fut, Lc, Rc, ml, m0):
    import pandas as pd
    F = label_feats(v0, fut)
    T = pd.DataFrame(F)
    sg = np.sign(F["h4"])
    T["ml"], T["m0"] = ml, m0
    T["inr"] = (fut[:, :, 0].max(1) < 52.0) & (np.abs(fut[:, :, 1]).max(1) < 20.0)       # label inside the raster (x < 56, |y| < 24 m)
    T["r_end"], T["l_end"], T["r0"], T["l0"] = Rc[:, 8], Lc[:, 8], Rc[:, 0], Lc[:, 0]
    T["r_min"], T["l_min"] = Rc[:, 1:].min(1), Lc[:, 1:].min(1)
    ins, out = np.where(sg[:, None] >= 0, Lc, Rc), np.where(sg[:, None] >= 0, Rc, Lc)          # inside = the side the label turns to
    T["ins0"], T["ins_min"], T["out_min"] = ins[:, 0], ins[:, 1:].min(1), out[:, 1:].min(1)
    return T


STATES = {  # state classes on the label (v0 = fed speed, m/s; y4 left +; h = max |heading| over the label)
    "cruise straight (h < 5 deg, v_end >= 0.8 v0)": lambda T: (T.hmax < 5) & (T.vend >= 0.8 * T.v0),
    "curve / bend (15-45 deg, v0 >= 4)": lambda T: (T.hmax >= 15) & (T.hmax < 45) & (T.v0 >= 4),
    "junction turn (>= 45 deg)": lambda T: T.hmax >= 45,
    "decelerating (v_end < 0.6 v0, h < 20 deg)": lambda T: (T.vend < 0.6 * T.v0) & (T.hmax < 20),
    "pull right while slowing (v_end < 0.6 v0, y4 < -0.8 m, h < 20 deg)": lambda T: (T.vend < 0.6 * T.v0) & (T.y4 < -0.8) & (T.hmax < 20),
    "pull left while slowing (v_end < 0.6 v0, y4 > 0.8 m, h < 20 deg)": lambda T: (T.vend < 0.6 * T.v0) & (T.y4 > 0.8) & (T.hmax < 20),
    "fast (v0 >= 25)": lambda T: T.v0 >= 25,
    "fast (v0 >= 30)": lambda T: T.v0 >= 30,
    "slow-down to a stop (v0 >= 4, v_end < 0.5, h < 20 deg)": lambda T: (T.v0 >= 4) & (T.vend < 0.5) & (T.hmax < 20),
    "slow-down, no stop (v0 >= 4, 0.5 <= v_end < 0.6 v0, h < 20 deg)": lambda T: (T.v0 >= 4) & (T.vend >= 0.5) & (T.vend < 0.6 * T.v0) & (T.hmax < 20),
}
FIX = {  # candidate data-only fixes: rows set to weight 0 (part renormalised)
    "F1 stretch-1.5 rows with label lat. accel > 3 m/s^2 or fed v0 > 30 m/s": ("ph15", lambda T: (T.alat > 3) | (T.v0 > 30)),
    "F1a stretch-1.5 rows with label lat. accel > 3 m/s^2": ("ph15", lambda T: T.alat > 3),
    "F1b stretch-1.5 rows fed v0 > 30 m/s": ("ph15", lambda T: T.v0 > 30),
    "F2 PAI rows slowing without a lead and without stopping (v0 >= 4, 0.5 <= v_end < 0.6 v0, h < 20 deg, no LEAD)": (
        "ph ph05 ph15", lambda T: (T.v0 >= 4) & (T.vend >= 0.5) & (T.vend < 0.6 * T.v0) & (T.hmax < 20) & ~T.lead),
    "F2a PAI rows slowing and pulling right (v_end < 0.6 v0, y4 < -0.8 m, h < 20 deg)": (
        "ph ph05 ph15", lambda T: (T.vend < 0.6 * T.v0) & (T.y4 < -0.8) & (T.hmax < 20)),
}


def wstats(T, w, m):
    """draw-weighted read-outs of the rows in mask m."""
    m = np.asarray(m) & (w > 0)
    if m.sum() < 10:
        return dict(n=int(m.sum()))
    ww = w[m] / w[m].sum()
    X = T[m]

    def q(x, p):
        i = np.argsort(x.values)
        c = np.cumsum(ww[i])
        return float(x.values[i][min(np.searchsorted(c, p), len(c) - 1)])

    def sh(b):
        return float((ww * np.asarray(b, float)).sum())
    r = X.inr.values                                   # margin / clearance read-outs only on labels inside the raster (border clamp beyond)
    wr = ww * r / max((ww * r).sum(), 1e-12)

    def shr(b):
        return float((wr * np.asarray(b, float)).sum())

    def qr(x, p):
        x, w_ = x.values[r], wr[r]
        if not len(x):
            return np.nan
        i = np.argsort(x)
        c = np.cumsum(w_[i])
        return float(x[i][min(np.searchsorted(c, p), len(c) - 1)])
    lead = dict(lead=sh(X.lead)) if "lead" in X else {}
    return lead | dict(in_raster=sh(r), ml_lt03=shr(X.ml < 0.3), ml_lt0=shr(X.ml < 0), r_end_lt05=shr(X.r_end < 1.6), l_end_lt05=shr(X.l_end < 1.6),
                       ins_min50=qr(X.ins_min, 0.5), ins_tight=shr(X.ins_min < 1.6), ins_drop50=qr(X.ins0 - X.ins_min, 0.5), out_min50=qr(X.out_min, 0.5)) | dict(n=int(m.sum()), alat50=q(X.alat, 0.5), alat90=q(X.alat, 0.9), alat99=q(X.alat, 0.99), alat_gt2=sh(X.alat > 2), alat_gt3=sh(X.alat > 3),
                y4_mean=float((ww * X.y4).sum()), y4_right08=sh(X.y4 < -0.8), y4_left08=sh(X.y4 > 0.8), absy4_90=q(X.y4.abs(), 0.9),
                vend_v0=q(X.vend / X.v0.clip(lower=0.5), 0.5),
                acc_gt115=sh(X.vend > 1.15 * X.v0), v50=q(X.v0, 0.5))


def cmd_data(a):
    import pandas as pd
    import torch
    sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round3/scripts")]
    from drivable_hinge import Hinge
    from jevdrive.common import n_cpus
    import r3
    torch.set_num_threads(n_cpus())
    CR = DATA / "runs/op_parity/cache"
    B = np.load(DATA / "runs/defect_fix/w/R3MV.npz")
    idx = {x: i for i, x in enumerate(B["names"].astype(str))}
    wl = np.load(DATA / "runs/round5/r5/w/R5LC.npz")["w"]
    res, tabs = dict(states=list(STATES)), {}
    for p, pool in r3.PARTS.items():
        t = r3.load_part(p)
        nm = t["names"].astype(str)
        ix = np.array([idx[f"on:{p}:{x[3:]}"] for x in nm])
        H = Hinge(all_rows_index(pool), nm, "cpu", 0.3, bank=True)
        rows = np.flatnonzero(H.ok.numpy())
        assert len(rows) == len(nm), f"{p}: rows without a raster"
        Lc, Rc, ml, m0 = clearances(H, rows, t["fut"].astype(np.float64))
        del H
        T = row_table(t["speed"].astype(np.float64), t["fut"].astype(np.float64), Lc, Rc, ml, m0)
        T["name"], T["w3"], T["w5"] = nm, B["w"][ix].astype(np.float64), wl[ix].astype(np.float64)
        k = np.array([int(x.rsplit(":", 1)[1]) for x in nm])
        sc = np.array([x.split(":")[2] for x in nm])
        kmax = pd.Series(k).groupby(sc).transform("max").values
        T["kpos"] = k / np.maximum(kmax, 1)
        LR = np.load(DATA / "runs/round5/r5/N/lead_rows.npz")
        lp = {x: i for i, x in enumerate(LR[f"{p}_names"].astype(str))}
        T["lead"] = LR[f"{p}_lead"][np.array([lp[f"on:{p}:{x[3:]}"] for x in nm])]
        tabs[p] = T
        mv = T.v0 >= 2
        res[p] = {"all moving": wstats(T, T.w3.values, mv)} | {s: wstats(T, T.w3.values, mv & f(T)) for s, f in STATES.items()}
        res[p + " MV-LC"] = {"all moving": wstats(T, T.w5.values, mv)} | {s: wstats(T, T.w5.values, mv & f(T)) for s, f in STATES.items()}
        pr = mv & STATES["pull right while slowing (v_end < 0.6 v0, y4 < -0.8 m, h < 20 deg)"](T) & (T.w3 > 0)
        res[p]["pull right: row position in the clip (k / k_max) median, share in the last 20 %"] = [float(T.kpos[pr].median()), float((T.kpos[pr] > 0.8).mean()),
                                                                                                   float((T.kpos[mv & (T.w3 > 0)] > 0.8).mean())]
        res[p]["pull right: rows a batch (64 x draw mass)"] = float(r3.K_ON * T.w3[pr].sum())
        res[p]["pull right MV-LC: rows a batch"] = float(r3.K_ON * T.w5[pr].sum())
        print(p, json.dumps(res[p]["all moving"]), flush=True)
    fx = {}
    for fn, (parts, rule) in FIX.items():
        r_ = dict(rows=0, drawn_rows=0, per_batch=0.0, per_batch_of_part={}, scenes=0)
        sc_ = set()
        for p in parts.split():
            T = tabs[p]
            m = (T.v0 >= 2) & rule(T) & (T.w3 > 0)
            r_["drawn_rows"] += int(m.sum())
            r_["per_batch"] += float(r3.K_ON * T.w3[m].sum())
            r_["per_batch_of_part"][p] = float(T.w3[m].sum() / T.w3[T.v0 >= 2].sum())
            sc_ |= set(x.split(":")[2] for x in T.name[m])
        r_["scenes"] = len(sc_)
        fx[fn] = r_
    res["fixes"] = fx
    # stretch f = 1.5 rows against their plain base row (same t0 pose, the base row's 6 s path in 4 s)
    ph, p15 = tabs["ph"], tabs["ph15"]
    base = {x.split(":", 2)[2]: i for i, x in enumerate(ph.name)}
    j = np.array([base.get(x.split(":", 2)[2], -1) for x in p15.name])
    m = (j >= 0) & (p15.w3.values > 0)
    A, Bp = p15[m].reset_index(drop=True), ph.iloc[j[m]].reset_index(drop=True)
    res["stretch15"] = dict(paired=int(m.sum()), v_ratio_med=float(np.median(A.v0 / Bp.v0.clip(lower=0.1))),
                            alat_ratio_med=float(np.median(A.alat[Bp.alat > 0.3] / Bp.alat[Bp.alat > 0.3])),
                            alat_gt3_ph15=float((A.alat > 3).mean()), alat_gt3_base=float((Bp.alat > 3).mean()),
                            ml_lt03_ph15_inr=float((A.ml[A.inr] < 0.3).mean()), ml_lt03_base_same_rows=float((Bp.ml[A.inr] < 0.3).mean()),
                            v_gt30_ph15=float((A.v0 > 30).mean()), v_gt30_base=float((Bp.v0 > 30).mean()))
    # navtrain (uniform; the nuPlan road raster of the S rows' road hinge)
    nz = [np.load(CR / f"navtrain_full.s{i}of12" / "tab.npz") for i in range(12)]
    tok = np.concatenate([z["names"] for z in nz]).astype(str)
    fut = np.concatenate([z["fut"] for z in nz]).astype(np.float64)
    v0 = np.concatenate([z["speed"] for z in nz]).astype(np.float64)
    H = Hinge(DATA / "runs/body1/labels_road/navtrain_s01234567891011.npz", tok, "cpu", 0.3, bank=True)
    rows = np.flatnonzero(H.ok.numpy() & np.isfinite(fut.reshape(len(fut), -1)).all(1))
    Lc, Rc, ml, m0 = clearances(H, rows, fut[rows])
    del H
    T = row_table(v0[rows], fut[rows], Lc, Rc, ml, m0)
    w = np.ones(len(T))
    mv = T.v0 >= 2
    res["navtrain"] = {"all moving": wstats(T, w, mv)} | {s: wstats(T, w, mv & f(T)) for s, f in STATES.items()}
    res["navtrain"]["raster_rows"] = int(len(rows))
    print("navtrain", json.dumps(res["navtrain"]["all moving"]), flush=True)
    O.mkdir(parents=True, exist_ok=True)
    (O / "data.json").write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------- report
GEN, MARK = "<!-- generated: offroad_mv.py report -->", "<!-- end generated -->"
MECH = ("slow-down drift", "over-speed drift", "turn / curve: cut inside", "turn / curve: wide outside", "straight: lateral offset")


def mechanism(r):
    """one event -> mechanism (order matters): the ego far below the human's speed with the model's own speed winning; far above it on a
    straight road; on a turn / curve (the log turns >= 10 deg over the 60 m around the event) inside or outside; else a straight-road offset."""
    tsum = r.turn_prev40 + r.turn_next20
    if r.v_ego < 0.6 * r.v_human and r.vp1 < 0.6 * r.vh1:
        return MECH[0]
    if abs(tsum) < 10 and r.v_ego > 1.2 * r.v_human:
        return MECH[1]
    if abs(tsum) >= 10:
        return MECH[2] if np.sign(r.lat_ev) == np.sign(tsum) else MECH[3]
    return MECH[4]


def md_table(df, cols=None, fmt="{:.2f}"):
    cols = cols or list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(":--" if df[c].dtype == object else "--:" for c in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return out


def cmd_report(a):
    import pandas as pd
    stage()
    D = _d1()
    E = pd.read_csv(RES / "offroad_mv_events.csv")
    P = pd.read_csv(RES / "offroad_mv_plan.csv")
    S = pd.read_csv(RES / "offroad_mv_pullscan.csv")
    X = json.loads((O / "data.json").read_text())
    for d in (E, P, S):
        d["sc"] = d.scene.str[7:15]
    q = P[(P.who == "read") & (P.lead == 1.0)].set_index(["read", "scene"])
    E["vp1"] = [q.v_plan.get((r.read, r.scene), np.nan) for r in E.itertuples()]
    E["vh1"] = [q.v_h.get((r.read, r.scene), np.nan) for r in E.itertuples()]
    E["mech"] = [mechanism(r) for r in E.itertuples()]
    FJ = D.facts("R2J-L18-s0-281ag")
    E["R2J"] = [FJ.fclass.get(s, "rpc") for s in E.scene]
    n4 = E.groupby("scene").read.nunique()
    E["n_of_4"] = E.scene.map(n4)
    E.to_csv(RES / "offroad_mv_events.csv", index=False, float_format="%.3f")
    # per scene
    g = E.groupby("scene")
    SC = pd.DataFrame(dict(n_of_4=n4, reads=g.read.agg(lambda x: " ".join(sorted(x))), mech=g.mech.agg(lambda x: x.mode()[0]),
                           d1_class=g.cls.agg(lambda x: x.mode()[0]), v_ego=g.v_ego.median(), v_human=g.v_human.median(),
                           lat_ev=g.lat_ev.median(), turn60=(g.turn_prev40.median() + g.turn_next20.median()), ctx=g.ctx.agg(lambda x: x.mode()[0]),
                           NTV0=g.ntv0.first(), R2J=g.R2J.first())).reset_index()
    SC["scene"] = SC.scene.str[7:15]
    SC = SC.sort_values(["n_of_4", "mech"], ascending=[False, True])
    SC.to_csv(RES / "offroad_mv.csv", index=False, float_format="%.3f")
    L = ["# OFFROAD-MV: the PAI-human-row family's extra offroad zeros on PAI 281 ACC + GAP (ROUND5, diagnosis)", "",
         "Stored reads only, no GPU. Reads: R3MV-L18 s0 / s1, MV-LC (R5LC-L18) s0 / s1, PAI 281 under ACC + GAP; NTV0-L18-s0 the same serve "
         "as reference. Event = an offroad zero (pai_fix fclass) on a scene where NTV0 is not offroad. Code: `scripts/offroad_mv.py` "
         "(`events`, `data`, `report`), lane `scripts/offroad_mv.sh` (pool CPU jobs, PRIO 15). Files: per event "
         "[offroad_mv_events.csv](offroad_mv_events.csv), per scene [offroad_mv.csv](offroad_mv.csv), plans at 3 / 2 / 1 s "
         "[offroad_mv_plan.csv](offroad_mv_plan.csv), read-wide plan scan [offroad_mv_pullscan.csv](offroad_mv_pullscan.csv).", ""]
    # 1. events
    L += ["## 1. Events", "", f"{len(E)} offroad zeros on {E.scene.nunique()} scenes (R3MV s0 / s1 / MV-LC s0 / s1: "
          + " / ".join(str((E.read == k).sum()) for k in READS) + "). Recurrence across the four reads (scenes): "
          + ", ".join(f"{k} of 4: {int((n4 == k).sum())}" for k in (4, 3, 2, 1)) + ".", "",
          "Mechanism per event (rule, in this order): **slow-down drift** = ego speed < 0.6 x the human's at the event and the model plan 1 s "
          "before < 0.6 x the human's speed there; **over-speed drift** = straight (log turns < 10 deg over the 60 m around the event) and ego "
          "> 1.2 x the human; **turn / curve** = the log turns >= 10 deg over those 60 m, inside / outside by the sign of the ego's lateral "
          "offset; else **straight: lateral offset**. `d1_class` = D1's geometry class (d1_mac.lateral_one; its 10-15 deg bends are always "
          "called `cut`). Lateral: + left of the human log.", ""]
    L += md_table(SC, ["scene", "n_of_4", "mech", "d1_class", "v_ego", "v_human", "lat_ev", "turn60", "ctx", "NTV0", "R2J", "reads"])
    M = pd.crosstab(E.mech, E.n_of_4.map(lambda k: "stable (3-4 of 4)" if k >= 3 else "2 of 4" if k == 2 else "1 of 4"))
    M["events"] = M.sum(1)
    M["scenes"] = E.groupby("mech").scene.nunique()
    L += ["", "Events by mechanism and recurrence (count of events; `scenes` = distinct scenes):", ""] + md_table(M.reset_index().rename(columns={"mech": "mechanism"}), fmt="{:.0f}")
    # 2. stable anatomy
    st = sorted(set(E.scene[E.n_of_4 >= 3]))
    rows = []
    for s in st:
        for L_ in LEADS:
            for who in ("read", "NTV0"):
                x = P[(P.scene == s) & (P.lead == L_) & (P.who == who)]
                if not len(x):
                    continue
                ts = np.sign(x.turn_log.median()) if abs(x.turn_log.median()) >= 5 else 0
                rows.append(dict(scene=s[7:15], mech=E.mech[E.scene == s].mode()[0], t=f"-{L_:.0f} s", who="MV (4 reads)" if who == "read" else "NTV0",
                                 v0=x.v0.median(), v_h=x.v_h.median(), v_plan=x.v_plan.median(), lat_ego=x.lat0.median(), lat_plan4=x.lat4.median(),
                                 hdg_ego=x.hdg0.median(), turn_plan=x.turn_plan.median(), turn_log=x.turn_log.median(),
                                 curv_ratio=(x.turn_plan / x.turn_log).median() if ts else np.nan, won=x.won.mode()[0], lead_d=x.lead_d.median()))
    A = pd.DataFrame(rows)
    L += ["", "## 2. Stable events (offroad in >= 3 of the 4 reads): plan vs the human log and vs NTV0 at the same log station", "",
          "Plan = the model plan (`poses_model`) at the decision 3 / 2 / 1 s before the event (MV: median over the reads that have the event); "
          "NTV0 = its decision whose ego is closest to the same log station (within 3 m). `lat_ego` / `lat_plan4` = lateral offset of the ego / "
          "the plan's 4 s point from the human log (m, + left); `hdg_ego` = ego heading minus the log's (deg); `turn_plan` / `turn_log` = the "
          "plan's 4 s heading change and the log's over the same arc from the ego's station (deg); `curv_ratio` = their ratio when the log "
          "turns >= 5 deg; `won` = the ACC term that set the speed (model = the model plan's own speed); `lead_d` = ACC lead gap (m, blank = no lead).", ""]
    L += md_table(A)
    # 3. read-wide scan
    G = S.groupby("read").agg(scenes=("scene", "size"), decisions=("n_dec", "sum"), slow=("slow", "sum"), right=("right", "sum"), both=("both", "sum"),
                              scenes_slow5=("slow", lambda x: int((x >= 5).sum())), scenes_both3=("both", lambda x: int((x >= 3).sum())))
    for c in ("slow", "right", "both"):
        G[c + " %"] = 100 * G[c] / G.decisions
    G = G.reindex(["NTV0", "R2C", "R2J"] + list(READS)).reset_index()
    L += ["", "## 3. Read-wide: unforced slow-down plans", "",
          "Every 0.5 s decision with ego v0 >= 4 m/s where the human, at the ego's log station, keeps >= 0.8 of its speed (>= 3 m/s) over the "
          "next 4 s: `slow` = the model plan's end speed < 0.6 v0; `right` = the plan's 4 s point > 0.8 m right with |yaw| < 20 deg; `both`. "
          "`scenes_slow5` = scenes with >= 5 such slow plans (>= 2.5 s of them).", ""]
    L += md_table(G, fmt="{:.2f}")
    # 4. data
    parts = ["ph", "ph05", "ph15", "navtrain"]
    L += ["", "## 4. Training rows by state (moving rows, fed v0 >= 2 m/s; PAI parts under R3MV's draw weights, navtrain uniform)", "",
          "`alat` = peak lateral acceleration of the label path (v^2 x curvature of consecutive label points, m/s^2); `y4 R / L` = share of "
          "rows whose 4 s point is > 0.8 m right / left; `lead` = r5 LEAD flag (PAI only). Road read-outs only on labels inside the raster "
          "(x < 52 m, |y| < 20 m; `in raster` share): `ml<0` = the label footprint leaves the road surface (PAI: round 5 road_pai raster = "
          "the evaluator's NuRec surface; navtrain: the nuPlan road raster of the S rows' hinge, a different surface: compare PAI parts with "
          "each other first); `ins tight` = the car centre comes within 1.6 m of the inside edge on the label (bends / turns).", ""]
    cols = [("n", "{:.0f}"), ("v50", "{:.1f}"), ("alat50", "{:.2f}"), ("alat90", "{:.2f}"), ("alat_gt3", "{:.1%}"), ("y4_right08", "{:.1%}"),
            ("y4_left08", "{:.1%}"), ("vend_v0", "{:.2f}"), ("lead", "{:.0%}"), ("in_raster", "{:.0%}"), ("ml_lt0", "{:.1%}"), ("ins_tight", "{:.1%}"),
            ("ins_min50", "{:.2f}")]
    for stn in ["all moving"] + X["states"]:
        L += ["", f"**{stn}**", "", "| part | " + " | ".join(c for c, _ in cols) + " |", "|:--|" + "--:|" * len(cols)]
        for p in parts:
            r = X[p].get(stn, {})
            cells = [(f.format(r[c]) if c in r and r[c] is not None and not (isinstance(r[c], float) and np.isnan(r[c])) else "") for c, f in cols]
            L.append(f"| {p} | " + " | ".join(cells) + " |")
    s15 = X["stretch15"]
    L += ["", f"Stretch f = 1.5 rows vs their plain base row ({s15['paired']} drawn pairs): fed speed x {s15['v_ratio_med']:.2f}, label peak lateral "
          f"acceleration x {s15['alat_ratio_med']:.2f} (median); > 3 m/s^2 {s15['alat_gt3_ph15']:.1%} vs {s15['alat_gt3_base']:.1%}; fed v0 > 30 m/s "
          f"{s15['v_gt30_ph15']:.1%} vs {s15['v_gt30_base']:.1%}; label footprint within 0.3 m of the edge (in-raster ph15 rows) "
          f"{s15['ml_lt03_ph15_inr']:.1%} vs the same base rows {s15['ml_lt03_base_same_rows']:.1%}.", ""]
    for p in ("ph", "ph15"):
        k = [x for x in X[p] if x.startswith("pull right: row position")]
        if k:
            v = X[p][k[0]]
            L.append(f"- {p}: pull-right-while-slowing rows sit at clip position k / k_max median {v[0]:.2f}; {v[1]:.0%} in the last 20 % of the clip "
                     f"(all drawn moving rows {v[2]:.0%}): not an end-of-clip pull-over; {X[p]['pull right: rows a batch (64 x draw mass)']:.2f} rows a batch.")
    L += ["", "Candidate fixes (rows at weight 0 under R3MV's draw; drawn moving rows, rows a batch of 64, share of the part's drawn moving mass):", "",
          "| rule | drawn rows | rows a batch | share of part | scenes |", "|:--|--:|--:|--:|--:|"]
    for fn, r in X["fixes"].items():
        L.append(f"| {fn} | {r['drawn_rows']} | {r['per_batch']:.2f} | " + " / ".join(f"{p} {v:.1%}" for p, v in r["per_batch_of_part"].items())
                 + f" | {r['scenes']} |")
    # 5. R5-R
    fr, fm, fn_ = D.facts("R5R-L18-s0-vl"), D.facts("R3MV-L18-s0-vl"), D.facts("NTV0-L18-s0-vl")
    ex = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv").set_index("scene")
    o_r, o_m, o_n = (set(f.index[f.fclass == "offroad"]) for f in (fr, fm, fn_))
    common = set(fr.index) & set(fm.index)
    ev = set(E.scene)
    L += ["", "## 5. R5-R (R3MV + PAI road hinge, PAI-FIX native VC + LEAD, 118 scenes)", "",
          f"- offroad zeros: R5-R {len(o_r)}, R3MV {len(o_m)}, NTV0 {len(o_n)} (same 118-scene read, native serve); R5-R and R3MV share "
          f"{len(o_r & o_m)}; new in R5-R {len(o_r - o_m)} ({', '.join(sorted(f'{s[7:15]} ({ex.subset.get(s, chr(63))}, {fr.primary[s]})' for s in o_r - o_m))}); "
          f"fixed {len(o_m - o_r)} ({', '.join(sorted(s[7:15] for s in o_m - o_r))}).",
          f"- of the {len(ev)} OFFROAD-MV event scenes, {len(ev & common)} are in the PAI-FIX exam; R5-R offroad on {len(ev & o_r)} of them, R3MV-vl on "
          f"{len(ev & o_m)}: {', '.join(sorted(s[7:15] for s in ev & common)) or 'none'}.",
          f"- R5-R's offroads are {sum(ex.subset.get(s, '').endswith('TURN') for s in o_r)} / {len(o_r)} on TURN-subset scenes: the PAI-FIX exam samples "
          "turns and stops, not the straight-road / slow-down scenes where the 281 events sit, so R5-R cannot say whether the hinge would touch them."]
    md = RES / "offroad_mv.md"                         # hand-written reading below the marker is kept across re-runs
    keep = md.read_text().split(MARK, 1) if md.exists() else []
    head = keep[0].split(GEN, 1)[0] if keep and GEN in keep[0] else ""
    md.write_text(head + GEN + "\n\n" + "\n".join(L[1:]) + "\n\n" + MARK + (keep[1] if len(keep) == 2 else "\n"))
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    for c in ("events", "data", "report"):
        sp.add_parser(c)
    a = p.parse_args()
    dict(events=cmd_events, data=cmd_data, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
