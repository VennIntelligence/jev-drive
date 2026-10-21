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
    for n in list(READS.values()) + [NTV0, "R3MV-L18-s0-vl", "R5R-L18-s0-vl", "NTV0-L18-s0-vl"]:
        for sub, ext in (("facts", "csv"), ("tax", "pkl")):
            src, dst = DATA / f"runs/pai_fix/{sub}/{n}.{ext}", S / sub / f"{n}.{ext}"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
    for f in list((O / "dec").glob("dec_*.pkl")) + [DATA / "runs/round5/d1/dec_NTV0-281ag.pkl"]:
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
             arc=arc, v_plan=arc / 4.0, ds4=s4 - s0)
    if U is not None:
        r["m_plan"] = foot_margin(U, W, a[2] + pm[:, 2], dims)
        r["m_ego"] = foot_margin(U, a[None, :2], np.array([a[2]]), dims)
    return r


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
    E.to_csv(RES / "offroad_mv_events.csv", index=False, float_format="%.3f")
    P.to_csv(RES / "offroad_mv_plan.csv", index=False, float_format="%.3f")
    print(E[["read", "scene", "cls", "v_ego", "turn_deg", "lat_ev", "ntv0"]].to_string())


# ---------------------------------------------------------------- data (training rows)
VB = (2, 5, 10, 15, 20, 25, 30, 60)
TURN_CLS = (("straight", 0, 10), ("bend", 10, 25), ("turn", 25, 360))


def label_geo(v0, fut):
    """per row: peak |lateral acceleration| along the label (v x yaw rate on 0.5 s segments), |heading| at 4 s, its max, arc, |y4|, mean curvature."""
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[:, :, :2]], 1)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    yaw = np.concatenate([np.zeros((len(fut), 1)), np.unwrap(fut[:, :, 2], axis=1)], 1)
    dyaw = np.diff(yaw, axis=1)
    alat = np.abs(seg / 0.5 * dyaw / 0.5)
    arc = seg.sum(1)
    hmax = np.degrees(np.abs(yaw).max(1))
    return dict(alat=alat.max(1), alat_mean=alat.mean(1), h4=np.degrees(np.abs(yaw[:, -1])), hmax=hmax, arc=arc, y4=np.abs(fut[:, 7, 1]),
                kap=np.abs(yaw[:, -1]) / np.maximum(arc, 1.0), v4=seg[:, -1] / 0.5, sgn=np.sign(yaw[:, -1]))


def sdf_points(H, rows, pts):
    """SDF at points pts (B, P, 2) (rig frame) of the Hinge rows."""
    import torch
    from drivable_hinge import X0, Y0
    g = torch.stack([(pts[..., 1] - Y0) / 24.0 - 1.0, (pts[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
    return torch.nn.functional.grid_sample(H.sdf[rows].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]


OFFS = np.arange(0.0, 8.01, 0.25)


def clearances(H, rows, fut, bs=2048):
    """left / right clearance (m, the first lateral offset where the SDF goes below 0, capped 8 m) of the car centre (rear axle + 1.45 m)
    at t0 and at the 8 label poses -> (n, 9) left, (n, 9) right; margins of the label footprint (min, n) and of t0 (n)."""
    import torch
    sys.path[:0] = [str(HERE)]
    from road_pai import _margins
    L, R = [], []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs])
            f = np.concatenate([np.zeros((len(r), 1, 3)), fut[i:i + bs]], 1)              # (B, 9, 3)
            c, s = np.cos(f[..., 2]), np.sin(f[..., 2])
            cx, cy = f[..., 0] + 1.45 * c, f[..., 1] + 1.45 * s
            out = []
            for side in (1.0, -1.0):                                                     # left normal (-s, c)
                px = cx[..., None] - side * s[..., None] * OFFS
                py = cy[..., None] + side * c[..., None] * OFFS
                v = sdf_points(H, r, torch.as_tensor(np.stack([px, py], -1).reshape(len(r), -1, 2), dtype=torch.float32)).numpy()
                v = v.reshape(len(r), 9, len(OFFS))
                neg = v < 0
                out.append(np.where(neg.any(-1), OFFS[np.argmax(neg, -1)], 8.0))
            L.append(out[0]), R.append(out[1])
    m0, ml, _ = _margins(H, rows, fut.astype(np.float32))
    return np.concatenate(L), np.concatenate(R), ml, m0


def summarise(name, v0, fut, w, Lc, Rc, ml, m0):
    """tables by state class (turn class of the label x speed bin), draw-weighted."""
    G = label_geo(v0.astype(np.float64), fut.astype(np.float64))
    mov = v0 >= 2
    out = dict(name=name, rows=int(mov.sum()), w_moving=float(w[mov].sum()))

    def wq(x, ww, q):
        i = np.argsort(x)
        c = np.cumsum(ww[i])
        return [float(x[i][min(np.searchsorted(c, qq * c[-1]), len(x) - 1)]) for qq in q] if len(x) else [np.nan] * len(q)

    def wm(m, ww):
        return float((ww * m).sum() / max(ww.sum(), 1e-12))

    cls = {}
    for cn, lo, hi in TURN_CLS:
        for vlo, vhi in zip(VB[:-1], VB[1:]):
            m = mov & (G["hmax"] >= lo) & (G["hmax"] < hi) & (v0 >= vlo) & (v0 < vhi)
            if m.sum() < 20:
                continue
            ww = w[m]
            # inside / outside clearance for bend / turn rows: inside = the side the label turns to
            sg = G["sgn"][m]
            ins = np.where(sg[:, None] > 0, Lc[m], Rc[m])
            ous = np.where(sg[:, None] > 0, Rc[m], Lc[m])
            jmin = ins[:, 1:].argmin(1) + 1
            ins_min = ins[np.arange(m.sum()), jmin]
            rec = dict(n=int(m.sum()), w=float(ww.sum()), alat_q=wq(G["alat"][m], ww, (0.5, 0.9, 0.99)),
                       alat_gt3=wm(G["alat"][m] > 3, ww), alat_gt4=wm(G["alat"][m] > 4, ww), kap_q=wq(G["kap"][m], ww, (0.5, 0.9)),
                       y4_q=wq(G["y4"][m], ww, (0.5, 0.9, 0.99)), y4_gt1=wm(G["y4"][m] > 1.0, ww), y4_gt2=wm(G["y4"][m] > 2.0, ww),
                       mlab_q=wq(ml[m], ww, (0.05, 0.25, 0.5)), mlab_lt03=wm(ml[m] < 0.3, ww), mlab_lt0=wm(ml[m] < 0, ww),
                       m0_lt03=wm(m0[m] < 0.3, ww))
            if cn != "straight":
                rec |= dict(ins0_q=wq(ins[:, 0], ww, (0.25, 0.5)), ins_min_q=wq(ins_min, ww, (0.1, 0.25, 0.5)), out_min_q=wq(ous[:, 1:].min(1), ww, (0.25, 0.5)),
                            ins_drop_q=wq(ins[:, 0] - ins_min, ww, (0.5, 0.9)), ins_tight=wm(ins_min < 1.6, ww),
                            near_inside=wm(ins.min(1) < ous.min(1), ww))
            cls[f"{cn}|{vlo}-{vhi}"] = rec
    out["cls"] = cls
    # speed coverage and lateral acceleration overall (moving, draw-weighted)
    out["v_q"] = wq(v0[mov], w[mov], (0.5, 0.9, 0.99))
    out["v_gt30"] = wm(v0[mov] > 30, w[mov])
    out["alat_gt3"] = wm(G["alat"][mov] > 3, w[mov])
    out["alat_gt4"] = wm(G["alat"][mov] > 4, w[mov])
    out["mlab_lt03"] = wm(ml[mov] < 0.3, w[mov])
    return out, G


def cmd_data(a):
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
    res, per = {}, {}
    for p, pool in r3.PARTS.items():
        t = r3.load_part(p)
        nm = t["names"].astype(str)
        ix = np.array([idx[f"on:{p}:{x[3:]}"] for x in nm])
        w3, w5 = B["w"][ix].astype(np.float64), wl[ix].astype(np.float64)
        H = Hinge(DATA / f"runs/round5/road_pai/road_{pool}.npz", nm, "cpu", 0.3, bank=True)
        ok = H.ok.numpy()
        rows = np.flatnonzero(ok)
        Lc, Rc, ml, m0 = (np.full((len(nm), 9), np.nan), np.full((len(nm), 9), np.nan), np.full(len(nm), np.nan), np.full(len(nm), np.nan))
        lc, rc, mlr, m0r = clearances(H, rows, t["fut"][rows].astype(np.float64))
        Lc[rows], Rc[rows], ml[rows], m0[rows] = lc, rc, mlr, m0r
        v0 = t["speed"].astype(np.float64)
        sel = ok
        res[p], G = summarise(p, v0[sel], t["fut"][sel], w3[sel], Lc[sel], Rc[sel], ml[sel], m0[sel])
        res[p + "|R5LC"], _ = summarise(p + " (MV-LC draw)", v0[sel], t["fut"][sel], w5[sel], Lc[sel], Rc[sel], ml[sel], m0[sel])
        res[p]["raster_ok_drawn"] = float((w3 * ok).sum() / w3.sum())
        per[p] = dict(names=nm, v0=v0, ok=ok, ml=ml, alat=label_geo(v0, t["fut"].astype(np.float64))["alat"], hmax=label_geo(v0, t["fut"].astype(np.float64))["hmax"], w=w3)
        del H
        print(p, json.dumps({k: v for k, v in res[p].items() if k != "cls"}), flush=True)
    # stretch f = 1.5 rows against their plain base row (same t0 pose, path over 1.5 x the time)
    ph, p15 = per["ph"], per["ph15"]
    base = {x.split(":", 2)[2]: i for i, x in enumerate(ph["names"])}
    j = np.array([base.get(x.split(":", 2)[2], -1) for x in p15["names"]])
    m = (j >= 0) & (p15["w"] > 0)
    r15 = dict(paired=int(m.sum()), v_ratio_med=float(np.median(p15["v0"][m] / np.maximum(ph["v0"][j[m]], 0.1))),
               alat_ratio_med=float(np.median(p15["alat"][m] / np.maximum(ph["alat"][j[m]], 0.05))),
               alat_gt3_ph15=float((p15["alat"][m] > 3).mean()), alat_gt3_base=float((ph["alat"][j[m]] > 3).mean()),
               mlab_lt03_ph15=float(np.nanmean(p15["ml"][m] < 0.3)), mlab_lt03_base=float(np.nanmean(ph["ml"][j[m]] < 0.3)),
               turn_rows_alat_gt3_ph15=float((p15["alat"][m & (p15["hmax"] >= 25)] > 3).mean()))
    res["stretch15"] = r15
    # navtrain (uniform; the nuPlan road raster of the S rows' hinge)
    tabs = [np.load(CR / f"navtrain_full.s{i}of12" / "tab.npz") for i in range(12)]
    tok = np.concatenate([z["names"] for z in tabs]).astype(str)
    fut = np.concatenate([z["fut"] for z in tabs]).astype(np.float64)
    v0 = np.concatenate([z["speed"] for z in tabs]).astype(np.float64)
    H = Hinge(DATA / "runs/body1/labels_road/navtrain_s01234567891011.npz", tok, "cpu", 0.3, bank=True)
    ok = H.ok.numpy() & np.isfinite(fut.reshape(len(fut), -1)).all(1)
    rows = np.flatnonzero(ok)
    lc, rc, mlr, m0r = clearances(H, rows, fut[rows])
    res["navtrain"], _ = summarise("navtrain", v0[rows], fut[rows], np.ones(len(rows)), lc, rc, mlr, m0r)
    res["navtrain"]["raster_ok"] = float(ok.mean())
    print("navtrain", json.dumps({k: v for k, v in res["navtrain"].items() if k != "cls"}), flush=True)
    O.mkdir(parents=True, exist_ok=True)
    (O / "data.json").write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------- report
def cmd_report(a):
    import pandas as pd
    stage()
    D = _d1()
    E = pd.read_csv(RES / "offroad_mv_events.csv")
    P = pd.read_csv(RES / "offroad_mv_plan.csv")
    X = json.loads((O / "data.json").read_text()) if (O / "data.json").exists() else None
    print(len(E), len(P), X is not None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    for c in ("events", "data", "report"):
        sp.add_parser(c)
    a = p.parse_args()
    dict(events=cmd_events, data=cmd_data, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
