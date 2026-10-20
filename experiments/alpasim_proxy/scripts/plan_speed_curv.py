#!/usr/bin/env python3
"""PLAN-SPEED-CURV: does the plan's speed react to the curvature of the road ahead the way the human's does?
Diagnostic, CPU only, stored reads; pre-registered in plans/2026-10-20-plan-speed-curv-prereg.md.

  $DATA/envs/jevdrive/bin/python plan_speed_curv.py logs --out F   (box) the full nuPlan ego log around every held arena window
        (frames start_idx - 10 s .. + 30 s, 20 Hz: xy, yaw, speed) -> F (pickle: id -> dict(xy, yaw, v, f0))
  tmp/pcs_local/.venv/bin/python plan_speed_curv.py feat     (Mac) per-decision plan / human speed response and road geometry ahead,
        5 reads -> tmp/plan_speed_curv/feat.parquet
  tmp/pcs_local/.venv/bin/python plan_speed_curv.py stats    (Mac) slopes, ratios, window-clustered bootstrap, phase and fail / pass
        splits -> results/plan_speed_curv.json, results/plan_speed_curv/*.csv
  tmp/pcs_local/.venv/bin/python plan_speed_curv.py fig      (Mac) -> results/figs/plan_speed_curv.{png,pdf}

Frames: decisions (x, y, yaw, poses) live in the window frame shared with hum.pkl (the 5.5 s logged track). The full log is moved into
that frame by a 2D rigid fit of log frames start_idx + round(20 T) onto hum.pkl's points (median residual reported; windows > 0.10 m
dropped). s_e = the ego anchor projected onto the full log polyline near the window; t_h = the first time the human was at s_e.
Speeds: plan dv02 = a(2 s) / 2 - v_ego, dv24 = (a(4 s) - a(2 s)) / 2 - v_ego (a = plan arc); human dv02_h, dv24_h the same from t_h on
the log, relative to v_h = the human's speed at s_e. Geometry: kmax = turn_sep.kmax_1m of the logged path at 1 m stations from s_e
(the trainer's gpath view, 0 .. 20 m ahead), dh20 = heading change over the next 20 m (+ = left), done20 = over the last 20 m.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
RES = REPO / "experiments/alpasim_proxy/results"
TMP = REPO / "tmp"
OUT = TMP / "plan_speed_curv"
HZ, PRE_S, POST_S = 20, 10.0, 30.0
TAGS = ("STANDFMXH-L18-s1", "STANDFMXH-L18-s0", "JTN-L18-s1", "JTN-L18-s0", "RSG-L18-s0")
SHORT = {"STANDFMXH-L18-s1": "B2 s1", "STANDFMXH-L18-s0": "B2 s0", "JTN-L18-s1": "JTN s1", "JTN-L18-s0": "JTN s0", "RSG-L18-s0": "RSG s0"}
# pre-registered
V_MIN, TURN_MIN, LEAD_MIN, STOP_V, K_CLIP, K_TIGHT = 2.0, 15.0, 30.0, 1.0, 0.25, 0.108
FIT_MAX, NBOOT, SEED = 0.10, 2000, 0
BINS = (0.0, 0.03, 0.06, 0.09, 0.108, 0.15, 0.25)
REACT, NOREACT = 0.5, 0.25


# ---------------------------------------------------------------- box: full logs
def _yaw_q(qw, qx, qy, qz):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _log_job(job):
    log, wins, raw = job
    c = sqlite3.connect(f"file:{raw}/{log}.db?mode=ro", uri=True)
    r = c.execute("select e.x, e.y, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy from lidar_pc l join ego_pose e on e.token = l.ego_pose_token "
                  "order by l.timestamp").fetchall()
    a = np.array(r, float)
    xy, yaw, v = a[:, :2], _yaw_q(*a[:, 2:6].T), np.hypot(a[:, 6], a[:, 7])
    out = {}
    for wid, s0 in wins:
        lo, hi = max(0, s0 - int(PRE_S * HZ)), min(len(xy), s0 + int(POST_S * HZ) + 1)
        out[wid] = dict(xy=xy[lo:hi], yaw=yaw[lo:hi], v=v[lo:hi].astype(np.float32), f0=s0 - lo)
    return out


def cmd_logs(a):
    from jevdrive.par import pmap
    data = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
    S = pd.read_csv(data / "runs/alpasim_proxy/arena/set/set.csv.gz").drop_duplicates("id")
    S = S[S.split == "held"]
    jobs = [(log, [(w.id, int(w.start_idx)) for w in g.itertuples()], str(data / "datasets/nuplan_raw/test")) for log, g in S.groupby("log")]
    res = pmap(_log_job, jobs, desc="logs")
    out = {k: v for d in res.values if d for k, v in d.items()}
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(f"{len(out)} windows from {len(jobs)} logs, {len(res.errors)} failed logs")
    for e in list(res.errors.values())[:3]:
        print(e)


# ---------------------------------------------------------------- Mac: per-decision features
def rigid(A, B):
    """2D rigid transform (R, t) with R A + t ~ B (Kabsch)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return R, cb - R @ ca


def path_1m(S, X, s, lo, hi):
    g = s + np.arange(lo, hi + 1.0)
    return np.c_[np.interp(g, S, X[:, 0]), np.interp(g, S, X[:, 1])]


def heading(P, i, j):
    d = P[j] - P[i]
    return np.arctan2(d[1], d[0])


def dang(a, b):
    return float(np.degrees(np.angle(np.exp(1j * (b - a)))))


def _feat_one(job):
    from turn_sep import kmax_1m
    wid, h, L, D, t_ev = job
    T = np.asarray(h["T"], float)
    f = L["f0"] + np.round(HZ * T / 1e6).astype(int)
    ok = f < len(L["xy"])
    if ok.sum() < 20:
        return [], dict(id=wid, fit=np.nan)
    R, tr = rigid(L["xy"][f[ok]], np.asarray(h["P"], float)[ok])
    X = L["xy"] @ R.T + tr
    res = float(np.median(np.hypot(*(X[f[ok]] - np.asarray(h["P"], float)[ok]).T)))
    tl = (np.arange(len(X)) - L["f0"]) / HZ                          # window time, s
    seg = np.hypot(*np.diff(X, axis=0).T)
    S = np.r_[0.0, np.cumsum(seg)]
    keep = np.r_[True, seg > 0.01]                                   # strictly increasing stations for interpolation
    Sk, Xk = S[keep], X[keep]
    v = L["v"].astype(float)
    rows = []
    if res > FIT_MAX:
        return rows, dict(id=wid, fit=res)
    near = (tl >= -2.0) & (tl <= 12.0)                                # projection only onto the window's stretch of the log
    a_, b_ = X[:-1][near[:-1]], X[1:][near[:-1]]
    s_a = S[:-1][near[:-1]]
    for q in D.itertuples():
        if q.k < 2 or (np.isfinite(t_ev) and q.t0 >= t_ev):
            continue
        p = np.array([q.x, q.y])
        ab = b_ - a_
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        u = np.clip(((p - a_) * ab).sum(1) / L2, 0, 1)
        d = np.hypot(*(a_ + u[:, None] * ab - p).T)
        i = int(d.argmin())
        s_e = float(s_a[i] + u[i] * np.sqrt(L2[i]))
        j = int(np.searchsorted(S, s_e))                              # first passage of the human at s_e
        if j <= 0 or j >= len(S):
            continue
        w = (s_e - S[j - 1]) / max(S[j] - S[j - 1], 1e-9)
        t_h = tl[j - 1] + w * (tl[j] - tl[j - 1])
        if t_h + 4.0 > tl[-1] or S[-1] - s_e < 22:
            continue
        b = lambda t: np.interp(t_h + t, tl, S) - s_e  # noqa: E731
        vh = float(np.interp(t_h, tl, v))
        P = path_1m(Sk, Xk, s_e, -21, 22)                              # stations s_e - 21 .. s_e + 22, index 21 = s_e
        km = float(kmax_1m(P[None, 20:44], 0, 20)[0])               # s_e - 1 .. s_e + 23, turn_sep's gpath slice
        dh20 = dang(heading(P, 21, 22), heading(P, 40, 41))
        done20 = dang(heading(P, 1, 2), heading(P, 21, 22)) if s_e - 21 >= Sk[0] else np.nan
        pz = np.asarray(q.poses, float).reshape(8, 3)
        ca = np.r_[0, np.cumsum(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], pz[:, :2]], axis=0).T))]
        tt = np.arange(0, 4.01, 0.05)
        vmin_h = float(np.interp(t_h + tt, tl, v).min())
        rows.append(dict(id=wid, k=int(q.k), t0=int(q.t0), v=float(q.v), vh=vh, t_h=t_h, s_e=s_e - S[L["f0"]], s_e_hum=float(q.s_e),
                         lead_gap=float(q.lead_gap), kmax=km, dh20=dh20, done20=done20,
                         dv02=ca[4] / 2 - q.v, dv24=(ca[8] - ca[4]) / 2 - q.v, arc4=ca[8],
                         dv02_h=b(2.0) / 2 - vh, dv24_h=(b(4.0) - b(2.0)) / 2 - vh, arc4_h=b(4.0), vmin_h=vmin_h))
    return rows, dict(id=wid, fit=res)


def cmd_feat(a):
    from multiprocessing import Pool
    from turn_sep import exit_time
    H = pickle.load(open(TMP / "lat_zero/hum.pkl", "rb"))
    LG = pickle.load(open(OUT / "logs.pkl", "rb"))
    F, FIT = [], []
    for t in TAGS:
        ext = TMP / ("easy_items_B2/ext_full_H" if t == "STANDFMXH-L18-s0" else "easy_items_B2/ext_full_H_s1")
        D = pd.read_parquet(ext / "dec.parquet")
        D = D[D.tag == t]
        St = pd.read_parquet(ext / "steps.parquet")
        tev = St[St.tag == t].groupby("id").apply(exit_time).to_dict()
        jobs = [(i, H[i], LG[i], g, tev.get(i, np.nan)) for i, g in D.groupby("id") if i in H and i in LG]
        with Pool(a.workers) as p:
            out = p.map(_feat_one, jobs, chunksize=8)
        G = pd.DataFrame([r for o, _ in out for r in o])
        G["tag"] = t
        G["t_ev"] = G.id.map(tev)
        W = pd.read_parquet(ext / "win.parquet")
        G = G.merge(W[W.tag == t][["id", "score", "type"]].drop_duplicates("id"), on="id", how="left")
        F.append(G)
        FIT += [dict(tag=t, **m) for _, m in out]
        print(t, len(G), G.id.nunique(), flush=True)
    F = pd.concat(F, ignore_index=True)
    F.to_parquet(OUT / "feat.parquet")
    FT = pd.DataFrame(FIT)
    FT.to_csv(OUT / "fit.csv", index=False)
    print("fit residual q50/q90/q99", FT.fit.quantile([0.5, 0.9, 0.99]).round(3).tolist(), "dropped", int((FT.fit > FIT_MAX).sum()))
    print("s_e - s_e_hum (m) q01/q50/q99", (F.s_e - F.s_e_hum).quantile([0.01, 0.5, 0.99]).round(2).tolist())


# ---------------------------------------------------------------- Mac: statistics
def frame(F, tag, clean=True):
    G = F[(F.tag == tag) & (F.v >= V_MIN)].copy()
    G["x"] = G.kmax.clip(0, K_CLIP)
    G["side"] = np.where(G.dh20 <= -TURN_MIN, "right", np.where(G.dh20 >= TURN_MIN, "left", "straight"))
    G["lead"] = G.lead_gap < LEAD_MIN
    G["stop"] = G.vmin_h < STOP_V
    G["clean"] = ~G.lead & ~G.stop
    return G[G.clean] if clean else G


def ols_boot(G, y, x="x", nboot=NBOOT, cov=None, rng=None):
    """slope of y on x (optionally with one covariate), point + window-clustered bootstrap draws."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    ids, inv = np.unique(G.id.to_numpy(), return_inverse=True)
    cols = [np.ones(len(G)), G[x].to_numpy()] + ([G[cov].to_numpy()] if cov else [])
    Xm = np.stack(cols, 1)
    Y = np.stack([G[c].to_numpy() for c in y], 1)                    # several responses share the resampling
    p = Xm.shape[1]
    XX = np.zeros((len(ids), p, p))
    XY = np.zeros((len(ids), p, Y.shape[1]))
    np.add.at(XX, inv, Xm[:, :, None] * Xm[:, None, :])
    np.add.at(XY, inv, Xm[:, :, None] * Y[:, None, :])
    W = np.r_[[np.ones(len(ids))], rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=nboot)]
    A = np.einsum("bw,wij->bij", W, XX)
    B = np.einsum("bw,wij->bij", W, XY)
    beta = np.linalg.solve(A + 1e-9 * np.eye(p), B)                   # (nboot + 1, p, ny)
    return beta[:, 1, :]                                              # slope on x, row 0 = point estimate


def ci(v):
    return [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]


def verdict(bm, bh, r):
    if not (ci(bh)[1] < 0):
        return "not judged (human slope CI includes 0)"
    lo, hi = ci(bm)
    if hi >= 0 or r[0] < NOREACT:
        return "does not react"
    if r[0] >= REACT and hi < 0:
        return "reacts"
    return "partial"


def cmd_stats(a):
    F = pd.read_parquet(OUT / "feat.parquet")
    sh = pd.read_csv(TMP / "lat_zero/sharp_windows.csv")
    rows, phase, binr, fp = [], [], [], []
    J = dict(rule=dict(v_min=V_MIN, turn_min=TURN_MIN, lead_min=LEAD_MIN, stop_v=STOP_V, k_clip=K_CLIP, react=REACT, noreact=NOREACT,
                       nboot=NBOOT))
    for t in TAGS:
        for clean in (True, False):
            G = frame(F, t, clean)
            for side in ("right", "left", "turns", "all"):
                g = G[G.side == side] if side in ("right", "left") else G[G.side != "straight"] if side == "turns" else G
                if g.id.nunique() < 10:
                    continue
                for resp in ("dv24", "dv02", "arc4"):
                    yh = resp + "_h"
                    b = ols_boot(g, [resp, yh])
                    bm, bh = b[:, 0], b[:, 1]
                    r = bm / bh
                    rec = dict(tag=t, read=SHORT[t], set="clean" if clean else "all", side=side, resp=resp, n_dec=len(g),
                               n_win=g.id.nunique(), b_m=bm[0], b_m_lo=ci(bm[1:])[0], b_m_hi=ci(bm[1:])[1], b_h=bh[0],
                               b_h_lo=ci(bh[1:])[0], b_h_hi=ci(bh[1:])[1], ratio=r[0], ratio_lo=ci(r[1:])[0], ratio_hi=ci(r[1:])[1],
                               verdict=verdict(bm[1:], bh[1:], r) if resp == "dv24" else "")
                    if resp == "dv24":                                # robustness: current speed as covariate (plan v_ego, human v_h)
                        bmc = ols_boot(g, ["dv24"], cov="v")[:, 0]
                        bhc = ols_boot(g, ["dv24_h"], cov="vh")[:, 0]
                        rec.update(b_m_cov=bmc[0], b_m_cov_lo=ci(bmc[1:])[0], b_m_cov_hi=ci(bmc[1:])[1], b_h_cov=bhc[0],
                                   b_h_cov_lo=ci(bhc[1:])[0], b_h_cov_hi=ci(bhc[1:])[1], ratio_cov=bmc[0] / bhc[0],
                                   ratio_cov_ci=ci(bmc[1:] / bhc[1:]))
                    rows.append(rec)
            if not clean:
                continue
            # binned curves (clean, per side incl. straight)
            for side in ("right", "left"):
                g = G[G.side.isin([side, "straight"])]
                g = g.assign(bin=pd.cut(g.kmax, BINS, include_lowest=True, right=False))
                for bn, gb in g.groupby("bin", observed=True):
                    if len(gb) < 15:
                        continue
                    rng = np.random.default_rng(SEED)
                    ids = gb.id.unique()
                    dm = (gb.dv24 - gb.dv24_h).groupby(gb.id).agg(["sum", "count"])
                    mm, mh = gb.dv24.groupby(gb.id).sum(), gb.dv24_h.groupby(gb.id).sum()
                    Wb = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=NBOOT)
                    n = Wb @ dm["count"].reindex(ids).to_numpy()
                    binr.append(dict(read=SHORT[t], side=side, bin=str(bn), k_mid=float(gb.kmax.median()), n_dec=len(gb), n_win=len(ids),
                                     plan=gb.dv24.mean(), plan_ci=ci(Wb @ mm.reindex(ids).to_numpy() / n),
                                     human=gb.dv24_h.mean(), human_ci=ci(Wb @ mh.reindex(ids).to_numpy() / n),
                                     diff=(gb.dv24 - gb.dv24_h).mean(), diff_ci=ci(Wb @ dm["sum"].reindex(ids).to_numpy() / n)))
            # phase in sharp turns (total >= 45 deg over -20 .. +20 m, same sign)
            tot = G.done20 + G.dh20
            S2 = G[(np.sign(G.done20) * np.sign(G.dh20) >= 0) & (tot.abs() >= 45)].copy()
            S2["frac"] = S2.done20.abs() / (S2.done20.abs() + S2.dh20.abs())
            S2["phase"] = pd.cut(S2.frac, [0, 1 / 3, 2 / 3, 1.0001], labels=["approach", "inside", "exit"], right=False)
            S2["tside"] = np.where((S2.done20 + S2.dh20) < 0, "right", "left")
            for (sd, ph), gp in S2.groupby(["tside", "phase"], observed=True):
                if gp.id.nunique() < 5:
                    continue
                d = (gp.dv24 - gp.dv24_h).groupby(gp.id).agg(["sum", "count"])
                rng = np.random.default_rng(SEED)
                Wb = rng.multinomial(len(d), np.full(len(d), 1 / len(d)), size=NBOOT)
                phase.append(dict(read=SHORT[t], side=sd, phase=ph, n_dec=len(gp), n_win=len(d), plan=gp.dv24.mean(), human=gp.dv24_h.mean(),
                                  diff=(gp.dv24 - gp.dv24_h).mean(), diff_ci=ci(Wb @ d["sum"].to_numpy() / (Wb @ d["count"].to_numpy())),
                                  v=gp.v.mean(), vh=gp.vh.mean(), kmax=gp.kmax.median()))
            # d397's sharp set: failing (3 .. 1 s before exit) vs passing, tight decisions
            s = sh[sh.tag == t].drop_duplicates("id").set_index("id")
            H = G[G.id.isin(s.index) & (G.kmax >= K_TIGHT) & (G.side != "straight")].copy()
            H["fail"] = H.id.map(s.latzero).fillna(False).astype(bool)
            H["pass"] = H.score >= 1 - 1e-4
            pre = H.fail & (H.t0 >= H.t_ev - 3e6) & (H.t0 <= H.t_ev - 1e6)
            for nm, m in (("fail 3-1 s pre-exit", pre), ("pass", H["pass"])):
                g = H[m]
                if not len(g):
                    continue
                fp.append(dict(read=SHORT[t], group=nm, n_dec=len(g), n_win=g.id.nunique(), plan=g.dv24.mean(), human=g.dv24_h.mean(),
                               diff=(g.dv24 - g.dv24_h).mean(), v_ratio=float((g.v / g.vh.clip(lower=0.3)).median()),
                               kmax=float(g.kmax.median())))
    A = pd.DataFrame(rows)
    (RES / "plan_speed_curv").mkdir(parents=True, exist_ok=True)
    A.to_csv(RES / "plan_speed_curv/slopes.csv", index=False, float_format="%.4f")
    for nm, X in (("bins", binr), ("phase", phase), ("failpass", fp)):
        pd.DataFrame(X).to_csv(RES / f"plan_speed_curv/{nm}.csv", index=False, float_format="%.4f")
    G0 = pd.concat([frame(F, t, False) for t in TAGS])
    J["counts"] = {SHORT[t]: dict(dec=int((G0.tag == t).sum()), clean=int(((G0.tag == t) & G0.clean).sum()),
                                  lead=int(((G0.tag == t) & G0.lead).sum()), stop=int(((G0.tag == t) & G0.stop).sum()),
                                  right=int(((G0.tag == t) & G0.clean & (G0.side == "right")).sum()),
                                  left=int(((G0.tag == t) & G0.clean & (G0.side == "left")).sum())) for t in TAGS}
    J["slopes"] = A.to_dict(orient="records")
    J["phase"], J["failpass"], J["bins"] = phase, fp, binr
    json.dump(J, open(RES / "plan_speed_curv.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    c = ["read", "set", "side", "resp", "n_dec", "n_win", "b_m", "b_m_lo", "b_m_hi", "b_h", "b_h_lo", "b_h_hi", "ratio", "ratio_lo", "ratio_hi", "verdict"]
    print(A[c].to_string(float_format=lambda v: f"{v:.3f}"))
    print(A[A.resp == "dv24"][["read", "set", "side", "b_m_cov", "b_m_cov_lo", "b_m_cov_hi", "b_h_cov", "b_h_cov_lo", "b_h_cov_hi", "ratio_cov"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(phase).to_string(float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(fp).to_string(float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(binr)[["read", "side", "bin", "n_dec", "n_win", "plan", "human", "diff"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(J["counts"], indent=1))


def cmd_fig(a):
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "research"))
    import plot_style as PS
    PS.apply()
    C = PS.PALETTE
    col = {"B2 s1": C["vermillion"], "B2 s0": C["orange"], "JTN s1": C["blue"], "JTN s0": C["sky_blue"], "RSG s0": C["green"]}
    A = pd.read_csv(RES / "plan_speed_curv/slopes.csv")
    Bn = pd.read_csv(RES / "plan_speed_curv/bins.csv")
    Ph = pd.read_csv(RES / "plan_speed_curv/phase.csv")
    fig, ax = plt.subplots(2, 2, figsize=(PS.DOUBLE_COLUMN_IN, 4.6))
    for j, side in enumerate(("right", "left")):
        a_ = ax[0, j]
        b = Bn[Bn.side == side]
        h = b[b.read == "B2 s1"]
        lo, hi = np.array([eval(x) for x in h.human_ci]).T
        a_.fill_between(h.k_mid, lo, hi, color=PS.BASELINE, alpha=0.25, lw=0)
        a_.plot(h.k_mid, h.human, "-o", color=PS.BASELINE, ms=3, label="human (same states, B2 s1)")
        for r in ("B2 s1", "B2 s0", "JTN s1", "RSG s0"):
            g = b[b.read == r]
            a_.plot(g.k_mid, g.plan, "-o", color=col[r], ms=3, label=f"plan {r}")
        a_.axvline(K_TIGHT, color="#999999", ls="--", lw=0.6)
        PS.zero_line(a_)
        a_.set_xlabel("max path curvature in next 20 m (1/m)")
        a_.set_ylabel("mean speed 2-4 s minus now (m/s)")
        PS.panel(a_, f"({'ab'[j]}) {side} turns + straight, clean")
        if j == 0:
            a_.legend(fontsize=6, frameon=False, loc="upper left")
    a_ = ax[1, 0]
    S = A[(A.resp == "dv24") & (A.set == "clean") & A.side.isin(["right", "left"])]
    reads = list(SHORT.values())
    for i, r in enumerate(reads):
        for side in ("right", "left"):
            g = S[(S.read == r) & (S.side == side)]
            if not len(g):
                continue
            g = g.iloc[0]
            y = i + (-0.2 if side == "right" else 0.2)
            mk = "o" if side == "right" else "s"
            a_.errorbar(g.b_h_cov, y - 0.08, xerr=[[g.b_h_cov - g.b_h_cov_lo], [g.b_h_cov_hi - g.b_h_cov]], fmt=mk, color=PS.BASELINE,
                        mfc=PS.BASELINE if side == "right" else "white", ms=3.5, capsize=1.5, lw=0.8)
            a_.errorbar(g.b_m_cov, y + 0.08, xerr=[[g.b_m_cov - g.b_m_cov_lo], [g.b_m_cov_hi - g.b_m_cov]], fmt=mk, color=col[r],
                        mfc=col[r] if side == "right" else "white", ms=3.5, capsize=1.5, lw=0.8)
    a_.axvline(0, color="#999999", lw=0.6)
    a_.set_yticks(range(len(reads)), reads)
    a_.invert_yaxis()
    a_.set_xlabel("slope of dv 2-4 s on curvature, speed held fixed (m/s per 1/m)")
    PS.panel(a_, "(c) grey = human, colour = plan; filled right, open left")
    a_ = ax[1, 1]
    ph = Ph[Ph.side == "right"]
    order = ["approach", "inside", "exit"]
    for i, r in enumerate(reads):
        g = ph[ph.read == r].set_index("phase").reindex(order)
        x = np.arange(3) + (i - 2) * 0.13
        lo, hi = np.array([eval(v) if isinstance(v, str) else [np.nan, np.nan] for v in g.diff_ci]).T
        a_.errorbar(x, g["diff"], yerr=[g["diff"] - lo, hi - g["diff"]], fmt="o", color=col[r], ms=3, capsize=1.5, lw=0.8, label=r)
    PS.zero_line(a_)
    a_.set_xticks(range(3), order)
    a_.set_ylabel("plan minus human, dv 2-4 s (m/s)")
    PS.panel(a_, "(d) sharp right turns (>= 45 deg) by phase")
    a_.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    PS.save(fig, RES / "figs/plan_speed_curv")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("logs")
    p.add_argument("--out", required=True)
    p = sp.add_parser("feat")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    sp.add_parser("stats")
    sp.add_parser("fig")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
