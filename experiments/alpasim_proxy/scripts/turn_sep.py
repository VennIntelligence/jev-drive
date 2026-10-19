#!/usr/bin/env python3
"""TURN-SEP: is there a decision-time state that separates B2's failing sharp turns from the ones it passes? (decision 396's open question)
Diagnostic, CPU only, stored reads; candidates and success line pre-registered in plans/2026-10-20-turn-sep-prereg.md.

  $DATA/envs/jevdrive/bin/python turn_sep.py traces --out F   (box) executed controller traces (10 Hz) of garage 0014 and B2 s1 / s0 on
        the held windows -> F (parquet: tag, id, t, x, y, yaw, v)
  tmp/pcs_local/.venv/bin/python turn_sep.py feat              (Mac) per-decision candidate states, 5 reads x all held windows
        -> tmp/turn_sep/feat.parquet
  tmp/pcs_local/.venv/bin/python turn_sep.py auc               (Mac) AUC table, 1 % fire thresholds, coverage, combo (leave-one-window-out)
        -> results/turn_sep.json, results/turn_sep/auc.csv
  tmp/pcs_local/.venv/bin/python turn_sep.py garage            (Mac) garage vs B2 executed turns -> results/turn_sep/garage.csv
  tmp/pcs_local/.venv/bin/python turn_sep.py fig               (Mac) -> results/figs/turn_sep.{png,pdf}

Frames: the driver anchors (dec x, y, yaw), the human track (hum.pkl) and the controller traces share the window's local frame.
Signs: sigma = sign of the human path's heading change over the next 20 m from the ego's station (+ = left); outward = -sigma x the
left-positive offset from the human path, relative to the ego's offset at the decision. Every candidate is oriented larger = riskier.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/scripts")]
RES = REPO / "experiments/alpasim_proxy/results"
TMP = REPO / "tmp"
OUT = TMP / "turn_sep"
B2 = {"s1": "STANDFMXH-L18-s1", "s0": "STANDFMXH-L18-s0"}
EXT = {"STANDFMXH-L18-s0": TMP / "easy_items_B2/ext_full_H"}          # every other tag from ext_full_H_s1 (lat_zero's convention)
EXT_DEF = TMP / "easy_items_B2/ext_full_H_s1"
TAGS = ("STANDFMXH-L18-s1", "STANDFMXH-L18-s0", "JTN-L18-s1", "JTN-L18-s0", "RSG-L18-s0")
TARGETS = ("098f5c53", "202f583c", "3f805a18", "476b54f9", "6f27563e", "87705868", "9e7c5a46", "efe356db", "f2305669")
A2_S0 = TARGETS + ("41a358be",)
# pre-registered (plans/2026-10-20-turn-sep-prereg.md)
V_MIN, TURN_MIN, SPAN = 2.0, 15.0, (3.0, 1.0)
AUC_LINE, FIRE_LINE, TGT_LINE, FAIL_LINE = 0.85, 0.01, 6, 0.5
MAPFREE = ("ratio", "arc", "out4", "out8", "out_end", "kdef", "alat", "alat_plan", "lag", "lat0")
GEOM = ("kmax", "kmax1m", "turn_deg", "right")
ORACLE = ("edge",)
FEATS = MAPFREE + GEOM + ORACLE


# ---------------------------------------------------------------- box: executed traces
def cmd_traces(a):
    sys.path.insert(0, str(REPO / "experiments/defect_fix/scripts"))
    import free_gate as FG
    data = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
    reads = {"garage": data / "runs/alpasim_proxy/arena/reads/G0014",
             **{t: data / f"runs/defect_fix/jt/arena/{t}@off" for t in B2.values()}}
    S = pd.read_csv(data / "runs/alpasim_proxy/arena/set/set.csv.gz").drop_duplicates("id")
    held = set(S.id[S.split == "held"])
    rows = []
    for tag, d in reads.items():
        for run in sorted(glob.glob(f"{d}/c*/run")):
            f = Path(run) / "aggregate/results-summary.json"
            if not f.exists():
                continue
            for r in json.load(open(f))["rollouts"]:
                c = Path(run) / f"controller/alpasim_controller_{r['rollout_id']}.csv"
                if r["clipgt_id"] not in held or not c.exists():
                    continue
                C = FG.ctrl(c)
                rows.append(pd.DataFrame(dict(tag=tag, id=r["clipgt_id"], t=C["t"], x=C["xy"][:, 0], y=C["xy"][:, 1], yaw=C["yaw"], v=C["v"])))
    T = pd.concat(rows, ignore_index=True)
    T.to_parquet(a.out)
    print(T.groupby("tag").id.nunique().to_dict())


def cmd_pool(a):
    """(box) the draft term's domain in the on-policy pool: S rows, ego >= V_MIN, path heading change >= TURN_MIN over 20 m,
    kmax_1m >= --k; and how many of the stored served plans sit outward of the logged path at 4 m of arc by > --m."""
    data = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
    CR, GI = data / "runs/op_parity/cache", 15
    rows_dir = {"rb0": "L18RB0", "rtb1": "L18RTB1", "ys": "L18YS", "y": "L18Y", **{f"pn{i}": f"PN{i}" for i in range(7)}}
    tab = np.load(CR / a.pool / "tab.npz")
    names = tab["names"].astype(str)
    S = tab["mode"].astype(str) == "self"
    part = np.array([n.split(":")[1] for n in names])
    km, dh, o4 = (np.full(len(names), np.nan) for _ in range(3))
    for p, d in rows_dir.items():
        r = np.flatnonzero(S & (part == p))
        if not len(r):
            continue
        z = np.load(data / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        pos = {q: i for i, q in enumerate(z["names"].astype(str).tolist())}
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in r])
        g = z["gpath"][mi].astype(np.float64)
        km[r] = kmax_1m(g[:, GI - 1:GI + 23], 0, 20)
        h = np.arctan2(*np.diff(g[:, [GI, GI + 1, GI + 19, GI + 20]], axis=1)[:, [0, 2]].transpose(2, 0, 1)[::-1])
        dh[r] = np.degrees(np.angle(np.exp(1j * (h[:, 1] - h[:, 0]))))
        P = z["poses"][mi][:, :, :2].astype(np.float64)                  # ego frame, origin = ego
        pl = np.concatenate([np.zeros((len(r), 1, 2)), P], 1)
        ca = np.concatenate([np.zeros((len(r), 1)), np.cumsum(np.hypot(*np.diff(pl, axis=1).transpose(2, 0, 1)), 1)], 1)
        want = (tab["speed"][r] >= V_MIN) & (np.abs(dh[r]) >= TURN_MIN) & (km[r] >= a.k)
        for j in np.flatnonzero(want):
            if ca[j, -1] < 4:
                continue
            q = np.array([np.interp(4.0, ca[j], pl[j, :, 0]), np.interp(4.0, ca[j], pl[j, :, 1])])
            tr = Track(dict(T=np.arange(86) * 1e5, P=g[j], Y=np.zeros(86)))
            off = tr.proj(np.r_[[[0.0, 0.0]], [q]])[1]
            o4[r[j]] = -np.sign(dh[r[j]]) * (off[1] - off[0])
    v = tab["speed"]
    dom = S & (v >= V_MIN) & (np.abs(dh) >= TURN_MIN) & (km >= a.k)
    w = np.load(data / "runs/defect_fix/w/JTX00.npz")
    ww = w["w"].astype(np.float64) / w["w"].sum() if np.array_equal(w["names"].astype(str), names) else None
    rec = dict(pool=a.pool, s_rows=int(S.sum()), moving_turn=int((S & (v >= V_MIN) & (np.abs(dh) >= TURN_MIN)).sum()), k=a.k, m=a.m,
               domain=int(dom.sum()), domain_right=int((dom & (dh < 0)).sum()), domain_share_of_s=float(dom.sum() / S.sum()),
               domain_logs=int(len(np.unique(tab["log"].astype(str)[dom]))), served_out4_fire=int((dom & (o4 > a.m)).sum()),
               served_out4_q=np.nanquantile(o4[dom], [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if dom.any() else [],
               draw_share_jtx00=float(ww[dom].sum()) if ww is not None else None, kmax_q_moving_turn=np.nanquantile(
                   km[S & (v >= V_MIN) & (np.abs(dh) >= TURN_MIN)], [0.5, 0.9, 0.99]).round(4).tolist())
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- Mac: candidate states per decision
class Track:
    """human track: station grid, heading by station, curvature, signed (left +) projection."""

    def __init__(self, h):
        T, P, Y = np.asarray(h["T"], float), np.asarray(h["P"], float), np.unwrap(np.asarray(h["Y"], float))
        seg = np.hypot(*np.diff(P, axis=0).T)
        self.T, self.P = T, P
        self.S = np.r_[0.0, np.cumsum(seg)]
        keep = np.r_[True, seg > 0.02]
        self.Sk, self.Yk = self.S[keep], Y[keep]
        g = np.arange(self.Sk[0], self.Sk[-1] + 0.25, 0.25)
        y = np.interp(g, self.Sk, self.Yk)
        k = np.gradient(y, g) if len(g) > 2 else np.zeros_like(g)
        n = 8                                                       # 2 m running mean
        self.gk, self.kap = g, np.convolve(k, np.ones(n) / n, "same") if len(k) > n else k

    def proj(self, xy):
        q = np.atleast_2d(np.asarray(xy, float))[:, None]
        a, b = self.P[None, :-1], self.P[None, 1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(-1), 1e-9)
        u = np.clip(((q - a) * ab).sum(-1) / L2, 0, 1)
        c = a + u[..., None] * ab
        d = np.hypot(*(c - q).transpose(2, 0, 1))
        i = d.argmin(1)
        r = np.arange(len(i))
        cr = ab[0, i, 0] * (q[r, 0, 1] - a[0, i, 1]) - ab[0, i, 1] * (q[r, 0, 0] - a[0, i, 0])
        return self.S[i] + u[r, i] * np.sqrt(L2[0, i]), np.sign(cr) * d[r, i]

    def head(self, s):
        return np.interp(s, self.Sk, self.Yk)

    def path1m(self, s_e):
        """the logged path at 1 m stations over [s_e - 1, s_e + 22] m (clamped at the track ends), as the trainer's gpath."""
        g = s_e + np.arange(-1.0, 23.0)
        return np.c_[np.interp(g, self.S, self.P[:, 0]), np.interp(g, self.S, self.P[:, 1])]

    def kmax(self, s0, s1):
        m = (self.gk >= s0) & (self.gk <= s1)
        return float(np.abs(self.kap[m]).max()) if m.any() else np.nan


def kmax_1m(path, i0, i1):
    """path (n, m, 2) at 1 m stations -> (n,) max |curvature| (rad/m) over segments i0 .. i1, curvature = heading change between
    consecutive 1 m segments averaged over 2 m (the trainer's view of the logged path, rows meta `gpath`). Segments < 0.2 m (clamped
    ends) carry the previous heading."""
    P = np.asarray(path, np.float64)
    d = np.diff(P, axis=1)
    h = np.where(np.hypot(d[..., 0], d[..., 1]) > 0.2, np.arctan2(d[..., 1], d[..., 0]), np.nan)
    for j in range(1, h.shape[1]):
        h[:, j] = np.where(np.isnan(h[:, j]), h[:, j - 1], h[:, j])
    h = np.unwrap(np.nan_to_num(h), axis=1)
    k = np.diff(h, axis=1)
    k2 = 0.5 * (k[:, 1:] + k[:, :-1])
    return np.abs(k2[:, i0:i1]).max(1)


def plan_geom(q):
    """plan in the world frame with the anchor prepended: points (9, 2), yaw (9,) relative to the anchor, cumulative arc (9,)."""
    P = np.asarray(q.poses, float).reshape(8, 3)
    c, s = np.cos(q.yaw), np.sin(q.yaw)
    W = np.c_[q.x + c * P[:, 0] - s * P[:, 1], q.y + s * P[:, 0] + c * P[:, 1]]
    pl = np.r_[[[q.x, q.y]], W]
    yw = np.r_[0.0, np.unwrap(P[:, 2])]
    return pl, yw, np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]


def at_arc(pl, ca, L):
    return np.array([np.interp(L, ca, pl[:, 0]), np.interp(L, ca, pl[:, 1])]) if ca[-1] >= L else None


def decision(q, tr, edge):
    s_e = q.s_e
    dh20 = float(tr.head(s_e + 20) - tr.head(s_e))
    sig = 1.0 if dh20 >= 0 else -1.0
    pl, yw, ca = plan_geom(q)
    _, o = tr.proj(pl)
    off0 = o[0]
    out = lambda off: float(-sig * (off - off0))  # noqa: E731
    d = dict(dh20=np.degrees(dh20), ratio=q.v / max(q.vh, 0.3), lat0=float(-sig * off0), out_end=out(o[-1]), edge=-edge,
             kmax=tr.kmax(s_e, s_e + 20), kmax1m=float(kmax_1m(tr.path1m(s_e)[None], 0, 20)[0]), turn_deg=abs(np.degrees(dh20)), right=float(sig < 0), plan4=ca[-1])
    for L, nm in ((4.0, "out4"), (8.0, "out8")):
        p = at_arc(pl, ca, L)
        d[nm] = out(tr.proj(p)[1][0]) if p is not None else np.nan
    # plan arc / the human's arc over the same horizon from the time it was at the ego's station (4 s, or less where the 5.5 s
    # human track ends; < 1.5 s left -> nan)
    hz = min(4.0, (tr.T[-1] - q.t_h) / 1e6)
    sh = np.interp(q.t_h + hz * 1e6, tr.T, tr.S) - np.interp(q.t_h, tr.T, tr.S)
    d["arc"] = float(np.interp(hz, 0.5 * np.arange(9), ca) / max(sh, 0.5)) if hz >= 1.5 else np.nan
    L8 = min(8.0, ca[-1])
    d["kdef"] = float(sig * ((tr.head(s_e + L8) - tr.head(s_e)) - np.interp(L8, ca, yw))) if L8 >= 1.0 else np.nan
    d["alat"] = q.v ** 2 * d["kmax"]
    vp = np.diff(ca) / 0.5
    dy = np.diff(yw) / 0.5
    d["alat_plan"] = float(np.abs(vp * dy).max())
    th = 0.5 * abs(tr.head(s_e + 15) - tr.head(s_e))
    if th >= np.radians(5):
        g = np.arange(0, 15.01, 0.25)
        hh = np.abs(tr.head(s_e + g) - tr.head(s_e))
        a_h = float(g[np.argmax(hh >= th)])
        yp = sig * yw
        j = np.flatnonzero(yp >= th)
        if len(j):
            i = j[0]
            a_p = float(np.interp(th, [yp[i - 1], yp[i]], [ca[i - 1], ca[i]])) if i > 0 else 0.0
        else:
            a_p = float(ca[-1])
        d["lag"] = a_p - a_h
    else:
        d["lag"] = np.nan
    return d


def _feat_one(job):
    i, h, D, E = job
    tr = Track(h)
    out = []
    for q in D.itertuples():
        if q.k < 2:
            continue
        e = E.get((q.tag, q.t0), np.nan)
        out.append(dict(tag=q.tag, id=i, k=int(q.k), t0=int(q.t0), v=q.v, vh=q.vh, **decision(q, tr, e)))
    return out


def ext(tag):
    return EXT.get(tag, EXT_DEF)


def exit_time(st):
    s = st[st.eval_relevant > 0].sort_values("t_us")
    ev = s[(s.offroad > 0) | (s.left_corridor_laterally > 0) | (s.collision_any > 0) | (s.dist_to_gt_trajectory >= 4)]
    return float(ev.t_us.iloc[0]) if len(ev) else np.nan


def cmd_feat(a):
    from multiprocessing import Pool
    H = pickle.load(open(TMP / "lat_zero/hum.pkl", "rb"))
    Ds, Ss, Ws = [], [], []
    for t in TAGS:
        d = pd.read_parquet(ext(t) / "dec.parquet")
        Ds.append(d[d.tag == t])
        s = pd.read_parquet(ext(t) / "steps.parquet")
        Ss.append(s[s.tag == t])
        w = pd.read_parquet(ext(t) / "win.parquet")
        Ws.append(w[w.tag == t])
    D, St, W = pd.concat(Ds), pd.concat(Ss), pd.concat(Ws)
    E = St.set_index(["id", "tag", "t_us"]).min_distance_to_lane_boundary_m
    Eg = {i: g.droplevel(0).to_dict() for i, g in E.groupby(level=0)}
    ids = [i for i in D.id.unique() if i in H]
    jobs = [(i, H[i], g, Eg.get(i, {})) for i, g in D[D.id.isin(ids)].groupby("id")]
    with Pool(a.workers) as p:
        F = pd.DataFrame([r for x in p.imap_unordered(_feat_one, jobs, chunksize=8) for r in x])
    tev = St.groupby(["tag", "id"]).apply(exit_time).rename("t_ev").reset_index()
    F = F.merge(tev, on=["tag", "id"], how="left").merge(W[["tag", "id", "score", "type"]].drop_duplicates(["tag", "id"]), on=["tag", "id"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    F.to_parquet(OUT / "feat.parquet")
    print(len(F), F.tag.value_counts().to_dict())


# ---------------------------------------------------------------- Mac: separation
def auc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logit_fit(X, y, lam=1.0, w=None):
    """L2 logistic regression by Newton steps (intercept unpenalised)."""
    X1 = np.c_[np.ones(len(X)), X]
    b = np.zeros(X1.shape[1])
    w = np.ones(len(y)) if w is None else w
    R = lam * np.eye(X1.shape[1])
    R[0, 0] = 0
    for _ in range(30):
        p = 1 / (1 + np.exp(-X1 @ b))
        g = X1.T @ (w * (p - y)) + R @ b
        Hs = (X1 * (w * p * (1 - p))[:, None]).T @ X1 + R
        db = np.linalg.solve(Hs, g)
        b -= db
        if np.abs(db).max() < 1e-8:
            break
    return b


def logit_pred(b, X):
    return 1 / (1 + np.exp(-np.c_[np.ones(len(X)), X] @ b))


def frame(F, tag):
    """the read's decisions with their roles: dom (turn domain), role = fail-pre (primary span) / fail-pre2 (secondary) / pass."""
    sh = pd.read_csv(TMP / "lat_zero/sharp_windows.csv")
    sh = sh[sh.tag == tag].drop_duplicates("id").set_index("id")
    G = F[F.tag == tag].copy()
    G["dom"] = (G.v >= V_MIN) & (G.turn_deg >= TURN_MIN)
    G["sharp"] = G.id.isin(sh.index)
    G["wfail"] = G.id.map(sh.latzero).fillna(False).astype(bool) & G.sharp
    G["wpass"] = G.sharp & (G.score >= 1 - 1e-4)
    lo, hi = G.t_ev - SPAN[0] * 1e6, G.t_ev - SPAN[1] * 1e6
    G["pre"] = G.wfail & G.dom & (G.t0 >= lo) & (G.t0 <= hi)
    G["pre2"] = G.wfail & G.dom & (G.t0 >= lo) & (G.t0 < G.t_ev)
    G["pdec"] = G.wpass & G.dom
    G["short"] = G.id.str[-8:]
    return G


def combo_scores(G, Gfit=None):
    """leave-one-window-out logistic on the map-free features over the read's eligible decisions; Gfit given: fit on Gfit, apply to G."""
    cols = list(MAPFREE)
    E = G[G.pre | G.pdec]
    ref = (Gfit[Gfit.pre | Gfit.pdec] if Gfit is not None else E)
    med, sd = ref[cols].median(), ref[cols].std().replace(0, 1)
    Z = lambda X: ((X[cols].fillna(med) - med) / sd).clip(-6, 6).to_numpy()  # noqa: E731
    y = ref.pre.to_numpy(float)
    wt = np.where(y > 0, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
    Zr = Z(ref)
    b_all = logit_fit(Zr, y, w=wt)
    s = pd.Series(logit_pred(b_all, Z(G)), index=G.index)
    if Gfit is None:                                               # leave-one-window-out for the eligible windows' own decisions
        idr = ref.id.to_numpy()
        for w in np.unique(idr):
            m = idr != w
            b = logit_fit(Zr[m], y[m], w=wt[m])
            ix = G.index[G.id == w]
            s.loc[ix] = logit_pred(b, Z(G.loc[ix]))
    return s, dict(zip(["intercept", *cols], b_all.round(3).tolist()))


def evaluate(G, f, thr=None):
    """window / decision AUC, 1 % fire threshold on B2's held passing decisions, coverage."""
    x = G[f].where(G.dom)
    win = lambda m: x[m].groupby(G.id[m]).max()  # noqa: E731
    fw, pw, fw2 = win(G.pre), win(G.pdec), win(G.pre2)
    r = dict(n_fail=int(fw.notna().sum()), n_pass=int(pw.notna().sum()), auc_win=auc(fw, pw), auc_win_span2=auc(fw2, pw),
             auc_dec=auc(x[G.pre], x[G.pdec]))
    held1 = (G.score >= 1 - 1e-4)
    xs = x[held1].fillna(-np.inf).to_numpy()
    own = float(np.quantile(xs, 1 - FIRE_LINE, method="higher")) if len(xs) else np.nan
    t = own if thr is None else thr
    r.update(thr_own=own, thr=t, fire_pass=float((xs > t).mean()), fire_pass_win=float(pd.Series(xs > t).groupby(G.id[held1].to_numpy()).any().mean()))
    fired = x[G.pre] > t
    fwin = fired.groupby(G.id[G.pre]).any()
    r.update(fail_cov=float(fwin.mean()) if len(fwin) else np.nan, fail_dec_cov=float(fired.mean()) if len(fired) else np.nan,
             tgt_cov=int(fwin[fwin.index.str[-8:].isin(TARGETS)].sum()), tgt_n=int(fwin.index.str[-8:].isin(TARGETS).sum()))
    return r


def cmd_auc(a):
    F = pd.read_parquet(OUT / "feat.parquet")
    Gs = {t: frame(F, t) for t in TAGS}
    s1, s0 = Gs[B2["s1"]], Gs[B2["s0"]]
    for t, G in Gs.items():
        G["combo"], _ = combo_scores(G)
    s0["combo_x"], coef = combo_scores(s0, s1)                     # fitted on s1, applied to s0
    s1["combo_x"] = s1["combo"]
    _, coef1 = combo_scores(s1)
    rows, J = [], dict(rule=dict(v_min=V_MIN, turn_min=TURN_MIN, span=SPAN, auc_line=AUC_LINE, fire_line=FIRE_LINE, tgt_line=TGT_LINE,
                                 fail_line=FAIL_LINE), combo_coef_s1=coef1)
    for f in (*FEATS, "combo", "combo_x"):
        e1 = evaluate(s1, f)
        e0 = evaluate(s0, f, thr=e1["thr"])
        e0own = evaluate(s0, f)
        ok = (e1["auc_win"] >= AUC_LINE and e0["auc_win"] >= AUC_LINE and e1["tgt_cov"] >= TGT_LINE and e1["fail_cov"] >= FAIL_LINE
              and e0["fail_cov"] >= FAIL_LINE and e0["fire_pass"] <= 2 * FIRE_LINE)
        rows.append(dict(feature=f, kind="oracle" if f in ORACLE else "geometry" if f in GEOM else "map-free",
                         **{f"s1_{k}": v for k, v in e1.items()}, **{f"s0_{k}": v for k, v in e0.items()},
                         s0_auc_own_thr=e0own["fail_cov"], passes=bool(ok)))
        for t in TAGS[2:]:
            rows[-1][f"{t}_auc_win"] = evaluate(Gs[t], f)["auc_win"] if f != "combo_x" else np.nan
    A = pd.DataFrame(rows)
    (RES / "turn_sep").mkdir(parents=True, exist_ok=True)
    A.to_csv(RES / "turn_sep/auc.csv", index=False, float_format="%.4f")
    # target windows: per-window pre-exit medians of the main candidates (s1)
    tw = s1[s1.pre].groupby("short")[["ratio", "arc", "out4", "out8", "kdef", "lag", "lat0", "alat", "edge"]].median().round(3)
    pw = s1[s1.pdec][["ratio", "arc", "out4", "out8", "kdef", "lag", "lat0", "alat", "edge"]].quantile([0.5, 0.9, 0.99]).round(3)
    J["targets_s1"] = tw.to_dict(orient="index")
    J["pass_decisions_s1_q50_90_99"] = pw.to_dict()
    J["auc"] = A.to_dict(orient="records")
    J["counts"] = {t: dict(fail_windows=int(G[G.pre].id.nunique()), pass_windows=int(G[G.pdec].id.nunique()), fail_dec=int(G.pre.sum()),
                           pass_dec=int(G.pdec.sum()), held_pass_dec=int((G.score >= 1 - 1e-4).sum())) for t, G in Gs.items()}
    json.dump(J, open(RES / "turn_sep.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    c = ["feature", "kind", "s1_auc_win", "s0_auc_win", "s1_auc_win_span2", "s1_auc_dec", "s0_auc_dec", "s1_thr", "s1_fire_pass", "s1_fail_cov",
         "s1_tgt_cov", "s0_fail_cov", "s0_fire_pass", "JTN-L18-s1_auc_win", "JTN-L18-s0_auc_win", "RSG-L18-s0_auc_win", "passes"]
    print(A[c].to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(J["counts"], indent=1))
    print(tw.to_string())
    print(pw.to_string())


# ---------------------------------------------------------------- Mac: garage vs B2 executed turns
def turn_exec(tr, g, t_cut):
    """one executed trace on one window -> entry / apex speed, turn-in lag, outward offsets (up to t_cut s)."""
    g = g[g.t <= t_cut + 1e-6]
    if len(g) < 5:
        return None
    s, off = tr.proj(g[["x", "y"]].to_numpy())
    t0, tH = tr.T[0] / 1e6, (tr.T[-1]) / 1e6
    s_a = float(np.interp(1.0, tr.T / 1e6 - t0, tr.S))                # the human's station at t = 1 s (policy takes over at ~1 s)
    s_b = float(np.interp(5.0, tr.T / 1e6 - t0, tr.S))
    if s_b - s_a < 3:
        return None
    m = (tr.gk >= s_a) & (tr.gk <= s_b)
    s_ap = float(tr.gk[m][np.argmax(np.abs(tr.kap[m]))])
    dh = tr.head(s_b) - tr.head(s_a)
    sig = 1.0 if dh >= 0 else -1.0
    th = 0.5 * abs(dh)
    gg = np.arange(s_a, s_b, 0.25)
    s_h = float(gg[np.argmax(np.abs(tr.head(gg) - tr.head(s_a)) >= th)])
    yaw0 = np.interp(s_a, s, g.yaw.to_numpy()) if s[0] <= s_a else g.yaw.iloc[0] + (tr.head(s_a) - tr.head(s[0]))
    j = np.flatnonzero((sig * (g.yaw.to_numpy() - yaw0) >= th) & (s >= s_a))
    s_e = float(s[j[0]]) if len(j) else np.nan
    vh = lambda st: float(np.interp(np.interp(st, tr.S, tr.T), tr.T, np.gradient(tr.S, tr.T / 1e6)))  # noqa: E731
    after = s >= s_a
    v_ap = float(np.interp(s_ap, s, g.v.to_numpy())) if s.max() >= s_ap else np.nan
    v_in = float(g.v.to_numpy()[np.argmin(np.abs(g.t.to_numpy() - 1.0))])
    ow = -sig * off
    return dict(turn_deg=float(np.degrees(abs(dh))), right=sig < 0, v_in=v_in, r_in=v_in / max(vh(np.interp(1.0, g.t, s)), 0.3),
                v_apex=v_ap, r_apex=v_ap / max(vh(s_ap), 0.3), turnin_lag=s_e - s_h,
                out_apex=float(np.interp(s_ap, s, ow)) if s.max() >= s_ap else np.nan, out_max=float(ow[after].max()) if after.any() else np.nan,
                v_max=float(g.v[after].max()) if after.any() else np.nan, dist=float(s[-1] - s[0]))


def cmd_garage(a):
    H = pickle.load(open(TMP / "lat_zero/hum.pkl", "rb"))
    T = pd.read_parquet(OUT / "traces.parquet")
    F = pd.read_parquet(OUT / "feat.parquet")
    sh = pd.read_csv(TMP / "lat_zero/sharp_windows.csv")
    tev = F.groupby(["tag", "id"]).t_ev.first()
    rows = []
    for i in sh.id.unique():
        if i not in H:
            continue
        tr = Track(H[i])
        lab = sh[sh.id == i].set_index("tag")
        for sd, tag in B2.items():
            te = tev.get((tag, i), np.nan)
            cut = te / 1e6 if np.isfinite(te) else 5.1
            for who in ("garage", tag):
                g = T[(T.tag == who) & (T.id == i)].sort_values("t")
                r = turn_exec(tr, g, cut) if len(g) else None
                if r:
                    rows.append(dict(id=i, short=i[-8:], seed=sd, who="garage" if who == "garage" else "B2", cut=cut,
                                     b2_latzero=bool(lab.latzero.get(tag, False)), b2_score=float(lab.score.get(tag, np.nan)), **r))
    R = pd.DataFrame(rows)
    R.to_csv(RES / "turn_sep/garage.csv", index=False, float_format="%.3f")
    P = R.pivot_table(index=["seed", "short", "b2_latzero"], columns="who",
                      values=["r_in", "r_apex", "turnin_lag", "out_apex", "out_max", "v_max"]).round(2)
    a2 = P[P.index.get_level_values("short").isin(A2_S0)]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(a2.to_string())
    for sd in B2:
        for lz in (True, False):
            q = R[(R.seed == sd) & (R.b2_latzero == lz) & ((R.b2_score >= 1 - 1e-4) | lz)]
            print(sd, "B2 fails" if lz else "B2 passes", q.groupby("who")[["r_in", "r_apex", "turnin_lag", "out_apex", "out_max"]].median().round(2).to_dict(orient="index"))

# ---------------------------------------------------------------- Mac: figure
def cmd_fig(a):
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "research"))
    import plot_style as PS
    PS.apply()
    C = PS.PALETTE
    A = pd.read_csv(RES / "turn_sep/auc.csv")
    F = pd.read_parquet(OUT / "feat.parquet")
    H = pickle.load(open(TMP / "lat_zero/hum.pkl", "rb"))
    Tr = pd.read_parquet(OUT / "traces.parquet")
    R = pd.read_csv(RES / "turn_sep/garage.csv")
    fig, ax = plt.subplots(1, 4, figsize=(PS.DOUBLE_COLUMN_IN * 1.6, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1.1, 0.9, 1.1]))
    # (a) window AUC per candidate, B2 s1 / s0 and the three other reads
    A = A[~A.feature.isin(["combo_x", "right"])].reset_index(drop=True)
    y = np.arange(len(A))[::-1]
    for c, col, m, nm in (("JTN-L18-s1_auc_win", "#BBBBBB", "o", "JTN s1"), ("JTN-L18-s0_auc_win", "#BBBBBB", "s", "JTN s0"),
                          ("RSG-L18-s0_auc_win", "#BBBBBB", "^", "RSG s0"), ("s0_auc_win", C["sky_blue"], "D", "B2 s0"), ("s1_auc_win", C["vermillion"], "o", "B2 s1")):
        ax[0].scatter(A[c], y, s=16, color=col, marker=m, label=nm, zorder=3)
    ax[0].axvline(AUC_LINE, color="k", lw=0.8, ls="--")
    ax[0].axvline(0.5, color="#999999", lw=0.6)
    ax[0].set_yticks(y, [f"{f} ({k[:3]})" for f, k in zip(A.feature, A.kind)], fontsize=7)
    ax[0].set_xlabel("window AUC, failing vs passing sharp turns")
    ax[0].legend(fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.55), framealpha=0.95)
    PS.panel(ax[0], "(a) candidate states (line 0.85)")
    # (b) window max speed ratio vs the path's max curvature ahead, B2 s1 + s0
    for sd, mk in (("s1", "o"), ("s0", "D")):
        G = frame(F, B2[sd])
        x = G[["kmax1m", "ratio"]].where(G.dom, np.nan)
        for role, col, nm in (("pdec", "#AAAAAA", "passes"), ("pre", C["vermillion"], "lateral zero")):
            w = x[G[role]].groupby(G.id[G[role]]).max()
            ax[1].scatter(w.kmax1m, w.ratio, s=12 if role == "pdec" else 20, color=col, marker=mk, alpha=0.8 if role == "pre" else 0.6,
                          label=f"B2 {sd} {nm}", zorder=3 if role == "pre" else 2)
    ax[1].axvline(0.108, color="k", lw=0.8, ls="--")
    ax[1].axhline(1.05, color="#999999", lw=0.6, ls=":")
    ax[1].set_xlabel("max curvature of the human path, next 20 m (1/m)")
    ax[1].set_ylabel("ego / human speed (window max)")
    ax[1].legend(fontsize=6, loc="upper right")
    PS.panel(ax[1], "(b) tight turns fail, at any speed ratio")
    # (c) BEV of one target: human, B2 s1, garage
    i = [k for k in H if k.endswith("3f805a18")][0]
    tr = Track(H[i])
    ax[2].plot(tr.P[:, 0], tr.P[:, 1], color="#222222", lw=3.2, alpha=0.3, label="human (log)")
    for who, col, nm in ((B2["s1"], C["vermillion"], "B2 s1"), ("garage", C["blue"], "garage 0014")):
        g = Tr[(Tr.tag == who) & (Tr.id == i)].sort_values("t")
        ax[2].plot(g.x, g.y, color=col, lw=1.4, label=nm)
    te = F[(F.tag == B2["s1"]) & (F.id == i)].t_ev.iloc[0] / 1e6
    g = Tr[(Tr.tag == B2["s1"]) & (Tr.id == i)]
    j = (g.t - te).abs().idxmin()
    ax[2].plot(g.x[j], g.y[j], "x", color="k", ms=7, mew=1.5, label="B2 exit (scored)")
    ax[2].plot(tr.P[0, 0], tr.P[0, 1], "o", color="k", ms=4, label="start")
    ax[2].set_aspect("equal", adjustable="datalim")
    ax[2].set_xticks([]), ax[2].set_yticks([])
    ax[2].legend(fontsize=6, loc="best")
    PS.panel(ax[2], "(c) 3f805a18, 99 deg right turn")
    # (d) executed turn: turn-in lag and outward offset, medians with IQR
    grp = [("B2, B2 fails", (R.who == "B2") & R.b2_latzero), ("garage, B2 fails", (R.who == "garage") & R.b2_latzero),
           ("B2, B2 passes", (R.who == "B2") & ~R.b2_latzero & (R.b2_score >= 1 - 1e-4)),
           ("garage, B2 passes", (R.who == "garage") & ~R.b2_latzero & (R.b2_score >= 1 - 1e-4))]
    cols = [C["vermillion"], C["blue"], "#E8A27A", C["sky_blue"]]
    for k, (met, nm) in enumerate((("turnin_lag", "turn-in lag (m)"), ("out_apex", "outward at apex (m)"), ("out_max", "max outward (m)"))):
        for j, ((lab, m), col) in enumerate(zip(grp, cols)):
            q = R.loc[m, met].quantile([0.25, 0.5, 0.75])
            xx = k + (j - 1.5) * 0.2
            ax[3].bar(xx, q[0.5], 0.18, color=col, label=lab if k == 0 else None)
            ax[3].errorbar(xx, q[0.5], yerr=[[q[0.5] - q[0.25]], [q[0.75] - q[0.5]]], color="k", lw=0.7, capsize=1.5)
        ax[3].text(k, ax[3].get_ylim()[0], "", ha="center")
    ax[3].set_xticks(range(3), ["turn-in lag\n(m, + = late)", "outward at\napex (m)", "max outward\n(m)"], fontsize=7)
    PS.zero_line(ax[3])
    PS.bars(ax[3])
    ax[3].set_ylim(-3.6, 3.6)
    ax[3].legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.95)
    PS.panel(ax[3], "(d) executed sharp turns, B2 s1 + s0\nwindows vs garage (median, IQR)")
    fig.tight_layout(w_pad=0.8)
    print(PS.save(fig, RES / "figs/turn_sep"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("traces")
    p.add_argument("--out", required=True)
    p = sp.add_parser("pool")
    p.add_argument("--pool", default="on1_JTX"), p.add_argument("--k", type=float, default=0.107), p.add_argument("--m", type=float, default=0.2)
    p.add_argument("--out", required=True)
    p = sp.add_parser("feat")
    p.add_argument("--workers", type=int, default=8)
    sp.add_parser("auc")
    sp.add_parser("garage")
    sp.add_parser("fig")
    a = ap.parse_args()
    dict(traces=cmd_traces, pool=cmd_pool, feat=cmd_feat, auc=cmd_auc, garage=cmd_garage, fig=cmd_fig)[a.cmd](a)
