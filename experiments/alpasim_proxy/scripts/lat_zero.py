#!/usr/bin/env python3
"""LAT-ZERO: B2's remaining easy lateral zeros (decision 393) split into overspeed at curvature / path error at normal speed / other.
Diagnostic, CPU only, stored reads; rule pre-registered in plans/2026-10-20-lat-zero-prereg.md.

  tmp/pcs_local/.venv/bin/python lat_zero.py table   (Mac) per-window table, class counts, frozen-pool PCS of the levers
        -> results/lat_zero.json, results/lat_zero/windows.csv
  tmp/pcs_local/.venv/bin/python lat_zero.py fig     (Mac) BEV + speed panels of representative windows -> results/figs/lat_zero.{png,pdf}

Inputs: tmp/easy_items_B2/ext_full_H{,_s1} (native_gap extracts of the B2 reads, JTN-L18-s1 on the same windows), tmp/lat_zero/hum.pkl
(the human tracks of stand_b/ext_full_H_s1), tmp/easy_items/steps_extra.parquet (garage per-step metrics).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/scripts")]
RES = REPO / "experiments/alpasim_proxy/results"
TMP = REPO / "tmp"
READS = {"s1": ("STANDFMXH-L18-s1", TMP / "easy_items_B2/ext_full_H_s1", "misses_STANDFMXH-s1",
                REPO / "experiments/defect_fix/results/stand_b/arena_windows_full_H_s1.csv.gz"),
         "s0": ("STANDFMXH-L18-s0", TMP / "easy_items_B2/ext_full_H", "misses_STANDFMXH-s0",
                REPO / "experiments/defect_fix/results/stand_b/arena_windows_full_H.csv.gz")}
REF = "JTN-L18-s1"
LAT = ["zero: turn window, offroad / corridor", "zero: corridor, moving non-turn", "zero: offroad, non-turn"]
# pre-registered thresholds (plans/2026-10-20-lat-zero-prereg.md)
T_EARLY, R_OVER, SLOW, KAPPA, OFF_MIN, LEAD_MIN = 1.517, 1.05, 0.10, 0.02, 0.5, 8.0


class Track:
    """human rear-axle track: station grid, heading, speed; signed projection (left of travel = +)."""

    def __init__(self, h):
        T, P, Y = np.asarray(h["T"], float), np.asarray(h["P"], float), np.unwrap(np.asarray(h["Y"], float))
        seg = np.hypot(*np.diff(P, axis=0).T)
        self.T, self.P, self.Y = T, P, Y
        self.S = np.r_[0.0, np.cumsum(seg)]
        v = seg / np.maximum(np.diff(T) / 1e6, 1e-3)
        self.V = np.r_[v[0], 0.5 * (v[1:] + v[:-1]), v[-1]]
        keep = np.r_[True, seg > 0.02]                     # heading on a station grid without the standing jitter
        self.Sk, self.Yk = self.S[keep], Y[keep]

    def proj(self, xy):
        q = np.atleast_2d(np.asarray(xy, float))[:, None]
        a, b = self.P[None, :-1], self.P[None, 1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(-1), 1e-9)
        u = np.clip(((q - a) * ab).sum(-1) / L2, 0, 1)
        c = a + u[..., None] * ab
        d = np.hypot(*(c - q).transpose(2, 0, 1))
        i = d.argmin(1); r = np.arange(len(i))
        cr = ab[0, i, 0] * (q[r, 0, 1] - a[0, i, 1]) - ab[0, i, 1] * (q[r, 0, 0] - a[0, i, 0])
        return self.S[i] + u[r, i] * np.sqrt(L2[0, i]), np.sign(cr) * d[r, i]

    def head(self, s):
        return np.interp(s, self.Sk, self.Yk)

    def kappa(self, s0, s1):
        """mean |d heading / d s| over [s0, s1] on a 0.5 m grid (rad/m)."""
        if s1 - s0 < 1.0:
            s1 = s0 + 1.0
        g = np.arange(s0, s1 + 1e-6, 0.5)
        return float(np.abs(np.diff(self.head(g))).sum() / (g[-1] - g[0]))

    def vmin(self, s0, s1):
        g = np.linspace(s0, s1, 41)
        i = np.clip(np.searchsorted(self.S, g), 0, len(self.S) - 1)
        return float(self.V[i].min()), float(np.interp(s0, self.S, self.V))


def plan_world(r):
    P = np.asarray(r.poses, float).reshape(8, 3)
    c, s = np.cos(r.yaw), np.sin(r.yaw)
    return np.c_[r.x + c * P[:, 0] - s * P[:, 1], r.y + s * P[:, 0] + c * P[:, 1]], P


def exit_time(st):
    s = st[st.eval_relevant > 0].sort_values("t_us")
    ev = s[(s.offroad > 0) | (s.left_corridor_laterally > 0) | (s.collision_any > 0) | (s.dist_to_gt_trajectory >= 4)]
    return (float(ev.t_us.iloc[0]), ev.iloc[0]) if len(ev) else (np.nan, None)


def window(i, tag, D, St, tr):
    """the pre-registered quantities of one read on one window."""
    g = D[(D.tag == tag) & (D.id == i)].sort_values("k")
    st = St[(St.tag == tag) & (St.id == i)]
    t_ev, e = exit_time(st)
    out = dict(t_ev=t_ev / 1e6 if np.isfinite(t_ev) else np.nan)
    if e is None:
        return out | dict(lat_max=float(st[st.eval_relevant > 0].lateral_dist_to_gt_trajectory.max()))
    out |= dict(ev_kind="offroad" if e.offroad > 0 else "corridor" if e.left_corridor_laterally > 0 else
                "collision" if e.collision_any > 0 else "far", exit_off=float(e.lateral_dist_to_gt_trajectory))
    pre = g[(g.k >= 2) & (g.t0 < t_ev) & (g.t0 >= t_ev - 3e6)]
    out["n_pre"], out["k_ev"] = len(pre), int(round((t_ev - 17000) / 5e5))
    if not len(pre):
        return out
    r = pre.v / np.maximum(pre.vh, 0.3)
    s_a, s_b = float(pre.s_e.iloc[0]), float(pre.s_e.iloc[-1])
    vmin, va = tr.vmin(s_a, s_b + 5)
    kap = tr.kappa(s_a, s_b + 5)
    # signed side of the exit: the ego's signed offset at the decision nearest the exit
    j = (g.t0 - t_ev).abs().idxmin()
    _, side = tr.proj([g.loc[j, "x"], g.loc[j, "y"]])
    side = float(np.sign(side[0]))
    turn = float(np.sign(tr.head(s_b + 5) - tr.head(s_a)))
    # plan geometry at equal arc (first 2 s of the plan), per pre-exit decision with k >= 2
    kr, dep, al = [], [], []
    for _, q in pre.iterrows():
        W, P = plan_world(q)
        arc = np.r_[0, np.cumsum(np.hypot(*np.diff(np.r_[[[0, 0]], P[:4, :2]], axis=0).T))][-1]
        if arc < 1.0:
            continue
        _, o0 = tr.proj([q.x, q.y])
        _, o4 = tr.proj(W[3])
        dep.append(float((o4[0] - o0[0]) * side))
        dh = tr.head(q.s_e + arc) - tr.head(q.s_e)
        dp = P[3, 2]
        if abs(dh) >= np.radians(3):
            kr.append(dp / dh)
            al.append((q.v ** 2 * abs(dp)) / max(q.vh ** 2 * abs(dh), 1e-6))
    lg = pre.lead_gap.min()
    return out | dict(r3=float(r.median()), r3max=float(r.max()), v_pre=float(pre.v.median()), vh_pre=float(pre.vh.median()),
                      slow=float(1 - vmin / max(va, 0.3)), kappa=kap, turn_deg=float(np.degrees(abs(tr.head(s_b + 5) - tr.head(s_a)))),
                      side=side, turn=turn, outward=bool(turn != 0 and side == -turn),
                      k_ratio=float(np.median(kr)) if kr else np.nan, alat_ratio=float(np.median(al)) if al else np.nan,
                      dep2=float(np.median(dep)) if dep else np.nan, lead_min=float(lg) if np.isfinite(lg) else np.nan,
                      cmd=",".join(str(int(c)) for c in pre.cmd), lat_max=float(st[st.eval_relevant > 0].lateral_dist_to_gt_trajectory.max()))


def _geom_one(job):
    """per-decision plan geometry against the human track for every read on one window (valid decisions before the read's exit)."""
    i, h, Dg, Sg = job
    tr = Track(h)
    out = []
    for tag, g in Dg.groupby("tag"):
        t_ev, _ = exit_time(Sg[Sg.tag == tag])
        st = Sg[(Sg.tag == tag) & (Sg.eval_relevant > 0)].sort_values("t_us")
        for _, q in g[(g.k >= 2) & ~(g.t0 >= t_ev)].iterrows():
            W, P = plan_world(q)
            arc = float(np.hypot(*np.diff(np.r_[[[0, 0]], P[:4, :2]], axis=0).T).sum())
            if arc < 2.0 or q.vh < 1.0:
                continue
            dh = float(tr.head(q.s_e + arc) - tr.head(q.s_e))
            # plan points at equal arc (4 / 8 m along the plan polyline from the anchor)
            pl = np.r_[[[q.x, q.y]], W]
            ca = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
            ea = [np.c_[np.interp(L, ca, pl[:, 0]), np.interp(L, ca, pl[:, 1])][0] if ca[-1] >= L else [np.nan, np.nan] for L in (4.0, 8.0)]
            _, o = tr.proj(np.r_[[[q.x, q.y]], W[[1, 3]]])
            oe = [float(tr.proj(e)[1][0]) if np.isfinite(e[0]) else np.nan for e in ea]
            # executed position 1 s later against the plan's 1 s point (lateral, in the plan's frame)
            nx = g[g.k == q.k + 2]
            ex = np.nan
            if len(nx):
                d = np.array([nx.x.iloc[0], nx.y.iloc[0]]) - W[1]
                hy = q.yaw + P[1, 2]
                ex = float(-np.sin(hy) * d[0] + np.cos(hy) * d[1])
            out.append(dict(tag=tag, id=i, k=int(q.k), v=q.v, vh=q.vh, arc=arc, dh=dh, dp=float(P[3, 2]),
                            off0=float(o[0]), off1=float(o[1]), off2=float(o[2]), off4m=oe[0], off8m=oe[1], exec1=ex, lead=q.lead_gap))
    return out


def cmd_geom(a):
    """per-decision geometry table over all held non-static windows -> tmp/lat_zero/geom.parquet."""
    from multiprocessing import Pool
    H, _, data = load_all()
    D = pd.concat([data["s1"][1], data["s0"][1][data["s0"][1].tag == "STANDFMXH-L18-s0"]]).drop_duplicates(["tag", "id", "k"])
    St = pd.concat([data["s1"][2], data["s0"][2][data["s0"][2].tag == "STANDFMXH-L18-s0"]]).drop_duplicates(["tag", "id", "t_us"])
    S = pd.read_csv(TMP / "native_gap/set.csv.gz").drop_duplicates("id").set_index("id")
    ids = [i for i in D.id.unique() if i in H and S.stratum.get(i) != "static"]
    Dg, Sg = dict(tuple(D[D.id.isin(ids)].groupby("id"))), dict(tuple(St[St.id.isin(ids)].groupby("id")))
    with Pool(a.workers) as p:
        rows = [r for x in p.imap_unordered(_geom_one, [(i, H[i], Dg[i], Sg[i]) for i in ids], chunksize=8) for r in x]
    G = pd.DataFrame(rows)
    G.to_parquet(TMP / "lat_zero/geom.parquet")
    print(len(G), G.tag.value_counts().to_dict())


def classify(w, typ):
    if not np.isfinite(w.get("t_ev", np.nan)) or w.get("n_pre", 0) == 0 or w["t_ev"] <= T_EARLY or w.get("k_ev", 0) <= 3:
        return "c: early / no policy decision"
    if typ not in ("offroad", "corridor", "offroad+corridor"):
        return "c: not lateral (scorer type)"
    if w["exit_off"] < OFF_MIN:
        return "c: human track marginal (on track at exit)"
    if np.isfinite(w["lead_min"]) and w["lead_min"] < LEAD_MIN:
        return "c: lead interaction"
    if w["r3"] >= R_OVER and (w["slow"] >= SLOW or w["kappa"] >= KAPPA):
        return "a: overspeed at curvature / slow-down"
    return "b: path error at normal speed"


def bsub(w):
    if w["kappa"] >= KAPPA and np.isfinite(w["k_ratio"]):
        if w["k_ratio"] < 0.8:
            return "under-turn"
        if w["k_ratio"] > 1.2:
            return "over-turn"
    if np.isfinite(w["dep2"]) and w["dep2"] >= 0.3:
        return "drift"
    return "execution / unclear"


def dist_ratio(E, tag, i, tr, t_ev):
    """distance driven / the human's arc from the first scored step to t_ev (s), and the max lateral offset over all scored steps."""
    g = E[(E.tag == tag) & (E.id == i)].sort_values("t_us")
    if not len(g) or not np.isfinite(t_ev):
        return np.nan, np.nan
    s = g[g.eval_relevant > 0]
    hs = np.interp(t_ev * 1e6, tr.T, tr.S) - np.interp(s.t_us.iloc[0], tr.T, tr.S)
    gs = np.interp(t_ev * 1e6, g.t_us, g.dist_traveled_m) - s.dist_traveled_m.iloc[0]
    return float(gs / max(hs, 0.5)), float(s.lateral_dist_to_gt_trajectory.max())


def load_all():
    H = pickle.load(open(TMP / "lat_zero/hum.pkl", "rb"))
    E = pd.read_parquet(TMP / "easy_items/steps_extra.parquet")
    data = {}
    for sd, (tag, ext, mis, rows) in READS.items():
        D = pd.read_parquet(ext / "dec.parquet")
        St = pd.read_parquet(ext / "steps.parquet")
        M = pd.read_csv(RES / f"easy_items/{mis}.csv.gz")
        data[sd] = (tag, D, St, M, pd.read_csv(rows))
    return H, E, data


def cmd_table(a):
    H, E, data = load_all()
    rows = []
    for sd, (tag, D, St, M, R) in data.items():
        tgt = M[M.cause.isin(LAT) & (M.strat == "a: easy (garage = 1)")]
        oth = "s0" if sd == "s1" else "s1"
        Mo = data[oth][3].set_index("id")
        for _, m in tgt.iterrows():
            tr = Track(H[m.id])
            w = window(m.id, tag, D, St, tr)
            jw = window(m.id, REF, D, St, tr)
            c = classify(w, m.type)
            # JTN-s1 over the same time span as the subject's pre-exit window (how it drives it)
            gj = D[(D.tag == REF) & (D.id == m.id) & (D.k >= 2)]
            if np.isfinite(w.get("t_ev", np.nan)):
                gj = gj[(gj.t0 < w["t_ev"] * 1e6) & (gj.t0 >= w["t_ev"] * 1e6 - 3e6)]
            gr, glat = dist_ratio(E, "garage", m.id, tr, w.get("t_ev", np.nan))
            br, _ = dist_ratio(St, tag, m.id, tr, w.get("t_ev", np.nan))
            jr, _ = dist_ratio(St, REF, m.id, tr, w.get("t_ev", np.nan))
            rows.append(dict(seed=sd, id=m.id, short=m.id[-8:], cause=m.cause, type=m.type, stratum=m.stratum, a=m.a, cls=c,
                             bsub=bsub(w) if c.startswith("b") else "", **{k: v for k, v in w.items()},
                             jtn_score=m.JTN1, jtn_r=float((gj.v / np.maximum(gj.vh, 0.3)).median()) if len(gj) else np.nan,
                             jtn_lat_max=jw.get("lat_max", np.nan), jtn_r3=jw.get("r3", np.nan),
                             other_seed_score=float(Mo.score.get(m.id, 1.0)), dist_r=br, jtn_dist_r=jr, garage_r=gr,
                             garage_lat_max=glat))
    W = pd.DataFrame(rows)
    out = RES / "lat_zero"; out.mkdir(exist_ok=True)
    W.to_csv(out / "windows.csv", index=False, float_format="%.3f")
    J = dict(counts={sd: W[W.seed == sd].groupby(["cause", "cls"]).size().unstack(fill_value=0).to_dict() for sd in READS},
             totals={sd: W[W.seed == sd].cls.value_counts().to_dict() for sd in READS},
             bsub={sd: W[(W.seed == sd) & W.cls.str.startswith("b")].bsub.value_counts().to_dict() for sd in READS})
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["seed", "short", "cause", "cls", "bsub", "k_ev", "ev_kind", "exit_off", "r3", "r3max", "slow", "kappa", "turn_deg", "outward",
            "k_ratio", "alat_ratio", "dep2", "lead_min", "cmd", "jtn_score", "jtn_r", "jtn_lat_max", "other_seed_score", "dist_r", "jtn_dist_r", "garage_r", "garage_lat_max"]
    print(W[cols].to_string(float_format=lambda x: f"{x:.2f}"))
    for sd in READS:
        print(sd, pd.crosstab(W[W.seed == sd].cause, W[W.seed == sd].cls), sep="\n")
    json.dump(J, open(RES / "lat_zero.json", "w"), indent=1, default=float)


def sub_a(w):
    """post-hoc sub-split of class a (descriptive, written after the class counts): stop overrun / sharp turn / other slow-down."""
    if w.slow >= 0.9:
        return "a1: stop overrun (human brakes to a stop, no lead)"
    if w.turn_deg >= 25 and w.kappa >= KAPPA:
        return "a2: speed in a sharp turn"
    return "a3: other slow-down / mild bend"


def cmd_price(a):
    """frozen-pool PCS (board-mix / g15, log-clustered bootstrap) of fixing each class and each lever's target set, per seed."""
    import easy_items as EI
    import pcs_local as PL
    import slow_rescore as SR
    sc, _, _ = PL.load(RES / "pcs_local")
    items = list(pd.read_csv(RES / "pcs_local/items_boardmix.csv.gz", index_col=0).index)
    S = pd.read_csv(TMP / "native_gap/set.csv.gz").drop_duplicates("id").set_index("id")
    PR = {k: EI.Pricer(SR.Frozen(p, sc), S.log.reindex(items).to_numpy()) for k, p in SR.FITS.items()}
    W = pd.read_csv(RES / "lat_zero/windows.csv")
    W["sub"] = [sub_a(w) if w.cls.startswith("a") else w.cls[:1] for w in W.itertuples()]
    W.to_csv(RES / "lat_zero/windows.csv", index=False, float_format="%.3f")
    sharp = pd.read_csv(TMP / "lat_zero/sharp_windows.csv")
    rng = np.random.default_rng(0)
    J = json.load(open(RES / "lat_zero.json"))
    J["price"], J["sub_counts"] = {}, {}
    for sd, (tag, _, mis, rows) in READS.items():
        y = pd.read_csv(rows).drop_duplicates("id").set_index("id").score.reindex(items).to_numpy(float)
        M = pd.read_csv(RES / f"easy_items/{mis}.csv.gz")
        w = W[W.seed == sd]
        red = set(M.id[(M.cause == "zero: static, red line, no lead") & (M.strat == "a: easy (garage = 1)")])
        a2 = set(w.id[w["sub"].str.startswith("a2")])
        groups = {"all 20-window lateral group": set(w.id), "a (all)": set(w.id[w.cls.str.startswith("a")]),
                  **{s: set(w.id[w["sub"] == s]) for s in sorted(w["sub"].unique())},
                  "lever STAND-R: a1 + easy static red-line zeros": set(w.id[w["sub"].str.startswith("a1")]) | red,
                  "lever TURN-CAP: a2": a2}
        P = {}
        for nm, ids in groups.items():
            m = np.isin(items, list(ids))
            y1 = np.where(m, 1.0, y)
            P[nm] = dict(windows=int(m.sum()), **{k: p.diff(y, y1) for k, p in PR.items()})
        # expected TURN-CAP value: each a2 window fixed with p = 1 - (zero rate at r <= 1.05) / (zero rate at its r bin), right turns, all reads
        rate = sharp.assign(rb=pd.cut(sharp.r, [0, 1.05, 1.2, 1.4, 9])).groupby("rb", observed=True).latzero.mean()
        pf = {i: float(np.clip(1 - rate.iloc[0] / rate[pd.cut([r], [0, 1.05, 1.2, 1.4, 9])[0]], 0, 1)) if r > 1.05 else 0.0
              for i, r in zip(w.id[w.id.isin(a2)], w.r3[w.id.isin(a2)])}
        ev = {k: [] for k in PR}
        for _ in range(400):
            fix = [i for i, p in pf.items() if rng.random() < p]
            y1 = np.where(np.isin(items, fix), 1.0, y)
            for k, p in PR.items():
                ev[k].append(p.pcs(y1) - p.pcs(y))
        P["lever TURN-CAP: a2, expected (fix prob. from the zero-rate ratio)"] = dict(
            windows=float(sum(pf.values())), **{k: [float(np.mean(v)), float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))] for k, v in ev.items()})
        P["base"] = {k: p.pcs(y) for k, p in PR.items()}
        J["price"][sd] = P
        J["sub_counts"][sd] = w["sub"].value_counts().to_dict()
        J["turn_zero_rate_by_ratio"] = {str(k): float(v) for k, v in rate.items()}
    json.dump(J, open(RES / "lat_zero.json", "w"), indent=1, default=float)
    for sd, P in J["price"].items():
        print(sd, J["sub_counts"][sd])
        for nm, r in P.items():
            print(f"  {nm}: {r}")


FIG = [("a1: stop overrun", "915451d1"), ("a2: wide right turn", "efe356db"), ("a2: inside of a right turn", "202f583c"),
       ("b: drift at normal speed", "5742534c")]


def cmd_fig(a):
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "research"))
    import plot_style as PS
    PS.apply()
    H = pickle.load(open(TMP / "lat_zero/hum.pkl", "rb"))
    tag, ext = READS["s1"][0], READS["s1"][1]
    D, St = pd.read_parquet(ext / "dec.parquet"), pd.read_parquet(ext / "steps.parquet")
    W = pd.read_csv(RES / "lat_zero/windows.csv")
    W = W[W.seed == "s1"].set_index("short")
    C = dict(h="#222222", b=PS.PALETTE["vermillion"], j=PS.PALETTE["blue"])
    fig, ax = plt.subplots(2, 5, figsize=(PS.DOUBLE_COLUMN_IN * 1.6, 5.0), gridspec_kw=dict(height_ratios=[1.35, 1]))
    for c, (title, sh) in enumerate(FIG):
        w = W.loc[sh]; i = w.id; tr = Track(H[i])
        bx, sx = ax[0, c], ax[1, c]
        bx.plot(tr.P[:, 0], tr.P[:, 1], color=C["h"], lw=3.2, alpha=.3, label="human (log)")
        for t, col, nm in ((tag, C["b"], "B2 s1"), (REF, C["j"], "JTN-s1")):
            g = D[(D.tag == t) & (D.id == i)].sort_values("k")
            bx.plot(g.x, g.y, "-o", color=col, ms=2.2, lw=1, label=nm)
            sx.plot(g.t0 / 1e6, g.v, "-o", color=col, ms=2.2, lw=1, label=f"{nm} speed")
            if t == tag:
                sx.plot(g.t0 / 1e6, g.vh, "--", color=C["h"], lw=1, label="human at ego's station")
                t_ev = w.t_ev
                pre = g[(g.k >= 2) & (g.t0 <= (t_ev - 1.0) * 1e6)]
                if len(pre):
                    q = pre.iloc[-1]; Pw, _ = plan_world(q)
                    bx.plot(np.r_[q.x, Pw[:, 0]], np.r_[q.y, Pw[:, 1]], ":", color=col, lw=1.3, label=f"B2 plan at {q.t0 / 1e6:.1f} s")
                e = g.iloc[(g.t0 - t_ev * 1e6).abs().argmin()]
                bx.plot(e.x, e.y, "x", color="k", ms=7, mew=1.5, label="exit (scored)")
                sx.axvline(t_ev, color="#999999", lw=.6)
        bx.set_aspect("equal", adjustable="datalim"); bx.set_xticks([]); bx.set_yticks([])
        PS.panel(bx, f"({'abcd'[c]}) {title}\n{sh}: r3 {w.r3:.2f}, turn {w.turn_deg:.0f}$^\\circ$,\nJTN-s1 score {w.jtn_score:.0f}")
        sx.set_xlabel("time in window (s)"); sx.set_ylim(bottom=0)
        if c == 0:
            sx.set_ylabel("speed (m/s)"); bx.legend(loc="lower left", fontsize=6.5); sx.legend(loc="upper left", fontsize=6.5)
    # summary column: lateral-zero rate in sharp turns by speed ratio (all 5 reads), equal-arc outward departure by speed ratio
    sh = pd.read_csv(TMP / "lat_zero/sharp_windows.csv")
    sh["rb"] = pd.cut(sh.r, [0, 1.05, 1.2, 1.4, 9])
    lab = ["$\\leq$1.05", "1.05-\n1.2", "1.2-\n1.4", "$>$1.4"]
    for d, col, off in (("right", PS.PALETTE["vermillion"], 0.18), ("left", PS.PALETTE["sky_blue"], -0.18)):
        q = sh[sh.dir == d].groupby("rb", observed=False).latzero.agg(["mean", "size"])
        ax[0, 4].bar(np.arange(4) + off, q["mean"], 0.36, color=col, label=f"{d} turns")
        for x, (m, n) in enumerate(zip(q["mean"], q["size"])):
            ax[0, 4].text(x + off, m + .01, f"{n}", ha="center", fontsize=6)
    ax[0, 4].set_xticks(range(4), lab, fontsize=7); ax[0, 4].set_ylabel("lateral-zero rate"); ax[0, 4].legend(fontsize=6.5)
    PS.bars(ax[0, 4])
    PS.panel(ax[0, 4], "(e) sharp-turn windows, 5 reads:\nlateral zeros by speed ratio (n above)")
    G = pd.read_parquet(TMP / "lat_zero/geom.parquet")
    G = G[G.dh.abs() >= np.radians(15)].assign(r=lambda x: x.v / x.vh)
    G["o8"] = -(G.off8m - G.off0) * np.sign(G.dh)
    G["rb"] = pd.cut(G.r, [0, 1.05, 1.2, 1.4, 9])
    q = G.groupby("rb", observed=False).o8.quantile([.25, .5, .75]).unstack()
    ax[1, 4].errorbar(range(4), q[.5], yerr=[q[.5] - q[.25], q[.75] - q[.5]], fmt="o-", color=PS.PALETTE["vermillion"], ms=3, capsize=2)
    ax[1, 4].set_xticks(range(4), lab, fontsize=7); ax[1, 4].set_xlabel("ego / human speed")
    ax[1, 4].set_ylabel("outward offset (m)"); PS.zero_line(ax[1, 4])
    PS.panel(ax[1, 4], "(f) turning decisions: plan outward\noffset at 8 m arc (median, IQR)")
    fig.tight_layout(h_pad=1.6, w_pad=0.8)
    print(PS.save(fig, RES / "figs/lat_zero"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("table")
    sp.add_parser("fig")
    gp = sp.add_parser("geom")
    gp.add_argument("--workers", type=int, default=8)
    sp.add_parser("price")
    a = ap.parse_args()
    {"table": cmd_table, "geom": cmd_geom, "price": cmd_price, "fig": cmd_fig}[a.cmd](a)
