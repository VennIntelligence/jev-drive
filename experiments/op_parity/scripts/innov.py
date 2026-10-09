"""op_parity innovation gain (plans/2026-10-09-innovation-gain-prereg.md, lane IG1): does SH30 lack scene-caused change?

Measurement only. innovation = (logged 4 s future) - (extrapolation of the last 1.5 s of ego history), in the Frenet frame of the
extrapolated path: lateral = signed offset of the 4 s point from that path, longitudinal = own arc length - extrapolated arc length.
The plan gets the same subtraction; gain = OLS slope of plan innovation on logged innovation.

  report      (.venv, CPU)   gain ladder, gap share by innovation decile, failure shares, false innovation, P3, robustness, navhard
                             stage 1, image-ablation rows when present -> $OUT/report/{*.csv, tables.md, summary.json, figs/}
  ablate      (op-train, GPU, one small pool job)  SH30-F-s0 / s1 plans on navtest with the front tokens of another log:
                             ID (unchanged, gate), KIN (kinematically nearest token of another log), SHUF (random) -> $OUT/ablate_plans.npz
  abl-export  (.venv, CPU)   those plans through the bench export adapter (op_interp `base`) -> $OUT/ablate_poses.npz, ID gate <= 1e-3 m

Extrapolations: cc (constant curvature in arc length + constant acceleration, primary), ccv (curvature, constant speed), cv (straight,
constant speed). kappa = 0 when the 1.5 s history arc is below 1.5 m. Everything else is fixed in the pre-registration.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent), str(REPO / "research")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/op_parity/innov"
RES = OUT / "report"
CR = D / "runs/op_parity/cache"
T_OUT = np.arange(1, 9) * 0.5
MODES = ("cc", "ccv", "cv")
FLOOR = {"lat": 0.5, "lon": 1.0}                      # |x| floor of the sign-agreement rate (m)
MARGIN = 0.10                                          # "clearly below": WA-JEPA gain - SH30 gain
WA = "WA-JEPA"
# family -> members (bench pred stems unless prefixed); gains are reported per member and as the family mean
NAVTEST = {
    "cinque (shipped, G frames)": ["cinque-gimm"], "P2 (no hinge)": ["P2-F-s0-warp"], "RH0 (P2H pilot)": ["RH0-F-s0-warp", "RH0-F-s1-warp"],
    "RMH10 (replay hinge)": ["RMH10-F-s0-warp", "RMH10-F-s1-warp"], "SHP (SH30 pilot)": ["SHP-F-s0-warp", "SHP-F-s1-warp"],
    "GH0 (pilot, no memory)": ["GH0-F-s0-warp", "GH0-F-s1-warp"], "GEX (shuffled geometry)": ["GEX-F-s0-warp", "GEX-F-s1-warp"],
    "GEB (geometry, random init)": ["GEB-F-s0-warp", "GEB-F-s1-warp"], "GEW (geometry, warm start)": ["GEW-F-s0-warp", "GEW-F-s1-warp"],
    "GEP (leaked log path) s0": ["GEP-F-s0-warp"], "GEP (leaked log path) s1": ["GEP-F-s1-warp"],
    "SH30": ["SH30-F-s0-warp", "SH30-F-s1-warp"], "OT30 (off-track rows)": ["OT30-F-s0-warp", "OT30-F-s1-warp"],
    "SH30 (AlpaSim inputs, m 4)": ["ap2:SH30_m4"], "AP2 (AlpaSim inputs, m 4)": ["ap2:AP2_m4"], "AP2 (NAVSIM inputs)": ["ap2:AP2_nav"],
    WA: ["wa:navtest"],
}
NAVHARD = {"P0 (shipped, G frames)": ["P0-gimm"], "RMH10 (replay hinge)": ["RMH10-F-s0-gimm", "RMH10-F-s1-gimm"],
           "SHP (SH30 pilot)": ["SHP-F-s0-gimm", "SHP-F-s1-gimm"], "SH30": ["SH30-F-s0-gimm", "SH30-F-s1-gimm"],
           "OT30 (off-track rows)": ["OT30-F-s0-gimm", "OT30-F-s1-gimm"], WA: ["wa:navhard"]}
SYNTH = ["EXT (extrapolation)", "CONST (navtrain mean future)", "BLIND-E (ego history only)", "BLIND-EC (+ command)", "LOG"]
TURNS = ["<5", "5-20", "20-45", ">45"]


# ---------------------------------------------------------------- geometry
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def extrap(pose, speed, mode="cc"):
    """History (n, 4, 3) at -1.5 .. 0 s in the t0 rear-axle frame, t0 speed -> (kappa, a, s_E (n, 8) at T_OUT, history arc)."""
    seg = np.linalg.norm(np.diff(pose[:, :, :2].astype(np.float64), axis=1), axis=2)
    Lh = seg.sum(1)
    a = np.clip((seg[:, 2] - seg[:, 0]) / 0.5 / 1.0, -4.0, 3.0)
    kap = np.where(Lh >= 1.5, wrap(pose[:, 3, 2].astype(np.float64) - pose[:, 0, 2]) / np.maximum(Lh, 1e-9), 0.0)
    kap = np.clip(kap, -0.2, 0.2)
    if mode == "cv":
        kap = np.zeros_like(kap)
    if mode in ("cv", "ccv"):
        a = np.zeros_like(a)
    v0 = np.maximum(speed.astype(np.float64), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stop = np.where(a < 0, v0 / -a, np.inf)
    tt = np.minimum(T_OUT[None], t_stop[:, None])
    return kap, a, v0[:, None] * tt + 0.5 * a[:, None] * tt ** 2, Lh


def frenet_d(xy, kap):
    """Signed offset (left positive) of points (n, 2) from the path of curvature kap through the origin along +x."""
    x, y = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    st = np.abs(kap) < 1e-6
    R = 1.0 / np.where(st, 1.0, np.abs(kap))
    sg = np.sign(kap)
    return np.where(st, y, sg * (R - np.hypot(x, y - sg * R)))


def arclen(P):
    """Poses (n, 8, 3) -> cumulative polyline length from the origin (n, 9)."""
    xy = np.concatenate([np.zeros((len(P), 1, 2)), P[:, :, :2].astype(np.float64)], 1)
    return np.concatenate([np.zeros((len(P), 1)), np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=2), 1)], 1), xy


def at_arc(P, s):
    """Point (n, 2) of each polyline at arc length s (n,), clipped to its end."""
    cum, xy = arclen(P)
    s = np.minimum(s, cum[:, -1])
    k = np.clip((cum < s[:, None]).sum(1), 1, 8)
    r = np.arange(len(P))
    c0, c1 = cum[r, k - 1], cum[r, k]
    w = np.where(c1 > c0, (s - c0) / np.maximum(c1 - c0, 1e-9), 0.0)[:, None]
    return xy[r, k - 1] * (1 - w) + xy[r, k] * w


def innov(P, kap, sE, h):
    """Trajectory (n, 8, 3) -> (lateral m, longitudinal m, heading deg) innovation at horizon index h."""
    cum, _ = arclen(P)
    return frenet_d(P[:, h, :2], kap), cum[:, h + 1] - sE[:, h], np.degrees(wrap(P[:, h, 2].astype(np.float64) - kap * sE[:, h]))


# ---------------------------------------------------------------- statistics (log-cluster bootstrap on sufficient statistics)
class Boot:
    """Cluster bootstrap over logs with the repo convention (default_rng(0).integers(G, size=(B, G))), kept as a count matrix W (B, G)
    so any statistic that is a function of per-log sums is resampled by one matrix product."""

    def __init__(self, logs, B=10_000, seed=0):
        import pandas as pd
        self.c, u = pd.factorize(np.asarray(logs))
        self.G, self.B = len(u), B
        idx = np.random.default_rng(seed).integers(self.G, size=(B, self.G))
        self.W = np.zeros((B, self.G))
        np.add.at(self.W, (np.arange(B)[:, None], idx), 1.0)

    def sums(self, V):
        """Per-token terms (n, k) -> (point sums (k,), bootstrap sums (B, k))."""
        V = np.asarray(V, np.float64)
        S = np.zeros((self.G, V.shape[1]))
        np.add.at(S, self.c, np.nan_to_num(V))
        return S.sum(0), self.W @ S


def ci(b):
    return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def pc(point, b, f="{:+.3f}"):
    lo, hi = ci(b)
    return f"{f.format(point)} [{f.format(lo)}, {f.format(hi)}]"


def reg(bt, x, y, m, floor):
    """OLS y = a + g x on mask m: dict of (point, boot) for slope, icpt, r2, ratio (through origin), sign agreement."""
    m = m & np.isfinite(x) & np.isfinite(y)
    xm, ym = np.where(m, x, 0.0), np.where(m, y, 0.0)
    f = m & (np.abs(x) >= floor)
    V = np.stack([m.astype(float), xm, ym, xm * xm, xm * ym, ym * ym, f.astype(float), (f & (np.sign(x) == np.sign(y))).astype(float)], 1)

    def st(s):
        n, sx, sy, sxx, sxy, syy, nf, ag = np.moveaxis(s, -1, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vx, vy, cxy = sxx - sx * sx / n, syy - sy * sy / n, sxy - sx * sy / n
            g = cxy / vx
            return dict(slope=g, icpt=(sy - g * sx) / n, r2=cxy * cxy / (vx * vy), ratio=sxy / sxx, sign=ag / nf)
    p, b = bt.sums(V)
    P, Bt = st(p), st(b)
    return {k: (float(P[k]), Bt[k]) for k in P} | {"n": int(m.sum())}


def ols_boot(bt, X, y, m):
    """OLS coefficients of y on X (n, p) over mask m: (beta (p,), boot (B, p))."""
    m = m & np.isfinite(y) & np.isfinite(X).all(1)
    Xm, ym = np.where(m[:, None], X, 0.0), np.where(m, y, 0.0)
    p = X.shape[1]
    A, bA = bt.sums((Xm[:, :, None] * Xm[:, None, :]).reshape(len(X), -1))
    c, bc = bt.sums(Xm * ym[:, None])
    eye = 1e-9 * np.eye(p)
    return np.linalg.solve(A.reshape(p, p) + eye, c), np.linalg.solve(bA.reshape(-1, p, p) + eye, bc[:, :, None])[:, :, 0]


def deciles(v, m, k=10):
    """Equal-count bins 0 .. k-1 of |v| by rank inside mask m (ties by token order); -1 outside."""
    out = np.full(len(v), -1)
    i = np.flatnonzero(m)
    out[i[np.argsort(np.abs(v[i]), kind="stable")]] = np.arange(len(i)) * k // len(i)
    return out


def shares(bt, w, dec, m, k=10):
    """Share of sum(w) falling in each bin: rows of dict(bin, n_frac, share, lo, hi) + the top-three row."""
    wm = np.where(m, w, 0.0)
    V = np.stack([wm * (dec == q) for q in range(k)] + [wm], 1)
    p, b = bt.sums(V)
    rows = []
    for q in range(k):
        lo, hi = ci(b[:, q] / b[:, k])
        rows.append(dict(bin=q + 1, tokens=int((m & (dec == q)).sum()), share=p[q] / p[k], lo=lo, hi=hi, mean_w=float(wm[dec == q].sum() / max((m & (dec == q)).sum(), 1))))
    t3, bt3 = p[k - 3:k].sum() / p[k], b[:, k - 3:k].sum(1) / b[:, k]
    lo, hi = ci(bt3)
    return rows, dict(share=float(t3), lo=lo, hi=hi, total=float(p[k] / max(m.sum(), 1)))


# ---------------------------------------------------------------- data
def load_tab(data):
    t = np.load(CR / data / "tab.npz")
    return {k: t[k] for k in t.files}


def member_poses(mem, names, data):
    """Stored (n, 8, 3) poses of one ladder member in tab order (NaN rows where a token is missing), or None when nothing is stored."""
    def align(tok, P):
        pos = {t: i for i, t in enumerate(tok)}
        out = np.full((len(names), 8, 3), np.nan)
        ix = np.array([pos.get(t, -1) for t in names])
        out[ix >= 0] = P[ix[ix >= 0]]
        return out
    if mem == "wa:navtest":
        w = pickle.load(open(D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl", "rb"))["trajectories"]
        return align(list(w), np.stack([np.asarray(v, np.float64)[:, :3] for v in w.values()]))
    if mem == "wa:navhard":
        z = np.load(D / "runs/op_parity/navhard/wajepa/navhard_preds.npz")
        return align(z["tokens"].astype(str).tolist(), z["poses"].astype(np.float64))
    if mem.startswith("ap2:"):
        f = D / "runs/alpasim/ap2/offline/full-AB/poses.npz"
        if not f.exists():
            return None
        z = np.load(f)
        return align(z["tokens"].astype(str).tolist(), z[mem[4:]].astype(np.float64))
    f = D / "runs/bench/ol" / data / "preds" / f"{mem}__base.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return align(z["tokens"].astype(str).tolist(), z["poses"].astype(np.float64))


def blind_features(tab, kap, a, Lh, cmd):
    X = [tab["pose"][:, :3].reshape(len(kap), -1), tab["vel"].reshape(len(kap), -1), tab["acc"].reshape(len(kap), -1),
         tab["speed"][:, None], a[:, None], kap[:, None], Lh[:, None]]
    if cmd:
        X.append(tab["cmd"][:, -1])
    return np.concatenate([np.asarray(x, np.float64) for x in X], 1)


def navtrain_tab():
    tabs = [load_tab(p.name) for p in sorted(CR.glob("navtrain_full.s*of12")) if "@" not in p.name and (p / "tab.npz").exists()]
    t = {k: np.concatenate([x[k] for x in tabs]) for k in tabs[0]}
    ok = np.isfinite(t["fut"]).all((1, 2)) & np.isfinite(t["pose"]).all((1, 2))
    return {k: v[ok] for k, v in t.items()}


class Board:
    """Per-token innovations of the log (x) and of every ladder member (y) on one board, for every extrapolation and the 2 s / 4 s horizons."""

    def __init__(self, data, ladder, run, blind=True):
        from threadpoolctl import threadpool_limits
        from jevdrive.common import n_cpus
        self.tab = tab = load_tab(data)
        self.ok = np.isfinite(tab["fut"]).all((1, 2))
        self.names, self.log = tab["names"].astype(str), tab["log"].astype(str)
        self.fut = np.nan_to_num(tab["fut"].astype(np.float64))
        self.P, self.fam, self.missing = {}, {}, []
        for fam, mems in ladder.items():
            got = []
            for m in mems:
                p = member_poses(m, self.names, data)
                if p is None:
                    self.missing.append(m)
                else:
                    self.P[m] = p
                    got.append(m)
            if got:
                self.fam[fam] = got
        tr = navtrain_tab() if blind else None
        self.ext, self.x, self.y = {}, {}, {}
        for mode in MODES:
            kap, a, sE, Lh = self.ext[mode] = extrap(tab["pose"], tab["speed"], mode)
            if tr is not None:
                kt, at, st, lt = extrap(tr["pose"], tr["speed"], mode)
                const = np.broadcast_to(tr["fut"].astype(np.float64).mean(0), self.fut.shape)
            for h in (7, 3):
                self.x[mode, h] = innov(self.fut, kap, sE, h)
                Y = {m: innov(np.nan_to_num(p), kap, sE, h) for m, p in self.P.items()}
                for m, p in self.P.items():
                    bad = ~np.isfinite(p).all((1, 2))
                    Y[m] = tuple(np.where(bad, np.nan, v) for v in Y[m])
                Y["EXT (extrapolation)"] = (np.zeros(len(kap)), np.zeros(len(kap)), np.zeros(len(kap)))
                Y["LOG"] = self.x[mode, h]
                if tr is not None and mode != "ccv":
                    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
                    Y["CONST (navtrain mean future)"] = innov(const, kap, sE, h)
                    xt = innov(tr["fut"].astype(np.float64), kt, st, h)
                    with threadpool_limits(limits=max(1, min(16, n_cpus() // 4))):
                        for lab, cmd in (("BLIND-E (ego history only)", False), ("BLIND-EC (+ command)", True)):
                            Xt, Xe = blind_features(tr, kt, at, lt, cmd), blind_features(tab, kap, a, Lh, cmd)
                            pr = [HGB(max_iter=300, learning_rate=0.1, random_state=0).fit(Xt, xt[k]).predict(Xe) for k in (0, 1)]
                            Y[lab] = (pr[0], pr[1], np.full(len(kap), np.nan))
                    run.info(f"{data}: blind rungs fitted on {len(tr['names'])} navtrain tokens, mode {mode}, h {T_OUT[h]} s")
                self.y[mode, h] = Y
        syn = [f for f in SYNTH if f in self.y["cc", 7]]
        self.fam = {f: [f] for f in syn if f != "LOG"} | self.fam | {"LOG": ["LOG"]}
        # tokens where every stored member has a plan and the log future exists
        self.common = self.ok & np.all([np.isfinite(p).all((1, 2)) for p in self.P.values()], 0)
        self.bt = Boot(self.log)

    def xy(self, mem, mode="cc", h=7, axis="lat"):
        k = {"lat": 0, "lon": 1, "psi": 2}[axis]
        return self.x[mode, h][k], self.y[mode, h][mem][k]

    def arc_matched(self, fams, mode="cc"):
        """Lateral gain with log and plan compared at the same arc length s* = min(L_log, L_plan) (s* >= 2 m): fam -> [(gain, boot), (top-decile gain, boot)]."""
        kap = self.ext[mode][0]
        cumL, _ = arclen(self.fut)
        res = {}
        for fam in fams:
            per = []
            for q_ in self.fam[fam]:
                Pm = np.nan_to_num(self.P[q_])
                s_ = np.minimum(cumL[:, -1], arclen(Pm)[0][:, -1])
                mm = self.common & (s_ >= 2.0)
                xa, ya = frenet_d(at_arc(self.fut, s_), kap), frenet_d(at_arc(Pm, s_), kap)
                per.append((reg(self.bt, xa, ya, mm, 0.5), reg(self.bt, xa, ya, mm & (deciles(xa, mm) == 9), 0.5)))
            res[fam] = [(float(np.mean([p_[i]["slope"][0] for p_ in per])), np.mean([p_[i]["slope"][1] for p_ in per], 0)) for i in (0, 1)]
        return res

    def gains(self, m, mode="cc", h=7, axis="lat", fams=None, extra=None):
        """Gain rows of every family (member mean) with CIs, and the paired difference WA-JEPA - family."""
        x = self.x[mode, h][{"lat": 0, "lon": 1, "psi": 2}[axis]]
        top = m & (deciles(x, m) == 9)
        floor = FLOOR.get(axis, 1.0)
        R = {}
        fam_all = dict(self.fam) | (extra or {})
        for fam, mems in fam_all.items():
            if fams is not None and fam not in fams:
                continue
            if not all(q in self.y[mode, h] for q in mems):
                continue
            per = [(reg(self.bt, x, self.y[mode, h][q][{"lat": 0, "lon": 1, "psi": 2}[axis]], m, floor),
                    reg(self.bt, x, self.y[mode, h][q][{"lat": 0, "lon": 1, "psi": 2}[axis]], top, floor)) for q in mems]
            R[fam] = per
        rows, B = [], {}
        for fam, per in R.items():
            def avg(i, k):
                return float(np.mean([r[i][k][0] for r in per])), np.mean([r[i][k][1] for r in per], 0)
            B[fam] = dict(slope=avg(0, "slope"), top=avg(1, "slope"))
            row = dict(model=fam, n=per[0][0]["n"], n_top=per[0][1]["n"], members=len(per))
            for lab, (i, k) in dict(slope=(0, "slope"), icpt=(0, "icpt"), r2=(0, "r2"), sign=(0, "sign"), top_slope=(1, "slope"),
                                    top_ratio=(1, "ratio"), top_sign=(1, "sign")).items():
                p, b = avg(i, k)
                row[lab], (row[lab + "_lo"], row[lab + "_hi"]) = p, ci(b)
            row["per_member_slope"] = " / ".join(f"{r[0]['slope'][0]:.3f}" for r in per)
            row["per_member_top"] = " / ".join(f"{r[1]['slope'][0]:.3f}" for r in per)
            rows.append(row)
        if WA in B:
            for row in rows:
                for k in ("slope", "top"):
                    d, b = B[WA][k][0] - B[row["model"]][k][0], B[WA][k][1] - B[row["model"]][k][1]
                    row[f"wa_minus_{k}"], (row[f"wa_minus_{k}_lo"], row[f"wa_minus_{k}_hi"]) = d, ci(b)
        return rows, B


# ---------------------------------------------------------------- report
def md(rows, cols=None, f=".3f"):
    import pandas as pd
    d = pd.DataFrame(rows)
    return d[cols or list(d.columns)].to_markdown(index=False, floatfmt=f)


def gain_md(rows):
    out = []
    for r in rows:
        c = lambda k, f="{:.3f}": f"{f.format(r[k])} [{f.format(r[k + '_lo'])}, {f.format(r[k + '_hi'])}]"  # noqa: E731
        o = {"model": r["model"], "n": r["n"], "gain": c("slope"), "intercept (m)": c("icpt", "{:+.3f}"), "R2": f"{r['r2']:.3f}",
             "sign agreement": f"{r['sign']:.3f}", "top-decile gain": c("top_slope"), "top-decile ratio": f"{r['top_ratio']:.3f}",
             "members (gain)": r["per_member_slope"], "members (top)": r["per_member_top"]}
        if "wa_minus_top" in r:
            o["WA-JEPA - model, gain"] = c("wa_minus_slope", "{:+.3f}")
            o["WA-JEPA - model, top-decile gain"] = c("wa_minus_top", "{:+.3f}")
        out.append(o)
    return md(out)


def verdict_axis(D_, lo, hi, S):
    gain_ok = D_ >= MARGIN and lo > 0
    if (lo <= 0) or S < 0.40:
        return "FALSIFIED"
    return "SUPPORTED" if gain_ok and S >= 0.60 else "PARTIAL"


def cmd_report(a):
    import pandas as pd
    from scipy.stats import spearmanr
    from jevdrive import stats
    from jevdrive.bench import tables as BT
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "innov/report", seed=0, config=vars(a)) as run:
        for s in ("navsim/navtest", "navsim/navtrain", "navsim/navhard_two_stage"):
            run.use_split(splits.load(s))
        RES.mkdir(parents=True, exist_ok=True)
        Bd = Board("lb_navtest", NAVTEST, run)
        bt, names, tab = Bd.bt, Bd.names, Bd.tab
        v0 = tab["speed"].astype(np.float64)
        txt, summ = [], dict(n_tokens=len(names), n_common=int(Bd.common.sum()), logs=bt.G, missing_members=Bd.missing, margin=MARGIN)
        H = lambda s: txt.append(f"\n## {s}\n")  # noqa: E731
        # image-ablation members, when the pool job has run
        abl = OUT / "ablate_poses.npz"
        extra = {}
        if abl.exists():
            z = np.load(abl)
            assert (z["tokens"].astype(str) == names).all()
            for mode in MODES:
                kap, _, sE, _ = Bd.ext[mode]
                for h in (7, 3):
                    for k in z.files:
                        if "|" in k:
                            Bd.y[mode, h][k] = innov(z[k].astype(np.float64), kap, sE, h)
            for var in ("ID", "KIN", "SHUF"):
                extra[f"SH30 image {var}"] = [f"SH30-F-s{s}|{var}" for s in (0, 1)]

        # ---- 0. the innovation itself
        H("0. Logged innovation (navtest, E-cc, 4 s)")
        xl, xo, xp = Bd.x["cc", 7]
        kap, acc, sE, Lh = Bd.ext["cc"]
        q = lambda v, m: " / ".join(f"{np.percentile(np.abs(v[m]), p):.2f}" for p in (10, 30, 50, 70, 90, 97))  # noqa: E731
        txt.append(md([dict(axis="lateral (m)", tokens=int(Bd.ok.sum()), **{"|x| P10 / P30 / P50 / P70 / P90 / P97": q(xl, Bd.ok)}, mean=xl[Bd.ok].mean(), sd=xl[Bd.ok].std()),
                       dict(axis="longitudinal (m)", tokens=int(Bd.ok.sum()), **{"|x| P10 / P30 / P50 / P70 / P90 / P97": q(xo, Bd.ok)}, mean=xo[Bd.ok].mean(), sd=xo[Bd.ok].std())]))
        txt.append(f"\nkappa set to 0 (history arc < 1.5 m): {int((Lh < 1.5).sum())} tokens; v0 < 2 m/s: {int((v0 < 2).sum())}; "
                   f"tokens in the gain set (every stored member has a plan): {int(Bd.common.sum())}; not stored: {Bd.missing or 'none'}.")

        # ---- 1. gain ladder
        G = {}
        for ax in ("lat", "lon"):
            rows, B = Bd.gains(Bd.common, axis=ax, extra=extra)
            G[ax] = (rows, B)
            pd.DataFrame(rows).to_csv(RES / f"gain_ladder_{ax}.csv", index=False)
            H(f"1. Gain ladder, {'lateral' if ax == 'lat' else 'longitudinal'} (navtest, E-cc, 4 s, {int(Bd.common.sum())} tokens; 95% CI over logs)")
            txt.append(gain_md(rows))

        # ---- 2. score link
        U = {s: BT.load("navtest", f"SH30-F-s{s}")[0].reindex(names) for s in (0, 1)}
        UW = BT.load("navtest", WA)[0].reindex(names)
        sc = 100 * (U[0].score.to_numpy(float) + U[1].score.to_numpy(float)) / 2
        scw = 100 * UW.score.to_numpy(float)
        gap = sc - scw
        ms = Bd.ok & np.isfinite(gap)
        summ["gap_total"] = float(gap[ms].mean())
        from turn_oracle import Data
        TD = Data(["sh"])
        Fd = [TD.one(f"SH30-F-s{s}")[0].reindex(names) for s in (0, 1)]
        fl = {
            "NC failure": sum((U[s].NC < 1).to_numpy(float) for s in (0, 1)) / 2,
            "NC failure, SH30-specific": sum(((U[s].NC < 1) & (UW.NC >= 1)).to_numpy(float) for s in (0, 1)) / 2,
            "TTC-only failure": sum(((U[s].TTC < 1) & (U[s].NC >= 1)).to_numpy(float) for s in (0, 1)) / 2,
            "TTC-only, SH30-specific": sum(((U[s].TTC < 1) & (U[s].NC >= 1) & (UW.TTC >= 1)).to_numpy(float) for s in (0, 1)) / 2,
            "DAC failure": sum(f["DAC fail %"].to_numpy(float) for f in Fd) / 200,
            "DAC inside-cut": sum(f["inside-cut %"].to_numpy(float) for f in Fd) / 200,
            "DAC cannot-make-turn": sum(f["cannot-make-turn %"].to_numpy(float) for f in Fd) / 200,
            "DAC other": sum(f["other DAC fail %"].to_numpy(float) for f in Fd) / 200,
            "WA-JEPA DAC failure": (UW.DAC < 1).to_numpy(float), "WA-JEPA NC failure": (UW.NC < 1).to_numpy(float),
        }

        def share_block(mode, mask, tag):
            xl_, xo_, _ = Bd.x[mode, 7]
            dl, do = deciles(xl_, mask), deciles(xo_, mask)
            rk = lambda d: np.where(mask, d, -1)  # noqa: E731
            n = mask.sum()
            rl = np.zeros(len(mask)); rl[np.flatnonzero(mask)[np.argsort(np.abs(xl_[mask]), kind="stable")]] = np.arange(n) / n
            ro = np.zeros(len(mask)); ro[np.flatnonzero(mask)[np.argsort(np.abs(xo_[mask]), kind="stable")]] = np.arange(n) / n
            dj = deciles(np.maximum(rl, ro), mask)
            out, top = {}, {}
            for ax, d in (("lat", rk(dl)), ("lon", rk(do)), ("joint", rk(dj))):
                rows, t3 = shares(bt, gap, d, mask)
                l_s, _ = shares(bt, 100 - sc, d, mask)
                l_w, _ = shares(bt, 100 - scw, d, mask)
                xx = np.abs({"lat": xl_, "lon": xo_}.get(ax, np.maximum(rl, ro)))
                for r, s_, w_ in zip(rows, l_s, l_w):
                    sel = mask & (d == r["bin"] - 1)
                    r.update(axis=ax, set=tag, x_lo=float(xx[sel].min()), x_hi=float(xx[sel].max()), gap=float(gap[sel].mean()),
                             sh30_lost_share=s_["share"], wa_lost_share=w_["share"])
                out[ax], top[ax] = rows, t3
                for nm, f in fl.items():
                    fr, f3 = shares(bt, f, d, mask)
                    top[ax, nm] = f3 | dict(count=float(f[mask].sum()), by_bin=[r["share"] for r in fr])
            return out, top, (dl, do, dj)

        SH, TOP, (dl, do, dj) = share_block("cc", ms, "all")
        for ax, nm in (("lat", "lateral"), ("lon", "longitudinal"), ("joint", "joint (larger of the two axis ranks)")):
            H(f"2. Share of the SH30 - WA-JEPA EPDMS gap by decile of logged |innovation|, {nm} (navtest, {int(ms.sum())} tokens, gap {summ['gap_total']:+.2f})")
            txt.append(md([{"decile": r["bin"], "|x| range" if ax != "joint" else "rank range": f"{r['x_lo']:.2f}-{r['x_hi']:.2f}", "tokens": r["tokens"],
                            "gap (points)": f"{r['gap']:+.2f}", "share of gap": f"{100 * r['share']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]",
                            "share of SH30 lost points": f"{100 * r['sh30_lost_share']:.1f}%", "share of WA-JEPA lost points": f"{100 * r['wa_lost_share']:.1f}%"} for r in SH[ax]]))
            t = TOP[ax]
            txt.append(f"\nTop three deciles: **{100 * t['share']:.1f}% [{100 * t['lo']:.1f}, {100 * t['hi']:.1f}]** of the gap "
                       f"= {t['share'] * summ['gap_total']:+.2f} EPDMS (upper bound of an innovation method on this axis).")
        pd.DataFrame([r for ax in SH for r in SH[ax]]).to_csv(RES / "gap_share.csv", index=False)
        H("2b. Failure shares in the top three deciles of logged |innovation| (seed-mean counts; share [95% CI]); uniform = 30%")
        rows = []
        for nm in fl:
            r = {"set": nm, "tokens (seed mean)": f"{TOP['lat', nm]['count']:.1f}"}
            for ax in ("lat", "lon", "joint"):
                t = TOP[ax, nm]
                r[f"top-3 share, {ax}"] = f"{100 * t['share']:.0f}% [{100 * t['lo']:.0f}, {100 * t['hi']:.0f}]"
                r[f"lowest-5 share, {ax}"] = f"{100 * sum(t['by_bin'][:5]):.0f}%"
            rows.append(r)
        txt.append(md(rows))
        pd.DataFrame([dict(set=nm, axis=ax, **{f"d{i + 1}": v for i, v in enumerate(TOP[ax, nm]["by_bin"])}) for nm in fl for ax in ("lat", "lon", "joint")]).to_csv(RES / "failure_share.csv", index=False)
        # 3 x 3 tercile grid
        tl, to = deciles(xl, ms, 3), deciles(xo, ms, 3)
        tot = gap[ms].sum()
        H("2c. Gap share on the lateral x longitudinal tercile grid (share of gap / tokens / gap in points)")
        txt.append(md([{"lateral tercile": i + 1, **{f"lon tercile {j + 1}": f"{100 * gap[ms & (tl == i) & (to == j)].sum() / tot:.1f}% / {int((ms & (tl == i) & (to == j)).sum())} / {gap[ms & (tl == i) & (to == j)].mean():+.2f}" for j in range(3)}} for i in range(3)]))

        # ---- 3. false innovation
        H("3. False innovation: plan innovation where the logged innovation is near zero")
        fam_show = [f for f in list(Bd.fam) + list(extra) if f not in ("EXT (extrapolation)",)]
        allf = dict(Bd.fam) | extra
        cont = ms & (dl <= 2) & (do <= 2) & (v0 >= 2)
        sets = {"lat": {"lowest 2 deciles": ms & (dl <= 1), "|x_lat| < 0.25 m": ms & (np.abs(xl) < 0.25), "log continues (both axes lowest 3 deciles, v0 >= 2)": cont},
                "lon": {"lowest 2 deciles": ms & (do <= 1), "|x_lon| < 0.5 m": ms & (np.abs(xo) < 0.5), "log continues (both axes lowest 3 deciles, v0 >= 2)": cont}}
        thr = {"lat": (0.5, 1.0), "lon": (1.0, 2.0)}
        frows = []
        for ax in ("lat", "lon"):
            for sn, m_ in sets[ax].items():
                ya = {}
                for fam in fam_show:
                    ys = [Bd.xy(q, axis=ax)[1] for q in allf[fam]]
                    mm = m_ & np.all([np.isfinite(y) for y in ys], 0)
                    ab = np.mean([np.abs(y) for y in ys], 0)
                    ya[fam] = ab
                    frows.append(dict(axis=ax, set=sn, model=fam, n=int(mm.sum()), mean_abs=float(ab[mm].mean()),
                                      median_abs=float(np.mean([np.median(np.abs(y[mm])) for y in ys])), p90_abs=float(np.mean([np.percentile(np.abs(y[mm]), 90) for y in ys])),
                                      signed_mean=float(np.mean([y[mm].mean() for y in ys])),
                                      share_ge_a=float(np.mean([(np.abs(y[mm]) >= thr[ax][0]).mean() for y in ys])),
                                      share_ge_b=float(np.mean([(np.abs(y[mm]) >= thr[ax][1]).mean() for y in ys]))))
                for fam in fam_show:
                    if fam != WA and WA in ya:
                        mm = m_ & np.isfinite(ya[fam]) & np.isfinite(ya[WA])
                        r = stats.paired(ya[fam][mm], ya[WA][mm], groups=Bd.log[mm])
                        frows[[i for i, x in enumerate(frows) if (x["axis"], x["set"], x["model"]) == (ax, sn, fam)][0]].update(minus_wa=r["mean"], minus_wa_lo=r["lo"], minus_wa_hi=r["hi"])
        pd.DataFrame(frows).to_csv(RES / "false_innovation.csv", index=False)
        for ax in ("lat", "lon"):
            for sn in sets[ax]:
                rr = [r for r in frows if r["axis"] == ax and r["set"] == sn]
                txt.append(f"\n**{'Lateral' if ax == 'lat' else 'Longitudinal'}, {sn}** ({rr[0]['n']} tokens; |y| in m)\n")
                txt.append(md([{"model": r["model"], "mean |y|": f"{r['mean_abs']:.3f}", "median": f"{r['median_abs']:.3f}", "P90": f"{r['p90_abs']:.3f}", "signed mean": f"{r['signed_mean']:+.3f}",
                                f"|y| >= {thr[ax][0]} m": f"{100 * r['share_ge_a']:.1f}%", f"|y| >= {thr[ax][1]} m": f"{100 * r['share_ge_b']:.1f}%",
                                "mean |y| - WA-JEPA [95% CI]": (f"{r['minus_wa']:+.3f} [{r['minus_wa_lo']:+.3f}, {r['minus_wa_hi']:+.3f}]" if "minus_wa" in r else "")} for r in rr]))

        # ---- 4. P3
        H("4. P3: per-token EPDMS gap on turn bucket alone, innovation alone, both")
        turn = BT.navtest_strata().reindex(names).turn.to_numpy(str)
        dyaw = BT.navtest_strata().reindex(names).dyaw.to_numpy(float)
        XT = np.stack([turn == t for t in TURNS[1:]], 1).astype(float)
        XI = np.concatenate([np.stack([dl == q for q in range(1, 10)], 1), np.stack([do == q for q in range(1, 10)], 1)], 1).astype(float)
        one = np.ones((len(names), 1))
        Xs = {"M_T": np.concatenate([one, XT], 1), "M_I": np.concatenate([one, XI], 1), "M_TI": np.concatenate([one, XT, XI], 1)}
        c_lat, c_lon = np.zeros(18), np.zeros(18)
        c_lat[8], c_lat[:4], c_lon[17], c_lon[9:13] = 1, -0.2, 1, -0.2                      # top decile - mean of the lowest five (decile 1 is the reference, 0)
        beta = {k: ols_boot(bt, X, gap, ms) for k, X in Xs.items()}
        prow = []
        for i, t in enumerate(TURNS[1:]):
            bT, bTI = beta["M_T"], beta["M_TI"]
            prow.append({"contrast": f"turn {t} deg vs < 5", "alone": pc(bT[0][1 + i], bT[1][:, 1 + i], "{:+.2f}"), "with the other factor": pc(bTI[0][1 + i], bTI[1][:, 1 + i], "{:+.2f}"),
                         "shrink": f"{100 * (1 - abs(bTI[0][1 + i]) / abs(bT[0][1 + i])):.0f}%"})
            summ[f"p3_turn_{t}"] = dict(alone=float(bT[0][1 + i]), both=float(bTI[0][1 + i]), both_ci=ci(bTI[1][:, 1 + i]))
        for nm, c in (("lateral top decile vs lowest five", c_lat), ("longitudinal top decile vs lowest five", c_lon)):
            bI, bTI = beta["M_I"], beta["M_TI"]
            pI, pTI = bI[0][1:] @ c, bTI[0][4:] @ c
            prow.append({"contrast": nm, "alone": pc(pI, bI[1][:, 1:] @ c, "{:+.2f}"), "with the other factor": pc(pTI, bTI[1][:, 4:] @ c, "{:+.2f}"), "shrink": f"{100 * (1 - abs(pTI) / abs(pI)):.0f}%"})
            summ[f"p3_{nm.split()[0]}"] = dict(alone=float(pI), both=float(pTI), both_ci=ci(bTI[1][:, 4:] @ c))
        txt.append(md(prow))
        # out-of-fold R2, 5 folds of logs
        ul = np.unique(Bd.log)
        fold = dict(zip(np.random.default_rng(0).permutation(ul), np.arange(len(ul)) % 5))
        fo = np.array([fold[x] for x in Bd.log])
        E = {}
        for k, X in ({"M_0": one} | Xs).items():
            pr = np.zeros(len(names))
            for f in range(5):
                tr_, te = ms & (fo != f), ms & (fo == f)
                pr[te] = X[te] @ np.linalg.lstsq(X[tr_], gap[tr_], rcond=None)[0]
            E[k] = np.where(ms, (gap - pr) ** 2, 0.0)
        p, b = bt.sums(np.stack([E["M_0"], E["M_T"], E["M_I"], E["M_TI"]], 1))
        r2 = lambda i: (1 - p[i] / p[0], 1 - b[:, i] / b[:, 0])  # noqa: E731
        d2 = lambda i, j: ((p[j] - p[i]) / p[0], (b[:, j] - b[:, i]) / b[:, 0])  # noqa: E731
        orow = [{"model": n_, "out-of-fold R2 [95% CI]": pc(*r2(i), "{:.4f}")} for i, n_ in ((1, "M_T turn bucket"), (2, "M_I innovation deciles"), (3, "M_TI both"))]
        orow += [{"model": "M_TI - M_T (innovation adds)", "out-of-fold R2 [95% CI]": pc(*d2(3, 1), "{:+.4f}")}, {"model": "M_TI - M_I (turn bucket adds)", "out-of-fold R2 [95% CI]": pc(*d2(3, 2), "{:+.4f}")},
                 {"model": "M_I - M_T", "out-of-fold R2 [95% CI]": pc(*d2(2, 1), "{:+.4f}")}]
        summ["p3_oof"] = dict(M_T=float(r2(1)[0]), M_I=float(r2(2)[0]), M_TI=float(r2(3)[0]), innov_adds=[float(d2(3, 1)[0]), *ci(d2(3, 1)[1])], turn_adds=[float(d2(3, 2)[0]), *ci(d2(3, 2)[1])])
        txt.append("\n" + md(orow))
        q5 = deciles(xl, ms, 5)
        rho = spearmanr(np.abs(dyaw[ms]), np.abs(xl[ms])).statistic
        rho_o = spearmanr(np.abs(dyaw[ms]), np.abs(xo[ms])).statistic
        summ["spearman_dyaw_xlat"], summ["spearman_dyaw_xlon"] = float(rho), float(rho_o)
        txt.append(f"\nCollinearity: Spearman(|dyaw|, |x_lat|) = {rho:.3f}, Spearman(|dyaw|, |x_lon|) = {rho_o:.3f}. Cells: tokens / mean gap (points); cells under 50 tokens are not identified.\n")
        txt.append(md([{"turn bucket": t, **{f"lat quintile {j + 1}": f"{int((ms & (turn == t) & (q5 == j)).sum())} / " + (f"{gap[ms & (turn == t) & (q5 == j)].mean():+.2f}" if (ms & (turn == t) & (q5 == j)).sum() else "-") for j in range(5)}} for t in TURNS]))
        q5o = deciles(xo, ms, 5)
        txt.append("\n" + md([{"turn bucket": t, **{f"lon quintile {j + 1}": f"{int((ms & (turn == t) & (q5o == j)).sum())} / " + (f"{gap[ms & (turn == t) & (q5o == j)].mean():+.2f}" if (ms & (turn == t) & (q5o == j)).sum() else "-") for j in range(5)}} for t in TURNS]))

        # ---- 5. robustness
        H("5. Robustness of the definition (SH30 vs WA-JEPA; D = WA-JEPA - SH30 top-decile gain, S = gap share in the top three deciles)")
        rrow, V = [], {}
        for tag, mode, h, sub in (("primary: E-cc, 4 s, all tokens", "cc", 7, None), ("E-cc, 4 s, v0 >= 2 m/s", "cc", 7, v0 >= 2), ("E-cv, 4 s, all tokens", "cv", 7, None),
                                  ("E-cv, 4 s, v0 >= 2 m/s", "cv", 7, v0 >= 2), ("E-ccv, 4 s, all tokens", "ccv", 7, None), ("E-cc, 2 s, all tokens", "cc", 3, None)):
            mg = Bd.common if sub is None else Bd.common & sub
            mk = ms if sub is None else ms & sub
            if h == 7:
                _, top_, _ = share_block(mode, mk, tag)
            for ax in ("lat", "lon"):
                rows, B = Bd.gains(mg, mode=mode, h=h, axis=ax, fams=["SH30", WA, "BLIND-EC (+ command)"])
                r = {x["model"]: x for x in rows}
                S = top_[ax]["share"] if h == 7 else np.nan
                Dv, lo, hi = r["SH30"]["wa_minus_top"], r["SH30"]["wa_minus_top_lo"], r["SH30"]["wa_minus_top_hi"]
                V[tag, ax] = dict(D=Dv, lo=lo, hi=hi, S=S, verdict=verdict_axis(Dv, lo, hi, S) if h == 7 else "n/a",
                                  sh30=r["SH30"]["slope"], wa=r[WA]["slope"], sh30_top=r["SH30"]["top_slope"], wa_top=r[WA]["top_slope"],
                                  d_all=r["SH30"]["wa_minus_slope"], d_all_lo=r["SH30"]["wa_minus_slope_lo"], d_all_hi=r["SH30"]["wa_minus_slope_hi"], n=r["SH30"]["n"])
                v = V[tag, ax]
                rrow.append({"definition": tag, "axis": ax, "n": v["n"], "SH30 gain": f"{v['sh30']:.3f}", "WA-JEPA gain": f"{v['wa']:.3f}",
                             "WA - SH30 gain": f"{v['d_all']:+.3f} [{v['d_all_lo']:+.3f}, {v['d_all_hi']:+.3f}]", "SH30 top": f"{v['sh30_top']:.3f}", "WA-JEPA top": f"{v['wa_top']:.3f}",
                             "D": f"{Dv:+.3f} [{lo:+.3f}, {hi:+.3f}]", "S": "" if h != 7 else f"{100 * S:.1f}% [{100 * top_[ax]['lo']:.1f}, {100 * top_[ax]['hi']:.1f}]", "cell": v["verdict"]})
        txt.append(md(rrow))
        summ["robustness"] = {f"{k[0]} | {k[1]}": v for k, v in V.items()}
        # supplementary read-outs on the primary set: heading innovation, arc-length-matched lateral, partial gain
        srow = []
        rows, _ = Bd.gains(Bd.common, axis="psi", fams=["SH30", WA])
        for r in rows:
            srow.append({"read-out": "heading innovation (deg)", "model": r["model"], "gain": f"{r['slope']:.3f} [{r['slope_lo']:.3f}, {r['slope_hi']:.3f}]", "top-decile gain": f"{r['top_slope']:.3f} [{r['top_slope_lo']:.3f}, {r['top_slope_hi']:.3f}]",
                         "WA-JEPA - model (top)": f"{r.get('wa_minus_top', 0):+.3f} [{r.get('wa_minus_top_lo', 0):+.3f}, {r.get('wa_minus_top_hi', 0):+.3f}]"})
        am = [f for f in ("cinque (shipped, G frames)", "P2 (no hinge)", "SH30", "OT30 (off-track rows)", "AP2 (AlpaSim inputs, m 4)", WA) if f in Bd.fam]
        res = Bd.arc_matched(am)
        for fam in am:
            srow.append({"read-out": "lateral at matched arc length (m)", "model": fam, "gain": pc(*res[fam][0], "{:.3f}"), "top-decile gain": pc(*res[fam][1], "{:.3f}"),
                         "WA-JEPA - model (top)": pc(res[WA][1][0] - res[fam][1][0], res[WA][1][1] - res[fam][1][1]) if fam != WA else ""})
        summ["arc_matched_lat_top_D"] = [float(res[WA][1][0] - res["SH30"][1][0]), *ci(res[WA][1][1] - res["SH30"][1][1])]
        for ax in ("lat", "lon"):
            x_ = Bd.x["cc", 7][0 if ax == "lat" else 1]
            pg = {}
            for fam in ("CONST (navtrain mean future)", "BLIND-EC (+ command)", "SH30", WA):
                bb = [ols_boot(bt, np.stack([np.ones(len(x_)), x_, sE[:, 7], v0, acc, kap], 1), Bd.xy(q_, axis=ax)[1], Bd.common) for q_ in Bd.fam[fam]]
                pg[fam] = (float(np.mean([b_[0][1] for b_ in bb])), np.mean([b_[1][:, 1] for b_ in bb], 0))
            for fam in pg:
                srow.append({"read-out": f"partial gain, {ax} (covariates s_E, v0, a, kappa)", "model": fam, "gain": pc(*pg[fam], "{:.3f}"), "top-decile gain": "",
                             "WA-JEPA - model (top)": ("all-token: " + pc(pg[WA][0] - pg[fam][0], pg[WA][1] - pg[fam][1])) if fam != WA else ""})
        txt.append("\n" + md(srow))

        # ---- 6. image ablation
        if extra:
            H("6. Image ablation (SH30-F-s0 / s1, navtest; front tokens swapped, ego / history / command intact; x = the token's own logged innovation)")
            arow = []
            for ax in ("lat", "lon"):
                rows, B = G[ax]
                r = {x["model"]: x for x in rows}
                for fam in ["BLIND-EC (+ command)", "SH30"] + list(extra):
                    o = {"axis": ax, "model": fam, "gain": f"{r[fam]['slope']:.3f} [{r[fam]['slope_lo']:.3f}, {r[fam]['slope_hi']:.3f}]", "R2": f"{r[fam]['r2']:.3f}", "sign agreement": f"{r[fam]['sign']:.3f}",
                         "top-decile gain": f"{r[fam]['top_slope']:.3f} [{r[fam]['top_slope_lo']:.3f}, {r[fam]['top_slope_hi']:.3f}]"}
                    if fam in extra:
                        for k in ("slope", "top"):
                            p_, b_ = B[fam][k][0] / B["SH30"][k][0], B[fam][k][1] / B["SH30"][k][1]
                            o[f"retention ({'all' if k == 'slope' else 'top decile'})"] = pc(p_, b_, "{:.2f}")
                            summ[f"ablate_{ax}_{fam.split()[-1]}_{k}"] = [float(p_), *ci(b_)]
                    arow.append(o)
            txt.append(md(arow))
            ade = lambda A, Bp, m_: float(np.linalg.norm(A[m_, :, :2] - Bp[m_, :, :2], axis=-1).mean())  # noqa: E731
            z = np.load(abl)
            txt.append("\nADE to the log (m, gain set): " + ", ".join(f"{var} {np.mean([ade(z[f'SH30-F-s{s}|{var}'].astype(np.float64), Bd.fut, Bd.common) for s in (0, 1)]):.3f}" for var in ("ID", "KIN", "SHUF"))
                       + f"; ADE of KIN / SHUF plans to the intact plan: " + ", ".join(f"{var} {np.mean([ade(z[f'SH30-F-s{s}|{var}'].astype(np.float64), z[f'SH30-F-s{s}|ID'].astype(np.float64), Bd.common) for s in (0, 1)]):.3f}" for var in ("KIN", "SHUF")) + ".")

        # ---- 7. navhard stage 1
        H("7. navhard stage 1 (G frames, real tokens with a logged future; gain ladder only)")
        Bh = Board("lb_navhard", NAVHARD, run, blind=False)
        summ["navhard_tokens"], summ["navhard_missing"] = int(Bh.common.sum()), Bh.missing
        for ax in ("lat", "lon"):
            rows, _ = Bh.gains(Bh.common, axis=ax)
            pd.DataFrame(rows).to_csv(RES / f"navhard_gain_{ax}.csv", index=False)
            txt.append(f"\n**{'Lateral' if ax == 'lat' else 'Longitudinal'}** ({int(Bh.common.sum())} tokens, {Bh.bt.G} logs)\n\n" + gain_md(rows))
        resh = Bh.arc_matched([f for f in NAVHARD if f in Bh.fam])
        txt.append("\n**Lateral at matched arc length** (navhard stage 1)\n\n" + md([{"model": f, "gain": pc(*r_[0], "{:.3f}"), "top-decile gain": pc(*r_[1], "{:.3f}"),
                   "WA-JEPA - model": pc(resh[WA][0][0] - r_[0][0], resh[WA][0][1] - r_[0][1]), "WA-JEPA - model (top)": pc(resh[WA][1][0] - r_[1][0], resh[WA][1][1] - r_[1][1])} for f, r_ in resh.items()]))
        summ["navhard_arc_matched_lat_D"] = [float(resh[WA][0][0] - resh["SH30"][0][0]), *ci(resh[WA][0][1] - resh["SH30"][0][1])]

        # ---- verdict
        vd = {}
        for ax in ("lat", "lon"):
            v = V["primary: E-cc, 4 s, all tokens", ax]
            vd[ax] = dict(v, P1=bool(v["d_all"] >= MARGIN and v["d_all_lo"] > 0), P1_top=bool(v["D"] >= MARGIN and v["lo"] > 0))
        vd["joint_share_top3"] = TOP["joint"]
        cells = [vd["lat"]["verdict"], vd["lon"]["verdict"]]
        vd["overall"] = "SUPPORTED" if "SUPPORTED" in cells else "FALSIFIED" if cells == ["FALSIFIED"] * 2 else "PARTIAL"
        summ["verdict"] = vd
        summ["top3"] = {ax: TOP[ax] for ax in ("lat", "lon", "joint")}
        (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        head = [f"# innovation gain: tables (navtest unless stated; `innov.py report`)\n",
                f"Registered verdict cells (E-cc, 4 s, all tokens): lateral **{cells[0]}**, longitudinal **{cells[1]}**, overall **{vd['overall']}** (margin {MARGIN}).\n"]
        (RES / "tables.md").write_text("\n".join(head + txt) + "\n")
        np.savez_compressed(RES / "per_token.npz", token=names, x_lat=xl, x_lon=xo, gap=gap, dec_lat=dl, dec_lon=do, dec_joint=dj,
                            **{f"y_lat|{q_}": Bd.xy(q_, axis="lat")[1] for f_ in ("SH30", WA) for q_ in Bd.fam[f_]}, **{f"y_lon|{q_}": Bd.xy(q_, axis="lon")[1] for f_ in ("SH30", WA) for q_ in Bd.fam[f_]})
        figures(G, SH, frows, extra)
        run.summary.update(verdict=cells, overall=vd["overall"])
        run.info(json.dumps(vd, indent=1, default=float))


def figures(G, SH, frows, extra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    fd = RES / "figs"
    fd.mkdir(exist_ok=True)
    col = lambda f: ps.PALETTE["vermillion"] if f == WA else ps.PREDICTION if f.startswith("SH30") and "image" not in f and "AlpaSim" not in f else ps.PALETTE["green"] if "leaked" in f else ps.PALETTE["orange"] if "image" in f else ps.BASELINE  # noqa: E731
    # 1. gain ladder
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 4.6), sharey=True)
    for ax, key, lab in zip(axs, ("lat", "lon"), ("lateral", "longitudinal")):
        rows = [r for r in G[key][0] if r["model"] not in ("LOG",)]
        y = np.arange(len(rows))[::-1]
        for yy, r in zip(y, rows):
            c = col(r["model"])
            ax.errorbar(r["slope"], yy + 0.13, xerr=[[r["slope"] - r["slope_lo"]], [r["slope_hi"] - r["slope"]]], fmt="o", ms=3.2, color=c, lw=0.8, capsize=1.5)
            ax.errorbar(r["top_slope"], yy - 0.13, xerr=[[r["top_slope"] - r["top_slope_lo"]], [r["top_slope_hi"] - r["top_slope"]]], fmt="s", ms=3.0, mfc="white", color=c, lw=0.8, capsize=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels([r["model"] for r in rows], fontsize=6.5)
        ax.axvline(1, color="#999999", lw=0.5)
        ax.axvline(0, color="#999999", lw=0.5)
        ax.set_xlabel(f"{lab} innovation gain")
        ax.grid(axis="y", visible=False)
    axs[0].plot([], [], "o", ms=3.2, color="#444444", label="all tokens")
    axs[0].plot([], [], "s", ms=3.0, mfc="white", color="#444444", label="top decile of logged |innovation|")
    axs[0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, fd / "gain_ladder")
    plt.close(fig)
    # 2. gap share by decile
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.3), sharey=True)
    for ax, key, lab in zip(axs, ("lat", "lon", "joint"), ("lateral", "longitudinal", "joint")):
        r = SH[key]
        s = np.array([100 * x["share"] for x in r])
        ax.bar(np.arange(1, 11), s, 0.72, color=ps.PREDICTION, yerr=[s - [100 * x["lo"] for x in r], [100 * x["hi"] for x in r] - s], error_kw=dict(lw=0.6, capsize=1.5))
        ax.plot(np.arange(1, 11), [100 * x["wa_lost_share"] for x in r], "-o", ms=2.2, lw=0.8, color=ps.PALETTE["vermillion"], label="WA-JEPA lost points")
        ax.axhline(10, color="#777777", lw=0.6, ls="--")
        ps.zero_line(ax)
        ps.bars(ax)
        ax.set_xticks(np.arange(1, 11))
        ax.set_xlabel(f"decile of logged |{lab} innovation|")
    axs[0].set_ylabel("share of SH30 - WA-JEPA gap (%)")
    axs[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, fd / "gap_share")
    plt.close(fig)
    # 3. false innovation
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 4.2), sharey=True)
    for ax, key, lab in zip(axs, ("lat", "lon"), ("|lateral plan innovation| >= 0.5 m", "|longitudinal plan innovation| >= 1 m")):
        rr = [r for r in frows if r["axis"] == key and r["set"].startswith("log continues") and r["model"] != "LOG"]
        y = np.arange(len(rr))[::-1]
        ax.barh(y, [100 * r["share_ge_a"] for r in rr], 0.7, color=[col(r["model"]) for r in rr])
        ax.set_yticks(y)
        ax.set_yticklabels([r["model"] for r in rr], fontsize=6.5)
        ax.set_xlabel(f"tokens with {lab} (%)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    ps.save(fig, fd / "false_innovation")
    plt.close(fig)


# ---------------------------------------------------------------- image ablation
def swap_rows(tab, seed=0):
    """KIN: the kinematically nearest token of another log (standardised v0, a, yaw rate); SHUF: a random token of another log."""
    kap, a, _, _ = extrap(tab["pose"], tab["speed"], "cc")
    v0 = tab["speed"].astype(np.float64)
    Z = np.stack([v0, a, v0 * kap], 1)
    Z = (Z - Z.mean(0)) / Z.std(0)
    _, lg = np.unique(tab["log"], return_inverse=True)
    n = len(v0)
    kin = np.zeros(n, int)
    for i in range(0, n, 512):
        d = ((Z[i:i + 512, None] - Z[None]) ** 2).sum(-1)
        d[lg[i:i + 512, None] == lg[None]] = np.inf
        kin[i:i + 512] = d.argmin(1)
    rng = np.random.default_rng(seed)
    shuf = rng.integers(0, n, n)
    while (bad := lg[shuf] == lg).any():
        shuf[bad] = rng.integers(0, n, int(bad.sum()))
    return dict(ID=np.arange(n), KIN=kin, SHUF=shuf), Z


def cmd_ablate(a):
    import torch
    from jevdrive.bench import navsim as BN
    from jevdrive.bench.models import resolve
    from jevdrive.data import splits
    from jevdrive.run import Run
    BN._pp_path()
    import pp_train as T
    with Run("op_parity", "innov/ablate", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        OUT.mkdir(parents=True, exist_ok=True)
        dev, data = torch.device("cuda"), "lb_navtest"
        ms = [resolve(f"SH30-F-s{s}", check=True) for s in (0, 1)]
        side_ok = (BN.cache_dir(data, "gimm") / "side.npy").exists()
        S = T.Store([data], dev, need_side=side_ok, frames=ms[0].frames, host=True)     # tokens stay on the host: a few GB of VRAM
        rows, Z = swap_rows(S.tab)
        n = S.n if not a.limit else min(S.n, a.limit)
        out = dict(names=S.tab["names"][:n], **{f"rows|{k}": v[:n] for k, v in rows.items()})
        for m in ms:
            model = T.load_pmodel(m.name, dev) if not m.ckpt else BN._load_ckpt(T, m.ckpt, dev)
            assert getattr(model, "mem", None) is None
            sl = model.net.slices
            pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
            for var, perm in rows.items():
                mu = np.zeros((n, 33, 15), np.float32)
                with torch.no_grad():
                    for i in range(0, n, 128):
                        r = torch.arange(i, min(i + 128, n), device=dev)
                        side = S.side[r] if side_ok else None
                        o = model(S.front[perm[i:i + len(r)]], S.ego[r], S.tc[r], side, None).float().cpu().numpy()
                        mu[i:i + len(r)] = o[:, pi].reshape(-1, 33, 15)
                out[f"{m.name}|{var}|pos"], out[f"{m.name}|{var}|yaw"] = mu[:, :, 0:3], mu[:, :, 11]
                run.info(f"{m.name} {var}: {n} plans, peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
        np.savez(OUT / ("ablate_plans.npz" if not a.limit else "ablate_plans_smoke.npz"), **out)
        dz = np.linalg.norm(Z[rows["KIN"]] - Z, axis=1)
        run.summary.update(tokens=n, peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30, kin_match_median_sd=float(np.median(dz)), kin_match_p90_sd=float(np.percentile(dz, 90)))


def cmd_abl_export(a):
    from jevdrive import navsim_zs as Z
    from jevdrive.bench import navsim as BN
    from jevdrive.run import Run
    os.environ["OPI_ROOT"] = BN.OL_REL
    sys.path.insert(0, str(REPO / "experiments/op_openloop/lib"))
    import op_interp as OP
    with Run("op_parity", "innov/abl-export", config=vars(a)) as run:
        z = np.load(OUT / "ablate_plans.npz")
        mt = OP.meta("lb_navtest")
        assert z["names"].tolist() == mt["names"]
        out, gate = dict(tokens=z["names"]), {}
        for k in [k[:-4] for k in z.files if k.endswith("|pos")]:
            zz = dict(plan_pos=z[k + "|pos"], plan_yaw=z[k + "|yaw"])
            out[k] = np.stack([OP.adapt(zz, i, mt, Z.T_OUT, **OP.ADAPTERS["base"])[0] for i in range(len(mt["names"]))])
            if k.endswith("|ID"):
                ref = np.load(D / "runs/bench/ol/lb_navtest/preds" / f"{k[:-3]}-warp__base.npz")
                assert (ref["tokens"].astype(str) == z["names"].astype(str)).all()
                gate[k] = float(np.abs(out[k][:, :, :2] - ref["poses"][:, :, :2]).max())
        run.info(json.dumps(gate))
        run.summary.update(id_gate_max_abs_m=gate)
        if max(gate.values()) > 1e-3:
            raise SystemExit(f"identity gate failed: {gate}")
        np.savez(OUT / "ablate_poses.npz", **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("report")
    q = sp.add_parser("ablate")
    q.add_argument("--limit", type=int, default=0)
    sp.add_parser("abl-export")
    a = ap.parse_args()
    {"report": cmd_report, "ablate": cmd_ablate, "abl-export": cmd_abl_export}[a.cmd](a)
