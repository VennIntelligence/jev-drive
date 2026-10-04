"""Tables + figure of the turn-calibration analysis (experiments/op_closed_loop/results/turn_calibration.md). CPU only.

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/turn_calibration_report.py [--out DIR]

Needs the three tables written by turn_calibration_{b2d_extract,openloop_extract,sparse}.py. Writes tables.md and turn_calibration.png into DIR.

Statistic everywhere: ratio = sum(model_k * sign(req_k)) / sum(|req_k|) over the ticks of a bin (a wrong-sign tick counts negative; 1 = the model turns exactly as
much as the reference requires), 95% CI from a cluster bootstrap (B2D: route; WOD: scene; NAVSIM: log), 2000 draws.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir  # noqa: E402

T = data_dir() / "runs/op_closed_loop/turn_calibration"
EDGES = np.array([0.003, 0.006, 0.012, 0.025, 0.05, 0.1, 0.25])
PRIMARY = ("v2-drive-s2-q", "v2-drive-s3-q", "v2-opc-s2-q", "v2-opc-s3-q", "v2-lsc-s2-q", "v2-lsc-s3-q")
rng = np.random.default_rng(0)


def ratio_ci(req, mod, cl, B=2000):
    """(ratio, lo, hi, n, n_clusters, same-sign share)."""
    n = len(req)
    if n < 5:
        return (np.nan,) * 3 + (n, 0, np.nan)
    u, inv = np.unique(cl, return_inverse=True)
    num = np.bincount(inv, mod * np.sign(req), len(u))
    den = np.bincount(inv, np.abs(req), len(u))
    r = num.sum() / den.sum()
    if len(u) < 3:
        return (r, np.nan, np.nan, n, len(u), float((np.sign(mod) == np.sign(req)).mean()))
    w = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=B)
    bs = (w @ num) / np.maximum(w @ den, 1e-12)
    return (r, *np.nanpercentile(bs, [2.5, 97.5]), n, len(u), float((np.sign(mod) == np.sign(req)).mean()))


def binned(req, mod, cl, mask=None, edges=EDGES):
    mask = np.ones(len(req), bool) if mask is None else mask
    ok = mask & np.isfinite(req) & np.isfinite(mod)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (np.abs(req) >= lo) & (np.abs(req) < hi)
        out.append(ratio_ci(req[m], mod[m], cl[m]))
    return out


def fmt(t):
    if t[3] < 5 or np.isnan(t[0]):
        return "n/a (n %d)" % t[3]
    ci = "[%.2f, %.2f]" % (t[1], t[2]) if not np.isnan(t[1]) else "[n/a]"
    return "%.2f %s (n %d, %d cl)" % (t[0], ci, t[3], t[4])


def row_label(lo, hi):
    return "%.3f-%.3f (R %d-%d m)" % (lo, hi, round(1 / hi), round(1 / lo))


def load():
    b = {k: v for k, v in np.load(T / "b2d_ticks.npz", allow_pickle=True).items()}
    o = {k: v for k, v in np.load(T / "openloop.npz", allow_pickle=True).items()}
    s = {k: v for k, v in np.load(T / "sparse.npz", allow_pickle=True).items()}
    return b, o, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(T))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b, o, s = load()
    L = []
    P = lambda *x: L.append(" ".join(str(i) for i in x))  # noqa: E731
    prim = np.array([any(x.startswith(p) for p in PRIMARY) for x in b["arm"]])
    proutes = set(b["route"][prim])
    held = ~np.isin(b["route"], list(proutes))
    P("B2D rows: %d total; primary arms (drive / opc / lsc, seeds 2-3): %d rows, %d routes; held-out routes (other arms, routes not in the primary 19): %d rows, %d routes"
      % (len(prim), prim.sum(), len(proutes), held.sum(), len(set(b["route"][held]))))
    act, plan1, plan2 = b["act_k"], -b["plan_k1"], -b["plan_k2"]
    req = b["req_v"]
    srcs = {
        "B2D action head (primary 19 routes)": (req[prim], act[prim], b["route"][prim]),
        "B2D action head (held-out routes)": (req[held], act[held], b["route"][held]),
        "B2D plan, 1 s point (primary)": (b["req_p1"][prim], plan1[prim], b["route"][prim]),
        "B2D plan, 1 s point (held-out)": (b["req_p1"][held], plan1[held], b["route"][held]),
        "WOD plan (real 10 Hz history)": (o["wod_real_req10"], o["wod_real_mod10"], o["wod_real_cluster"]),
        "NAVSIM plan (2 Hz history interpolated)": (o["nav_warp_req10"], o["nav_warp_mod10"], o["nav_warp_cluster"]),
        "NAVSIM plan (native 2 Hz exam protocol)": (o["nav_native_req10"], o["nav_native_mod10"], o["nav_native_cluster"]),
    }
    P("\n## 1. Ratio (model / required) by required |kappa| (bins of the reference; reference = dense route chord curvature over max(4, min(v, 20)) m on B2D; logged human future chord curvature at 10 m on WOD / NAVSIM)\n")
    P("| |kappa_req| bin | " + " | ".join(srcs) + " |")
    P("|---|" + "---|" * len(srcs))
    res = {k: binned(*v) for k, v in srcs.items()}
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        P("| %s | " % row_label(lo, hi) + " | ".join(fmt(res[k][i]) for k in srcs) + " |")
    # same-sign shares
    P("\nSame-sign share per bin (action head primary / plan primary / NAVSIM plan): " + "; ".join(
        "%s %.0f%% / %.0f%% / %.0f%%" % (row_label(lo, hi), 100 * res["B2D action head (primary 19 routes)"][i][5], 100 * res["B2D plan, 1 s point (primary)"][i][5],
                                         100 * res["NAVSIM plan (2 Hz history interpolated)"][i][5]) for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:]))))
    # ---- 1b arcs sensitivity
    P("\n## 1b. Sensitivity of the open-loop ratios to the arc of the chord (a = 10 / 15 / 25 m), required |kappa| 0.012-0.1\n")
    P("| source | a=10 | a=15 | a=25 |\n|---|---|---|---|")
    for nm, k in (("WOD plan", "wod_real"), ("NAVSIM plan (interp.)", "nav_warp"), ("NAVSIM plan (native)", "nav_native")):
        cells = []
        for ar in (10, 15, 25):
            r_, m_ = o["%s_req%d" % (k, ar)], o["%s_mod%d" % (k, ar)]
            ok = np.isfinite(r_) & np.isfinite(m_) & (np.abs(r_) >= 0.012) & (np.abs(r_) < 0.1)
            cells.append(fmt(ratio_ci(r_[ok], m_[ok], o[k + "_cluster"][ok])))
        P("| %s | " % nm + " | ".join(cells) + " |")
    P("\nB2D action head, required over A = 5 m / 10 m / max(4, min(v, 20)) m, |kappa_req| >= 0.04 (primary):")
    cells = []
    for nm in ("req5", "req10", "req_v"):
        r_ = b[nm][prim]
        ok = np.abs(r_) >= 0.04
        cells.append("%s %s" % (nm, fmt(ratio_ci(r_[ok], act[prim][ok], b["route"][prim][ok]))))
    P("; ".join(cells))
    # ---- 2 B2D action head breakdowns
    ap_ = prim
    P("\n## 2. B2D action head: what else the ratio depends on (primary arms; turn ticks |kappa_req| >= 0.04 unless stated)\n")
    big = np.abs(req) >= 0.04
    mid = (np.abs(req) >= 0.012) & (np.abs(req) < 0.04)
    P("| split | level | action head (turn ticks >= 0.04) | action head (0.012-0.04) | plan 1 s (>= 0.04) |\n|---|---|---|---|---|")

    def split(name, levels):
        for lv, m in levels:
            c = []
            for sel, rr, mm in ((big, req, act), (mid, req, act), (big, b["req_p1"], plan1)):
                k = ap_ & m & sel if rr is req else ap_ & m & (np.abs(b["req_p1"]) >= 0.04)
                c.append(fmt(ratio_ci(rr[k], mm[k], b["route"][k])))
            P("| %s | %s | %s |" % (name, lv, " | ".join(c)))

    v = b["v"]
    split("speed (m/s)", [("%g-%g" % (lo, hi), (v >= lo) & (v < hi)) for lo, hi in ((1, 2.5), (2.5, 4), (4, 6), (6, 9), (9, 30))])
    split("side", [("left (req < 0)", req < 0), ("right (req > 0)", req > 0)])
    split("owner", [("op (action head steers)", b["lat"] == "op"), ("route zone (route steers)", (b["lat"] == "route") & (b["why"] == "zone")),
                    ("div (route steers)", (b["lat"] == "route") & (b["why"] == "div"))])
    split("arm", [(nm, np.char.startswith(b["arm"].astype(str), "v2-%s-" % nm)) for nm in ("drive", "opc", "lsc")])
    split("route desire input", [("desire = 0", b["desire"] == 0), ("desire != 0 (route turn desire given)", b["desire"] != 0)])
    P("\nmaneuver radius (route minimum radius of the turn the tick belongs to, smoothed 1.5 m), action head:\n")
    P("| min radius | action head | plan 1 s |\n|---|---|---|")
    mt = b["mtype"]
    for lo, hi in ((0, 6), (6, 9), (9, 13), (13, 40)):
        m = ap_ & (mt >= 2) & (b["mrmin"] >= lo) & (b["mrmin"] < hi) & big
        m2 = ap_ & (mt >= 2) & (b["mrmin"] >= lo) & (b["mrmin"] < hi) & (np.abs(b["req_p1"]) >= 0.04)
        P("| %d-%d m | %s | %s |" % (lo, hi, fmt(ratio_ci(req[m], act[m], b["route"][m])), fmt(ratio_ci(b["req_p1"][m2], plan1[m2], b["route"][m2]))))
    P("\nB2D turn types present (route maneuvers, dense route): ticks by type (0 straight, 1 bend 8-25 deg, 2 turn 25-60, 3 turn 60-120, 4 tight > 120): " + str({int(t): int((mt[prim] == t).sum()) for t in range(5)}))
    P("img command given (non-'None') ticks in these arms: %d" % int(np.sum(b["img"].astype(str) != "None")))

    # ---- 2b speed x curvature (pooled over all B2D routes) and a joint log-ratio regression
    P("\n## 2b. Speed x curvature, action head, all B2D routes pooled (primary + held-out; cells with >= 100 ticks)\n")
    cb = [(0.012, 0.04), (0.04, 0.1), (0.1, 0.3)]
    vb = [(1, 2.5), (2.5, 4), (4, 6), (6, 9)]
    P("| speed (m/s) | " + " | ".join("|k| %g-%g" % c for c in cb) + " |\n|---|" + "---|" * len(cb))
    for lo, hi in vb:
        cells = []
        for c0, c1 in cb:
            m = (b["v"] >= lo) & (b["v"] < hi) & (np.abs(req) >= c0) & (np.abs(req) < c1)
            cells.append(fmt(ratio_ci(req[m], act[m], b["route"][m])) if m.sum() >= 100 else "n/a (n %d)" % m.sum())
        P("| %g-%g | " % (lo, hi) + " | ".join(cells) + " |")
    P("\nJoint fit on all routes, log(ratio of a cell) = a + b ln|k| + c ln v (cells of the table above weighted by ticks; cluster bootstrap over routes):\n")
    P("| quantity | b (curvature exponent - 1) | c (speed exponent) |\n|---|---|---|")
    uR, invR = np.unique(b["route"], return_inverse=True)
    for nm, rq, md in (("action head", req, act), ("plan 1 s", b["req_p1"], plan1)):
        cells, fits = [], []
        for lo, hi in vb:
            for c0, c1 in cb:
                m = (b["v"] >= lo) & (b["v"] < hi) & (np.abs(rq) >= c0) & (np.abs(rq) < c1)
                cells.append((m, np.log(np.sqrt(c0 * c1)), np.log(b["v"][m].mean()) if m.any() else 0.0))
        nums = np.array([np.bincount(invR[m], (md * np.sign(rq))[m], len(uR)) for m, _, _ in cells])
        dens = np.array([np.bincount(invR[m], np.abs(rq)[m], len(uR)) for m, _, _ in cells])
        cnt = np.array([np.bincount(invR[m], None, len(uR)) for m, _, _ in cells])
        X = np.array([[1.0, lk, lv] for _, lk, lv in cells])

        def fit(w):
            n_, d_, c_ = nums @ w, dens @ w, cnt @ w
            r = n_ / np.maximum(d_, 1e-12)
            ok = (c_ >= 100) & (r > 0)
            if ok.sum() < 4:
                return np.array([np.nan] * 3)
            sw = np.sqrt(c_[ok])
            return np.linalg.lstsq(X[ok] * sw[:, None], np.log(r[ok]) * sw, rcond=None)[0]
        est = fit(np.ones(len(uR)))
        bs = np.array([fit(rng.multinomial(len(uR), np.ones(len(uR)) / len(uR)).astype(float)) for _ in range(1000)])
        ci = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        P("| %s | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] |" % (nm, est[1], ci[0, 1], ci[1, 1], est[2], ci[0, 2], ci[1, 2]))
    # ---- 2c integrated turn: what the action head alone would have steered over each junction turn
    P("\n## 2c. Integrated heading change over each junction turn (sum of kappa * v * dt over the turn's ticks / the route's turn angle), runs that cover the turn\n")
    P("| model curvature | ratio over (run, turn) groups [route-cluster CI] | groups | routes |\n|---|---|---|---|")
    grp = {}
    for i in np.nonzero((b["mtype"] >= 2) & np.isfinite(b["mang"]))[0]:
        grp.setdefault((b["arm"][i], b["route"][i], round(float(b["mang"][i]), 1)), []).append(i)
    rows = []
    for (arm_, rt, ang_), ii in grp.items():
        ii = np.array(ii)
        cover = np.sum(np.abs(b["kloc"][ii]) * b["v"][ii] * 0.05) / np.radians(abs(ang_))
        if cover < 0.85 or cover > 1.6:
            continue
        rows.append((rt, np.sum(act[ii] * np.sign(ang_) * b["v"][ii] * 0.05) / np.radians(abs(ang_)), np.sum(plan1[ii] * np.sign(ang_) * b["v"][ii] * 0.05) / np.radians(abs(ang_)) ))
    if rows:
        rt_ = np.array([r[0] for r in rows])
        for j, nm in ((1, "action head"), (2, "plan 1 s point")):
            val = np.array([r[j] for r in rows])
            u_, inv_ = np.unique(rt_, return_inverse=True)
            sm, ct = np.bincount(inv_, val, len(u_)), np.bincount(inv_, None, len(u_))
            w = rng.multinomial(len(u_), np.ones(len(u_)) / len(u_), size=2000)
            bs = (w @ sm) / np.maximum(w @ ct, 1e-9)
            P("| %s | %.2f [%.2f, %.2f] | %d | %d |" % (nm, sm.sum() / ct.sum(), *np.percentile(bs, [2.5, 97.5]), len(val), len(u_)))
    # open-loop splits
    P("\n## 3. Open-loop plan ratio by turn angle of the logged future, speed and side (|kappa_req(10 m)| >= 0.006)\n")
    P("| source | split | level | ratio |\n|---|---|---|---|")
    for nm, k in (("WOD plan", "wod_real"), ("NAVSIM plan (interp.)", "nav_warp"), ("NAVSIM plan (native)", "nav_native")):
        r_, m_, cl, vv, ang, sd = o[k + "_req10"], o[k + "_mod10"], o[k + "_cluster"], o[k + "_v"], np.abs(o[k + "_ang"]), o[k + "_side"]
        base = np.isfinite(r_) & np.isfinite(m_) & (np.abs(r_) >= 0.006)
        for sp, lv_list in (("turn angle (deg)", [("<10", ang < 10), ("10-25", (ang >= 10) & (ang < 25)), ("25-60", (ang >= 25) & (ang < 60)), (">=60", ang >= 60)]),
                            ("speed (m/s)", [("<5", vv < 5), ("5-9", (vv >= 5) & (vv < 9)), ("9-14", (vv >= 9) & (vv < 14)), (">=14", vv >= 14)]),
                            ("side (sign of required)", [("left", r_ < 0), ("right", r_ > 0)])):
            for lv, m in lv_list:
                kk = base & m
                P("| %s | %s | %s | %s |" % (nm, sp, lv, fmt(ratio_ci(r_[kk], m_[kk], cl[kk]))))
    # ---- 4 linear vs nonlinear
    P("\n## 4. Linear or nonlinear: slope of the binned ratio on log|kappa_req| (bins >= 0.006), cluster-bootstrapped\n")
    P("| source | slope per ln-unit of |kappa| | 95% CI | reading |\n|---|---|---|---|")

    def lslope(req_, mod_, cl_, edges=EDGES[1:]):
        u, inv = np.unique(cl_, return_inverse=True)
        mids = np.sqrt(edges[:-1] * edges[1:])
        num = np.zeros((len(u), len(mids)))
        den = np.zeros_like(num)
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (np.abs(req_) >= lo) & (np.abs(req_) < hi) & np.isfinite(mod_)
            num[:, j] = np.bincount(inv[m], (mod_ * np.sign(req_))[m], len(u))
            den[:, j] = np.bincount(inv[m], np.abs(req_)[m], len(u))

        def fit(w):
            n, d = w @ num, w @ den
            ok = d > 0
            if ok.sum() < 3:
                return np.nan
            return np.polyfit(np.log(mids[ok]), n[ok] / d[ok], 1)[0]
        w0 = np.ones(len(u))
        est = fit(w0)
        bs = np.array([fit(rng.multinomial(len(u), np.ones(len(u)) / len(u)).astype(float)) for _ in range(1000)])
        return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)

    for k, v_ in srcs.items():
        e, lo, hi = lslope(*v_)
        P("| %s | %+.3f | [%+.3f, %+.3f] | %s |" % (k, e, lo, hi, "ratio falls with sharpness" if hi < 0 else "ratio rises with sharpness" if lo > 0 else "flat within CI (linear)"))
    # ---- 5 candidate map
    P("\n## 5. Candidate calibration map (not adopted): fitted on the primary 19 routes, checked on held-out routes\n")
    r_p, m_p, c_p = req[prim], act[prim], b["route"][prim]
    ok = np.abs(r_p) >= 0.01
    g_lin = np.sum(m_p[ok] * np.sign(r_p[ok])) / np.sum(np.abs(r_p[ok]))
    # power law on the binned ratio: ratio = g0 * (|k| / 0.05)^q  (log-log fit on bin ratios >= 0.006, weighted by the bin's |req| mass)
    bb = binned(r_p, m_p, c_p, edges=EDGES[1:])
    mids = np.sqrt(EDGES[1:-1] * EDGES[2:])
    rat = np.array([t[0] for t in bb])
    wt = np.array([t[3] for t in bb], float)
    good = np.isfinite(rat) & (rat > 0)
    q, lg = np.polyfit(np.log(mids[good] / 0.05), np.log(rat[good]), 1, w=np.sqrt(wt[good]))
    P("- linear gain: model = %.3f x required (ratio of sums over |kappa_req| >= 0.01); the calibration is kappa_cal = kappa_act / %.3f" % (g_lin, g_lin))
    P("- power law: ratio(|k|) = %.3f x (|k| / 0.05)^%+.3f; kappa_cal solves ratio(|k|) |k| = |kappa_act|" % (np.exp(lg), q))
    # piecewise-linear gain in log k from the bins (constant outside)
    gl = lambda k: np.interp(np.log(np.maximum(np.abs(k), 1e-6)), np.log(mids[good]), rat[good])  # noqa: E731

    def invert(kact, gfun):
        """Required |k| whose model output equals |kact| (monotone search on a log grid)."""
        grid = np.geomspace(1e-4, 0.5, 600)
        fwd = gfun(grid) * grid
        return np.sign(kact) * np.interp(np.abs(kact), np.maximum.accumulate(fwd), grid)

    # speed-dependent gain from the primary routes' turn ticks (>= 0.04) in speed bins, interpolated on the bin's mean speed
    sb = [(1, 2.5), (2.5, 4), (4, 9)]
    gv, vm = [], []
    for lo, hi in sb:
        m = prim & (np.abs(req) >= 0.04) & (b["v"] >= lo) & (b["v"] < hi)
        gv.append(np.sum(act[m] * np.sign(req[m])) / np.sum(np.abs(req[m])))
        vm.append(b["v"][m].mean())
    P("- speed-dependent gain (turn ticks >= 0.04, speed bins %s, mean speeds %s): %s" % (sb, np.round(vm, 2).tolist(), np.round(gv, 3).tolist()))
    maps = {"identity": lambda k, v_: k, "linear gain": lambda k, v_: k / g_lin,
            "power law": lambda k, v_: invert(k, lambda x: np.exp(lg) * (x / 0.05) ** q), "piecewise-linear gain": lambda k, v_: invert(k, gl),
            "speed-dependent gain": lambda k, v_: k / np.interp(v_, vm, gv)}
    P("\n| map | set | ratio of sums (turn ticks >= 0.012) | CI | RMSE of kappa_cal vs required (1/m) | ratio in 0.012-0.04 | in 0.04-0.1 | in >= 0.1 |\n|---|---|---|---|---|---|---|---|")
    for setname, sel in (("fit (primary)", prim), ("held-out routes", held)):
        for nm, f in maps.items():
            k = sel & (np.abs(req) >= 0.012)
            cal = f(act[k], b["v"][k])
            t = ratio_ci(req[k], cal, b["route"][k])
            rm = float(np.sqrt(np.mean((cal - req[k]) ** 2)))
            cells = []
            for lo, hi in ((0.012, 0.04), (0.04, 0.1), (0.1, 0.3)):
                kk = k & (np.abs(req) >= lo) & (np.abs(req) < hi)
                cells.append("%.2f" % ratio_ci(req[kk], f(act[kk], b["v"][kk]), b["route"][kk])[0])
            P("| %s | %s | %.2f | [%.2f, %.2f] | %.4f | %s |" % (nm, setname, t[0], t[1], t[2], rm, " | ".join(cells)))
    # plan-derived map: fit on half of the NAVSIM logs, check on the other half, WOD and B2D
    cl = o["nav_warp_cluster"].astype(str)
    half = np.array([hash_(c) % 2 for c in cl]) == 0
    P("\nPlan-derived curvature (NAVSIM, interpolated protocol): linear gain fitted on half of the logs, checked on the other half / WOD / B2D (|kappa_req| >= 0.012):")
    r_, m_ = o["nav_warp_req10"], o["nav_warp_mod10"]
    ok = np.isfinite(r_) & np.isfinite(m_) & (np.abs(r_) >= 0.012) & half
    g_p = np.sum(m_[ok] * np.sign(r_[ok])) / np.sum(np.abs(r_[ok]))
    P("\ngain fitted: %.3f (n %d). Residual ratio after dividing the plan curvature by it:\n" % (g_p, ok.sum()))
    P("| set | before | after dividing by the gain |\n|---|---|---|")
    for nm, (rr, mm, cc) in (("NAVSIM other half", (r_[~half], m_[~half], cl[~half])), ("WOD (real 10 Hz)", (o["wod_real_req10"], o["wod_real_mod10"], o["wod_real_cluster"])),
                             ("B2D plan 1 s (primary)", (b["req_p1"][prim], plan1[prim], b["route"][prim])), ("B2D plan 1 s (held-out)", (b["req_p1"][held], plan1[held], b["route"][held]))):
        k = np.isfinite(rr) & np.isfinite(mm) & (np.abs(rr) >= 0.012)
        P("| %s | %s | %s |" % (nm, fmt(ratio_ci(rr[k], mm[k], cc[k])), fmt(ratio_ci(rr[k], mm[k] / g_p, cc[k]))))
    # ---- 6 sparse route
    P("\n## 6. Dense vs sparse route: kinematic pure pursuit through the %d route turns (>= 25 deg) of %d routes\n" % (len(set(zip(s["route"][s["variant"] == "dense"], s["man"][s["variant"] == "dense"]))), len(set(s["route"]))))
    P("| path given | speed (m/s) | peak cross-track vs dense centreline, median [p90] (m) | runs with peak > 1 m | > 1.75 m (lane half-width) | peak commanded |kappa| (1/m), median | heading change delivered / route's |\n|---|---|---|---|---|---|---|")
    names = [("dense", "dense route (shipped zones)"), ("lb50", "leaderboard downsample (50 m)"), ("road_0", "road polyline, 10 m decimation, no noise"),
             ("road5_0", "road polyline, 5 m decimation, no noise"), ("road_1", "road polyline 10 m, noise sd 1 m"), ("road_2", "... sd 2 m"), ("road_4", "... sd 4 m")]
    for vv in (3.0, 5.0, 8.0):
        for key, lab in names:
            m = (s["v"] == vv) & (s["variant"] == key)
            P("| %s | %g | %.2f [%.2f] | %.0f%% | %.0f%% | %.3f | %.2f |" % (lab, vv, np.median(s["peak"][m]), np.percentile(s["peak"][m], 90), 100 * (s["peak"][m] > 1).mean(),
                                                                             100 * (s["peak"][m] > 1.75).mean(), np.median(s["kpeak"][m]), np.median(s["dpsi"][m] / s["turn_deg"][m])))
    P("\nBy turn direction and radius, leaderboard downsample at 5 m/s (peak cross-track median, m; peak |kappa| sparse / dense):\n")
    P("| group | n turns | peak cross-track (lb50) | peak cross-track (dense) | peak |kappa| lb50 / dense |\n|---|---|---|---|---|")
    d0 = (s["v"] == 5.0) & (s["variant"] == "dense")
    l0 = (s["v"] == 5.0) & (s["variant"] == "lb50")
    key = lambda m: dict(zip(zip(s["route"][m], s["man"][m]), range(m.sum())))  # noqa: E731
    kd, kl = key(d0), key(l0)
    common = sorted(set(kd) & set(kl))
    pk_d, pk_l = s["peak"][d0], s["peak"][l0]
    kp_d, kp_l = s["kpeak"][d0], s["kpeak"][l0]
    ang = s["turn_deg"][d0]
    rmin = s["rmin"][d0]
    for lab, sel in (("left turns", ang < 0), ("right turns", ang > 0), ("R_min < 6 m", rmin < 6), ("R_min 6-9 m", (rmin >= 6) & (rmin < 9)), ("R_min 9-13 m", (rmin >= 9) & (rmin < 13)), ("R_min >= 13 m", rmin >= 13)):
        ii = [kd[c] for c in common]
        jj = [kl[c] for c in common]
        sel_c = sel[ii]
        if sel_c.sum() == 0:
            continue
        P("| %s | %d | %.2f | %.2f | %.2f |" % (lab, sel_c.sum(), np.median(pk_l[jj][sel_c]), np.median(pk_d[ii][sel_c]), np.median(kp_l[jj][sel_c] / kp_d[ii][sel_c])))
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figure(out / "turn_calibration.png", srcs, res, b, o, s, prim, req, act, plan1)


def hash_(c):
    import zlib
    return zlib.crc32(c.encode())


def figure(path, srcs, res, b, o, s, prim, req, act, plan1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"B2D action head (primary 19 routes)": ("#1f4e9c", "-", "o"), "B2D action head (held-out routes)": ("#1f4e9c", "--", "s"),
           "B2D plan, 1 s point (primary)": ("#d9731a", "-", "o"), "B2D plan, 1 s point (held-out)": ("#d9731a", "--", "s"),
           "WOD plan (real 10 Hz history)": ("#6b6b6b", "-", "^"), "NAVSIM plan (2 Hz history interpolated)": ("#2a9d8f", "-", "o"),
           "NAVSIM plan (native 2 Hz exam protocol)": ("#2a9d8f", ":", "x")}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1.1]})
    mids = np.sqrt(EDGES[:-1] * EDGES[1:])
    A = ax[0]
    for k, (c, ls, mk) in col.items():
        r = res[k]
        y = np.array([t[0] for t in r])
        lo = np.array([t[1] for t in r])
        hi = np.array([t[2] for t in r])
        n = np.array([t[3] for t in r])
        g = n >= 20
        xs = mids * (1 + 0.025 * list(col).index(k) - 0.07)
        A.plot(xs[g], y[g], ls, color=c, marker=mk, ms=5, lw=1.8, label=k)
        A.fill_between(xs[g], lo[g], hi[g], color=c, alpha=0.10, lw=0)
    A.axhline(1, color="k", lw=0.8)
    A.set_xscale("log")
    A.set_ylim(-0.1, 2.4)
    A.set_xlabel("required |curvature| (1/m); top axis: radius (m)")
    A.set_ylabel("model / required (ratio of sums, 95% CI)")
    A.set_title("A. Ratio vs required curvature: action head flat ~0.5 above 0.012 (below: noise,\nsame-sign 64-80%); plan-derived 1.2-1.5", fontsize=10)
    sec = A.secondary_xaxis("top", functions=(lambda x: 1 / np.maximum(x, 1e-9), lambda x: 1 / np.maximum(x, 1e-9)))
    sec.set_xticks([300, 100, 50, 25, 12, 6])
    sec.set_xticklabels(["300", "100", "50", "25", "12", "6"])
    sec.minorticks_off()
    A.legend(fontsize=7, loc="upper right", ncol=1, frameon=False)
    B = ax[1]
    sp = [(1, 2.5), (2.5, 4), (4, 6), (6, 9), (9, 30)]
    xm = [np.mean(x) if x[1] < 30 else 11 for x in sp]
    for nm, mod, rq, c in (("B2D action head", act, req, "#1f4e9c"), ("B2D plan 1 s", plan1, b["req_p1"], "#d9731a")):
        yy, ll, hh = [], [], []
        for lo, hi in sp:
            m = prim & (b["v"] >= lo) & (b["v"] < hi) & (np.abs(rq) >= 0.04)
            t = ratio_ci(rq[m], mod[m], b["route"][m])
            yy.append(t[0] if t[3] >= 100 else np.nan); ll.append(t[1] if t[3] >= 100 else np.nan); hh.append(t[2] if t[3] >= 100 else np.nan)
        B.errorbar(xm, yy, yerr=[np.array(yy) - np.array(ll), np.array(hh) - np.array(yy)], color=c, marker="o", capsize=3, label=nm + " (turns, |k| >= 0.04)")
    for nm, k, c in (("NAVSIM plan (interp.)", "nav_warp", "#2a9d8f"),):
        yy, ll, hh, xx = [], [], [], []
        for lo, hi in ((3, 6), (6, 9), (9, 14), (14, 30)):
            m = np.isfinite(o[k + "_req10"]) & np.isfinite(o[k + "_mod10"]) & (np.abs(o[k + "_req10"]) >= 0.012) & (o[k + "_v"] >= lo) & (o[k + "_v"] < hi)
            t = ratio_ci(o[k + "_req10"][m], o[k + "_mod10"][m], o[k + "_cluster"][m])
            yy.append(t[0]); ll.append(t[1]); hh.append(t[2]); xx.append((lo + min(hi, 20)) / 2)
        B.errorbar(xx, yy, yerr=[np.array(yy) - np.array(ll), np.array(hh) - np.array(yy)], color=c, marker="s", capsize=3, label=nm + " (|k| >= 0.012)")
    B.axhline(1, color="k", lw=0.8)
    B.set_ylim(0, 2.0)
    B.set_xlabel("speed (m/s)")
    B.set_title("B. Same ratio by speed", fontsize=10)
    B.legend(fontsize=7, frameon=False, loc="upper right")
    C = ax[2]
    keys = [("dense", "dense"), ("road5_0", "road 5 m"), ("road_0", "road 10 m"), ("road_1", "road 10 m\n+noise 1 m"), ("road_2", "+noise 2 m"), ("lb50", "leaderboard\n50 m")]
    for j, vv in enumerate((3.0, 5.0, 8.0)):
        data = [s["peak"][(s["v"] == vv) & (s["variant"] == k)] for k, _ in keys]
        pos = np.arange(len(keys)) + (j - 1) * 0.27
        bp = C.boxplot(data, positions=pos, widths=0.24, showfliers=False, patch_artist=True)
        for p_ in bp["boxes"]:
            p_.set_facecolor(["#c6dbef", "#6baed6", "#08519c"][j]); p_.set_alpha(0.9)
        C.plot([], [], color=["#c6dbef", "#6baed6", "#08519c"][j], lw=6, label="%g m/s" % vv)
    C.axhline(1.0, color="r", lw=0.8, ls="--")
    C.axhline(1.75, color="r", lw=0.8)
    C.set_xticks(range(len(keys)))
    C.set_xticklabels([k for _, k in keys], fontsize=8)
    C.set_ylabel("peak cross-track error to the dense centreline (m)")
    C.set_title("C. Pure pursuit through the %d route turns on a sparse route\n(dashed 1 m margin; solid 1.75 m = leaves the lane)" % len(set(zip(s["route"][s["variant"] == "dense"], s["man"][s["variant"] == "dense"]))), fontsize=10)
    C.legend(fontsize=8, frameon=False, loc="upper left")
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
