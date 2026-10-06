"""Turn gain: is P2's sharp-turn deficit proportional shrinkage of turning, or a cap (model output ceiling / openpilot lateral clip)?  CPU only, stored outputs.

  navtest  (box)  per-token table: logged future vs the plans of P0 (W frames), P2 s0/s1, P2+hinge s0/s1, WA-JEPA -> $OUT/navtest_turn.csv.gz
  hugsim   (box)  per-step table of the HUGSIM 64 runs (spec + exam presets, P2 / P2+hinge, 2 seeds): requested curvature (kappa), openpilot lateral
                  chain trace (act / des / real from sim.log `op_ctrl` lines), speed, steer, pose -> $OUT/hugsim_steps.csv.gz
  report   (Mac)  statistics + figures + the markdown from the two tables (copied to results/turn-gain/data/)

  box:  .venv/bin/python experiments/op_probe/scripts/opj_turn_gain.py navtest|hugsim [--out DIR]
  Mac:  .venv/bin/python experiments/op_probe/scripts/opj_turn_gain.py report
"""
import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results/turn-gain"
OUTB = D / "runs/op_probe/turn_gain"
PRED = D / "runs/op_lb/lb_navtest/preds"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
WA_TRAJ = D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl"
HS = D / "runs/op_parity/hugsim"
MODELS = {"P0": ["warp-cinque_PPP0__base.npz"], "P2": ["warp-cinque_PPP2-F-s0__base.npz", "warp-cinque_PPP2-F-s1__base.npz"],
          "P2H": ["warp-cinque_PPP2H10-F-s0__base.npz", "warp-cinque_PPP2H10-F-s1__base.npz"]}
EDGES = [0, 5, 20, 45, 400]
LABELS = {"P0": "P0 shipped (W)", "P2": "P2", "P2H": "P2 + hinge", "WA": "WA-JEPA"}
COL = {"P0": "#7f7f7f", "P2": "#d95f02", "P2H": "#7570b3", "WA": "#1b9e77"}
B, SEED = 4000, 0


# ---------------------------------------------------------------- plan descriptors (t0 rear-axle frame, x forward, y left, 8 poses at 0.5 s)
def desc(P):
    """P (n, 8, 3) -> dict of arrays. yaw/y at 2 s (idx 3) and 4 s (idx 7), arc length, mean curvature over [0, T], peak segment lateral accel / curvature."""
    P = np.asarray(P, float)
    n = len(P)
    P0 = np.concatenate([np.zeros((n, 1, 3)), P], 1)
    yaw = np.unwrap(P0[:, :, 2], axis=1)
    seg = np.linalg.norm(np.diff(P0[:, :, :2], axis=1), axis=2)                 # (n, 8)
    dyaw = np.diff(yaw, axis=1)
    v = seg / 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        kseg = np.where(seg > 0.5, dyaw / seg, np.nan)                          # curvature of segments longer than 0.5 m
    alat = dyaw / 0.5 * v                                                       # signed, v^2 * kappa
    s2, s4 = seg[:, :4].sum(1), seg.sum(1)
    out = dict(yaw2=np.degrees(yaw[:, 4]), yaw4=np.degrees(yaw[:, 8]), y2=P0[:, 4, 1], y4=P0[:, 8, 1], s2=s2, s4=s4,
               k2=np.where(s2 > 3, yaw[:, 4] / np.maximum(s2, 1e-6), np.nan), k4=np.where(s4 > 6, yaw[:, 8] / np.maximum(s4, 1e-6), np.nan),
               v_mean=s4 / 4.0)
    sgn = np.sign(yaw[:, 8])                                                    # side of the turn: peaks are taken in the turn direction
    out["kpk"] = np.nanmax(np.where(np.isfinite(kseg), kseg * sgn[:, None], -np.inf), 1)
    out["kpk"] = np.where(np.isfinite(out["kpk"]), out["kpk"], np.nan)
    out["apk"] = np.max(alat * sgn[:, None], 1)
    out["amax_abs"] = np.max(np.abs(alat), 1)
    return out


def cmd_navtest(a):
    z = np.load(TAB)
    names, log, fut, v0 = z["names"], z["log"], z["fut"], z["speed"]
    ok = ~np.isnan(fut[:, 0, 0])
    df = pd.DataFrame({"token": names, "log": log, "v0": v0, "ok": ok})
    for k, v in desc(np.nan_to_num(fut)).items():
        df[f"L_{k}"] = np.where(ok, v, np.nan)
    plans = {}
    for m, fs in MODELS.items():
        for i, f in enumerate(fs):
            q = np.load(PRED / f)
            pos = {t: j for j, t in enumerate(q["tokens"].tolist())}
            ix = np.array([pos.get(t, -1) for t in names])
            plans[f"{m}s{i}" if len(fs) > 1 else m] = np.where((ix >= 0)[:, None, None], q["poses"][np.maximum(ix, 0)], np.nan)
    w = pickle.load(open(WA_TRAJ, "rb"))["trajectories"]
    plans["WA"] = np.stack([np.asarray(w[t], float) if t in w else np.full((8, 3), np.nan) for t in names])
    for m, P in plans.items():
        have = ~np.isnan(P[:, 0, 0])
        for k, v in desc(np.nan_to_num(P)).items():
            df[f"{m}_{k}"] = np.where(have, v, np.nan)
    out = Path(a.out or OUTB)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "navtest_turn.csv.gz", index=False, float_format="%.5g")
    print(len(df), "tokens;", {m: int((~np.isnan(P[:, 0, 0])).sum()) for m, P in plans.items()})


# ---------------------------------------------------------------- hugsim per-step table
def cmd_hugsim(a):
    res = pd.read_csv(HS / "results.csv")
    rows = []
    for preset, pre in (("spec", "pp-spec-"), ("exam", "pp-")):
        for arm in ("P2-F-s0", "P2-F-s1", "P2H10-F-s0", "P2H10-F-s1"):
            tag = pre + arm
            r = res[res.tag == tag].drop_duplicates("scenario", keep="last")
            for _, x in r.iterrows():
                d = Path(x.run_dir)
                if not (d / "zs_steps.jsonl").exists():
                    continue
                st = [s for s in map(json.loads, open(d / "zs_steps.jsonl")) if "step" in s and "pos" in s]
                oc = []
                if (d / "sim.log").exists():
                    oc = [json.loads(m.group(1)) for m in (re.match(r"op_ctrl (\{.*\})", l) for l in open(d / "sim.log")) if m]
                for s in st:
                    o = oc[s["step"]] if s["step"] < len(oc) else {}
                    rows.append(dict(preset=preset, arm=arm, scenario=x.scenario, scene=x.scene, step=s["step"], t=s["t"], x=s["pos"][0], z=s["pos"][1],
                                     theta=s["theta"], v=s["v"], steer=s["steer"], kappa=s.get("kappa"), act=o.get("act"), des=o.get("des"),
                                     real=o.get("real"), kmean=o.get("kmean"), n_steps=int(x.steps), end=x.end, hd=x.hdscore, dac=x.dac, nc=x.nc))
    out = Path(a.out or OUTB)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "hugsim_steps.csv.gz", index=False, float_format="%.6g")
    print(len(rows), "steps")


# ---------------------------------------------------------------- report helpers
def cboot(fn, g, B=B, seed=SEED, **cols):
    """Cluster bootstrap over groups g of a statistic fn(**sliced cols) -> (est, lo, hi). Resamples groups; concatenates their rows."""
    g = np.asarray(g)
    codes, uniq = pd.factorize(g)
    order = np.argsort(codes, kind="stable")
    cuts = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    cols = {k: np.asarray(v)[order] for k, v in cols.items()}
    est = fn(**cols)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([np.arange(cuts[p], cuts[p + 1]) for p in pick])
        vals.append(fn(**{k: v[idx] for k, v in cols.items()}))
    return est, float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def slope0(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float((x[m] * y[m]).sum() / (x[m] ** 2).sum()) if m.any() else np.nan


def ratio_of_sums(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(y[m].sum() / x[m].sum()) if m.any() and x[m].sum() != 0 else np.nan


def fmt(r, k=2):
    return f"{r[0]:.{k}f} [{r[1]:.{k}f}, {r[2]:.{k}f}]"



# ---------------------------------------------------------------- report (Mac)
def load_nav():
    d = pd.read_csv(RES / "data/navtest_turn.csv.gz")
    d = d[d.ok & d.WA_yaw4.notna()].reset_index(drop=True)                     # common set: logged future and all 6 plan sets exist
    sg = np.sign(d.L_yaw4).replace(0, 1).values
    d["sg"] = sg
    d["mag"] = d.L_yaw4.abs()
    d["bin"] = pd.cut(d.mag, EDGES, right=False, labels=["<5", "5-20", "20-45", ">45"])
    d["side"] = np.where(sg > 0, "left", "right")
    return d


def stack(d, m):
    """Rows of model m (seeds stacked, same logs) with turn-direction-folded columns: logged x_*, predicted p_* (sign of the logged turn)."""
    ks = {"P2": ["P2s0", "P2s1"], "P2H": ["P2Hs0", "P2Hs1"]}.get(m, [m])
    out = []
    for k in ks:
        o = pd.DataFrame({"log": d.log, "v0": d.v0, "bin": d.bin, "side": d.side, "mag": d.mag, "Lapk": d.L_apk, "Lkpk": d.L_kpk})
        for v in ("yaw2", "yaw4", "y2", "y4", "k2", "k4", "apk", "kpk"):
            sg = d.sg.values if v not in ("apk", "kpk") else 1.0
            o[f"x_{v}"] = d[f"L_{v}"].values * sg
            o[f"p_{v}"] = d[f"{k}_{v}"].values * sg
        out.append(o)
    return pd.concat(out, ignore_index=True)


def gain_tables(d):
    rows = []
    strata = [("all", np.ones(len(d), bool))]
    for b in ("<5", "5-20", "20-45", ">45"):
        strata += [(f"{b} left", ((d.bin == b) & (d.side == "left")).values), (f"{b} right", ((d.bin == b) & (d.side == "right")).values)]
    strata += [(">20 left", ((d.mag >= 20) & (d.side == "left")).values), (">20 right", ((d.mag >= 20) & (d.side == "right")).values),
               (">20 both", (d.mag >= 20).values)]
    for m in ("P0", "P2", "P2H", "WA"):
        S = stack(d, m)
        n = len(d)
        for lab, msk in strata:
            mm = np.tile(msk, len(S) // n)
            s = S[mm]
            for v in ("yaw4", "y4", "y2", "yaw2", "k4"):
                if lab == "all" and v != "yaw4":
                    pass
                r_s = cboot(lambda x, y: slope0(x, y), s.log.values, B=1000, x=s[f"x_{v}"].values, y=s[f"p_{v}"].values)
                r_r = cboot(lambda x, y: ratio_of_sums(x, y), s.log.values, B=1000, x=s[f"x_{v}"].values, y=s[f"p_{v}"].values)
                rows.append(dict(model=m, stratum=lab, n=int(msk.sum()), metric=v, slope0=r_s[0], slope_lo=r_s[1], slope_hi=r_s[2],
                                 ratio=r_r[0], ratio_lo=r_r[1], ratio_hi=r_r[2]))
    return pd.DataFrame(rows)


def edges_fold(d):
    return np.array([0, 2, 5, 10, 20, 30, 45, 60, 80, 120])


def fig_gain(d, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ed = edges_fold(d)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True, sharey=True)
    for a, m in zip(ax, ("P0", "P2", "P2H", "WA")):
        S = stack(d, m)
        a.hexbin(S.x_yaw4.abs().clip(upper=118), (S.p_yaw4).clip(-30, 118), gridsize=45, mincnt=1, bins="log", cmap="Greys", extent=(0, 118, -30, 118))
        S["b"] = pd.cut(S.x_yaw4, ed, right=False)
        q = S.groupby("b", observed=True).apply(lambda g: pd.Series(dict(x=g.x_yaw4.median(), q05=g.p_yaw4.quantile(.05), q25=g.p_yaw4.quantile(.25),
                                                                       q50=g.p_yaw4.median(), q75=g.p_yaw4.quantile(.75), q95=g.p_yaw4.quantile(.95))))
        a.fill_between(q.x, q.q05, q.q95, color=COL[m], alpha=.15, lw=0)
        a.fill_between(q.x, q.q25, q.q75, color=COL[m], alpha=.3, lw=0)
        a.plot(q.x, q.q50, "-o", color=COL[m], ms=3, label="median, IQR, 5-95%")
        a.plot([0, 118], [0, 118], "k--", lw=.8)
        a.set_title(LABELS[m]); a.set_xlabel("logged heading change over 4 s, turn direction (deg)")
    ax[0].set_ylabel("predicted heading change (deg)")
    ax[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig1_gain_scatter")
    # ratio of sums per fine bin
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, v, t in zip(ax, ("yaw4", "y4", "k4"), ("heading change at 4 s", "lateral offset at 4 s", "mean curvature over 4 s")):
        for m in ("P0", "P2", "P2H", "WA"):
            S = stack(d, m)
            S = S[S.x_yaw4 >= 0]
            xs, rs, lo, hi = [], [], [], []
            for l, h in zip(ed[:-1], ed[1:]):
                s = S[(S.x_yaw4 >= l) & (S.x_yaw4 < h)]
                if l < 5 or len(s) < 50:
                    continue
                r = cboot(lambda x, y: ratio_of_sums(x, y), s.log.values, B=500, x=s[f"x_{v}"].values, y=s[f"p_{v}"].values)
                xs.append(s.x_yaw4.median()); rs.append(r[0]); lo.append(r[1]); hi.append(r[2])
            off = {"P0": 0.97, "P2": 1.0, "P2H": 1.03, "WA": 1.06}[m]
            a.errorbar(np.array(xs) * off, rs, [np.array(rs) - lo, np.array(hi) - rs], color=COL[m], marker="o", ms=3, lw=1.2, capsize=2, label=LABELS[m])
        a.axhline(1, color="k", lw=.7, ls="--"); a.set_title(t); a.set_xlabel("logged heading change, turn direction (deg)"); a.set_xscale("log")
        a.set_ylabel("sum predicted / sum logged")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_gain_by_bin")


def save(fig, name):
    (RES).mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / f"{name}.png", dpi=140)
    fig.savefig(RES / f"{name}.pdf")


def fig_ceiling(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9))
    qs = [.5, .75, .9, .95, .99, .995, .999]
    for a, v, t, u in zip(ax[:2], ("apk", "kpk"), ("peak lateral acceleration v^2 k of the plan (turn direction)", "peak segment curvature of the plan (turn direction)"),
                          ("m/s^2", "1/m")):
        for m, k in (("L", "L"), ("P0", "P0"), ("P2", "P2s0"), ("P2H", "P2Hs0"), ("WA", "WA")):
            x = d[f"{k}_{v}"].values * (d.sg.values if False else 1.0)
            x = np.sort(x[np.isfinite(x)])
            a.plot(np.linspace(0, 1, len(x))[int(.5 * len(x)):], x[int(.5 * len(x)):], color="k" if m == "L" else COL[m], lw=2 if m == "L" else 1.3,
                   label="logged" if m == "L" else LABELS[m])
        a.set_title(t, fontsize=9); a.set_xlabel("quantile"); a.set_ylabel(u)
        if v == "apk":
            a.axhline(3.0, color="r", ls=":", lw=1); a.text(.5, 3.02, "openpilot limit 3 m/s^2", color="r", fontsize=8)
        else:
            a.axhline(0.2, color="r", ls=":", lw=1); a.text(.5, 0.202, "openpilot limit 0.2 /m", color="r", fontsize=8)
        a.set_xlim(.5, 1)
    ax[0].legend(fontsize=8)
    # pred peak lat accel vs logged peak lat accel, binned
    ed = np.array([0, .5, 1, 1.5, 2, 2.5, 3, 3.5])
    for m, k in (("P0", "P0"), ("P2", "P2s0"), ("P2H", "P2Hs0"), ("WA", "WA")):
        b = pd.cut(d.L_apk, ed)
        q = d.groupby(b, observed=True).apply(lambda g: pd.Series(dict(x=g.L_apk.median(), m=g[f"{k}_apk"].median(), hi=g[f"{k}_apk"].quantile(.9))))
        ax[2].plot(q.x, q.m, "-o", ms=3, color=COL[m], label=LABELS[m])
        ax[2].plot(q.x, q.hi, ":", color=COL[m])
    ax[2].plot([0, 3.5], [0, 3.5], "k--", lw=.8); ax[2].axhline(3, color="r", ls=":", lw=1)
    ax[2].set_xlabel("logged peak lateral acceleration (m/s^2)"); ax[2].set_ylabel("predicted (median solid, p90 dotted)"); ax[2].set_title("predicted vs logged peak lateral acceleration", fontsize=9)
    fig.tight_layout()
    save(fig, "fig3_ceiling")


def speed_tables(d):
    rows = []
    t = d[d.mag >= 20]
    from scipy.stats import spearmanr
    for m, k in (("P0", ["P0"]), ("P2", ["P2s0", "P2s1"]), ("P2H", ["P2Hs0", "P2Hs1"]), ("WA", ["WA"])):
        pred = np.mean([t[f"{c}_yaw4"].values * t.sg.values for c in k], 0)
        x = t.L_yaw4.values * t.sg.values
        D = x - pred
        r1 = spearmanr(D, t.v0)[0]; r2 = spearmanr(D / x, t.v0)[0]; r3 = spearmanr(D / x, t.v0 ** 2)[0]
        r4 = spearmanr(D / x, t.L_apk)[0]
        q = pd.qcut(t.v0, 3, labels=["slow", "mid", "fast"])
        out = dict(model=m, n=len(t), rho_deficit_v0=r1, rho_relative_deficit_v0=r2, rho_relative_deficit_v0sq=r3, rho_relative_deficit_loggedAlat=r4)
        for lv in ("slow", "mid", "fast"):
            mm = (q == lv).values
            out[f"ratio_{lv}"] = pred[mm].sum() / x[mm].sum()
            out[f"v0_{lv}"] = float(t.v0[mm].median())
        for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 9)):
            mm = ((t.L_apk >= lo) & (t.L_apk < hi)).values
            out[f"ratio_alat{lo}-{hi}"] = pred[mm].sum() / x[mm].sum()
            out[f"n_alat{lo}-{hi}"] = int(mm.sum())
        rows.append(out)
    return pd.DataFrame(rows)


def resid_tables(d):
    rows = []
    for m, k in (("P0", ["P0"]), ("P2", ["P2s0", "P2s1"]), ("P2H", ["P2Hs0", "P2Hs1"]), ("WA", ["WA"])):
        for b in ("<5", "5-20", "20-45", ">45"):
            s = d[d.bin == b]
            x = s.L_yaw4.values * s.sg.values
            for c in k:
                e = s[f"{c}_yaw4"].values * s.sg.values - x
                rows.append(dict(model=m, seed=c, bin=b, n=len(s), bias=e.mean(), rms=np.sqrt((e ** 2).mean()), sd=e.std(), under_gt_30pct=float(((e < -0.3 * np.maximum(x, 5)) & (x > 5)).mean()) if b != "<5" else np.nan,
                                 yerr_rms=float(np.sqrt(((s[f"{c}_y4"].values * s.sg.values - s.L_y4.values * s.sg.values) ** 2).mean()))))
    r = pd.DataFrame(rows)
    return r.groupby(["model", "bin"], as_index=False, sort=False).mean(numeric_only=True)


# ---------------------------------------------------------------- closed loop
def route_k(v, w=4):
    xz = np.asarray(v["xz"], float)
    yaw = np.unwrap(np.asarray(v["yaw"], float))
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    n = len(s)
    k = np.array([(yaw[min(i + w, n - 1)] - yaw[max(i - w, 0)]) / max(s[min(i + w, n - 1)] - s[max(i - w, 0)], 1e-3) for i in range(n)])
    return xz, k


def load_hs():
    import sys
    sys.path.insert(0, str(HERE.parents[1] / "lib"))
    import op_ctrl
    h = pd.read_csv(RES / "data/hugsim_steps.csv.gz")
    R = json.load(open(RES / "data/routes.json"))
    turn = {k: float(np.degrees(np.ptp(np.unwrap(np.asarray(v["yaw"], float))))) for k, v in R.items()}
    h["route_turn"] = h.scene.map(turn)
    h["turning"] = h.route_turn >= 30
    h = h.sort_values(["preset", "arm", "scenario", "step"]).reset_index(drop=True)
    parts = []
    for _, g in h.groupby(["preset", "arm", "scenario"], sort=False):
        xz, k = route_k(R[g.scene.iloc[0]])
        P = g[["x", "z"]].values
        dd = np.linalg.norm(P[:, None, :] - xz[None], axis=2)
        i = dd.argmin(1)
        g = g.assign(kr=k[i], dev=dd.min(1))
        ds = np.hypot(g.x.diff(), g.z.diff())
        g["kach"] = g.theta.diff() / ds.where(ds > 0.3)
        # replay through the openpilot lateral chain (dt 0.25 sim s, delay 0.25) on the logged requested curvature and speed
        st = op_ctrl.OpLateral({"delay": 0.25})
        des, act = [], []
        for kap, v in zip(g.kappa.values, g.v.values):
            _, lg = st.step(float(kap), float(v), 0.25)
            des.append(lg["des"]); act.append(lg["act"])
        g["des_replay"], g["act_replay"] = des, act
        parts.append(g)
    h = pd.concat(parts, ignore_index=True)
    h["alat"] = h.kappa.abs() * h.v ** 2                                     # requested lateral accel at the simulator's speed
    h["vmax_lim"] = 3.0 / np.maximum(h.v, 1.0) ** 2                          # lateral-accel curvature limit
    print("replay vs logged des (spec): max |diff|", float((h[h.preset == "spec"].des - h[h.preset == "spec"].des_replay).abs().max()))
    return h


def clip_tables(h):
    s = h[(h.preset == "spec")].copy()
    s["clipped"] = (s.act - s.des).abs() > 1e-6
    s["clip_replay"] = (s.act_replay - s.des_replay).abs() > 1e-6
    rows = []
    for grp, m in (("all 64", s.turning | ~s.turning), ("23 turning", s.turning), ("41 straight", ~s.turning)):
        for arm, g in s[m].groupby("arm"):
            runs = g.groupby("scenario")
            rows.append(dict(group=grp, arm=arm, runs=g.scenario.nunique(), steps=len(g), clip_step_pct=100 * g.clipped.mean(), replay_clip_step_pct=100 * g.clip_replay.mean(),
                             runs_with_clip=int(runs.clipped.any().sum()), max_abs_kappa=g.kappa.abs().max(), p99_abs_kappa=g.kappa.abs().quantile(.99),
                             p99_alat=g.alat.quantile(.99), max_alat=g.alat.max(), max_alat_model_clock=1.5625 * g.alat.max(),
                             max_frac_of_alat_limit=(g.kappa.abs() / g.vmax_lim).max(), steps_over_limit_at_model_clock=int((g.alat * 1.5625 > 3).sum()), max_abs_act_minus_des=(g.act - g.des).abs().max()))
    ex = h[h.preset == "exam"].copy()
    ex["clip_replay"] = (ex.act_replay - ex.des_replay).abs() > 1e-6
    for grp, m in (("all 64", ex.turning | ~ex.turning), ("23 turning", ex.turning)):
        for arm, g in ex[m].groupby("arm"):
            rows.append(dict(group=grp + " (exam, replay only)", arm=arm, runs=g.scenario.nunique(), steps=len(g), clip_step_pct=np.nan, replay_clip_step_pct=100 * g.clip_replay.mean(),
                             runs_with_clip=int(g.groupby("scenario").clip_replay.any().sum()), max_abs_kappa=g.kappa.abs().max(), p99_abs_kappa=g.kappa.abs().quantile(.99),
                             p99_alat=g.alat.quantile(.99), max_alat=g.alat.max(), max_alat_model_clock=1.5625 * g.alat.max(),
                             max_frac_of_alat_limit=(g.kappa.abs() / g.vmax_lim).max(), steps_over_limit_at_model_clock=int((g.alat * 1.5625 > 3).sum()), max_abs_act_minus_des=np.nan))
    return pd.DataFrame(rows)


def end_tables(h):
    s = h[(h.preset == "spec") & h.turning]
    r = s.groupby(["arm", "scenario"]).agg(end=("end", "first"), n=("step", "size"), any_clip=("des", lambda x: False)).reset_index()
    cl = s.assign(c=(s.act - s.des).abs() > 1e-6).groupby(["arm", "scenario"]).c.sum().reset_index(name="clips")
    r = r.drop(columns="any_clip").merge(cl, on=["arm", "scenario"])
    last = s.groupby(["arm", "scenario"]).tail(8).assign(c=lambda g: (g.act - g.des).abs() > 1e-6).groupby(["arm", "scenario"]).c.sum().reset_index(name="clips_last8")
    r = r.merge(last, on=["arm", "scenario"])
    return r


def curve_gain(h):
    rows = []
    for (pre, arm), g in h[h.turning].groupby(["preset", "arm"]):
        c = g[(g.kr.abs() > 0.02) & g.kach.notna() & (g.dev < 3)]
        rows.append(dict(preset=pre, arm=arm, curve_steps=len(c), runs=c.scenario.nunique(), median_v=c.v.median(), frac_v_lt_2=(c.v < 2).mean(),
                         slope_requested=slope0(c.kr.values, c.kappa.values), slope_achieved=slope0(c.kr.values, c.kach.values),
                         slope_requested_v_gt_4=slope0(c[c.v > 4].kr.values, c[c.v > 4].kappa.values),
                         max_route_alat=(c.kr.abs() * c.v ** 2).max(), p90_route_alat=(c.kr.abs() * c.v ** 2).quantile(.9)))
    return pd.DataFrame(rows)


def fig_closed(h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = h[(h.preset == "spec")]
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9))
    for arm, c in (("P2-F-s0", COL["P2"]), ("P2H10-F-s0", COL["P2H"])):
        g = s[(s.arm == arm) & s.turning]
        ax[0].hist(np.log10(np.maximum(g.kappa.abs() / g.vmax_lim, 1e-4)), bins=60, histtype="step", color=c, label=LABELS["P2" if arm.startswith("P2-") else "P2H"] + " s0")
    ax[0].axvline(0, color="r", ls=":"); ax[0].set_xlabel("log10 (requested curvature / lateral-accel curvature limit 3/max(v,1)^2)"); ax[0].set_ylabel("steps (23 turning scenarios)")
    ax[0].set_title("requested curvature vs the lateral-accel limit (1 = clip)", fontsize=9); ax[0].legend(fontsize=8)
    g = s[s.turning & (s.arm.isin(["P2-F-s0", "P2-F-s1"]))]
    a = ax[1]
    a.scatter(g.v, g.alat, s=3, alpha=.3, color=COL["P2"])
    g2 = s[s.turning & (s.arm.isin(["P2H10-F-s0", "P2H10-F-s1"]))]
    a.scatter(g2.v, g2.alat, s=3, alpha=.3, color=COL["P2H"])
    a.axhline(3, color="r", ls=":"); a.set_xlabel("simulator speed (m/s)"); a.set_ylabel("requested lateral accel |kappa| v^2 (m/s^2)"); a.set_title("requested lateral accel (orange P2, purple hinge)", fontsize=9)
    a.set_ylim(0, 3.4)
    c = h[(h.turning) & (h.kr.abs() > 0.02) & (h.dev < 3) & (h.preset == "spec")]
    ax[2].scatter(c.kr, c.kappa, s=4, alpha=.4, color=COL["P2"]); ax[2].plot([-.1, .1], [-.1, .1], "k--", lw=.8)
    ax[2].set_xlabel("route curvature at the nearest route point (1/m)"); ax[2].set_ylabel("model requested curvature (1/m)"); ax[2].set_title("requested vs route curvature, curve steps (spec, P2 and hinge)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_closed_loop")


def fig_speed(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = d[d.mag >= 20]
    q = pd.qcut(t.v0, 4)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for m, ks in (("P0", ["P0"]), ("P2", ["P2s0", "P2s1"]), ("P2H", ["P2Hs0", "P2Hs1"]), ("WA", ["WA"])):
        pred = np.mean([t[f"{k}_yaw4"].values * t.sg.values for k in ks], 0)
        x = t.L_yaw4.values * t.sg.values
        xs, ys = [], []
        for lv in q.cat.categories:
            mm = (q == lv).values
            xs.append(t.v0[mm].median()); ys.append(pred[mm].sum() / x[mm].sum())
        ax[0].plot(xs, ys, "-o", color=COL[m], label=LABELS[m])
        ed = [0, 1.5, 2.5, 4]
        b = pd.cut(t.L_apk, ed)
        xs, ys = [], []
        for lv in b.cat.categories:
            mm = (b == lv).values
            if mm.sum() > 30:
                xs.append(t.L_apk[mm].median()); ys.append(pred[mm].sum() / x[mm].sum())
        ax[1].plot(xs, ys, "-o", color=COL[m], label=LABELS[m])
    ax[0].set_xlabel("ego speed at t0 (m/s), quartiles of turns > 20 deg"); ax[0].set_ylabel("sum predicted / sum logged heading change"); ax[0].axhline(1, color="k", ls="--", lw=.7)
    ax[1].set_xlabel("logged peak lateral acceleration (m/s^2)"); ax[1].axhline(1, color="k", ls="--", lw=.7); ax[0].legend(fontsize=8)
    ax[0].set_title("gain vs speed", fontsize=9); ax[1].set_title("gain vs how hard the logged turn is", fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_speed")


def report(a):
    d = load_nav()
    RES.mkdir(parents=True, exist_ok=True)
    T = RES / "tables"
    T.mkdir(exist_ok=True)
    g = gain_tables(d); g.to_csv(T / "gain.csv", index=False)
    sp = speed_tables(d); sp.to_csv(T / "speed.csv", index=False)
    rs = resid_tables(d); rs.to_csv(T / "resid.csv", index=False)
    fig_gain(d, ""); fig_ceiling(d); fig_speed(d)
    q = {}
    for m in ("L", "P0", "P2s0", "P2s1", "P2Hs0", "P2Hs1", "WA"):
        q[m] = {v: np.nanquantile(np.abs(d[f"{m}_{v}"]) if v != "apk" and v != "kpk" else d[f"{m}_{v}"], [.5, .9, .99, .999, 1]).round(4).tolist() for v in ("yaw4", "y4", "k4", "kpk", "apk")}
    json.dump(q, open(T / "quantiles.json", "w"), indent=1)
    h = load_hs()
    ct = clip_tables(h); ct.to_csv(T / "clip.csv", index=False)
    et = end_tables(h); et.to_csv(T / "ends.csv", index=False)
    cg = curve_gain(h); cg.to_csv(T / "curve_gain.csv", index=False)
    fig_closed(h)
    print(f"{len(d)} navtest tokens; tables in {T}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["navtest", "hugsim", "report"])
    ap.add_argument("--out")
    a = ap.parse_args()
    {"navtest": cmd_navtest, "hugsim": cmd_hugsim, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
