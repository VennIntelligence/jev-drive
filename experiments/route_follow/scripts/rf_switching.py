#!/usr/bin/env python3
"""PLAN SWITCHING (plans/2026-10-15-plan-switching-prereg.md; decision 307 M4). Does the served plan switch between a full-turn mode and a
straight / partial mode across consecutive decisions in failing turns, and does the car execute the time average of the sequence?
CPU only; reads the RECEDING extracts (rf_receding.py extract-pai / extract-held), writes nothing next to them.

  read  (Mac, .venv)  rf_switching.py read --pai <RECEDING pai.pkl> --held <RECEDING held.pkl> --tax NAME=<csv> ... (as rf_receding.sh)
        --receding results/receding/turns.csv --out DIR --fig PNG
        -> DIR/{turns.csv, decisions.csv.gz, switching.json, switching.md}; the figure.

Per window decision (s_e in [entry - 20, entry + 20] m, t0 < t_f), sign sigma = +1 left / -1 right, lateral + = toward the inside:
  r  = plan 4 s heading change / the route's heading change over the plan's own arc (route change >= 15 deg), d307 Amendment 3;
  j  = signed perpendicular offset of the plan's 2.0 s point from the previous decision's plan carried into the current anchor frame (m);
  e  = signed perpendicular offset of the car (anchor) from the plan served 1.0 s earlier (m);
  human "plan" (RECEDING Log.plan_at_station) -> r_h, j_h (sanity).
Per turn: switching rate (pairs with one r >= 0.6 and the other <= 0.3), flip rate (F <-> S per s), median |j|, median |e|, mean r,
p90 r, executed ratio R_exec = ego yaw change / route heading change over the window.
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
sys.path[:0] = [str(HERE)]
import rf_pai_follow as F  # noqa: E402
import rf_receding as RC  # noqa: E402

WIN_M, H_MIN, R_HI, R_LO, T_PLAN, T_TRACK, TOL_US, MIN_PAIRS, N_BOOT = 20.0, 15.0, 0.6, 0.3, 3, 1_000_000, 20_000, 3, 10_000
PERIOD = {"pai": 100_000, "held": 500_000}
PRIMARY = "pai_RSG-L18-s0"
CLAIM_POPS = ("pai_RSG-L18-s0", "pai_RV2a-L18-s0", "pai_RV2b-L18-s1", "pai_APP1X-L18-s0", "held_RSG")
CLASSES = ("late_wide", "straight", "over_turn", "no_zero", "other")
CTRL_M = (40.0, 80.0)
XO = {"pai_RSG-L18-s0": "x_pai_RSG-L18-s0", "pai_all4": "x_pai_", "held_RSG": "x_held_RSG-L18-", "held_all6": "x_held_"}   # d299 over-turn zeros


# ---------------------------------------------------------------- geometry
def path_of(A, P):
    """Plan (8, 3) in the anchor A's frame -> world polyline (9, 2) with the anchor as its origin."""
    return np.r_[np.asarray(A[:2], float)[None], RC.to_world(A, np.asarray(P, float)[:, :2])]


def offset(Q, p):
    """Signed perpendicular offset (left +) of point p from polyline Q; the last segment is extended past its end. NaN if degenerate."""
    a, b = Q[:-1], Q[1:]
    d = b - a
    L2 = (d ** 2).sum(1)
    ok = L2 > 1e-6
    if not ok.any():
        return np.nan
    a, d, L2 = a[ok], d[ok], L2[ok]
    t = ((p - a) * d).sum(1) / L2
    hi = np.ones(len(t))
    hi[-1] = np.inf
    t = np.clip(t, 0.0, hi)
    f = a + t[:, None] * d
    i = int(np.argmin(((p - f) ** 2).sum(1)))
    v = p - f[i]
    return float(np.sign(d[i, 0] * v[1] - d[i, 1] * v[0]) * np.hypot(*v))


# ---------------------------------------------------------------- per turn
def turn_switching(rec, tax):
    tr, dd = RC.read_turn(rec, tax)
    if tr is None:
        return None, []
    pop = "pai" if rec["src"].startswith("pai") else "held"
    per = PERIOD[pop]
    R = rec["R"]
    S = F.stations(R)
    sh, hh = F.heading(R, S)
    sg = 1.0 if rec["row"]["side"] == "L" else -1.0
    ent = float(rec["row"]["entry"])
    Pl, C = rec["plans"], rec["ctrl"]
    LG = RC.Log(rec["log"], R, S)
    t0s = Pl["t0"].astype(np.int64)
    yaw_c = np.unwrap(C["yaw"])
    route = lambda s: float(np.interp(min(s, sh[-1]), sh, hh))  # noqa: E731
    rows = []
    for i, d in enumerate(dd):
        if not (ent - WIN_M <= d["s_e"] <= ent + WIN_M and d["pre"]):
            continue
        A = Pl["anchor"][i]
        ra = sg * np.degrees(route(d["s_e"] + d["a4"]) - route(d["s_e"]))
        o = dict(i=i, t0=int(t0s[i]), s_e=d["s_e"], d_e=d["d_e"], v=d["v"], dpsi=d["dpsi"], a4=d["a4"], route_arc=ra,
                 r=d["dpsi"] / ra if ra >= H_MIN else np.nan, j=np.nan, e=np.nan, r_h=np.nan, j_h=np.nan,
                 frame_xy=float(np.hypot(np.interp(t0s[i], C["t"], C["x"]) - A[0], np.interp(t0s[i], C["t"], C["y"]) - A[1])),
                 frame_yaw=float(np.degrees(abs(RC.wrap(np.interp(t0s[i], C["t"], yaw_c) - A[2])))),
                 yaw_ego=float(np.interp(t0s[i], C["t"], yaw_c)))
        psi = sg * np.degrees(np.unwrap(np.r_[0.0, Pl["poses"][i][:, 2]]))     # Amendment 1c: planned vs executed heading at 1 / 2 s
        for T_, ip in ((1, 2), (2, 4)):
            te = t0s[i] + T_ * 1_000_000
            ok = te < min(rec["tz"], C["t"][-1])
            o[f"hp{T_}"] = float(psi[ip]) if ok else np.nan
            o[f"hx{T_}"] = float(sg * np.degrees(np.interp(te, C["t"], yaw_c) - o["yaw_ego"])) if ok else np.nan
        # tracking error: the plan served 1.0 s earlier, the car at t0
        k = int(np.searchsorted(t0s, t0s[i] - T_TRACK))
        for kk in (k - 1, k):
            if 0 <= kk < len(t0s) and abs(t0s[kk] - (t0s[i] - T_TRACK)) <= TOL_US:
                o["e"] = sg * offset(path_of(Pl["anchor"][kk], Pl["poses"][kk]), np.asarray(A[:2], float))
                break
        # human log as a plan at the same station
        H, H0 = LG.plan_at_station(d["s_e"])
        if H is not None:
            so, _, hdp, ha4 = RC.onset(H, sg)
            rah = sg * np.degrees(route(d["s_e"] + ha4) - route(d["s_e"]))
            o["r_h"] = hdp / rah if rah >= H_MIN else np.nan
            o["_H"] = path_of(H0, H)
        rows.append(o)
    # consecutive pairs
    n_skip = 0
    for a, b in zip(rows[:-1], rows[1:]):
        if abs(b["t0"] - a["t0"] - per) > TOL_US:
            n_skip += 1
            continue
        b["pair"] = True
        q = RC.to_world(Pl["anchor"][b["i"]], Pl["poses"][b["i"]][T_PLAN, :2])[0]
        b["j"] = sg * offset(path_of(Pl["anchor"][a["i"]], Pl["poses"][a["i"]]), q)
        if "_H" in a and "_H" in b:
            b["j_h"] = sg * offset(a["_H"], b["_H"][1 + T_PLAN])
        b["r_prev"], b["r_h_prev"] = a["r"], a["r_h"]
        b["ds"] = b["s_e"] - a["s_e"]
    for o in rows:
        o.pop("_H", None)
        o.setdefault("pair", False)
        o.setdefault("r_prev", np.nan)
        o.setdefault("r_h_prev", np.nan)
        o.setdefault("ds", np.nan)
    tr.update(popn=pop, n_win=len(rows), n_pair_skip=n_skip, **stats(rows, per, "r", "j"), **{f"h_{k}": v for k, v in stats(rows, per, "r_h", "j_h").items()
                                                                                               if k in ("sw_rate", "n_sw_pairs", "flip_rate", "abs_j", "j_signed")})
    e = np.array([o["e"] for o in rows], float)
    e = e[np.isfinite(e)]
    tr.update(abs_e=float(np.median(np.abs(e))) if len(e) >= MIN_PAIRS else np.nan, e_signed=float(np.median(e)) if len(e) >= MIN_PAIRS else np.nan, n_e=int(len(e)))
    r = np.array([o["r"] for o in rows], float)
    rf = r[np.isfinite(r)]
    tr.update(r_mean=float(rf.mean()) if len(rf) >= MIN_PAIRS else np.nan, r_p90=float(np.percentile(rf, 90)) if len(rf) >= MIN_PAIRS else np.nan,
              n_r=int(len(rf)), share_r_gt2=float((rf > 2).mean()) if len(rf) else np.nan)
    if len(rows) >= 2:
        dr = sg * np.degrees(route(rows[-1]["s_e"]) - route(rows[0]["s_e"]))
        dy = sg * np.degrees(rows[-1]["yaw_ego"] - rows[0]["yaw_ego"])
        tr.update(route_win=dr, yaw_win=dy, R_exec=dy / dr if dr >= H_MIN else np.nan)
    else:
        tr.update(route_win=np.nan, yaw_win=np.nan, R_exec=np.nan)
    tr.update(creep(rows), **{f"x{T_}": med([o[f"hx{T_}"] - o[f"hp{T_}"] for o in rows]) for T_ in (1, 2)},
              **{f"hp{T_}_med": med([o[f"hp{T_}"] for o in rows]) for T_ in (1, 2)}, **{f"hx{T_}_med": med([o[f"hx{T_}"] for o in rows]) for T_ in (1, 2)})
    ci = [i for i, d in enumerate(dd) if CTRL_M[0] <= d["d_e"] <= CTRL_M[1] and d["pre"]]
    cj = []
    for ia, ib in zip(ci[:-1], ci[1:]):
        if ib == ia + 1 and abs(t0s[ib] - t0s[ia] - per) <= TOL_US:
            q = RC.to_world(Pl["anchor"][ib], Pl["poses"][ib][T_PLAN, :2])[0]
            cj.append(dict(pair=True, j=sg * offset(path_of(Pl["anchor"][ia], Pl["poses"][ia]), q), ds=dd[ib]["s_e"] - dd[ia]["s_e"]))
    tr.update({f"ctrl_{k}": v for k, v in creep(cj).items()})
    tr["frame_xy"] = float(np.median([o["frame_xy"] for o in rows])) if rows else np.nan
    tr["frame_yaw"] = float(np.median([o["frame_yaw"] for o in rows])) if rows else np.nan
    for o in rows:
        o.update(key=None, scene=rec["scene"], cls=tr["cls"], completed=tr["completed"], popn=pop)
    return tr, rows


def med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) >= MIN_PAIRS else np.nan


def creep(rows):
    """Amendment 1a: j per metre of ego travel (median over pairs with ds > 0.05 m) and the summed signed j (m)."""
    p = [o for o in rows if o.get("pair") and np.isfinite(o.get("j", np.nan))]
    jm = [o["j"] / o["ds"] for o in p if o["ds"] > 0.05]
    return dict(jpm=med(jm), jsum=float(np.sum([o["j"] for o in p])) if len(p) >= MIN_PAIRS else np.nan, n_jpm=len(jm))


def stats(rows, per, rk, jk):
    pr = [o for o in rows if o.get("pair")]
    a = np.array([o[f"{rk}_prev"] if rk == "r" else o["r_h_prev"] for o in pr], float)
    b = np.array([o[rk] for o in pr], float)
    ok = np.isfinite(a) & np.isfinite(b)
    sw = (np.minimum(a, b) <= R_LO) & (np.maximum(a, b) >= R_HI)
    j = np.array([o[jk] for o in pr], float)
    j = j[np.isfinite(j)]
    # flip rate: F / S states with partial dropped, changes per second of r-defined window time
    r = np.array([o[rk] for o in rows], float)
    st = np.where(r >= R_HI, 1, np.where(r <= R_LO, 0, -1))[np.isfinite(r)]
    st = st[st >= 0]
    n_def = int(np.isfinite(r).sum())
    return dict(sw_rate=float(sw[ok].mean()) if ok.sum() >= MIN_PAIRS else np.nan, n_sw_pairs=int(ok.sum()),
                flip_rate=float((np.diff(st) != 0).sum() / (n_def * per / 1e6)) if n_def >= MIN_PAIRS else np.nan,
                abs_j=float(np.median(np.abs(j))) if len(j) >= MIN_PAIRS else np.nan, j_signed=float(np.median(j)) if len(j) >= MIN_PAIRS else np.nan,
                p90_j=float(np.percentile(np.abs(j), 90)) if len(j) >= MIN_PAIRS else np.nan, n_j=int(len(j)))


# ---------------------------------------------------------------- statistics
def ratio_boot(v, failed, cl, how, n_boot=N_BOOT, seed=0):
    """stat(v[failed]) / stat(v[completed]) with a cluster bootstrap; how = 'mean' | 'median'."""
    v, failed, cl = np.asarray(v, float), np.asarray(failed, bool), np.asarray(cl)
    m = np.isfinite(v)
    v, failed, cl = v[m], failed[m], cl[m]
    fn = np.mean if how == "mean" else np.median
    res = dict(n_f=int(failed.sum()), n_c=int((~failed).sum()))
    if not failed.any() or failed.all():
        return res | dict(f=np.nan, c=np.nan, ratio=np.nan, lo=np.nan, hi=np.nan, dropped=0, inf=0)

    def rt(a, b):
        return a / b if b > 0 else (np.inf if a > 0 else np.nan)
    f, c = float(fn(v[failed])), float(fn(v[~failed]))
    u = np.unique(cl)
    idx = {k: np.flatnonzero(cl == k) for k in u}
    rng = np.random.default_rng(seed)
    out, drop = [], 0
    for _ in range(n_boot):
        r = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        a, b = v[r][failed[r]], v[r][~failed[r]]
        x = rt(fn(a), fn(b)) if len(a) and len(b) else np.nan
        if np.isnan(x):
            drop += 1
            continue
        out.append(x)
    out = np.array(out)
    q = np.percentile(out, [2.5, 97.5], method="inverted_cdf") if len(out) else [np.nan, np.nan]
    return res | dict(f=f, c=c, ratio=rt(f, c), lo=float(q[0]), hi=float(q[1]), dropped=drop, inf=int(np.isinf(out).sum()))


def diff_boot(v, failed, cl, n_boot=N_BOOT, seed=0):
    v, failed, cl = np.asarray(v, float), np.asarray(failed, bool), np.asarray(cl)
    m = np.isfinite(v)
    v, failed, cl = v[m], failed[m], cl[m]
    if not failed.any() or failed.all():
        return dict(n_f=int(failed.sum()), n_c=int((~failed).sum()), f=np.nan, c=np.nan, d=np.nan, lo=np.nan, hi=np.nan)
    u = np.unique(cl)
    idx = {k: np.flatnonzero(cl == k) for k in u}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        r = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        a, b = v[r][failed[r]], v[r][~failed[r]]
        if len(a) and len(b):
            out.append(np.median(a) - np.median(b))
    f, c = float(np.median(v[failed])), float(np.median(v[~failed]))
    return dict(n_f=int(failed.sum()), n_c=int((~failed).sum()), f=f, c=c, d=f - c, lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)))


def verdict(b):
    if b["n_f"] < 3 or b["n_c"] < 3 or not np.isfinite(b["f"]):
        return "inconclusive (group < 3)"
    if b["ratio"] >= 2 and b["lo"] > 1:
        return "supported"
    if b["hi"] < 2 or b["ratio"] <= 1:
        return "refuted"
    return "inconclusive"


def fit(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return dict(n=int(m.sum()))
    sl, ic = np.polyfit(x[m], y[m], 1)
    r2 = 1 - ((y[m] - (sl * x[m] + ic)) ** 2).sum() / max(((y[m] - y[m].mean()) ** 2).sum(), 1e-12)
    return dict(n=int(m.sum()), slope=float(sl), icpt=float(ic), r2=float(r2), mad=float(np.median(np.abs(y[m] - x[m]))))


def s3_reading(a, b):
    if a.get("n", 0) < 4:
        return "too few turns"
    if a["mad"] < b["mad"] and 0.6 <= a["slope"] <= 1.4:
        return "averages the sequence"
    if b["mad"] < a["mad"] and 0.6 <= b["slope"] <= 1.4:
        return "follows the committed plans"
    return "neither"


def read(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    taxes = {}
    for spec in a.tax or []:
        n, f = spec.split("=", 1)
        x = pd.read_csv(f)
        taxes[n] = dict(zip(x.scene, x.primary))
    recs = []
    for f in (a.pai, a.held):
        if f:
            recs += pickle.load(open(f, "rb"))
    TR, DR = [], []
    for r in recs:
        key = ("x_" if r["src"].endswith("_x") else "") + ("pai_" if r["src"].startswith("pai") else "held_") + r["set"]
        tr, rows = turn_switching(r, taxes.get(key))
        if tr is None:
            continue
        tr["key"] = key
        for o in rows:
            o["key"] = key
        TR.append(tr)
        DR += rows
    T, D = pd.DataFrame(TR), pd.DataFrame(DR)
    T["failed"] = ~T.completed.astype(bool)
    # class cross-check against RECEDING
    if a.receding:
        Rt = pd.read_csv(a.receding)[["key", "scene", "cls"]].rename(columns={"cls": "cls_rec"})
        m = T.merge(Rt, on=["key", "scene"], how="left")
        cls_agree = float((m.cls == m.cls_rec).mean())
    else:
        cls_agree = np.nan
    T.to_csv(out / "turns.csv", index=False)
    D.to_csv(out / "decisions.csv.gz", index=False, float_format="%.4g")
    pops = {k: T[T.key == k] for k in sorted(T.key.unique()) if not k.startswith("x_")}
    pops["held_RSG"] = T[T.key.isin(["held_RSG-L18-s0", "held_RSG-L18-s1"])]
    pops["held_all6"] = T[T.src == "held"]
    pops["pai_all4"] = T[T.src == "pai"]
    res = dict(gates={}, claims={}, classes={}, s3={}, n={}, cls_agree_receding=cls_agree)
    for k, x in pops.items():
        res["gates"][k] = dict(frame_xy=float(x.frame_xy.median()), frame_yaw=float(x.frame_yaw.median()))
    P0 = pops[PRIMARY]
    hs, hj = float(P0.h_sw_rate.mean()), float(P0.h_abs_j.median())
    res["gates"]["human"] = dict(mean_sw=hs, median_abs_j=hj, n_sw=int(P0.h_sw_rate.notna().sum()), n_j=int(P0.h_abs_j.notna().sum()),
                                 passed=bool(hs <= 0.02 and hj <= 0.10))
    res["gates"]["frame_passed"] = bool(all(v["frame_xy"] <= 0.3 and v["frame_yaw"] <= 1.0 for kk, v in res["gates"].items() if isinstance(v, dict) and "frame_xy" in v))
    for k, x in pops.items():
        cl = x.cluster.to_numpy()
        s1 = ratio_boot(x.sw_rate, x.failed, cl, "mean")
        s2 = ratio_boot(x.abs_j, x.failed, cl, "median")
        fl = ratio_boot(x.flip_rate, x.failed, cl, "mean")
        e = diff_boot(x.abs_e, x.failed, cl)
        es = diff_boot(x.e_signed, x.failed, cl)
        js = diff_boot(x.j_signed, x.failed, cl)
        c = dict(turns=int(len(x)), failed=int(x.failed.sum()), completed=int((~x.failed).sum()), S1=s1, S2=s2, flip=fl, abs_e=e, e_signed=es,
                 j_signed=js, p90_j={g: float(x[x.failed == (g == "failed")].p90_j.median()) for g in ("failed", "completed")},
                 human_sw={g: float(x[x.failed == (g == "failed")].h_sw_rate.mean()) for g in ("failed", "completed")},
                 human_abs_j={g: float(x[x.failed == (g == "failed")].h_abs_j.median()) for g in ("failed", "completed")},
                 pair_skips=int(x.n_pair_skip.sum()), no_rate=int(x.sw_rate.isna().sum()), no_j=int(x.abs_j.isna().sum()),
                 jpm=diff_boot(x.jpm, x.failed, cl), jsum=diff_boot(x.jsum, x.failed, cl), x1=diff_boot(x.x1, x.failed, cl), x2=diff_boot(x.x2, x.failed, cl),
                 ctrl_jpm={g: float(x[x.failed == (g == "failed")].ctrl_jpm.median()) for g in ("failed", "completed")},
                 ctrl_n={g: int(x[x.failed == (g == "failed")].ctrl_jpm.notna().sum()) for g in ("failed", "completed")},
                 hp={g: [float(x[x.failed == (g == "failed")][f"hp{t}_med"].median()) for t in (1, 2)] for g in ("failed", "completed")},
                 hx={g: [float(x[x.failed == (g == "failed")][f"hx{t}_med"].median()) for t in (1, 2)] for g in ("failed", "completed")},
                 v={g: float(x[x.failed == (g == "failed")].v_app_med.median()) for g in ("failed", "completed")})
        if k in CLAIM_POPS:
            c["S1"]["verdict"], c["S2"]["verdict"] = verdict(s1), verdict(s2)
            if np.isinf(s1["ratio"]) and s1["f"] < 0.01:                   # Amendment 1: completed rate 0, failed ~0
                c["S1"]["verdict"] += " (degenerate: completed 0, failed < 0.01; no switching)"
            elif s1["f"] == 0 and s1["c"] == 0:
                c["S1"]["verdict"] += " (0 / 0: no switching in either group)"
            if not res["gates"]["human"]["passed"]:
                c["S1"]["verdict"] = c["S2"]["verdict"] = "inconclusive (G-human failed)"
        res["claims"][k] = c
    # classes vs completed
    for k in ("pai_RSG-L18-s0", "pai_all4", "held_RSG", "held_all6"):
        x = pops[k]
        xo = T[T.key.str.startswith(XO[k])]
        res["classes"][k] = {}
        for cname in CLASSES:
            y = pd.concat([x[x.completed], (xo if cname == "over_turn" else x[x.cls == cname])])
            y = y.assign(failed=~y.completed.astype(bool))
            if not y.failed.any():
                continue
            cl = y.cluster.to_numpy()
            res["classes"][k][cname] = dict(n=int(y.failed.sum()), sw=ratio_boot(y.sw_rate, y.failed, cl, "mean"), abs_j=ratio_boot(y.abs_j, y.failed, cl, "median"),
                                            flip=ratio_boot(y.flip_rate, y.failed, cl, "mean"), abs_e=diff_boot(y.abs_e, y.failed, cl),
                                            e_signed=float(y[y.failed].e_signed.median()), j_signed=float(y[y.failed].j_signed.median()),
                                            r_mean=float(y[y.failed].r_mean.median()), R_exec=float(y[y.failed].R_exec.median()),
                                            jpm=diff_boot(y.jpm, y.failed, cl), jsum=diff_boot(y.jsum, y.failed, cl), x2=float(y[y.failed].x2.median()))
    # S3: executed ratio vs time-average / p90 of r
    for k in ("pai_RSG-L18-s0", "pai_all4", "held_RSG", "held_all6"):
        x = pops[k]
        res["s3"][k] = {}
        for g, y in (("all", x), ("failed", x[x.failed]), ("completed", x[~x.failed])):
            fa, fm = fit(y.R_exec, y.r_mean), fit(y.R_exec, y.r_p90)
            res["s3"][k][g] = dict(mean=fa, p90=fm, reading=s3_reading(fa, fm) if g != "completed" else "-",
                                   med=dict(R_exec=float(y.R_exec.median()), r_mean=float(y.r_mean.median()), r_p90=float(y.r_p90.median())))
    res["m4"] = {}
    for k in ("pai_RSG-L18-s0", "pai_RV2a-L18-s0", "pai_RV2b-L18-s1", "pai_APP1X-L18-s0", "pai_all4", "held_RSG"):
        keys = pops[k].key.unique()
        m = D[D.key.isin(keys) & (D.d_e >= 0) & (D.d_e < 10) & np.isfinite(D.r) & ~D.completed.astype(bool)]
        if not len(m):
            continue
        st = np.select([m.r < 0.25, m.r < 0.6], ["S", "P"], "F")
        sets = pd.Series(st, index=m.index).groupby([m.key, m.scene]).agg(lambda q: "".join(sorted(set(q))))
        mu = m.groupby(["key", "scene"]).r.transform("mean")
        res["m4"][k] = dict(decisions=int(len(m)), turns=int(len(sets)), shares={c: float((st == c).mean()) for c in "SPF"},
                            sets={a_: int(b_) for a_, b_ in sets.value_counts().items()},
                            between=float(1 - ((m.r - mu) ** 2).sum() / ((m.r - m.r.mean()) ** 2).sum()))
    res["n"] = dict(decisions=int(len(D)), pairs=int(D.pair.sum()), r_defined=int(np.isfinite(D.r).sum()), e_defined=int(np.isfinite(D.e).sum()))
    (out / "switching.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    write_md(out, res, T)
    if a.fig:
        fig(T, D, a.fig)


def write_md(out, res, T):
    f = lambda v, p=2: "-" if v is None or not np.isfinite(v) else f"{v:+.{p}f}"  # noqa: E731
    u = lambda v, p=2: "inf" if v is not None and np.isinf(v) else ("-" if v is None or not np.isfinite(v) else f"{v:.{p}f}")  # noqa: E731
    g = res["gates"]
    md = ["# PLAN SWITCHING (rf_switching.py)", "",
          "Prereg `plans/2026-10-15-plan-switching-prereg.md`. Window: decisions with the ego in entry - 20 .. entry + 20 m, before the first zero. "
          "r = plan 4 s heading change / route heading change over the plan's own arc (route >= 15 deg); switching rate = share of consecutive "
          "decision pairs with one r >= 0.6 and the other <= 0.3; j = perpendicular offset of the plan's 2 s point from the previous plan carried "
          "forward by the ego motion (+ = inside); e = offset of the car from the plan served 1 s earlier (+ = inside). Ratios failed / completed "
          "(S1: means of the per-turn rate; S2: medians of the per-turn median |j|) with a scene-clustered (held: log) bootstrap 95 % CI, B 10 000. "
          "PAI decides every 0.1 s, held every 0.5 s: pair statistics compare within a population only.", "",
          f"Gates: G-frame {'pass' if g['frame_passed'] else 'FAIL'} (median |anchor - controller| per population "
          + ", ".join(f"{k} {v['frame_xy']:.3f} m / {v['frame_yaw']:.2f} deg" for k, v in g.items() if isinstance(v, dict) and "frame_xy" in v and not k.endswith("all6") and not k.endswith("all4"))
          + f"); G-human (pai_RSG, human log as a plan) mean switching rate {g['human']['mean_sw']:.4f} (n {g['human']['n_sw']}), median |j_h| "
          f"{g['human']['median_abs_j']:.3f} m (n {g['human']['n_j']}): {'pass' if g['human']['passed'] else 'FAIL'}. Classes agree with RECEDING turns.csv "
          f"on {res['cls_agree_receding']:.3f} of the turns. Decisions {res['n']['decisions']}, consecutive pairs {res['n']['pairs']}, r defined "
          f"{res['n']['r_defined']}, e defined {res['n']['e_defined']}.", "",
          "## S1 / S2", "",
          "| population | turns f / c | S1 mean switching rate f / c | ratio [CI] | S1 | S2 median abs j f / c (m) | ratio [CI] | S2 | flip /s f / c | ratio [CI] |",
          "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, c in res["claims"].items():
        s1, s2, fl = c["S1"], c["S2"], c["flip"]
        md.append(f"| {k} | {c['failed']} / {c['completed']} (rate {s1['n_f']} / {s1['n_c']}) | {u(s1['f'], 3)} / {u(s1['c'], 3)} | {u(s1['ratio'])} [{u(s1['lo'])}, {u(s1['hi'])}] | "
                  f"{s1.get('verdict', 'descriptive')} | {u(s2['f'], 3)} / {u(s2['c'], 3)} | {u(s2['ratio'])} [{u(s2['lo'])}, {u(s2['hi'])}] | {s2.get('verdict', 'descriptive')} | "
                  f"{u(fl['f'])} / {u(fl['c'])} | {u(fl['ratio'])} [{u(fl['lo'])}, {u(fl['hi'])}] |")
    md += ["", "Bootstrap bookkeeping (dropped replicates / replicates with ratio +inf) and the human log treated as a plan (mean switching rate, "
           "median abs j_h; f / c):", ""]
    for k, c in res["claims"].items():
        md.append(f"- {k}: S1 dropped {c['S1']['dropped']} inf {c['S1']['inf']}; S2 dropped {c['S2']['dropped']} inf {c['S2']['inf']}; turns without a rate "
                  f"{c['no_rate']}, without j {c['no_j']}; pairs skipped (gap) {c['pair_skips']}; human switching {u(c['human_sw']['failed'], 4)} / "
                  f"{u(c['human_sw']['completed'], 4)}, human abs j {u(c['human_abs_j']['failed'], 3)} / {u(c['human_abs_j']['completed'], 3)} m")
    md += ["", "## S3: tracking error and what the car executes", "",
           "| population | median abs e f / c (m) | diff [CI] | signed e f / c (m) | diff [CI] | signed j f / c (m) | p90 abs j f / c (m) |", "|:--|:--|:--|:--|:--|:--|:--|"]
    for k, c in res["claims"].items():
        e, es, js = c["abs_e"], c["e_signed"], c["j_signed"]
        md.append(f"| {k} | {u(e['f'])} / {u(e['c'])} | {f(e['d'])} [{f(e['lo'])}, {f(e['hi'])}] | {f(es['f'])} / {f(es['c'])} | {f(es['d'])} [{f(es['lo'])}, {f(es['hi'])}] | "
                  f"{f(js['f'], 3)} / {f(js['c'], 3)} | {u(c['p90_j']['failed'])} / {u(c['p90_j']['completed'])} |")
    md += ["", "Executed ratio R_exec (ego yaw change / route heading change over the window) against the time-average of r and against its p90 "
           "(robust max), across turns: OLS slope, intercept, R^2, median abs (R_exec - x); reading per the prereg.", "",
           "| population | turns | median R_exec / mean r / p90 r | vs mean r: slope, icpt, R^2, mad | vs p90 r: slope, icpt, R^2, mad | reading |", "|:--|:--|:--|:--|:--|:--|"]
    for k, v in res["s3"].items():
        for gname, z in v.items():
            fa, fm = z["mean"], z["p90"]
            ff = lambda q: f"{f(q['slope'])}, {f(q['icpt'])}, {u(q['r2'])}, {u(q['mad'])}" if "slope" in q else "-"  # noqa: E731
            md.append(f"| {k} {gname} | {fa['n']} | {u(z['med']['R_exec'])} / {u(z['med']['r_mean'])} / {u(z['med']['r_p90'])} | {ff(fa)} | {ff(fm)} | {z['reading']} |")
    md += ["", "## By failure class (vs the completed turns of the same population; ratio [CI], descriptive)", "",
           "| population | class | n | switching ratio [CI] | abs j ratio [CI] | flip ratio [CI] | abs e f / c (m) | signed e | signed j | median mean r | median R_exec |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, v in res["classes"].items():
        for cname, z in v.items():
            r3 = lambda q: f"{u(q['ratio'])} [{u(q['lo'])}, {u(q['hi'])}] ({u(q['f'], 3)} / {u(q['c'], 3)})"  # noqa: E731
            md.append(f"| {k} | {cname} | {z['n']} | {r3(z['sw'])} | {r3(z['abs_j'])} | {r3(z['flip'])} | {u(z['abs_e']['f'])} / {u(z['abs_e']['c'])} | "
                      f"{f(z['e_signed'])} | {f(z['j_signed'], 3)} | {u(z['r_mean'])} | {u(z['R_exec'])} |")
    md += ["", "## Amendment 1 (post hoc, descriptive)", "",
           "(a) Plan creep: signed j per metre of ego travel (per-turn median; - = each new plan's 2 s point lies outside the previous plan's path) "
           "and the summed signed j over the window (m); median failed / completed, difference [CI]; control = the same per-metre j 40-80 m before "
           "the entry (turns that reach back that far). (c) Executed - planned heading toward the turn over the next 1 s / 2 s (deg, per-turn "
           "median; planned = the plan's own heading at 1 / 2 s, executed = the car's yaw change), with the planned / executed medians.", "",
           "| population | approach v f / c | j per m f / c | diff [CI] | summed j f / c (m) | diff [CI] | control j per m f / c (n) | exec - plan 1 s f / c | exec - plan 2 s f / c | diff 2 s [CI] | plan / exec 2 s f | plan / exec 2 s c |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, c in res["claims"].items():
        a1, a2, x1, x2 = c["jpm"], c["jsum"], c["x1"], c["x2"]
        md.append(f"| {k} | {u(c['v']['failed'], 1)} / {u(c['v']['completed'], 1)} | {f(a1['f'], 3)} / {f(a1['c'], 3)} | {f(a1['d'], 3)} [{f(a1['lo'], 3)}, {f(a1['hi'], 3)}] | "
                  f"{f(a2['f'])} / {f(a2['c'])} | {f(a2['d'])} [{f(a2['lo'])}, {f(a2['hi'])}] | {f(c['ctrl_jpm']['failed'], 3)} / {f(c['ctrl_jpm']['completed'], 3)} "
                  f"({c['ctrl_n']['failed']} / {c['ctrl_n']['completed']}) | {f(x1['f'], 1)} / {f(x1['c'], 1)} | {f(x2['f'], 1)} / {f(x2['c'], 1)} | "
                  f"{f(x2['d'], 1)} [{f(x2['lo'], 1)}, {f(x2['hi'], 1)}] | {f(c['hp']['failed'][1], 1)} / {f(c['hx']['failed'][1], 1)} | "
                  f"{f(c['hp']['completed'][1], 1)} / {f(c['hx']['completed'][1], 1)} |")
    md += ["", "(a) by class (vs completed of the same population): j per m failed / completed, diff [CI]; summed j; exec - plan 2 s (failed median):", "",
           "| population | class | n | j per m | diff [CI] | summed j (m) | diff [CI] | exec - plan 2 s |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, v in res["classes"].items():
        for cname, z in v.items():
            a1, a2 = z["jpm"], z["jsum"]
            md.append(f"| {k} | {cname} | {z['n']} | {f(a1['f'], 3)} / {f(a1['c'], 3)} | {f(a1['d'], 3)} [{f(a1['lo'], 3)}, {f(a1['hi'], 3)}] | {f(a2['f'])} / {f(a2['c'])} | "
                      f"{f(a2['d'])} [{f(a2['lo'])}, {f(a2['hi'])}] | {f(z['x2'], 1)} |")
    md += ["", "(b) M4 composition, failed turns, 0-10 m before the entry (S r < 0.25, P, F r >= 0.6 as d307 M4): decision shares, the set of states "
           "each turn shows, and the share of the variance of r that lies between turns:", ""]
    for k, z in res["m4"].items():
        md.append(f"- {k}: {z['decisions']} decisions, {z['turns']} turns; shares S {z['shares']['S']:.2f} / P {z['shares']['P']:.2f} / F {z['shares']['F']:.2f}; "
                  f"state sets {', '.join(f'{a_} {b_}' for a_, b_ in z['sets'].items())}; between-turn variance share {z['between']:.2f}")
    md += ["", "## Per turn (pai_RSG-L18-s0)", "",
           "| scene | side | class | win dec | sw pairs | switching | flip /s | abs j m | signed j | j per m | summed j | abs e m | signed e | mean r | p90 r | R_exec | exec - plan 2 s | human sw |",
           "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in T[T.key == PRIMARY].sort_values(["completed", "cls", "scene"]).iterrows():
        md.append(f"| {r.scene[7:15]} | {r.side} | {r.cls} | {r.n_win} | {r.n_sw_pairs} | {u(r.sw_rate, 3)} | {u(r.flip_rate)} | {u(r.abs_j, 3)} | {f(r.j_signed, 3)} | "
                  f"{f(r.jpm, 3)} | {f(r.jsum)} | {u(r.abs_e)} | {f(r.e_signed)} | {u(r.r_mean)} | {u(r.r_p90)} | {u(r.R_exec)} | {f(r.x2, 1)} | {u(r.h_sw_rate, 3)} |")
    (out / "switching.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:22]))


def picks(T):
    x = T[(T.key == PRIMARY) & T.sw_rate.notna()].sort_values("scene")
    p = list(x[x.completed].scene[:4])
    for c in ("late_wide", "straight"):
        p += list(x[x.cls == c].scene[:2])
    return p


def fig(T, D, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    sel = picks(T)
    x = T[T.key == PRIMARY].set_index("scene")
    fg, ax = plt.subplots(4, 2, figsize=(13, 12.5), sharex=True, sharey="row")
    groups = (("failed", [s for s in sel if not x.loc[s].completed]), ("completed", [s for s in sel if x.loc[s].completed]))
    for c, (gname, ss) in enumerate(groups):
        for i, s in enumerate(ss):
            r = x.loc[s]
            d = D[(D.key == PRIMARY) & (D.scene == s)].sort_values("t0")
            lab = f"{s[7:15]} {r.cls}: switching {r.sw_rate:.2f}, median |j| {r.abs_j:.2f} m, summed j {r.jsum:+.1f} m"
            ax[0, c].plot(d.d_e, np.clip(d.r, -0.5, 2.0), color=col[i], lw=2, label=lab)
            ax[1, c].plot(d.d_e, d.j, color=col[i], lw=1.5)
            ax[2, c].plot(d.d_e, d.e, color=col[i], lw=2)
            ax[3, c].plot(d.d_e, d.j.fillna(0).cumsum(), color=col[i], lw=2)      # Amendment 1d
        ax[0, c].axhspan(R_HI, 2.0, color="#888", alpha=0.08)
        ax[0, c].axhspan(-0.5, R_LO, color="#888", alpha=0.08)
        ax[0, c].set_title(f"PAI 281 RSG-L18-s0, {gname} turns", fontsize=11)
        ax[3, c].legend(*ax[0, c].get_legend_handles_labels(), fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
        for rr in range(4):
            ax[rr, c].axhline(0, color="#888", lw=0.8)
            ax[rr, c].axvline(0, color="#888", lw=0.8, ls=":")
            ax[rr, c].grid(alpha=0.25)
            ax[rr, c].spines[["top", "right"]].set_visible(False)
        ax[3, c].set_xlabel("ego distance to the turn entry d_e (m; < 0 = past the entry)")
    ax[0, 0].set_xlim(20, -20)
    ax[0, 0].set_ylabel("r = plan / route heading over\nthe plan's arc (shaded: full / straight)")
    ax[1, 0].set_ylabel("j: plan-to-plan jump at 2 s (m)\n+ = toward the inside")
    ax[2, 0].set_ylabel("e: car vs the plan of 1 s ago (m)\n+ = inside, - = wide")
    ax[3, 0].set_ylabel("summed j over the window (m)\n- = plans creep outward")
    fg.tight_layout(rect=(0, 0.03, 1, 1))
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read")
    r.add_argument("--pai", default=""), r.add_argument("--held", default=""), r.add_argument("--tax", nargs="*")
    r.add_argument("--receding", default=""), r.add_argument("--out", required=True), r.add_argument("--fig", default="")
    a = ap.parse_args()
    {"read": read}[a.cmd](a)
