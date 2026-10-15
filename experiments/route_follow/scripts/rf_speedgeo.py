#!/usr/bin/env python3
"""SPEED-GEOMETRY reading (plans/2026-10-16-speed-geometry-prereg.md; research/next-round/turn-night.md section 10). Part A: is the wide
turn a speed effect (the plan's curvature against the human's on the same stretch of road, against the speed ratio; the plan's implied
lateral acceleration)? Part B: does the plan slow for what it sees (plan vs human 2 s speed drop at collision zeros, in scenes where the
human braked for a lead, in the oncoming lane)? CPU only; stored reads are read, never written.

  brake-extract  (box, repo .venv)  rf_speedgeo.py brake-extract --tax <rf_tax PAI pkl> --plans <rf_corridor plans pkl>
                 --taxcsv <pai281_rsg.csv> --laneoff <laneoff turns.csv> --out <dir> [--workers N]
                 -> <dir>/brake_dec.csv.gz (every infer decision: plan / human drops, ego / human leads), <dir>/brake_scenes.csv
  read           (Mac, .venv)  rf_speedgeo.py read --pai <RECEDING pai.pkl> --held <RECEDING held.pkl> --cls <receding turns.csv>
                 --brake <dir> --out <results dir> --fig <png>
                 -> speed_geo.md, speed_geo.json, turns.csv, decisions.csv.gz (Part A), brake_scenes.csv (Part B), the figure.

Definitions: the prereg. Plan / human curvature: chord headings of the 8 pose segments at their mid arcs plus the start heading at arc 0;
kappa_j = |delta heading| / delta arc (segments >= 0.3 m). Human arc-matched = the log path sampled at the plan's own arcs from the point
where the log first reaches the decision's route station. Plan 2 s drop dP = v - mean(segment speeds 4, 5); human dH = v_h(t_s) -
v_h(t_s + 2 s) at the log's first passage t_s of the anchor's log-path station.
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
sys.path[:0] = [str(HERE), str(HERE.parents[2])]

N_BOOT, SEED = 10_000, 0
KH_MIN, V_MIN, SEG_MIN, K_FLOOR, ALAT_GATE = 0.02, 2.0, 0.3, 1e-3, 4.5
LEAD_G, LEAD_PSI, HB_DROP, HB_GAP, TTC_MAX = 40.0, np.radians(60.0), 1.0, 25.0, 3.0
CLS_ORDER = ("completed", "late_wide", "straight", "over_turn", "no_zero", "other", "wrong_exit")
COLL = ("coll_veh_oncoming", "coll_veh_side", "coll_veh_crossing", "coll_veh_ahead", "coll_stationary", "coll_cyclist", "coll_ped",
        "coll_unknown")
LEADQ = ("low_slow_behind_lead", "low_queue", "low_follow_lead", "low_stopped_lead")


def box5(v):
    v = np.asarray(v, float)
    return np.convolve(np.r_[v[:2], v, v[-2:]], np.ones(5) / 5, "valid")


def log_speed(T, P):
    """Log 10 Hz positions -> cumulative arc and 5-sample boxed step speed at T."""
    step = np.hypot(*np.diff(P, axis=0).T)
    arc = np.r_[0.0, np.cumsum(step)]
    v = np.r_[0.0, step / np.maximum(np.diff(T) * 1e-6, 1e-3)]
    v[0] = v[1] if len(v) > 1 else 0.0
    return arc, box5(v)


def curv(X, h0):
    """Points (9, 2) incl. the start, start heading h0 -> kappa (8,), sample speeds (8,) (0.5 s poses), segment ok mask (8,)."""
    d = np.diff(X, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    H = np.unwrap(np.r_[h0, np.arctan2(d[:, 1], d[:, 0])])
    a = np.r_[0.0, np.cumsum(L)]
    m = np.r_[0.0, a[:-1] + L / 2]
    k = np.abs(np.diff(H)) / np.maximum(np.diff(m), 1e-6)
    ok = L >= SEG_MIN
    ok[1:] &= L[:-1] >= SEG_MIN
    v = np.r_[L[0], (L[:-1] + L[1:]) / 2] / 0.5
    return np.where(ok, k, np.nan), v, ok


def peak(x, n):
    x = np.asarray(x, float)[:n]
    return float(np.nanmax(x)) if np.isfinite(x).any() and np.isfinite(x).all() else np.nan


# ---------------------------------------------------------------- Part B extract (box)
def lead(st, path, front, hw, v_subj):
    """Nearest in-path same-direction vehicle / cyclist ahead: (gap m, ttc s, id) or (nan, inf, '')."""
    import rf_tax as TX
    if not len(st["id"]) or len(path) < 2:
        return np.nan, np.inf, ""
    X = np.c_[st["x"], st["y"]]
    _, s, lat, h = TX.poly_proj(path, X)
    best = (np.nan, np.inf, "")
    for n in range(len(X)):
        if TX.cgroup(str(st["cls"][n])) not in ("vehicle", "cyclist"):
            continue
        g = s[n] - front - st["L"][n] / 2
        dpsi = abs(float(TX.wrap(st["yaw"][n] - h[n])))
        if not (-1.0 <= g <= LEAD_G and abs(lat[n]) <= hw + st["W"][n] / 2 + 0.3 and dpsi < LEAD_PSI):
            continue
        if np.isfinite(best[0]) and g >= best[0]:
            continue
        cl = v_subj - float(st["v"][n]) * np.cos(dpsi)
        best = (float(g), float(max(g, 0.0) / cl) if cl > 0.1 else np.inf, str(st["id"][n]))
    return best


def fpath(t, X, Y, Yaw, t0):
    """Path of a 10 Hz track from its interpolated point at t0 over the next 5 s, + 20 m along the last heading."""
    import rf_tax as TX
    m = (t > t0) & (t <= t0 + 5e6)
    P = TX.dedupe(np.vstack([[np.interp(t0, t, X), np.interp(t0, t, Y)], np.c_[X[m], Y[m]]]))
    h = float(np.interp(min(t0 + 5e6, t[-1]), t, Yaw))
    return TX.dedupe(np.vstack([P, P[-1] + 20.0 * np.array([np.cos(h), np.sin(h)])]))


def brake_one(job):
    o, pl, tx, onc = job
    import rf_tax as TX
    if "ctrl" not in o or pl is None:
        return [], dict(scene=o["scene"], rid=o["rid"], ok=False)
    r = TX.Roll(o)
    C = o["ctrl"]
    t, x, y, yaw, vx = C["t"], C["x"], C["y"], C["yaw"], C["vx"]
    L = o["log"]
    T, P, Y = np.asarray(L["T"], float), np.asarray(L["P"], float), np.unwrap(np.asarray(L["Y"], float))
    arc, vh = log_speed(T, P)
    au, iu = np.unique(arc, return_index=True)                              # first passage of each log arc
    LPd = TX.dedupe(P)
    tev = r.t0 + tx["t_ev_s"] * 1e6 if np.isfinite(tx["t_ev_s"]) else np.nan
    tf = tev if (np.isfinite(tev) and float(o["score"] or 0) == 0) else np.inf
    rear, front, hw = o["dims"]
    tg0, tg1 = r.tg[0], r.tg[-1]
    hit = tx.get("hit_id") or ""
    vhs = lambda q: float(np.interp(q, T, vh, left=np.nan, right=np.nan))  # noqa: E731

    def human_at(s):
        if not np.isfinite(s) or s < au[0] or s > au[-1]:
            return np.nan
        return float(np.interp(s, au, T[iu]))

    def hlead(ts):
        if not np.isfinite(ts) or ts < tg0 or ts > tg1:
            return np.nan, np.inf
        g, ttc, _ = lead(r.objs_at(ts), fpath(T, P[:, 0], P[:, 1], Y, ts), front, hw, vhs(ts))
        return g, ttc

    rows = []
    for i in range(len(pl["t0"])):
        t0 = float(pl["t0"][i])
        if t0 < t[0] or t0 > t[-1]:
            continue
        A, Q = pl["anchor"][i], np.asarray(pl["poses"][i], float)
        seg = np.hypot(*np.diff(np.r_[[[0.0, 0.0]], Q[:, :2]], axis=0).T) / 0.5
        v = float(np.interp(t0, t, vx))
        st = r.objs_at(t0) if tg0 <= t0 <= tg1 else dict(id=np.array([]))
        eg, ettc, eid = lead(st, fpath(t, x, y, yaw, t0), front, hw, v) if len(st["id"]) else (np.nan, np.inf, "")
        s_log = float(TX.poly_proj(LPd, A[:2])[1][0]) if len(LPd) > 1 else np.nan
        ts = human_at(s_log)
        hg, httc = hlead(ts)
        rng = np.nan
        if hit and len(st["id"]):
            j = np.flatnonzero(st["id"] == hit)
            if len(j):
                rng = float(np.hypot(st["x"][j[0]] - A[0], st["y"][j[0]] - A[1]))
        vp2 = float((seg[3] + seg[4]) / 2)
        rows.append(dict(scene=o["scene"], rid=o["rid"], t0=t0, t_rel=(t0 - tev) / 1e6 if np.isfinite(tev) else np.nan, pre=bool(t0 < tf),
                         v=v, vp1=float((seg[1] + seg[2]) / 2), vp2=vp2, vp4=float(seg[7]), dP=v - vp2, s_log=s_log, ts=ts, vh=vhs(ts),
                         vh2=vhs(ts + 2e6) if np.isfinite(ts) else np.nan, dH=(vhs(ts) - vhs(ts + 2e6)) if np.isfinite(ts) else np.nan,
                         e_gap=eg, e_ttc=ettc, e_lead=eid, h_gap=hg, h_ttc=httc, hit_rng=rng,
                         v_ex2=float(np.interp(t0, t, vx) - np.interp(t0 + 2e6, t, vx)) if t0 + 2e6 <= t[-1] else np.nan))
    # human brake-for-lead moments on the traffic grid
    hb = []
    for tt in r.tg:
        if tt + 2e6 > T[-1] or tt < T[0]:
            continue
        d = vhs(tt) - vhs(tt + 2e6)
        if d >= HB_DROP and vhs(tt) >= 1.0:
            g, _ = hlead(tt)
            if np.isfinite(g) and g <= HB_GAP:
                hb.append((tt, d, g))
    sc = dict(scene=o["scene"], rid=o["rid"], ok=True, score=float(o["score"] or 0), primary=tx["primary"], t_ev=tev, t_f=tf,
              n_hb=len(hb), hb_t=hb[0][0] if hb else np.nan, hb_drop=max((h[1] for h in hb), default=np.nan),
              onc_t=onc if onc is not None else np.nan, hit_id=hit, v_ev=float(np.interp(tev, t, vx)) if np.isfinite(tev) else np.nan,
              v_ev2=float(np.interp(tev - 2e6, t, vx)) if np.isfinite(tev) else np.nan)
    return rows, sc


def brake_extract(a):
    import pandas as pd
    from jevdrive.common import n_cpus
    from jevdrive.par import pmap
    rolls = pickle.load(open(a.tax, "rb"))["rolls"]
    plans = pickle.load(open(a.plans, "rb"))
    tax = pd.read_csv(a.taxcsv).set_index("rid")
    lo = pd.read_csv(a.laneoff)
    lo = lo[(lo.set == "pai") & (lo.kind == "strict") & (lo.point == "e0") & lo.oncoming.astype(bool) & lo.failed.astype(bool)]
    onc = dict(zip(lo.rid, lo.t))
    jobs = []
    for o in rolls:
        if o["rid"] not in tax.index:
            continue
        tr = tax.loc[o["rid"]]
        jobs.append((o, plans.get(o["rid"]), dict(t_ev_s=float(tr.t_ev_s), primary=str(tr.primary),
                                                   hit_id="" if pd.isna(tr.hit_id) else str(tr.hit_id)), onc.get(o["rid"])))
    res = pmap(brake_one, jobs, workers=a.workers or max(1, n_cpus() // 2), desc="brake")
    res.raise_if_failed()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for v in res.values for r in v[0]]).to_csv(out / "brake_dec.csv.gz", index=False)
    pd.DataFrame([v[1] for v in res.values]).to_csv(out / "brake_scenes.csv", index=False)
    print(len(jobs), "rollouts,", len(onc), "oncoming events ->", out)


# ---------------------------------------------------------------- Part A (Mac)
def geo_turn(rec, cls):
    import rf_pai_follow as F
    import rf_receding as RC
    row = rec["row"]
    if not row.get("turn") or rec.get("plans") is None:
        return []
    R = rec["R"]
    S = F.stations(R)
    C = rec["ctrl"]
    t = C["t"]
    Pc = np.c_[C["x"], C["y"]]
    stc, _ = F.track(R, S, Pc)
    tf = rec["tz"]
    LG = RC.Log(rec["log"], R, S)
    _, vh = log_speed(LG.T, LG.P)
    ent, ext = float(row["entry"]), float(row["exit"])
    Pl = rec["plans"]
    D = []
    for i in range(len(Pl["t0"])):
        t0, A = float(Pl["t0"][i]), Pl["anchor"][i]
        ic = int(np.clip(np.searchsorted(t, t0), 0, len(t) - 1))
        s_e, _ = F.project(R, S, A[:2], stc[ic] - 30.0, stc[ic] + 60.0)
        D.append((i, t0, A, s_e))
    se = np.array([d[3] for d in D])
    reach = np.flatnonzero(se >= ent)
    k_ent = int(reach[0]) if len(reach) else len(D)
    out = []
    for n, (i, t0, A, s_e) in enumerate(D):
        app = bool(0.0 <= ent - s_e <= 40.0 and t0 < tf and n < k_ent)
        trn = bool(ent <= s_e <= ext + 10.0 and t0 < tf)
        if not (app or trn):
            continue
        Q = np.asarray(Pl["poses"][i], float)
        X = np.r_[[[0.0, 0.0]], Q[:, :2]]
        kp, vp, _ = curv(X, 0.0)
        arcs = np.r_[0.0, np.cumsum(np.hypot(*np.diff(X, axis=0).T))]
        d = dict(set=rec["set"], src=rec["src"], scene=rec["scene"], cls=cls, completed=bool(row.get("completed")), k=int(Pl["k"][i]), t0=t0,
                 s_e=s_e, d_e=ent - s_e, win="approach" if app else "turn", v=float(np.interp(t0, t, C["vx"])),
                 kp2=peak(kp, 4), kp4=peak(kp, 8), alat4=peak(kp * vp ** 2, 8), arc2=float(arcs[4]), arc4=float(arcs[8]))
        d.update(vh=np.nan, kh2=np.nan, kh4=np.nan, kht2=np.nan, kht4=np.nan)
        if LG.su[0] <= s_e <= LG.su[-1]:
            ts = float(np.interp(s_e, LG.su, LG.tu))
            q0 = float(np.interp(ts, LG.T, LG.arc))
            d["vh"] = float(np.interp(ts, LG.T, vh))
            if q0 + arcs[8] <= LG.arc[-1]:
                kh, _, _ = curv(LG.xy_arc(q0 + arcs), float(LG.pose(np.array([ts]))[0, 2]))
                d["kh2"], d["kh4"] = peak(kh, 4), peak(kh, 8)
            H, _ = LG.plan_at_station(s_e)
            if H is not None and ts + 4e6 <= LG.T[-1]:
                kt, _, _ = curv(np.r_[[[0.0, 0.0]], H[:, :2]], 0.0)
                d["kht2"], d["kht4"] = peak(kt, 4), peak(kt, 8)
        out.append(d)
    return out


def wls(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    vx = (w * (x - mx) ** 2).sum()
    return float((w * (x - mx) * (y - my)).sum() / vx) if vx > 0 else np.nan


def wls_multi(X, y, w):
    Xa = np.c_[np.ones(len(y)), X]
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(Xa * sw[:, None], y * sw, rcond=None)
    return b[1:]


def cluster_boot(keys, fn, n_boot=N_BOOT, seed=SEED):
    """keys: cluster label per row; fn(row index array) -> float. Percentile 95 % CI over cluster resamples."""
    keys = np.asarray(keys)
    u, inv = np.unique(keys, return_inverse=True)
    idx = [np.flatnonzero(inv == k) for k in range(len(u))]
    rng = np.random.default_rng(seed)
    V = np.array([fn(np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])) for _ in range(n_boot)], float)
    return float(np.nanpercentile(V, 2.5)), float(np.nanpercentile(V, 97.5))


def a1_fit(E, xcol="lq", ycol="lr2", n_boot=N_BOOT):
    """Turn-weighted WLS slope with a turn-cluster CI; A1c (log kappa_plan on log v + log v_h + log kappa_h, coefficient on log v) and
    the within-turn (turn-demeaned) slope."""
    nt = int(E.turn.nunique())
    if nt < 2:
        return dict(n_turns=nt, n_dec=int(len(E)))
    tid = np.unique(E.turn.to_numpy(), return_inverse=True)[1]
    w = 1.0 / np.bincount(tid)[tid]
    x, y = E[xcol].to_numpy(float), E[ycol].to_numpy(float)
    hcol = {"lr2": "lkh2", "lr4": "lkh4", "lrt2": "lkht2"}[ycol]
    kp = E["lkp4" if ycol == "lr4" else "lkp2"].to_numpy(float)
    X3 = E[["lv", "lvh", hcol]].to_numpy(float)
    f = lambda r: wls(x[r], y[r], w[r])  # noqa: E731
    g = lambda r: wls_multi(X3[r], kp[r], w[r])[0]  # noqa: E731

    def within(r):
        t_, ww, xx, yy = tid[r], w[r], x[r], y[r]
        sw = np.bincount(t_, ww, minlength=tid.max() + 1)
        mx = np.bincount(t_, ww * xx, minlength=tid.max() + 1) / np.maximum(sw, 1e-12)
        my = np.bincount(t_, ww * yy, minlength=tid.max() + 1) / np.maximum(sw, 1e-12)
        zx, zy = xx - mx[t_], yy - my[t_]
        den = (ww * zx ** 2).sum()
        return float((ww * zx * zy).sum() / den) if den > 0 else np.nan
    al = np.arange(len(E))
    cl = E.cl.to_numpy()
    lo, hi = cluster_boot(cl, f, n_boot)
    clo, chi = cluster_boot(cl, g, n_boot)
    wlo, whi = cluster_boot(cl, within, n_boot)
    return dict(n_turns=nt, n_dec=int(len(E)), slope=f(al), lo=lo, hi=hi, c_lv=float(g(al)), c_lo=clo, c_hi=chi, within=within(al), w_lo=wlo,
                w_hi=whi)


def verdict_a1(r):
    if r.get("n_turns", 0) < 8 or not np.isfinite(r.get("slope", np.nan)):
        return "inconclusive (< 8 turns)"
    if r["slope"] <= -0.5 and r["hi"] < 0:
        return "supported" + ("" if r["c_hi"] < 0 else " (A1c: not attributable to the ego's speed)")
    if (r["slope"] > -0.5 and r["lo"] > -0.5) or r["slope"] >= 0:
        return "refuted"
    return "inconclusive"


def p90_boot(df, col, n_boot=N_BOOT):
    x = df[np.isfinite(df[col])]
    if not len(x):
        return dict(n=0)
    v = x[col].to_numpy(float)
    lo, hi = cluster_boot(x.cl.to_numpy(), lambda r: np.percentile(v[r], 90), n_boot)
    return dict(n=int(len(x)), turns=int(x.turn.nunique()), p90=float(np.percentile(v, 90)), lo=lo, hi=hi, median=float(np.median(v)))


def part_a(a, res):
    import pandas as pd
    import rf_receding as RC  # noqa: F401  (Log, imported in geo_turn)
    cls = pd.read_csv(a.cls)
    cmap = {(k, s): c for k, s, c in zip(cls.key, cls.scene, cls.cls)}
    clus = {(k, s): c for k, s, c in zip(cls.key, cls.scene, cls.cluster)}
    recs = pickle.load(open(a.pai, "rb")) + pickle.load(open(a.held, "rb"))
    want = {"RSG-L18-s0", "RSG-L18-s1", "APP1X-L18-s0"}
    rows = []
    for r in recs:
        if r["set"] not in want or (r["src"].startswith("held") and r["set"] == "APP1X-L18-s0"):
            continue
        key = ("x_" if r["src"].endswith("_x") else "") + ("pai_" if r["src"].startswith("pai") else "held_") + r["set"]
        if key.startswith("held_APP1X") or key.startswith("x_held_APP1X"):
            continue
        c = cmap.get((key, r["scene"]))
        if c is None:
            continue
        for d in geo_turn(r, c):
            d.update(key=key, turn=f"{key}|{r['scene']}", cl=clus.get((key, r["scene"]), r["scene"]))
            rows.append(d)
    D = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        D["q"] = D.v / D.vh
        for T in ("2", "4"):
            D[f"r{T}"] = np.maximum(D[f"kp{T}"], K_FLOOR) / D[f"kh{T}"]
            D[f"rt{T}"] = np.maximum(D[f"kp{T}"], K_FLOOR) / D[f"kht{T}"]
            D[f"lkp{T}"] = np.log(np.maximum(D[f"kp{T}"], K_FLOOR))
            D[f"lkh{T}"] = np.log(D[f"kh{T}"])
            D[f"lkht{T}"] = np.log(D[f"kht{T}"])
        D["lq"], D["lv"], D["lvh"] = np.log(D.q), np.log(D.v), np.log(D.vh)
        D["lr2"], D["lr4"], D["lrt2"] = np.log(D.r2), np.log(D.r4), np.log(D.rt2)
        D["lkpt2"] = D.lkp2
        D["cf_alat4"] = D.kh4 * D.v ** 2
        D["h_alat4"] = D.kh4 * D.vh ** 2
    D.to_csv(Path(a.out) / "decisions.csv.gz", index=False)
    base = (D.v >= V_MIN) & (D.vh >= V_MIN)
    elig = lambda T, h="kh": base & np.isfinite(D[f"kp{T}"]) & (D[f"{h}{T}"] >= KH_MIN)  # noqa: E731
    sets = {"pai_RSG": D.key == "pai_RSG-L18-s0", "held_RSG": D.key.isin(["held_RSG-L18-s0", "held_RSG-L18-s1"]),
            "pai_APP1X": D.key == "pai_APP1X-L18-s0"}
    A = res["A"] = dict(n_dec={k: int(m.sum()) for k, m in sets.items()}, A1={}, A2={}, A3={}, classes={})
    for nm, m in sets.items():
        A["A1"][nm] = {}
        for lab, (T, h, y) in {"T2": ("2", "kh", "lr2"), "T4": ("4", "kh", "lr4"), "T2_time": ("2", "kht", "lrt2")}.items():
            E = D[m & elig(T, h) & np.isfinite(D[y])]
            r = a1_fit(E, ycol=y)
            r["verdict"] = verdict_a1(r)
            A["A1"][nm][lab] = r
            print("A1", nm, lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    for nm, m in sets.items():
        x = D[m & (D.v >= V_MIN) & np.isfinite(D.alat4)]
        pa = p90_boot(x, "alat4")
        cf = p90_boot(D[m & (D.v >= V_MIN) & (D.v > D.vh) & np.isfinite(D.cf_alat4)], "cf_alat4")
        hu = p90_boot(D[m & (D.vh >= V_MIN) & np.isfinite(D.h_alat4)], "h_alat4")
        v = "inconclusive"
        if pa.get("n") and cf.get("n"):
            if pa["p90"] <= ALAT_GATE and cf["lo"] > ALAT_GATE:
                v = "supported"
            elif pa["lo"] > ALAT_GATE or cf["hi"] <= ALAT_GATE:
                v = "refuted"
        bins = {}
        for lo_, hi_ in ((2, 4), (4, 6), (6, 8), (8, 12), (12, 40)):
            y = x[(x.v >= lo_) & (x.v < hi_)]
            bins[f"{lo_}-{hi_}"] = dict(n=int(len(y)), p50=float(y.alat4.median()) if len(y) else np.nan,
                                        p90=float(np.percentile(y.alat4, 90)) if len(y) else np.nan)
        A["A2"][nm] = dict(plan=pa, counterfactual=cf, human=hu, verdict=v, by_speed=bins)
    # A3: APP1X vs RSG turn medians on the shared scenes (eligible T2 decisions)
    E = D[elig("2") & np.isfinite(D.lr2)]
    tm = E.groupby(["key", "scene"]).agg(r2=("r2", "median"), q=("q", "median"), n=("r2", "size")).reset_index()
    a1x, rsg = tm[tm.key == "pai_APP1X-L18-s0"].set_index("scene"), tm[tm.key == "pai_RSG-L18-s0"].set_index("scene")
    sh = sorted(set(a1x.index) & set(rsg.index))
    import rf_receding as RC2
    A["A3"] = dict(n=len(sh), r2_app1x=float(a1x.loc[sh].r2.median()) if sh else np.nan, r2_rsg=float(rsg.loc[sh].r2.median()) if sh else np.nan,
                   q_app1x=float(a1x.loc[sh].q.median()) if sh else np.nan, q_rsg=float(rsg.loc[sh].q.median()) if sh else np.nan,
                   d_log_r2=RC2.paired_median([np.log(a1x.loc[s].r2) - np.log(rsg.loc[s].r2) for s in sh]),
                   d_q=RC2.paired_median([a1x.loc[s].q - rsg.loc[s].q for s in sh]),
                   abs_log_r2_app1x=float(np.median(np.abs(np.log(a1x.loc[sh].r2)))) if sh else np.nan,
                   abs_log_r2_rsg=float(np.median(np.abs(np.log(rsg.loc[sh].r2)))) if sh else np.nan,
                   completed_app1x=int(D[(D.key == "pai_APP1X-L18-s0") & D.scene.isin(sh)].groupby("scene").completed.first().sum()),
                   completed_rsg=int(D[(D.key == "pai_RSG-L18-s0") & D.scene.isin(sh)].groupby("scene").completed.first().sum()))
    # classes (x_ sets join their base set for the class table)
    D["grp"] = D.key.str.replace("x_", "", regex=False).map({"pai_RSG-L18-s0": "pai_RSG", "held_RSG-L18-s0": "held_RSG", "held_RSG-L18-s1": "held_RSG",
                                                                "pai_APP1X-L18-s0": "pai_APP1X"})
    for g in ("pai_RSG", "held_RSG", "pai_APP1X"):
        A["classes"][g] = {}
        for c in CLS_ORDER:
            x = D[(D.grp == g) & (D.cls == c)]
            if not len(x):
                continue
            e = x[elig("2").loc[x.index] & np.isfinite(x.lr2)]
            t_ = e.groupby("turn").agg(r2=("r2", "median"), q=("q", "median"))
            r = dict(turns=int(x.turn.nunique()), elig_turns=int(e.turn.nunique()), n_elig=int(len(e)),
                     r2=float(t_.r2.median()) if len(t_) else np.nan, q=float(t_.q.median()) if len(t_) else np.nan,
                     alat_p90=float(np.percentile(x.alat4.dropna(), 90)) if x.alat4.notna().any() else np.nan,
                     v_med=float(x.v.median()), vh_med=float(x.vh.median()))
            if e.turn.nunique() >= 5:
                f = a1_fit(e, n_boot=2000)
                r.update(slope=f["slope"], lo=f["lo"], hi=f["hi"])
            A["classes"][g][c] = r
    # per-turn csv
    pt = []
    for (k, s), x in D.groupby(["key", "scene"]):
        e = x[elig("2").loc[x.index] & np.isfinite(x.lr2)]
        pt.append(dict(key=k, scene=s, cls=x.cls.iloc[0], completed=bool(x.completed.iloc[0]), n_dec=len(x), n_elig=len(e),
                       r2_med=float(e.r2.median()) if len(e) else np.nan, q_med=float(e.q.median()) if len(e) else np.nan,
                       v_med=float(x.v.median()), vh_med=float(x.vh.median()), kp2_max=float(x.kp2.max()), kh2_max=float(x.kh2.max()),
                       alat4_max=float(x.alat4.max()), cf_alat4_max=float(x.cf_alat4.max()), h_alat4_max=float(x.h_alat4.max()),
                       slope=wls(e.lq.to_numpy(), e.lr2.to_numpy(), np.ones(len(e))) if len(e) >= 5 and np.ptp(e.lq) > 0.1 else np.nan))
    pd.DataFrame(pt).to_csv(Path(a.out) / "turns.csv", index=False)
    return D


# ---------------------------------------------------------------- Part B (Mac)
def ratio_boot(S, n_boot=N_BOOT, seed=SEED):
    """S: per-scene rows with dP, dH -> mean dP / mean dH with a scene bootstrap CI."""
    S = S[np.isfinite(S.dP) & np.isfinite(S.dH)]
    if not len(S):
        return dict(n=0)
    p, h = S.dP.to_numpy(), S.dH.to_numpy()
    rng = np.random.default_rng(seed)
    I = rng.integers(0, len(S), (n_boot, len(S)))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = p[I].mean(1) / h[I].mean(1)
    return dict(n=int(len(S)), dP=float(p.mean()), dH=float(h.mean()), R=float(p.mean() / h.mean()) if h.mean() != 0 else np.nan,
                lo=float(np.nanpercentile(R, 2.5)), hi=float(np.nanpercentile(R, 97.5)), dP_med=float(np.median(p)), dH_med=float(np.median(h)),
                diff=float((p - h).mean()))


def verdict_b(r, line, min_n=1):
    if r.get("n", 0) < min_n or not np.isfinite(r.get("dH", np.nan)) or r["dH"] < 0.5:
        return "untestable" + (" (human did not brake: mean dH < 0.5)" if r.get("n", 0) >= min_n else f" (< {min_n} scenes)")
    if r["R"] <= line and r["hi"] < 1:
        return "supported"
    if r["R"] > line and r["hi"] >= 1:
        return "refuted"
    return "inconclusive"


def part_b(a, res):
    import pandas as pd
    B = pd.read_csv(Path(a.brake) / "brake_dec.csv.gz")
    Sc = pd.read_csv(Path(a.brake) / "brake_scenes.csv")
    out = res["B"] = {}
    coll = Sc[Sc.primary.isin(COLL)]
    W = B[B.scene.isin(coll.scene) & (B.t_rel >= -2) & (B.t_rel < 0)]
    per = W.groupby("scene").agg(dP=("dP", "mean"), dH=("dH", "mean"), n=("dP", "size"), vp2_min=("vp2", "min"), v=("v", "mean"), vh=("vh", "mean"),
                                 e_ttc=("e_ttc", "min")).reset_index()
    first = B[B.scene.isin(coll.scene) & (B.t_rel >= -2.05) & (B.t_rel < 0)].sort_values("t0").groupby("scene").first()
    last = B[B.scene.isin(coll.scene) & (B.t_rel < 0)].sort_values("t0").groupby("scene").last()
    per = per.merge(coll[["scene", "primary", "v_ev", "v_ev2"]], on="scene")
    per["v_exec_drop"] = per.v_ev2 - per.v_ev
    per["hit_rng2"] = per.scene.map(first.hit_rng)
    rr = B[B.scene.isin(coll.scene) & (B.t_rel >= -2.6) & (B.t_rel < -1.4)].groupby("scene").apply(
        lambda z: float(np.polyfit(z.t_rel, z.hit_rng, 1)[0]) if z.hit_rng.notna().sum() >= 3 else np.nan, include_groups=False)
    per["hit_rr"] = per.scene.map(rr)
    per["hit_ttc2"] = np.where(per.hit_rr < 0, per.hit_rng2 / -per.hit_rr, np.inf)
    per["vh_ev"] = per.scene.map(last.vh)
    r1 = ratio_boot(per)
    out["B1"] = r1 | dict(verdict=verdict_b(r1, 0.25), scenes=per.to_dict("records"))
    # baseline: lead-free decisions
    free = B[B.pre & (B.v >= 2) & ~np.isfinite(B.e_gap)]
    bl = free.groupby("scene").dP.mean()
    out["baseline"] = dict(n_scenes=int(len(bl)), dP_mean=float(bl.mean()), dP_med=float(bl.median()),
                           dH_free_mean=float(free.groupby("scene").dH.mean().mean()))
    # B2
    match = Sc[~Sc.primary.isin(COLL) & (Sc.n_hb > 0)]
    E = B[B.scene.isin(match.scene) & B.pre & (B.v >= 1) & (B.e_ttc <= TTC_MAX)]
    p2 = E.groupby("scene").agg(dP=("dP", "mean"), dH=("dH", "mean"), n=("dP", "size")).reset_index()
    r2 = ratio_boot(p2)
    out["B2"] = r2 | dict(verdict=verdict_b(r2, 0.5, 5), matched=int(len(match)), scenes_elig=int(len(p2)), decisions=int(len(E)),
                          baseline_matched=float(bl.reindex(match.scene).mean()))
    Eb = E[E.h_ttc <= TTC_MAX]
    out["B2_both"] = ratio_boot(Eb.groupby("scene").agg(dP=("dP", "mean"), dH=("dH", "mean")).reset_index()) | dict(decisions=int(len(Eb)))
    Hown = B[B.scene.isin(match.scene)].copy()
    # own-time human: dH at the human's own times with its lead TTC <= 3 s (decision rows carry the human at t_s; use rows where h_ttc <= 3)
    ho = Hown[Hown.h_ttc <= TTC_MAX].groupby("scene").dH.mean()
    po = E.groupby("scene").dP.mean()
    j = pd.DataFrame(dict(dP=po, dH=ho)).dropna().reset_index()
    out["B2_own"] = ratio_boot(j)
    lq = Sc[Sc.primary.isin(LEADQ)]
    El = B[B.scene.isin(lq.scene) & B.pre & (B.v >= 1) & (B.e_ttc <= TTC_MAX)]
    out["B2_leadq"] = ratio_boot(El.groupby("scene").agg(dP=("dP", "mean"), dH=("dH", "mean")).reset_index()) | dict(
        scenes=int(len(lq)), in_matched=int(lq.scene.isin(match.scene).sum()))
    # B3 oncoming-lane events
    on = Sc[np.isfinite(Sc.onc_t)]
    b3 = []
    for _, s in on.iterrows():
        x = B[B.scene == s.scene]
        pre = x[(x.t0 >= s.onc_t - 2e6) & (x.t0 < s.onc_t)]
        dur = x[(x.t0 >= s.onc_t) & (x.t0 < min(s.onc_t + 2e6, s.t_f))]
        b3.append(dict(scene=s.scene, primary=s.primary, dP_pre=pre.dP.mean(), dH_pre=pre.dH.mean(), dP_in=dur.dP.mean(), dH_in=dur.dH.mean(),
                       v=pre.v.mean(), vh=pre.vh.mean(), n_pre=len(pre), n_in=len(dur), e_ttc_min=float(pd.concat([pre, dur]).e_ttc.min())))
    b3 = pd.DataFrame(b3)
    out["B3"] = dict(n=int(len(b3)), pre=ratio_boot(b3.rename(columns=dict(dP_pre="dP", dH_pre="dH"))[["dP", "dH"]]),
                     inside=ratio_boot(b3.rename(columns=dict(dP_in="dP", dH_in="dH"))[["dP", "dH"]]), scenes=b3.to_dict("records"))
    per.assign(part="B1").to_csv(Path(a.out) / "brake_scenes.csv", index=False)
    with open(Path(a.out) / "brake_scenes.csv", "a") as f:
        p2.merge(match[["scene", "primary", "n_hb", "hb_drop"]], on="scene").assign(part="B2").to_csv(f, index=False)
        b3.assign(part="B3").to_csv(f, index=False)
    return B, Sc, per


# ---------------------------------------------------------------- report
def fnum(v, p=2, sign=True):
    return "-" if v is None or not np.isfinite(v) else (f"{v:+.{p}f}" if sign else f"{v:.{p}f}")


def write_md(out, res):
    A, B = res["A"], res["B"]
    f = fnum
    md = ["# SPEED-GEOMETRY reading (rf_speedgeo.py)", "",
          "Prereg `plans/2026-10-16-speed-geometry-prereg.md`. Part A: approach + turn decisions of the strict turns; r_T = plan peak curvature "
          "over 0-T s / human peak curvature over the same stretch of road (arc-matched), q = ego speed / human speed at the same route station. "
          "Turn-weighted WLS of log r on log q, turn-cluster bootstrap 95 % CI (B 10 000). Part B: plan 2 s speed drop dP = v - v_plan(2 s) vs the "
          "human's dH = v_h(t_s) - v_h(t_s + 2 s) at the same log station; R = mean dP / mean dH over scenes, scene bootstrap CI.", "",
          "## Part A", "", "A1 (slope of log r on log q; A1c = coefficient on log v with log v_h and log kappa_h as covariates; within = turn-demeaned):", "",
          "| set | variant | turns / decisions | slope [CI] | A1c log v [CI] | within [CI] | verdict |", "|:--|:--|:--|:--|:--|:--|:--|"]
    for nm, v in A["A1"].items():
        for lab, r in v.items():
            if "slope" not in r:
                md.append(f"| {nm} | {lab} | {r.get('n_turns', 0)} / {r.get('n_dec', 0)} | - | - | - | {r['verdict']} |")
                continue
            md.append(f"| {nm} | {lab} | {r['n_turns']} / {r['n_dec']} | {f(r['slope'])} [{f(r['lo'])}, {f(r['hi'])}] | {f(r['c_lv'])} "
                      f"[{f(r['c_lo'])}, {f(r['c_hi'])}] | {f(r['within'])} [{f(r['w_lo'])}, {f(r['w_hi'])}] | {r['verdict']} |")
    md += ["", "A2 (peak lateral acceleration over the plan's 0-4 s, m/s^2; counterfactual = human curvature on the same stretch x ego speed^2, "
           "decisions with v > v_h; human = human curvature x human speed^2):", "",
           "| set | plan p90 [CI] (n) | counterfactual p90 [CI] (n) | human p90 [CI] | verdict |", "|:--|:--|:--|:--|:--|"]
    for nm, r in A["A2"].items():
        p, c, h = r["plan"], r["counterfactual"], r["human"]
        cell = lambda z: f"{f(z['p90'], 2, False)} [{f(z['lo'], 2, False)}, {f(z['hi'], 2, False)}] ({z['n']})" if z.get("n") else "-"  # noqa: E731
        md.append(f"| {nm} | {cell(p)} | {cell(c)} | {cell(h)} | {r['verdict']} |")
    md += ["", "Plan a_lat by ego speed (pai_RSG, m/s: n, median, p90): " + "; ".join(
        f"{k} {b['n']}, {f(b['p50'], 2, False)}, {f(b['p90'], 2, False)}" for k, b in A["A2"]["pai_RSG"]["by_speed"].items()), ""]
    a3 = A["A3"]
    md += [f"A3 (APP1X vs RSG, {a3['n']} shared turns with eligible decisions): median r_2 {f(a3['r2_app1x'], 2, False)} vs {f(a3['r2_rsg'], 2, False)}, "
           f"median |log r_2| {f(a3['abs_log_r2_app1x'], 2, False)} vs {f(a3['abs_log_r2_rsg'], 2, False)}, median q {f(a3['q_app1x'], 2, False)} vs "
           f"{f(a3['q_rsg'], 2, False)}; paired log r_2 difference {f(a3['d_log_r2'].get('median'))} [{f(a3['d_log_r2'].get('lo'))}, "
           f"{f(a3['d_log_r2'].get('hi'))}], paired q difference {f(a3['d_q'].get('median'))} [{f(a3['d_q'].get('lo'))}, {f(a3['d_q'].get('hi'))}]; "
           f"completed {a3['completed_app1x']} vs {a3['completed_rsg']}.", "",
           "Per class (turn medians of the eligible T2 decisions; slope with a 2 000-resample CI when >= 5 turns):", "",
           "| set | class | turns (eligible) | decisions | median r_2 | median q | v / v_h median | plan a_lat p90 | slope [CI] |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for g, v in A["classes"].items():
        for c, r in v.items():
            md.append(f"| {g} | {c} | {r['turns']} ({r['elig_turns']}) | {r['n_elig']} | {f(r['r2'], 2, False)} | {f(r['q'], 2, False)} | "
                      f"{f(r['v_med'], 1, False)} / {f(r['vh_med'], 1, False)} | {f(r['alat_p90'], 2, False)} | "
                      + (f"{f(r['slope'])} [{f(r['lo'])}, {f(r['hi'])}] |" if "slope" in r else "- |"))
    b1 = B["B1"]
    rc = lambda r: (f"{f(r['R'])} [{f(r['lo'])}, {f(r['hi'])}] (n {r['n']}; dP {f(r['dP'])}, dH {f(r['dH'])} m/s)" if r.get("n") else "-")  # noqa: E731
    md += ["", "## Part B", "",
           f"B1 (collision zeros, window [t_ev - 2 s, t_ev)): R1 = {rc(b1)} -> **{b1['verdict']}**.",
           f"B2 (human braked for a lead, no collision; ego lead TTC <= 3 s): {B['B2']['matched']} matched scenes, {B['B2']['scenes_elig']} with "
           f"eligible decisions ({B['B2']['decisions']}): R2 = {rc(B['B2'])} -> **{B['B2']['verdict']}**.",
           f"- both facing a lead (human TTC <= 3 s at t_s too, {B['B2_both']['decisions']} decisions): {rc(B['B2_both'])}",
           f"- own times (plan at ego TTC <= 3 s vs human at human TTC <= 3 s): {rc(B['B2_own'])}",
           f"- taxonomy lead / queue scenes ({B['B2_leadq']['scenes']}, {B['B2_leadq']['in_matched']} in the matched set): {rc(B['B2_leadq'])}",
           f"- plan baseline drop on lead-free decisions (v >= 2): mean over {B['baseline']['n_scenes']} rollouts {f(B['baseline']['dP_mean'])} m/s "
           f"(human there {f(B['baseline']['dH_free_mean'])}); matched scenes {f(B['B2']['baseline_matched'])}",
           f"B3 (d299 oncoming-lane events, n {B['B3']['n']}): before [t_e - 2 s, t_e) {rc(B['B3']['pre'])}; in the lane [t_e, t_e + 2 s) {rc(B['B3']['inside'])}.",
           "", "B1 per scene (m/s; executed drop = ego v(t_ev - 2 s) - v(t_ev); hit range / TTC at t_ev - 2 s):", "",
           "| scene | primary | dP | dH | v / v_h | v_plan(2 s) min | executed drop | v at t_ev | human v at the event station | hit range m | hit TTC s | ego lead TTC min |",
           "|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for s in b1["scenes"]:
        md.append(f"| {s['scene'][7:15]} | {s['primary']} | {f(s['dP'])} | {f(s['dH'])} | {f(s['v'], 1, False)} / {f(s['vh'], 1, False)} | "
                  f"{f(s['vp2_min'], 1, False)} | {f(s['v_exec_drop'])} | {f(s['v_ev'], 1, False)} | {f(s['vh_ev'], 1, False)} | "
                  f"{f(s['hit_rng2'], 1, False)} | {f(s['hit_ttc2'], 1, False)} | {f(s['e_ttc'], 1, False)} |")
    md += ["", "B3 per event:", "", "| scene | primary | dP / dH before | dP / dH in lane | v / v_h | ego lead TTC min |", "|:--|:--|:--|:--|:--|--:|"]
    for s in B["B3"]["scenes"]:
        md.append(f"| {s['scene'][7:15]} | {s['primary']} | {f(s['dP_pre'])} / {f(s['dH_pre'])} | {f(s['dP_in'])} / {f(s['dH_in'])} | "
                  f"{f(s['v'], 1, False)} / {f(s['vh'], 1, False)} | {f(s['e_ttc_min'], 1, False)} |")
    (out / "speed_geo.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def fig(D, B, per, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    col = {"completed": "#2a78d6", "late_wide": "#eb6834", "straight": "#1baf7a", "over_turn": "#eda100", "no_zero": "#8c8c8c", "other": "#b07aa1"}
    fg = plt.figure(figsize=(17, 10))
    gs = GridSpec(2, 12, figure=fg, hspace=0.38, wspace=0.9)
    el = (D.v >= V_MIN) & (D.vh >= V_MIN) & np.isfinite(D.kp2) & (D.kh2 >= KH_MIN) & np.isfinite(D.lr2)
    ax = fg.add_subplot(gs[0, 0:4])
    for k, mk, nm in (("pai_RSG-L18-s0", "o", "PAI RSG"), ("held", "^", "held RSG")):
        x = D[el & (D.key.str.startswith(k) if k == "held" else D.key == k)]
        for c, y in x.groupby("cls"):
            ax.scatter(y.lq, y.lr2, s=3, alpha=0.12, color=col.get(c, "#444"), marker=mk, lw=0)
            t = y.groupby("turn")[["lq", "lr2"]].median()
            ax.scatter(t.lq, t.lr2, s=46 if k != "held" else 30, color=col.get(c, "#444"), marker=mk, edgecolor="white", lw=0.6,
                       label=f"{c} ({nm})")
    xx = np.linspace(-1.2, 1.2, 10)
    ax.plot(xx, 0 * xx, color="#888", lw=1, ls="--")
    ax.plot(xx, -2 * xx, color="#888", lw=1, ls=":")
    ax.text(0.9, 0.08, "copy the human line", fontsize=8, color="#52514e", ha="right")
    ax.text(0.55, -1.25, "a_lat cap (slope -2)", fontsize=8, color="#52514e")
    ax.set(xlim=(-1.2, 1.2), ylim=(-4, 2), xlabel="log speed ratio  ego / human at the same station",
           ylabel="log curvature ratio  plan / human (0-2 s, same road)", title="A1: curvature ratio vs speed ratio (big = turn median)")
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="lower left")
    ax = fg.add_subplot(gs[0, 4:8])
    x = D[el & D.key.isin(["pai_RSG-L18-s0", "pai_APP1X-L18-s0"])].groupby(["key", "scene"])[["lq", "lr2"]].median().reset_index()
    a1x, rsg = x[x.key == "pai_APP1X-L18-s0"].set_index("scene"), x[x.key == "pai_RSG-L18-s0"].set_index("scene")
    for s in sorted(set(a1x.index) & set(rsg.index)):
        ax.annotate("", (a1x.loc[s].lq, a1x.loc[s].lr2), (rsg.loc[s].lq, rsg.loc[s].lr2),
                    arrowprops=dict(arrowstyle="-|>", color="#9aa5b1", lw=0.8, mutation_scale=8))
    ax.scatter(rsg.lq, rsg.lr2, s=30, color="#eb6834", label="RSG turn median", zorder=3)
    ax.scatter(a1x.lq, a1x.lr2, s=30, color="#2a78d6", label="APP1X turn median", zorder=3)
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.axvline(0, color="#888", lw=1, ls="--")
    ax.set(xlim=(-1.2, 1.2), ylim=(-4, 2), xlabel="log speed ratio", ylabel="log curvature ratio (0-2 s)", title="A3: same turn, RSG -> APP1X (slower entry)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax = fg.add_subplot(gs[0, 8:12])
    x = D[(D.key == "pai_RSG-L18-s0") & (D.v >= V_MIN)]
    ax.scatter(x.v, x.alat4, s=4, alpha=0.3, color="#eb6834", lw=0, label="plan, peak 0-4 s")
    y = x[x.v > x.vh]
    ax.scatter(y.v, y.cf_alat4, s=4, alpha=0.3, color="#2a78d6", lw=0, label="human line at ego speed (v > v_h)")
    ax.axhline(ALAT_GATE, color="#52514e", lw=1, ls="--")
    ax.text(2.2, ALAT_GATE + 0.2, "4.5 m/s^2 (d293 gate)", fontsize=8, color="#52514e")
    ax.set(ylim=(0, 15), xlabel="ego speed (m/s)", ylabel="lateral acceleration (m/s^2)", title="A2: implied lateral acceleration, PAI RSG turns")
    ax.legend(fontsize=8, frameon=False, loc="upper left", markerscale=3)
    picks = []
    for grp in (("coll_veh_oncoming",), ("coll_veh_crossing",), ("coll_veh_side",), ("coll_cyclist", "coll_ped")):
        s = sorted(per[per.primary.isin(grp)].scene)
        if s:
            picks.append(s[0])
    for n, s in enumerate(picks[:4]):
        ax = fg.add_subplot(gs[1, 3 * n:3 * n + 3])
        x = B[(B.scene == s) & (B.t_rel >= -5) & (B.t_rel <= 0.05)].sort_values("t0")
        ax.plot(x.t_rel, x.v, color="#52514e", lw=2, label="ego speed")
        ax.plot(x.t_rel, x.vp2, color="#eb6834", lw=1.6, label="plan speed at +2 s")
        ax.plot(x.t_rel, x.vh, color="#2a78d6", lw=1.6, ls="--", label="human speed at this station")
        ax.plot(x.t_rel, x.vh2, color="#2a78d6", lw=1.2, ls=":", label="human speed 2 s later")
        ax.axvspan(-2, 0, color="#f2f2f2", zorder=0)
        a2 = ax.twinx()
        a2.plot(x.t_rel, x.hit_rng, color="#9c36b5", lw=1, alpha=0.7)
        a2.set_ylabel("range to the hit object (m)", color="#9c36b5", fontsize=8)
        a2.set_ylim(0, max(40, np.nanmax(x.hit_rng) if x.hit_rng.notna().any() else 40))
        p = per[per.scene == s].iloc[0]
        ax.set(xlabel="time to the collision (s)", ylabel="speed (m/s)", ylim=(0, None),
               title=f"{s[7:15]} {p.primary.replace('coll_', '')}: dP {p.dP:+.2f} vs dH {p.dH:+.2f}")
        ax.title.set_fontsize(9)
        if n == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower left")
    for a_ in fg.axes:
        a_.grid(alpha=0.2)
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(png, dpi=120, bbox_inches="tight")
    print("figure ->", png)


def read(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    D = part_a(a, res)
    B, Sc, per = part_b(a, res)
    (out / "speed_geo.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    write_md(out, res)
    if a.fig:
        fig(D, B, per, a.fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("brake-extract")
    e.add_argument("--tax", required=True), e.add_argument("--plans", required=True), e.add_argument("--taxcsv", required=True)
    e.add_argument("--laneoff", required=True), e.add_argument("--out", required=True), e.add_argument("--workers", type=int, default=0)
    r = sp.add_parser("read")
    r.add_argument("--pai", required=True), r.add_argument("--held", required=True), r.add_argument("--cls", required=True)
    r.add_argument("--brake", required=True), r.add_argument("--out", required=True), r.add_argument("--fig", default="")
    a = ap.parse_args()
    {"brake-extract": brake_extract, "read": read}[a.cmd](a)
