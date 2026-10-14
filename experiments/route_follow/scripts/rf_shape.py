#!/usr/bin/env python3
"""Turn-shape read for the turning school's demonstration rows (results/p3_branch/shape.md): is a P3 pushed branch steeper / less smooth
than the human on the same turn? CPU only, stored rollouts read, never written.

  extract-p3  (Tokyo, plain python3 + numpy)  rf_shape.py extract-p3 --arms /data/runs/route_night/p3/arms --out <pkl>
              per P3 rollout (arm dir x scene): the controller trace (10 Hz t, x, y, yaw, vx) and per decision (k, now, branch on,
              served plan = poses_branch inside the window else poses, raw plan poses).
  report      (Mac, .venv)  rf_shape.py report --r281 <rf_pai_follow extract pkl> --p3 <extract-p3 pkl> --read <p3 read.json>
              --exam <hard_exam.txt> --out <dir> --fig <png>   -> <dir>/shape.json, <dir>/shape_tables.md, the figure.

Definitions.
  turn      rf_pai_follow's strict turn on RSG's AlpaSim route (MAP, 40 m; the same route the evaluator saw): 30 m window with net heading
            change >= 60 deg, entry / exit at 10 / 90 %, exit + 10 m inside the log's travel; first such turn per scene. U = |turn| >= 150 deg.
            Topo class: rf_pai_follow.topo with the HUMAN log as the ego (ego lane at entry - 10 m, its successor chains).
  window    route stations [entry - 40, exit] (clipped to where the track starts); every track is cut at its first zero-score event.
            P3 also: the push window = decision k_eff .. k_eff + 30 (3 s) by the driver's `now`.
  kappa     track resampled at 0.25 m of its own arc length; heading = chord over +-0.5 m; kappa = heading difference over +-1 m / 2 m
            (support ~3 m); dkappa/ds = kappa difference over +-1 m / 2 m. Peak = max |.| inside the window.
  speed     human: log positions -> step speed, 5-sample box (rf_pai_follow's v_log); rollouts: controller vx. a_lat = v^2 kappa at the
            same arc point; decel = -(v(t + 0.5 s) - v(t - 0.5 s)) / 1 s on the 5-sample-boxed speed, peak inside the window.
  v_-40/-20/-10/0  speed when the route station first crosses entry - d (rf_pai_follow.speed_at); v_min = min speed in [entry, exit];
            dec_start = first route station after entry - 40 where v < 0.9 v_-40, relative to the entry (m; nan: never, or v_-40 missing
            (the track starts inside entry - 40) or < 2 m/s).
  plan      served P3 plan (8 poses, 0.5 s, ego frame, after the push transform) with the ego at the origin: yaw interpolated on the pose
            arc length, same kappa operator; plan a_lat = (segment length / 0.5 s)^2 kappa. Peak over the 30 window decisions.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

DS, HCH, KCH = 0.25, 0.5, 1.0                      # resample step, heading chord half-width, kappa / dkappa half-width (m)
WIN_PRE, U_DEG, ALAT_CAP = 40.0, 150.0, 3.3
QS = (5, 25, 50, 75, 95)


# ---------------------------------------------------------------- extract-p3 (Tokyo, no pandas)
def yawq(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def extract_p3(a):
    out = []
    for arm_dir in sorted(p for p in glob.glob(os.path.join(a.arms, "*")) if os.path.isdir(p)):
        arm = os.path.basename(arm_dir)
        for run in sorted(glob.glob(os.path.join(arm_dir, "runs", "*"))):
            f = os.path.join(run, "sim", "aggregate", "results-summary.json")
            if not os.path.exists(f):
                continue
            dec = {}
            for line in open(os.path.join(run, "driver", "drive.jsonl")):
                if '"kind": "drive"' not in line:
                    continue
                r = json.loads(line)
                if "poses" not in r:
                    continue
                on = bool((r.get("branch") or {}).get("on"))
                dec.setdefault(r["session"], []).append((r["k"], r["now"], on, r.get("poses_branch", r["poses"]), r["poses"], r.get("switch")))
            for ro in json.load(open(f))["rollouts"]:
                rid = ro["rollout_id"]
                cf = os.path.join(run, "sim", "controller", f"alpasim_controller_{rid}.csv")
                if not os.path.exists(cf):
                    continue
                rows = list(csv.reader(open(cf)))
                h = {k: i for i, k in enumerate(rows[0])}
                A = np.array(rows[1:], float)
                c = lambda k: A[:, h[k]]  # noqa: E731
                D = sorted(dec.get(rid, []), key=lambda x: x[0])
                out.append(dict(arm=arm, scene=ro["clipgt_id"], rid=rid, score=ro.get("score"),
                                ctrl=dict(t=c("timestamp_us"), x=c("x"), y=c("y"), yaw=np.unwrap(yawq(c("qx"), c("qy"), c("qz"), c("qw"))), vx=c("vx")),
                                k=np.array([d[0] for d in D]), now=np.array([d[1] for d in D], float), on=np.array([d[2] for d in D]),
                                served=np.array([d[3] for d in D], float), raw=np.array([d[4] for d in D], float),
                                switch=next((d[5] for d in D if d[5] is not None), None)))
        print(arm, len([o for o in out if o["arm"] == arm]), flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "rollouts ->", a.out)


# ---------------------------------------------------------------- shape operators
def box5(v):
    v = np.asarray(v, float)
    return np.convolve(np.r_[v[:2], v, v[-2:]], np.ones(5) / 5, "valid")


def resample(xy, extra):
    """Track (n, 2) + per-sample arrays -> 0.25 m arc grid: s, xy, arrays interpolated on the arc (stops collapse)."""
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    keep = np.r_[True, np.diff(d) > 1e-4]
    d, xy = d[keep], xy[keep]
    if len(d) < 2:
        return None
    s = np.arange(0.0, d[-1] + 1e-9, DS)
    g = np.c_[np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])]
    return s, g, {k: np.interp(s, d, np.asarray(v, float)[keep]) for k, v in extra.items()}


def kappa_ops(s, g=None, yaw=None):
    """Heading (chord +-HCH m, or a given yaw on the grid), kappa and dkappa/ds (differences over +-KCH m) on the 0.25 m grid."""
    n, hc, kc = len(s), int(round(HCH / DS)), int(round(KCH / DS))
    i = np.arange(n)
    if yaw is None:
        a, b = np.clip(i - hc, 0, n - 1), np.clip(i + hc, 0, n - 1)
        h = np.unwrap(np.arctan2(g[b, 1] - g[a, 1], g[b, 0] - g[a, 0]))
    else:
        h = np.unwrap(yaw)
    a, b = np.clip(i - kc, 0, n - 1), np.clip(i + kc, 0, n - 1)
    span = np.maximum(s[b] - s[a], 1e-6)
    k = (h[b] - h[a]) / span
    k[span < KCH] = np.nan                            # the ends: not enough arc for the operator
    dk = (k[b] - k[a]) / span
    dk[span < KCH] = np.nan
    return h, k, dk


def nanmaxabs(x):
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    return float(x.max()) if len(x) else np.nan


def track_shape(t, xy, v, st, lo, hi, tz=np.inf, t_lo=-np.inf, t_hi=np.inf):
    """Peak |kappa|, |a_lat|, |dkappa/ds|, decel of a 10 Hz track inside route stations [lo, hi], time [t_lo, t_hi] and before tz."""
    t, v = np.asarray(t, float), box5(v)
    dec = np.full(len(v), np.nan)
    dec[5:-5] = -(v[10:] - v[:-10]) / ((t[10:] - t[:-10]) * 1e-6)
    pre = t < tz
    R = resample(xy[pre], dict(t=t[pre], v=v[pre], st=st[pre]))
    out = dict(kmax=np.nan, alat=np.nan, dk=np.nan, dec=np.nan, arc=0.0, st_kpk=np.nan, st_apk=np.nan, v_apk=np.nan)
    if R is None:
        return out
    s, g, E = R
    _, k, dk = kappa_ops(s, g)
    m = (E["st"] >= lo) & (E["st"] <= hi) & (E["t"] >= t_lo) & (E["t"] <= t_hi)
    if m.sum() < 4:
        return out
    mt = pre & (st >= lo) & (st <= hi) & (t >= t_lo) & (t <= t_hi)
    ka, aa = np.where(m & np.isfinite(k), np.abs(k), -1.0), np.where(m & np.isfinite(k), np.abs(E["v"] ** 2 * k), -1.0)
    out.update(st_kpk=float(E["st"][ka.argmax()]), st_apk=float(E["st"][aa.argmax()]), v_apk=float(E["v"][aa.argmax()]))
    out.update(kmax=nanmaxabs(k[m]), alat=nanmaxabs(E["v"][m] ** 2 * k[m]), dk=nanmaxabs(dk[m]), arc=float(m.sum() * DS),
               dec=float(np.nanmax(dec[mt])) if np.isfinite(dec[mt]).any() else np.nan)
    return out


def speeds(t, st, v, ent, ext, tz=np.inf):
    import rf_pai_follow as F
    v = box5(v)
    o = {f"v_{d}": F.speed_at(t, st, v, ent - d, tz) if ent - d >= st[0] - 0.5 else np.nan for d in (40, 20, 10, 0)}   # nan: track starts later
    m = (st >= ent) & (st <= ext) & (t < tz)
    o["v_min"] = float(v[m].min()) if m.any() else np.nan
    o["dec_start"] = np.nan
    if np.isfinite(o["v_40"]) and o["v_40"] >= 2.0:
        j = np.flatnonzero((st > ent - WIN_PRE) & (st <= ext) & (t < tz) & (v < 0.9 * o["v_40"]))
        o["dec_start"] = float(st[j[0]] - ent) if len(j) else np.nan
    return o


def plan_shape(P):
    """Served plan (8, 3) ego frame -> peak |kappa|, |a_lat| (0.5 s poses), |dkappa/ds| with the same operator (yaw on the arc)."""
    xy = np.r_[[[0.0, 0.0]], P[:, :2]]
    yaw = np.r_[0.0, P[:, 2]]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    d = np.r_[0.0, np.cumsum(seg)]
    keep = np.r_[True, np.diff(d) > 1e-3]
    if keep.sum() < 3 or d[-1] < 2 * KCH:
        return np.nan, np.nan, np.nan
    s = np.arange(0.0, d[keep][-1] + 1e-9, DS)
    yw = np.interp(s, d[keep], np.unwrap(yaw[keep]))
    vs = np.interp(s, d[keep][1:], (seg / 0.5)[keep[1:]]) if keep[1:].any() else np.zeros_like(s)
    _, k, dk = kappa_ops(s, yaw=yw)
    return nanmaxabs(k), nanmaxabs(vs ** 2 * k), nanmaxabs(dk)


# ---------------------------------------------------------------- report (Mac)
def rel(o, ent):
    """Peak stations relative to the turn entry."""
    return dict(o, st_kpk=o["st_kpk"] - ent, st_apk=o["st_apk"] - ent)


def human_and_rsg(r281, exam):
    """Per strict PAI turn: the human (log) and RSG (own rollout) shape + speeds, turn geometry, topo, split."""
    import rf_pai_follow as F
    rows, routes = [], {}
    for sc in r281["scenes"]:
        o = sc["rolls"].get("rsg")
        if o is None or o.get("ctrl") is None:
            continue
        R = F.dedup(o["route"])
        S = F.stations(R)
        C = o["ctrl"]
        P = np.c_[C["x"], C["y"]]
        st, _ = F.track(R, S, P)
        LG = sc["log"]
        lt, llat = F.track(R, S, LG["P"])
        s0, s_log = st[0], float(lt.max())
        s, h = F.heading(R, S)
        tt = F.find_turn(s, h, s0, s_log)
        if tt is None or tt[1] + F.PAST > s_log:
            continue
        ent, ext, th = tt
        deg = float(np.degrees(th))
        side = "U" if abs(deg) >= U_DEG else ("L" if deg > 0 else "R")
        vlog = np.r_[0.0, np.hypot(*np.diff(LG["P"], axis=0).T) / np.maximum(np.diff(LG["T"]) / 1e6, 1e-3)]
        # topo with the human as the ego (ego lane at entry - 10 m): stop the "ego" at the entry so the turn found is this one
        i_ent = int(np.flatnonzero(lt >= ent)[0])
        lyaw = np.unwrap(np.arctan2(*np.gradient(LG["P"], axis=0)[:, ::-1].T))
        fake = dict(ctrl=dict(t=LG["T"], x=LG["P"][:, 0], y=LG["P"][:, 1], yaw=lyaw, vx=box5(vlog)))
        z = F.topo(sc, fake, R, S, s, h, LG["P"], LG["T"], lt, llat, lt, vlog, float(LG["T"][i_ent]), s_log)
        topo = z.get("topo") if abs(z.get("z_entry", -1e9) - ent) < 1e-6 else "mismatch"
        base = dict(scene=sc["scene"], short=sc["scene"][7:15], side=side, deg=deg, entry=ent, exit=ext, topo=topo,
                    split="hard_exam" if sc["scene"] in exam else "rest", rsg_score=float(o["score"] or 0))
        hs = rel(track_shape(LG["T"], LG["P"], vlog, lt, ent - WIN_PRE, ext), ent)
        rows.append(dict(base, who="human", **hs, **speeds(LG["T"], lt, vlog, ent, ext)))
        tz = o["tz"] if o["tz"] is not None and not o["score"] else np.inf
        rs = rel(track_shape(C["t"], P, C["vx"], st, ent - WIN_PRE, ext, tz), ent)
        rows.append(dict(base, who="rsg", **rs, **speeds(C["t"], st, C["vx"], ent, ext, tz)))
        routes[sc["scene"]] = (R, S, ent, ext, LG)
    return rows, routes


def gate_route_zeros(r281, pf):
    """rf_pai_follow's 20-turn figure: the log's speed at the entry of the turn at each RSG route zero (z_entry), where the ego got there."""
    import rf_pai_follow as F
    z = pf[(pf.label == "rsg") & (pf.route_zero == True) & pf.v_ego_0.notna()].set_index("scene")  # noqa: E712
    here = {}
    for sc in r281["scenes"]:
        if sc["scene"] not in z.index:
            continue
        R = F.dedup(sc["rolls"]["rsg"]["route"])
        S = F.stations(R)
        LG = sc["log"]
        lt, _ = F.track(R, S, LG["P"])
        vlog = np.r_[0.0, np.hypot(*np.diff(LG["P"], axis=0).T) / np.maximum(np.diff(LG["T"]) / 1e6, 1e-3)]
        here[sc["scene"]] = F.speed_at(LG["T"], lt, box5(vlog), z.loc[sc["scene"], "z_entry"])
    h = pd_series(here)
    return dict(n=int(h.notna().sum()), median_here=float(h.median()), median_pf=float(z.v_log_0.median()),
                max_absdiff=float((h - z.v_log_0.reindex(h.index)).abs().max()))


def pd_series(d):
    import pandas as pd
    return pd.Series(d, dtype=float)


def p3_rows(p3, read, routes):
    import rf_pai_follow as F
    rd = {(r["arm"], r["scene"]): r for r in read["rows"]}
    rows = []
    for o in p3:
        r = rd.get((o["arm"], o["scene"]))
        if r is None or o["scene"] not in routes:
            continue
        R, S, ent, ext, LG = routes[o["scene"]]
        C = o["ctrl"]
        P = np.c_[C["x"], C["y"]]
        st, _ = F.track(R, S, P)
        tz = C["t"][0] + r["t_zero"] * 1e6 if r.get("t_zero") is not None else np.inf
        ho, tf = o["arm"].split("@")
        k_eff = r["k"]
        ik = np.flatnonzero(o["k"] == k_eff)
        iw = np.flatnonzero((o["k"] >= k_eff) & (o["k"] < k_eff + 30))
        t_lo = float(o["now"][ik[0]]) if len(ik) else np.nan
        t_hi = t_lo + 3.0e6
        base = dict(arm=o["arm"], ho=ho, tf=tf, ident=tf == "g1s1", scene=o["scene"], short=o["scene"][7:15], done=bool(r.get("done")),
                    score=float(r["score"]), zero=r.get("zero") or r.get("end"), k_eff=k_eff, n_on=int(o["on"][iw].sum()) if len(iw) else 0)
        whole = rel(track_shape(C["t"], P, C["vx"], st, ent - WIN_PRE, ext, tz), ent)
        s_ho = float(np.interp(t_lo, C["t"], st) - ent) if np.isfinite(t_lo) else np.nan
        push = track_shape(C["t"], P, C["vx"], st, -np.inf, np.inf, tz, t_lo, t_hi)
        pl = np.array([plan_shape(o["served"][i]) for i in iw]) if len(iw) else np.full((1, 3), np.nan)
        pr = np.array([plan_shape(o["raw"][i]) for i in iw]) if len(iw) else np.full((1, 3), np.nan)
        rows.append(dict(base, s_ho=s_ho, **whole, **{f"w_{k}": v for k, v in push.items()},
                         plan_kmax=float(np.nanmax(pl[:, 0])), plan_alat=float(np.nanmax(pl[:, 1])), plan_dk=float(np.nanmax(pl[:, 2])),
                         raw_kmax=float(np.nanmax(pr[:, 0])), raw_alat=float(np.nanmax(pr[:, 1])),
                         **speeds(C["t"], st, C["vx"], ent, ext, tz)))
    return rows


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(n=int(len(x)), **{f"p{p}": (float(np.percentile(x, p)) if len(x) else None) for p in QS})


QTY = ("kmax", "alat", "dk", "dec", "v_40", "v_20", "v_10", "v_0", "v_min", "dec_start", "st_kpk", "st_apk", "v_apk")


def report(a):
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    exam = {x.strip() for x in open(a.exam) if x.strip()}
    r281 = pickle.load(open(a.r281, "rb"))
    rows, routes = human_and_rsg(r281, exam)
    H = pd.DataFrame(rows)
    T = pd.DataFrame(p3_rows(pickle.load(open(a.p3, "rb")), json.load(open(a.read)), routes))
    hu, rs = H[H.who == "human"].set_index("scene"), H[H.who == "rsg"].set_index("scene")
    T["topo"], T["side"] = T.scene.map(hu.topo), T.scene.map(hu.side)
    for k in ("kmax", "alat", "dk"):
        T[f"r_{k}"] = T[k] / T.scene.map(hu[k])
    out = Path(a.out)
    J = dict(defs=__doc__.split("Definitions.")[1].strip(), n_turns=int(len(hu)),
             counts=dict(side=hu.side.value_counts().to_dict(), topo=hu.topo.value_counts().to_dict(), split=hu.split.value_counts().to_dict()))
    # reproduction gate against rf_pai_follow's numbers
    pf = pd.read_csv(a.pf) if a.pf else None
    if pf is not None:
        r = pf[(pf.label == "rsg") & (pf.turn == True)].set_index("scene")  # noqa: E712
        J["gate"] = dict(turns_pf=int(len(r)), turns_here=int(len(hu)), same_scenes=sorted(set(r.index)) == sorted(set(hu.index)),
                         same_entry_exit=bool((r.entry.reindex(hu.index) == hu.entry).all() and (r.exit.reindex(hu.index) == hu.exit).all()),
                         v_log_entry_pf_median=float(r.v_log_entry.median()), v_0_here_median=float(hu.v_0.median()),
                         v_0_max_absdiff=float((hu.v_0 - r.v_log_entry.reindex(hu.index)).abs().max()),
                         route_zero_turns=gate_route_zeros(r281, pf))
    groups = {"all": hu.index}
    for c in ("side", "topo", "split"):
        for v, g in hu.groupby(c):
            groups[f"{c}={v}"] = g.index
    J["human"] = {g: {k: q(hu.loc[ix, k]) for k in QTY} for g, ix in groups.items()}
    J["rsg"] = {g: {k: q(rs.loc[ix, k]) for k in QTY} for g, ix in groups.items()}
    done_push = T[T.done & ~T.ident]
    sets = {"p3 all": T, "p3 done": T[T.done], "p3 fail": T[~T.done], "p3 done identity": T[T.done & T.ident], "p3 done pushed": done_push,
            "p3 fail pushed": T[~T.done & ~T.ident], "p3 fail identity": T[~T.done & T.ident],
            "p3 done curvature push": done_push[done_push.tf.str.endswith("s1")], "p3 done speed push": done_push[~done_push.tf.str.endswith("s1")]}
    J["p3"] = {nm: dict(n=int(len(x)), whole={k: q(x[k]) for k in QTY}, push={k: q(x[f"w_{k}"]) for k in ("kmax", "alat", "dk", "dec")},
                        plan={k: q(x[f"plan_{k}"]) for k in ("kmax", "alat", "dk")}, raw={k: q(x[f"raw_{k}"]) for k in ("kmax", "alat")})
               for nm, x in sets.items()}
    # human band per topo class (p5, p95) and the paired read on completing branches
    band = {c: {k: (float(np.nanpercentile(g[k], 5)), float(np.nanpercentile(g[k], 95))) for k in ("kmax", "alat", "dk")} for c, g in hu.groupby("topo")}
    allb = {k: (float(np.nanpercentile(hu[k], 5)), float(np.nanpercentile(hu[k], 95))) for k in ("kmax", "alat", "dk")}
    D = T[T.done].copy()
    for k in ("kmax", "alat", "dk"):
        D[f"in_{k}"] = [allb[k][0] <= v <= allb[k][1] if (band.get(c) is None or (hu.topo == c).sum() < 5) else band[c][k][0] <= v <= band[c][k][1]
                        for v, c in zip(D[k], D.topo)]
        D[f"steeper_{k}"] = D[f"r_{k}"] > 1.0
    D["in_all3"] = D.in_kmax & D.in_alat & D.in_dk
    D["steeper_any"] = D.steeper_kmax | D.steeper_alat | D.steeper_dk
    J["band_topo"], J["band_all"] = band, allb
    J["paired"] = dict(n=int(len(D)), n_pushed=int((~D.ident).sum()),
                       **{f"in_{k}": int(D[f"in_{k}"].sum()) for k in ("kmax", "alat", "dk", "all3")},
                       **{f"steeper_{k}": int(D[f"steeper_{k}"].sum()) for k in ("kmax", "alat", "dk", "any")},
                       **{f"pushed_in_{k}": int(D[~D.ident][f"in_{k}"].sum()) for k in ("kmax", "alat", "dk", "all3")},
                       **{f"pushed_steeper_{k}": int(D[~D.ident][f"steeper_{k}"].sum()) for k in ("kmax", "alat", "dk", "any")},
                       ratio_median={k: float(D[f"r_{k}"].median()) for k in ("kmax", "alat", "dk")},
                       ratio_median_pushed={k: float(D[~D.ident][f"r_{k}"].median()) for k in ("kmax", "alat", "dk")})
    # gate: class p95 (all-turn p95 for classes with < 5 turns) on kappa and dkappa, a_lat cap
    def p95(c, k):
        return band[c][k][1] if c in band and (hu.topo == c).sum() >= 5 else allb[k][1]
    D["gate_k"] = [v <= p95(c, "kmax") for v, c in zip(D.kmax, D.topo)]
    D["gate_a"] = D.alat <= ALAT_CAP
    D["gate_dk"] = [v <= p95(c, "dk") for v, c in zip(D.dk, D.topo)]
    D["gate"] = D.gate_k & D.gate_a & D.gate_dk
    J["gate_rule"] = dict(n=int(len(D)), keep=int(D.gate.sum()), keep_pushed=int(D[~D.ident].gate.sum()), n_pushed=int((~D.ident).sum()),
                          fail_k=int((~D.gate_k).sum()), fail_a=int((~D.gate_a).sum()), fail_dk=int((~D.gate_dk).sum()),
                          kept_scenes=sorted(D[D.gate].short.unique().tolist()), kept_pushed_scenes=sorted(D[D.gate & ~D.ident].short.unique().tolist()))
    hal = D.scene.map(hu.alat)
    var = {"A: kmax<=p95c & dk<=p95c & alat<=3.3": D.gate,
           "B: kmax<=p95c & dk<=p95c & alat<=human p95 (all turns)": D.gate_k & D.gate_dk & (D.alat <= allb["alat"][1]),
           "C: kmax<=p95c & dk<=p95c & alat<=max(3.3, human same scene)": D.gate_k & D.gate_dk & (D.alat <= np.maximum(ALAT_CAP, hal)),
           "D: kmax<=p95c & dk<=p95c (geometry only)": D.gate_k & D.gate_dk}
    J["gate_variants"] = {k: dict(keep=int(v.sum()), keep_pushed=int(v[~D.ident].sum()), scenes=sorted(D[v].short.unique().tolist()),
                                  pushed_scenes=sorted(D[v & ~D.ident].short.unique().tolist()),
                                  new_scenes=sorted(set(D[v & ~D.ident].short) - set(D[D.ident].short))) for k, v in var.items()}
    J["paired"]["steeper_tol25"] = {k: int((D[f"r_{k}"] > 1.25).sum()) for k in ("kmax", "alat", "dk")}
    J["paired"]["steeper_tol25_pushed"] = {k: int((D[~D.ident][f"r_{k}"] > 1.25).sum()) for k in ("kmax", "alat", "dk")}
    J["paired"]["steeper_tol25_any"] = int(((D.r_kmax > 1.25) | (D.r_alat > 1.25) | (D.r_dk > 1.25)).sum())
    J["paired"]["vapk_ratio_median"] = float((D.v_apk / D.scene.map(hu.v_apk)).median())
    J["paired"]["st_kpk_median"] = dict(p3_done=float(D.st_kpk.median()), human_same=float(D.scene.map(hu.st_kpk).median()))
    sm = T[T.done & ~T.ident].copy()
    sm["cut"] = np.where(sm.tf.str.endswith("s1"), "curvature (s = 1)", "speed (s < 1)")
    J["smoothing"] = {c: dict(n=int(len(g)), realized_over_plan_kmax=q(g.w_kmax / g.plan_kmax), plan_over_raw_kmax=q(g.plan_kmax / g.raw_kmax),
                              whole_over_plan_kmax=q(g.kmax / g.plan_kmax)) for c, g in sm.groupby(["ho", "cut"]) for c in [" ".join(c)]}
    J["entry_targets"] = {c: {d: dict(median=float(np.nanmedian(g[f"v_{d}"])), p25=float(np.nanpercentile(g[f"v_{d}"], 25)),
                                      n=int(g[f"v_{d}"].notna().sum())) for d in (40, 20, 10, 0)} for c, g in [("all", hu)] + list(hu.groupby("topo"))}
    out.mkdir(parents=True, exist_ok=True)
    H.to_csv(out / "shape_turns.csv", index=False)
    cols = ["arm", "short", "side", "topo", "done", "ident", "score", "zero", "s_ho", "kmax", "alat", "dk", "dec", "st_kpk", "v_apk", "w_kmax", "w_alat", "w_dk", "w_dec",
            "plan_kmax", "plan_alat", "raw_kmax", "v_0", "v_min", "r_kmax", "r_alat", "r_dk"]
    T[cols].to_csv(out / "shape_p3.csv", index=False, float_format="%.4f")
    D["gate_k"], D["gate_dk"] = D.gate_k.astype(bool), D.gate_dk.astype(bool)
    J["done_rows"] = D[cols[:4] + ["kmax", "alat", "dk", "plan_kmax", "plan_alat", "w_kmax", "r_kmax", "r_alat", "r_dk", "in_all3", "steeper_any",
                                   "gate_k", "gate_a", "gate_dk", "gate"]].to_dict("records")
    J["done_rows"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in J["done_rows"]]
    for r in J["done_rows"]:
        r["human_kmax"], r["human_alat"], r["human_dk"] = (round(float(hu.loc[[s for s in hu.index if s[7:15] == r["short"]][0], k]), 4) for k in ("kmax", "alat", "dk"))
    (out / "shape.json").write_text(json.dumps(J, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
    tables(J, out / "shape_tables.md")
    fig(hu, rs, T, band, allb, Path(a.fig), routes)
    print(json.dumps(dict(gate=J.get("gate"), paired=J["paired"], gate_rule=J["gate_rule"], counts=J["counts"]), indent=1))


def tables(J, path):
    f = lambda d, p: "-" if d is None or d.get(f"p{p}") is None else f"{d[f'p{p}']:.3g}"  # noqa: E731
    L = ["# Turn shape tables (rf_shape.py report)", "", "Units: kmax 1/m, alat m/s^2, dk 1/m^2, dec m/s^2 (positive = braking), v m/s, dec_start m from the entry.", ""]
    for who in ("human", "rsg"):
        for g, d in J[who].items():
            L += [f"## {who}, {g}", "", "| quantity | n | p5 | p25 | p50 | p75 | p95 |", "|:--|--:|--:|--:|--:|--:|--:|"]
            L += [f"| {k} | {d[k]['n']} | " + " | ".join(f(d[k], p) for p in QS) + " |" for k in QTY]
            L.append("")
    for nm, d in J["p3"].items():
        for part in ("whole", "push", "plan", "raw"):
            L += [f"## {nm} (n {d['n']}), {part}", "", "| quantity | n | p5 | p25 | p50 | p75 | p95 |", "|:--|--:|--:|--:|--:|--:|--:|"]
            L += [f"| {k} | {v['n']} | " + " | ".join(f(v, p) for p in QS) + " |" for k, v in d[part].items()]
            L.append("")
    Path(path).write_text("\n".join(L) + "\n")


def fig(hu, rs, T, band, allb, path, routes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.4))
    D = T[T.done]
    cur = D.tf.str.endswith("s1") & ~D.ident
    K = [("kmax", "peak curvature |kappa| (1/m)"), ("alat", "peak lateral accel v^2 kappa (m/s^2)"),
         ("dk", "peak curvature rate |dkappa/ds| (1/m^2)"), ("v_apk", "speed at the peak lateral accel (m/s)")]
    for ax, (k, lab) in zip(axs, K):
        x = hu[k].reindex(D.scene).to_numpy()
        y = D[k].to_numpy()
        if k != "v_apk":
            ax.axhspan(*allb[k], color="0.9", zorder=0, label="human p5-p95 (36 turns)")
        hi = np.nanmax(np.r_[x, y]) * 1.08
        ax.plot([0, hi], [0, hi], color="0.5", lw=1, ls="--", zorder=1, label="equal to the human")
        for m, mk, c, nm in ((D.ident.to_numpy(), "o", "tab:blue", "P3 done, identity (policy alone)"),
                             (cur.to_numpy(), "^", "tab:red", "P3 done, curvature push (s = 1)"),
                             ((~D.ident & ~cur).to_numpy(), "D", "tab:purple", "P3 done, speed push (s < 1)")):
            ax.scatter(x[m], y[m], marker=mk, color=c, s=48, zorder=3, label=nm, edgecolor="k", lw=0.5)
        if k in ("kmax", "alat"):
            m = cur.to_numpy()
            pv = D[f"plan_{k}"].to_numpy()
            ax.scatter(x[m], pv[m], marker="x", color="tab:orange", s=44, zorder=2, label="served pushed plan, window peak")
            for xi, a_, b_ in zip(x[m], y[m], pv[m]):
                ax.plot([xi, xi], [a_, b_], color="tab:orange", lw=0.7, zorder=1)
        sc = D.scene.unique()
        ax.scatter(hu[k].reindex(sc).to_numpy(), rs[k].reindex(sc).to_numpy(), marker="s", facecolor="none", edgecolor="tab:green", s=48,
                   zorder=2, label="RSG driving alone from the start")
        if k == "alat":
            ax.axhline(ALAT_CAP, color="k", lw=1, ls=":", label=f"a_lat {ALAT_CAP} m/s^2")
        ax.set_xlabel(f"human, same turn: {lab}", fontsize=9)
        ax.set_ylabel(f"rollout: {lab}", fontsize=9)
        ax.set_title(lab, fontsize=10)
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"P3 completing branches ({len(D)} cells, 10 scenes) vs the human log on the same turn; window entry - 40 m .. exit, cut at the "
                 "first zero event", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print("figure ->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract-p3")
    e.add_argument("--arms", required=True)
    e.add_argument("--out", required=True)
    r = sp.add_parser("report")
    r.add_argument("--r281", required=True)
    r.add_argument("--p3", required=True)
    r.add_argument("--read", required=True)
    r.add_argument("--exam", required=True)
    r.add_argument("--pf", default="", help="rf_pai_follow scenes.csv (reproduction gate)")
    r.add_argument("--out", required=True)
    r.add_argument("--fig", required=True)
    a = ap.parse_args()
    extract_p3(a) if a.cmd == "extract-p3" else report(a)
