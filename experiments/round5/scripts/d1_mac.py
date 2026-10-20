#!/usr/bin/env python3
"""ROUND5 D1 read-outs (Mac, stored data only; inputs pulled from the box into tmp/round5/: facts/, tax/, dec/, mapcov_train.csv).

  budget     points lost per class and read (PAI 281 ACC + GAP: NTV0 / R2C / R2J; PAI-FIX native: NTV0 / R2C / R2J / R3JW / R3JNF) and the
             per-scene best-of-stored-reads ceiling (calib's 21 stored 281 reads + R2C / R2J 281 + the five PAI-FIX native T281 reads)
  offroad    every lateral zero (offroad, cor_drift, lane change) of NTV0 / R2J 281 ACC + GAP: geometry class from the rollout vs the human log
  launch     R2J vs R2C (281 ACC + GAP, PAI-FIX G-LAUNCH native): plan arc / served speed from the human's launch over 4 s, the follower's gap
  overshoot  J family overshoot zeros: does the model plan ever contain a stop, braking onset vs the human, speed at the human's rest point
  all        the four, then results/d1_*.csv and the tables of results/d1_budget.md (between the generated markers)

1 point = 1 scene. Classes = pai_fix.py fclass (overshoot / route-turn / offroad / AF-front / side / zero-other; rear-hit / slow / hit).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts"), str(REPO)]
import pai_fix as PF  # noqa: E402

T = REPO / "tmp/round5"
RES = REPO / "experiments/round5/results"
CLS = PF.ZCLS + PF.PCLS
R281 = {"NTV0": "NTV0-L18-s0-281ag", "R2C": "R2C-L18-s0-281ag", "R2J": "R2J-L18-s0-281ag"}
RFIX = {"NTV0": "NTV0-L18-s0-vl", "R2C": "R2C-L18-s0-vl", "R2J": "R2J-L18-s0-vl", "R3JW": "R3JW-L18-s0-vl", "R3JNF": "R3JNF-L18-s0-vl"}
LAT = ("cor_drift", "cor_lane_change")      # lateral zero-other sub-types, read with offroad


def facts(n):
    F = pd.read_csv(T / "facts" / f"{n}.csv").set_index("scene")
    return F[~F.rpc]


def loss_cls(F):
    """per scene: (class, points lost) with partial points = 1 - score."""
    return pd.DataFrame(dict(cls=F.fclass, pts=1.0 - F.score.clip(0, 1)))


# ---------------------------------------------------------------- 1. budget
def budget():
    cal = pd.read_csv(REPO / "experiments/pai_fix/results/calib/calib_scores.csv.gz", index_col=0)
    cal = cal.drop(columns=[c for c in ("subset", "a0_fail") if c in cal])
    F = {k: facts(v) for k, v in R281.items()}
    common = sorted(set.intersection(*[set(x.index) for x in F.values()]))
    ex = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv").set_index("scene")
    FX = {k: facts(v) for k, v in RFIX.items()}
    # stored scores per 281 scene: calib (21 reads, 270 scenes), R2C / R2J 281 ACC + GAP, PAI-FIX native T281 scenes
    S = pd.DataFrame({f"{k} 281ag": F[k].score.reindex(common) for k in F})
    S = S.join(cal.reindex(common), how="left")
    for k, x in FX.items():
        S[f"{k} fix-vl"] = x.score.reindex(common)
    ag_cols = [c for c in S if c.endswith("281ag")] + [c for c in list(PF.CAL) + ["NTV0-s0 rerun (R0)", "NTV0-s0-hl"] if c in S]
    best_all, best_ag = S.max(axis=1), S[ag_cols].max(axis=1)
    who = S.idxmax(axis=1)
    rows, per = [], []
    for k, x in F.items():
        L = loss_cls(x.reindex(common))
        for c in CLS:
            m = L.cls == c
            lost = L.pts[m].sum()
            reach = (best_all[m] - x.score.reindex(common)[m].clip(0, 1)).clip(lower=0).sum()
            reach_ag = (best_ag[m] - x.score.reindex(common)[m].clip(0, 1)).clip(lower=0).sum()
            rows.append(dict(read=f"{k} 281 AG", cls=c, scenes=int(m.sum()), lost=lost, reach_any=reach, reach_ag=reach_ag, never=lost - reach))
        rows.append(dict(read=f"{k} 281 AG", cls="all", scenes=int((L.pts > 0.02).sum()), lost=L.pts.sum(),
                         reach_any=(best_all - x.score.reindex(common)).clip(lower=0).sum(),
                         reach_ag=(best_ag - x.score.reindex(common)).clip(lower=0).sum(), never=(1 - best_all).sum()))
    B = pd.DataFrame(rows)
    # PAI-FIX native: subsets
    fx_rows = []
    sub = ex["subset"]
    for k, x in FX.items():
        for grp, sel in (("TARGET", sub.isin(["STOP", "STOP-L", "TURN"])), ("G-LAUNCH", sub == "G-LAUNCH"), ("G-LEAD", sub == "G-LEAD"),
                         ("T281", sub.str.startswith("T281"))):
            sc = sorted(set(sel.index[sel]) & set(x.index))
            L = loss_cls(x.loc[sc])
            r = dict(read=f"{k} fix VL", subset=grp, n=len(sc), mean=x.loc[sc].score.mean(), lost=L.pts.sum())
            for c in CLS:
                r[c] = L.pts[L.cls == c].sum()
            fx_rows.append(r)
    FXB = pd.DataFrame(fx_rows)
    # ceiling counts
    n = len(common)
    ceil = dict(n=n, reads=int(S.notna().sum(axis=1).median()), solved_any=int((best_all >= PF.OK).sum()), partial_any=int(((best_all > 0) & (best_all < PF.OK)).sum()),
                zero_all=int((best_all == 0).sum()), mean_best_all=float(best_all.mean()), mean_best_ag=float(best_ag.mean()),
                solved_ag=int((best_ag >= PF.OK).sum()), zero_ag=int((best_ag == 0).sum()))
    per = pd.DataFrame(dict(best_all=best_all, best_ag=best_ag, best_read=who, **{f"{k}_score": F[k].score.reindex(common) for k in F},
                            **{f"{k}_cls": F[k].fclass.reindex(common) for k in F},
                            **{f"{k}_primary": F[k].primary.reindex(common) for k in F}))
    return B, FXB, ceil, per


def md_budget(B, FXB, ceil, per):
    L = ["### 1a. PAI 281 under ACC + GAP: points lost per class (1 point = 1 scene; partials count 1 - score)", "",
         f"{ceil['n']} scenes all three reads score without an RPC error. `reach any` = points that the per-scene best of the stored reads "
         f"(median {ceil['reads']} reads a scene: calib's 21 stored 281 reads incl. 6 native / rerun rows, R2C / R2J 281 AG, PAI-FIX native on the T281 scenes) "
         "gets above this read; `reach AG` = the same over ACC + GAP reads only (the submission serve); `never` = lost - reach any (no stored read gets it).", "",
         "| class | " + " | ".join(f"{r} scenes / lost / reach any / reach AG / never" for r in B.read.unique()) + " |",
         "|---|" + "--:|" * B.read.nunique()]
    for c in CLS + ["all"]:
        cells = []
        for r in B.read.unique():
            x = B[(B.read == r) & (B.cls == c)].iloc[0]
            cells.append(f"{x.scenes} / {x.lost:.1f} / {x.reach_any:.1f} / {x.reach_ag:.1f} / {x.never:.1f}")
        L.append(f"| {'**all**' if c == 'all' else c} | " + " | ".join(cells) + " |")
    L += ["", f"Ceiling: of {ceil['n']} scenes, some stored read scores >= 0.98 on {ceil['solved_any']}, only partially on {ceil['partial_any']}, "
          f"and **no stored read scores at all on {ceil['zero_all']}**; per-scene best-of mean {ceil['mean_best_all']:.3f} (ACC + GAP reads only "
          f"{ceil['mean_best_ag']:.3f}: solved {ceil['solved_ag']}, zero in every read {ceil['zero_ag']}). The human log scores 1.000 on 277 / 277 (d381).", ""]
    # ranking of NTV0's classes by reachable vs never
    x = B[(B.read == "NTV0 281 AG") & (B.cls != "all")].sort_values("lost", ascending=False)
    L += ["### 1b. NTV0-L18-s0 + ACC + GAP (the PAI candidate), classes ranked by points lost: reachable by some stored checkpoint vs never", "",
          "| rank | class | lost | reachable (any read) | reachable (AG reads) | never got by any read | scenes where some read scores >= 0.98 |", "|--:|---|--:|--:|--:|--:|--:|"]
    for i, r in enumerate(x.itertuples(), 1):
        m = per.NTV0_cls == r.cls
        L.append(f"| {i} | {r.cls} | {r.lost:.1f} | {r.reach_any:.1f} | {r.reach_ag:.1f} | {r.never:.1f} | {int((per.best_all[m] >= PF.OK).sum())} / {int(m.sum())} |")
    L += ["", "### 1c. PAI-FIX native (VC + LEAD) reads: points lost per class and subset", "",
          "| read | subset | n | mean | lost | " + " | ".join(CLS) + " |", "|---|---|--:|--:|--:|" + "--:|" * len(CLS)]
    for r in FXB.itertuples():
        L.append(f"| {r.read} | {r.subset} | {r.n} | {r.mean:.3f} | {r.lost:.1f} | " + " | ".join(f"{FXB.loc[r.Index, c]:.1f}" for c in CLS) + " |")
    return L


# ---------------------------------------------------------------- shared geometry
def load_tax(n):
    return {o["scene"]: o for o in pickle.load(open(T / "tax" / f"{n}.pkl", "rb"))["rolls"] if "ctrl" in o}


def load_dec(n):
    p = T / "dec" / f"dec_{n}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else {}


def arclen(P):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(P, float)[:, :2], axis=0).T))]


def project(Pl, Sl, Yl, x, y):
    """station and signed lateral (left +) of point (x, y) on the polyline Pl (nearest segment vertex, refined; the first / last moving
    segment extended, so stations before the start / past the end of the log are negative / beyond its length)."""
    keep = np.r_[True, np.diff(Sl) > 0.05]
    Pl, Sl = Pl[keep], Sl[keep]
    d = np.hypot(Pl[:, 0] - x, Pl[:, 1] - y)
    i = int(np.argmin(d))
    j = min(max(i, 0), len(Pl) - 2)
    a, b = Pl[j], Pl[j + 1]
    v = b - a
    L2 = max(float(v @ v), 1e-9)
    u = float(((x, y) - a) @ v / L2)
    u = float(np.clip(u, -np.inf if j == 0 else 0, np.inf if j == len(Pl) - 2 else 1))   # extend the first / last segment
    q = a + u * v
    s = Sl[j] + u * np.sqrt(L2)
    lat = float(np.cross(v / np.sqrt(L2), np.array([x, y]) - q))
    return s, lat


def plan_world(r, k, anchor_xyh):
    """model plan of decision k in the world frame given the anchor pose (x, y, yaw) of that decision."""
    P = r["pm"][k]
    c, s = np.cos(anchor_xyh[2]), np.sin(anchor_xyh[2])
    return np.c_[anchor_xyh[0] + c * P[:, 0] - s * P[:, 1], anchor_xyh[1] + s * P[:, 0] + c * P[:, 1]]


def ego_at(C, t):
    return np.array([np.interp(t, C["t"], C["x"]), np.interp(t, C["t"], C["y"]), np.interp(t, C["t"], np.unwrap(C["yaw"])),
                     np.interp(t, C["t"], np.hypot(C["vx"], C["vy"]))])


def first_event(o, key):
    s = o["ser"].get(key)
    if s is None:
        return np.nan
    s = np.asarray(s, float)
    m = np.nonzero(s[:, 1] > 0.5)[0]
    return float(s[m[0], 0]) if len(m) else np.nan


# ---------------------------------------------------------------- 2. offroad / lateral anatomy
def _turn_deg(Yl, Sl, s0, s1):
    a, b = np.interp([s0, s1], Sl, Yl)
    return float(np.degrees(b - a))


def lane_dist(o, pts, yaws):
    """distance of each (x, y) to the nearest lane centre segment whose direction is within 35 deg of yaw (fig lanes of the extract)."""
    fg = o.get("fig") or {}
    segs = [(np.asarray(c, float)[:-1], np.asarray(c, float)[1:]) for c in fg.get("lanes", []) if len(c) >= 2]
    if not segs:
        return np.full(len(pts), np.nan)
    A = np.concatenate([a for a, _ in segs])
    B = np.concatenate([b for _, b in segs])
    V = B - A
    L2 = np.maximum((V ** 2).sum(1), 1e-9)
    hd = np.arctan2(V[:, 1], V[:, 0])
    out = []
    for (x, y), yw in zip(pts, yaws):
        u = np.clip(((x - A[:, 0]) * V[:, 0] + (y - A[:, 1]) * V[:, 1]) / L2, 0, 1)
        dd = np.hypot(A[:, 0] + u * V[:, 0] - x, A[:, 1] + u * V[:, 1] - y)
        ok = np.abs((hd - yw + np.pi) % (2 * np.pi) - np.pi) < np.radians(35)
        out.append(float(dd[ok].min()) if ok.any() else np.nan)
    return np.array(out)


def lateral_one(o, d, fclass, primary, t_ev):
    C, Lg = o["ctrl"], o["log"]
    Pl, Yl = np.asarray(Lg["P"], float), np.asarray(Lg["Y"], float)
    Sl = arclen(Pl)
    t0 = float(C["t"][0])
    # event time: first offroad sample, else the scorer's event time (facts t_ev_s after the start), else the end
    te = first_event(o, "offroad")
    if not np.isfinite(te):
        te = first_event(o, "left_corridor_laterally")
    if not np.isfinite(te):
        te = t0 + (t_ev * 1e6 if np.isfinite(t_ev) else (C["t"][-1] - t0))
    ts = np.asarray(C["t"], float)
    ts = ts[(ts <= te)]
    st, la = np.array([project(Pl, Sl, Yl, *ego_at(C, t)[:2]) for t in ts]).T
    e = ego_at(C, te)
    s_e, lat_e = project(Pl, Sl, Yl, e[0], e[1])
    # human at the same station: time and speed
    th = float(np.interp(s_e, Sl, Lg["T"]))
    vh = float(np.interp(th, Lg["T"][1:], np.hypot(*np.diff(Pl, axis=0).T) / np.maximum(np.diff(Lg["T"]) * 1e-6, 1e-3)))
    turn_here = _turn_deg(Yl, Sl, s_e - 20, s_e + 10)            # the log's heading change over the 30 m around the event station
    # time lag: ego reaches station s_e at te; the human at th (both from clip start)
    lag = (te - t0) * 1e-6 - (th - Lg["T"][0]) * 1e-6
    # lateral history: when did |lat| first exceed 0.5 m, and the growth over the last 4 s
    w4 = ts >= te - 4e6
    lat_on = ts[np.argmax(np.abs(la) > 0.5)] if (np.abs(la) > 0.5).any() else te
    stood = bool(((np.hypot(np.interp(ts, C["t"], C["vx"]), np.interp(ts, C["t"], C["vy"])) < 0.3) & (ts < te - 2e6)).any())
    rear = first_event(o, "collision_rear")
    j = int(np.argmin(np.abs(np.asarray(o["tg"]) - te)))
    junc = bool(np.asarray(o["junc"])[max(0, j - 4):j + 1].any())
    side = np.sign(lat_e)
    turn_sign = np.sign(turn_here) if abs(turn_here) >= 15 else 0
    # class
    if abs(turn_here) >= 25:
        cls = "turn wide" if side == -turn_sign else "turn cut"
        cls = ("junction " if junc else "curve ") + cls
    elif abs(turn_here) >= 10:
        cls = "bend " + ("wide" if side == -turn_sign else "cut")
    else:
        cls = "straight drift"
        if np.abs(la[w4]).size and (np.abs(la[w4]).max() - np.abs(la[w4]).min()) > 2.5 and (te - lat_on) < 3e6:
            cls = "straight swerve (< 3 s)"
    if primary == "cor_lane_change":
        cls = "lane change"
    # whose lane change: over the 10 s before the event, max distance of ego / human to the nearest aligned lane centre (> 1.2 m = crossed a
    # lane line, a lane is about 3.5 m), and the ego's distance at the event (small = it sits in a lane, that is in another lane than the human)
    tw = np.arange(max(te - 10e6, t0), te + 1, 0.5e6)
    E = np.array([ego_at(C, t) for t in tw])
    Th = np.clip(tw, Lg["T"][0], Lg["T"][-1])
    Hx, Hy, Hyaw = np.interp(Th, Lg["T"], Pl[:, 0]), np.interp(Th, Lg["T"], Pl[:, 1]), np.interp(Th, Lg["T"], Yl)
    de, dh = lane_dist(o, E[:, :2], E[:, 2]), lane_dist(o, np.c_[Hx, Hy], Hyaw)
    lanes = dict(ego_lane_max=float(np.nanmax(de)) if np.isfinite(de).any() else np.nan, ego_lane_end=float(de[-1]),
                 hum_lane_max=float(np.nanmax(dh)) if np.isfinite(dh).any() else np.nan)
    if cls.startswith("straight") and abs(lat_e) >= 3:
        hx, ex_ = lanes["hum_lane_max"] > 1.2, lanes["ego_lane_max"] > 1.2
        cls = "straight: " + ("human changed lane, ego kept" if hx and not ex_ else "ego left its lane" if ex_ and not hx else
                              "both crossed" if hx and ex_ else "parallel lane, no crossing seen")
    elif cls.startswith("straight"):
        cls = "straight: edge (|lat| < 3 m)"
    ctx = []
    if stood:
        ctx.append("stood before")
    if np.isfinite(rear) and rear < te:
        ctx.append("after rear-hit")
    if lag > 3:
        ctx.append(f"late (+{lag:.0f} s)")
    # plan at the decision 2-3 s before the event: lateral offset of its 4 s end from the log, plan speed vs human
    pl = dict(plan_lat4=np.nan, plan_v=np.nan, plan_heading_err=np.nan)
    if d is not None and len(d["t0"]):
        k = int(np.searchsorted(d["t0"], te - 2.5e6))
        k = min(max(k, 0), len(d["t0"]) - 1)
        tk = float(d["t0"][k])
        a = ego_at(C, tk)
        W = plan_world(d, k, a)
        s4, lat4 = project(Pl, Sl, Yl, *W[-1])
        s_now, lat_now = project(Pl, Sl, Yl, a[0], a[1])
        hdg = float(np.degrees(np.arctan2(*(W[-1] - W[-3])[::-1]) - np.interp(s4, Sl, Yl)))
        pl = dict(plan_lat4=lat4, plan_dlat=lat4 - lat_now, plan_v=float(arclen(np.r_[[[0, 0]], d["pm"][k][:, :2]])[-1] / 4),
                  ego_v=float(d["v0"][k]), plan_heading_err=(hdg + 180) % 360 - 180, plan_ds_vs_human=s4 - float(np.interp(np.interp(s_now, Sl, Lg["T"]) + 4e6, Lg["T"], Sl)))
    return dict(cls=cls, ctx="+".join(ctx) or "-", t_ev=(te - t0) * 1e-6, lat_ev=lat_e, turn_deg=turn_here, junc=junc, v_ego=e[3], v_human=vh,
                v_ratio=e[3] / max(vh, 0.5), lag_s=lag, lat_grow_s=(te - lat_on) * 1e-6, fclass=fclass, primary=primary, **lanes, **pl)


def offroad():
    out = []
    for k in ("NTV0", "R2C", "R2J"):
        F = facts(R281[k])
        A, D = load_tax(R281[k]), load_dec(f"{k}-281ag")
        m = (F.fclass == "offroad") | ((F.fclass == "zero-other") & F.primary.isin(LAT))
        for s in F.index[m]:
            if s not in A:
                continue
            r = lateral_one(A[s], D.get(s), F.fclass[s], F.primary[s], F.t_ev_s[s])
            out.append(dict(read=k, scene=s, **r))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 3. launch anatomy
HAND = 0.5e6


def follower(o, t, ex, ey, eyaw):
    """nearest vehicle behind the ego (rear axle frame) within 1.8 m lateral: (gap to its centre, its speed)."""
    j = int(np.argmin(np.abs(np.asarray(o["tg"]) - t)))
    tr = o["traffic"][j]
    if not len(tr["id"]):
        return np.nan, np.nan
    c, s = np.cos(eyaw), np.sin(eyaw)
    dx, dy = np.asarray(tr["x"]) - ex, np.asarray(tr["y"]) - ey
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    veh = np.array([("bicycle" not in str(k)) and ("pedestrian" not in str(k)) for k in tr["cls"]])
    q = (lx < 0) & (lx > -40) & (np.abs(ly) < 1.8) & veh
    if not q.any():
        return np.nan, np.nan
    i = np.argmax(np.where(q, lx, -1e9))
    L = np.asarray(tr["L"])[i] if "L" in tr else 4.5
    return float(-lx[i] - L / 2 - 1.2), float(np.asarray(tr["v"])[i])  # bumper gap approx (ego rear overhang ~1.2 m)


def launch_one(o, d):
    C, Lg = o["ctrl"], o["log"]
    Pl, Tl = np.asarray(Lg["P"], float), np.asarray(Lg["T"], float)
    Sl = arclen(Pl)
    th = float(C["t"][0]) + HAND
    ph = np.array([np.interp(th, Tl, Pl[:, 0]), np.interp(th, Tl, Pl[:, 1])])
    dd = np.hypot(*(Pl - ph).T)
    m = np.nonzero((Tl >= th) & (dd > 0.5))[0]
    if not len(m):
        return None
    tL = float(Tl[m[0]])
    sH = lambda t: float(np.interp(t, Tl, Sl))  # noqa: E731
    out = dict(t_L=(tL - th) * 1e-6)
    eh = ego_at(C, th)
    se0 = project(Pl, Sl, Lg["Y"], eh[0], eh[1])[0]
    for h in (1, 2, 3, 4):
        t = tL + h * 1e6
        e = ego_at(C, t)
        out[f"h_d{h}"] = sH(t) - sH(tL)                                 # human distance since its launch
        out[f"e_d{h}"] = project(Pl, Sl, Lg["Y"], e[0], e[1])[0] - max(se0, sH(tL) - 0.5)  # ego distance along the log from its start pose
        out[f"e_v{h}"] = e[3]
        out[f"h_v{h}"] = (sH(t + 0.25e6) - sH(t - 0.25e6)) / 0.5
    # ego move time (0.5 m past its handoff pose)
    ex, ey = np.asarray(C["x"], float), np.asarray(C["y"], float)
    me = np.nonzero((C["t"] >= th) & (np.hypot(ex - eh[0], ey - eh[1]) > 0.5))[0]
    out["ego_move"] = (C["t"][me[0]] - tL) * 1e-6 if len(me) else np.nan
    # plans: arc of the model plan and the served plan at the decision nearest t_L + h, and the ACC set / v_model
    if d is not None and len(d["t0"]):
        for h in (0, 1, 2, 3, 4):
            k = int(np.searchsorted(d["t0"], tL + h * 1e6))
            if k >= len(d["t0"]):
                continue
            out[f"pm_arc{h}"] = float(arclen(np.r_[[[0, 0]], d["pm"][k][:, :2]])[-1])
            out[f"ps_arc{h}"] = float(arclen(np.r_[[[0, 0]], d["ps"][k][:, :2]])[-1])
            out[f"cut{h}"] = float(d["cut"][k])
        w = (d["t0"] >= tL) & (d["t0"] <= tL + 4e6)
        if w.any():
            A = np.array([arclen(np.r_[[[0, 0]], p[:, :2]]) for p in d["pm"][w]])          # (n, 9): arc at 0, 0.5 .. 4 s
            out |= dict(win_pm_a1=float(np.median(A[:, 2])), win_pm_a2=float(np.median(A[:, 4])), win_pm_a4=float(np.median(A[:, 8])),
                        win_v0=float(np.median(d["v0"][w])),
                        # "deferred start": standing, the plan moves < 0.3 m in its first 1 s but >= 2 m by 4 s (go later, re-planned every 0.1 s)
                        win_still=float((d["v0"][w] < 0.3).mean()),
                        win_defer=float(((d["v0"][w] < 0.3) & (A[:, 2] < 0.3) & (A[:, 8] >= 2)).mean()))
        out["won_mode"] = Counter(d["won"][w]).most_common(1)[0][0] if w.any() else ""
        out["cut_share"] = float((d["cut"][w] > 0.3).mean()) if w.any() else np.nan
        # first decision after t_L - 1 s whose model plan covers >= 2 m in 4 s
        g = np.nonzero((d["t0"] >= tL - 1e6) & np.array([arclen(np.r_[[[0, 0]], p[:, :2]])[-1] >= 2.0 for p in d["pm"]]))[0]
        out["plan_go"] = (d["t0"][g[0]] - tL) * 1e-6 if len(g) else np.nan
    # follower gap at t_L and its minimum over 4 s; the rear-hit time
    gaps = []
    for h in np.arange(0, 4.01, 0.5):
        t = tL + h * 1e6
        e = ego_at(C, t)
        gaps.append(follower(o, t, e[0], e[1], e[2]))
    g = np.array(gaps)
    out["fol_gap0"], out["fol_v0"] = g[0]
    out["fol_gap_min4"] = np.nanmin(g[:, 0]) if np.isfinite(g[:, 0]).any() else np.nan
    rr = first_event(o, "collision_rear")
    out["rear_t"] = (rr - tL) * 1e-6 if np.isfinite(rr) else np.nan
    if np.isfinite(rr):
        e = ego_at(C, rr)
        out["v_at_rear"] = e[3]
        out["h_v_at_rear"] = (sH(rr + 0.25e6) - sH(rr - 0.25e6)) / 0.5
        out["e_d_at_rear"] = project(Pl, Sl, Lg["Y"], e[0], e[1])[0] - max(se0, sH(tL) - 0.5)
        out["h_d_at_rear"] = sH(rr) - sH(tL)
    return out


def launch():
    rows = []
    ex = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv").set_index("scene")
    gl = set(ex.index[ex.subset == "G-LAUNCH"])
    spec = [(k, R281[k], f"{k}-281ag", "281") for k in ("NTV0", "R2C", "R2J")] + \
           [(k, RFIX[k], f"{k}-vl", "fix") for k in ("NTV0", "R2C", "R2J", "R3JW", "R3JNF")]
    for k, fn, dn, grp in spec:
        F = facts(fn)
        Lm = PF.launch_mask(F)
        sc = F.index[Lm]
        if grp == "fix":
            sc = [s for s in sc if s in gl or ex.subset.get(s, "") in ("STOP", "STOP-L", "TURN")]
        A, D = load_tax(fn), load_dec(dn)
        for s in sc:
            if s not in A:
                continue
            r = launch_one(A[s], D.get(s))
            if r is None:
                continue
            rows.append(dict(read=k, set=grp, scene=s, g_launch=s in gl, score=F.score[s], fclass=F.fclass[s], primary=F.primary[s],
                             launched=bool(PF.launched(F.loc[[s]]).iloc[0]), **r))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. overshoot anatomy
def overshoot_one(o, d):
    C, Lg = o["ctrl"], o["log"]
    Pl, Tl, Yl = np.asarray(Lg["P"], float), np.asarray(Lg["T"], float), np.asarray(Lg["Y"], float)
    Sl = arclen(Pl)
    vl = np.r_[0, np.diff(Sl) / np.maximum(np.diff(Tl) * 1e-6, 1e-3)]
    vl = np.convolve(np.r_[[vl[0]] * 5, vl, [vl[-1]] * 5], np.ones(11) / 11, "valid")          # 1 s moving average (10 Hz log)
    al = np.gradient(vl, Tl * 1e-6)
    # the human's rest station: first time the log is below 0.3 m/s and stays below 0.5 until the end of the log
    i_rest = next((i for i in range(len(vl)) if vl[i] < 0.3 and (vl[i:] < 0.5).all()), len(vl) - 1)
    s_rest, t_rest = float(Sl[i_rest]), float(Tl[i_rest])
    # the human's braking onset: the last time before rest that the smoothed deceleration is weaker than 0.4 m/s^2 (start of the final stop)
    j = i_rest
    while j > 0 and vl[j - 1] < 2.0:                                   # skip the final creep into rest
        j -= 1
    while j > 0 and al[j - 1] < -0.4:
        j -= 1
    s_hb, v_hb = float(Sl[j]), float(vl[j])
    ts = np.asarray(C["t"], float)
    st = np.array([project(Pl, Sl, Yl, x, y)[0] for x, y in zip(C["x"], C["y"])])
    ve = np.hypot(C["vx"], C["vy"])
    # ego speed when it passes the human's rest station / the human's braking station
    def at_station(sq):
        k = np.nonzero(st >= sq)[0]
        return (float(ve[k[0]]), float(ts[k[0]])) if len(k) else (np.nan, np.nan)
    v_rest, t_e_rest = at_station(s_rest)
    v_hbs, _ = at_station(s_hb)
    # ego braking onset: the first time (after passing s_hb - 40 m) ego speed falls by >= 1.5 m/s from its running max
    out = dict(s_rest=s_rest, h_v_brake=v_hb, h_brake_dist=s_rest - s_hb, h_brake_s=(t_rest - Tl[j]) * 1e-6, ego_v_at_rest_station=v_rest, ego_v_at_h_brake_station=v_hbs,
               ego_min_v_before_rest=float(ve[(st >= s_hb - 5) & (st <= s_rest)].min()) if ((st >= s_hb - 5) & (st <= s_rest)).any() else np.nan,
               ego_reach_rest=bool((st >= s_rest - 0.5).any()))
    for dq in (40, 20, 10):
        k = np.nonzero(st >= s_rest - dq)[0]
        out[f"e_v@{dq}"] = float(ve[k[0]]) if len(k) else np.nan
        out[f"h_v@{dq}"] = float(np.interp(s_rest - dq, Sl, vl)) if s_rest - dq > 0 else np.nan
        out[f"pl_end@{dq}"] = np.nan
        if d is not None and len(d["t0"]) and len(k):
            kk = min(int(np.searchsorted(d["t0"], ts[k[0]])), len(d["t0"]) - 1)
            W = plan_world(d, kk, ego_at(C, float(d["t0"][kk])))
            out[f"pl_end@{dq}"] = project(Pl, Sl, Yl, *W[-1])[0] - s_rest     # model plan's 4 s end vs the human's rest station
            out[f"pl_v4@{dq}"] = float(np.hypot(*(d["pm"][kk][-1, :2] - d["pm"][kk][-2, :2])) / 0.5)
    # plan stops: for decisions with the ego within 60 m before the rest station, does the model plan end within 4 s at < 0.5 m/s
    # (last segment < 0.25 m in 0.5 s) and where would it stop relative to the human's rest station
    if d is not None and len(d["t0"]):
        k_ok, stop_k, stop_dist, first_stop_ds, srv = 0, 0, [], np.nan, []
        brake_onset_ds = np.nan
        for k in range(len(d["t0"])):
            tk = float(d["t0"][k])
            if tk > ts[-1]:
                break
            a = ego_at(C, tk)
            s_now = project(Pl, Sl, Yl, a[0], a[1])[0]
            if not (s_rest - 60 <= s_now <= s_rest + 5):
                continue
            k_ok += 1
            P = d["pm"][k][:, :2]
            if s_now >= s_rest - 40:
                srv.append(arclen(np.r_[[[0, 0]], d["ps"][k][:, :2]])[-1] / max(arclen(np.r_[[[0, 0]], P])[-1], 0.5))
            seg = np.hypot(*np.diff(np.r_[[[0, 0]], P], axis=0).T)
            arc = float(seg.sum())
            if seg[-1] < 0.25:
                stop_k += 1
                ds = s_now + arc - s_rest
                stop_dist.append(ds)
                if not np.isfinite(first_stop_ds):
                    first_stop_ds = s_rest - s_now
            # model plan braking: plan end speed (last segment / 0.5 s) below 0.6 x current speed
            if not np.isfinite(brake_onset_ds) and d["v0"][k] > 2 and seg[-1] / 0.5 < 0.6 * d["v0"][k]:
                brake_onset_ds = s_rest - s_now
        out.update(served_over_model_40m=float(np.median(srv)) if srv else np.nan, dec_in_approach=k_ok, dec_plan_stops=stop_k, plan_stop_end_vs_rest_med=float(np.median(stop_dist)) if stop_dist else np.nan,
                   plan_first_stop_dist=first_stop_ds, plan_brake_onset_dist=brake_onset_ds)
    return out


def overshoot():
    rows = []
    spec = [(k, R281[k], f"{k}-281ag", "281") for k in ("NTV0", "R2C", "R2J")] + \
           [(k, RFIX[k], f"{k}-vl", "fix") for k in ("NTV0", "R2C", "R2J", "R3JW", "R3JNF")]
    for k, fn, dn, grp in spec:
        F = facts(fn)
        A, D = load_tax(fn), load_dec(dn)
        for s in F.index[F.fclass == "overshoot"]:
            if s not in A:
                continue
            rows.append(dict(read=k, set=grp, scene=s, rest_wl=F.rest_wl.get(s), rest_lead=F.rest_lead.get(s), **overshoot_one(A[s], D.get(s))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- markdown tables of sections 2-4 (from the csvs above)
def md(df, fmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|---|" + "--:|" * len(cols)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]) + " |")
    return out


def tables():
    L = []
    # 2. lateral
    D = pd.read_csv(RES / "d1_lateral.csv")
    t = D.groupby(["cls", "read"]).size().unstack(1).reindex(columns=["NTV0", "R2C", "R2J"]).fillna(0).astype(int)
    t.loc["all"] = t.sum()
    t.index.name = "lateral zero (offroad + cor_drift + lane change), geometry class"
    L += ["### 2a. Lateral zeros by geometry (PAI 281 ACC + GAP)", ""] + md(t) + [""]
    n = D[D.read == "NTV0"].set_index("scene")
    j = D[D.read == "R2J"].set_index("scene")
    F = {k: facts(v) for k, v in R281.items()}
    new = [s for s in j.index if s not in n.index]
    x = pd.DataFrame(dict(R2J_class=j.loc[new, "cls"], NTV0=F["NTV0"].fclass.reindex(new).values, R2C=F["R2C"].fclass.reindex(new).values,
                          v_ego=j.loc[new, "v_ego"], lat_ev=j.loc[new, "lat_ev"], lat_grow_s=j.loc[new, "lat_grow_s"],
                          plan_lat4=j.loc[new, "plan_lat4"], plan_dlat=j.loc[new, "plan_dlat"],
                          ego_lane_max=j.loc[new, "ego_lane_max"], hum_lane_max=j.loc[new, "hum_lane_max"]))
    x.index = [s[7:15] for s in x.index]
    x.index.name = "R2J new lateral zero"
    L += [f"### 2b. R2J's {len(new)} lateral zeros that NTV0 does not have (NTV0 / R2C class on the same scene; plan at the decision 2.5 s before "
          "the event, lateral vs the log; lane distance = max over the last 10 s to the nearest aligned lane centre)", ""] + md(x.sort_values("R2J_class")) + [""]
    # 3. launch
    D = pd.read_csv(RES / "d1_launch.csv")
    D["stall"] = (D.e_d4 < 1) & (D.h_d4 > 3)
    agg = dict(n=("scene", "size"), launched=("launched", "sum"), ego_move_s=("ego_move", "median"), plan_1s=("win_pm_a1", "median"),
               plan_2s=("win_pm_a2", "median"), plan_4s=("win_pm_a4", "median"), deferred=("win_defer", "mean"), ego_4s=("e_d4", "median"),
               human_4s=("h_d4", "median"), ego_v2=("e_v2", "median"), human_v2=("h_v2", "median"), stall=("stall", "sum"),
               rear_hit=("rear_t", lambda q: int(q.notna().sum())), fol_gap0=("fol_gap0", "median"), rear_after_s=("rear_t", "median"),
               v_at_rear=("v_at_rear", "median"), ego_d_rear=("e_d_at_rear", "median"), human_d_rear=("h_d_at_rear", "median"))
    c = D[(D.set == "281") & (D.read == "R2C")].set_index("scene")
    jj = D[(D.set == "281") & (D.read == "R2J")].set_index("scene")
    lost = [s for s in c.index if c.launched[s] and not jj.launched[s]]
    for title, sel in (("PAI 281 ACC + GAP, all 37 launch scenes", D.set == "281"),
                       (f"PAI 281 ACC + GAP, the {len(lost)} launches R2C makes and R2J loses", (D.set == "281") & D.scene.isin(lost)),
                       ("PAI-FIX G-LAUNCH (6 scenes, native VC + LEAD)", (D.set == "fix") & D.g_launch)):
        t = D[sel].groupby("read").agg(**agg)
        t = t.reindex([r for r in ("NTV0", "R2C", "R2J", "R3JW", "R3JNF") if r in t.index]).T
        t.index.name = "from the human's launch t_L"
        L += [f"### 3. {title}", ""] + md(t) + [""]
    t = pd.concat([c.loc[lost], jj.loc[lost]], keys=["R2C", "R2J"]).reset_index()
    t = t.assign(scene=t.scene.str[7:15]).set_index(["scene", "level_0"])[["fclass", "ego_move", "win_pm_a1", "win_pm_a2", "win_pm_a4", "win_defer",
                                                                           "e_d4", "h_d4", "fol_gap0", "rear_t", "v_at_rear"]].sort_index()
    t.index = [f"{a} {b}" for a, b in t.index]
    t.index.name = "scene, read"
    L += ["### 3d. The lost launches per scene", ""] + md(t) + [""]
    # 4. overshoot
    D = pd.read_csv(RES / "d1_overshoot.csv")
    D = D[D.h_brake_dist > 15]                                         # the human brakes from speed (drops the creep / standstill rests)
    agg = dict(n=("scene", "size"), plan_has_stop=("dec_plan_stops", lambda q: int((q > 0).sum())), human_brake_m=("h_brake_dist", "median"),
               human_v_brake=("h_v_brake", "median"), **{f"{w}@{d_}": (f"{w}@{d_}", "median") for d_ in (40, 20, 10) for w in ("h_v", "e_v", "pl_end", "pl_v4")},
               plan_brake_m=("plan_brake_onset_dist", "median"), first_stop_plan_m=("plan_first_stop_dist", "median"),
               stop_plan_end_m=("plan_stop_end_vs_rest_med", "median"), ego_v_at_rest=("ego_v_at_rest_station", "median"),
               served_over_model=("served_over_model_40m", "median"))
    for title, st in (("PAI 281 ACC + GAP", "281"), ("PAI-FIX native (T281-STOP + STOP + STOP-L)", "fix")):
        t = D[D.set == st].groupby("read").agg(**agg)
        t = t.reindex([r for r in ("NTV0", "R2C", "R2J", "R3JW", "R3JNF") if r in t.index]).T
        t.index.name = "overshoot zeros (human brakes >= 15 m)"
        L += [f"### 4. {title}", ""] + md(t) + [""]
    # map coverage
    M = pd.read_csv(RES / "d1_mapcov_train.csv")
    mv = M.log_m > 5
    tm = M[M.t_raster > 1e-3]
    L += ["### 2c. NuRec vector maps of the curated-train scenes (the evaluator's offroad surface)", "",
          f"- scenes with a vector map: {int((M.n_lanes > 0).sum())} / {len(M)}; lanes per scene median {M.n_lanes.median():.0f} (min {M.n_lanes.min()}, max {M.n_lanes.max()})",
          f"- human footprint inside the lane union over the clip (moving scenes, {int(mv.sum())}): min margin >= -0.3 m in {(M.log_margin_min[mv] >= -0.3).mean():.1%}, "
          f">= -0.5 m in {(M.log_margin_min[mv] >= -0.5).mean():.1%}; share of poses inside median {M.log_in[mv].median():.3f}",
          f"- cost: map load {M.t_load.median():.2f} s, lane union {M.t_map.median():.3f} s per scene; a 128 x 96 signed-distance raster (shapely, one core) "
          f"{tm.t_raster.median():.2f} s median ({len(tm)} timed scenes, p90 {tm.t_raster.quantile(0.9):.2f} s)", ""]
    return L


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["budget", "offroad", "launch", "overshoot", "tables", "all"])
    a = p.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("budget", "all"):
        B, FXB, ceil, per = budget()
        B.to_csv(RES / "d1_budget_281.csv", index=False, float_format="%.3f")
        FXB.to_csv(RES / "d1_budget_fix.csv", index=False, float_format="%.3f")
        per.to_csv(RES / "d1_scenes_281.csv", float_format="%.3f")
        print("\n".join(md_budget(B, FXB, ceil, per)))
    if a.cmd in ("offroad", "all"):
        D = offroad()
        D.to_csv(RES / "d1_lateral.csv", index=False, float_format="%.3f")
        print(D.groupby(["read", "cls"]).size().unstack(0).fillna(0).astype(int).to_string())
    if a.cmd in ("launch", "all"):
        D = launch()
        D.to_csv(RES / "d1_launch.csv", index=False, float_format="%.3f")
        print(D.groupby(["set", "read"]).median(numeric_only=True).T.to_string())
    if a.cmd in ("overshoot", "all"):
        D = overshoot()
        D.to_csv(RES / "d1_overshoot.csv", index=False, float_format="%.3f")
        print(D.groupby(["set", "read"]).median(numeric_only=True).T.to_string())
    if a.cmd in ("tables", "all"):
        print("\n".join(tables()))


if __name__ == "__main__":
    main()
