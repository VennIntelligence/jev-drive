#!/usr/bin/env python3
"""RECEDING-TURN fingerprint (plans/2026-10-15-receding-turn-prereg.md; research/next-round/turn-night.md section 9). Does the served plan
place the turn onset at a fixed world point as the car approaches (slope of onset arc vs distance to the entry ~ 1), or keep it at a fixed
distance ahead of the car (slope ~ 0, the turn recedes)? CPU only; stored reads are read, never written.

  extract-pai   (Tokyo, host python3 + numpy)  rf_receding.py extract-pai --set NAME=<follow pkl>:<pkl label>:<run root> ... --strict <txt> --out <pkl>
                per strict scene: rf_pai_follow.one's turn row on the label's own route, the route, controller trace, human log, zero time,
                and every `infer` decision's t0 / anchor / poses / poses_model (runs and reruns under the root, keyed by the rollout id).
  extract-held  (box, repo .venv)  rf_receding.py extract-held --set NAME=<rf_tax extract pkl> ... --follow <scenes.csv> --out <pkl>
                per strict turn of label NAME@18: the scenes.csv row, the rf_tax route / controller / log, the decisions of the rollout's session.
  read          (Mac, .venv)  rf_receding.py read --pai <pkl> --held <pkl> --tax NAME=<csv> ... --runway <txt> --out DIR --fig PNG
                -> DIR/turns.csv, decisions.parquet, receding.json, receding.md; the figure.

Definitions (prereg): plan = 8 poses 0.5 .. 4 s in the anchor frame plus the origin; psi = sigma x unwrapped yaw (into the turn);
s_on = arc where psi first exceeds 15 deg (linear between poses); dpsi = psi at 4 s; world onset W = the s_on point composed with the
anchor, projected on the route; route onset s_onR = first station past the ego where sigma (h(s) - h(s_e)) > 15 deg, minus s_e.
Approach = decisions with d_e = entry - s_e in [0, 40] m, t0 < t_f, before the ego first reaches the entry. Per turn: slope = Theil-Sen
of s_on on d_e (pairs |dd_e| >= 1 m; >= 3 points spanning >= 5 m); drift = Theil-Sen of W_st on s_e x the s_e span (m, + = forward).
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

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import rf_pai_follow as F  # noqa: E402

ON_DEG, PRE_M, MIN_N, MIN_SPAN, PAIR_DX, N_BOOT = 15.0, 40.0, 3, 5.0, 1.0, 10_000
T0_HELD, DT_HELD = 17_000, 500_000
TURN_CLS = {"cor_straight_at_turn": "straight", "off_junc_straight": "straight", "cor_turn_under": "late_wide", "off_junc_outside": "late_wide",
            "off_curve_outside": "late_wide", "cor_turn_over": "over_turn", "off_junc_inside": "over_turn", "off_curve_inside": "over_turn",
            "cor_wrong_exit": "wrong_exit"}
CLASSES = ("completed", "late_wide", "straight", "over_turn", "wrong_exit", "other", "no_zero")
PAI_SETS = ("RSG-L18-s0", "RV2a-L18-s0", "RV2b-L18-s1", "APP1X-L18-s0")
HELD_SETS = ("RSG-L18-s0", "RSG-L18-s1", "RV2a-L18-s0", "RV2a-L18-s1", "RV2b-L18-s0", "RV2b-L18-s1")


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- extract (Tokyo / box)
def plan_records(paths, rids):
    """{rid: dict(t0, k, anchor, poses, poses_model)} from drive.jsonl files, infer decisions only, sorted by t0."""
    out = {}
    for f in paths:
        for line in open(f):
            if '"kind": "drive"' not in line or '"poses"' not in line:
                continue
            r = json.loads(line)
            if r.get("session") not in rids or r.get("infer") is False or "anchor" not in r:
                continue
            out.setdefault(r["session"], []).append((int(r["t0"]), int(r["k"]), r["anchor"], r["poses"], r.get("poses_model") or r["poses"]))
    res = {}
    for rid, v in out.items():
        v.sort(key=lambda x: (x[0], x[1]))
        res[rid] = dict(t0=np.array([x[0] for x in v], np.int64), k=np.array([x[1] for x in v]), anchor=np.array([x[2] for x in v], float),
                        poses=np.array([x[3] for x in v], float), poses_model=np.array([x[4] for x in v], float))
    return res


def extract_pai(a):
    strict = [x.split()[0] for x in open(a.strict) if x.strip()]
    out = []
    for spec in a.set:
        name, rest = spec.split("=", 1)
        pk, lab, root = rest.split(":")
        D = {sc["scene"]: sc for sc in pickle.load(open(pk, "rb"))["scenes"]}
        recs = []
        for s in strict:
            o = D.get(s, {}).get("rolls", {}).get(lab)
            if o is None or o.get("ctrl") is None:
                continue
            row, _, _ = F.one(D[s], lab, o, zeros=False)
            recs.append(dict(set=name, src="pai", scene=s, rid=o["rid"], score=float(o["score"] or 0), kind=row.get("kind", ""),
                             tz=o["tz"] if o["tz"] is not None and not o["score"] else np.inf, row=row, R=F.dedup(np.asarray(o["route"], float)),
                             ctrl={k: np.asarray(v, float) for k, v in o["ctrl"].items()}, log=dict(T=np.asarray(D[s]["log"]["T"], float),
                                                                                                    P=np.asarray(D[s]["log"]["P"], float))))
        files = glob.glob(os.path.join(root, "**", "driver", "drive.jsonl"), recursive=True)
        P = plan_records(files, {r["rid"] for r in recs})
        for r in recs:
            r["plans"] = P.get(r["rid"])
        print(name, len(recs), "strict scenes,", sum(r["plans"] is not None for r in recs), "with plans,", len(files), "driver logs", flush=True)
        out += recs
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "->", a.out)


def extract_held(a):
    import csv
    fo = list(csv.DictReader(open(a.follow)))
    out = []
    for spec in a.set:
        name, pk = spec.split("=", 1)
        rows = {r["scene"]: r for r in fo if r["label"] == f"{name}@18" and r["turn"] == "True"}
        rolls = {o["scene"]: o for o in pickle.load(open(pk, "rb"))["rolls"] if o["scene"] in rows}
        recs, by_run = [], {}
        for s, r in rows.items():
            o = rolls.get(s)
            if o is None or o.get("ctrl") is None:
                continue
            row = dict(turn=True, side=r["side"], entry=float(r["entry"]), exit=float(r["exit"]), outcome=r["outcome"], completed=r["completed"] == "True",
                       turn_deg=float(r["turn_deg"]), s0=float(r["s0"]), s_log_end=float(r["s_log_end"]), kz=int(r["kz"]))
            kz = row["kz"]
            recs.append(dict(set=name, src="held", scene=s, rid=o["rid"], score=float(o["score"] or 0), kind="", row=row,
                             tz=float(T0_HELD + DT_HELD * kz) if kz >= 0 else np.inf, R=F.dedup(np.asarray(o["route"], float)),
                             ctrl={k: np.asarray(o["ctrl"][k], float) for k in ("t", "x", "y", "yaw", "vx")},
                             log=dict(T=np.asarray(o["log"]["T"], float), P=np.asarray(o["log"]["P"], float)), run=o["run"]))
            by_run.setdefault(o["run"], set()).add(o["rid"])
        P = {}
        for run, rids in by_run.items():
            P |= plan_records([os.path.join(run, "driver-logs", "drive.jsonl")], rids)
        for r in recs:
            r["plans"] = P.get(r["rid"])
        print(name, len(rows), "turns,", len(recs), "extracted,", sum(r["plans"] is not None for r in recs), "with plans", flush=True)
        out += recs
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "->", a.out)


# ---------------------------------------------------------------- geometry (read)
def to_world(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.asarray(a[:2], float) + np.c_[c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]]


def onset(P, sg):
    """Plan (8, 3) in its own frame -> (s_on or nan, onset point (2,) or nan, dpsi deg, horizon arc)."""
    xy = np.r_[[[0.0, 0.0]], np.asarray(P, float)[:, :2]]
    psi = sg * np.degrees(np.unwrap(np.r_[0.0, np.asarray(P, float)[:, 2]]))
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    i = np.flatnonzero(psi > ON_DEG)
    if not len(i) or i[0] == 0:
        return np.nan, np.full(2, np.nan), float(psi[-1]), float(arc[-1])
    j = int(i[0])
    w = (ON_DEG - psi[j - 1]) / max(psi[j] - psi[j - 1], 1e-9)
    return float(arc[j - 1] + w * (arc[j] - arc[j - 1])), xy[j - 1] + w * (xy[j] - xy[j - 1]), float(psi[-1]), float(arc[-1])


def theil_sen(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < MIN_N or np.ptp(x) < MIN_SPAN:
        return np.nan, float(np.ptp(x)) if len(x) else 0.0
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) >= PAIR_DX
    return (float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else np.nan), float(np.ptp(x))


class Log:
    """Human log path: route stations (first passage), pose at any time with the chord heading over +-1 m of log arc."""

    def __init__(self, L, R, S):
        self.T, self.P = L["T"], L["P"]
        self.arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.P, axis=0).T))]
        lt, _ = F.track(R, S, self.P)
        m = np.maximum.accumulate(lt)
        self.su, iu = np.unique(m, return_index=True)
        self.tu = self.T[iu]

    def xy_arc(self, q):
        q = np.clip(q, 0, self.arc[-1])
        return np.c_[np.interp(q, self.arc, self.P[:, 0]), np.interp(q, self.arc, self.P[:, 1])]

    def pose(self, t):
        q = np.interp(t, self.T, self.arc)
        d = self.xy_arc(q + 1.0) - self.xy_arc(q - 1.0)
        return np.c_[self.xy_arc(q), np.arctan2(d[:, 1], d[:, 0])]

    def plan_at_station(self, s):
        """The log's own 4 s 'plan' from where its route station first reaches s -> (8, 3) in that pose's frame, origin pose) or None."""
        if s < self.su[0] or s > self.su[-1]:
            return None, None
        t = float(np.interp(s, self.su, self.tu))
        Q = self.pose(t + 0.5e6 * np.arange(9))
        Q[:, 2] = np.unwrap(Q[:, 2])
        c, sn = np.cos(Q[0, 2]), np.sin(Q[0, 2])
        d = Q[1:, :2] - Q[0, :2]
        return np.c_[c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1], Q[1:, 2] - Q[0, 2]], Q[0]


def lite_class(rec, R, S, tf):
    """rf_tax's corridor heading rule at tf on the extract (log heading from its path chord): straight / late_wide / over_turn / wrong_exit /
    route_end / other."""
    C, L = rec["ctrl"], rec["log"]
    P = np.c_[C["x"], C["y"]]
    yaw = np.unwrap(C["yaw"])
    tb = max(tf - 3e6, C["t"][0])
    pe = np.array([np.interp(tf, C["t"], P[:, 0]), np.interp(tf, C["t"], P[:, 1])])
    pb = np.array([np.interp(tb, C["t"], P[:, 0]), np.interp(tb, C["t"], P[:, 1])])
    LP = F.dedup(L["P"])
    LS = F.stations(LP)
    s_ev, s_b = F.project(LP, LS, pe)[0], F.project(LP, LS, pb)[0]
    s_hi = min(s_ev + 15.0, LS[-1])
    xy = lambda q: np.array([np.interp(q, LS, LP[:, 0]), np.interp(q, LS, LP[:, 1])])  # noqa: E731

    def hd(q):
        d = xy(min(q + 1.0, LS[-1])) - xy(max(q - 1.0, 0.0))
        return np.arctan2(d[1], d[0])
    qs = np.linspace(s_b, s_hi, 30)
    hl = np.unwrap([hd(q) for q in qs])
    dl = float(np.degrees(hl[-1] - hl[0]))
    de = float(np.degrees(np.interp(tf, C["t"], yaw) - np.interp(tb, C["t"], yaw)))
    if s_ev >= LS[-1] - 2.0:
        return "route_end"
    if abs(dl) >= 30 and (abs(de) < 10 or np.sign(de) != np.sign(dl)):
        return "straight"
    if abs(dl) >= 30:
        return "late_wide" if abs(de) < abs(dl) else "over_turn"
    return "wrong_exit" if abs(de) >= 30 else "other"


def read_turn(rec, tax):
    """One stored turn -> (turn row, decision rows)."""
    row = rec["row"]
    if not row.get("turn") or rec.get("plans") is None:
        return None, []
    R = rec["R"]
    S = F.stations(R)
    sh, hh = F.heading(R, S)
    sg = 1.0 if row["side"] == "L" else -1.0
    ent, ext = float(row["entry"]), float(row["exit"])
    C = rec["ctrl"]
    t = C["t"]
    Pc = np.c_[C["x"], C["y"]]
    stc, _ = F.track(R, S, Pc)
    tf = rec["tz"]
    LG = Log(rec["log"], R, S)
    Pl = rec["plans"]
    psi_exit = float(np.interp(ext + 10.0, sh, hh))
    out, frame_d = [], []
    for i in range(len(Pl["t0"])):
        t0, A = float(Pl["t0"][i]), Pl["anchor"][i]
        ic = int(np.clip(np.searchsorted(t, t0), 0, len(t) - 1))
        if t[0] <= t0 <= t[-1]:
            frame_d.append(float(np.hypot(np.interp(t0, t, Pc[:, 0]) - A[0], np.interp(t0, t, Pc[:, 1]) - A[1])))
        s_e, lat = F.project(R, S, A[:2], stc[ic] - 30.0, stc[ic] + 60.0)
        d = dict(k=int(Pl["k"][i]), t0=t0, s_e=s_e, lat=lat, d_e=ent - s_e, pre=bool(t0 < tf), v=float(np.interp(t0, t, C["vx"])),
                 yaw_rem=float(np.degrees(sg * wrap(psi_exit - A[2]))))
        for tag, P in (("", Pl["poses"][i]), ("m_", Pl["poses_model"][i])):
            so, q, dp, a4 = onset(P, sg)
            d[tag + "s_on"], d[tag + "dpsi"], d[tag + "a4"] = so, dp, a4
            d[tag + "W"] = F.project(R, S, to_world(A, q)[0], s_e - 30.0, s_e + 120.0)[0] if np.isfinite(so) else np.nan
        j = np.flatnonzero((sh > s_e) & (sg * (hh - np.interp(s_e, sh, hh)) > np.radians(ON_DEG)))
        d["s_onR"] = float(sh[j[0]] - s_e) if len(j) else np.nan
        H, H0 = LG.plan_at_station(s_e)
        if H is not None:
            so, q, dp, a4 = onset(H, sg)
            d["h_s_on"], d["h_dpsi"], d["h_a4"] = so, dp, a4
            d["h_W"] = F.project(R, S, to_world(H0, q)[0], s_e - 30.0, s_e + 120.0)[0] if np.isfinite(so) else np.nan
        else:
            d.update(h_s_on=np.nan, h_dpsi=np.nan, h_a4=np.nan, h_W=np.nan)
        out.append(d)
    se = np.array([d["s_e"] for d in out])
    reach = np.flatnonzero(se >= ent)
    k_ent = int(reach[0]) if len(reach) else len(out)
    for i, d in enumerate(out):
        d["approach"] = bool(0.0 <= d["d_e"] <= PRE_M and d["pre"] and i < k_ent)
        d["inside"] = bool(ent <= d["s_e"] <= ext + 10.0 and d["pre"])
    # class
    done = bool(row.get("completed"))
    if done:
        cls, src = "completed", "-"
    elif rec["score"] > 0 or not np.isfinite(tf):
        cls, src = "no_zero", "-"
    else:
        prim = tax.get(rec["scene"]) if tax is not None else None
        if prim is not None and prim in TURN_CLS:
            cls, src = TURN_CLS[prim], "tax"
        else:
            cls, src = lite_class(rec, R, S, tf), "rule"
            cls = cls if cls in CLASSES else "other"
    tr = dict(set=rec["set"], src=rec["src"], scene=rec["scene"], cluster=rec["scene"].rsplit("-", 1)[0] if rec["src"] == "held" else rec["scene"],
              side=row["side"], turn_deg=float(row["turn_deg"]), entry=ent, exit=ext, outcome=row["outcome"], completed=done, cls=cls, cls_src=src,
              primary=(tax.get(rec["scene"]) if tax is not None else None), score=rec["score"], frame_med=float(np.median(frame_d)) if frame_d else np.nan,
              start_to_entry=float(ent - se[0]), n_dec=len(out))
    if rec["src"] == "held":
        e2 = F.find_turn(sh, hh, row["s0"], row["s_log_end"])
        tr["entry_recomp"] = float(e2[0]) if e2 else np.nan
    ap = [d for d in out if d["approach"]]
    tr["n_app"] = len(ap)
    g = lambda k: np.array([d[k] for d in ap], float)  # noqa: E731
    de, s_e = g("d_e"), g("s_e")
    for tag in ("", "m_", "h_"):
        so, W = g(tag + "s_on"), g(tag + "W")
        tr[tag + "slope"], tr[tag + "span"] = theil_sen(de, so) if len(ap) else (np.nan, 0.0)
        ds, sp = theil_sen(s_e[np.isfinite(W)], W[np.isfinite(W)]) if len(ap) else (np.nan, 0.0)
        tr[tag + "drift"] = ds * sp if np.isfinite(ds) else np.nan
        tr[tag + "n_on"] = int(np.isfinite(so).sum())
        tr[tag + "visible"] = float(np.isfinite(so).mean()) if len(ap) else np.nan
    if ap:
        so, sR, a4 = g("s_on"), g("s_onR"), g("a4")
        vis = np.flatnonzero(np.isfinite(so))
        tr["de_first_visible"] = float(de[vis[0]]) if len(vis) else np.nan
        near = de <= 15.0
        lag = so - sR
        tr["lag15"] = float(np.nanmedian(lag[near])) if np.isfinite(lag[near]).any() else np.nan
        cov = np.isfinite(sR) & (a4 >= sR)
        tr["unseen"] = float((~(so[cov] <= sR[cov] + 5.0)).mean()) if cov.any() else np.nan
        tr["n_cov"] = int(cov.sum())
        tr["v_app_med"] = float(np.median(g("v")))
    for d in out:
        d.update(set=rec["set"], src=rec["src"], scene=rec["scene"], cls=cls, completed=done)
    return tr, out


# ---------------------------------------------------------------- statistics
def boot_groups(T, col, n_boot=N_BOOT, seed=0):
    """Failed vs completed medians of T[col] with a cluster bootstrap of D = median failed - median completed."""
    x = T[np.isfinite(T[col].to_numpy(float))]
    f, c = x[~x.completed], x[x.completed]
    res = dict(n_f=int(len(f)), n_c=int(len(c)), med_f=float(f[col].median()) if len(f) else np.nan, med_c=float(c[col].median()) if len(c) else np.nan)
    if not len(f) or not len(c):
        return res | dict(D=np.nan, lo=np.nan, hi=np.nan, dropped=0)
    cl = np.unique(x.cluster)
    idx = {k: np.flatnonzero(x.cluster.to_numpy() == k) for k in cl}
    v, done = x[col].to_numpy(float), x.completed.to_numpy(bool)
    rng = np.random.default_rng(seed)
    D, drop = [], 0
    for _ in range(n_boot):
        r = np.concatenate([idx[k] for k in rng.choice(cl, len(cl))])
        a, b = v[r][~done[r]], v[r][done[r]]
        if not len(a) or not len(b):
            drop += 1
            continue
        D.append(np.median(a) - np.median(b))
    D = np.array(D)
    return res | dict(D=res["med_f"] - res["med_c"], lo=float(np.percentile(D, 2.5)), hi=float(np.percentile(D, 97.5)), dropped=drop)


def verdict_c1(b):
    """b: D = failed - completed. Registered on M_c - M_f: CI lower > 0 <=> (failed - completed) CI upper < 0."""
    if b["n_f"] < 3 or b["n_c"] < 3:
        return "inconclusive (group < 3)"
    lo_cf, hi_cf = -b["hi"], -b["lo"]
    if b["med_f"] <= 0.3 and b["med_c"] >= 0.7 and lo_cf > 0:
        return "supported"
    if b["med_f"] >= 0.7 or hi_cf < 0.4:
        return "refuted"
    return "inconclusive"


def verdict_c2(b):
    if b["n_f"] < 3 or b["n_c"] < 3:
        return "inconclusive (group < 3)"
    if b["med_f"] >= 10.0 and b["med_c"] <= 3.0 and b["lo"] > 0:
        return "supported"
    if b["med_f"] <= 5.0 or b["hi"] < 5.0:
        return "refuted"
    return "inconclusive"


def paired_median(d, n_boot=N_BOOT, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0)
    m = np.median(d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))], axis=1)
    return dict(n=int(len(d)), median=float(np.median(d)), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))


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
        key = ("pai_" if r["src"] == "pai" else "held_") + r["set"]
        tr, dd = read_turn(r, taxes.get(key))
        if tr is None:
            continue
        tr["key"] = key
        for d in dd:
            d["key"] = key
        TR.append(tr)
        DR += dd
    T, Dx = pd.DataFrame(TR), pd.DataFrame(DR)
    T.to_csv(out / "turns.csv", index=False)
    Dx.to_parquet(out / "decisions.parquet", index=False)
    keys = [f"pai_{s}" for s in PAI_SETS] + [f"held_{s}" for s in HELD_SETS]
    res = dict(gates={}, sets={}, classes={}, c3={}, c4={}, app1x={})
    # gates
    res["gates"]["frame"] = {k: float(T[T.key == k].frame_med.median()) for k in keys if (T.key == k).any()}
    H = T[T.src == "held"]
    ok = (H.entry - H.entry_recomp).abs() <= 1.0
    res["gates"]["entry"] = dict(n=int(len(H)), share_within_1m=float(ok.mean()) if len(H) else np.nan, passed=bool(ok.mean() >= 0.9) if len(H) else None)
    P0 = T[T.key == "pai_RSG-L18-s0"]
    hs, hd = float(P0.h_slope.median()), float(P0.h_drift.abs().median())
    res["gates"]["human"] = dict(median_slope=hs, median_abs_drift=hd, n=int(P0.h_slope.notna().sum()), passed=bool(0.8 <= hs <= 1.2 and hd <= 3.0))
    tc = T[(T.cls_src == "tax")]
    res["gates"]["rule_vs_tax"] = None
    # claims per set
    groups = {k: T[T.key == k] for k in keys} | {"held_pooled": H}
    for k, x in groups.items():
        if not len(x):
            continue
        r = dict(turns=int(len(x)), failed=int((~x.completed).sum()), completed=int(x.completed.sum()))
        for tag in ("", "m_", "h_"):
            if tag == "m_" and k.startswith("held"):
                continue
            b1, b2 = boot_groups(x, tag + "slope"), boot_groups(x, tag + "drift")
            r[tag + "C1"] = b1 | dict(verdict=verdict_c1(b1))
            r[tag + "C2"] = b2 | dict(verdict=verdict_c2(b2))
        for c in ("visible", "de_first_visible", "lag15", "unseen", "v_app_med", "n_app"):
            if c in x:
                r[c] = {g: float(x[x.completed == (g == "completed")][c].median()) for g in ("failed", "completed")}
        res["sets"][k] = r
    if not res["gates"]["human"]["passed"]:
        for r in res["sets"].values():
            for c in ("C1", "C2"):
                r[c]["verdict"] = "inconclusive (G-human failed)"
    # classes
    for k, x in list(groups.items()) + [("pai_all4", T[T.src == "pai"])]:
        res["classes"][k] = {c: dict(n=int((x.cls == c).sum()), n_slope=int(x[x.cls == c].slope.notna().sum()),
                                     slope=float(x[x.cls == c].slope.median()), drift=float(x[x.cls == c].drift.median()),
                                     lag15=float(x[x.cls == c].lag15.median()), unseen=float(x[x.cls == c].unseen.median()),
                                     visible=float(x[x.cls == c].visible.median()))
                             for c in CLASSES if (x.cls == c).any()}
    # C3: dpsi by d_e bin and class
    A = Dx[Dx.approach]
    bins = [0, 10, 20, 30, 40.01]
    A = A.assign(bin=pd.cut(A.d_e, bins, right=False, labels=["0-10", "10-20", "20-30", "30-40"]))
    for k, x in (("pai_RSG-L18-s0", A[A.key == "pai_RSG-L18-s0"]), ("pai_all4", A[A.src == "pai"]), ("held_all6", A[A.src == "held"])):
        res["c3"][k] = {c: {str(b): dict(n=int(len(y)), dpsi=float(y.dpsi.median()), h_dpsi=float(y.h_dpsi.median()))
                            for b, y in x[x.cls == c].groupby("bin", observed=False)} for c in CLASSES if (x.cls == c).any()}
    # C4: inside the turn
    I = Dx[Dx.inside]
    for k, x in (("pai_RSG-L18-s0", I[I.key == "pai_RSG-L18-s0"]), ("pai_all4", I[I.src == "pai"]), ("held_all6", I[I.src == "held"])):
        res["c4"][k] = {}
        for c in CLASSES:
            y = x[x.cls == c]
            if not len(y):
                continue
            past = y[y.yaw_rem <= 0]
            ok2 = np.isfinite(y.dpsi) & np.isfinite(y.yaw_rem)
            sl = float(np.polyfit(y.yaw_rem[ok2], y.dpsi[ok2], 1)[0]) if ok2.sum() >= 3 else np.nan
            res["c4"][k][c] = dict(n=int(len(y)), turns=int(y.scene.nunique()), n_past=int(len(past)), turns_past=int(past.scene.nunique()),
                                   share_more=float((past.dpsi > 5).mean()) if len(past) else np.nan,
                                   dpsi_past=float(past.dpsi.median()) if len(past) else np.nan, excess=float((y.dpsi - y.yaw_rem).median()),
                                   ols=sl, h_excess=float((y.h_dpsi - y.yaw_rem).median()))
    # APP1X vs RSG on the runway turns
    rw = {x.split()[0] for x in open(a.runway) if x.strip()} if a.runway else set()
    a1, rs = T[T.key == "pai_APP1X-L18-s0"].set_index("scene"), T[T.key == "pai_RSG-L18-s0"].set_index("scene")
    for nm, S in (("runway22", sorted(rw)), ("strict36", sorted(set(a1.index) & set(rs.index)))):
        S = [s for s in S if s in a1.index and s in rs.index]
        res["app1x"][nm] = dict(slope_app1x=float(a1.loc[S].slope.median()) if S else np.nan, slope_rsg=float(rs.loc[S].slope.median()) if S else np.nan,
                                paired=paired_median([a1.loc[s].slope - rs.loc[s].slope for s in S]),
                                drift_app1x=float(a1.loc[S].drift.median()) if S else np.nan, drift_rsg=float(rs.loc[S].drift.median()) if S else np.nan,
                                completed_app1x=int(a1.loc[S].completed.sum()) if S else 0, completed_rsg=int(rs.loc[S].completed.sum()) if S else 0, n=len(S))
    # rule vs taxonomy agreement (sets with a taxonomy; zeros with a turn-class primary)
    agree = []
    for rec in recs:
        key = ("pai_" if rec["src"] == "pai" else "held_") + rec["set"]
        tx = taxes.get(key)
        if tx is None or not rec["row"].get("turn") or rec["row"].get("completed") or rec["score"] > 0 or not np.isfinite(rec["tz"]):
            continue
        p = tx.get(rec["scene"])
        if p in TURN_CLS:
            R = rec["R"]
            agree.append((key, rec["scene"], TURN_CLS[p], lite_class(rec, R, F.stations(R), rec["tz"])))
    res["gates"]["rule_vs_tax"] = dict(n=len(agree), agree=float(np.mean([x[2] == x[3] for x in agree])) if agree else np.nan,
                                       pairs=[list(x) for x in agree])
    (out / "receding.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    write_md(out, res, T)
    if a.fig:
        fig(T, Dx, a.fig)


def write_md(out, res, T):
    f = lambda v, p=2: "-" if v is None or not np.isfinite(v) else f"{v:+.{p}f}"  # noqa: E731
    g = res["gates"]
    md = ["# RECEDING-TURN fingerprint (rf_receding.py)", "",
          "Prereg `plans/2026-10-15-receding-turn-prereg.md`. Per strict turn: slope = Theil-Sen of the served plan's 15 deg onset arc s_on on the "
          "distance to the entry d_e over the approach (1 = onset fixed in the world, 0 = fixed ahead of the car: receding); drift = forward "
          "movement of the world-frame onset along the route over the approach (m). Medians failed / completed, D = failed - completed with a "
          "scene-clustered bootstrap 95 % CI (B 10 000).", "",
          f"Gates: G-frame median |anchor - controller| per set {', '.join(f'{k} {v:.3f}' for k, v in g['frame'].items())} m; "
          f"G-entry (held) {g['entry']['share_within_1m']:.2f} of {g['entry']['n']} within 1 m ({'pass' if g['entry']['passed'] else 'FAIL'}); "
          f"G-human (pai_RSG) median slope {g['human']['median_slope']:.2f}, median |drift| {g['human']['median_abs_drift']:.2f} m "
          f"(n {g['human']['n']}, {'pass' if g['human']['passed'] else 'FAIL'}); class rule vs taxonomy {g['rule_vs_tax']['agree']:.2f} (n {g['rule_vs_tax']['n']}).", "",
          "| set | turns f / c | C1 slope med f / c | D slope [CI] | C1 | C2 drift med f / c (m) | D drift [CI] | C2 | human slope f / c | model-plan slope f / c |",
          "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, r in res["sets"].items():
        c1, c2, h = r["C1"], r["C2"], r["h_C1"]
        m = r.get("m_C1")
        md.append(f"| {k} | {r['failed']} / {r['completed']} (slope {c1['n_f']} / {c1['n_c']}) | {f(c1['med_f'])} / {f(c1['med_c'])} | "
                  f"{f(c1['D'])} [{f(c1['lo'])}, {f(c1['hi'])}] | {c1['verdict']} | {f(c2['med_f'], 1)} / {f(c2['med_c'], 1)} | "
                  f"{f(c2['D'], 1)} [{f(c2['lo'], 1)}, {f(c2['hi'], 1)}] | {c2['verdict']} | {f(h['med_f'])} / {f(h['med_c'])} | "
                  + (f"{f(m['med_f'])} / {f(m['med_c'])} |" if m else "- |"))
    md += ["", "Descriptives (median failed / completed): onset visible share of approach decisions, d_e at first visibility (m), lag = s_on - route "
           "onset over d_e <= 15 m (m), unseen = share of decisions whose 4 s horizon reaches the route onset but the plan's onset is absent or > 5 m "
           "beyond it, approach speed (m/s), approach decisions.", "",
           "| set | visible | d_e first visible | lag15 | unseen | v | n_app |", "|:--|:--|:--|:--|:--|:--|:--|"]
    for k, r in res["sets"].items():
        md.append(f"| {k} | " + " | ".join(f"{f(r[c]['failed'])} / {f(r[c]['completed'])}" if c in r else "-"
                                           for c in ("visible", "de_first_visible", "lag15", "unseen", "v_app_med", "n_app")) + " |")
    md += ["", "Fingerprint by class (turns, turns with a slope; medians):", "",
           "| set | class | n (slope) | slope | drift m | lag15 m | unseen | visible |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k in ("pai_RSG-L18-s0", "pai_all4", "held_pooled"):
        for c, v in res["classes"].get(k, {}).items():
            md.append(f"| {k} | {c} | {v['n']} ({v['n_slope']}) | {f(v['slope'])} | {f(v['drift'], 1)} | {f(v['lag15'], 1)} | {f(v['unseen'])} | {f(v['visible'])} |")
    md += ["", "C3: plan heading change over 4 s toward the turn (deg, median) by distance to the entry; human log in brackets:", "",
           "| set | class | 30-40 m | 20-30 m | 10-20 m | 0-10 m |", "|:--|:--|:--|:--|:--|:--|"]
    for k, v in res["c3"].items():
        for c, b in v.items():
            md.append(f"| {k} | {c} | " + " | ".join(f"{f(b[x]['dpsi'], 1)} ({f(b[x]['h_dpsi'], 1)}, n {b[x]['n']})" for x in ("30-40", "20-30", "10-20", "0-10")) + " |")
    md += ["", "C4: inside the turn (entry .. exit + 10 m). past = decisions with the ego at / past the exit heading; more = share of those whose plan "
           "still asks > +5 deg into the turn; excess = median (plan dpsi - true remaining heading), human in brackets; OLS slope of plan on true remaining.", "",
           "| set | class | decisions (turns) | past (turns) | more | plan dpsi past | excess (human) | OLS |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, v in res["c4"].items():
        for c, b in v.items():
            md.append(f"| {k} | {c} | {b['n']} ({b['turns']}) | {b['n_past']} ({b['turns_past']}) | {f(b['share_more'])} | {f(b['dpsi_past'], 1)} | "
                      f"{f(b['excess'], 1)} ({f(b['h_excess'], 1)}) | {f(b['ols'])} |")
    md += ["", "APP1X vs RSG (PAI): median slope, paired median difference APP1X - RSG [CI], drift medians, completed:", ""]
    for nm, v in res["app1x"].items():
        p = v["paired"]
        md.append(f"- {nm} (n {v['n']}): slope {f(v['slope_app1x'])} vs {f(v['slope_rsg'])}, paired " +
                  (f"{f(p['median'])} [{f(p['lo'])}, {f(p['hi'])}] (n {p['n']})" if p.get("n") else "-") +
                  f"; drift {f(v['drift_app1x'], 1)} vs {f(v['drift_rsg'], 1)} m; completed {v['completed_app1x']} vs {v['completed_rsg']}")
    md += ["", "Per turn (pai_RSG-L18-s0):", "", "| scene | side deg | outcome | class (src) | start to entry m | n app | slope | drift | human slope | visible | lag15 | unseen |",
           "|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in T[T.key == "pai_RSG-L18-s0"].sort_values(["completed", "cls", "scene"]).iterrows():
        md.append(f"| {r.scene[7:15]} | {r.side} {r.turn_deg:+.0f} | {r.outcome} | {r.cls} ({r.cls_src}) | {r.start_to_entry:.1f} | {r.n_app} | {f(r.slope)} | "
                  f"{f(r.drift, 1)} | {f(r.h_slope)} | {f(r.get('visible'))} | {f(r.get('lag15'), 1)} | {f(r.get('unseen'))} |")
    (out / "receding.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:24]))


def fig(T, Dx, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"completed": "#2a78d6", "late_wide": "#eb6834", "straight": "#1baf7a", "over_turn": "#eda100"}
    ls = {"completed": "-", "late_wide": "--", "straight": "-.", "over_turn": ":"}
    x = T[(T.key == "pai_RSG-L18-s0")].sort_values("scene")
    picks = list(x[x.completed].scene)
    for c in ("late_wide", "straight", "over_turn"):
        picks += list(x[(x.cls == c) & x.slope.notna()].scene[:2])
    D = Dx[(Dx.key == "pai_RSG-L18-s0") & Dx.approach & Dx.scene.isin(picks)]
    fg, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for s in picks:
        r = x[x.scene == s].iloc[0]
        d = D[D.scene == s].sort_values("t0")
        c = r.cls if r.cls in col else "late_wide"
        lab = f"{s[7:15]} {r.cls} (slope {r.slope:+.2f})" if np.isfinite(r.slope) else f"{s[7:15]} {r.cls}"
        ax[0].plot(d.d_e, d.s_on, ls[c], color=col[c], lw=2, label=lab)
        ax[1].plot(d.d_e, d.W - r.entry, ls[c], color=col[c], lw=2)
    lim = [0, 40]
    ax[0].plot(lim, lim, color="#888", lw=1)
    ax[0].text(30, 31.5, "onset at the entry (slope 1)", color="#52514e", fontsize=8, rotation=0)
    ax[0].set(xlim=(40, 0), xlabel="ego distance to the turn entry d_e (m)", ylabel="plan onset arc s_on (m, 15 deg into the turn)",
              title="Served plan onset vs distance to the entry (PAI 281, RSG-L18-s0)")
    ax[1].axhline(0, color="#888", lw=1)
    ax[1].set(xlim=(40, 0), xlabel="ego distance to the turn entry d_e (m)", ylabel="world onset station - entry (m)",
              title="World-frame onset: flat = fixed in the world, rising = receding")
    for a_ in ax:
        a_.grid(alpha=0.25)
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, loc="upper left", frameon=False)
    fg.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract-pai")
    e.add_argument("--set", nargs="+", required=True), e.add_argument("--strict", required=True), e.add_argument("--out", required=True)
    h = sp.add_parser("extract-held")
    h.add_argument("--set", nargs="+", required=True), h.add_argument("--follow", required=True), h.add_argument("--out", required=True)
    h.add_argument("--read", default="")
    r = sp.add_parser("read")
    r.add_argument("--pai", default=""), r.add_argument("--held", default=""), r.add_argument("--tax", nargs="*")
    r.add_argument("--runway", default=""), r.add_argument("--out", required=True), r.add_argument("--fig", default="")
    a = ap.parse_args()
    {"extract-pai": extract_pai, "extract-held": extract_held, "read": read}[a.cmd](a)
