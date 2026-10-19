#!/usr/bin/env python3
"""PAI-GAP read-out (plans/2026-10-19-pai-gap-prereg.md). Box, repo .venv; stored reads are read, never written.

  pai_gap.py all --o $DATA_DIR/runs/route_follow/pai_gap --out experiments/route_follow/results/pai_gap

Inputs: A0 = runs/defect_fix/pai_accgap/NTV0-L18-s0 (+ its rf_tax extract runs/route_follow/pai_input_loop/tax/A0.pkl), N0 =
runs/route_follow/pai_input_loop/N0, the other stored PAI 281 ACC + GAP reads (READS), <o>/tax/A0asl.pkl (A0 with the rollout.asl actors, the
traffic gate), <o>/H (log-replay read) and <o>/tax/H.pkl. Writes <out>/tables.md, <out>/pai_gap.json; per-scene / per-cut tables to <o>/.
Item 1 classes / strata, item 2 cut classes, item 3 attribution: as in the prereg (constants below carry its numbers).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent), str(REPO / "experiments/defect_fix/scripts"), str(REPO / "experiments/wait_cap/scripts")]
import rf_tax as TX  # noqa: E402
from pai281 import ptype, zclass  # noqa: E402

D = os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs") + "/runs"
A0 = f"{D}/defect_fix/pai_accgap/NTV0-L18-s0"
PIL = f"{D}/route_follow/pai_input_loop"
READS = {"NTV0-s0": A0, "NTV0-s1": f"{D}/defect_fix/pai_accgap/NTV0-L18-s1", "NTV0-s0-hl": f"{D}/defect_fix/pai_accgap/NTV0-L18-s0-hl",
         "JTN-s0": f"{D}/defect_fix/pai_accgap/JTN-L18-s0", "JTN-s1": f"{D}/defect_fix/pai_accgap/JTN-L18-s1",
         "NTVX-s0": f"{D}/defect_fix/pai_accgap/NTVX-L18-s0", "RSG-s0": f"{D}/route_follow/acc_gap/read/s2"}
HAND = 1.6e6                       # us after the first controller sample: the handoff (tau 1.7 s)
OK = 0.98
LANE, EXT = 3.7, 40.0               # adjacent band (m), straight extension of the plan path (m)
LON = ("rear-hit", "slow", "creep", "no-launch")
LAT_ZERO = ("corridor", "offroad", "AF-side")


# ---------------------------------------------------------------- loading
def summaries(root):
    """-> {scene: results-summary row}, {scenes with a driver RPC error}."""
    R, bad = {}, set()
    for f in glob.glob(f"{root}/runs/*/sim/aggregate/results-summary.json"):
        for r in json.load(open(f))["rollouts"]:
            R.setdefault(r["clipgt_id"], r)
            if "RpcError" in str(r.get("failure_reason") or ""):
                bad.add(r["clipgt_id"])
    return R, bad


def drive(root):
    """-> {scene: inference records sorted by t0}."""
    out = defaultdict(list)
    for f in glob.glob(f"{root}/runs/*/driver/drive.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "drive" and r.get("infer"):
                out[r["scene"]].append(r)
    return {s: sorted(v, key=lambda r: r["t0"]) for s, v in out.items()}


def rolls(pkl):
    return {o["scene"]: o for o in pickle.load(open(pkl, "rb"))["rolls"] if "ctrl" in o}


# ---------------------------------------------------------------- geometry
def to_world(a, P):
    c, s = np.cos(a[2]), np.sin(a[2])
    P = np.asarray(P, float)[:, :2]
    return np.c_[a[0] + c * P[:, 0] - s * P[:, 1], a[1] + s * P[:, 0] + c * P[:, 1]]


def corridor(a, P):
    """Plan path in the world from the ego's rear axle, extended EXT m straight; a straight heading line when the plan is < 1 m."""
    W = np.vstack([a[None, :2], to_world(a, P)])
    W = TX.dedupe(W)
    if len(W) < 2 or np.hypot(*(W[-1] - W[0])) < 1.0:
        W = np.vstack([a[:2], a[:2] + 0.01 * np.array([np.cos(a[2]), np.sin(a[2])])])
    h = np.arctan2(*(W[-1] - W[-2])[::-1])
    return np.vstack([W, W[-1] + EXT * np.array([np.cos(h), np.sin(h)])])


def ahead(R, a, P, st, gmax):
    """Objects ahead of the ego's front bumper along the plan path (and along the straight heading line): list of (gap, |lat| - lim, n)."""
    if not len(st["id"]):
        return []
    X = np.c_[st["x"], st["y"]]
    out = []
    for path in (corridor(a, P), corridor(a, np.zeros((1, 3)))):
        _, s, lat, _ = TX.poly_proj(path, X)
        for n in range(len(X)):
            g = s[n] - R.front - st["L"][n] / 2
            if -1.0 <= g <= gmax:
                out.append((float(g), float(abs(lat[n]) - (R.hw + st["W"][n] / 2 + 0.3)), n))
    return out


def cut_class(R, r):
    """One inference record with a lead cut -> dict(cls, gap, d, err, obj static / v)."""
    ld = (r.get("fix") or {}).get("lead") or {}
    d = ld.get("d")
    if d is None:
        return dict(cls="no_lead")
    t = r["t0"]
    a = R.pose(t)
    st = R.objs_at(t)
    dmax = d + max(5.0, 0.3 * d)
    A = ahead(R, a, r.get("poses_model") or r["poses"], st, dmax)
    inp = [x for x in A if x[1] <= 0]
    if inp:
        g, _, n = min(inp)
        return dict(cls="real", gap=g, d=d, err=g - d, static=bool(st["static"][n]), v_obj=float(st["v"][n]), grp=TX.cgroup(str(st["cls"][n])))
    adj = [x for x in A if x[1] <= LANE]
    if adj:
        g, _, n = min(adj)
        return dict(cls="adjacent", gap=g, d=d, static=bool(st["static"][n]), v_obj=float(st["v"][n]))
    return dict(cls="nothing", d=d)


# ---------------------------------------------------------------- per scene
def human(R):
    """Human facts from the log: handoff speed, a later stop, lead share, turn."""
    T, P = np.asarray(R.o["log"]["T"], float), np.asarray(R.o["log"]["P"], float)
    v = np.r_[np.hypot(*np.diff(P, axis=0).T) / np.maximum(np.diff(T) * 1e-6, 1e-3), 0.0]
    th = R.t[0] + HAND
    v_hand = float(np.interp(th, T, v))
    stop, i = False, 0
    while i < len(v):
        if v[i] < 0.5 and T[i] > th:
            j = i
            while j < len(v) and v[j] < 0.5:
                j += 1
            if (T[min(j, len(T) - 1)] - T[i]) * 1e-6 >= 1.0:
                stop = True
                break
            i = j
        i += 1
    Y = np.unwrap(np.asarray(R.o["log"]["Y"], float))
    lead, mov = 0, 0
    for j, t in enumerate(R.tg):
        if t < th or float(np.interp(t, T, v)) < 1.0:
            continue
        mov += 1
        a = np.array([np.interp(t, T, P[:, 0]), np.interp(t, T, P[:, 1]), np.interp(t, T, Y)])
        k = (T >= t) & (T <= t + 5e6)
        F = P[k] - a[:2]
        c, s = np.cos(a[2]), np.sin(a[2])
        Pl = np.c_[c * F[:, 0] + s * F[:, 1], -s * F[:, 0] + c * F[:, 1], np.zeros(len(F))] if len(F) else np.zeros((1, 3))
        st = R.tr[j]
        veh = [x for x in ahead(R, a, Pl, st, 30.0) if x[1] <= 0 and TX.cgroup(str(st["cls"][x[2]])) == "vehicle" and x[0] >= 0]
        lead += bool(veh)
    turn = float(np.degrees(Y.max() - Y.min())) if len(Y) else 0.0
    gt = float((R.o["metrics"] or {}).get("gt_dist_traveled_m") or 0.0)
    stratum = ("static" if gt < 1.0 else "launch" if v_hand < 1.0 else "stop/lead" if stop or (mov and lead / mov >= 0.3)
               else "turn" if turn > 30 else "straight")
    return dict(stratum=stratum, v_hand=v_hand, h_stop=stop, lead_share=lead / mov if mov else np.nan, turn_deg=turn)


def scene_job(job):
    s, o, recs = job
    R = TX.Roll(o)
    row, _, _ = TX.classify(o)
    sc = float(o["score"] or 0.0)
    pt = ptype(dict(score=sc, metrics=o["metrics"], failure_reason=""))
    out = dict(scene=s, score=sc, loss=1 - sc, ptype=pt, zclass=zclass(pt) if sc == 0 else "", primary=row.get("primary", ""),
               secondary=row.get("secondary", ""), t_ev_s=row.get("t_ev_s", np.nan), rear_t_s=row.get("rear_t_s", np.nan),
               speed_ratio=row.get("speed_ratio", np.nan), **human(R))
    t_tr = np.nanmin([R.t0 + 1e6 * x for x in (out["t_ev_s"], out["rear_t_s"]) if np.isfinite(x)] + [R.t[-1]])
    cuts = []
    for r in recs:
        ld = (r.get("fix") or {}).get("lead") or {}
        v = r["ego"][4] * 10
        if v < 1 or (ld.get("cut") or 0) <= 0:
            continue
        c = cut_class(R, r)
        cuts.append(dict(scene=s, t_s=(r["t0"] - R.t0) / 1e6, cut=float(ld["cut"]), v=v, before_event=bool(r["t0"] <= t_tr),
                         won=((r.get("fix") or {}).get("acc") or {}).get("won", ""), src=ld.get("src", ""), **c))
    out["n_dec"] = sum(r["ego"][4] * 10 >= 1 for r in recs)
    return out, cuts


# ---------------------------------------------------------------- item 3 helpers
def window(R, t_ev_us):
    """Human time (us) when the log passed A0's ego station at t_ev."""
    s = float(R.log_station(R.pose(t_ev_us)[None, :2])[0])
    return float(np.interp(s, R.LS[:len(R.LT)], R.LT))


def onlog(recs, t_lo, t_hi):
    """H records in [t_lo, t_hi]: median |lateral error at 4 s| of the model plan against the log, median 2 s arc ratio model / log."""
    lat, arc, off = [], [], []
    for r in recs:
        if not (t_lo <= r["t0"] <= t_hi) or r.get("log_poses") is None:
            continue
        L, M = np.asarray(r["log_poses"], float), np.asarray(r["poses_model"] or r["poses"], float)
        n = int(r.get("log_valid") or 8) - 1
        if n < 0:
            continue
        h = L[n, 2] if L.shape[1] > 2 else np.arctan2(*(L[n, :2] - L[max(n - 1, 0), :2])[::-1])
        e = M[n, :2] - L[n, :2]
        lat.append(abs(-np.sin(h) * e[0] + np.cos(h) * e[1]))
        l0 = np.asarray(r.get("log_now") or [0, 0, 0], float)
        off.append(float(np.hypot(*l0[:2])))
        if n >= 3:
            la = float(np.hypot(*(L[0, :2] - l0[:2])) + np.hypot(*np.diff(L[:4, :2], axis=0).T).sum())
            ma = float(np.hypot(*M[0, :2]) + np.hypot(*np.diff(M[:4, :2], axis=0).T).sum())
            if la >= 2.0:
                arc.append(ma / la)
    md = lambda x: float(np.median(x)) if x else np.nan  # noqa: E731
    return dict(n_h=len(lat), lat4=md(lat), arc2=md(arc), off=md(off))


def a0_window(R, recs, t_lo, t_hi):
    """A0 decisions in [t_lo, t_hi]: share of the served-speed terms, served / model and served / human speed (medians)."""
    won, rm, rh = Counter(), [], []
    for r in recs:
        if not (t_lo <= r["t0"] <= t_hi):
            continue
        acc = (r.get("fix") or {}).get("acc") or {}
        won[acc.get("won", "")] += 1
        a = R.pose(r["t0"])
        vh = R.vlog_at(float(R.log_station(a[None, :2])[0]))
        if acc.get("v15") is not None and (acc.get("v_model") or 0) > 0.5:
            rm.append(acc["v15"] / acc["v_model"])
        if acc.get("v15") is not None and vh > 1.0:
            rh.append(acc["v15"] / vh)
    n = sum(won.values())
    return dict(n_a=n, won={k: v / n for k, v in won.items()} if n else {}, served_model=float(np.median(rm)) if rm else np.nan,
                served_human=float(np.median(rh)) if rh else np.nan)


# ---------------------------------------------------------------- main
def traffic_gate(o):
    """Logged traffic (A0.pkl) against the asl actors (A0asl.pkl): nearest-object distance of every asl actor at every 0.5 s step."""
    a, b = rolls(f"{PIL}/tax/A0.pkl"), rolls(f"{o}/tax/A0asl.pkl")
    d = []
    for s in set(a) & set(b):
        for j in range(min(len(a[s]["traffic"]), len(b[s]["traffic"]))):
            A, B = a[s]["traffic"][j], b[s]["traffic"][j]
            if len(B["id"]) and len(A["id"]):
                d += list(np.hypot(B["x"][:, None] - A["x"][None], B["y"][:, None] - A["y"][None]).min(1))
            elif len(B["id"]):
                d += [np.inf] * len(B["id"])
    d = np.asarray(d)
    n = len({s for s in set(a) & set(b) if any(len(x["id"]) for x in b[s]["traffic"])})
    return dict(rollouts=n, pairs=int(len(d)), share_005=float((d <= 0.05).mean()) if len(d) else np.nan,
                p99=float(np.quantile(d[np.isfinite(d)], 0.99)) if np.isfinite(d).any() else np.nan, passed=bool(len(d) and (d <= 0.05).mean() >= 0.99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"]), ap.add_argument("--o", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    NUM, L = {}, ["# PAI-GAP tables (generated by scripts/pai_gap.py)", ""]

    # ---- gate
    if Path(f"{a.o}/tax/A0asl.pkl").exists():
        NUM["gate"] = g = traffic_gate(a.o)
        L += [f"Traffic gate (logged traffic vs rollout.asl actors, {g['rollouts']} A0 rollouts with an asl): {g['pairs']} actor-steps, "
              f"<= 0.05 m on {100 * g['share_005']:.2f} %, p99 {g['p99']:.3f} m -> {'PASS' if g['passed'] else 'FAIL'}", ""]

    # ---- item 1 / 2 on A0
    S0, bad = summaries(A0)
    A = rolls(f"{PIL}/tax/A0.pkl")
    DR = drive(A0)
    jobs = [(s, A[s], DR.get(s, [])) for s in sorted(A) if s not in bad]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(scene_job, jobs, chunksize=4))
    T = pd.DataFrame([x for x, _ in res]).set_index("scene")
    C = pd.DataFrame([c for _, cs in res for c in cs])
    n = len(T)
    # best of reads, N0, H
    for k, root in READS.items():
        Sk, bk = summaries(root)
        T[k] = [float(Sk[s].get("score") or 0) if s in Sk and s not in bk else np.nan for s in T.index]
    T["best"] = T[list(READS)].max(1)
    SN, bN = summaries(f"{PIL}/N0")
    T["N0"] = [float(SN[s].get("score") or 0) if s in SN and s not in bN else np.nan for s in T.index]
    SH, bH = summaries(f"{a.o}/H") if Path(f"{a.o}/H").exists() else ({}, set())
    T["H"] = [float(SH[s].get("score") or 0) if s in SH and s not in bH else np.nan for s in T.index]
    T["H_type"] = [ptype(SH[s]) if s in SH else "" for s in T.index]
    T["cls"] = [f"zero {z}" if sc == 0 else f"partial {p}" if sc < OK else "scored" for z, p, sc in zip(T.zclass, T.ptype, T.score)]
    T["sub"] = [p if sc == 0 and z in ("corridor", "offroad") else "" for p, z, sc in zip(T.primary, T.zclass, T.score)]
    T["h_lost"] = np.minimum(T.loss, 1 - T.H.fillna(1.0))
    T["best_rec"] = (T.best - T.score).clip(lower=0)

    L += [f"## Item 1: A0 = NTV0-L18-s0 + ACC + GAP, {n} scenes, mean {T.score.mean():.4f}, points lost {T.loss.sum():.1f} "
          f"(best of {len(READS)} reads {T.best.mean():.4f}; H read {T.H.notna().sum()} scenes, mean {T.H.mean():.4f})", "",
          "| class | scenes | points lost | human also loses | best-of recovers | points-to-gain (lost - human) | N0 scores >= 0.98 |", "|---|--:|--:|--:|--:|--:|--:|"]
    g = T.groupby("cls")
    tab = pd.DataFrame(dict(n=g.size(), lost=g.loss.sum(), h=g.h_lost.sum(), rec=g.best_rec.sum(), n0ok=g.N0.apply(lambda x: int((x >= OK).sum()))))
    tab["gain"] = tab.lost - tab.h
    for k, x in tab.sort_values("gain", ascending=False).iterrows():
        L.append(f"| {k} | {x.n} | {x.lost:.1f} | {x.h:.1f} | {x.rec:.1f} | **{x.gain:.1f}** | {x.n0ok} |")
        NUM[f"1/class/{k}"] = x.to_dict()
    L += ["", "Zero sub-types (corridor / offroad, rf_tax):", "", "| sub-type | zeros | human also zero | best-of scores |", "|---|--:|--:|--:|"]
    for k, x in T[T["sub"] != ""].groupby("sub"):
        L.append(f"| {k} | {len(x)} | {int((x.H < OK).sum())} | {int((x.best >= OK).sum())} |")
        NUM[f"1/sub/{k}"] = dict(n=len(x), h_zero=int((x.H < OK).sum()))
    L += ["", "Partial primaries (rf_tax):", "", "| partial | primary | scenes | points lost |", "|---|---|--:|--:|"]
    for (p, q), x in T[(T.score > 0) & (T.score < OK)].groupby(["ptype", "primary"]):
        L.append(f"| {p} | {q} | {len(x)} | {x.loss.sum():.1f} |")
    L += ["", "By stratum:", "", "| stratum | scenes | mean | points lost | zeros | lat zeros | lon partial points | human also loses |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for k, x in T.groupby("stratum"):
        L.append(f"| {k} | {len(x)} | {x.score.mean():.3f} | {x.loss.sum():.1f} | {int((x.score == 0).sum())} | "
                 f"{int(x.zclass.isin(LAT_ZERO).sum())} | {x[x.ptype.isin(LON) & (x.score > 0)].loss.sum():.1f} | {x.h_lost.sum():.1f} |")
        NUM[f"1/stratum/{k}"] = dict(n=len(x), mean=x.score.mean(), lost=x.loss.sum())
    L += ["", "Class x stratum (points lost):", ""]
    ct = T.pivot_table(index="cls", columns="stratum", values="loss", aggfunc="sum", fill_value=0).round(1)
    L += ["| class | " + " | ".join(ct.columns) + " |", "|---|" + "--:|" * len(ct.columns)]
    L += [f"| {k} | " + " | ".join(f"{v:.1f}" for v in x) + " |" for k, x in ct.iterrows()]

    # ---- item 2
    if len(C):
        C["phantom"] = C.cls.isin(["adjacent", "nothing", "no_lead"])
        L += ["", f"## Item 2: lead cuts in A0 (inference decisions, ego >= 1 m/s; {T.n_dec.sum()} decisions, {len(C)} with a cut > 0)", "",
              "| subset | decisions | real | adjacent | nothing | no lead reported | phantom share (decisions) | phantom share (cut m) | mean cut m |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for lab, x in (("cut > 0", C), ("cut >= 0.5 m", C[C.cut >= 0.5]), ("cut > 0, before the first event", C[C.before_event]),
                       ("cut > 0, ACC won = lead", C[C.won == "lead"])):
            if not len(x):
                continue
            cc = x.cls.value_counts()
            L.append(f"| {lab} | {len(x)} | {cc.get('real', 0)} | {cc.get('adjacent', 0)} | {cc.get('nothing', 0)} | {cc.get('no_lead', 0)} | "
                     f"{x.phantom.mean():.3f} | {x[x.phantom].cut.sum() / x.cut.sum():.3f} | {x.cut.mean():.2f} |")
            NUM[f"2/{lab}"] = dict(n=len(x), **{k: int(v) for k, v in cc.items()}, phantom=x.phantom.mean(), phantom_m=x[x.phantom].cut.sum() / x.cut.sum())
        R_ = C[C.cls == "real"]
        if len(R_):
            L += ["", f"Real leads: gap - reported d median {R_.err.median():+.2f} m (IQR {R_.err.quantile(0.25):+.2f} .. {R_.err.quantile(0.75):+.2f}), "
                  f"static {R_.static.mean():.2f}, object speed median {R_.v_obj.median():.2f} m/s, vehicle {(R_.grp == 'vehicle').mean():.2f}."]
        pre = C[C.before_event]
        ph = pre.groupby("scene").apply(lambda x: x[x.phantom].cut.sum() / max(x.cut.sum(), 1e-9))
        T["phantom_m_share"] = ph.reindex(T.index)
        lon = T[T.ptype.isin(LON) & (T.score > 0)]
        b1 = lon[lon.phantom_m_share >= 0.5].loss.sum()
        share = pre[pre.phantom].cut.sum() / max(pre.cut.sum(), 1e-9)
        b2 = lon.loss.sum() * share
        NUM["2/worth"] = dict(lon_points=lon.loss.sum(), strict=b1, loose=b2, share=share, n_scenes_strict=int((lon.phantom_m_share >= 0.5).sum()))
        L += ["", f"Worth (ceiling): longitudinal partial points {lon.loss.sum():.1f} in {len(lon)} scenes; scenes whose phantom cuts carry >= 50 % of the "
              f"cut metres before the first event: {int((lon.phantom_m_share >= 0.5).sum())}, {b1:.1f} points; pooled phantom share {share:.3f} x "
              f"longitudinal points = {b2:.1f} points (+{b2 / n:.4f} mean at most)."]
        C.to_csv(f"{a.o}/cuts.csv.gz", index=False)

    # ---- item 3
    if T.H.notna().any() and Path(f"{a.o}/tax/H.pkl").exists():
        HR = drive(f"{a.o}/H")
        Hrolls = rolls(f"{a.o}/tax/H.pkl")
        t0d = [HR[s][0]["t0"] - DR[s][0]["t0"] for s in T.index if s in HR and DR.get(s)]
        NUM["3/clock"] = dict(n=len(t0d), max_abs_us=float(np.max(np.abs(t0d))) if t0d else np.nan)
        H_off = [float(np.median([np.hypot(*np.asarray(r["log_now"])[:2]) for r in HR[s] if r.get("log_now") is not None and r["k"] >= 20] or [np.nan]))
                 for s in T.index if s in HR]
        rows = []
        for s, x in T[T.score < OK].iterrows():
            R = TX.Roll(A[s])
            lon_ = x.ptype in LON and x.score > 0
            if np.isfinite(x.t_ev_s) and x.score == 0:
                te = R.t0 + 1e6 * x.t_ev_s
            elif np.isfinite(x.rear_t_s):
                te = R.t0 + 1e6 * x.rear_t_s
            else:
                te = R.t[-1]
            th = window(R, te)
            hw = onlog(HR.get(s, []), th - (4e6 if not (lon_ and x.ptype != "rear-hit") else 30e6), th)
            if lon_ and x.ptype != "rear-hit":                               # slow / creep / no-launch: the whole valid rollout
                aw = a0_window(R, DR.get(s, []), R.t[0] + HAND, te)
            else:
                aw = a0_window(R, DR.get(s, []), te - 3e6, te)
            lat = x.zclass in LAT_ZERO
            if 1 - (x.H if np.isfinite(x.H) else 1.0) >= 0.5:
                at = "iii"
            elif lat and x.N0 >= OK and aw["served_model"] >= 1.1 and aw["served_human"] >= 1.1:
                at = "ii"
            elif lon_ and (aw["won"].get("lead", 0) + aw["won"].get("curve", 0)) >= 0.5:
                at = "ii"
            elif lat and np.isfinite(hw["lat4"]):
                at = "i" if hw["lat4"] >= 1.0 else "i-cl"
            elif lon_ and np.isfinite(hw["arc2"]) and hw["arc2"] < 0.9:
                at = "i"
            else:
                at = "mixed"
            rows.append(dict(scene=s, cls=x.cls, sub=x["sub"], loss=x.loss, H=x.H, H_type=x.H_type, N0=x.N0, attr=at, **hw,
                             served_model=aw["served_model"], served_human=aw["served_human"], **{f"won_{k}": v for k, v in aw["won"].items()}))
        AT = pd.DataFrame(rows)
        AT.to_csv(f"{a.o}/attr.csv", index=False)
        L += ["", f"## Item 3: attribution (H decisions in the 4 s before A0's event station; slow / creep / no-launch: the whole rollout). "
              f"Clock check A0 vs H first t0: max |diff| {NUM['3/clock']['max_abs_us']:.0f} us over {len(t0d)} scenes; "
              f"H ego off the log (median |log_now| at k >= 20, per scene): median {np.nanmedian(H_off):.2f} m, p90 {np.nanquantile(H_off, 0.9):.2f} m", "",
              "| class | scenes | points | (i) plan wrong on the human's state | (i-cl) right on the human's, wrong on its own | (ii) serve | (iii) human also loses | mixed | median lat4 on H (m) | median arc2 model / log on H |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for k, x in AT.groupby("cls"):
            p = x.groupby("attr").loss.sum()
            L.append(f"| {k} | {len(x)} | {x.loss.sum():.1f} | " + " | ".join(f"{p.get(q, 0):.1f}" for q in ("i", "i-cl", "ii", "iii", "mixed"))
                     + f" | {x.lat4.median():.2f} | {x.arc2.median():.2f} |")
            NUM[f"3/{k}"] = dict(n=len(x), points=x.loss.sum(), **{q: float(p.get(q, 0)) for q in ("i", "i-cl", "ii", "iii", "mixed")},
                                 lat4=x.lat4.median(), arc2=x.arc2.median())
        L += ["", "Corridor / offroad sub-types:", "", "| sub-type | zeros | i | i-cl | ii | iii | mixed | lat4 median |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for k, x in AT[AT["sub"] != ""].groupby("sub"):
            c = x.attr.value_counts()
            L.append(f"| {k} | {len(x)} | " + " | ".join(str(c.get(q, 0)) for q in ("i", "i-cl", "ii", "iii", "mixed")) + f" | {x.lat4.median():.2f} |")
        W = [c for c in AT.columns if c.startswith("won_")]
        lp = AT[AT.cls.str.startswith("partial")]
        if len(lp):
            L += ["", "Longitudinal partials: served-speed term shares (A0 window) and on-log arc ratio:", "",
                  "| partial | scenes | " + " | ".join(w[4:] for w in W) + " | served / human | arc2 on H |", "|---|--:|" + "--:|" * (len(W) + 2)]
            for k, x in lp.groupby("cls"):
                L.append(f"| {k} | {len(x)} | " + " | ".join(f"{x[w].fillna(0).mean():.2f}" for w in W) + f" | {x.served_human.median():.2f} | {x.arc2.median():.2f} |")
        # H's own losses and the on-log lead cuts (sanity: cuts at the human's state)
        HC = []
        for s, recs in HR.items():
            if s not in Hrolls:
                continue
            R = TX.Roll(Hrolls[s])
            for r in recs:
                ld = (r.get("fix") or {}).get("lead") or {}
                if r["ego"][4] * 10 >= 1 and (ld.get("cut") or 0) > 0 and r.get("k", 0) >= 17:
                    HC.append(dict(scene=s, cut=float(ld["cut"]), **cut_class(R, r)))
        HC = pd.DataFrame(HC)
        if len(HC):
            cc = HC.cls.value_counts()
            ph = HC.cls.isin(["adjacent", "nothing", "no_lead"])
            NUM["2/H"] = dict(n=len(HC), **{k: int(v) for k, v in cc.items()}, phantom=ph.mean(), phantom_m=HC[ph].cut.sum() / HC.cut.sum())
            L += ["", f"Lead cuts at the human's own states (H, k >= 17, ego >= 1 m/s): {len(HC)} decisions; real {cc.get('real', 0)}, adjacent "
                  f"{cc.get('adjacent', 0)}, nothing {cc.get('nothing', 0)}, no lead {cc.get('no_lead', 0)}; phantom share {ph.mean():.3f} "
                  f"(cut m {HC[ph].cut.sum() / HC.cut.sum():.3f})."]
        L += ["", "H's own losses: " + ", ".join(f"{k} {v}" for k, v in Counter(T.H_type[T.H < OK]).items())]
    T.to_csv(f"{a.o}/scenes.csv")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    (out / "pai_gap.json").write_text(json.dumps(NUM, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
