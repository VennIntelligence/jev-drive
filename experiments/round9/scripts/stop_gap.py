#!/usr/bin/env python3
"""STOP-GAP: rest gap to the lead, creep at standstill and what they cost on PAI 281 under ACC + GAP (0.7, 2.7). Stored reads only, CPU.

  scan    (box, repo .venv)  stop_gap.py scan [--reads NAME ...]  -> $DATA_DIR/runs/round9/stop_gap/{scenes.csv, decs.csv.gz}
  report  (box or Mac)       stop_gap.py report --src <dir with scenes.csv, decs.csv.gz> --out experiments/round9/results/stop_gap

Definitions (all on the simulator's ground truth; the rf_tax pkl carries the ego at 0.1 s and the logged traffic at 0.5 s, which on PAI is
what the simulator replays, pai_gap's traffic gate):
- standstill onset: the first time after the handoff (t0 + 1.6 s) that the ego is below V_STILL for >= T_STILL.
- lead at a time: the nearest vehicle in the ego's path within 30 m ahead of the front bumper along the heading line (pai_gap.ahead, as
  pai_gap.rest_cue does for the human's rest gap): true gap = front bumper to the lead's rear.
- lead moves: the lead's position leaves its onset position by > 1 m (0.5 s traffic steps, linear interpolation of the time).
- creep: ego path length from the standstill onset to min(lead moves, first event, end of rollout); for no-lead stops to min(event, end).
- reading: the served lead (fix.lead.d = model lead distance - camera-to-front, None when radard's filtered probability <= 0.5).
- creep attribution: every 0.1 s ego displacement in the creep window is charged to the latest decision's served state: `ref0` (served
  2 s arc < 0.1 m: the reference does not move, the controller creeps), else ACC's binding term at 1.5 s (`won`: model = the stop guard
  passes the model's own profile, lead = the lead MPC cap, set / curve = guard released and ACC's cruise target, ramp).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/defect_fix/scripts"),
                str(REPO / "experiments/wait_cap/scripts")]

D = os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs") + "/runs"
OUT = f"{D}/round9/stop_gap"
READS = {  # name -> (read root, rf_tax pkl / pai_fix facts stem)
    "R6DAG-s0": (f"{D}/round6/dag/pai281/R6DAG-L18-s0-ag", "R6DAG-L18-s0-281ag"),
    "R6DAG-s1": (f"{D}/round6/dag/pai281/R6DAG-L18-s1-ag", "R6DAG-L18-s1-281ag"),
    "R6DAG-s2": (f"{D}/round8/s2/pai281/R6DAG-L18-s2-ag", "R6DAG-L18-s2-281ag"),
    "R8K-s2": (f"{D}/round8/s2/pai281/R8K-L18-s2-ag", "R8K-L18-s2-281ag"),
    "NTV0-s0": (f"{D}/defect_fix/pai_accgap/NTV0-L18-s0", "NTV0-L18-s0-281ag"),
}
HAND = 1.6e6
V_STILL, T_STILL = 0.2, 1.0
GMAX, LEAD_MOVE = 30.0, 1.0
STOP_D = 2.7                       # served JEV_GAP stop distance


# ---------------------------------------------------------------- scan (box)
def arc2(P):
    """2 s arc length of a pose list (0.5 s steps from the rig origin)."""
    if P is None or not len(P):
        return np.nan
    P = np.asarray(P, float)[:4, :2]
    return float(np.hypot(*P[0]) + np.hypot(*np.diff(P, axis=0).T).sum())


def load_drive(f):
    out = defaultdict(list)
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") != "drive" or not r.get("infer"):
            continue
        fx = r.get("fix") or {}
        ld, ac = fx.get("lead") or {}, fx.get("acc") or {}
        out[r["scene"]].append(dict(t=r["t0"], v=r["ego"][4] * 10, d=ld.get("d"), vl=ld.get("vl"), p=(ld.get("p") or [np.nan])[0],
                                    cut=ld.get("cut"), guard=ac.get("guard", ""), won=ac.get("won", ""), v15=ac.get("v15"),
                                    vmod=ac.get("v_model"), vlead=ac.get("v_lead"), vset=ac.get("v_set"),
                                    a_mod=arc2(r.get("poses_model")), a_srv=arc2(r.get("poses_track") or r.get("poses"))))
    return out


def lead_at(R, t, ids=None):
    """Nearest in-path vehicle ahead at time t (heading line) -> (gap, id, v) or (nan, '', nan); ids: restrict to these ids."""
    import pai_gap as PG
    import rf_tax as TX
    a = R.pose(t)
    st = R.objs_at(t)
    c = [x for x in PG.ahead(R, a, np.zeros((1, 3)), st, GMAX) if x[1] <= 0 and x[0] >= -1.0
         and TX.cgroup(str(st["cls"][x[2]])) == "vehicle" and (ids is None or str(st["id"][x[2]]) in ids)]
    if not c:
        return np.nan, "", np.nan
    g, _, n = min(c)
    return float(g), str(st["id"][n]), float(st["v"][n])


def track(R, oid):
    """(t, x, y) of object oid over the 0.5 s traffic steps."""
    T, X = [], []
    for t, st in zip(R.tg, R.tr):
        m = np.flatnonzero(np.asarray(st["id"]).astype(str) == oid)
        if len(m):
            T.append(t), X.append((st["x"][m[0]], st["y"][m[0]]))
    return np.asarray(T, float), np.asarray(X, float).reshape(-1, 2)


def follower(R):
    """The logged vehicle behind the human's rest pose (last traffic step), in its lane: gap from its front to the ego box's rear."""
    L = R.o["log"]
    a = np.array([L["P"][-1][0], L["P"][-1][1], L["Y"][-1]])
    st = R.tr[-1]
    if not len(st["id"]):
        return np.nan
    c, s = np.cos(a[2]), np.sin(a[2])
    D_ = np.c_[st["x"], st["y"]] - a[:2]
    ex, ey = c * D_[:, 0] + s * D_[:, 1], -s * D_[:, 0] + c * D_[:, 1]
    g = R.rear - (ex + np.asarray(st["L"]) / 2)
    import rf_tax as TX
    ok = (ex < R.rear) & (np.abs(ey) <= R.hw + np.asarray(st["W"]) / 2) & (g <= 30) & np.array([TX.cgroup(str(k)) == "vehicle" for k in st["cls"]])
    return float(g[ok].min()) if ok.any() else np.nan


def stills(R, t_from):
    """Standstill onsets (t) after t_from: v < V_STILL for >= T_STILL."""
    out, i, n = [], 0, len(R.t)
    while i < n:
        if R.v[i] < V_STILL and R.t[i] >= t_from:
            j = i
            while j < n and R.v[j] < V_STILL:
                j += 1
            if R.t[min(j, n - 1)] - R.t[i] >= T_STILL * 1e6 - 1:
                out.append(float(R.t[i]))
            i = j
        i += 1
    return out


def path_len(R, t1, t2):
    m = (R.t >= t1) & (R.t <= t2)
    return float(np.hypot(*np.diff(np.c_[R.x[m], R.y[m]], axis=0).T).sum()) if m.sum() > 1 else 0.0


def scene_job(job):
    import rf_tax as TX
    name, s, o, recs, F = job
    R = TX.Roll(o)
    t_hand = R.t[0] + HAND
    ev = [R.t0 + 1e6 * x for x in (F.get("t_ev_s"), F.get("rear_t_s")) if x is not None and np.isfinite(x)]
    t_ev = R.t0 + 1e6 * F["t_ev_s"] if F.get("t_ev_s") is not None and np.isfinite(F["t_ev_s"]) else np.nan
    t_rear = R.t0 + 1e6 * F["rear_t_s"] if F.get("rear_t_s") is not None and np.isfinite(F["rear_t_s"]) else np.nan
    t_end = R.t[-1]
    st_on = stills(R, t_hand)
    LX = np.vstack([R.LP, R.LP[-1] + np.outer(np.arange(1, 41) * 5.0, np.r_[np.cos(o["log"]["Y"][-1]), np.sin(o["log"]["Y"][-1])])])
    s_hrest = float(TX.poly_proj(LX, np.asarray(o["log"]["P"][-1:], float))[1][0])
    sta = lambda t: float(TX.poly_proj(LX, R.pose(t)[None, :2])[1][0])  # noqa: E731   station on the log path extended 200 m
    RT = np.array([r["t"] for r in recs]) if recs else np.zeros(0)
    rec_at = lambda t: recs[int(np.clip(np.searchsorted(RT, t, side="right") - 1, 0, len(recs) - 1))] if len(recs) else {}  # noqa: E731
    row = dict(read=name, scene=s, n_still=len(st_on), s_hrest=s_hrest, f_gap=follower(R), t_ev=(t_ev - R.t0) / 1e6, t_rear=(t_rear - R.t0) / 1e6,
               v_ev=R.speed(t_ev) if np.isfinite(t_ev) else np.nan, d_end=sta(t_end) - s_hrest, still_end=bool(R.v[-1] < V_STILL))
    g_end, _, _ = lead_at(R, t_end)
    row.update(g_end=g_end, rd_end=rec_at(t_end).get("d"))
    if np.isfinite(t_ev):                                   # the event: partner, speed, reading against truth over the last 2 s
        g_e, id_e, v_e = lead_at(R, t_ev - 0.3e6)
        row.update(ev_gap=g_e, ev_id=id_e, ev_obj_v=v_e, ev_d_hrest=sta(t_ev) - s_hrest)
        er = [(r["d"], lead_at(R, r["t"])[0], r["v"]) for r in recs if t_ev - 2e6 <= r["t"] <= t_ev]
        row.update(ev_nread=len(er), ev_noread=float(np.mean([d is None for d, _, _ in er])) if er else np.nan,
                   ev_err=float(np.nanmedian([d - g for d, g, _ in er if d is not None and np.isfinite(g)])) if any(d is not None and np.isfinite(g) for d, g, _ in er) else np.nan)
    decs, eps = [], []
    for e, t1 in enumerate(st_on):                         # every standstill episode: rest gap, reading, creep until the lead moves
        g0, oid, vo = lead_at(R, t1)
        r1 = rec_at(t1)
        ep = dict(read=name, scene=s, ep=e, t_still=(t1 - R.t0) / 1e6, v_in=float(np.interp(t1 - 2e6, R.t, R.v)), d_still=sta(t1) - s_hrest,
                  g_still=g0, lead_id=oid, lead_v=vo, rd_still=r1.get("d"), vl_still=r1.get("vl"), won_still=r1.get("won", ""),
                  guard_still=r1.get("guard", ""), before_ev=bool(np.isfinite(t_ev) and t1 < t_ev))
        t_lm = np.nan
        if oid:
            T, X = track(R, oid)
            k = T >= R.tg[max(np.searchsorted(R.tg, t1) - 1, 0)]
            if k.any():
                dx = np.hypot(*(X[k] - X[k][0]).T)
                j = np.flatnonzero(dx > LEAD_MOVE)
                if len(j):
                    i = j[0]
                    t_lm = float(np.interp(LEAD_MOVE, [dx[i - 1], dx[i]], [T[k][i - 1], T[k][i]])) if i else float(T[k][0])
        nxt = st_on[e + 1] if e + 1 < len(st_on) else np.inf
        tw = min([t_end, t_lm, nxt] + [x for x in ev if x > t1], key=lambda x: x if np.isfinite(x) else np.inf)
        ep.update(t_lm=(t_lm - R.t0) / 1e6, t_win=(tw - t1) / 1e6, creep=path_len(R, t1, tw), d_win=sta(tw) - s_hrest,
                  win_end="event" if any(abs(tw - x) < 1 for x in ev) else "lead_moves" if tw == t_lm else "next_stop" if tw == nxt else "end")
        if oid:                                             # true gap to the same lead over the window
            gs = [lead_at(R, t, {oid})[0] for t in np.arange(t1, tw + 1, 0.5e6)]
            gs = [g for g in gs if np.isfinite(g)]
            ep.update(g_min=min(gs) if gs else np.nan, g_win=lead_at(R, tw, {oid})[0])
        m = (R.t >= t1) & (R.t <= tw)                       # creep attribution
        ti, xi, yi = R.t[m], R.x[m], R.y[m]
        att = defaultdict(float)
        for i in range(1, len(ti)):
            r = rec_at(ti[i - 1])
            lab = "ref0" if np.isfinite(r.get("a_srv", np.nan)) and r["a_srv"] < 0.1 else (r.get("won") or "none")
            att[lab] += float(np.hypot(xi[i] - xi[i - 1], yi[i] - yi[i - 1]))
        ep.update({f"cr_{k}": v for k, v in att.items()})
        eps.append(ep)
        for r in recs:
            if t1 <= r["t"] <= min(tw, t1 + 30e6):
                gt = lead_at(R, r["t"], {oid})[0] if oid else np.nan
                decs.append(dict(read=name, scene=s, ep=e, ts=(r["t"] - t1) / 1e6, v=r["v"], g=gt, **{k: r[k] for k in
                                 ("d", "vl", "p", "cut", "guard", "won", "v15", "vmod", "vlead", "a_mod", "a_srv")}))
    return row, eps, decs


def cmd_scan(a):
    from jevdrive.common import n_cpus
    os.makedirs(OUT, exist_ok=True)
    rows, eps, decs = [], [], []
    W = max(2, min(n_cpus() // 2, 24))
    for name in a.reads:
        root, stem = READS[name]
        F = pd.read_csv(f"{D}/pai_fix/facts/{stem}.csv").set_index("scene")
        A = {o["scene"]: o for o in pickle.load(open(f"{D}/pai_fix/tax/{stem}.pkl", "rb"))["rolls"] if "ctrl" in o}
        with ProcessPoolExecutor(W) as ex:
            DR = defaultdict(list)
            for d in ex.map(load_drive, glob.glob(f"{root}/runs/*/driver/drive.jsonl")):
                for k, v in d.items():
                    DR[k] += v
        DR = {k: sorted(v, key=lambda r: r["t"]) for k, v in DR.items()}
        jobs = [(name, s, A[s], DR.get(s, []), F.loc[s].to_dict()) for s in sorted(A) if s in F.index][:a.limit or None]
        with ProcessPoolExecutor(W) as ex:
            for r, e, d in ex.map(scene_job, jobs, chunksize=2):
                rows.append(r), eps.extend(e), decs.extend(d)
        print(name, len(jobs), "scenes", flush=True)
    if a.limit:
        print(pd.DataFrame(eps).T.to_string()[:12000]); return
    pd.DataFrame(rows).to_csv(f"{OUT}/scenes.csv", index=False, float_format="%.4f")
    pd.DataFrame(eps).to_csv(f"{OUT}/episodes.csv", index=False, float_format="%.4f")
    pd.DataFrame(decs).to_csv(f"{OUT}/decs.csv.gz", index=False, float_format="%.3f")
    print("->", OUT)


# ---------------------------------------------------------------- report (Mac or box)
FACTS = {k: v for k, (_, v) in READS.items()}
GX = {4.0: 4.5, 5.0: 5.5, 6.0: 6.8}     # stop distance X -> true rest gap where the standstill reading first drops below X (decs, v < 0.3)
MARGIN = 0.5                            # m: the replay follower hits the ego when it rests more than (follower gap - MARGIN) behind the human


def q(x, p=(0.1, 0.5, 0.9), f="{:.2f}"):
    x = pd.Series(x).dropna()
    return " / ".join(f.format(v) for v in x.quantile(list(p))) + f" (n {len(x)})" if len(x) else "-"


def cmd_report(a):
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fdir = Path(a.facts) if a.facts else Path(f"{D}/pai_fix/facts")
    F = pd.concat([pd.read_csv(fdir / f"{v}.csv").assign(read=k) for k, v in FACTS.items()])
    S = pd.read_csv(src / "scenes.csv").merge(F, on=["read", "scene"])
    E = pd.read_csv(src / "episodes.csv").merge(F[["read", "scene", "score", "loss", "fclass", "primary", "stratum", "v_hand", "h_end_still",
                                                     "rest_lead", "rest_lead_gap"]], on=["read", "scene"])
    Dd = pd.read_csv(src / "decs.csv.gz")
    nR = S.read.nunique()
    E["t_ev"] = S.set_index(["read", "scene"]).t_ev.reindex(list(zip(E.read, E.scene))).values
    E["closure"] = (E.g_still - E.g_min).clip(lower=0)
    L, J = ["# STOP-GAP tables (generated by stop_gap.py report)", ""], {}
    H = F[F.read == "R6DAG-s0"].set_index("scene")
    endL = H.h_end_still.astype(bool) & H.rest_lead.astype(bool)
    L += [f"Human: a vehicle within 15 m ahead at the log's end (`rest_lead`) in {int(H.rest_lead.sum())} scenes, gap p10 / p50 / p90 "
          f"{q(H.rest_lead_gap[H.rest_lead.astype(bool)])}; of those the human is still at the end in {int(endL.sum())}: {q(H.rest_lead_gap[endL])}; "
          f"moving at the handoff (>= 1 m/s) {int((endL & (H.v_hand >= 1)).sum())}.", ""]
    # ---- 1: rest gap in the human-rest-behind-lead scenes
    pre = E[(E.g_still <= 15) & (E.t_still < E.t_ev.fillna(np.inf))]
    R = pre.sort_values("t_still").groupby(["read", "scene"]).tail(1)
    P = R[R.h_end_still.astype(bool) & R.rest_lead.astype(bool)].copy()
    P["diff"] = P.g_min - P.rest_lead_gap
    L += ["## 1. Rest gap behind the lead (human still at the end with a lead; the ego's last standstill behind a lead before any event)", "",
          f"{len(P)} scene-reads ({P.scene.nunique()} scenes of {int(endL.sum())}; per read " + ", ".join(f"{k} {v}" for k, v in P.groupby('read').size().items()) + ").", "",
          "| quantity | p10 / p50 / p90 |", "|---|---|",
          f"| ego true gap at the standstill onset (m) | {q(P.g_still)} |", f"| ego lead reading at the onset (m) | {q(P.rd_still)} |",
          f"| ego true gap, minimum before the lead moves (m) | {q(P.g_min)} |", f"| human rest gap, same scenes (m) | {q(P.rest_lead_gap)} |",
          f"| ego minus human (m, paired) | {q(P['diff'])} |", f"| ego station at the onset minus the human's rest station (m; + = ahead) | {q(P.d_still)} |",
          f"| same, at the end of the creep window (m) | {q(P.d_win)} |", ""]
    mo = P[P.v_in > 1]
    L += [f"Stops from motion only (ego > 1 m/s 2 s before the onset): {len(mo)} scene-reads; ego min gap {q(mo.g_min)}, human {q(mo.rest_lead_gap)}, "
          f"ego - human {q(mo.g_min - mo.rest_lead_gap)}; ego - human station {q(mo.d_win)}; true gap < 1 m in {int((mo.g_min < 1).sum())}, < 2 m in {int((mo.g_min < 2).sum())}.", ""]
    J["1"] = dict(n=len(P), scenes=int(P.scene.nunique()), ego_gmin_med=P.g_min.median(), human_med=P.rest_lead_gap.median(), diff_med=P["diff"].median(),
                  d_win_med=P.d_win.median(), gmin_lt1=int((P.g_min < 1).sum()))
    per = P.groupby("scene").agg(human=("rest_lead_gap", "first"), ego_onset=("g_still", "median"), ego_min=("g_min", "median"), reading=("rd_still", "median"),
                                 creep=("creep", "median"), d_rest=("d_win", "median"), reads=("read", "size"), fclass=("fclass", lambda x: ",".join(sorted(set(x) - {"ok"}))))
    per.index = per.index.str[7:15]
    per.round(2).to_csv(out / "rest_gap_scenes.csv")
    L += ["Per scene (median over the reads that stop there): [rest_gap_scenes.csv](rest_gap_scenes.csv).", "",
          "| scene | human | ego onset | ego min | reading | creep | ego - human station | reads | classes |", "|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    L += [f"| {s} | {r.human:.2f} | {r.ego_onset:.2f} | {r.ego_min:.2f} | {r.reading:.2f} | {r.creep:.2f} | {r.d_rest:+.2f} | {r.reads} | {r.fclass} |"
          for s, r in per.sort_values("human").iterrows()]
    # reading against truth at standstill
    x = Dd[Dd.g.notna() & Dd.d.notna() & (Dd.v < 0.3)]
    tb = x.groupby(pd.cut(x.g, [-1, 1, 2, 3, 4, 5, 6, 8, 15]), observed=True).d.describe()[["count", "25%", "50%", "75%"]]
    L += ["", f"Lead reading at standstill (ego < 0.3 m/s, {len(x)} decisions in the creep windows) against the true gap:", "",
          "| true gap (m) | decisions | reading p25 / p50 / p75 (m) |", "|---|--:|---|"]
    L += [f"| {k} | {int(r['count'])} | {r['25%']:.2f} / {r['50%']:.2f} / {r['75%']:.2f} |" for k, r in tb.iterrows()]
    inv = {X: x.g[(x.d - X).abs() < 0.25] for X in (2.7, 3.5, 4.0, 5.0, 6.0)}
    L += ["", "True gap when the reading is X +- 0.25 m (p10 / p50 / p90): " + "; ".join(f"X {k:g}: {q(v)}" for k, v in inv.items()), ""]
    J["reading_true_at_X"] = {str(k): float(v.median()) for k, v in inv.items()}
    # ---- 2: creep
    A = pre.assign(kind=np.where(pre.v_in > 1, "stop from motion", "standing at the handoff / queue"))
    A = A[A.creep <= A.g_still + 1.0]                       # drop windows where the ego drives past the object (not a lead)
    L += ["## 2. Creep after the standstill onset, before the lead moves (lead within 15 m at the onset)", "",
          "| episodes | n | path p25 / p50 / p75 / p90 (m) | > 0.5 m | > 1 m | gap closure p50 / p90 (m) | window ends: lead moves / end / event / next stop | metres by served term |",
          "|---|--:|---|--:|--:|---|---|---|"]
    for k, y in A.groupby("kind"):
        cr = y.filter(like="cr_").sum()
        we = y.win_end.value_counts()
        L.append(f"| {k} | {len(y)} | {q(y.creep, (0.25, 0.5, 0.75, 0.9)).split(' (')[0]} | {(y.creep > 0.5).mean():.2f} | {(y.creep > 1).mean():.2f} | "
                 f"{y.closure.median():.2f} / {y.closure.quantile(0.9):.2f} | {we.get('lead_moves', 0)} / {we.get('end', 0)} / {we.get('event', 0)} / {we.get('next_stop', 0)} | "
                 + ", ".join(f"{c[3:]} {v / cr.sum():.2f}" for c, v in cr.sort_values(ascending=False).items() if v > 0) + f" (sum {cr.sum():.1f} m) |")
        J[f"2/{k}"] = dict(n=len(y), creep_med=y.creep.median(), creep_p90=y.creep.quantile(0.9), gt05=(y.creep > 0.5).mean(), share={c[3:]: v / cr.sum() for c, v in cr.items()})
    xs = x[(x.won == "lead")]
    L += ["", f"Served state while standing (decisions above): binding term lead {(x.won == 'lead').mean():.2f}, model {(x.won == 'model').mean():.2f}; "
          f"lead speed read p10 / p50 / p90 {q(x.vl)}; served 2 s arc {q(x.a_srv)} m; the model's own 2 s arc {q(x.a_mod)} m; "
          f"ego speed {q(x.v)} m/s; reading where the lead term binds {q(xs.d)}.", ""]
    N = E[(E.g_still.isna() | (E.g_still > 15)) & (E.v_in > 1) & (E.t_still < E.t_ev.fillna(np.inf))]
    N = N[~N.rest_lead.astype(bool) & N.h_end_still.astype(bool)]
    L += [f"Stops without a lead (human still at the end, no vehicle within 15 m; ego stops from motion): {len(N)} episodes, creep to the end / event "
          f"p50 / p90 {N.creep.median():.2f} / {N.creep.quantile(0.9):.2f} m; ego station at the onset minus the human's rest {q(N.d_still)} m.", ""]
    # ---- 3: losses
    Af = S[S.fclass == "AF-front"].copy()
    def af_kind(r):
        e = E[(E.read == r.read) & (E.scene == r.scene) & (E.t_still < r.t_ev)]
        if len(e) and e.iloc[-1].win_end == "event" and e.iloc[-1].g_still <= 15:
            return "after standstill + creep (lead not moved)"
        if not np.isfinite(r.ev_gap) if r.ev_gap is not None else True:
            return "partner not on the heading line (cut-in / turn / lateral)"
        if r.primary == "coll_veh_oncoming":
            return "oncoming"
        if r.v_ev < 2:
            return "slow approach, close-range over-read"
        if (r.ev_noread or 0) >= 0.5 or (r.ev_err or 0) > 2:
            return "lead misread at speed (no reading / reads long)"
        return "approach overspeed (reading ok)"
    Af["kind"] = [af_kind(r) for _, r in Af.iterrows()]
    L += ["## 3. Loss attribution", "", f"AF-front zeros ({len(Af)} over {nR} reads):", "",
          "| read | scene | stratum | primary | speed at the event (m/s) | partner speed | reading err (m) | no-reading share | kind |", "|---|---|---|---|--:|--:|--:|--:|---|"]
    L += [f"| {r.read} | {r.scene[7:15]} | {r.stratum} | {r.primary} | {r.v_ev:.2f} | {r.ev_obj_v:.2f} | {r.ev_err:.2f} | {r.ev_noread:.2f} | {r.kind} |"
          for _, r in Af.sort_values(["kind", "scene"]).iterrows()]
    J["3/af"] = Af.kind.value_counts().to_dict()
    L += ["", "Kinds: " + ", ".join(f"{k} {v}" for k, v in Af.kind.value_counts().items()), ""]
    O = S[S.fclass == "overshoot"].copy()
    O["stopped"] = [bool(((E.read == r.read) & (E.scene == r.scene) & (E.t_still < r.t_ev)).any()) for _, r in O.iterrows()]
    L += [f"Stop overshoot zeros (cor_route_end): {len(O)} over {nR} reads ({O.scene.nunique()} scenes); stopped (< {V_STILL} m/s for >= {T_STILL:g} s) "
          f"before the event: {int(O.stopped.sum())}; ego speed at the event {q(O.v_ev)} m/s, {q(O.ev_d_hrest)} m past the human's rest point; "
          f"a lead within 15 m of the human's rest point in {int(O.rest_lead.astype(bool).sum())}.", ""]
    J["3/overshoot"] = dict(n=len(O), stopped=int(O.stopped.sum()), v_med=O.v_ev.median())
    last = E.sort_values("t_still").groupby(["read", "scene"]).tail(1).set_index(["read", "scene"])
    S2 = S.set_index(["read", "scene"])
    S2["d_rest"], S2["rear"] = last.d_win, S2.fclass == "rear-hit"
    X2 = S2[S2.h_end_still.astype(bool) & S2.d_rest.notna() & (S2.v_hand >= 1)]
    g = X2.groupby(pd.cut(X2.d_rest, [-99, -3, -1, 0, 1, 3, 99]), observed=True).agg(n=("rear", "size"), rear=("rear", "sum"))
    Rh = S2[S2.rear]
    L += ["Rear-hit partials against the ego's rest position (human moving at the handoff and still at the end; the ego's last standstill; "
          "station at the end of the creep window minus the human's rest station):", "", "| ego - human station (m) | scene-reads | rear-hit |", "|---|--:|--:|"]
    L += [f"| {k} | {r.n} | {r.rear} |" for k, r in g.iterrows()]
    fol = X2.f_gap.notna() & (X2.f_gap < 15)
    L += ["", f"With a logged follower within 15 m behind the human's rest pose and the ego behind the human: {int((fol & (X2.d_rest < 0)).sum())} scene-reads, "
          f"rear-hit {int((fol & (X2.d_rest < 0) & X2.rear).sum())}. All rear-hit partials: {len(Rh)} (strata " + ", ".join(f"{k} {v}" for k, v in Rh.stratum.value_counts().items())
          + f"); in scenes where the human ends still {int(Rh.h_end_still.astype(bool).sum())}, of them launch {int((Rh.h_end_still.astype(bool) & (Rh.stratum == 'launch')).sum())}. "
          f"Mean loss of a rear-hit partial {Rh.loss.mean():.3f}.", ""]
    J["3/rear"] = dict(n=len(Rh), stop_from_motion_rear=int(X2.rear.sum()), stop_from_motion_n=len(X2), loss_mean=Rh.loss.mean())
    # ---- 4: counterfactual
    lrh = float(Rh.loss.mean())
    C = R.merge(S[["read", "scene", "f_gap", "fclass", "t_rear"]], on=["read", "scene"], suffixes=("", "_s"))
    C = C[C.h_end_still.astype(bool) & (C.v_in > 1)]          # stops from motion where the human ends still (queue-at-handoff launches excluded)
    now = C.f_gap.notna() & (C.d_win < -(C.f_gap - MARGIN))
    Call = R.merge(S[["read", "scene", "f_gap", "t_rear"]], on=["read", "scene"], suffixes=("", "_s"))
    Call = Call[Call.h_end_still.astype(bool) | Call.rest_lead.astype(bool)]
    pr, ob = Call.f_gap.notna() & (Call.d_win < -(Call.f_gap - MARGIN)), Call.t_rear.notna()
    J["4/rule_check"] = dict(tp=int((pr & ob).sum()), fp=int((pr & ~ob).sum()), fn=int((~pr & ob).sum()), tn=int((~pr & ~ob).sum()))
    C = C[~now]
    rows = []
    for X, gx in GX.items():
        for lab, gxx in (("p50", gx), ("low", gx - 0.7), ("high", gx + 0.7)):
            sh = (gxx - C.g_min).clip(lower=0)
            nd = C.d_win - sh
            risk = C.f_gap.notna() & (nd < -(C.f_gap - MARGIN)) & C.t_rear.isna()
            rows.append(dict(X=X, case=lab, shifted=int((sh > 0).sum()), shift_med=float(sh[sh > 0].median()) if (sh > 0).any() else 0.0,
                             new_rear=int(risk.sum()), scenes=",".join(sorted(C.scene[risk].str[7:15]))))
    CF = pd.DataFrame(rows)
    fixX = {4.0: ["after standstill + creep (lead not moved)"], 5.0: ["after standstill + creep (lead not moved)", "slow approach, close-range over-read"],
            6.0: ["after standstill + creep (lead not moved)", "slow approach, close-range over-read"]}
    L += ["## 4. Counterfactual (offline)", "", f"Population: the last standstill behind a lead before any event in scenes where the human ends still or "
          f"... restricted to stops from motion (ego > 1 m/s 2 s before) where the human ends still and the rule does not already predict a rear-hit "
          f"({len(C)} scene-reads). Rule check on the current reads (all last standstills, human still or lead at rest): predicted & rear-hit "
          f"{J['4/rule_check']['tp']}, predicted only {J['4/rule_check']['fp']}, rear-hit only {J['4/rule_check']['fn']}, neither {J['4/rule_check']['tn']}. Under stop distance X the ego rests where the standstill reading first drops below X: "
          f"true gap GX = {GX} (+- 0.7 m as low / high); shift = max(0, GX - ego min gap); a new rear-hit when a logged follower exists and the new "
          f"station is more than (follower gap - {MARGIN} m) behind the human's rest point, scene not already rear-hit; cost {lrh:.3f} per new rear-hit.", "",
          "| X | case | rests moved back | median shift (m) | new rear-hit risks | scenes |", "|---|---|--:|--:|--:|---|"]
    L += [f"| {r.X:g} | {r.case} | {r.shifted} | {r.shift_med:.2f} | {r.new_rear} | {r.scenes} |" for _, r in CF.iterrows()]
    L += [""]
    est = []
    for X in GX:
        fixed = int(Af.kind.isin(fixX[X]).sum())
        for lab in ("low", "p50", "high"):
            nr = int(CF[(CF.X == X) & (CF.case == lab)].new_rear.iloc[0])
            est.append(dict(arm=f"stop {X:g} m", case=lab, zeros_fixed=fixed, new_rear=nr, pts_per_seed=(fixed - nr * lrh) / nR))
    nc = int(Af.kind.isin(fixX[4.0]).sum())
    est.append(dict(arm="no creep until the lead moves", case="p50", zeros_fixed=nc, new_rear=0, pts_per_seed=nc / nR))
    ES = pd.DataFrame(est)
    L += ["Expected PAI-281 points per seed (zeros fixed - new rear-hits x cost, over the reads; the moving-following cost of a larger X is not in it):", "",
          "| arm | case | zeros fixed (all reads) | new rear-hits (all reads) | points / seed |", "|---|---|--:|--:|--:|"]
    L += [f"| {r.arm} | {r.case} | {r.zeros_fixed} | {r.new_rear} | {r.pts_per_seed:+.2f} |" for _, r in ES.iterrows()]
    J["4"] = ES.to_dict("records")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    CF.to_csv(out / "counterfactual.csv", index=False, float_format="%.3f")
    json.dump(J, open(out / "summary.json", "w"), indent=1, default=float)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--reads", nargs="+", default=list(READS)), s.add_argument("--limit", type=int, default=0)
    r = sp.add_parser("report")
    r.add_argument("--src", default=OUT), r.add_argument("--out", default=str(REPO / "experiments/round9/results/stop_gap")), r.add_argument("--facts", default="")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
