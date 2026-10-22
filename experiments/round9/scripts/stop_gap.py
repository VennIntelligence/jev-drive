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


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--reads", nargs="+", default=list(READS)), s.add_argument("--limit", type=int, default=0)
    r = sp.add_parser("report")
    r.add_argument("--src", default=OUT), r.add_argument("--out", default=str(REPO / "experiments/round9/results/stop_gap"))
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
