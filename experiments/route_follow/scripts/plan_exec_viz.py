#!/usr/bin/env python3
"""PLAN vs EXECUTION viewer: where does a turn fail between the served plan and the car? Every turn of RSG-L18-s0: the 36 PAI 281 strict
turns (+ 4 non-strict PAI scenes of the first 15-row page) and the 24 held-460 nuPlan turns, one row each: BEV of every served plan around
the junction, the front-camera clip (PAI only), traces vs time. Grouped by outcome, completed first, then by class. Static page, no server.
CPU only; stored reads are read, never written.

  extract  (Tokyo, host python3 + numpy)  plan_exec_viz.py extract --pkl <r281.pkl> --root <r281/rsg run root> --out <pkl>
           per scene: route, controller trace, human log, first zero event, mp4 path, every `infer` decision (t0, k, anchor, served poses).
  clips    (Tokyo, ffmpeg)  plan_exec_viz.py clips --pkl <pkl> --meta <meta.json> --out DIR
           re-encode of the stored front-camera eval video (turn_gallery.py's rendering, originals untouched) with flags HANDOFF / ENTRY /
           EVENT and the clip time; meta.json comes from `render`.
  held     (box, repo .venv)  plan_exec_viz.py held --held <rf_receding held.pkl> --lanes <rf_corridor lanes_held.pkl> --out <pkl>
           the 24 held-460 turns of RSG-L18-s0 (src held) in this script's record layout + their lanes: {recs: [...], lanes: {scene: ...}}.
  render   (Mac, repo .venv)  plan_exec_viz.py render --pkl <pkl> --lanes <rf_corridor lanes pkl> [--held <pkl>] --out DIR
           -> DIR/<s8>_bev.png, <s8>_tr.png, meta.json, check.json, index.html (Chinese text, synchronized cursor in plain JS).

Frames and times (verified in `render`, numbers in the page footer): a decision's served plan `poses` (8 x (x, y, yaw), 0.5 .. 4 s) is in the
rear-axle frame of its `anchor` (x, y, yaw in the controller's local frame; the anchor is the controller pose at t0); world = anchor composed
with the pose. Clip time of decision k = (k + 1) x 0.1 s (turn_gallery.py); force_gt 1.7 s, so the policy drives from decision 17 (HANDOFF).
Traces: car speed (controller) vs the human's speed when the log first reached the car's route station; the plan's heading change at 2 s and
4 s (sign: + = into the turn) vs the human's own 2 s / 4 s heading change from the same station; the car's lateral offset from the plan served
1 s earlier (+ = toward the turn's inside); the lateral offset of plan k's 2 s point from plan k-1's path, both composed with their own anchors
(the measured ego motion between them), + = toward the inside.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + ([str(HERE.parents[1] / "junction_eyes" / "lib")] if len(HERE.parents) > 2 else [])
import rf_pai_follow as F  # noqa: E402

RES = HERE.parent / "results"
# PAI rows: the 36 strict turns (results/pai281/strict36.txt, classes from results/receding/turns.csv) + 4 non-strict scenes of the first
# 15-row page (class as given); held rows: the 24 held-460 turns of RSG-L18-s0 (receding/turns.csv, src held). Classes follow rf_receding.
EXTRA = {"1a69f757": "late_wide", "2466266e": "over_turn", "300465f7": "over_turn", "de9db5dc": "other"}
ORDER = ("completed", "late_wide", "straight", "over_turn", "wrong_exit", "other", "no_zero")
CN = {"late_wide": "晚而宽", "straight": "直走", "over_turn": "过转", "wrong_exit": "错出口", "other": "其他", "completed": "完成",
      "no_zero": "未完成、未零分"}
HANDOFF_K, DT, DT_HELD, PRE_M, BOX, MIN_WIN = 17, 0.1, 0.5, 20.0, 60.0, 3.0


def strict36():
    return [x.split()[0][7:15] for x in open(RES / "pai281/strict36.txt") if x.strip()]
FAIL = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor"}


# ---------------------------------------------------------------- extract (Tokyo)
def extract(a):
    want = set(strict36()) | set(EXTRA)
    D = {sc["scene"]: sc for sc in pickle.load(open(a.pkl, "rb"))["scenes"] if sc["scene"][7:15] in want}
    rids = {sc["rolls"]["rsg"]["rid"]: s for s, sc in D.items() if "rsg" in sc["rolls"]}
    dec = {}
    for f in glob.glob(os.path.join(a.root, "runs", "*", "driver", "drive.jsonl")):
        for line in open(f):
            if '"kind": "drive"' not in line or '"poses"' not in line:
                continue
            r = json.loads(line)
            if r.get("session") in rids and r.get("infer") is not False and "anchor" in r:
                dec.setdefault(r["session"], []).append((int(r["t0"]), int(r["k"]), r["anchor"], r["poses"]))
    mp4 = {}
    for summ in glob.glob(os.path.join(a.root, "runs", "*", "sim", "aggregate", "results-summary.json")):
        run = Path(summ).parents[1]
        for r in json.loads(Path(summ).read_text())["rollouts"]:
            if r["rollout_id"] in rids:
                m = sorted((run / "rollouts" / r["clipgt_id"] / r["rollout_id"]).glob("*camera_front_wide*.mp4"))
                mp4[r["rollout_id"]] = dict(mp4=str(m[0]) if m else None, metrics=r.get("score_metrics") or {})
    out = []
    for s, sc in sorted(D.items()):
        o = sc["rolls"]["rsg"]
        v = sorted(dec.get(o["rid"], []), key=lambda x: (x[0], x[1]))
        mm = mp4.get(o["rid"], {})
        out.append(dict(scene=s, rid=o["rid"], score=float(o["score"] or 0), tz=o["tz"], zflag=o["zflag"], route=np.asarray(o["route"], float),
                        ctrl={k: np.asarray(x, float) for k, x in o["ctrl"].items()}, log={k: np.asarray(x, float) for k, x in sc["log"].items()},
                        mp4=mm.get("mp4"), kind=next((k2 for k, k2 in FAIL.items() if (mm.get("metrics") or {}).get(k)), "ok"),
                        t0=np.array([x[0] for x in v], np.int64), k=np.array([x[1] for x in v]), anchor=np.array([x[2] for x in v], float),
                        poses=np.array([x[3] for x in v], float)))
        print(s[7:15], len(v), "decisions, mp4", bool(mm.get("mp4")), flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "scenes ->", a.out)


def held_extract(a):
    H = [r for r in pickle.load(open(a.held, "rb")) if r["set"] == "RSG-L18-s0" and r["src"] == "held"]
    L = pickle.load(open(a.lanes, "rb"))["scenes"]
    recs = [dict(scene=r["scene"], rid=r["rid"], score=r["score"], tz=r["tz"] if np.isfinite(r["tz"]) else None, zflag="", route=r["R"], ctrl=r["ctrl"],
                 log=r["log"], mp4=None, kind="zero" if r["score"] == 0 else "ok", t0=r["plans"]["t0"], k=r["plans"]["k"], anchor=r["plans"]["anchor"],
                 poses=r["plans"]["poses"], row=r["row"], run=r["run"]) for r in H]
    pickle.dump(dict(recs=recs, lanes={r["scene"]: L[r["scene"]] for r in H if r["scene"] in L}), open(a.out, "wb"), protocol=4)
    print(len(recs), "held turns ->", a.out)


# ---------------------------------------------------------------- clips (Tokyo)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def clips(a):
    from concurrent.futures import ThreadPoolExecutor
    meta = json.loads(Path(a.meta).read_text())
    recs = {r["scene"][7:15]: r for r in pickle.load(open(a.pkl, "rb"))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def esc(s):
        return s.replace(":", r"\:").replace("'", "")

    def tx(txt, x, y, size, col, en=""):
        return (f"drawtext=fontfile={FONT}:text='{txt}':x={x}:y={y}:fontsize={size}:fontcolor={col}:box=1:boxcolor=black@0.55:boxborderw=5"
                + (f":enable='{en}'" if en else ""))

    def one(s8):
        r, m = recs[s8], meta["rows"][s8]
        if not r["mp4"]:
            return s8, "no stored clip"
        vf = [tx("RSG " + s8, 8, 8, 24, "white"),
              tx(r"t %{eif\:floor(t)\:d}.%{eif\:mod(floor(t*10+0.5)\,10)\:d} s", 8, "h-th-14", 26, "white"),
              tx("HANDOFF", "(w-text_w)/2", 8, 30, "cyan", f"between(t,{m['t_handoff']},{m['t_handoff'] + 1.0})")]
        if m.get("t_entry") is not None:
            vf.append(tx("ENTRY", "(w-text_w)/2", 44, 30, "yellow", f"between(t,{m['t_entry']},{m['t_entry'] + 1.5})"))
        if m.get("t_event") is not None:
            vf += [f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=8:enable='gte(t,{m['t_event']})'",
                   tx(esc(m["event"]).upper(), "(w-text_w)/2", 80, 30, "red", f"gte(t,{m['t_event']})")]
        dst = out / f"{s8}.mp4"
        tmp = str(dst) + ".part.mp4"
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", r["mp4"], "-vf", ",".join(vf), "-an", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "24", "-pix_fmt", "yuv420p", "-threads", "4", "-movflags", "+faststart", tmp], capture_output=True, text=True)
        if p.returncode:
            return s8, "ffmpeg failed: " + p.stderr[-300:]
        Path(tmp).rename(dst)
        n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames,r_frame_rate",
                            "-of", "csv=p=0", str(dst)], capture_output=True, text=True).stdout.strip()
        return s8, f"ok ({n}; decisions {len(r['t0'])})"

    with ThreadPoolExecutor(6) as ex:
        res = dict(ex.map(one, [k for k, v in meta["rows"].items() if v.get("set", "pai") == "pai"]))
    (out / "clips.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)


# ---------------------------------------------------------------- render (Mac)
def to_world(A, X):
    c, s = np.cos(A[2]), np.sin(A[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.asarray(A[:2], float) + np.c_[c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]]


def plan_world(A, P):
    """(9, 2) world path of a served plan incl. its origin, (9,) times."""
    return to_world(A, np.r_[[[0.0, 0.0]], P[:, :2]]), np.r_[0.0, np.arange(1, 9) * 0.5]


def lat_of(Q, p):
    """Signed lateral (left +) of point p from polyline Q; nan when p projects beyond Q's ends (clamped)."""
    S = F.stations(Q)
    if S[-1] < 0.5:
        return np.nan
    s, la = F.project(Q, S, np.asarray(p, float))
    return la if 1e-3 < s < S[-1] - 1e-3 else np.nan


def turn_defs():
    """{scene8: dict(entry, exit, side, turn_deg, primary, src)}: rf_receding turns.csv (RSG-L18-s0, strict 36 + d299 over-turns), else the d299
    zeros.csv near turn (bends)."""
    out = {}
    for r in csv.DictReader(open(RES / "progress/zeros.csv")):
        if r["set"] == "pai" and r["turn_entry"]:
            d = float(r["turn_deg"])
            out[r["scene"][7:15]] = dict(entry=float(r["turn_entry"]), exit=float(r["turn_exit"]), turn_deg=d, side="L" if d > 0 else "R",
                                         src=f"d299 zeros.csv ({r['near_turn']})", primary=r["primary"], t_f_s=float(r["t_f_s"]))
    for r in csv.DictReader(open(RES / "receding/turns.csv")):
        if r["set"] == "RSG-L18-s0" and r["src"] in ("pai", "pai_x"):
            s8 = r["scene"][7:15]
            out[s8] = dict(out.get(s8, {}), entry=float(r["entry"]), exit=float(r["exit"]), turn_deg=float(r["turn_deg"]), side=r["side"],
                           src="receding/turns.csv (" + ("strict 36" if r["src"] == "pai" else "d299 over-turn") + ")", outcome=r["outcome"],
                           primary=r["primary"] or out.get(s8, {}).get("primary", ""), cls=r["cls"])
        elif r["set"] == "RSG-L18-s0" and r["src"] == "held":
            out[r["scene"]] = dict(entry=float(r["entry"]), exit=float(r["exit"]), turn_deg=float(r["turn_deg"]), side=r["side"], cls=r["cls"],
                                   src="receding/turns.csv (held)", outcome=r["outcome"], primary=r["primary"])
    tax = {r["scene"][7:15]: r for r in csv.DictReader(open(RES / "taxonomy/pai281_rsg.csv"))}
    for s8, d in out.items():
        if s8 in tax:
            d.update(primary=tax[s8]["primary"], t_ev_s=tax[s8]["t_ev_s"])
    return out


def analyse(r, td, dt=DT, off=DT, handoff=True):
    """One scene -> dict of arrays and scalars used by the panels and the check. dt = decision period, off = time of decision 0 on the
    display axis (PAI: the clip time 0.1 s; held: 0)."""
    R = F.dedup(r["route"])
    S = F.stations(R)
    sh, hh = F.heading(R, S)
    C = r["ctrl"]
    t, P = C["t"], np.c_[C["x"], C["y"]]
    st, _ = F.track(R, S, P)
    sg = 1.0 if td["side"] == "L" else -1.0
    ent, ext = td["entry"], td["exit"]
    LP, LT = r["log"]["P"], r["log"]["T"]
    lt, _ = F.track(R, S, LP)
    vlog = np.r_[0.0, np.hypot(*np.diff(LP, axis=0).T) / np.maximum(np.diff(LT) / 1e6, 1e-3)]
    vlog[0] = vlog[1]
    vlog = np.convolve(np.r_[vlog[:2], vlog, vlog[-2:]], np.ones(5) / 5, "valid")
    lm = np.maximum.accumulate(lt)
    su, iu = np.unique(lm, return_index=True)
    larc = F.stations(LP)
    t0 = r["t0"].astype(float)
    T0 = t0[0]
    tau = (t0 - T0) / 1e6 + off                              # clip time of each decision (PAI)
    A, PO = r["anchor"], r["poses"]
    n = len(t0)
    tz = r["tz"] if r["tz"] is not None and not r["score"] else np.inf
    xy = lambda q: np.c_[np.interp(q, larc, LP[:, 0]), np.interp(q, larc, LP[:, 1])]  # noqa: E731

    def log_plan(s):
        """Human's own 4 s path from where its route station first reached s, as heading changes at 2 s / 4 s (deg, signed into the turn)."""
        if s < su[0] or s > su[-1]:
            return np.nan, np.nan
        tl = float(np.interp(s, su, LT[iu]))
        q = np.interp(tl + np.array([0.0, 2e6, 4e6]), LT, larc)
        d = xy(q + 1.0) - xy(q - 1.0)
        y = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        return sg * np.degrees(y[1] - y[0]), sg * np.degrees(y[2] - y[0])

    W = np.stack([plan_world(A[i], PO[i])[0] for i in range(n)])          # (n, 9, 2)
    s_e = np.array([F.project(R, S, A[i, :2], st[min(np.searchsorted(t, t0[i]), len(t) - 1)] - 30, st[min(np.searchsorted(t, t0[i]), len(t) - 1)] + 60)[0]
                    for i in range(n)])
    v_car = np.interp(t0, t, C["vx"])
    v_hum = np.array([np.interp(float(np.interp(s, su, LT[iu])), LT, vlog) if su[0] <= s <= su[-1] else np.nan for s in s_e])
    v_plan = np.hypot(*PO[:, 1, :2].T) / 1.0                 # served plan's mean speed over its first second
    yaw = np.unwrap(np.c_[np.zeros(n), PO[:, :, 2]], axis=1)
    d2, d4 = sg * np.degrees(yaw[:, 4]), sg * np.degrees(yaw[:, 8])
    hum = np.array([log_plan(s) for s in s_e])
    by_t = {int(x): i for i, x in enumerate(r["t0"])}
    e1 = np.full(n, np.nan)
    jump = np.full(n, np.nan)
    for i in range(n):
        j = by_t.get(int(r["t0"][i]) - 1_000_000)
        if j is not None:
            e1[i] = sg * lat_of(W[j], A[i, :2])
        if i and abs((t0[i] - t0[i - 1]) / 1e6 - dt) < 0.3 * dt:
            jump[i] = sg * lat_of(W[i - 1], W[i, 4])
    # window: entry - 20 m .. event, or exit + 10 m when completed
    reach = np.flatnonzero(s_e >= ent - PRE_M)
    i0 = int(reach[0]) if len(reach) else 0
    x10 = np.flatnonzero(s_e >= ext + 10.0)
    i_ent = np.flatnonzero(s_e >= ent)
    i1 = int(np.flatnonzero(t0 <= tz)[-1]) if np.isfinite(tz) and (t0 <= tz).any() else n - 1
    if len(x10) and x10[0] < i1:
        i1 = int(x10[0])
    i0 = int(min(i0, np.searchsorted(tau, tau[i1] - MIN_WIN)))          # at least MIN_WIN s of plans (fast approaches)
    i1 = max(i1, i0)
    t_ent = float(tau[i_ent[0]]) if len(i_ent) and t0[i_ent[0]] <= min(tz, t0[-1]) else None
    # check: anchor vs controller at t0; plan 0.5 / 1.0 s vs the executed track (displacement, cross-track)
    an = np.where((t0 >= t[0]) & (t0 <= t[-1]), np.hypot(np.interp(t0, t, P[:, 0]) - A[:, 0], np.interp(t0, t, P[:, 1]) - A[:, 1]), np.nan)
    chk = {}
    for h, col in ((0.5, 1), (1.0, 2)):
        tt = t0 + h * 1e6
        ok = tt <= t[-1]
        cp = np.c_[np.interp(tt, t, P[:, 0]), np.interp(tt, t, P[:, 1])]
        disp = np.hypot(*(W[:, col] - cp).T)
        Sx = F.stations(P)
        xt = np.array([abs(F.project(P, Sx, W[i, col], *(lambda q: (q - 15, q + 15))(F.project(P, Sx, cp[i])[0]))[1]) for i in range(n)])
        chk[h] = dict(disp=disp, xt=xt, ok=ok)
    return dict(R=R, S=S, sh=sh, hh=hh, P=P, t=t, st=st, sg=sg, LP=LP, W=W, A=A, s_e=s_e, tau=tau, v_car=v_car, v_hum=v_hum, v_plan=v_plan, d2=d2, d4=d4,
                h2=hum[:, 0], h4=hum[:, 1], e1=e1, jump=jump, i0=i0, i1=i1, t_ent=t_ent, tz=tz, T0=T0, an=an, chk=chk, n=n,
                t_handoff=float(tau[np.flatnonzero(r["k"] >= HANDOFF_K)[0]]) if handoff and (r["k"] >= HANDOFF_K).any() else None,
                t_event=(tz - T0) / 1e6 + off if np.isfinite(tz) else None, v_ent=F.speed_at(t, st, C["vx"], ent, tz),
                v_ent_h=F.speed_at(LT, lt, vlog, ent), t_end=float(tau[-1]))


def lanes_view(lsc, R, S, LP):
    """Route lane chain polygons (rf_corridor SceneMap.match) and every lane polygon of the scene (shapely)."""
    from rf_corridor import SceneMap
    M = SceneMap(lsc)
    m = M.match(R, S, LP)
    return [p["poly"] for p in m["pieces"] if p["poly"] is not None], M.poly, m


def draw_bev(r, td, g, lsc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    R, S = g["R"], g["S"]
    ent, ext = td["entry"], td["exit"]
    at = lambda q: np.array([np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])])  # noqa: E731
    h_app = float(np.interp(ent - 10.0, g["sh"], g["hh"]))
    qs = np.linspace(max(ent - PRE_M, 0), min(ext + 10.0, S[-1]), 40)
    pts = np.array([at(q) for q in qs])
    ctr = (pts.min(0) + pts.max(0)) / 2
    half = max(BOX / 2, float(np.ptp(pts, 0).max()) / 2 + 6.0)
    th = np.pi / 2 - h_app                                     # approach heading -> up
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    tf = lambda X: (np.asarray(X, float).reshape(-1, 2) - ctr) @ Rm.T  # noqa: E731
    fig = plt.figure(figsize=(5.0, 5.0), dpi=110)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    chain, allp = [], []
    if lsc is not None:
        chain, allp, _ = lanes_view(lsc, R, S, g["LP"])
    import shapely
    box = shapely.box(*(ctr - 1.5 * half), *(ctr + 1.5 * half))

    def polys(geom):
        if geom is None or geom.is_empty:
            return []
        geom = geom.intersection(box)
        if geom.is_empty:
            return []
        gs = getattr(geom, "geoms", [geom])
        return [np.asarray(x.exterior.coords) for x in gs if x.geom_type == "Polygon"] + [q for x in gs if x.geom_type != "Polygon" for q in polys(x)]
    for pg in allp:
        for c in polys(pg):
            ax.fill(*tf(c).T, color="#f2f2f2", ec="#c8c8c8", lw=0.4, zorder=0)
    for pg in chain:
        for c in polys(pg):
            ax.fill(*tf(c).T, color="#cfe3f7", ec="#7fa9d6", lw=0.6, zorder=1)
    ax.plot(*tf(R).T, color="#7fa9d6", lw=0.6, ls=(0, (4, 3)), zorder=2)
    ax.plot(*tf(g["LP"]).T, color="#8c8c8c", lw=2.2, alpha=0.8, zorder=3, solid_capstyle="round")
    # plans
    cm = plt.get_cmap("viridis")
    i0, i1 = g["i0"], g["i1"]
    span = max(g["tau"][i1] - g["tau"][i0], 1e-6)
    thin, thick, ct = [], [], []
    for i in range(i0, i1 + 1):
        Q = tf(g["W"][i])
        thin.append(Q)
        thick.append(Q[2:5])                                  # 1.0 .. 2.0 s
        ct.append(cm((g["tau"][i] - g["tau"][i0]) / span))
    ax.add_collection(LineCollection(thin, colors=ct, linewidths=0.55, alpha=0.75, zorder=4))
    ax.add_collection(LineCollection(thick, colors=ct, linewidths=2.0, alpha=0.9, zorder=5))
    # executed track up to the event (light after it)
    t, P, tz = g["t"], g["P"], g["tz"]
    pre = t <= tz
    ax.plot(*tf(P[pre]).T, color="black", lw=1.3, zorder=6)
    if (~pre).any():
        ax.plot(*tf(P[~pre]).T, color="black", lw=0.8, alpha=0.3, zorder=6)
    ts = np.arange(g["T0"], t[pre][-1] if pre.any() else t[-1], 1e6)
    ax.plot(*tf(np.c_[np.interp(ts, t, P[:, 0]), np.interp(ts, t, P[:, 1])]).T, "o", ms=2.6, color="black", zorder=7)
    # entry / exit lines, event / exit marker
    for q, col, ls, lab in ((ent, "#d4a000", "-", "entry"), (ext, "#d4a000", "--", "exit")):
        p0 = at(q)
        hd = float(np.interp(q, g["sh"], g["hh"]))
        nrm = np.array([-np.sin(hd), np.cos(hd)])
        L = tf(np.r_[[p0 - 5 * nrm], [p0 + 5 * nrm]])
        ax.plot(*L.T, color=col, lw=1.8, ls=ls, zorder=8)
        ax.text(*L[1], lab, color=col, fontsize=7, zorder=9)
    if np.isfinite(tz):
        pe = tf([np.interp(tz, t, P[:, 0]), np.interp(tz, t, P[:, 1])])[0]
        ax.plot(*pe, "X", ms=11, color="#d62728", mec="white", mew=0.8, zorder=10)
        ax.text(pe[0] + 1.2, pe[1] - 2.5, f"{r['kind']} {g['t_event']:.1f}s", color="#d62728", fontsize=8, zorder=10,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    sb = np.array([[-half + 3, -half + 3], [-half + 13, -half + 3]])
    ax.plot(*sb.T, color="black", lw=2)
    ax.text(-half + 3, -half + 4.2, "10 m", fontsize=7)
    ax.text(half - 1, half - 2, f"plans {g['tau'][i0]:.1f}-{g['tau'][i1]:.1f} s", ha="right", va="top", fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(g["tau"][i0], g["tau"][i1]))
    cax = fig.add_axes([0.6, 0.075, 0.35, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6)
    cb.set_label("decision time (clip s)", fontsize=6)
    fig.savefig(path, dpi=110)
    # cursor map: car position of every decision in image fractions
    pix = ax.transData.transform(tf(g["A"][:, :2]))
    wpx, hpx = fig.get_size_inches() * fig.dpi
    plt.close(fig)
    return dict(x=np.round(pix[:, 0] / wpx, 4).tolist(), y=np.round(1 - pix[:, 1] / hpx, 4).tolist())


def draw_traces(r, td, g, path, xlabel="clip time s (decision k = (k + 1) x 0.1 s)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tau, i0, i1 = g["tau"], g["i0"], g["i1"]
    tmax = min(g["t_end"], (g["t_event"] + 1.5) if g["t_event"] else g["t_end"])
    m = tau <= tmax
    fig, ax = plt.subplots(4, 1, figsize=(6.2, 5.0), dpi=110, sharex=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.08, hspace=0.12)
    ax[0].plot(tau[m], g["v_car"][m], color="black", lw=1.2, label="car")
    ax[0].plot(tau[m], g["v_hum"][m], color="#8c8c8c", lw=1.2, label="human @ same station")
    ax[0].plot(tau[m], g["v_plan"][m], color="#1f77b4", lw=0.8, ls=":", label="plan 0-1 s")
    ax[0].set_ylabel("speed m/s", fontsize=7)
    ax[1].plot(tau[m], g["d2"][m], color="#2ca02c", lw=1.1, label="plan 2 s")
    ax[1].plot(tau[m], g["d4"][m], color="#1f77b4", lw=1.1, label="plan 4 s")
    ax[1].plot(tau[m], g["h2"][m], color="#2ca02c", lw=0.9, ls="--", alpha=0.7, label="human 2 s")
    ax[1].plot(tau[m], g["h4"][m], color="#1f77b4", lw=0.9, ls="--", alpha=0.7, label="human 4 s")
    ax[1].set_ylabel("heading chg deg\n(+ into turn)", fontsize=7)
    ax[2].plot(tau[m], g["e1"][m], color="#d62728", lw=1.1)
    ax[2].set_ylabel("car - plan(t-1s)\nlat m (+ inside)", fontsize=7)
    ax[3].plot(tau[m], g["jump"][m], color="#9467bd", lw=0.9)
    ax[3].set_ylabel("plan k vs k-1\n@2 s lat m", fontsize=7)
    ax[3].set_xlabel(xlabel, fontsize=7)
    for a, lim in zip(ax, (None, None, 2.0, 1.0)):
        a.tick_params(labelsize=6.5)
        a.grid(alpha=0.25, lw=0.4)
        a.axvspan(tau[i0], tau[i1], color="#ffe9a8", alpha=0.35, lw=0, zorder=0)
        if lim:
            y = g["e1" if lim == 2.0 else "jump"][m]
            ymax = max(lim, float(np.nanmax(np.abs(y))) * 1.1) if np.isfinite(y).any() else lim
            a.set_ylim(-ymax, ymax)
            a.axhline(0, color="#999", lw=0.5)
        if g["t_handoff"] is not None:
            a.axvline(g["t_handoff"], color="#17becf", lw=1.0, ls="--")
        if g["t_ent"] is not None:
            a.axvline(g["t_ent"], color="#d4a000", lw=1.3)
        if g["t_event"] is not None:
            a.axvline(g["t_event"], color="#d62728", lw=1.3)
    ax[0].legend(fontsize=6, loc="best", ncol=3, framealpha=0.7)
    ax[1].legend(fontsize=6, loc="best", ncol=2, framealpha=0.7)
    ax[1].axhline(0, color="#999", lw=0.5)
    ax[3].set_xlim(0, tmax)
    yt = ax[0].get_ylim()[1]
    if g["t_handoff"] is not None:
        ax[0].text(g["t_handoff"], yt, " handoff", color="#17becf", fontsize=6, va="top")
    if g["t_ent"] is not None:
        ax[0].text(g["t_ent"], yt, " entry", color="#d4a000", fontsize=6, va="top")
    fig.savefig(path, dpi=110)
    wpx, hpx = fig.get_size_inches() * fig.dpi
    x0, x1 = ax[3].transData.transform([(0, 0), (tmax, 0)])[:, 0] / wpx
    ytop = 1 - ax[0].transAxes.transform((0, 1))[1] / hpx
    ybot = 1 - ax[3].transAxes.transform((0, 0))[1] / hpx
    plt.close(fig)
    return dict(x0=round(float(x0), 4), x1=round(float(x1), 4), tmax=round(float(tmax), 3), ytop=round(float(ytop), 4), ybot=round(float(ybot), 4))


def render(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = {r["scene"][7:15]: r for r in pickle.load(open(a.pkl, "rb"))}
    lanes = pickle.load(open(a.lanes, "rb"))["scenes"] if a.lanes else {}
    lanes = {k[7:15]: v for k, v in lanes.items()}
    TD = turn_defs()
    clipst = json.loads(Path(a.clips).read_text()) if a.clips and Path(a.clips).exists() else {}
    gal = set(a.gallery.split(",")) if a.gallery else set()
    S36 = strict36()
    jobs = [("pai", s8, TD[s8]["cls"] if s8 in S36 else EXTRA[s8], recs[s8], TD[s8], lanes.get(s8), s8 in S36) for s8 in S36 + list(EXTRA)]
    if a.held:
        H = pickle.load(open(a.held, "rb"))
        jobs += [("held", "h" + r["scene"][-8:], TD[r["scene"]]["cls"], r, TD[r["scene"]], H["lanes"].get(r["scene"]), True) for r in H["recs"]]
    rk = lambda j: (j[0] != "pai", not j[6], ORDER.index(j[2]), j[4].get("outcome", ""), j[1])  # noqa: E731
    rows, check = {}, {}
    for st_, key, cls, r, td, lsc, strict in sorted(jobs, key=rk):
        held = st_ == "held"
        g = analyse(r, td, *((DT_HELD, 0.0, False) if held else (DT, DT, True)))
        g["t0"] = r["t0"]
        bev = draw_bev(r, td, g, lsc, out / f"{key}_bev.png")
        tr = draw_traces(r, td, g, out / f"{key}_tr.png", *(["time s since decision 0 (2 Hz decisions)"] if held else []))
        w = slice(g["i0"], g["i1"] + 1)
        c = {}
        for h in (0.5, 1.0):
            x = g["chk"][h]
            ok = x["ok"][w]
            c[f"disp{h}"] = x["disp"][w][ok]
            c[f"xt{h}"] = x["xt"][w][ok]
        an = g["an"][r["k"] >= 2] if held else g["an"]          # held: decision 1 (0.5 s) still inside the sim's start-up; see the footer
        check[key] = dict(set=st_, cls=cls, n=int(g["i1"] - g["i0"] + 1), anchor_max=float(np.nanmax(an)), anchor_k1=float(g["an"][1]),
                          **{k: dict(med=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max())) for k, v in c.items() if len(v)})
        tw = g["tau"][w]
        inwin = lambda x: x[w][np.isfinite(x[w])]  # noqa: E731
        pre_m = float(td["entry"] - g["s_e"][0])
        rows[key] = dict(set=st_, scene=r["scene"], s8=key, cls=cls, cn=CN[cls], strict=strict, side=td["side"], turn_deg=round(td["turn_deg"]),
                         primary=td.get("primary", ""), tsrc=td["src"], score=round(r["score"], 3), kind=r["kind"], entry=td["entry"], exit=td["exit"],
                         outcome=td.get("outcome", ""), pre_m=round(pre_m, 1),
                         v_ent=None if not np.isfinite(g["v_ent"]) else round(g["v_ent"], 2), v_ent_h=None if not np.isfinite(g["v_ent_h"]) else round(g["v_ent_h"], 2),
                         t_handoff=g["t_handoff"], t_entry=g["t_ent"], t_event=None if g["t_event"] is None else round(g["t_event"], 2),
                         event=td.get("primary") or r["kind"], win=[round(float(tw[0]), 2), round(float(tw[-1]), 2)], n_dec=int(g["n"]),
                         lanes=lsc is not None, bev=bev, tr=tr, tau=np.round(g["tau"], 2).tolist(), clip=clipst.get(key, "not rendered"),
                         gallery=key in gal, mp4=bool(r.get("mp4")),
                         stats=dict(jump_p90=float(np.percentile(np.abs(inwin(g["jump"])), 90)) if len(inwin(g["jump"])) else None,
                                    e1_med=float(np.median(inwin(g["e1"]))) if len(inwin(g["e1"])) else None,
                                    e1_min=float(np.min(inwin(g["e1"]))) if len(inwin(g["e1"])) else None,
                                    d4_med=float(np.median(inwin(g["d4"]))) if len(inwin(g["d4"])) else None,
                                    h4_med=float(np.median(inwin(g["h4"]))) if len(inwin(g["h4"])) else None,
                                    d4_max=float(np.max(inwin(g["d4"]))) if len(inwin(g["d4"])) else None,
                                    v_med=float(np.median(g["v_car"][w])), vh_med=float(np.nanmedian(g["v_hum"][w])) if np.isfinite(g["v_hum"][w]).any() else None))
        print(key, st_, cls, "decisions", rows[key]["win"], "anchor max", f"{check[key]['anchor_max']:.3f}",
              " ".join(f"{k} {v['med']:.2f}/{v['p90']:.2f}" for k, v in check[key].items() if isinstance(v, dict)), flush=True)
    (out / "meta.json").write_text(json.dumps(dict(rows=rows), indent=1))
    (out / "check.json").write_text(json.dumps(check, indent=1))
    html(out, rows, check)


def html(out, rows, check):
    def pool(st_, k):
        v = [check[s][k]["med"] for s in check if check[s]["set"] == st_ and check[s]["cls"] == "completed" and k in check[s]]
        return f"{np.median(v):.2f}" if v else "-"
    cl = "".join(f"<tr><td>{s}</td><td>{check[s]['set']}</td><td>{CN[check[s]['cls']]}</td><td>{check[s]['n']}</td><td>{check[s]['anchor_max']:.3f}</td>"
                 + "".join(f"<td>{check[s][k]['med']:.2f} / {check[s][k]['p90']:.2f}</td>" if k in check[s] else "<td>-</td>"
                           for k in ("xt0.5", "xt1.0", "disp0.5", "disp1.0")) + "</tr>" for s in rows)
    page = PAGE.replace("__ROWS__", json.dumps(list(rows.values()))).replace("__CHECK__", cl).replace("__CN__", json.dumps(CN))
    k1 = [check[s]["anchor_k1"] for s in check if check[s]["set"] == "held" and np.isfinite(check[s]["anchor_k1"])]
    page = page.replace("__K1__", f"{max(k1):.2f}" if k1 else "-")
    for st_ in ("pai", "held"):
        page = page.replace(f"__XT_{st_}__", pool(st_, "xt1.0")).replace(f"__DISP_{st_}__", pool(st_, "disp1.0"))
    (out / "index.html").write_text(page)
    print("->", out / "index.html")


PAGE = r"""<!doctype html><html lang=zh><meta charset=utf-8><title>plan 与车：全部转弯（RSG-L18-s0）</title>
<style>
body{font:14px/1.55 -apple-system,"PingFang SC","Noto Sans CJK SC",system-ui,sans-serif;margin:0;background:#fafafa;color:#222}
header,#foot,.sec{max-width:1500px;margin:0 auto;padding:12px 18px}
h1{font-size:19px;margin:6px 0 8px}h2{font-size:17px;margin:18px 0 4px}h3{font-size:15px;margin:14px 0 2px;color:#444}
.count{background:#fff7e0;border:1px solid #e8d9a8;border-radius:6px;padding:8px 10px;font-size:13px}
.legend{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:8px 0}
.legend div{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 10px;font-size:13px}
.legend b{font-size:14px}
.key,.note{font-size:12.5px;color:#444}
#toc{font-size:13px}#toc a{margin-right:10px}
.row{max-width:1500px;margin:10px auto;background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 10px}
.row.done{border-left:5px solid #2ca02c}.row.fail{border-left:5px solid #d62728}.row.nz{border-left:5px solid #ff9f1c}
.hd{font-size:13px;margin-bottom:6px}.hd b{font-family:ui-monospace,monospace;font-size:14px}
.hd .c{display:inline-block;padding:0 6px;border-radius:3px;background:#eee;margin:0 4px}
.panels{display:flex;gap:10px;align-items:flex-start}
.pan{position:relative;flex:none}
.pan img{display:block}
.bev img{width:470px}.tr img{width:590px}
.vid{width:400px}.vid video{width:400px;display:block;background:#000}
.none{width:400px;height:200px;display:flex;align-items:center;justify-content:center;background:#eee;color:#666;text-align:center;padding:10px;box-sizing:border-box}
.held .vid,.held .none{width:180px}
.dot{position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:#ff2d55;border:2px solid #fff;box-shadow:0 0 3px #000;display:none;pointer-events:none}
.cur{position:absolute;width:2px;background:#ff2d55;display:none;pointer-events:none}
.tr img{cursor:crosshair}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:2px 6px;text-align:right}
#foot{font-size:12.5px;color:#444}
</style>
<header>
<h1>plan 与车：全部转弯（RSG-L18-s0；PAI 281 严格转弯 36 个 + 第一版的 4 个非严格场景；held 460 转弯 24 个）</h1>
<div class=count><b>转弯在各集合里占多少</b>（场景数，主会话汇总的口径）：PAI 281 = 严格转弯 36（12.8 %）+ 其余（弯道 / 直路 / 车道保持）；考试集 72 = 转弯场景 26（36 %），其中严格 15（21 %）；
held 460 = ≥ 60° 的转弯 24（5.2 %）；PAI train160 = 有转弯的场景 17（10.6 %）；nuPlan on-policy 训练池 = 转弯场景 300 / 1 020（29 %）。
严格转弯在 PAI 的两个集合里约 11–13 %，held 460 更少，训练池里多一些。本页把 PAI 和 held 上 RSG-L18-s0 的转弯全部列出。第一版 15 行的页面在 <a href="index_15.html">index_15.html</a>。</div>
<div class=legend>
<div><b>(a) plan 本身对不对</b><br>看鸟瞰：细线是每个决策的 plan（4 s，颜色 = 决策时间，紫早黄晚），粗段是它的 1–2 s（MPC 实际在跟的那一段）。plan 有没有拐进蓝色的路线车道？和灰色人类路径比，起弯早晚、弯的大小、在哪条车道。读数第二格：plan 2 s / 4 s 的航向变化（实线）对人在同一路线位置的航向变化（虚线）。</div>
<div><b>(b) 车有没有跟自己的 plan</b><br>看鸟瞰：黑色实际轨迹（每 1 s 一个点）是否落在粗段上。读数第三格：车此刻相对 1 s 前那个 plan 的横向偏差（+ = 偏向弯内，− = 偏向弯外）。接近 0 = 车在跟 plan。第一格的速度：车（黑）、人在同一路线位置（灰）、plan 自己前 1 s 的速度（蓝点线）。</div>
<div><b>(c) 连续 plan 之间有没有跳</b><br>看鸟瞰：相邻颜色的细线是重合还是交替地一条弯、一条直。读数第四格：第 k 个 plan 的 2 s 点离第 k−1 个 plan 路径的横向距离（两个 plan 都按各自决策时的实测位姿放到世界系，即已扣除车的实际运动；+ = 偏向弯内）。平稳 ≈ 0；锯齿 = plan 在两种形状之间切换。PAI 相邻决策隔 0.1 s，held 隔 0.5 s，两节的幅度不能直接比。</div>
</div>
<p class=key>标记：黄色实线 = 入口（路线转弯的 10 % 航向点），黄色虚线 = 出口；红 X = 第一个零分事件（类型 + 时间）；青色虚线 = 交接（PAI：force_gt 1.7 s 后，第 17 个决策起策略自己开）；浅黄底 = 鸟瞰图画了 plan 的时间段（入口前 20 m 起、且至少 3 s，到零分事件或出口后 10 m 止）。鸟瞰按路线对齐：入口前 10 m 处的路线方向朝上，60 × 60 m（转弯更大时放大）。浅灰 = 地图上所有车道，浅蓝 = 路线的车道链（rf_corridor 的 lane-chain），蓝虚线 = 路线中心线。
行首颜色：绿 = 完成，红 = 零分失败，橙 = 未完成但没有零分（开出路线 2.5 m 以外 / 到 rollout 结束还没过出口）。PAI 行播放视频时红点 / 红线跟着走；点读数图可跳到那一刻（held 行没有视频，点读数图移动红点）。</p>
<div id=toc></div>
</header>
<div id=list></div>
<div id=foot>
<h3>坐标核对（脚本 plan_exec_viz.py render 计算）</h3>
<p>plan 的 8 个点（0.5–4 s）在决策时刻后轴坐标系里（原点 = anchor，driver 日志记的 anchor 即控制器在 t0 的位姿），世界坐标 = anchor ∘ pose。
核对：anchor 与控制器在 t0 的位置之差（最大值列；只算控制器轨迹覆盖的决策；held 从第 2 个决策起算：第 1 个决策（0.5 s，rollout 起步阶段）在 24 个里差 __K1__ m 以内，之后都是 0.000 m）；plan 的 0.5 s / 1.0 s 点与车在 t0 + 0.5 / 1.0 s 实际位置的偏差，分「到实际轨迹的横向距离」和「同一时刻的位移」两种（中位 / p90，m，只算鸟瞰图画的窗口）。
<b>完成的弯：plan 1.0 s 点到实际轨迹的横向距离，各弯中位数的中位 PAI __XT_pai__ m / held __XT_held__ m；同时刻位移 PAI __DISP_pai__ m / held __DISP_held__ m</b>（位移含纵向）。门槛 ~0.3 m 是对横向距离说的。</p>
<table><tr><th>行</th><th>集</th><th>类</th><th>窗口决策数</th><th>anchor 偏差 max m</th><th>横向 0.5 s</th><th>横向 1.0 s</th><th>位移 0.5 s</th><th>位移 1.0 s</th></tr>__CHECK__</table>
<p>数据：PAI = Tokyo <code>/data/runs/route_night/r281/rsg</code>（RSG-L18-s0 在 PAI 281 上 gains 20 的存档 rollout：driver/drive.jsonl 的 infer 决策 = anchor + serve 的 poses；控制器轨迹、路线、人类日志、零分时刻来自 <code>pai_follow/r281.pkl</code>）；
held = box <code>runs/route_mem/read/RSG-L18-s0/c*/run</code>（rf_receding extract-held 的 <code>receding/held.pkl</code>：路线、控制器、日志、driver-logs 的决策；零分时刻 = 第一个零分的 2 Hz 决策）。
车道多边形 = rf_corridor lanes（<code>lanes_pai.pkl</code> / <code>lanes_held.pkl</code>），路线车道链 = rf_corridor SceneMap.match。
入口 / 出口与类别：<code>results/receding/turns.csv</code>（PAI 严格 36、d299 过转、held 24）；1a69f757（弯道）与 de9db5dc 来自 <code>results/progress/zeros.csv</code>；零分类型来自 taxonomy。视频时间：决策 k 在 (k + 1) × 0.1 s（turn_gallery.py 核过）；视频是存档前视视频的重编码（加了标签），没有重新渲染。</p>
</div>
<script>
const R=__ROWS__, CN=__CN__;
const f1=v=>v==null?"-":(+v).toFixed(1);
const GROUPS=[];
for(const r of R){const sec=r.set=="pai"?(r.strict?"A":"A2"):"B"; const g=sec+"|"+r.cls; if(!GROUPS.length||GROUPS[GROUPS.length-1].g!=g) GROUPS.push({g,sec,cls:r.cls,rows:[]}); GROUPS[GROUPS.length-1].rows.push(r)}
const SEC={A:"A. PAI 281 严格转弯（36）",A2:"A′. 不在严格 36 里、第一版 15 行中的 4 个 PAI 场景",B:"B. held 460 nuPlan 转弯（24）"};
const SECNOTE={A:"",A2:"1a69f757 是 −50° 的弯道；de9db5dc 是严格转弯但不在 36 个里（d299 列表）；2466266e、300465f7 是 d299 的过转零分。",
 B:"nuPlan held 460，2 Hz 决策、每个 rollout 18 个决策（9 s）；接近段只有入口前 3–19 m（rollout 起点就离路口很近，见每行的「起点距入口」），所以「入口前 20 m」常常从第 0 个决策开始。没有存档的摄像头视频，只有鸟瞰和读数；没有交接标记。"};
const outTxt=r=>r.cls=="completed"?`完成${r.t_event!=null?`（之后 ${f1(r.t_event)} s 零分：${r.event}）`:""}`:r.t_event!=null?`失败：${r.event||r.kind} @ ${f1(r.t_event)} s`:`未完成（${r.outcome}），没有零分`;
let html="", last="", toc=[];
for(const G of GROUPS){
  if(G.sec!=last){html+=`<div class=sec><h2 id=s${G.sec}>${SEC[G.sec]}</h2><p class=note>${SECNOTE[G.sec]}</p></div>`; last=G.sec; toc.push(`<b>${SEC[G.sec].split("（")[0]}</b>`)}
  const id=G.g.replace("|","_"); html+=`<div class=sec><h3 id=${id}>${CN[G.cls]}（${G.rows.length}）</h3></div>`; toc.push(`<a href="#${id}">${CN[G.cls]} ${G.rows.length}</a>`);
  for(const r of G.rows){const i=R.indexOf(r), st=r.stats, held=r.set=="held";
    const gal=held?"":r.gallery?` · <a href="../turn_gallery/index.html">turn_gallery</a>（搜 ${r.s8}，Tokyo 桌面）`:" · 画廊里没有这一行";
    const vid=held?`<div class=none>nuPlan held：没有摄像头视频</div>`:r.mp4&&r.clip.startsWith("ok")?`<video src="clips/${r.s8}.mp4" controls muted playsinline preload=none data-i=${i}></video>`:`<div class=none>没有存档的前视视频（${r.clip}），没有重新渲染</div>`;
    const cls=r.cls=="completed"?"done":r.t_event!=null?"fail":"nz";
    html+=`<div class="row ${cls}${held?" held":""}" id=r${r.s8} data-i=${i}><div class=hd><b>${r.s8}</b>${held?` <span style="color:#777">${r.scene}</span>`:""} <span class=c>${r.cn}</span> ${r.side=="L"?"左转":"右转"} ${Math.abs(r.turn_deg)}° ·
    入口速度 车 ${f1(r.v_ent)} / 人 ${f1(r.v_ent_h)} m/s · ${outTxt(r)} · 分数 ${r.score.toFixed(2)}${gal}
    <span style="color:#777">· 起点距入口 ${f1(r.pre_m)} m · 入口 / 出口 ${f1(r.entry)} / ${f1(r.exit)} m（${r.tsrc}）· plan 窗口 ${r.win[0]}–${r.win[1]} s · 窗口内 plan 4 s 航向中位 ${f1(st.d4_med)}° 对人 ${f1(st.h4_med)}° · 速度中位 车 ${f1(st.v_med)} / 人 ${f1(st.vh_med)} · |跳| p90 ${st.jump_p90==null?"-":st.jump_p90.toFixed(2)} m${r.lanes?"":" · 无车道数据"}</span></div>
    <div class=panels><div class="pan bev"><img loading=lazy src="${r.s8}_bev.png"><div class=dot></div></div><div class="pan vid">${vid}</div>
    <div class="pan tr"><img loading=lazy src="${r.s8}_tr.png"><div class=cur></div></div></div></div>`}}
document.getElementById("list").innerHTML=html;
document.getElementById("toc").innerHTML=toc.join(" ");
function nearest(a,x){let lo=0,hi=a.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(a[m]<x)lo=m;else hi=m}return Math.abs(a[lo]-x)<=Math.abs(a[hi]-x)?lo:hi}
function update(row,r,t){
  const bi=row.querySelector(".bev img"), dot=row.querySelector(".dot"), ti=row.querySelector(".tr img"), cur=row.querySelector(".cur");
  if(t<r.tau[0]-0.3||t>r.tau[r.tau.length-1]+0.3){dot.style.display="none"}else{const k=nearest(r.tau,t);
    const x=r.bev.x[k]*bi.clientWidth, y=r.bev.y[k]*bi.clientHeight;
    if(x>=0&&y>=0&&x<=bi.clientWidth&&y<=bi.clientHeight){dot.style.left=x+"px";dot.style.top=y+"px";dot.style.display="block"}else dot.style.display="none"}
  const T=r.tr; if(t>T.tmax){cur.style.display="none";return}
  cur.style.left=(T.x0+(T.x1-T.x0)*t/T.tmax)*ti.clientWidth+"px"; cur.style.top=T.ytop*ti.clientHeight+"px"; cur.style.height=(T.ybot-T.ytop)*ti.clientHeight+"px"; cur.style.display="block";
}
document.querySelectorAll(".row").forEach(row=>{const r=R[+row.dataset.i], v=row.querySelector("video");
  if(v){const tick=()=>{update(row,r,v.currentTime); if(!v.paused) requestAnimationFrame(tick)};
    v.addEventListener("play",tick); v.addEventListener("seeked",()=>update(row,r,v.currentTime)); v.addEventListener("timeupdate",()=>update(row,r,v.currentTime))}
  row.querySelector(".tr img").addEventListener("click",e=>{const T=r.tr, f=e.offsetX/e.target.clientWidth; const t=(f-T.x0)/(T.x1-T.x0)*T.tmax; if(t>=0&&t<=T.tmax){if(v) v.currentTime=t; update(row,r,t)}});
});
</script></html>"""


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument("--pkl", default="/data/runs/route_night/pai_follow/r281.pkl")
    p.add_argument("--root", default="/data/runs/route_night/r281/rsg")
    p.add_argument("--out", required=True)
    p = sp.add_parser("held")
    p.add_argument("--held", required=True)
    p.add_argument("--lanes", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("clips")
    p.add_argument("--pkl", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("render")
    p.add_argument("--pkl", required=True)
    p.add_argument("--lanes")
    p.add_argument("--clips", help="clips.json from `clips` (status per scene)")
    p.add_argument("--gallery", default="", help="comma list of scene8 that have a turn_gallery row")
    p.add_argument("--held", help="held-460 turns pkl: {recs: [...], lanes: {scene: lanes extract}} (box, from rf_receding held.pkl)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": extract, "held": held_extract, "clips": clips, "render": render}[a.cmd](a)


if __name__ == "__main__":
    main()
