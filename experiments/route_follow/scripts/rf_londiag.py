#!/usr/bin/env python3
"""LON-DIAG readings 1 and 2 (experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md): why the human stopped where the ego
did not (the 17 `cor_route_end` zeros) and where the slow scenes get slow (onset, triggers, fixed point, following gap), on RSG-L18-s0's
stored PAI 281 rollouts (Tokyo /data/runs/route_night/r281/rsg). CPU only; the run dirs are read, never written.

  slim     (Tokyo, host python3 + numpy)   rf_londiag.py slim --runs <run dir> ... --out <pkl>
      every driver decision of every session (driver/drive.jsonl, kind drive, 10 Hz): now, k, cmd, the fed ego row (20), the served plan
      `poses` and the model plan `poses_model` (8 x 0.5 s, decision rear-axle frame), the lead limit of `fix` (filtered lead distance d,
      lead speed vl, src, cut). Session id = rollout id.
  frames   (Tokyo, ffmpeg)   rf_londiag.py frames --list <json [{mp4, t_s, png}]> --out DIR      video frames at t_s (s after t0; the PAI
      mp4 is a composite: map + metric table on top, the front camera below, its clock = the controller's, checked to 0.1 s)
  route_end (Mac, .venv)   rf_londiag.py route_end --tax <rf_tax pkl> --wl <waitlines pkl> --dec <slim pkl> [--frames DIR]
      reading 1 -> results/lon_diag/route_end.csv, route_end.md, figs/lon_diag_route_end.png (front frame + BEV + v vs distance to stop)
  slow     (Mac, .venv)   rf_londiag.py slow --tax ... --wl ... --dec ... [--frames DIR] [--parquet <decisions.parquet>]
      reading 2 -> results/lon_diag/slow.md, slow.json, onsets.csv, figs/lon_diag_fixed_point.png, figs/lon_diag_onsets.png
  picks    (Mac)   rf_londiag.py picks --tax ... --wl ... --dec ... --out <json>   the frame list for `frames` (route-end arrivals, onsets)
  pai      (Mac / Tokyo, numpy + pandas)   rf_londiag.py pai --tax NEW.pkl --dec NEW_dec.pkl --ref-tax REF.pkl --ref-dec REF_dec.pkl --wl WL.pkl
      SLOW prereg line 3 (experiments/route_follow/plans/2026-10-15-slow-s1s2-prereg.md) for one PAI 281 read against the reference, scene-paired:
      speed ratio (rf_tax `speed_ratio`: median over the 0.5 s steps of ego speed / log speed at the ego's station) median on the reference
      taxonomy's 73 `low_slow_free` scenes (results/taxonomy/pai281_rsg.csv) >= 0.80; handoff (R2a's rows: valid decisions k 0-2, v_f >= 1)
      median v_p / v_f in the v_log >= 16 m/s bin >= 0.95; collision zeros (rf_tax zero_type `collision`: at-fault front / lateral) <= ref + 2;
      all zeros <= ref + 3; exam 72 mean-score difference CI low > -0.02 (scene bootstrap B 10 000, seed 0); rear-ended (rf_tax `rear_hit`)
      <= 0.7 x ref; `cor_route_end` primaries <= ref + 2. Inputs per read (Tokyo): `rf_tax.py extract --runs <read>/runs/* --out X.pkl
      --asl-actors` (alpasim-base image, as the taxonomy) and `rf_londiag.py slim --runs <read>/runs/* --out X_dec.pkl`; WL = the stored
      PAI wait lines (the taxonomy's `waitlines` pkl; same 281 scenes). --out F.json writes the numbers.

Definitions (prereg): v_f = fed vx (ego[4] x 10, m/s); v_p = arc length of the served plan from the rear axle through its 0.5 .. 2.0 s poses
/ 2 s (rf_follow.plan_arc2 / 2); v_log = the log's speed at the ego's station on the log path (rf_tax.Roll alignment: rear-axle position
projected on the logged rig path); short = v_p - v_f. Decision pose = the controller trace at the decision's `now`.
Scene features per decision (route = AlpaSim's route polyline from the rf_tax extract, objects = the rollout.asl actors at 0.5 s,
interpolated by id to the decision time):
  vehicle in corridor  a vehicle (automobile, truck, bus, trailer, other_vehicle, tram) with its centre within hw + W / 2 + 0.3 m of the
                       route and its near edge 0 .. 40 m ahead of the front bumper along the route, either stationary (< 0.5 m/s) or
                       moving in the route's direction (heading within 60 deg): nearest gap (m) and its speed
  stop line            a map wait line (any class) with its centre within 4 m of the route (rf_tax stop_lines' lateral rule), 0 .. 60 m
                       ahead of the front bumper: nearest distance and class
  turn                 max |heading change| of the route (chord heading over +-2 m) between the ego's station and any station up to +60 m
  junction             the evaluator lane map's intersection flag at the ego (rf_tax extract, 0.5 s grid, the step at or before)
  free                 no vehicle in corridor within 40 m, no stop line within 60 m, turn < 30 deg, not in a junction, and the ego
                       within 5 m of the route (else the route features are not about the ego's road)
Decisions after the first zero-score event, rear contact or any contact (rf_tax classify's truncation) are flagged `valid` = False.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402


# ---------------------------------------------------------------- Tokyo: slim decision extract
def slim_one(f):
    S = {}
    for line in open(f):
        if '"kind": "drive"' not in line:
            continue
        r = json.loads(line)
        L = (r.get("fix") or {}).get("lead") or {}
        S.setdefault(r["session"], dict(scene=r["scene"], rows=[]))["rows"].append(
            (int(r["k"]), int(r["now"]), int(r.get("cmd") or 0), r["ego"], r["poses"], r.get("poses_model") or r["poses"],
             np.nan if L.get("d") is None else float(L["d"]), np.nan if L.get("vl") is None else float(L["vl"]), L.get("src") == "mpc",
             float(L.get("cut") or 0.0), float((r.get("fix") or {}).get("v0", np.nan))))
    out = {}
    for s, d in S.items():
        R = sorted(d["rows"], key=lambda x: x[0])
        c = list(zip(*R))
        out[s] = dict(scene=d["scene"], k=np.array(c[0], int), now=np.array(c[1], np.int64), cmd=np.array(c[2], np.int8),
                      ego=np.array(c[3], np.float32), poses=np.array(c[4], np.float32), poses_model=np.array(c[5], np.float32),
                      lead_d=np.array(c[6], np.float32), lead_vl=np.array(c[7], np.float32), lead_mpc=np.array(c[8], bool),
                      lead_cut=np.array(c[9], np.float32), fix_v0=np.array(c[10], np.float32))
    return out


def slim(a):
    from concurrent.futures import ProcessPoolExecutor
    fs = [str(Path(r) / "driver/drive.jsonl") for r in a.runs if (Path(r) / "driver/drive.jsonl").exists()]
    out = {}
    with ProcessPoolExecutor(min(a.workers, len(fs))) as ex:
        for d in ex.map(slim_one, fs):
            out.update(d)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(fs), "drive logs,", len(out), "sessions,", sum(len(v["k"]) for v in out.values()), "decisions ->", a.out, flush=True)


def frames(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for x in json.loads(Path(a.list).read_text()):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(x['t_s'], 0.0):.2f}", "-i", x["mp4"], "-frames:v", "1", "-vf",
                        "scale=960:-1", str(out / x["png"])], check=False)
    print(len(list(out.glob("*.png"))), "frames ->", out)


# ---------------------------------------------------------------- Mac: per-decision table
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RES, FIGS = REPO / "experiments/route_follow/results/lon_diag", REPO / "experiments/route_follow/figs"
TAXCSV = REPO / "experiments/route_follow/results/taxonomy/pai281_rsg.csv"
RED24 = REPO / "experiments/route_follow/results/red/pai_red24_rsg.csv"
EXAM = REPO / "experiments/alpasim/results/pai_exam/hard_exam.txt"
VEHC = {"automobile", "heavy_truck", "trailer", "bus", "other_vehicle", "train_or_tram_car", "vehicle", "VEHICLE"}
SAME = np.radians(60.0)
D_VEH, D_LINE, D_TURN, TURN_FREE = 40.0, 60.0, 60.0, 30.0


def tx():
    sys.path.insert(0, str(HERE))
    import rf_tax
    return rf_tax


def plan_arc2(P):
    """(N, 8, >=2) plans -> arc length (m) from the rear axle through the 0.5 .. 2.0 s poses (rf_follow.plan_arc2, vectorised)."""
    P = np.asarray(P, float)[:, :4, :2]
    P = np.concatenate([np.zeros((len(P), 1, 2)), P], 1)
    return np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)).sum(1)


class Route:
    """Route polyline with stations, unwrapped chord heading on a 1 m grid, and the scene's wait lines projected on it."""

    def __init__(self, R, wl, TX):
        self.TX, self.R = TX, R
        self.S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]
        g = np.arange(0.0, self.S[-1] + 1.0, 1.0)
        xy = lambda s: np.c_[np.interp(s, self.S, R[:, 0]), np.interp(s, self.S, R[:, 1])]  # noqa: E731
        d = xy(np.minimum(g + 2, self.S[-1])) - xy(np.maximum(g - 2, 0))
        self.g, self.h = g, np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        L = [(w["kind"], *TX.poly_proj(R, w["pts"].mean(0)[None])) for w in wl]
        self.lines = [(k, float(s[0])) for k, dd, s, lat, _ in L if abs(lat[0]) <= 4.0]

    def proj(self, P):
        return self.TX.poly_proj(self.R, P)

    def turn(self, s0, ahead=D_TURN):
        i0 = int(np.clip(round(s0), 0, len(self.g) - 1))
        i1 = int(np.clip(round(s0 + ahead), 0, len(self.g) - 1))
        return float(np.degrees(np.abs(self.h[i0:i1 + 1] - self.h[i0]).max())) if i1 > i0 else 0.0

    def line(self, s_front, ahead=D_LINE):
        c = [(s - s_front, k) for k, s in self.lines if 0.0 <= s - s_front <= ahead]
        return min(c) if c else (np.nan, "")


def corridor_vehicles(r, rt, t, s_front, ahead):
    """Vehicles in the route corridor 0 .. ahead m before the front bumper (stationary, or moving in the route's direction) at time t
    -> list of (gap m, speed, id, cls)."""
    TX = rt.TX
    st = r.objs_at(t)
    if not len(st["id"]):
        return []
    X = np.c_[st["x"], st["y"]]
    _, s, lat, h = rt.proj(X)
    out = []
    for n in range(len(X)):
        if str(st["cls"][n]) not in VEHC:
            continue
        gap = s[n] - st["L"][n] / 2 - s_front
        if not (-1.0 <= gap <= ahead) or abs(lat[n]) > r.hw + st["W"][n] / 2 + 0.3:
            continue
        moving = st["v"][n] >= 0.5 and not st["static"][n]
        if moving and abs(TX.wrap(st["yaw"][n] - h[n])) >= SAME:
            continue
        out.append((float(gap), float(st["v"][n]) if moving else 0.0, str(st["id"][n]), str(st["cls"][n])))
    return sorted(out)


def t_trunc(TX, o, row, r):
    ser = o["ser"]
    t_ev = r.t0 + row["t_ev_s"] * 1e6 if np.isfinite(row.get("t_ev_s", np.nan)) else np.nan
    v = [t_ev, TX.first_pos(ser, "collision_rear"), TX.first_pos(ser, "collision_any")]
    t = np.nanmin(v) if np.isfinite(np.nanmin(v + [np.inf])) else r.t[-1]
    return min(t, r.t[-1])


def roll_table(job):
    """One rollout -> (gate row, per-decision rows)."""
    o, d, wl, exam = job
    TX = tx()
    row, _, _ = TX.classify(o, None, wl)
    r = TX.Roll(o)
    rt = Route(r.route, wl, TX)
    tr = t_trunc(TX, o, row, r)
    t = d["now"].astype(float)
    tc = np.clip(t, r.t[0], r.t[-1])
    P = np.c_[np.interp(tc, r.t, r.x), np.interp(tc, r.t, r.y)]
    vf = d["ego"][:, 4].astype(float) * 10.0
    vp = plan_arc2(d["poses"]) / 2.0
    vpm = plan_arc2(d["poses_model"]) / 2.0
    s_log = r.log_station(P)
    vlog = np.interp(s_log, r.LS, r.Lv)
    _, s_rt, lat_rt, _ = rt.proj(P)
    jg = np.clip(np.searchsorted(r.tg, tc, side="right") - 1, 0, len(r.tg) - 1)
    rows = []
    for n in range(len(t)):
        sf = s_rt[n] + r.front
        V = corridor_vehicles(r, rt, tc[n], sf, D_LINE)
        g40 = [x for x in V if x[0] <= D_VEH]
        dl, kl = rt.line(sf)
        turn = rt.turn(s_rt[n])
        junc = bool(o["junc"][jg[n]])
        rows.append(dict(scene=o["scene"], k=int(d["k"][n]), t=(t[n] - r.t0) / 1e6, v_f=vf[n], v_p=vp[n], v_pm=vpm[n], v_log=vlog[n],
                         short=vp[n] - vf[n], v_ctrl=r.speed(tc[n]), s_log=s_log[n], beyond_log=bool(s_log[n] >= r.LS[-1] - 0.5),
                         route_lat=lat_rt[n], veh_gap=V[0][0] if V else np.nan, veh_v=V[0][1] if V else np.nan, veh_id=V[0][2] if V else "",
                         veh40=bool(g40), line_d=dl, line_kind=kl, line60=bool(np.isfinite(dl)), turn60=turn, turn30=turn >= TURN_FREE, junc=junc,
                         free=not g40 and not np.isfinite(dl) and turn < TURN_FREE and not junc and abs(lat_rt[n]) <= 5.0, valid=bool(t[n] <= tr), exam=exam,
                         cmd=int(d["cmd"][n]), lead_d=float(d["lead_d"][n]), lead_mpc=bool(d["lead_mpc"][n]), lead_cut=float(d["lead_cut"][n])))
    g = dict(scene=o["scene"], rid=o["rid"], score=row["score"], zero_type=row["zero_type"], primary=row["primary"], t_tr=(tr - r.t0) / 1e6,
             ahead_of_log_s=row.get("ahead_of_log_s", np.nan), n_dec=len(t))
    return g, rows


def load_inputs(a):
    TX = tx()
    rolls = pickle.load(open(a.tax, "rb"))["rolls"]
    WL = pickle.load(open(a.wl, "rb"))
    D = pickle.load(open(a.dec, "rb"))
    exam = set(Path(EXAM).read_text().split())
    return TX, {o["scene"]: o for o in rolls}, WL, D, exam


def table(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    TX, rolls, WL, D, exam = load_inputs(a)
    jobs = [(o, D[o["rid"]], WL.get(s, []), s in exam) for s, o in sorted(rolls.items())]
    G, rows = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for g, rr in ex.map(roll_table, jobs, chunksize=4):
            G.append(g)
            rows += rr
    G, T = pd.DataFrame(G), pd.DataFrame(rows)
    # gate (a): score / zero type / primary against the taxonomy CSV
    C = pd.read_csv(TAXCSV).set_index("scene")
    m = G.set_index("scene").join(C[["score", "zero_type", "primary", "ahead_of_log_s"]], rsuffix="_csv")
    bad = m[~np.isclose(m.score.astype(float).fillna(-1), m.score_csv.astype(float).fillna(-1), rtol=1e-3, atol=1e-5) | (m.zero_type.fillna("") != m.zero_type_csv.fillna(""))
            | (m.primary != m.primary_csv)]
    print(f"gate a: {len(m) - len(bad)} / {len(m)} rows match the taxonomy CSV (score to its 4 significant digits, zero type, primary)")
    if len(bad):
        print(bad[["score", "score_csv", "zero_type", "zero_type_csv", "primary", "primary_csv"]].head(20))
    # gate (b): plan arc against rf_follow.plan_arc2 on a few decisions
    sys.path.insert(0, str(HERE))
    import rf_follow as RF
    rng = np.random.default_rng(0)
    err = []
    for rid in rng.choice(sorted(D), 5, replace=False):
        d = D[rid]
        for i in rng.choice(len(d["k"]), 4, replace=False):
            v = float(np.hypot(*d["ego"][i, 4:6].astype(float)) * 10.0)
            ref = RF.plan_arc2(d["poses"][i]) - 2.0 * v
            mine = plan_arc2(d["poses"][i:i + 1])[0] - 2.0 * v
            err.append(abs(ref - mine))
    print(f"gate b: max |plan arc - rf_follow.plan_arc2| over {len(err)} decisions = {max(err) * 100:.4f} cm")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    T.to_parquet(a.out)
    G.to_csv(Path(a.out).with_name("rolls.csv"), index=False)
    print(len(T), "decisions ->", a.out)


# ---------------------------------------------------------------- reading 1: the 17 route-end zeros
def ext_log(r, ahead=40.0):
    h = np.arctan2(*(r.LP[-1] - r.LP[-2])[::-1])
    return np.vstack([r.LP, r.LP[-1] + ahead * np.array([np.cos(h), np.sin(h)])])


def first_cross(t, s, lat, s0):
    """First time the station s(t) reaches s0 (|lateral| < 6 m), linear between samples; NaN when never."""
    i = np.flatnonzero((s >= s0) & (np.abs(lat) < 6.0))
    if not len(i):
        return np.nan
    i = i[0]
    if i == 0:
        return float(t[0])
    f = (s0 - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
    return float(t[i - 1] + np.clip(f, 0, 1) * (t[i] - t[i - 1]))


def route_end_scene(TX, o, d, wl, red24, exam, row_csv):
    r = TX.Roll(o)
    LT, LS, Lv = r.LT, r.LS[:len(r.LT)], r.Lv
    tail = len(Lv) - (np.flatnonzero(Lv >= 0.5)[-1] + 1 if (Lv >= 0.5).any() else 0)
    i0 = len(Lv) - tail if tail else len(Lv) - 1
    s_stop, t_stop = float(LS[i0]), float(LT[i0])
    v_end, log_len = float(Lv[-1]), float(LS[-1])
    Lx = ext_log(r)
    rt = Route(Lx, wl, TX)
    # queue: stationary vehicles in the route corridor 0 .. 15 m ahead of the human's front bumper when the human stopped (Vr, the
    # prereg's rule; V = the same on the straight-extended log path, reported only)
    tq = float(np.clip(t_stop, r.tg[0], r.tg[-1]))
    V = [x for x in corridor_vehicles(r, rt, tq, s_stop + r.front, 15.0) if x[1] < 0.5]
    rr = Route(r.route, wl, TX)
    Pst = np.c_[np.interp(s_stop, r.LS, r.LP[:, 0]), np.interp(s_stop, r.LS, r.LP[:, 1])]
    s_rr = float(rr.proj(Pst)[1][0])
    Vr = [x for x in corridor_vehicles(r, rr, tq, s_rr + r.front, 15.0) if x[1] < 0.5]
    # wait lines 0 .. 12 m ahead of the stop point (rear axle), centre within 4 m of the log path
    kinds = sorted({k for k, sl in rt.lines if 0.0 <= sl - s_stop <= 12.0})
    kinds_wide = sorted({k for k, sl in rt.lines if -3.0 <= sl - s_stop <= 12.0})
    if v_end >= 1.0 or log_len < 2.0:
        cause = "E"
    elif Vr:
        cause = "Q"
    elif "STOP" in kinds:
        cause = "L-STOP"
    elif kinds == ["UNKNOWN"]:
        cause = "L-LIGHT"
    else:
        cause = "N"
    # ego arrival at the stop point (station on the extended log path)
    _, se, le, _ = TX.poly_proj(Lx, np.c_[r.x, r.y])
    t_r = first_cross(r.t, se, le, s_stop)
    t = d["now"].astype(float)
    vf, vp, vpm = d["ego"][:, 4].astype(float) * 10.0, plan_arc2(d["poses"]) / 2.0, plan_arc2(d["poses_model"]) / 2.0
    w = (t >= t_r - 3e6) & (t <= t_r) & (vf > 0.5) if np.isfinite(t_r) else np.zeros(len(t), bool)
    rat, ratm = vp[w] / vf[w], vpm[w] / vf[w]
    # gate (c): ahead_of_log_s by this script's own alignment
    t_ev = TX.first_pos(o["ser"], "left_corridor_laterally")
    s_ev = float(r.log_station(r.pose(t_ev)[None, :2])[0]) if np.isfinite(t_ev) else np.nan
    ahead = (np.interp(s_ev, LS, LT) - t_ev) / 1e6 if np.isfinite(t_ev) else np.nan
    sd = np.interp(np.clip(t, r.t[0], r.t[-1]), r.t, se) - s_stop
    row = dict(scene=o["scene"], s8=o["scene"][7:15], exam=exam, red24=o["scene"] in red24, cause=cause, log_v_end=v_end, log_len=log_len,
               s_stop=s_stop, t_stop_s=(t_stop - r.t0) / 1e6, queue_veh=";".join(f"{c}@{g:.1f}m" for g, _, _, c in Vr),
               queue_veh_logpath=";".join(f"{c}@{g:.1f}m" for g, _, _, c in V), q_dt_s=(t_stop - tq) / 1e6, lines_0_12=("+".join(kinds)),
               lines_m3_12="+".join(kinds_wide), t_r_s=(t_r - r.t0) / 1e6 if np.isfinite(t_r) else np.nan,
               v_arrive=r.speed(t_r) if np.isfinite(t_r) else np.nan, v_log_40m=float(np.interp(max(s_stop - 40, 0), LS, Lv)),
               ego_early_at_stop_s=(t_stop - t_r) / 1e6 if np.isfinite(t_r) else np.nan, n_win=int(w.sum()),
               min_ratio=float(rat.min()) if w.any() else np.nan, intent=bool((rat <= 0.7).any()) if w.any() else False,
               min_ratio_model=float(ratm.min()) if w.any() else np.nan, intent_model=bool((ratm <= 0.7).any()) if w.any() else False,
               lead_present=bool(np.isfinite(d["lead_d"][w]).any()), lead_limit=bool(d["lead_mpc"][w].any() or (d["lead_cut"][w] > 0.05).any()),
               ahead_of_log_s=ahead, ahead_csv=float(row_csv["ahead_of_log_s"]), score=float(o["score"] or 0))
    cur = dict(sd_ego=se - s_stop, v_ego=r.v, sd_log=LS - s_stop, v_log=Lv, sd_dec=sd, v_f=vf, v_p=vp, t=t, win=w)
    return row, cur


def re_compute(a):
    import pandas as pd
    TX, rolls, WL, D, exam = load_inputs(a)
    C = pd.read_csv(TAXCSV).set_index("scene")
    red24 = set(pd.read_csv(RED24).scene)
    out = []
    for s in sorted(C.index[C.primary == "cor_route_end"]):
        o = rolls[s]
        out.append((o, *route_end_scene(TX, o, D[o["rid"]], WL.get(s, []), red24, s in exam, C.loc[s])))
    return TX, out


def route_end(a):
    import pandas as pd
    TX, out = re_compute(a)
    R = pd.DataFrame([x[1] for x in out])
    gate = (R.ahead_of_log_s - R.ahead_csv).abs().max()
    print(f"gate c: max |ahead_of_log_s - CSV| over {len(R)} route-end rows = {gate:.3f} s")
    RES.mkdir(parents=True, exist_ok=True)
    R.to_csv(RES / "route_end.csv", index=False, float_format="%.4g")
    nE = R[R.cause != "E"]
    no = int((~nE.intent).sum())
    l1 = "does not read the stop (signal problem)" if no >= 10 else "reads it but late / weak (magnitude problem)" if no <= 5 else "mixed"
    nLQ, nN = int(nE.cause.isin(["L-LIGHT", "Q"]).sum()), int((nE.cause == "N").sum())
    md = ["# LON-DIAG reading 1: the 17 `cor_route_end` zeros of RSG-L18-s0 on PAI 281", "",
          "Prereg: `experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md` (reading 1). Script: "
          "`experiments/route_follow/scripts/rf_londiag.py route_end`. Contact sheet: `figs/lon_diag_route_end.png`.", "",
          f"Gate c (this script's arc alignment vs the taxonomy CSV's `ahead_of_log_s`): max |difference| = {gate:.3f} s (line: < 0.2 s).", "",
          "Human stop point = start of the log's final run below 0.5 m/s. Cause (exclusive, in order): E log end speed >= 1 m/s or log < 2 m; "
          "Q stationary (< 0.5 m/s) vehicle in the route corridor (AlpaSim route, centre within hw + W / 2 + 0.3 m) 0-15 m ahead of the human's front bumper at the human's stop time (asl "
          "actors; the time is clipped to the rollout, `q_dt_s`; `queue_veh_logpath` in the CSV = the same on the log path extended straight); L-STOP a STOP wait line 0-12 m ahead of the stop point (centre within 4 m "
          "of the log path); L-LIGHT only UNKNOWN wait lines there; N otherwise. Intent = any decision in the 3 s before the ego reaches the "
          "stop point (arc on the extended log path) with v_p <= 0.7 v_f (served plan; `intent_model` = the same on `poses_model`). "
          "`lead_present` = the serve-side lead filter saw a lead (fix.lead.d set); `lead_limit` = the lead MPC cut the plan.", "",
          "| scene | exam | red24 | cause | wait lines 0-12 m | queue vehicle | intent | min v_p/v_f (3 s) | model plan intent | v arrive (m/s) "
          "| human v 40 m before | ego early at stop (s) | ahead_of_log_s | lead seen / limit |", "|:--|:-:|:-:|:--|:--|:--|:-:|--:|:-:|--:|--:|--:|--:|:-:|"]
    for _, x in R.iterrows():
        md.append(f"| {x.s8} | {'x' if x.exam else ''} | {'x' if x.red24 else ''} | {x.cause} | {x.lines_0_12 or '-'} | {x.queue_veh or '-'} | "
                  f"{'yes' if x.intent else 'no'} | {x.min_ratio:.2f} | {'yes' if x.intent_model else 'no'} | {x.v_arrive:.1f} | {x.v_log_40m:.1f} | "
                  f"{x.ego_early_at_stop_s:.1f} | {x.ahead_of_log_s:.2f} | {'y' if x.lead_present else 'n'} / {'y' if x.lead_limit else 'n'} |")
    cc = R.groupby("cause").agg(n=("scene", "size"), no_intent=("intent", lambda z: int((~z).sum())), red24=("red24", "sum"), exam=("exam", "sum"))
    md += ["", "| cause | n | no intent | red24 | exam |", "|:--|--:|--:|--:|--:|"]
    md += [f"| {k} | {v.n} | {v.no_intent} | {v.red24} | {v.exam} |" for k, v in cc.iterrows()]
    md += ["", "## Registered lines", "",
           f"- **L1** (does the car read 'stop'): {len(nE)} non-E scenes, {no} without deceleration intent -> **{l1}** "
           f"(lines: >= 10 'does not read', <= 5 'late / weak', else mixed).",
           f"- **L2** (where the signal is): L-LIGHT + Q = {nLQ} (line >= 8: labels exist in nuPlan) -> **{'pass' if nLQ >= 8 else 'fail'}**; "
           f"N = {nN} (line >= 5: needs the videos) -> **{'pass' if nN >= 5 else 'fail'}**.", "", RE_NOTE, "",
           "![route-end contact sheet](../../figs/lon_diag_route_end.png)"]
    (RES / "route_end.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-4:]))
    re_sheet(TX, out, Path(a.frames) if a.frames else None)


RE_NOTE = """## Not registered: what the frames show (one look at the camera half of each frame, 2026-10-15; for the user to confirm)

The cause classes come from the map's wait-line classes. In the frames at the ego's arrival, signal heads are visible over the ego's approach
in most L-STOP scenes (054b5901 and 35e082a4 show a red; 44df49fe, 4cb7f7ac, 9f8acd00, ba40be62, bc309fd6 show signal heads), and also at
68c3be44 (L-LIGHT) and 2d10a08d (N). 7d994ed3 is a light-rail crossing (a train passes on the left). 127c4f85 (the one scene with intent) shows no
signal head. 4ecc114d (N) has stopped cars ahead in the next lane. So the PAI `STOP` wait-line class does not mean a stop sign, and the L2
count (L-LIGHT + Q = 2) understates the signalised stops. The rule split stands as registered; read the contact sheet before acting on L2."""


def re_sheet(TX, out, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, nc = len(out), 3
    nr = int(np.ceil(n / nc))
    fig = plt.figure(figsize=(9.0 * nc, 9.4 * nr))
    gs = fig.add_gridspec(nr * 2, nc * 2, height_ratios=[1.0, 0.95] * nr, hspace=0.32, wspace=0.14, top=0.975, bottom=0.01, left=0.02,
                          right=0.99)
    for i, (o, x, c) in enumerate(out):
        rI, cI = 2 * (i // nc), 2 * (i % nc)
        axf = fig.add_subplot(gs[rI, cI:cI + 2])
        f = frames / f"re_{x['s8']}.png" if frames else None
        if f is not None and f.exists():
            axf.imshow(camera(f))
        axf.set_axis_off()
        lab = f"{x['s8']}  [{x['cause']}]{'  red24' if x['red24'] else ''}{'  exam' if x['exam'] else ''}  lines {x['lines_0_12'] or '-'}"
        axf.set_title(lab + f"\nfront at ego arrival t = {x['t_r_s']:.1f} s; intent {'yes' if x['intent'] else 'no'}, "
                      f"v {x['v_arrive']:.1f} m/s, {x['ego_early_at_stop_s']:.1f} s before the human", fontsize=12)
        axb = fig.add_subplot(gs[rI + 1, cI])
        r = TX.Roll(o)
        t_r = r.t0 + x["t_r_s"] * 1e6 if np.isfinite(x["t_r_s"]) else r.t[-1]
        TX.draw_bev(axb, o, t_r, "", None, 35.0)
        P = np.c_[np.interp(x["s_stop"], r.LS, r.LP[:, 0]), np.interp(x["s_stop"], r.LS, r.LP[:, 1])][0]
        axb.plot(*P, marker="X", ms=9, color="#2f9e44", mec="black", zorder=9)
        axv = fig.add_subplot(gs[rI + 1, cI + 1])
        m = (c["sd_log"] >= -45) & (c["sd_log"] <= 5)
        axv.plot(c["sd_log"][m], c["v_log"][m], color="black", lw=1.2, label="v_log (human)")
        m = (c["sd_ego"] >= -45) & (c["sd_ego"] <= 10)
        axv.plot(c["sd_ego"][m], c["v_ego"][m], color="#e03131", lw=1.2, label="v ego")
        m = (c["sd_dec"] >= -45) & (c["sd_dec"] <= 10)
        axv.plot(c["sd_dec"][m], c["v_p"][m], color="#f08c00", lw=0.9, label="v_p (served plan)")
        axv.plot(c["sd_dec"][m], c["v_f"][m], color="#1c7ed6", lw=0.7, ls="--", label="v_f (fed)")
        axv.axvline(0, color="#2f9e44", lw=1.0)
        axv.set_xlim(-45, 10)
        axv.set_ylim(0, None)
        axv.set_xlabel("distance to the human's stop point (m)", fontsize=10)
        axv.set_ylabel("m/s", fontsize=10)
        axv.tick_params(labelsize=9)
        if i == 0:
            axv.legend(fontsize=9, loc="lower left")
    fig.suptitle("LON-DIAG reading 1: RSG-L18-s0 PAI 281 route-end zeros (green X / line = the human's stop point; cause E evaluator, "
                 "Q queue, L-STOP STOP-class wait line, L-LIGHT only unlabelled wait lines, N none)", fontsize=16, y=0.995)
    FIGS.mkdir(parents=True, exist_ok=True)
    save_png(fig, FIGS / "lon_diag_route_end.png", dpi=58)
    plt.close(fig)


def camera(f):
    """The front-camera half of a PAI composite video frame (bottom 49 %)."""
    import matplotlib.pyplot as plt
    im = plt.imread(f)
    return im[int(0.515 * im.shape[0]):]


def save_png(fig, path, dpi):
    """PNG, palette-quantised when it would exceed 2 MB (photos in the contact sheets)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    if path.stat().st_size > 2e6:
        from PIL import Image
        Image.open(path).convert("RGB").quantize(256).save(path, optimize=True)
    print(path, f"{path.stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------- reading 2: slow scenes, onsets, fixed point, following
def cluster_boot(stat, units, B=10_000, seed=0):
    """Percentile CI of stat(index array of units) over B resamples of units (default_rng(seed), jevdrive.stats' convention)."""
    idx = np.random.default_rng(seed).integers(0, units, (B, units))
    b = np.array([stat(i) for i in idx])
    b = b[np.isfinite(b)]
    return (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) else (np.nan, np.nan)


def ols_sums(df, x, y, by):
    g = df.assign(xx=df[x] ** 2, xy=df[x] * df[y], one=1.0).groupby(by)[[x, y, "xx", "xy", "one"]].sum()
    return g.to_numpy()


def ols_from(Sx):
    sx, sy, sxx, sxy, n = Sx.sum(0)
    g = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return g, (sy - g * sx) / n


def onset_of(df):
    """First decision with v_p < 0.9 v_f whose next 3 decisions all stay below 0.95 v_f (valid decisions, v_f >= 1 m/s)."""
    q = (df.v_p / df.v_f).where(df.v_f >= 1.0).to_numpy()
    for i in range(len(q) - 3):
        if q[i] < 0.9 and np.all(q[i + 1:i + 4] < 0.95):
            return i
    return None


FEAT = [("veh40", "vehicle in corridor <= 40 m"), ("line60", "stop line <= 60 m"), ("turn30", "route turn >= 30 deg within 60 m"),
        ("junc", "in junction"), ("lead_seen", "(extra) serve lead filter sees a lead")]


def slow_compute(a):
    import pandas as pd
    T = pd.read_parquet(a.parquet)
    T["lead_seen"] = np.isfinite(T.lead_d)
    C = pd.read_csv(TAXCSV).set_index("scene")
    T["primary"] = T.scene.map(C.primary)
    T = T.sort_values(["scene", "k"]).reset_index(drop=True)
    V = T[T.valid]
    # R2a handoff
    H = V[(V.k <= 2) & (V.v_f >= 1.0)].assign(q=lambda z: z.v_p / z.v_f)
    bins = [0, 8, 12, 16, 100]
    H["bin"] = pd.cut(H.v_log, bins, right=False, labels=["<8", "8-12", "12-16", ">=16"])
    r2a = H.groupby("bin", observed=False).agg(n=("q", "size"), scenes=("scene", "nunique"), med=("q", "median"))
    r2a["med_k0"] = H[H.k == 0].groupby("bin", observed=False).q.median()
    hi, lo = r2a.loc[">=16", "med"], r2a.loc["<8", "med"]
    r2a_line = ("distribution ceiling" if hi < 0.85 and lo >= 0.95 else "not short at the handoff (all in the loop)" if (r2a.med >= 0.95).all()
                else "neither line")
    # R2b onsets
    slow = sorted(C.index[C.primary == "low_slow_free"])
    ons, on_rows = {}, []
    for s in slow:
        df = V[V.scene == s].reset_index(drop=True)
        i = onset_of(df)
        if i is None:
            on_rows.append(dict(scene=s, onset=False))
            continue
        t_on = df.t[i]
        w_on = df[(df.t >= t_on - 1.0) & (df.t <= t_on)]
        w_c = df[(df.t >= t_on - 6.0) & (df.t <= t_on - 3.0)]
        rec = dict(scene=s, s8=s[7:15], onset=True, k_on=int(df.k[i]), t_on=t_on, v_f_on=df.v_f[i], v_p_on=df.v_p[i], v_log_on=df.v_log[i],
                   exam=bool(df.exam[i]), n_ctrl=len(w_c), score=float(C.loc[s, "score"]))
        for f, _ in FEAT:
            rec[f + "_on"] = float(w_on[f].mean())
            rec[f + "_ctrl"] = float(w_c[f].mean()) if len(w_c) >= 10 else np.nan
        on_rows.append(rec)
        ons[s] = (df, i)
    On = pd.DataFrame(on_rows)
    Oc = On[On.onset & On.n_ctrl.ge(10)].reset_index(drop=True)
    rr = {}
    for f, lab in FEAT:
        x_on, x_c = Oc[f + "_on"].to_numpy(), Oc[f + "_ctrl"].to_numpy()
        st = lambda i: x_on[i].mean() / x_c[i].mean() if x_c[i].mean() > 0 else np.nan  # noqa: E731
        lo_, hi_ = cluster_boot(st, len(Oc))
        rr[f] = dict(label=lab, on=float(x_on.mean()), ctrl=float(x_c.mean()), rr=float(st(np.arange(len(Oc)))), lo=lo_, hi=hi_)
    reg = [f for f, _ in FEAT[:4]]
    trig = [f for f in reg if rr[f]["rr"] >= 2 and rr[f]["lo"] > 1]
    r2b_line = ("trigger: " + ", ".join(trig)) if trig else ("no trigger (ratchet without a scene cause)" if all(rr[f]["lo"] <= 1 <= rr[f]["hi"] for f in reg)
                                                         else "neither line (some CI excludes 1 without RR >= 2)")
    # R2c fixed point on free decisions
    Fd = V[V.free & (V.v_f >= 1.0)].copy()
    r2c = {}
    for name, sub in (("pooled", Fd), ("exam", Fd[Fd.exam]), ("non-exam", Fd[~Fd.exam])):
        Sx = ols_sums(sub, "v_f", "v_p", "scene")
        g, c0 = ols_from(Sx)
        lo_, hi_ = cluster_boot(lambda i: ols_from(Sx[i])[0], len(Sx))
        b = sub.assign(b=np.floor(sub.v_f)).groupby("b").agg(n=("v_p", "size"), vp=("v_p", "mean"), vf=("v_f", "mean"))
        b = b[b.n >= 30]
        dlt = (b.vp - b.vf).to_numpy()
        xs = b.vf.to_numpy()
        cr = [float(xs[j] + dlt[j] * (xs[j + 1] - xs[j]) / (dlt[j] - dlt[j + 1])) for j in range(len(dlt) - 1) if dlt[j] > 0 >= dlt[j + 1]]
        r2c[name] = dict(n=len(sub), scenes=int(sub.scene.nunique()), g=float(g), g_lo=lo_, g_hi=hi_, a=float(c0),
                         v_star_ols=float(c0 / (1 - g)) if g < 1 else np.nan, v_star_bins=cr, bins=b.reset_index().to_dict("records"),
                         below_all=bool((dlt <= 0).all()))
    return T, V, r2a, r2a_line, On, Oc, rr, r2b_line, r2c, ons


def follow_gaps(a, T):
    """R2d: per decision of the 12 low_slow_behind_lead scenes with a moving lead in the corridor <= 40 m: the ego's gap (m, s) and the
    human's gap to the same actor when the log was at the same station."""
    import pandas as pd
    TX, rolls, WL, D, exam = load_inputs(a)
    C = pd.read_csv(TAXCSV).set_index("scene")
    rows = []
    for s in sorted(C.index[C.primary == "low_slow_behind_lead"]):
        o = rolls[s]
        r = TX.Roll(o)
        rt = Route(r.route, WL.get(s, []), TX)
        df = T[(T.scene == s) & T.valid & (T.veh_v >= 0.5) & (T.veh_gap <= D_VEH) & (T.v_f >= 1.0)]
        for _, x in df.iterrows():
            tl = float(np.interp(x.s_log, r.LS[:len(r.LT)], r.LT))
            if not (r.tg[0] <= tl <= r.tg[-1]) or x.v_log < 1.0:
                continue
            st = r.objs_at(tl)
            k = np.flatnonzero(st["id"] == x.veh_id)
            if not len(k):
                continue
            k = k[0]
            Ph = np.c_[np.interp(x.s_log, r.LS, r.LP[:, 0]), np.interp(x.s_log, r.LS, r.LP[:, 1])]
            sh = float(rt.proj(Ph)[1][0])
            so = float(rt.proj(np.c_[st["x"][k], st["y"][k]])[1][0])
            gh = so - st["L"][k] / 2 - (sh + r.front)
            rows.append(dict(scene=s, s8=s[7:15], t=x.t, gap_e=x.veh_gap, gap_h=gh, v_e=x.v_f, v_h=x.v_log, th_e=x.veh_gap / x.v_f, th_h=gh / x.v_log,
                             v_lead=x.veh_v))
    return pd.DataFrame(rows)


def slow(a):
    import pandas as pd
    T, V, r2a, r2a_line, On, Oc, rr, r2b_line, r2c, ons = slow_compute(a)
    Fg = follow_gaps(a, T)
    RES.mkdir(parents=True, exist_ok=True)
    On.to_csv(RES / "onsets.csv", index=False, float_format="%.4g")
    Fg.to_csv(RES / "follow_gaps.csv", index=False, float_format="%.4g")
    if Fg.empty:
        sc, r2d_line = pd.DataFrame(), "no data"
    else:
        sc = Fg[Fg.gap_h > 0].groupby("s8").agg(n=("t", "size"), gap_e=("gap_e", "median"), gap_h=("gap_h", "median"), th_e=("th_e", "median"),
                                              th_h=("th_h", "median"), v_e=("v_e", "median"), v_h=("v_h", "median"), v_lead=("v_lead", "median"))
        sc["ratio"] = sc.th_e / sc.th_h
        med = float(sc.ratio.median())
        r2d_line = "follows far (gap)" if med >= 1.5 else "follows slowly (the lead is slow / normal gap)"
    on = On[On.onset]
    md = ["# LON-DIAG reading 2: the slow scenes of RSG-L18-s0 on PAI 281", "",
          "Prereg: `experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md` (reading 2). Script: `rf_londiag.py table` (per-decision "
          "table, Tokyo `/data/runs/route_follow/lon_diag/decisions.parquet`, not in git) and `rf_londiag.py slow`. Definitions in the script header. "
          f"{len(T)} decisions (10 Hz) of 281 rollouts; {int(T.valid.sum())} before the first zero-score event / contact (`valid`); "
          f"{int((T.valid & T.free & (T.v_f >= 1)).sum())} valid free decisions at v_f >= 1 m/s.", "",
          "## R2a handoff (k = 0-2, the fed history is still the log's)", "",
          "| v_log bin (m/s) | decisions | scenes | median v_p / v_f (k 0-2) | median at k = 0 |", "|:--|--:|--:|--:|--:|"]
    md += [f"| {b} | {int(x.n)} | {int(x.scenes)} | {x.med:.3f} | {x.med_k0:.3f} |" for b, x in r2a.iterrows()]
    md += ["", f"Line (>= 16 bin < 0.85 and < 8 bin >= 0.95 -> distribution ceiling; all bins >= 0.95 -> not short at the handoff): **{r2a_line}**.", "",
           "## R2b onset of the slow 73 (`low_slow_free`)", "",
           f"Onset = first valid decision (v_f >= 1 m/s) with v_p < 0.9 v_f whose next 3 decisions (0.3 s) stay below 0.95 v_f. Found in {len(on)} / {len(On)} "
           f"scenes; onset time median {on.t_on.median():.1f} s (IQR {on.t_on.quantile(.25):.1f}-{on.t_on.quantile(.75):.1f}), "
           f"{int((on.k_on <= 2).sum())} at k <= 2 (the handoff), {int((on.t_on < 3).sum())} before 3 s. At the onset v_f median {on.v_f_on.median():.1f} m/s, "
           f"v_log {on.v_log_on.median():.1f} m/s, v_p / v_f {np.median(on.v_p_on / on.v_f_on):.2f}. "
           f"RR uses the {len(Oc)} scenes with >= 1 s of control decisions in [onset - 6, onset - 3] s: per scene the fraction of decisions with the "
           "feature in [onset - 1 s, onset] against the fraction in the control window; RR = ratio of the scene means; scene bootstrap B 10 000, seed 0.", "",
           "| feature | onset window | control window | RR [95 % CI] |", "|:--|--:|--:|:--|"]
    md += [f"| {v['label']} | {v['on']:.3f} | {v['ctrl']:.3f} | " + (f"{v['rr']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" if np.isfinite(v['rr'])
           else "undefined (control fraction 0)") + " |" for v in rr.values()]
    h0, h1 = on[on.k_on <= 2], on[on.k_on > 2]
    md += ["", f"Handoff onsets (k <= 2, {len(h0)} scenes) against later onsets ({len(h1)}), medians: v_log at the onset {h0.v_log_on.median():.1f} vs "
           f"{h1.v_log_on.median():.1f} m/s, v_f {h0.v_f_on.median():.1f} vs {h1.v_f_on.median():.1f} m/s, v_p / v_f {np.median(h0.v_p_on / h0.v_f_on):.2f} vs "
           f"{np.median(h1.v_p_on / h1.v_f_on):.2f}; later onsets at t = {h1.t_on.median():.1f} s (median). Per-scene rows: `onsets.csv`."]
    md += ["", f"Line (any registered feature RR >= 2 and CI low > 1 -> trigger; all CIs contain 1 -> no trigger): **{r2b_line}**.", "",
           "## R2c fixed point (valid free decisions, v_f >= 1 m/s)", "",
           "| set | decisions | scenes | slope g [95 % CI] | intercept | v* = a / (1 - g) | v* from 1 m/s bins (mean v_p crosses v_f downward) |",
           "|:--|--:|--:|:--|--:|--:|:--|"]
    for k, v in r2c.items():
        md.append(f"| {k} | {int(v['n'])} | {v['scenes']} | {v['g']:.3f} [{v['g_lo']:.3f}, {v['g_hi']:.3f}] | {v['a']:+.2f} | {v['v_star_ols']:.1f} | "
                  f"{', '.join(f'{z:.1f}' for z in v['v_star_bins']) or ('none: below the diagonal everywhere' if v['below_all'] else 'none')} |")
    p = r2c["pooled"]
    vs = p["v_star_bins"][0] if p["v_star_bins"] else p["v_star_ols"]
    r2c_line = ("comfort speed: g < 1 and v* <= 10 m/s" if p["g"] < 1 and np.isfinite(vs) and vs <= 10 else
                "no fixed point: constant offset (g >= 0.95)" if p["g"] >= 0.95 else "neither line")
    md += ["", f"Line (g < 1 and v* <= 10 m/s -> a comfort speed the ratchet decays to; g >= 0.95 -> constant offset): pooled **{r2c_line}**. "
           "Bin table (pooled):", "", "| v_f bin | n | mean v_f | mean v_p | v_p - v_f |", "|--:|--:|--:|--:|--:|"]
    md += [f"| {int(b['b'])} | {b['n']} | {b['vf']:.2f} | {b['vp']:.2f} | {b['vp'] - b['vf']:+.2f} |" for b in p["bins"]]
    md += ["", "## R2d following (the 12 `low_slow_behind_lead`)", "",
           "Decisions with a moving (>= 0.5 m/s) same-direction vehicle in the corridor <= 40 m; the human's gap to the same actor id when the log "
           "was at the ego's station (asl actors replay the log); time gap = gap / own speed (both >= 1 m/s). Per-scene medians:", ""]
    if not sc.empty:
        md += ["| scene | decisions | ego gap m | human gap m | ego time gap s | human time gap s | ratio | ego v | human v | lead v |",
               "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        md += [f"| {k} | {int(x.n)} | {x.gap_e:.1f} | {x.gap_h:.1f} | {x.th_e:.2f} | {x.th_h:.2f} | {x.ratio:.2f} | {x.v_e:.1f} | {x.v_h:.1f} | {x.v_lead:.1f} |"
               for k, x in sc.iterrows()]
        md += ["", f"{len(sc)} of 12 scenes with paired decisions; median over scenes of the time-gap ratio = {float(sc.ratio.median()):.2f} "
               f"(ego {float(sc.th_e.median()):.2f} s, human {float(sc.th_h.median()):.2f} s). Line (median time gap >= 1.5 x the human's -> follows far, "
               f"else follows slowly): **{r2d_line}**."]
    md += ["", "## Figures", "", "![fixed point](../../figs/lon_diag_fixed_point.png)", "",
           "R2c: grey = decision density, orange = bin means; the bin mean sits a near-constant 0.3-0.5 m/s below the diagonal from 3 to 19 m/s "
           "and 1-2 m/s below above 20 m/s, so g < 1 comes mostly from the fast end and the bin crossing (v* ~ 3 m/s) is where the slow scenes settle "
           "(onset figure: the cars decay to 3-4 m/s and hold).", "", "![onsets](../../figs/lon_diag_onsets.png)", "",
           "R2b figure: the 8 lowest-score `low_slow_free` scenes with an onset at >= 1 s (most onsets are at the handoff and have no frame worth "
           "showing); red line = onset; the human (black) often speeds up while the plan decays."]
    (RES / "slow.md").write_text("\n".join(md) + "\n")
    js = dict(r2a=r2a.reset_index().astype({"bin": str}).to_dict("records"), r2a_line=r2a_line, r2b=rr, r2b_line=r2b_line, r2c=r2c, r2c_line=r2c_line,
              r2d=sc.reset_index().to_dict("records") if not sc.empty else [], r2d_line=r2d_line)
    (RES / "slow.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md))
    fp_fig(V, r2c)
    on_fig(a, ons, On, Path(a.frames) if a.frames else None)


def fp_fig(V, r2c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Fd = V[V.free & (V.v_f >= 1.0)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, (name, sub) in zip(axs, (("pooled", Fd), ("exam", Fd[Fd.exam]), ("non-exam", Fd[~Fd.exam]))):
        v = r2c[name]
        ax.hexbin(sub.v_f, sub.v_p, gridsize=60, extent=(0, 30, 0, 30), bins="log", cmap="Greys", mincnt=1)
        b = v["bins"]
        ax.plot([x["vf"] for x in b], [x["vp"] for x in b], "o-", color="#e8590c", ms=3.5, lw=1.4, label="mean v_p per 1 m/s bin (n >= 30)")
        ax.plot([0, 30], [0, 30], color="#1c7ed6", lw=1.0, ls="--", label="v_p = v_f")
        xx = np.array([0, 30])
        ax.plot(xx, v["a"] + v["g"] * xx, color="#2f9e44", lw=1.0, label=f"OLS g = {v['g']:.3f} [{v['g_lo']:.3f}, {v['g_hi']:.3f}]")
        for z in v["v_star_bins"]:
            ax.axvline(z, color="#e8590c", lw=0.8, ls=":")
        ax.set_title(f"{name}: {v['n']} free decisions, {v['scenes']} scenes; v* bins {', '.join(f'{z:.1f}' for z in v['v_star_bins']) or '-'}, "
                     f"OLS {v['v_star_ols']:.1f} m/s", fontsize=9)
        ax.set_xlabel("v_f fed speed (m/s)")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal")
        ax.legend(fontsize=7.5, loc="upper left")
    axs[0].set_ylabel("v_p served plan 2 s arc / 2 s (m/s)")
    fig.suptitle("LON-DIAG R2c: plan speed vs fed speed on free decisions (RSG-L18-s0, PAI 281 closed loop)", fontsize=11)
    fig.tight_layout()
    save_png(fig, FIGS / "lon_diag_fixed_point.png", dpi=110)
    plt.close(fig)


def on_pick(On):
    on = On[On.onset & (On.t_on >= 1.0)].sort_values("score")
    return on.head(8)


def on_fig(a, ons, On, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = on_pick(On)
    fig = plt.figure(figsize=(26, 14))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 0.75, 1.0, 0.75], hspace=0.38, wspace=0.15)
    for i, (_, x) in enumerate(P.iterrows()):
        df, j = ons[x.scene]
        axf = fig.add_subplot(gs[2 * (i // 4), i % 4])
        f = frames / f"on_{x.s8}.png" if frames else None
        if f is not None and f.exists():
            axf.imshow(camera(f))
        axf.set_axis_off()
        fl = [lab.split(" <=")[0].split(" >=")[0] for k, lab in FEAT[:4] if df.loc[max(j - 10, 0):j, k].any()]
        axf.set_title(f"{x.s8}{'  exam' if x.exam else ''}  score {x.score:.2f}: onset t = {x.t_on:.1f} s, v_f {x.v_f_on:.1f}, v_log {x.v_log_on:.1f} m/s"
                      f"\nin the 1 s before: {', '.join(fl) or 'nothing (free)'}", fontsize=8.5)
        ax = fig.add_subplot(gs[2 * (i // 4) + 1, i % 4])
        ax.plot(df.t, df.v_log, color="black", lw=1.1, label="v_log (human, same position)")
        ax.plot(df.t, df.v_f, color="#1c7ed6", lw=1.0, ls="--", label="v_f (fed)")
        ax.plot(df.t, df.v_p, color="#f08c00", lw=1.0, label="v_p (served plan)")
        ax.axvline(x.t_on, color="#e03131", lw=1.0)
        for k, c, yy in (("veh40", "#868e96", 0.4), ("line60", "#9c36b5", 0.9), ("turn30", "#2f9e44", 1.4), ("junc", "#e8590c", 1.9)):
            m = df[k].to_numpy()
            ax.scatter(df.t[m], np.full(m.sum(), yy), s=2, color=c, marker="|", label=k if i == 0 else None)
        ax.set_ylim(0, None)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=2)
    fig.suptitle("LON-DIAG R2b: onset of the plan deficit in 8 slow scenes (lowest score with onset >= 1 s); front frame at the onset; "
                 "ticks at the bottom = vehicle <= 40 m, stop line <= 60 m, turn >= 30 deg, junction", fontsize=11)
    save_png(fig, FIGS / "lon_diag_onsets.png", dpi=72)
    plt.close(fig)


def picks(a):
    TX, out = re_compute(a)
    L = [dict(mp4=o["files"]["mp4"], t_s=x["t_r_s"] if np.isfinite(x["t_r_s"]) else 0.0, png=f"re_{x['s8']}.png") for o, x, _ in out]
    *_, On, Oc, rr, r2b_line, r2c, ons = slow_compute(a)
    rolls = {o["scene"]: o for o in pickle.load(open(a.tax, "rb"))["rolls"]}
    L += [dict(mp4=rolls[x.scene]["files"]["mp4"], t_s=float(x.t_on), png=f"on_{x.s8}.png") for _, x in on_pick(On).iterrows()]
    Path(a.out).write_text(json.dumps(L, indent=1))
    print(len(L), "frames ->", a.out)


# ---------------------------------------------------------------- SLOW prereg line 3: one PAI read vs the reference
def pai_one(job):
    """One rollout -> (rf_tax classify row, handoff decisions k <= 2 with v_f, v_p, v_log, valid) (roll_table's definitions)."""
    o, d, wl = job
    TX = tx()
    row, _, _ = TX.classify(o, None, wl)
    out = []
    if d is not None and "ctrl" in o and o.get("ser") is not None:
        r = TX.Roll(o)
        tr = t_trunc(TX, o, row, r)
        m = d["k"] <= 2
        t = d["now"][m].astype(float)
        tc = np.clip(t, r.t[0], r.t[-1])
        P = np.c_[np.interp(tc, r.t, r.x), np.interp(tc, r.t, r.y)]
        vlog = np.interp(r.log_station(P), r.LS, r.Lv)
        vf, vp = d["ego"][m, 4].astype(float) * 10.0, plan_arc2(d["poses"][m]) / 2.0
        out = [dict(scene=o["scene"], k=int(k), v_f=f, v_p=q, v_log=v, valid=bool(tt <= tr)) for k, f, q, v, tt in zip(d["k"][m], vf, vp, vlog, t)]
    keep = ("scene", "score", "zero_type", "primary", "speed_ratio", "rear_hit")
    return {k: row.get(k) for k in keep}, out


def pai_read(tax, dec, wl, workers):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    rolls = pickle.load(open(tax, "rb"))["rolls"]
    D = pickle.load(open(dec, "rb"))
    WL = pickle.load(open(wl, "rb"))
    jobs = [(o, D.get(o["rid"]), WL.get(o["scene"], [])) for o in sorted(rolls, key=lambda o: o["scene"])]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(pai_one, jobs, chunksize=4))
    C = pd.DataFrame([r for r, _ in res]).set_index("scene")
    H = pd.DataFrame([x for _, h in res for x in h])
    return C, H


def pai(a):
    import pandas as pd
    C1, H1 = pai_read(a.tax, a.dec, a.wl, a.workers)
    C0, H0 = pai_read(a.ref_tax, a.ref_dec, a.wl, a.workers)
    T = pd.read_csv(TAXCSV).set_index("scene")
    slow = sorted(T.index[T.primary == "low_slow_free"])
    exam = sorted(set(Path(EXAM).read_text().split()) & set(C1.index) & set(C0.index))
    common = sorted(set(C1.index) & set(C0.index))

    def hand(H):
        V = H[H.valid & (H.k <= 2) & (H.v_f >= 1.0) & (H.v_log >= 16.0)]
        return float((V.v_p / V.v_f).median()), int(len(V)), int(V.scene.nunique())

    def counts(C):
        C = C.loc[common]
        sc = C.score.astype(float).fillna(0)
        return dict(scenes=len(C), mean=float(sc.mean()), zeros=int((sc == 0).sum()), coll_zeros=int(((sc == 0) & (C.zero_type == "collision")).sum()),
                    rear_ended=int(C.rear_hit.fillna(False).astype(bool).sum()), cor_route_end=int((C.primary == "cor_route_end").sum()),
                    slow_ratio_median=float(C.reindex(slow).speed_ratio.astype(float).median()), slow_scenes=int(C.reindex(slow).speed_ratio.notna().sum()))
    n, r = counts(C1), counts(C0)
    n["handoff_ge16"], n["handoff_rows"], n["handoff_scenes"] = hand(H1)
    r["handoff_ge16"], r["handoff_rows"], r["handoff_scenes"] = hand(H0)
    d = (C1.loc[exam].score.astype(float).fillna(0) - C0.loc[exam].score.astype(float).fillna(0)).to_numpy()
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10_000, len(d)))].mean(1)
    ex = [float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    lines = dict(slow_ratio=n["slow_ratio_median"] >= 0.80, handoff=n["handoff_ge16"] >= 0.95, coll_zeros=n["coll_zeros"] <= r["coll_zeros"] + 2,
                 zeros=n["zeros"] <= r["zeros"] + 3, exam=ex[1] > -0.02, rear=n["rear_ended"] <= 0.7 * r["rear_ended"],
                 cor_route_end=n["cor_route_end"] <= r["cor_route_end"] + 2)
    res = dict(run=a.tax, ref=a.ref_tax, new=n, ref_counts=r, diff={k: n[k] - r[k] for k in n if isinstance(n[k], (int, float))},
               exam72_mean_diff=ex, exam_scenes=len(exam), lines=lines, line3=all(lines.values()))
    f = lambda k, p=3: f"{n[k]:.{p}f} vs {r[k]:.{p}f} (d {n[k] - r[k]:+.{p}f})" if isinstance(n[k], float) else f"{n[k]} vs {r[k]} (d {n[k] - r[k]:+d})"  # noqa: E731
    print(f"# SLOW line 3: {a.tax} vs {a.ref_tax} ({len(common)} paired scenes)")
    print(f"slow-scene speed ratio median ({n['slow_scenes']} of {len(slow)} low_slow_free scenes): {f('slow_ratio_median')}  line >= 0.80: {lines['slow_ratio']}")
    print(f"handoff k 0-2 v_p / v_f, v_log >= 16 ({n['handoff_rows']} decisions / {n['handoff_scenes']} scenes): {f('handoff_ge16')}  line >= 0.95: {lines['handoff']}")
    print(f"collision zeros: {f('coll_zeros')}  line <= ref + 2: {lines['coll_zeros']}")
    print(f"all zeros: {f('zeros')}  line <= ref + 3: {lines['zeros']}; mean score {f('mean', 4)}")
    print(f"exam {len(exam)} mean-score difference {ex[0]:+.4f} [{ex[1]:+.4f}, {ex[2]:+.4f}]  line CI low > -0.02: {lines['exam']}")
    print(f"rear-ended: {f('rear_ended')}  line <= 0.7 x ref: {lines['rear']}")
    print(f"cor_route_end: {f('cor_route_end')}  line <= ref + 2: {lines['cor_route_end']}")
    print(f"LINE 3: {'PASS' if res['line3'] else 'FAIL'}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("slim")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    t = sp.add_parser("table")
    for k in ("tax", "wl", "dec", "out"):
        t.add_argument(f"--{k}", required=True)
    t.add_argument("--workers", type=int, default=8)
    for name in ("route_end", "slow", "picks"):
        q = sp.add_parser(name)
        for k in ("tax", "wl", "dec"):
            q.add_argument(f"--{k}", required=True)
        q.add_argument("--parquet", default="tmp/lon_diag/decisions.parquet")
        q.add_argument("--frames", default="")
        q.add_argument("--out", default="")
    q = sp.add_parser("pai")
    for k in ("tax", "dec", "ref-tax", "ref-dec", "wl"):
        q.add_argument(f"--{k}", required=True)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--out", default="")
    f = sp.add_parser("frames")
    f.add_argument("--list", required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"slim": slim, "frames": frames, "table": table, "route_end": route_end, "slow": slow, "picks": picks, "pai": pai}[a.cmd](a)


if __name__ == "__main__":
    main()
