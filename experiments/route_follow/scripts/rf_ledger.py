#!/usr/bin/env python3
"""RSGBOTH-LEDGER (research/route-follow/rsgboth-ledger.md): what RSGboth solved against RSG on the held 460 @18, by failure class, and how
every held turn is driven (completion, entry speed, onset, path shape, exit, jerkiness) against the human log. CPU only, Mac (.venv).

Inputs per label: an rf_tax.py extract pkl of the held read (box: rf_tax.py extract --runs $M/<tag>/c*/run) and its rf_tax.py report CSV;
rf_follow.py's scenes.csv of the same reads (the 24 strict turns, their outcome).

  rf_ledger.py report --set LABEL=<pkl>,<tax csv> ... --follow <scenes.csv> --pairs A:B ... --group NAME=L1,L2 ... --out DIR --figs DIR
      -> DIR/ledger.md, ledger.csv (class x label counts / points), ledger_pairs.csv, turns.md, turns.csv (label x turn), turn_pairs.csv,
         FIGS/ledger_bars.png, FIGS/ledger_turn_shape.png

Ledger
  turnzero   the turn-related zero classes regrouped: inside (off_*_inside, cor_turn_over), outside (off_*_outside, cor_turn_under),
             straight at the turn, wrong exit, junction straight-through.
  class      rf_tax primary label of the rollout (zeros: coll_* / off_* / cor_* / infra; low scores: low_*; ok); bucket = the taxonomy's
             section-5 grouping. Turn classes (24 strict rf_follow turns): turn_<outcome> (completed / event / off / short).
  points     sum over the class's scenes of (1 - score) / 460 x 100 (the mean gain if the class were fixed).
  pair A:B   per scene x = (1 - s_A) [class_A = c] - (1 - s_B) [class_B = c] (and the count analogue); mean x 100 with a log-clustered
             bootstrap 95 % CI (log = scene id without its hash). A group pair (NAME=seed labels) averages x over the seed pairs per scene.
  verdict    (A = RSGboth, B = RSG, group pair) solved: A has <= 1 scene in the class over both seeds while B has >= 3 and CI upper < 0;
             improved: CI upper < 0; worse: CI lower > 0; unchanged: CI holds 0 (trend sign noted when |diff| >= 0.2 points).

Turns (geometry on the human's logged path LP, stations s; ego = controller trace, 10 Hz, truncated at its first zero event)
  turn       LP between the ego's start station - 20 m and the log's station at the rollout's last controller time holds a turn: a 30 m
             window with net chord heading change (+-5 m) >= 45 deg, else a 60 m window >= 45 deg (rf_pai_follow find_turn, the change over the
             window +-10 m >= 75 %); entry / exit = 10 / 90 % of the change; exit must lie inside the log's travel (the held logs start at the
             rollout start and end ~9 s later, so most turns begin a few metres in). The first such turn per scene = the loose set; strict = the
             scenes rf_follow counted as turns (route, 60 deg / 30 m, exit + 10 m inside the log at decision 17), same LP geometry.
             left = counter-clockwise. Ego station / lateral by projection on LP with a -30 / +60 m window (left +).
  (a)        completion: strict = rf_follow's outcome; loose = the ego crosses min(exit + 10 m, log end - 0.5 m) before its zero event
             within 2.5 m of LP (else off when it crossed farther away or ends > 2.5 m off, event, short).
  (b)        speed when the ego first crosses entry - 40 / - 20 / 0 m, 50 % of the turn and exit + 10 m, against the log's speed at the same
             station; lag = ego time - human time at exit + 10 m (s). On held the log starts at the rollout start: -40 / -20 m rarely exist.
  (c)        onset = the first station (>= the path's first sample) where the heading, measured from the human's heading at entry - 15 m,
             has turned 15 % of the human's turn angle the turn's way; ego minus human (m, negative = early); also the 50 % station.
             Undefined when the human's onset lies before the ego's first sample.
  (d)        signed offset to LP, inside negative, at 25 / 50 / 75 / 100 % of [entry, exit]; peak inside / outside over [entry - 10,
             exit + 10]; shape: cut = peak inside < -1 m and peak outside < 1 m, wide = the mirror, weave = both, on = neither (also at
             0.5 m). The same offsets to AlpaSim's route (lane centre) for the ego and the human: the route-referenced "inside cut" of
             rf_inside.py is partly the human's own line.
  (e)        exit: |offset| at exit + 20 m (and + 10 m: the held logs often end before + 20 m); correct lane = |offset| <= 1.75 m and heading within 20 deg of the human's there.
  (f)        jerk: yaw rate (10 Hz, 0.3 s box smoothing; human from the logged yaw resampled to 10 Hz) over the time the path spends in
             [entry - 10, exit + 10]: peak |yaw rate| (deg/s) and reversals = turning points of the yaw rate with a 3 deg/s hysteresis
             minus one (a clean turn = 0).
  pairs      per turn A - B where both are defined; ego - human per label; log-clustered bootstrap 95 % CI (jevdrive.stats).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R_ = HERE.parents[2]
sys.path[:0] = [str(R_), str(HERE)]
import rf_tax as TX  # noqa: E402

BEND_WIN, BEND_DEG, HALF = 60.0, 45.0, 5.0
ONSET, MID = 0.15, 0.5
FRACS = (0.25, 0.5, 0.75, 1.0)
LANE_HALF, EXIT_PSI, HYST = 1.75, 20.0, 3.0
STATIONS = (40, 20, 0)
N = 460

BUCKET = {  # rf_tax primary -> taxonomy section-5 bucket
    "low_slow_free": "slow", "low_slow_behind_lead": "slow", "low_rear_ended": "stop / rear / other low",
    "low_with_log": "stop / rear / other low", "low_queue": "stop / rear / other low", "low_follow_lead": "stop / rear / other low",
    "low_other": "stop / rear / other low", "low_unexplained": "stop: unexplained / late / stopped lead",
    "low_late_start": "stop: unexplained / late / stopped lead", "low_stopped_lead": "stop: unexplained / late / stopped lead",
    "low_ped_yield": "stop / rear / other low",
    "cor_straight_at_turn": "corridor: route following", "cor_turn_under": "corridor: route following", "cor_turn_over": "corridor: route following",
    "cor_wrong_exit": "corridor: route following", "cor_lane_change": "corridor: route following", "cor_drift": "corridor: route following",
    "cor_route_end": "corridor: past the log end",
    "off_curve_inside": "offroad: curve / straight", "off_curve_outside": "offroad: curve / straight", "off_straight": "offroad: curve / straight",
    "off_junc_inside": "offroad: junction", "off_junc_outside": "offroad: junction", "off_junc_straight": "offroad: junction",
    "off_at_start": "infra (no rollout / start pose)", "infra": "infra (no rollout / start pose)",
    "coll_veh_ahead": "collision: vehicle", "coll_veh_side": "collision: vehicle", "coll_veh_oncoming": "collision: vehicle",
    "coll_veh_crossing": "collision: vehicle", "coll_stationary": "collision: vehicle", "coll_unknown": "collision: vehicle",
    "coll_ped": "collision: ped / cyclist", "coll_cyclist": "collision: ped / cyclist", "zero_other": "other zero", "ok": "ok (>= 0.9)",
}


TURNZ = {  # turn-related zero classes by what the ego did against the route turn (rf_tax offroad side rule / corridor rule)
    "off_junc_inside": "turn zero: inside (cut)", "off_curve_inside": "turn zero: inside (cut)", "cor_turn_over": "turn zero: inside (cut)",
    "off_junc_outside": "turn zero: outside (wide / under-turned)", "off_curve_outside": "turn zero: outside (wide / under-turned)",
    "cor_turn_under": "turn zero: outside (wide / under-turned)", "cor_straight_at_turn": "turn zero: went straight at the turn",
    "cor_wrong_exit": "turn zero: wrong exit", "off_junc_straight": "junction zero: straight through (no turn)"}


def logof(s):
    return np.array([x.rsplit("-", 1)[0] for x in np.asarray(s, str)])


def boot(x, g):
    from jevdrive import stats
    r = stats.bootstrap(np.asarray(x, float), g)
    return r["mean"], r["lo"], r["hi"], r["n"]


# ---------------------------------------------------------------- ledger
def ledger(tax, follow, groups, out):
    import pandas as pd
    rows, base = [], None
    for lab, df in tax.items():
        d = df[["scene", "score", "primary"]].copy()
        d["score"] = d.score.fillna(0.0).astype(float)
        d["primary"] = d.primary.fillna("infra").replace("", "infra")
        d["bucket"] = d.primary.map(BUCKET).fillna("other")
        d["turnzero"] = d.primary.map(TURNZ).where(d.score == 0).fillna("")
        f = follow[(follow.label == lab + "@18") & follow.turn.astype(bool)].set_index("scene").outcome
        d["turn"] = d.scene.map(f).map(lambda o: f"turn_{o}" if isinstance(o, str) else "")
        d["label"] = lab
        rows.append(d)
    D = pd.concat(rows)
    scenes = sorted(set.intersection(*[set(x.scene) for x in rows]))
    D = D[D.scene.isin(scenes)]
    lv = {"class": "primary", "bucket": "bucket", "turnzero": "turnzero", "turn": "turn"}
    tab = []
    for lvl, col in lv.items():
        for c in sorted(set(D[col]) - {""}):
            r = dict(level=lvl, cls=c)
            for lab in tax:
                x = D[D.label == lab]
                m = x[col] == c
                r[f"n|{lab}"], r[f"pts|{lab}"] = int(m.sum()), float((1 - x.score[m]).sum() / len(x) * 100)
            tab.append(r)
    T = pd.DataFrame(tab)
    T.to_csv(out / "ledger.csv", index=False, float_format="%.4g")
    W = {lab: D[D.label == lab].set_index("scene").loc[scenes] for lab in tax}
    lg = logof(scenes)
    P = []
    for name, (A, B) in groups.items():
        for lvl, col in lv.items():
            for c in sorted(set(D[col]) - {""}):
                xp, xn = 0, 0
                for a, b in zip(A, B):
                    wa, wb = W[a], W[b]
                    xp = xp + ((1 - wa.score) * (wa[col] == c) - (1 - wb.score) * (wb[col] == c)).to_numpy()
                    xn = xn + ((wa[col] == c).astype(float) - (wb[col] == c).astype(float)).to_numpy()
                xp, xn = xp / len(A) * 100, xn / len(A)
                m, lo, hi, _ = boot(xp, lg)
                na = sum(int((W[a][col] == c).sum()) for a in A)
                nb = sum(int((W[b][col] == c).sum()) for b in B)
                if na == 0 and nb == 0:
                    continue
                pa = sum(float(((1 - W[a].score) * (W[a][col] == c)).sum()) for a in A) / len(A) / len(scenes) * 100
                pb = sum(float(((1 - W[b].score) * (W[b][col] == c)).sum()) for b in B) / len(B) / len(scenes) * 100
                if c in ("ok (>= 0.9)", "ok", "turn_completed"):                # judged on the count: more scenes here is better
                    cb = boot(xn, lg)
                    v = "improved (more scenes)" if cb[1] > 0 else "worse (fewer scenes)" if cb[2] < 0 else "unchanged"
                elif na <= len(A) - 1 and nb >= 3 and hi < 0:
                    v = "solved"
                elif hi < 0:
                    v = "improved"
                elif lo > 0:
                    v = "worse"
                else:
                    v = "unchanged" + (" (trend better)" if m <= -0.2 else " (trend worse)" if m >= 0.2 else "")
                P.append(dict(pair=name, level=lvl, cls=c, n_a=na, n_b=nb, pts_a=pa, pts_b=pb, d_pts=m, lo=lo, hi=hi, d_n=float(xn.mean() * len(scenes)),
                              verdict=v))
    Pd = pd.DataFrame(P)
    Pd.to_csv(out / "ledger_pairs.csv", index=False, float_format="%.4g")
    return D, T, Pd, scenes


# ---------------------------------------------------------------- turns
def heading(R, S, step=1.0):
    s = np.arange(0.0, S[-1] + 1e-9, step)
    at = lambda q: np.c_[np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])]  # noqa: E731
    d = at(np.minimum(s + HALF, S[-1])) - at(np.maximum(s - HALF, 0.0))
    return s, np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def find_turn(s, h, lo, hi, win, deg):
    i0, i1 = np.searchsorted(s, lo), np.searchsorted(s, hi, side="right")
    w = int(round(win / (s[1] - s[0]))) if len(s) > 1 else 0
    for a in range(i0, max(i0, i1 - w)):
        if np.degrees(abs(h[a + w] - h[a])) >= deg:
            j0, j1 = max(0, a - 10), min(len(s) - 1, a + w + 10)
            th = h[j1] - h[j0]
            if np.degrees(abs(th)) < 0.75 * deg:
                continue
            f = (h[j0:j1 + 1] - h[j0]) / th
            k10, k90 = np.flatnonzero(f >= 0.1), np.flatnonzero(f >= 0.9)
            return float(s[j0 + k10[0]]), float(s[j0 + k90[0]]), float(th)
    return None


def track(R, S, P):
    """Path P -> stations, signed laterals (left +) on polyline R, local -30 / +60 m window."""
    st, la, hint = np.empty(len(P)), np.empty(len(P)), None
    a, b = R[:-1], R[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    for i, p in enumerate(P):
        u = np.clip(((p - a) * d).sum(1) / L2, 0, 1)
        q = a + u[:, None] * d
        s = S[:-1] + u * np.sqrt(L2)
        dist = np.hypot(*(p - q).T)
        ok = np.ones(len(s), bool) if hint is None else (s >= hint - 30) & (s <= hint + 60)
        if not ok.any():
            ok[:] = True
        j = int(np.flatnonzero(ok)[np.argmin(dist[ok])])
        cr = d[j, 0] * (p[1] - a[j, 1]) - d[j, 1] * (p[0] - a[j, 0])
        st[i], la[i], hint = s[j], np.sign(cr) * dist[j], s[j]
    return st, la


def mono(st, *ys):
    """Keep samples where the station sets a new maximum (first passage), so np.interp on station is defined."""
    k = np.r_[True, st[1:] > np.maximum.accumulate(st)[:-1]]
    return (st[k],) + tuple(y[k] for y in ys)


def at_station(st, y, s):
    return float(np.interp(s, st, y)) if len(st) and st[0] <= s <= st[-1] else np.nan


def smooth(x, k=3):
    return np.convolve(x, np.ones(k) / k, "same") if len(x) >= k else x


def reversals(w):
    """Turning points of w (deg/s) with HYST hysteresis, minus one."""
    if len(w) < 3:
        return np.nan
    n, hi, lo, d = 0, w[0], w[0], 0                                  # d: +1 rising, -1 falling, 0 not yet moved HYST
    for x in w[1:]:
        hi, lo = max(hi, x), min(lo, x)
        if d >= 0 and hi - x >= HYST:                                  # a maximum confirmed
            n, d, lo = n + 1, -1, x
        elif d <= 0 and x - lo >= HYST:                                # a minimum confirmed
            n, d, hi = n + 1, 1, x
    return max(n - 1, 0)


def turn_rows(lab, o, strict):
    """One extracted rollout -> turn row (None when no bend) + profile arrays for the figure."""
    if "ctrl" not in o or o.get("ser") is None:
        return None, None
    r = TX.Roll(o)
    ser = o["ser"]
    tz = np.nanmin([TX.first_pos(ser, k) for k in ("collision_front", "collision_lateral", "offroad", "left_corridor_laterally")]
                   if float(o["score"] or 0) == 0 else [np.nan])
    tz = tz if np.isfinite(tz) else np.inf
    LP, LS = r.LP, r.LS
    LT = r.LT
    Ly = np.unwrap(np.interp(LT, o["log"]["T"], o["log"]["Y"]))
    k = r.t <= tz
    E = np.c_[r.x[k], r.y[k]]
    if len(E) < 3:
        return None, None
    est, elat = track(LP, LS, E)
    eyaw, ev, et = r.yaw[k], r.v[k], r.t[k]
    ewz = smooth(np.degrees(o["ctrl"]["wz"][k]))
    s_log = float(np.interp(r.t[-1], LT, LS[:len(LT)]))
    s, h = heading(LP, LS)
    t = find_turn(s, h, max(est[0] - 20, 0), s_log, 30.0, BEND_DEG) or find_turn(s, h, max(est[0] - 20, 0), s_log, BEND_WIN, BEND_DEG)
    if t is None:
        return None, None
    ent, ext, th = t
    sg = np.sign(th)
    row = dict(label=lab, scene=o["scene"], log=o["scene"].rsplit("-", 1)[0], score=float(o["score"] or 0), strict=o["scene"] in strict,
               side="L" if th > 0 else "R", turn_deg=float(np.degrees(th)), entry=ent, exit=ext, ego_s0=float(est[0]), zero=bool(np.isfinite(tz)))
    # (a) loose completion
    i = np.flatnonzero(est >= min(ext + 10, LS[-1] - 0.5))                # the log may end before exit + 10 m
    row["completed_loose"] = bool(len(i) and abs(elat[i[0]]) <= 2.5)
    row["outcome_loose"] = ("completed" if row["completed_loose"] else "off" if len(i) else "event" if np.isfinite(tz) else
                            "off" if abs(elat[-1]) > 2.5 else "short")
    ms, mlat, myaw, mv, mt = mono(est, elat, eyaw, ev, et)
    # (b) speeds
    for d in STATIONS:
        q = ent - d
        row[f"v_ego_{d}"] = at_station(ms, mv, q) if q >= ms[0] - 0.5 else np.nan
        row[f"v_log_{d}"] = r.vlog_at(q) if q >= 0 else np.nan
    for nm, q in (("mid", (ent + ext) / 2), ("x10", ext + 10)):
        row[f"v_ego_{nm}"], row[f"v_log_{nm}"] = at_station(ms, mv, q), (r.vlog_at(q) if q <= LS[-1] else np.nan)
    q = ext + 10
    row["lag_x10"] = (at_station(ms, mt, q) - float(np.interp(q, LS[:len(LT)], LT))) / 1e6 if q <= LS[-1] else np.nan
    # (c) onset: heading relative to the human's at entry - 15 m
    hy = lambda q: float(np.interp(q, LS[:len(LT)], Ly))  # noqa: E731
    psi0 = hy(max(ent - 15, 0))
    THh = hy(ext + 10) - psi0
    for nm, fr in (("onset", ONSET), ("mid", MID)):
        thr = fr * abs(THh)
        sh = LS[:len(LT)]
        gh = sg * (Ly - psi0)
        jh = np.flatnonzero((gh >= thr) & (sh >= max(ent - 40, 0)))
        s_h = float(sh[jh[0]]) if len(jh) else np.nan
        ge = sg * (myaw - psi0)
        je = np.flatnonzero(ge >= thr)
        s_e = float(ms[je[0]]) if len(je) and je[0] > 0 else np.nan
        if not np.isfinite(s_h) or s_h < ms[0]:
            s_e = np.nan
        row[f"{nm}_h"], row[f"{nm}_e"], row[f"{nm}_d"] = s_h, s_e, s_e - s_h
    # (d) offsets to LP (inside negative) and to the route
    off = -sg * mlat
    Rt = r.route
    RS = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Rt, axis=0).T))]
    keep = np.r_[True, est[1:] > np.maximum.accumulate(est)[:-1]]
    Ep = E[keep]                                                      # ego positions of the first-passage samples (= ms)
    _, elr = track(Rt, RS, Ep)
    Hs = np.arange(max(ent - 30, 0), min(ext + 30, LS[-1]), 1.0)
    Hp = np.c_[np.interp(Hs, LS, LP[:, 0]), np.interp(Hs, LS, LP[:, 1])]
    _, hlr = track(Rt, RS, Hp)
    eoff_r, hoff_r = -sg * elr, -sg * hlr
    L = max(ext - ent, 1.0)
    for f in FRACS:
        q = ent + f * L
        row[f"off_{int(f * 100)}"] = at_station(ms, off, q)
        row[f"offr_ego_{int(f * 100)}"] = at_station(ms, eoff_r, q)
        row[f"offr_hum_{int(f * 100)}"] = at_station(Hs, hoff_r, q)
    w = (ms >= ent - 10) & (ms <= ext + 10)
    row["reached_turn"] = bool(w.sum() >= 3 and ms[-1] >= ext)
    if w.sum() >= 3:
        lo_, hi_ = float(off[w].min()), float(off[w].max())
        row.update(peak_in=lo_, peak_out=hi_, peak=lo_ if -lo_ > hi_ else hi_)
        for thr in (1.0, 0.5):
            c, wd = lo_ < -thr, hi_ > thr
            row[f"shape_{thr}"] = "weave" if c and wd else "cut" if c else "wide" if wd else "on"
        wr = (ms >= ent - 10) & (ms <= ext + 10)
        lo_r, hi_r = float(eoff_r[wr].min()), float(eoff_r[wr].max())
        wh = (Hs >= ent - 10) & (Hs <= ext + 10)
        row.update(route_peak_in_ego=lo_r, route_peak_out_ego=hi_r, route_peak_in_hum=float(hoff_r[wh].min()) if wh.any() else np.nan,
                   route_cut1_ego=bool(lo_r < -1.0), route_cut1_hum=bool(wh.any() and hoff_r[wh].min() < -1.0))
    # (e) exit
    for sfx, dq in (("", 20), ("10", 10)):
        q = ext + dq
        lat_ = abs(at_station(ms, mlat, q)) if q <= LS[-1] else np.nan
        psi_e, psi_h = at_station(ms, myaw, q), hy(q)
        row[f"exit{sfx}_lat"] = lat_
        row[f"exit{sfx}_psi"] = abs(np.degrees(TX.wrap(psi_e - psi_h))) if np.isfinite(psi_e) else np.nan
        row[f"exit{sfx}_ok"] = bool(lat_ <= LANE_HALF and row[f"exit{sfx}_psi"] <= EXIT_PSI) if np.isfinite(lat_) else np.nan
    # (f) jerk
    ke = (est >= ent - 10) & (est <= ext + 10)
    if ke.sum() >= 5 and row["reached_turn"]:
        row["yr_peak_e"] = float(np.abs(ewz[ke]).max())
        row["rev_e"] = reversals(ewz[ke])
    t0h, t1h = np.interp([max(ent - 10, 0), ext + 10], LS[:len(LT)], LT)
    tg = np.arange(t0h, t1h, 1e5)
    if len(tg) >= 5:
        wh = smooth(np.gradient(np.degrees(np.interp(tg, LT, Ly)), 0.1))
        row["yr_peak_h"], row["rev_h"] = float(np.abs(wh).max()), reversals(wh)
    # profiles for the figure: u = (s - entry) / (exit - entry)
    U = np.linspace(-1.0, 2.0, 61)
    qs = ent + U * L
    prof = dict(label=lab, scene=o["scene"], side=row["side"], strict=row["strict"], U=U,
                off=np.array([at_station(ms, off, x) for x in qs]),
                offr_e=np.array([at_station(ms, eoff_r, x) for x in qs]), offr_h=np.array([at_station(Hs, hoff_r, x) for x in qs]),
                hp_e=np.array([at_station(ms, sg * (myaw - psi0), x) for x in qs]) / max(abs(THh), 1e-3),
                hp_h=np.array([float(np.interp(x, LS[:len(LT)], sg * (Ly - psi0))) for x in qs]) / max(abs(THh), 1e-3),
                D=np.arange(-60, 31, 2.0))
    prof["v_e"] = np.array([at_station(ms, mv, ent + x) for x in prof["D"]])
    prof["v_h"] = np.array([r.vlog_at(ent + x) if ent + x >= 0 else np.nan for x in prof["D"]])
    return row, prof


def turns(pk, follow, out, pairs, groups):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    strict = set(follow[follow.turn.astype(bool)].scene)
    rows, profs = [], []
    jobs = []
    for lab, f in pk.items():
        jobs.append((lab, f, strict))
    with ProcessPoolExecutor(min(len(jobs), 4)) as ex:
        for R, Pp in ex.map(one_label, jobs):
            rows += R
            profs += Pp
    T = pd.DataFrame(rows)
    fo = follow[follow.turn.astype(bool)].copy()
    fo["label"] = fo.label.str.replace("@18", "", regex=False)
    T = T.merge(fo[["label", "scene", "outcome", "completed"]].rename(columns=dict(outcome="outcome_strict", completed="completed_strict")),
                on=["label", "scene"], how="left")
    T.to_csv(out / "turns.csv", index=False, float_format="%.4g")
    return T, profs


def one_label(job):
    lab, f, strict = job
    D = pickle.load(open(f, "rb"))["rolls"]
    R, P = [], []
    for o in D:
        try:
            r, p = turn_rows(lab, o, strict)
        except Exception as e:  # noqa: BLE001
            print("turn row failed", lab, o.get("scene"), repr(e)[:200], flush=True)
            continue
        if r is not None:
            R.append(r)
            P.append(p)
    return R, P


METRICS = [("completed_loose", "completion (loose rule)", 100), ("v_ego_40", "speed at entry - 40 m (m/s)", 1), ("v_ego_20", "speed at entry - 20 m", 1),
           ("v_ego_0", "speed at entry", 1), ("v_ego_mid", "speed at 50 % of the turn", 1), ("v_ego_x10", "speed at exit + 10 m", 1),
           ("lag_x10", "time lag at exit + 10 m vs human (s)", 1), ("onset_d", "onset - human (m, - = early)", 1), ("mid_d", "50 % heading - human (m)", 1),
           ("off_25", "offset at 25 % (m, inside -)", 1), ("off_50", "offset at 50 %", 1), ("off_75", "offset at 75 %", 1), ("off_100", "offset at 100 %", 1),
           ("peak_in", "peak inside (m)", 1), ("peak_out", "peak outside (m)", 1), ("cut1", "cut inside > 1 m (%)", 100), ("wide1", "wide > 1 m (%)", 100),
           ("weave1", "weave (both > 1 m) (%)", 100), ("route_cut1_ego", "inside the route > 1 m (rf_inside-like) (%)", 100),
           ("exit_lat", "|offset| at exit + 20 m (m)", 1), ("exit_ok", "correct lane at exit + 20 m (%)", 100),
           ("exit10_lat", "|offset| at exit + 10 m (m)", 1), ("exit10_ok", "correct lane at exit + 10 m (%)", 100), ("yr_peak_e", "peak yaw rate (deg/s)", 1),
           ("rev_e", "yaw-rate reversals", 1)]
HUMAN = {"v_ego_40": "v_log_40", "v_ego_20": "v_log_20", "v_ego_0": "v_log_0", "v_ego_mid": "v_log_mid", "v_ego_x10": "v_log_x10", "yr_peak_e": "yr_peak_h", "rev_e": "rev_h",
         "route_cut1_ego": "route_cut1_hum"}


def turn_stats(T, groups):
    """-> per (set, label) summaries, ego - human diffs, pair diffs (per seed pair and group)."""
    import pandas as pd
    T = T.copy()
    for thr, nm in ((1.0, "1"),):
        T[f"cut{nm}"] = (T[f"shape_{thr}"] == "cut").where(T[f"shape_{thr}"].notna())
        T[f"wide{nm}"] = (T[f"shape_{thr}"] == "wide").where(T[f"shape_{thr}"].notna())
        T[f"weave{nm}"] = (T[f"shape_{thr}"] == "weave").where(T[f"shape_{thr}"].notna())
    for c in ("completed_loose", "exit_ok", "exit10_ok", "route_cut1_ego", "route_cut1_hum", "completed_strict"):
        T[c] = T[c].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(bool(v)))
    sets = {"strict": T[T.strict], "loose": T, "loose L": T[T.side == "L"], "loose R": T[T.side == "R"]}
    S, H, P = [], [], []
    for sn, X in sets.items():
        for lab, g in X.groupby("label"):
            for m, _, sc in METRICS + [("completed_strict", "", 100)]:
                if m not in g:
                    continue
                mu, lo, hi, n = boot(g[m].astype(float) * sc, g.log)
                S.append(dict(set=sn, label=lab, metric=m, mean=mu, lo=lo, hi=hi, n=n))
                if m in HUMAN:
                    mu, lo, hi, n = boot((g[m].astype(float) - g[HUMAN[m]].astype(float)) * sc, g.log)
                    H.append(dict(set=sn, label=lab, metric=m, mean=mu, lo=lo, hi=hi, n=n))
            for hm in ("v_log_40", "v_log_20", "v_log_0", "v_log_mid", "v_log_x10", "yr_peak_h", "rev_h", "route_cut1_hum", "onset_h"):
                mu, lo, hi, n = boot(g[hm].astype(float) * (100 if hm == "route_cut1_hum" else 1), g.log)
                S.append(dict(set=sn, label=lab, metric=hm, mean=mu, lo=lo, hi=hi, n=n))
        for name, (A, B) in groups.items():
            for m, _, sc in METRICS + [("completed_strict", "", 100)]:
                xa = pd.concat([X[X.label == a].set_index("scene")[m].astype(float) for a in A], axis=1).mean(axis=1)
                xb = pd.concat([X[X.label == b].set_index("scene")[m].astype(float) for b in B], axis=1).mean(axis=1)
                j = xa.index.intersection(xb.index)
                d = (xa[j] - xb[j]) * sc
                mu, lo, hi, n = boot(d.to_numpy(), logof(j))
                P.append(dict(set=sn, pair=name, metric=m, a=float(xa[j][d.notna()].mean()) * sc, b=float(xb[j][d.notna()].mean()) * sc,
                              diff=mu, lo=lo, hi=hi, n=n))
    return pd.DataFrame(S), pd.DataFrame(H), pd.DataFrame(P), T


# ---------------------------------------------------------------- figures
C_H, C_B, C_N = "#3d3d3a", "#2a78d6", "#eb6834"


def fig_ledger(Pd, path, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = Pd[(Pd.pair == name) & (Pd.level == "bucket") & ~Pd.cls.str.startswith("ok")].sort_values("pts_b")
    t = Pd[(Pd.pair == name) & (Pd.level == "turnzero")].sort_values("pts_b")
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    for ax, d, title in ((axs[0], x, "held 460 @18: lost points by taxonomy bucket"), (axs[1], t, "turn-related zeros (what the ego did at the route turn)")):
        y = np.arange(len(d))
        ax.barh(y + 0.2, d.pts_b, 0.38, color=C_B, label="RSG-L18 (s0, s1 mean)")
        ax.barh(y - 0.2, d.pts_a, 0.38, color=C_N, label="RSGboth-L18 (s0, s1 mean)")
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.text(max(r.pts_a, r.pts_b) + 0.05, yi, f"{r.d_pts:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]  {r.verdict}", va="center", fontsize=8, color="#555")
        ax.set_yticks(y, [f"{c} ({int(r.n_b)}/{int(r.n_a)})" for c, (_, r) in zip(d.cls, d.iterrows())], fontsize=9)
        ax.set_xlabel("points of the held mean (sum of 1 - score / 460 x 100)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="x", color="#ddd", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_xlim(0, max(d.pts_a.max(), d.pts_b.max()) * 2.3 + 0.3)
    axs[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "labels: class (RSG / RSGboth scene count, both seeds summed); text: RSGboth - RSG paired points [95 % CI, log-clustered], verdict",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_shape(profs, path, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = next(iter(groups))
    A, B = groups[name]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, side in enumerate(("L", "R")):
        P = [p for p in profs if p["side"] == side]
        U = P[0]["U"] if P else np.linspace(-1, 2, 61)
        nsc = len({p["scene"] for p in P})
        for ax, key, ttl in ((axs[row, 0], "off", "offset to the human path (m, inside -)"),
                             (axs[row, 1], "offr", "offset to the route / lane centre (m, inside -)"),
                             (axs[row, 2], "hp", "heading change / human turn angle")):
            for labs, c, nm in ((B, C_B, "RSG-L18"), (A, C_N, "RSGboth-L18")):
                k = "offr_e" if key == "offr" else "hp_e" if key == "hp" else "off"
                M = np.array([p[k] for p in P if p["label"] in labs])
                if not len(M):
                    continue
                with np.errstate(all="ignore"):
                    md, q1, q3 = np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
                ax.plot(U, md, color=c, lw=2, label=nm)
                ax.fill_between(U, q1, q3, color=c, alpha=0.15, lw=0)
            if key in ("offr", "hp"):
                k = "offr_h" if key == "offr" else "hp_h"
                M = np.array([p[k] for p in P if p["label"] == B[0]])
                if len(M):
                    with np.errstate(all="ignore"):
                        ax.plot(U, np.nanmedian(M, 0), color=C_H, lw=2, ls="--", label="human (log)")
                        ax.fill_between(U, np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), color=C_H, alpha=0.10, lw=0)
            else:
                ax.axhline(0, color=C_H, lw=2, ls="--", label="human (log)")
            for v in (0, 1):
                ax.axvline(v, color="#bbb", lw=0.8, ls=":")
            ax.set_title(f"{'left' if side == 'L' else 'right'} turns ({nsc} scenes): {ttl}", fontsize=9.5, loc="left")
            ax.set_xlabel("turn progress u = (s - entry) / (exit - entry) on the human path")
            ax.grid(color="#eee", lw=0.6)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[row, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.01, 0.005, "loose turn set (>= 45 deg within 30 m, else 60 m, on the human log path), both seeds pooled; line = median, band = IQR; "
             "dotted verticals = entry (10 % heading) and exit (90 %).", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report
def fmt(m, lo, hi, f=".2f"):
    return "-" if not np.isfinite(m) else f"{m:{f}} [{lo:{f}}, {hi:{f}}]"


def report(a):
    import pandas as pd
    out, figs = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    pk, tax = {}, {}
    for spec in a.set:
        lab, fs = spec.split("=", 1)
        p, c = fs.split(",")
        pk[lab], tax[lab] = p, pd.read_csv(c)
    follow = pd.read_csv(a.follow)
    follow["turn"] = follow.turn.astype(str).isin(["True", "1", "1.0"])
    groups = {}
    for p in a.pairs:
        A, B = p.split(":")
        groups[p] = ([A], [B])
    for g in a.group or []:
        nm, spec = g.split("=", 1)
        A, B = spec.split(":")
        groups[nm] = (A.split(","), B.split(","))
    D, T, Pd, scenes = ledger(tax, follow, groups, out)
    gname = a.group[0].split("=", 1)[0] if a.group else a.pairs[0]
    fig_ledger(Pd, figs / "ledger_bars.png", gname)
    Tt, profs = turns(pk, follow, out, a.pairs, groups)
    S, H, P, Tt = turn_stats(Tt, groups)
    S.to_csv(out / "turn_summary.csv", index=False, float_format="%.4g")
    H.to_csv(out / "turn_vs_human.csv", index=False, float_format="%.4g")
    P.to_csv(out / "turn_pairs.csv", index=False, float_format="%.4g")
    fig_shape(profs, figs / "ledger_turn_shape.png", {gname: groups[gname]})
    # ledger.md
    md = ["# RSGboth vs RSG: solved / unsolved ledger on held 460 @18 (rf_ledger.py)", "",
          f"{len(scenes)} scenes common to all labels. Points = sum(1 - score) / {len(scenes)} x 100 in the class; pair = RSGboth - RSG per scene "
          "(log-clustered 95 % CI). Group pairs (seeds averaged): n = scenes summed over both seeds, points = seed mean. Rules in the script docstring.", ""]
    for name in groups:
        for lvl, title in (("bucket", "by taxonomy bucket"), ("turnzero", "turn-related zeros"), ("turn", "24 strict turns by rf_follow outcome"),
                           ("class", "by rf_tax primary class")):
            x = Pd[(Pd.pair == name) & (Pd.level == lvl)].sort_values("pts_b", ascending=False)
            md += [f"## {name}: {title}", "", "| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |",
                   "|:--|:--|:--|:--|--:|:--|"]
            md += [f"| {r.cls} | {r.n_b} / {r.n_a} | {r.pts_b:.2f} / {r.pts_a:.2f} | {fmt(r.d_pts, r.lo, r.hi)} | {r.d_n:+.1f} | {r.verdict} |"
                   for r in x.itertuples()]
            md.append("")
    (out / "ledger.md").write_text("\n".join(md) + "\n")
    # turns.md
    labs = list(pk)
    md = ["# Turning quality on held 460 @18: RSG vs RSGboth vs the human (rf_ledger.py)", "",
          "Turn sets: strict = rf_follow's 24 route turns; loose = the first bend >= 45 deg / 60 m on the human log path inside the rollout's span. "
          "Geometry on the human path; ego truncated at its zero event. Means with log-clustered 95 % CI. Rules in the script docstring.", ""]
    for sn in ("strict", "loose", "loose L", "loose R"):
        n = {lab: int(((Tt.label == lab) & (Tt.strict if sn == "strict" else Tt.side.isin(["L", "R"] if sn == "loose" else [sn[-1]]))).sum()) for lab in labs}
        md += [f"## {sn} ({', '.join(f'{k} {v}' for k, v in n.items())} turns)", "",
               "| metric | " + " | ".join(labs) + " | human |", "|:--|" + "--:|" * (len(labs) + 1)]
        for m, desc, _ in METRICS + [("completed_strict", "completion (rf_follow, strict only)", 100)]:
            if m == "completed_strict" and sn != "strict":
                continue
            cells = []
            for lab in labs:
                q = S[(S.set == sn) & (S.label == lab) & (S.metric == m)]
                cells.append(f"{q['mean'].iloc[0]:.2f} ({int(q['n'].iloc[0])})" if len(q) and np.isfinite(q["mean"].iloc[0]) else "-")
            hq = S[(S.set == sn) & (S.label == labs[0]) & (S.metric == HUMAN.get(m, "-"))]
            hm = {"onset_d": "0", "mid_d": "0", **{f"off_{int(f * 100)}": "0" for f in FRACS}, "exit_lat": "0", "exit_ok": "100", "exit10_lat": "0", "exit10_ok": "100"}
            hv = f"{hq['mean'].iloc[0]:.2f}" if len(hq) else hm.get(m, "")
            md.append(f"| {desc} | " + " | ".join(cells) + f" | {hv} |")
        X = Tt[Tt.strict] if sn == "strict" else Tt if sn == "loose" else Tt[Tt.side == sn[-1]]
        oc = X.groupby("label").outcome_loose.value_counts().unstack(fill_value=0).reindex(labs)
        md += ["", "Outcomes (loose rule): " + "; ".join(f"{lab} " + " / ".join(f"{c} {int(oc.loc[lab].get(c, 0))}" for c in ("completed", "event", "off", "short"))
                                                     for lab in labs)]
        md += ["", "Ego - human (paired per turn):", "", "| metric | " + " | ".join(labs) + " |", "|:--|" + ":--|" * len(labs)]
        for m in HUMAN:
            cells = []
            for lab in labs:
                q = H[(H.set == sn) & (H.label == lab) & (H.metric == m)]
                cells.append(fmt(q["mean"].iloc[0], q["lo"].iloc[0], q["hi"].iloc[0]) + f" ({int(q['n'].iloc[0])})" if len(q) else "-")
            md.append(f"| {m} | " + " | ".join(cells) + " |")
        md += ["", "Pairs (A - B per turn; group pairs average the seeds per turn):", "", "| metric | " + " | ".join(groups) + " |",
               "|:--|" + ":--|" * len(groups)]
        for m, desc, _ in METRICS + [("completed_strict", "completion (rf_follow)", 100)]:
            if m == "completed_strict" and sn != "strict":
                continue
            cells = []
            for g in groups:
                q = P[(P.set == sn) & (P.pair == g) & (P.metric == m)]
                cells.append(f"{q.a.iloc[0]:.2f} vs {q.b.iloc[0]:.2f}: " + fmt(q["diff"].iloc[0], q["lo"].iloc[0], q["hi"].iloc[0]) + f" ({int(q['n'].iloc[0])})"
                             if len(q) and q["n"].iloc[0] else "-")
            md.append(f"| {desc} | " + " | ".join(cells) + " |")
        md.append("")
    # per-turn table (strict, group means)
    md += ["## Per strict turn (both seeds listed: RSG s0 / s1 | RSGboth s0 / s1)", "",
           "| scene | side deg | outcome | onset - human (m) | offset 50 % (m) | peak in / out (m) | exit lat (m) | shape@1 m |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    st = Tt[Tt.strict]
    for sc, g in st.groupby("scene"):
        g = g.set_index("label").reindex(labs)
        v = lambda c, f=".1f": " / ".join("-" if not isinstance(x, (int, float, np.floating)) or not np.isfinite(x) else f"{x:{f}}" for x in g[c])  # noqa: E731
        md.append(f"| {sc[11:40]}…{sc[-6:]} | {g.side.dropna().iloc[0]} {g.turn_deg.dropna().iloc[0]:+.0f} | {' / '.join(str(x)[:5] for x in g.outcome_strict)} | "
                  f"{v('onset_d')} | {v('off_50')} | {v('peak_in')} ; {v('peak_out')} | {v('exit_lat')} | {' / '.join(str(x) for x in g['shape_1.0'])} |")
    (out / "turns.md").write_text("\n".join(md) + "\n")
    Tt.to_csv(out / "turns.csv", index=False, float_format="%.4g")
    print((out / "ledger.md").read_text()[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("report")
    r.add_argument("--set", nargs="+", required=True, help="LABEL=<rf_tax pkl>,<rf_tax report csv>")
    r.add_argument("--follow", required=True, help="rf_follow.py scenes.csv of the same reads (labels LABEL@18)")
    r.add_argument("--pairs", nargs="+", required=True, help="A:B label pairs (A = new)")
    r.add_argument("--group", nargs="*", help="NAME=A1,A2:B1,B2 seed-averaged pairs (the first one drives the figures)")
    r.add_argument("--out", required=True)
    r.add_argument("--figs", required=True)
    a = ap.parse_args()
    report(a)
