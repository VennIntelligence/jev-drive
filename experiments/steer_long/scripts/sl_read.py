#!/usr/bin/env python3
"""STEER-LONG read (research/next-round/steer-long.md, pre-registration section 3, commit 8f48d0c0): where the large steering before a
zero starts (plan or MPC steering), whether the model's yaw-history loop is visible, why the car is slow and stops, and what a driver-side
curvature cap would touch. Mac, CPU, on the compact copies made by sl_pull.py (tmp/steer_long/<track>/<label>/).

  sl_read.py [--root tmp/steer_long] [--out experiments/steer_long/results]

Frames: AlpaSim local frame, ego = rig (rear axle), x forward, y left. The returned trajectory is rebuilt as the drivers build it
(navsim_transfuser_challenge.trajectory: anchor + the 8 plan points at 0.5 s on the plan's own clock, linear in between).
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
WB = 2.85                                            # AlpaSim vehicle model wheelbase (Ford Fusion defaults)
LABELS = {"nuplan": ["YR10m10-F-s0", "YR10m10-F-s1", "YS-L18-s0", "YS-L18-s1", "YS2G-L18-s0", "YS2G-L18-s1"],
          "pai": ["ybase", "ys0", "ys1"]}
PAI_NAME = {"ybase": "YR10m10-F-s0", "ys0": "YS-L18-s0", "ys1": "YS-L18-s1"}
FLAGS = (("collision_at_fault", "collision_any", "collision"), ("offroad", "offroad", "offroad"),
         ("left_corridor_laterally", "left_corridor_laterally", "corridor"))
SER = ("dist_traveled_m", "gt_dist_traveled_m", "min_distance_to_obstacle_m", "collision_any", "offroad", "left_corridor_laterally")
TP = np.arange(9) * 0.5                               # plan clock of anchor + 8 points
CAPS_A = (2.0, 3.0, 4.0, 5.0)
CAPS_J = (2.5, 5.0, None)


def bucket(v):
    return np.where(v < 5, "low", np.where(v < 15, "mid", "high"))


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def yaw_q(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def log_of(track, scene):
    return scene.rsplit("-", 1)[0] if track == "nuplan" else scene


# ---------------------------------------------------------------- loading
def load(track, lab, root):
    d = root / track / lab
    D = defaultdict(list)
    for line in gzip.open(d / "drive.jsonl.gz", "rt"):
        r = json.loads(line)
        if r.get("infer", True):
            D[r["session"]].append(r)
    C = pd.read_csv(d / "ctrl.csv.gz")
    C["yaw"] = yaw_q(C.qx, C.qy, C.qz, C.qw)
    C = {s: g.sort_values("timestamp_us").reset_index(drop=True) for s, g in C.groupby("session")}
    M = {}                                           # metrics64.parquet: the per-rollout series with timestamps cast from i128 (sl_pull.py docstring)
    t = pq.read_table(d / "metrics64.parquet").to_pandas()
    t = t[t.name.isin(SER)].sort_values(["rid", "name", "timestamps_us"])
    for (rid, n), g in t.groupby(["rid", "name"]):
        M.setdefault(rid, {})[n] = (g.timestamps_us.to_numpy(np.int64), g["values"].astype(float).to_numpy())
    S = {r["rollout_id"]: r for r in json.load(open(d / "summary.json")) if r.get("rollout_id")}
    return D, C, M, S


class Plans:
    """The served (and model) plans of one session, rebuilt in the local frame on the plan clock."""

    def __init__(self, recs):
        recs = sorted(recs, key=lambda r: r["now"])
        self.now = np.array([r["now"] for r in recs], np.int64)
        self.t0 = np.array([r["t0"] for r in recs], np.int64)
        self.anchor = np.array([r["anchor"] for r in recs], float)
        self.poses = np.array([r["poses"] for r in recs], float)
        self.model = np.array([r.get("poses_model", r["poses"]) for r in recs], float)
        self.fix = [r.get("fix") for r in recs]
        c, s = np.cos(self.anchor[:, 2]), np.sin(self.anchor[:, 2])
        P = self.poses[:, :, :2]
        self.X = np.concatenate([self.anchor[:, None, :2], self.anchor[:, None, :2] + np.stack(
            [c[:, None] * P[..., 0] - s[:, None] * P[..., 1], s[:, None] * P[..., 0] + c[:, None] * P[..., 1]], -1)], 1)   # (n, 9, 2)

    def at(self, j, tau):
        tau = np.clip(tau, 0, 4)
        return np.c_[np.interp(tau, TP, self.X[j, :, 0]), np.interp(tau, TP, self.X[j, :, 1])]

    def kappa(self, j, ta, tb):
        """Mean curvature of plan j between plan-clock ta and tb (heading of the first and last 0.1 s chord over the arc between)."""
        tb = min(tb, 4.0)
        if tb - ta < 0.3:
            return np.nan, np.nan
        g = np.arange(ta, tb + 1e-9, 0.1)
        p = self.at(j, g)
        ds = np.hypot(*np.diff(p, axis=0).T)
        arc = ds.sum()
        den = arc - 0.5 * (ds[0] + ds[-1])
        if arc < 0.5 or den < 0.3:
            return np.nan, arc / (g[-1] - g[0])
        th1 = np.arctan2(*(p[1] - p[0])[::-1])
        th2 = np.arctan2(*(p[-1] - p[-2])[::-1])
        return float(wrap(th2 - th1) / den), arc / (g[-1] - g[0])


def arc(P, n):
    """Arc length from the anchor through the first n plan points (anchor frame)."""
    Q = np.vstack([[0.0, 0.0], P[:n, :2]])
    return float(np.hypot(*np.diff(Q, axis=0).T).sum())


# ---------------------------------------------------------------- the cap rule (section 3.7)
def native_kappa(P):
    """Curvature at the 0.5 s corners of a plan in the anchor frame: corner 0 = circle from the ego pose (heading 0) through point 1,
    corner i = heading change / mean adjacent segment length. Returns kappa (8,), segment lengths (8,), corner speeds (8,)."""
    Q = np.vstack([[0.0, 0.0], P[:, :2]])
    d = np.diff(Q, axis=0)
    ds = np.hypot(*d.T)
    th = np.arctan2(d[:, 1], d[:, 0])
    k = np.empty(8)
    k[0] = 2 * np.sin(th[0]) / max(ds[0], 1e-3)
    mid = 0.5 * (ds[1:] + ds[:-1])
    k[1:] = wrap(th[1:] - th[:-1]) / np.maximum(mid, 1e-3)
    v = np.r_[ds[0], mid] / 0.5
    return k, ds, v


def rebuild(k, ds):
    th = np.empty(8)
    th[0] = np.arcsin(np.clip(k[0] * ds[0] / 2, -1, 1))
    mid = 0.5 * (ds[1:] + ds[:-1])
    th[1:] = th[0] + np.cumsum(k[1:] * mid)
    return np.cumsum(np.c_[ds * np.cos(th), ds * np.sin(th)], axis=0)


def cap_session(pl, a_max, j_max):
    """Apply the cap to every plan of a session in order (the rate cap compares with the previous capped plan at the same absolute
    time). Returns per plan: max |p' - p| over plan points at 1.0 / 1.5 / 2.0 s, the capped y at 2 s."""
    n = len(pl.now)
    dev, y2c = np.zeros(n), np.zeros(n)
    prev = None
    for j in range(n):
        P = pl.poses[j]
        k, ds, v = native_kappa(P)
        lim = a_max / np.maximum(v, 3.0) ** 2
        kc = np.where(ds < 0.05, k, np.clip(k, -lim, lim))
        if j_max is not None and prev is not None:
            tc = (pl.t0[j] - prev[0]) / 1e6 + TP[:8]                     # this plan's corner times on the previous plan's clock
            kp = np.interp(tc, TP[:8], prev[1])
            ok = tc <= 3.5
            dt = (pl.now[j] - pl.now[j - 1]) / 1e6
            r = j_max / np.maximum(v, 3.0) ** 2 * dt
            kc = np.where(ok & (ds >= 0.05), np.clip(kc, kp - r, kp + r), kc)
        Pc = rebuild(kc, ds)
        dev[j] = np.abs(np.hypot(*(Pc[1:4] - P[1:4, :2]).T)).max()
        y2c[j] = Pc[3, 1]
        prev = (pl.t0[j], kc)
    return dev, y2c


# ---------------------------------------------------------------- per rollout
def series(Mr, name):
    return Mr.get(name, (np.zeros(0, np.int64), np.zeros(0)))


def first_flag_time(Mr):
    ts = [series(Mr, n)[0][series(Mr, n)[1] > 0] for n in ("collision_any", "offroad", "left_corridor_laterally")]
    ts = [t[0] for t in ts if len(t)]
    return min(ts) if ts else None


def event_of(summ, Mr):
    m = summ.get("metrics") or {}
    first = {}
    for flag, ser, kind in FLAGS:
        if float(m.get(flag) or 0) > 0:
            t, v = series(Mr, ser)
            hit = t[v > 0]
            if len(hit):
                first[kind] = int(hit[0])
    if not first:
        return "other", None
    kind = min(first, key=lambda q: first[q])
    return kind, first[kind]


def pos_aligned_vlog(Mr):
    """t -> the logged car's speed at the point the ego has reached (dist_traveled_m against gt_dist_traveled_m), and past-end test."""
    te, de = series(Mr, "dist_traveled_m")
    tg, dg = series(Mr, "gt_dist_traveled_m")
    if len(tg) < 3 or len(te) < 2:
        return None
    dg = np.maximum.accumulate(dg)
    vg = np.diff(dg) / (np.diff(tg) / 1e6)
    tm = 0.5 * (tg[1:] + tg[:-1])
    dgu, iu = np.unique(dg, return_index=True)

    def f(t):
        d = np.interp(t, te, de)
        past = d > dg[-1] - 1.0
        tq = np.interp(d, dgu, tg[iu])
        return np.interp(tq, tm, vg), past, d
    return f


def rollout(track, lab, sess, pl, C, Mr, summ):
    """Per-row table, per-plan table and the event record of one rollout."""
    t = C.timestamp_us.to_numpy(np.int64)
    x, y, psi = C.x.to_numpy(), C.y.to_numpy(), C.yaw.to_numpy()
    vx, vy, wz = C.vx.to_numpy(), C.vy.to_numpy(), C.wz.to_numpy()
    u, dl = C.u_steering_angle.to_numpy(), C.front_steering_angle.to_numpy()
    n = len(t)
    j = np.searchsorted(pl.now, t, side="right") - 1
    A_path = np.full(n, np.nan)
    A_req = np.full(n, np.nan)
    v_ref = np.full(n, np.nan)
    for i in range(n):
        if j[i] < 0:
            continue
        tau = (t[i] - pl.t0[j[i]]) / 1e6
        k, vr = pl.kappa(j[i], tau + 1.0, tau + 2.0)
        v_ref[i] = vr
        A_path[i] = abs(k) * vr ** 2 if np.isfinite(k) else np.nan
        q = pl.at(j[i], np.array([tau + 1.5]))[0] - [x[i], y[i]]
        c, s = np.cos(psi[i]), np.sin(psi[i])
        ex, ey = c * q[0] + s * q[1], -s * q[0] + c * q[1]
        dd = ex * ex + ey * ey
        A_req[i] = abs(2 * ey / dd) * max(vx[i], 1.0) ** 2 if dd >= 1.0 else np.nan
    A_cmd = vx ** 2 * np.abs(np.tan(u)) / WB
    A_exec = np.abs(vx * wz)
    ddl = np.r_[np.nan, np.abs(np.diff(dl)) / (np.diff(t) / 1e6)]
    R = pd.DataFrame(dict(track=track, label=lab, session=sess, t=t, v=vx, vy=vy, wz=wz, u=u, delta=dl, ddelta=ddl, A_path=A_path,
                          A_req=A_req, A_cmd=A_cmd, A_exec=A_exec, plan=j))
    # per plan: curvature-rate demand, offsets, model / served arcs, lead cut
    m = len(pl.now)
    J = np.full(m, np.nan)
    for q in range(1, m):
        tq = (pl.now[q] - pl.t0[q]) / 1e6
        k1, vr = pl.kappa(q, tq + 1.0, tq + 2.0)
        tp = (pl.now[q] - pl.t0[q - 1]) / 1e6
        k0, _ = pl.kappa(q - 1, tp + 1.0, tp + 2.0)
        if np.isfinite(k1) and np.isfinite(k0):
            J[q] = abs(k1 - k0) / ((pl.now[q] - pl.now[q - 1]) / 1e6) * vr ** 2
    lead_src = [((f or {}).get("lead") or {}).get("src") for f in pl.fix]
    lead_cut = [float(((f or {}).get("lead") or {}).get("cut") or 0) for f in pl.fix]
    iv = np.clip(np.searchsorted(t, pl.now), 0, n - 1)
    Pt = pd.DataFrame(dict(track=track, label=lab, session=sess, now=pl.now, J_plan=J, y1=pl.poses[:, 1, 1], y2=pl.poses[:, 3, 1],
                           h2=pl.poses[:, 3, 2], arc1_model=[arc(P, 2) for P in pl.model], arc2_model=[arc(P, 4) for P in pl.model],
                           arc2_served=[arc(P, 4) for P in pl.poses], lead_src=lead_src, lead_cut=lead_cut, v_ego=vx[iv]))
    kind, t_ev = event_of(summ, Mr) if float(summ.get("score") or 0) == 0 else ("", None)
    m_ = summ.get("metrics") or {}
    clean = float(summ.get("score") or 0) > 0 and all(float(m_.get(f) or 0) == 0 for f, _, _ in FLAGS)
    return R, Pt, dict(track=track, label=lab, session=sess, scene=summ["clipgt_id"], score=float(summ.get("score") or 0), kind=kind,
                       t_ev=t_ev, clean=clean, t_flag=first_flag_time(Mr))


# ---------------------------------------------------------------- reads
def runs(mask):
    """Start indices and lengths of runs of True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(s, e - s))


def onset_read(ev, R, Pt, th):
    """Section 3.5 on one zero event."""
    r = R[(R.t >= ev["t_ev"] - 3_000_000) & (R.t <= ev["t_ev"])].reset_index(drop=True)
    if not len(r):
        return None
    b = bucket(r.v.to_numpy())
    thp = np.array([th[(ev["track"], q)]["A_path"] for q in b])
    thc = np.array([th[(ev["track"], q)]["A_cmd"] for q in b])
    ip = np.flatnonzero(r.A_path.to_numpy() > thp)
    ic = np.flatnonzero(r.A_cmd.to_numpy() > thc)
    op = r.t[ip[0]] if len(ip) else None
    oc = r.t[ic[0]] if len(ic) else None
    first = min([i for i in (ip[:1].tolist() + ic[:1].tolist())], default=None)
    v_on = float(r.v[first]) if first is not None else float(np.interp(ev["t_ev"] - 1_000_000, R.t, R.v))
    if op is not None and (oc is None or op <= oc):
        verdict = "plan_first"
    elif oc is not None and (op is None or oc < op - 200_000):
        i = ic[0]
        verdict = "steer_first_amp" if r.A_cmd[i] > 1.5 * (r.A_req[i] if np.isfinite(r.A_req[i]) else np.inf) else "steer_first_ref"
    elif oc is not None:
        verdict = "plan_first"                                   # steering ahead by < 0.2 s: within the MPC's own response time
    else:
        verdict = "quiet"
    ipk = int(np.nanargmax(r.A_cmd.to_numpy()))
    amp = float(r.A_cmd[ipk] / r.A_req[ipk]) if np.isfinite(r.A_req[ipk]) and r.A_req[ipk] > 0 else np.nan
    spin = any(L >= 3 for s, L in runs(r.A_exec.to_numpy() >= 6.0)) or bool(((np.abs(r.vy / np.maximum(r.v, 1e-3)) >= 0.2) & (r.v >= 3)).any())
    # loop signatures on plans at 0.5 s spacing in [t_ev - 4 s, t_ev]
    p = Pt[(Pt.now >= ev["t_ev"] - 4_000_000) & (Pt.now <= ev["t_ev"])]
    if ev["track"] == "pai" and len(p):
        g = np.round((p.now - p.now.iloc[0]) / 100_000).astype(int)
        p = p[g % 5 == 0]
    O = R_ = R2 = False
    if len(p) >= 3:
        y2 = p.y2.to_numpy()
        k = y2[np.abs(y2) >= 0.3]
        if len(k) >= 3 and (np.diff(np.sign(k)) != 0).sum() >= 2:
            peaks = [np.abs(k[s:s + L]).max() for s, L in runs_sign(k)]
            O = peaks[-1] >= 1.3 * peaks[0]
        wr = np.array([R.wz[(R.t > tn - 500_000) & (R.t <= tn)].mean() for tn in p.now])
        h2 = p.h2.to_numpy()
        okk = (np.sign(h2[1:]) == np.sign(wr[1:])) & (np.abs(h2[1:]) >= 1.2 * np.abs(h2[:-1])) & (np.abs(wr[1:]) > np.abs(wr[:-1]))
        R_ = any(L >= 2 for s, L in runs(okk))
        R2 = any(L >= 2 and np.abs(h2[s + L]) >= np.radians(2) for s, L in runs(okk))
    return dict(onset_plan=op, onset_steer=oc, verdict=verdict, v_on=v_on, bucket=str(bucket(np.array([v_on]))[0]), amp=amp, spin=spin,
                loop_O=bool(O), loop_R=bool(R_), loop_R2deg=bool(R2), A_path_max=float(np.nanmax(r.A_path)) if r.A_path.notna().any() else np.nan,
                A_cmd_max=float(r.A_cmd.max()), A_exec_max=float(r.A_exec.max()), ddelta_max=float(np.nanmax(r.ddelta)),
                v_ev=float(np.interp(ev["t_ev"], R.t, R.v)))


def runs_sign(k):
    s = np.sign(k)
    out, i = [], 0
    while i < len(s):
        L = 1
        while i + L < len(s) and s[i + L] == s[i]:
            L += 1
        out.append((i, L))
        i += L
    return out


def boot_slope(df, xcol, ycol, ccol, n=1000):
    rng = np.random.default_rng(0)
    cl = df[ccol].unique()
    G = {c: g for c, g in df.groupby(ccol)}

    def slope(d):
        x, y = d[xcol].to_numpy(), d[ycol].to_numpy()
        return float(np.polyfit(x, y, 1)[0]) if len(x) > 2 and np.ptp(x) > 0 else np.nan
    b = [slope(pd.concat([G[c] for c in rng.choice(cl, len(cl))])) for _ in range(n)]
    return slope(df), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "tmp/steer_long"))
    ap.add_argument("--out", default=str(REPO / "experiments/steer_long/results"))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RR, PP, EE, gates, SL, ST, CAP = [], [], [], {}, [], [], []
    plans_by = {}
    for track, labs in LABELS.items():
        for lab in labs:
            D, C, M, S = load(track, lab, root)
            g = dict(summary=len(S), with_ctrl=0, with_drive=0, frame_pairs=0, frame_ok=0, rec_pairs=0, rec_ok=0)
            for rid, summ in S.items():
                if rid not in C or rid not in D:
                    continue
                g["with_ctrl"] += 1
                g["with_drive"] += 1
                pl = Plans(D[rid])
                c = C[rid]
                plans_by[(track, lab, rid)] = pl
                # G-frame: CSV pose at t0 vs anchor
                ct = c.timestamp_us.to_numpy(np.int64)
                for q, t0 in enumerate(pl.t0):
                    i = np.searchsorted(ct, t0)
                    if i < len(ct) and ct[i] == t0:
                        g["frame_pairs"] += 1
                        g["frame_ok"] += np.hypot(c.x[i] - pl.anchor[q, 0], c.y[i] - pl.anchor[q, 1]) <= 0.05
                # G-rec: x_ref_0 / y_ref_0 of row T = the reference at the solve time T - 0.1 s in the rig frame then
                for i in range(1, len(ct)):
                    ts = ct[i - 1]
                    jj = np.searchsorted(pl.now, ts, side="right") - 1
                    if jj < 0:
                        continue
                    # x_ref_0 / y_ref_0 are pose 0 of the reference (system.py `ref_in_rig.get_pose(0)`), i.e. the first returned point:
                    # the plan point at the first 0.1 s grid time >= now on the plan clock, not the interpolation at the solve time
                    tau0 = np.ceil(((pl.now[jj] - pl.t0[jj]) / 1e6) / 0.1 - 1e-9) * 0.1
                    pr = pl.at(jj, np.array([tau0]))[0] - [c.x[i - 1], c.y[i - 1]]
                    cc, ss = np.cos(c.yaw[i - 1]), np.sin(c.yaw[i - 1])
                    rx, ry = cc * pr[0] + ss * pr[1], -ss * pr[0] + cc * pr[1]
                    g["rec_pairs"] += 1
                    g["rec_ok"] += np.hypot(rx - c.x_ref_0[i], ry - c.y_ref_0[i]) <= 0.05
                Mr = M.get(rid, {})
                R, Pt, ev = rollout(track, lab, rid, pl, c, Mr, summ)
                RR.append(R)
                PP.append(Pt)
                EE.append(ev)
                # slowness rows at the metric times (section 3.6)
                f = pos_aligned_vlog(Mr)
                if f is None:
                    continue
                tm = series(Mr, "dist_traveled_m")[0]
                tm = tm[(tm >= ct[0]) & (tm <= ct[-1])]
                tf = ev["t_flag"]
                if tf is not None:
                    tm = tm[tm < tf]
                if not len(tm):
                    continue
                vl, past, de = f(tm)
                ve = np.interp(tm, ct, c.vx)
                stopped = c.vx.to_numpy() < 0.5
                last_stop = np.array([ct[:np.searchsorted(ct, x, side="right")][stopped[:np.searchsorted(ct, x, side="right")]].max(initial=ct[0])
                                      for x in tm])
                jp = np.searchsorted(pl.now, tm, side="right") - 1
                vplan = np.where(jp >= 0, Pt.arc1_model.to_numpy()[np.maximum(jp, 0)], np.nan)
                SL.append(pd.DataFrame(dict(track=track, label=lab, scene=summ["clipgt_id"], log=log_of(track, summ["clipgt_id"]), t=tm,
                                            v_e=ve, v_log=vl, past=past, d_e=de, since_stop=(tm - last_stop) / 1e6, v_plan=vplan)))
                # stop episodes
                obs_t, obs_v = series(Mr, "min_distance_to_obstacle_m")
                for s0, L in runs(c.vx.to_numpy() < 1.0):
                    if L < 10:
                        continue
                    ts = ct[s0]
                    if tf is not None and ts >= tf:
                        continue
                    vl0, past0, _ = f(np.array([ts]))
                    obs = float(np.interp(ts, obs_t, obs_v)) if len(obs_t) else np.nan
                    jj = np.searchsorted(pl.now, ts, side="right") - 1
                    jj = max(jj, 0)
                    a2m, a2s = Pt.arc2_model[jj], Pt.arc2_served[jj]
                    src, cut = Pt.lead_src[jj], Pt.lead_cut[jj]
                    if a2m < 1.0:
                        cls = "P"
                    elif track == "pai" and src == "mpc" and cut >= 0.5:
                        cls = "L"
                    elif track == "pai" and a2s < 1.0:
                        cls = "V"
                    elif a2s >= 1.0:
                        cls = "M"
                    else:
                        cls = "other"
                    ST.append(dict(track=track, label=lab, scene=summ["clipgt_id"], session=rid, t=int(ts), dur=L / 10, v_log=float(vl0[0]),
                                   past=bool(past0[0]), obs=obs, at_start=bool(s0 == 0 or ts - ct[0] < 500_000), cls=cls, arc2_model=a2m,
                                   arc2_served=a2s, lead_src=src, lead_cut=cut,
                                   unexplained=bool(vl0[0] >= 2.0 and not past0[0] and (np.isnan(obs) or obs >= 10.0))))
            gates[(track, lab)] = g
            print(track, lab, g, flush=True)
    R = pd.concat(RR, ignore_index=True)
    P = pd.concat(PP, ignore_index=True)
    E = pd.DataFrame(EE)
    SLd = pd.concat(SL, ignore_index=True)
    STd = pd.DataFrame(ST)
    # thresholds on clean rollouts (section 3.4)
    clean = set(E.session[E.clean])
    Rc = R[R.session.isin(clean) & (R.v >= 1.0)].copy()
    Rc["bucket"] = bucket(Rc.v.to_numpy())
    th = {(tr, b): dict(A_path=float(np.nanpercentile(g.A_path, 99)), A_cmd=float(np.nanpercentile(g.A_cmd, 99)),
                        A_exec=float(np.nanpercentile(g.A_exec, 99)), A_req=float(np.nanpercentile(g.A_req.dropna(), 99)),
                        ddelta=float(np.nanpercentile(g.ddelta.dropna(), 99)), n=len(g))
          for (tr, b), g in Rc.groupby(["track", "bucket"])}
    # events
    ER = []
    for ev in E[E.kind.isin(["collision", "offroad", "corridor"])].to_dict("records"):
        Rs = R[R.session == ev["session"]]
        Ps = P[P.session == ev["session"]]
        o = onset_read(ev, Rs, Ps, th)
        if o:
            ER.append({**ev, **o})
    ER = pd.DataFrame(ER)
    # cap counterfactual (section 3.7)
    for am in CAPS_A:
        for jm in CAPS_J:
            dev_all = []
            for (tr, lab, rid), pl in plans_by.items():
                dev, y2c = cap_session(pl, am, jm)
                dev_all.append(pd.DataFrame(dict(track=tr, label=lab, session=rid, now=pl.now, dev=dev, y2c=y2c, y2=pl.poses[:, 3, 1])))
            Dv = pd.concat(dev_all, ignore_index=True)
            Dv = Dv.merge(P[["session", "now", "v_ego"]], on=["session", "now"], how="left")
            for tr in ("nuplan", "pai"):
                dc = Dv[(Dv.track == tr) & Dv.session.isin(clean)]
                row = dict(track=tr, a_max=am, j_max=jm, coll_clean=float((dc.dev > 0.10).mean()),
                           coll_clean_high=float((dc.dev[dc.v_ego >= 15] > 0.10).mean()) if (dc.v_ego >= 15).any() else np.nan)
                for vb in ("low", "mid", "high"):
                    row[f"coll_clean_{vb}"] = float((dc.dev[bucket(dc.v_ego.to_numpy()) == vb] > 0.10).mean())
                ee = ER[ER.track == tr]
                cov, cov_hpf, n_hpf = 0, 0, 0
                for e in ee.to_dict("records"):
                    dv = Dv[(Dv.session == e["session"]) & (Dv.now >= e["t_ev"] - 3_000_000) & (Dv.now <= e["t_ev"])]
                    mod = dv[dv.dev > 0.10]
                    ok = False
                    if len(mod):
                        f0 = mod.iloc[0]
                        lim = e["onset_plan"] if e["onset_plan"] is not None and not pd.isna(e["onset_plan"]) else e["t_ev"]
                        ok = bool(f0.now <= lim and abs(f0.y2c) <= 0.5 * abs(f0.y2))
                    cov += ok
                    if e["verdict"] == "plan_first" and e["bucket"] == "high":
                        n_hpf += 1
                        cov_hpf += ok
                row.update(events=len(ee), cover=cov, high_plan_first=n_hpf, cover_high_plan_first=cov_hpf)
                CAP.append(row)
            print("cap", am, jm, flush=True)
    CAPd = pd.DataFrame(CAP)
    # write
    out.mkdir(parents=True, exist_ok=True)
    E.to_csv(out / "rollouts.csv", index=False)
    ER.to_csv(out / "events.csv", index=False)
    STd.to_csv(out / "stops.csv", index=False)
    CAPd.to_csv(out / "cap_grid.csv", index=False)
    SLd.to_parquet(out / "speed_rows.parquet", index=False)
    P.to_parquet(out / "plans.parquet", index=False)
    R.to_parquet(REPO / "tmp/steer_long/rows.parquet", index=False)
    json.dump(dict(gates={f"{k[0]}/{k[1]}": v for k, v in gates.items()},
                   thresholds={f"{k[0]}/{k[1]}": v for k, v in th.items()}), open(out / "gates_thresholds.json", "w"), indent=1, default=float)
    print("done", len(E), len(ER), len(STd), flush=True)


if __name__ == "__main__":
    main()
