#!/usr/bin/env python3
"""PROGRESS reading (plans/2026-10-15-progress-lane-readings-prereg.md, reading A): are RSG-L18's over-turn / shortcut turn zeros the ones
whose served route memory had no near route (0-20 m) shortly before the failure? CPU only; stored reads are read, never written.

  rmem    (Tokyo host / box, python3 + numpy only)
          rf_progress.py rmem --runs <run dir or PAI run root> ... --out <pkl>
          per rollout (driver session) of the read: decision t0 / k / anchor, the served memory's valid mask (40) and points (40 x 2, the
          decision's rear-axle frame), the served command. PAI roots: <root>/runs/*/{driver/drive.jsonl, sim/aggregate}; held runs:
          <run>/{driver-logs/drive.jsonl, aggregate}.
  report  (Mac, .venv; DATA_DIR for the Run dir)
          rf_progress.py report --set NAME=<rf_tax pkl>,<rf_tax csv>,<rmem pkl>,<follow scenes.csv>,<follow label> ... --out DIR --fig PNG
          -> DIR/progress.json, progress.md, zeros.csv (one row per turn zero), completed.csv (completed strict turns); the BEV figure.

Definitions (the prereg's, copied): ego = controller trace (rear axle, 10 Hz); route = rf_tax's rebuilt AlpaSim route (MAP, 40 m); ego
station / signed lateral (left +) on it with rf_follow's local window. t_f = the first zero-score event (rf_tax's rule). Memory at t = the rmem
of the last decision with t0 <= t (decision 0 if none). cov_0_20 / cov_20_40 = share of valid offsets in 0, 2, .., 18 / 20, .., 38 m; first
valid point = smallest valid offset >= 0 (offset, lateral, bearing). Turn zero = taxonomy turn class, or the route has a turn at the failure
(rf_pai_follow zero topology: first strict 60 deg / 30 m turn, else bend 45 deg / 60 m, with exit + 10 > s_reach and entry <= s_reach + 40).
Class = taxonomy mapping, else rf_tax's corridor heading rule at t_f. Shortcut flag: at t_f the ego heading turned >= 20 deg off the route
tangent toward the centroid of the far segment (route stations s_f + 40 .. s_f + 80) and the log at the same driven arc length did not.
Target T = over-turn class or shortcut. Covered = cov_0_20 >= 0.5 at t_f - 3 s. Claim A: RR = P(T | uncovered) / P(T | covered) >= 2, scene
bootstrap CI lower > 1 (unsupported: RR < 1 or CI upper < 2; inconclusive otherwise or a bucket < 5).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R_ = HERE.parents[2]
sys.path[:0] = [str(R_), str(HERE)]
N_PT, NF = 40, 5
OFFS = -6.0 + 2.0 * np.arange(N_PT)
NEAR, MID = (OFFS >= 0) & (OFFS < 20), (OFFS >= 20) & (OFFS < 40)
LAT_OK, DEV_DEG, N_BOOT = 2.5, 20.0, 10_000
TURN_CLS = {"cor_straight_at_turn": "straight", "off_junc_straight": "straight", "cor_turn_under": "late_wide", "off_junc_outside": "late_wide",
            "off_curve_outside": "late_wide", "cor_turn_over": "over_turn", "off_junc_inside": "over_turn", "off_curve_inside": "over_turn",
            "cor_wrong_exit": "wrong_exit"}
DROP = {"infra", "off_at_start", "cor_route_end"}


# ---------------------------------------------------------------- rmem (host python3 + numpy)
def rmem_extract(a):
    out = {}
    for root in a.runs:
        pai = bool(glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")))
        pairs = ([(f, os.path.join(os.path.dirname(os.path.dirname(f)), "sim")) for f in sorted(glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")))]
                 if pai else [(os.path.join(root, "driver-logs", "drive.jsonl"), root)])
        for f, sim in pairs:
            sf = os.path.join(sim, "aggregate", "results-summary.json")
            if not os.path.exists(sf):
                continue
            want = {r["rollout_id"]: r["clipgt_id"] for r in json.load(open(sf))["rollouts"]}
            rec = {}
            for line in open(f):
                if '"kind": "drive"' not in line:
                    continue
                r = json.loads(line)
                if r["session"] not in want or "rmem" not in r or r.get("infer") is False:
                    continue
                rec.setdefault(r["session"], []).append((int(r["t0"]), int(r["k"]), r["anchor"], r["rmem"], r.get("cmd")))
            for sid, rr in rec.items():
                rr.sort(key=lambda x: x[0])
                t0 = np.array([x[0] for x in rr], np.int64)
                keep = np.r_[True, np.diff(t0) > 0]                         # one record per decision time
                rr = [x for x, k in zip(rr, keep) if k]
                F = np.asarray([x[3] for x in rr], np.float32).reshape(-1, N_PT, NF)
                out[sid] = dict(scene=want[sid], t0=np.array([x[0] for x in rr], np.int64), k=np.array([x[1] for x in rr], np.int32),
                                anchor=np.asarray([x[2] for x in rr], np.float64), V=F[..., 0] > 0.5, XY=F[..., 1:3] * 20.0,
                                cmd=np.array([-1 if x[4] is None else int(x[4]) for x in rr], np.int8))
        print(root, len(out), "sessions so far", flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "sessions ->", a.out)


# ---------------------------------------------------------------- shared geometry (report side)
def wrap(x):
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi


def to_local(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.c_[a[0] + c * X[:, 0] - s * X[:, 1], a[1] + s * X[:, 0] + c * X[:, 1]]


def _imports():
    import rf_pai_follow as PF
    import rf_tax as TX
    return PF, TX


def load_set(spec):
    """NAME=<tax pkl>,<tax csv>,<rmem pkl>,<follow csv>,<follow label> -> (name, rollouts by rid, tax rows by scene, rmem, follow rows by scene)."""
    import pandas as pd
    name, rest = spec.split("=", 1)
    fp, fc, fr, ff, fl = rest.split(",")
    rolls = {o["rid"]: o for o in pickle.load(open(fp, "rb"))["rolls"]}
    tax = pd.read_csv(fc).set_index("scene")
    rm = pickle.load(open(fr, "rb"))
    fo = pd.read_csv(ff)
    fo = fo[fo.label == fl].set_index("scene")
    return name, rolls, tax, rm, fo


class Ego:
    """One rollout: route / log geometry, the ego track on the route, the memory at any time."""

    def __init__(self, o, rm, PF, TX):
        self.o, self.PF, self.TX = o, PF, TX
        self.r = TX.Roll(o)
        C = o["ctrl"]
        self.t, self.P, self.yaw = C["t"], np.c_[C["x"], C["y"]], C["yaw"]
        self.v = np.hypot(C["vx"], C["vy"])
        self.R = PF.dedup(np.asarray(o["route"], float))
        self.S = PF.stations(self.R)
        self.st, self.lat = PF.track(self.R, self.S, self.P)
        self.sh, self.hh = PF.heading(self.R, self.S)
        self.arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.P, axis=0).T))]
        self.rm = rm
        L = o["log"]
        self.LP, self.LS = self.r.LP, self.r.LS
        self.LY = np.unwrap(np.interp(self.r.LT, L["T"], L["Y"]))
        self.ls0 = float(TX.poly_proj(self.LP, self.P[:1])[1][0])

    def i_at(self, t):
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))

    def mem(self, t):
        """Memory of the last decision with t0 <= t -> (index, flagged before the first decision)."""
        if self.rm is None or not len(self.rm["t0"]):
            return None, True
        j = int(np.searchsorted(self.rm["t0"], t, side="right") - 1)
        return max(j, 0), j < 0

    def cov(self, t):
        j, pre = self.mem(t)
        if j is None:
            return dict(cov_0_20=np.nan, cov_20_40=np.nan, first_off=np.nan, first_lat=np.nan, first_brg=np.nan, first10_off=np.nan, first10_lat=np.nan,
                        first10_brg=np.nan, mem_pre=True, k_mem=-1)
        V, XY = self.rm["V"][j], self.rm["XY"][j]
        f = np.flatnonzero(V & (OFFS >= 0))
        fo = dict(first_off=float(OFFS[f[0]]), first_lat=float(XY[f[0], 1]), first_brg=float(np.degrees(np.arctan2(XY[f[0], 1], XY[f[0], 0])))) if len(f) \
            else dict(first_off=np.nan, first_lat=np.nan, first_brg=np.nan)
        f = np.flatnonzero(V & (OFFS >= 10))                                # Amendment 1: a point at 0 m sits beside the axle (bearing ~ +-90)
        fo |= dict(first10_off=float(OFFS[f[0]]), first10_lat=float(XY[f[0], 1]), first10_brg=float(np.degrees(np.arctan2(XY[f[0], 1], XY[f[0], 0])))) if len(f) \
            else dict(first10_off=np.nan, first10_lat=np.nan, first10_brg=np.nan)
        return dict(cov_0_20=float(V[NEAR].mean()), cov_20_40=float(V[MID].mean()), mem_pre=bool(pre), k_mem=int(self.rm["k"][j]), **fo)

    def geo_cov(self, t):
        """Amendment 2: route stations the memory covers, by geometry: the valid points projected on the rebuilt route span [lo, hi];
        share of the stations s_e + 0, 2, .., 18 (and 20, .., 38) inside it (s_e = the ego's station at t); nearest covered distance."""
        X, _ = self.mem_poly(t)
        s_e = float(self.st[self.i_at(t)])
        if not len(X):
            return dict(gcov_0_20=0.0, gcov_20_40=0.0, gnear=np.nan, s_e=s_e)
        sv = np.array([self.PF.project(self.R, self.S, q, s_e - 30.0, s_e + 120.0)[0] for q in X])
        lo, hi = sv.min() - 1.0, sv.max() + 1.0
        q = s_e + OFFS
        inn = (q >= lo) & (q <= hi)
        return dict(gcov_0_20=float(inn[NEAR].mean()), gcov_20_40=float(inn[MID].mean()), gnear=float(max(lo + 1.0 - s_e, 0.0)), s_e=s_e,
                    g_lo=float(lo), g_hi=float(hi))

    def mem_poly(self, t):
        """Valid memory points of the decision at t in the local frame (n, 2), with their offsets."""
        j, _ = self.mem(t)
        if j is None:
            return np.zeros((0, 2)), np.zeros(0)
        V = self.rm["V"][j]
        return to_local(self.rm["anchor"][j], self.rm["XY"][j][V]), OFFS[V]

    def route_xy(self, s):
        s = np.clip(s, 0, self.S[-1])
        return np.c_[np.interp(s, self.S, self.R[:, 0]), np.interp(s, self.S, self.R[:, 1])]

    def route_psi(self, s):
        return float(np.interp(np.clip(s, 0, self.S[-1]), self.sh, self.hh))

    def log_pose_at_arc(self, arc):
        """The log's pose at log station = the ego's start station on the log + arc."""
        s = float(np.clip(self.ls0 + arc, 0, self.LS[-1]))
        n = len(self.r.LT)
        return np.array([np.interp(s, self.LS, self.LP[:, 0]), np.interp(s, self.LS, self.LP[:, 1]), np.interp(s, self.LS[:n], self.LY)])

    def t_fail(self, zero_type):
        TX, ser = self.TX, self.o["ser"]
        if zero_type == "collision":
            return float(np.nanmin([TX.first_pos(ser, "collision_front"), TX.first_pos(ser, "collision_lateral")]))
        return float(TX.first_pos(ser, {"offroad": "offroad", "corridor": "left_corridor_laterally"}[zero_type]))

    def s_reach(self, tf):
        m = (self.t < tf) & (np.abs(self.lat) <= LAT_OK)
        return float(self.st[m][-1]) if m.any() else float(self.st[0])

    def near_turn(self, sr):
        """rf_pai_follow zero topology: first route turn (strict, else bend) with exit + 10 > s_reach and entry <= s_reach + 40."""
        PF = self.PF
        for kw, d in (({}, "strict"), (dict(win=PF.BEND_WIN, deg=PF.BEND_DEG), "bend")):
            for e, x, th in PF.all_turns(self.sh, self.hh, 0.0, self.S[-1], **kw):
                if x + PF.PAST > sr and e <= sr + PF.NEAR:
                    return dict(def_=d, entry=e, exit=x, deg=float(np.degrees(th)))
        return None

    def heading_rule(self, tf):
        """rf_tax's corridor rule applied at tf -> class name, log dpsi, ego dpsi."""
        r = self.r
        a = r.pose(tf)
        s_ev = float(r.log_station(a[None, :2])[0])
        tb = max(tf - 3e6, r.t[0])
        s_b = float(r.log_station(r.pose(tb)[None, :2])[0])
        s_hi = min(s_ev + 15, r.LS[-1])
        hl = lambda s: float(np.interp(s, r.LS[:len(r.LT)], np.unwrap(np.interp(r.LT, self.o["log"]["T"], self.o["log"]["Y"]))))  # noqa: E731
        dl = float(np.degrees(hl(s_hi) - hl(s_b)))
        de = float(np.degrees(r.pose(tf)[2] - r.pose(tb)[2]))
        W = np.c_[np.interp(np.linspace(s_b, s_hi, 20), r.LS, r.LP[:, 0]), np.interp(np.linspace(s_b, s_hi, 20), r.LS, r.LP[:, 1])]
        ch = W[-1] - W[0]
        nrm = np.array([-ch[1], ch[0]]) / max(np.hypot(*ch), 1e-6)
        jog = float(np.ptp((W - W[0]) @ nrm))
        if s_ev >= r.LS[-1] - 2.0:
            c = "route_end"
        elif abs(dl) >= 30 and (abs(de) < 10 or np.sign(de) != np.sign(dl)):
            c = "straight"
        elif abs(dl) >= 30:
            c = "late_wide" if abs(de) < abs(dl) else "over_turn"
        elif abs(de) >= 30:
            c = "wrong_exit"
        elif jog >= 2.5:
            c = "lane_change"
        else:
            c = "other"
        return c, dl, de

    def shortcut(self, tf):
        """Prereg flag at tf -> (flag or None when the far segment is < 10 m, ego deviation deg, log deviation deg)."""
        i = self.i_at(tf)
        sf = float(self.st[i])
        lo, hi = sf + 40.0, min(sf + 80.0, self.S[-1])
        if hi - lo < 10.0:
            return None, np.nan, np.nan
        cF = self.route_xy(np.linspace(lo, hi, 21)).mean(0)

        def dev(p, psi, s):
            pr = self.route_psi(s)
            d = float(wrap(psi - pr))
            b = float(wrap(np.arctan2(cF[1] - p[1], cF[0] - p[0]) - pr))
            return abs(np.degrees(d)) >= DEV_DEG and np.sign(d) == np.sign(b), float(np.degrees(d)) * (1 if np.sign(d) == np.sign(b) else -1)
        e_dev, e_deg = dev(self.P[i], self.yaw[i], sf)
        L = self.log_pose_at_arc(self.arc[i])
        sL, _ = self.PF.project(self.R, self.S, L[:2], sf - 30.0, sf + 60.0)
        l_dev, l_deg = dev(L[:2], L[2], sL)
        return bool(e_dev and not l_dev), e_deg, l_deg

    def legs(self, t):
        """At time t: far segment on another leg (|psi(s + 60) - psi(s)| >= 45), two opposite turns in [s, s + 80], a turn entry in [s, s + 40]."""
        PF = self.PF
        s = float(self.st[self.i_at(t)])
        dpsi = float(np.degrees(np.interp(min(s + 60, self.S[-1]), self.sh, self.hh) - np.interp(s, self.sh, self.hh)))
        T = PF.all_turns(self.sh, self.hh, max(s - 10, 0), min(s + 80, self.S[-1]), win=30.0, deg=30.0)
        sg = {np.sign(th) for _, _, th in T}
        return dict(far_dpsi=dpsi, far_other_leg=abs(dpsi) >= 45, double_opp=len(sg) == 2,
                    gap_turn=any(s <= e <= s + 40 for e, _, _ in PF.all_turns(self.sh, self.hh, max(s - 10, 0), min(s + 80, self.S[-1]))))

    def cross_t(self, s, tf=np.inf):
        """First controller time whose route station >= s before tf (None if the ego starts past it or never reaches it)."""
        if self.st[0] >= s:
            return None, "started_past"
        j = np.flatnonzero((self.st >= s) & (self.t < tf))
        return (float(self.t[j[0]]), "ok") if len(j) else (None, "not_reached")


# ---------------------------------------------------------------- reading A
def zero_job(job):
    name, o, tax, rm, frow = job
    PF, TX = _imports()
    e = Ego(o, rm, PF, TX)
    zt = tax["zero_type"]
    tf = e.t_fail(zt)
    sr = e.s_reach(tf)
    nt = e.near_turn(sr)
    hr, dl, de = e.heading_rule(tf)
    prim = tax["primary"]
    cls = TURN_CLS.get(prim)
    turn_zero = cls is not None or nt is not None
    if cls is None:
        cls = "lane_change" if prim == "cor_lane_change" else hr
    sc, e_deg, l_deg = e.shortcut(tf)
    row = dict(set=name, scene=o["scene"], rid=o["rid"], score=o["score"], zero_type=zt, primary=prim, t_f=tf, t_f_s=(tf - e.t[0] + 1e5) / 1e6,
               t_ev_s_tax=tax.get("t_ev_s"), s_reach=sr, turn_zero=bool(turn_zero), near_turn=None if nt is None else nt["def_"],
               turn_entry=np.nan if nt is None else nt["entry"], turn_exit=np.nan if nt is None else nt["exit"], turn_deg=np.nan if nt is None else nt["deg"],
               cls=cls, rule_cls=hr, log_dpsi=dl, ego_dpsi=de, shortcut=sc, ego_dev_deg=e_deg, log_dev_deg=l_deg,
               v_f=float(np.interp(tf, e.t, e.v)), strict_outcome=frow.get("outcome") if frow is not None else None)
    for tag, dt in (("m3", 3e6), ("m1", 1e6)):
        t = max(tf - dt, e.t[0])
        row.update({f"{k}_{tag}": v for k, v in e.cov(t).items()})
        row.update({f"{k}_{tag}": v for k, v in e.geo_cov(t).items()})
        row[f"drv_{tag}"] = float(e.arc[e.i_at(t)])
        row[f"lat_{tag}"] = float(e.lat[e.i_at(t)])
        X, _ = e.mem_poly(t)                                                # gate G-A2: the served memory lies on the rebuilt route
        row[f"memdist_{tag}"] = float(np.median([abs(PF.project(e.R, e.S, q)[1]) for q in X])) if len(X) else np.nan
    row.update(e.legs(max(tf - 3e6, e.t[0])))
    row["T"] = bool(cls == "over_turn" or sc is True)
    return row


def done_job(job):
    """A completed strict turn: coverage when the ego first crosses the entry station, and 3 s before."""
    name, o, rm, frow = job
    PF, TX = _imports()
    e = Ego(o, rm, PF, TX)
    t, why = e.cross_t(float(frow["entry"]))
    row = dict(set=name, scene=o["scene"], rid=o["rid"], entry=float(frow["entry"]), side=frow.get("side"), reach=why)
    if t is None:
        return row
    for tag, dt in (("e0", 0.0), ("m3", 3e6)):
        tt = max(t - dt, e.t[0])
        row.update({f"{k}_{tag}": v for k, v in e.cov(tt).items()})
        row.update({f"{k}_{tag}": v for k, v in e.geo_cov(tt).items()})
        row[f"drv_{tag}"] = float(e.arc[e.i_at(tt)])
    return row


def rr_boot(T, U, g, n_boot=N_BOOT, seed=0):
    """RR = P(T | U) / P(T | ~U) with a cluster bootstrap over g -> dict(rr, lo, hi, dropped, inf share)."""
    T, U = np.asarray(T, float), np.asarray(U, bool)
    codes, uniq = np.unique(np.asarray(g), return_inverse=True)[1], None
    nG = codes.max() + 1
    tu, nu = np.bincount(codes, T * U, nG), np.bincount(codes, U.astype(float), nG)
    tc, nc = np.bincount(codes, T * ~U, nG), np.bincount(codes, (~U).astype(float), nG)

    def rr(a, b, c, d):
        pu, pc = (a / b if b else np.nan), (c / d if d else np.nan)
        if not np.isfinite(pu) or not np.isfinite(pc):
            return np.nan
        return np.inf if pc == 0 and pu > 0 else (np.nan if pc == 0 else pu / pc)
    point = rr(tu.sum(), nu.sum(), tc.sum(), nc.sum())
    idx = np.random.default_rng(seed).integers(nG, size=(n_boot, nG))
    a, b, c, d = tu[idx].sum(1), nu[idx].sum(1), tc[idx].sum(1), nc[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pu, pc = a / b, c / d
        r = np.where((pc == 0) & (pu > 0), np.inf, pu / pc)
    ok = (b > 0) & (d > 0) & ~((pc == 0) & (pu == 0))
    r = r[ok]
    lo, hi = (np.quantile(r, [0.025, 0.975]) if len(r) else (np.nan, np.nan))
    return dict(rr=float(point), lo=float(lo), hi=float(hi), dropped=int((~ok).sum()), inf_share=float(np.isinf(r).mean()) if len(r) else np.nan,
                n_unc=int(nu.sum()), n_cov=int(nc.sum()), t_unc=int(tu.sum()), t_cov=int(tc.sum()))


def diff_boot(x, grp, g, n_boot=N_BOOT, seed=0):
    """Mean(x | grp) - mean(x | ~grp), cluster bootstrap over g."""
    x, grp = np.asarray(x, float), np.asarray(grp, bool)
    codes = np.unique(np.asarray(g), return_inverse=True)[1]
    nG = codes.max() + 1
    s1, n1 = np.bincount(codes, x * grp, nG), np.bincount(codes, grp.astype(float), nG)
    s0, n0 = np.bincount(codes, x * ~grp, nG), np.bincount(codes, (~grp).astype(float), nG)
    idx = np.random.default_rng(seed).integers(nG, size=(n_boot, nG))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = s1[idx].sum(1) / n1[idx].sum(1) - s0[idx].sum(1) / n0[idx].sum(1)
    b = b[np.isfinite(b)]
    return dict(diff=float(s1.sum() / max(n1.sum(), 1) - s0.sum() / max(n0.sum(), 1)), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)),
                n1=int(n1.sum()), n0=int(n0.sum()))


def verdict_a(r):
    if min(r["n_unc"], r["n_cov"]) < 5:
        return "inconclusive (a bucket < 5)"
    if r["rr"] >= 2 and r["lo"] > 1:
        return "supported"
    if r["rr"] < 1 or r["hi"] < 2:
        return "unsupported"
    return "inconclusive"


def report(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "progress", config=vars(a)) as run:
        zj, dj, gates = [], [], {}
        for spec in a.set:
            name, rolls, tax, rm, fo = load_set(spec)
            miss = 0
            for rid, o in rolls.items():
                if "ctrl" not in o or o["scene"] not in tax.index:
                    continue
                t = tax.loc[o["scene"]]
                m = rm.get(rid)
                miss += m is None
                frow = fo.loc[o["scene"]].to_dict() if o["scene"] in fo.index else None
                if isinstance(t["zero_type"], str) and t["zero_type"] in ("collision", "offroad", "corridor") and t["primary"] not in DROP:
                    zj.append((name, o, t.to_dict(), m, frow))
                if frow is not None and bool(frow.get("turn")) and frow.get("outcome") == "completed":
                    dj.append((name, o, m, frow))
            gates[name] = dict(rollouts=len(rolls), rmem_missing=miss)
            run.info(f"{name}: {len(rolls)} rollouts, rmem missing {miss}")
        rz = par.pmap(zero_job, zj, workers=None, run=run, desc="zeros")
        rd = par.pmap(done_job, dj, workers=None, run=run, desc="completed")
        if rz.errors or rd.errors:
            run.info(f"errors: {list(rz.errors.values())[:2]} {list(rd.errors.values())[:2]}")
        Z = pd.DataFrame([v for v in rz.values if v is not None])
        D = pd.DataFrame([v for v in rd.values if v is not None])
        Z["cluster"] = Z.scene
        Z.to_csv(out / "zeros.csv", index=False)
        D.to_csv(out / "completed.csv", index=False)
        # gate G-A1: the heading rule reproduces the taxonomy's corridor classes
        c = Z[Z.zero_type == "corridor"]
        cm = c.primary.map(lambda p: {"cor_straight_at_turn": "straight", "cor_turn_under": "late_wide", "cor_turn_over": "over_turn",
                                      "cor_wrong_exit": "wrong_exit", "cor_lane_change": "lane_change", "cor_drift": "other"}.get(p))
        gates["G_A1"] = dict(n=int(len(c)), agree=int((cm == c.rule_cls).sum()), share=float((cm == c.rule_cls).mean()) if len(c) else np.nan)
        gates["G_A2_memdist_median_m"] = float(Z.memdist_m3.median())
        gates["G_A2_memdist_p90_m"] = float(Z.memdist_m3.quantile(0.9))
        gates["t_f_vs_tax_max_abs_s"] = float(np.nanmax(np.abs(Z.t_f_s - Z.t_ev_s_tax.astype(float)))) if len(Z) else np.nan
        TZ = Z[Z.turn_zero].copy()
        res = dict(gates=gates, sets={}, claim={})
        for tag in ("m3", "m1"):
            TZ[f"unc_{tag}"] = TZ[f"cov_0_20_{tag}"] < 0.5
            TZ[f"unc_g{tag}"] = TZ[f"gcov_0_20_{tag}"] < 0.5
        groups = {"pooled": TZ, **{n: TZ[TZ.set == n] for n in TZ.set.unique()}, "held": TZ[TZ.set.str.startswith("held")]}
        for gname, x in groups.items():
            r = {}
            for tag in ("m3", "m1", "gm3", "gm1"):
                rr = rr_boot(x["T"].to_numpy(), x[f"unc_{tag}"].to_numpy(), x.cluster.to_numpy())
                rr["verdict"] = verdict_a(rr)
                r[tag] = rr
                if tag.startswith("g"):
                    continue
                r[f"T_shortcut_only_{tag}"] = rr_boot(x.shortcut.fillna(False).astype(bool).to_numpy(), x[f"unc_{tag}"].to_numpy(), x.cluster.to_numpy())
                r[f"T_overturn_only_{tag}"] = rr_boot((x.cls == "over_turn").to_numpy(), x[f"unc_{tag}"].to_numpy(), x.cluster.to_numpy())
            r["counts_m3"] = {f"{k[0]}|{'uncovered' if k[1] else 'covered'}": int(v) for k, v in x.groupby(["cls", "unc_m3"]).size().items()}
            r["counts_gm3"] = {f"{k[0]}|{'uncovered' if k[1] else 'covered'}": int(v) for k, v in x.groupby(["cls", "unc_gm3"]).size().items()}
            r["counts_m1"] = {f"{k[0]}|{'uncovered' if k[1] else 'covered'}": int(v) for k, v in x.groupby(["cls", "unc_m1"]).size().items()}
            r["shortcut_m3"] = {f"{k[0]}|{'uncovered' if k[1] else 'covered'}": int(v) for k, v in x.groupby([x.shortcut.astype(str), "unc_m3"]).size().items()}
            r["n"] = int(len(x))
            res["sets"][gname] = r
        res["claim"] = dict(primary=res["sets"]["pooled"]["m3"], verdict=res["sets"]["pooled"]["m3"]["verdict"],
                            geometric=res["sets"]["pooled"]["gm3"], verdict_geometric=res["sets"]["pooled"]["gm3"]["verdict"])
        # completed vs turn zeros: P(uncovered)
        if len(D):
            D["unc_e0"], D["unc_m3"] = D.cov_0_20_e0 < 0.5, D.cov_0_20_m3 < 0.5
            D["gunc_e0"], D["gunc_m3"] = D.gcov_0_20_e0 < 0.5, D.gcov_0_20_m3 < 0.5
            Dk = D[D.reach == "ok"]
            both = pd.concat([pd.DataFrame(dict(u=Dk.unc_m3.astype(float), z=False, g=Dk.scene)), pd.DataFrame(dict(u=TZ.unc_m3.astype(float), z=True, g=TZ.scene))])
            res["completed"] = dict(n=int(len(D)), reached=int(len(Dk)), p_unc_entry=float(Dk.unc_e0.mean()) if len(Dk) else np.nan,
                                    p_unc_entry_m3=float(Dk.unc_m3.mean()) if len(Dk) else np.nan, p_unc_zero_m3=float(TZ.unc_m3.mean()),
                                    diff_zero_minus_completed=diff_boot(both.u, both.z, both.g),
                                    gp_unc_entry=float(Dk.gunc_e0.mean()) if len(Dk) else np.nan, gp_unc_entry_m3=float(Dk.gunc_m3.mean()) if len(Dk) else np.nan,
                                    gp_unc_zero_m3=float(TZ.unc_gm3.mean()),
                                    cov_0_20_entry_median=float(Dk.cov_0_20_e0.median()) if len(Dk) else np.nan,
                                    per_set={n: dict(n=int((Dk.set == n).sum()), p_unc_entry=float(Dk[Dk.set == n].unc_e0.mean()),
                                                     p_unc_m3=float(Dk[Dk.set == n].unc_m3.mean())) for n in Dk.set.unique()})
        res["by_class"] = {c_: {k: float(TZ[TZ.cls == c_][k].median()) for k in ("cov_0_20_m3", "cov_20_40_m3", "gcov_0_20_m3", "gcov_20_40_m3", "gnear_m3", "first_off_m3", "first_lat_m3", "first_brg_m3", "first10_off_m3", "first10_brg_m3", "drv_m3", "v_f")}
                           | dict(n=int((TZ.cls == c_).sum()), shortcut=int((TZ[TZ.cls == c_].shortcut == True).sum()))  # noqa: E712
                           for c_ in sorted(TZ.cls.unique())}
        res["legs"] = dict(far_other_leg=int(TZ.far_other_leg.sum()), double_opp=int(TZ.double_opp.sum()), gap_turn=int(TZ.gap_turn.sum()), n=int(len(TZ)),
                           T_far_other_leg=int((TZ["T"] & TZ.far_other_leg).sum()), T_n=int(TZ["T"].sum()))
        res["non_turn_zeros"] = int((~Z.turn_zero).sum())
        (out / "progress.json").write_text(json.dumps(res, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
        write_md(out, res, Z, TZ, D)
        if a.fig:
            fig_bev(a, zj, Z)
        run.summary.update(verdict=res["claim"]["verdict"], rr=res["claim"]["primary"]["rr"])
        print((out / "progress.md").read_text())


def write_md(out, res, Z, TZ, D):
    f = lambda r: f"{r['rr']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]" + (f" (inf {100 * r['inf_share']:.0f} %)" if r.get("inf_share") else "")  # noqa: E731
    g = res["gates"]
    md = ["# PROGRESS reading A: near-route coverage before the failure (rf_progress.py)", "",
          "Prereg `plans/2026-10-15-progress-lane-readings-prereg.md`. Turn zeros of RSG-L18 (PAI 281 s0, held 460 s0 / s1); covered = >= 50 % of the "
          "memory's 0-20 m offsets valid at t_f - 3 s (primary) / - 1 s. T = over-turn class or shortcut flag. RR = P(T | uncovered) / P(T | covered), "
          "scene-clustered bootstrap 95 % CI (B = 10 000).", "",
          f"Gates: G-A1 heading rule = taxonomy corridor class {g['G_A1']['agree']} / {g['G_A1']['n']}; G-A2 served memory points to the rebuilt route "
          f"median {g['G_A2_memdist_median_m']:.2f} m (p90 {g['G_A2_memdist_p90_m']:.2f}); t_f vs taxonomy max |diff| {g['t_f_vs_tax_max_abs_s']:.2f} s; "
          + "; ".join(f"{k} rmem missing {v['rmem_missing']} / {v['rollouts']}" for k, v in g.items() if isinstance(v, dict) and "rollouts" in v), "",
          f"**Claim A (pooled, t_f - 3 s): RR {f(res['claim']['primary'])} -> {res['claim']['verdict']}** "
          f"(T {res['claim']['primary']['t_unc']} / {res['claim']['primary']['n_unc']} uncovered vs {res['claim']['primary']['t_cov']} / {res['claim']['primary']['n_cov']} covered).", "",
          f"Amendment 2 (geometric coverage: route stations s_e + 0..18 m inside the span of the valid memory points projected on the route): "
          f"**RR {f(res['claim']['geometric'])} -> {res['claim']['verdict_geometric']}** (T {res['claim']['geometric']['t_unc']} / {res['claim']['geometric']['n_unc']} "
          f"uncovered vs {res['claim']['geometric']['t_cov']} / {res['claim']['geometric']['n_cov']} covered).", "",
          "| set | turn zeros | RR t_f - 3 s (T unc / n vs T cov / n) | verdict | RR t_f - 1 s | geometric RR t_f - 3 s | verdict | geometric t_f - 1 s | shortcut only (- 3 s) | over-turn only (- 3 s) |",
          "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|"]
    q = lambda r: f"{f(r)} ({r['t_unc']}/{r['n_unc']} vs {r['t_cov']}/{r['n_cov']})"  # noqa: E731
    for k, r in res["sets"].items():
        md.append(f"| {k} | {r['n']} | {q(r['m3'])} | {r['m3']['verdict']} | {q(r['m1'])} | {q(r['gm3'])} | {r['gm3']['verdict']} | {q(r['gm1'])} | "
                  f"{f(r['T_shortcut_only_m3'])} | {f(r['T_overturn_only_m3'])} |")
    md += ["", "Counts per class x bucket (t_f - 3 s; pooled), shortcut flags inside each class, and the medians of the memory read at t_f - 3 s:", "",
           "| class | n | covered | uncovered | geo covered / uncovered | shortcut | cov 0-20 | cov 20-40 | geo cov 0-20 / 20-40 | geo nearest covered (m) | first valid offset (m) | first lateral (m) | first bearing (deg) | first >= 10 m: offset / bearing | driven since start (m) | v at t_f |",
           "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for c, r in res["by_class"].items():
        x = TZ[TZ.cls == c]
        md.append(f"| {c} | {r['n']} | {int((~x.unc_m3).sum())} | {int(x.unc_m3.sum())} | {int((~x.unc_gm3).sum())} / {int(x.unc_gm3.sum())} | {r['shortcut']} | {r['cov_0_20_m3']:.2f} | {r['cov_20_40_m3']:.2f} | {r['gcov_0_20_m3']:.2f} / {r['gcov_20_40_m3']:.2f} | {r['gnear_m3']:.0f} | "
                  f"{r['first_off_m3']:.0f} | {r['first_lat_m3']:+.1f} | {r['first_brg_m3']:+.0f} | {r['first10_off_m3']:.0f} / {r['first10_brg_m3']:+.0f} | {r['drv_m3']:.0f} | {r['v_f']:.1f} |")
    c = res.get("completed")
    if c:
        d = c["diff_zero_minus_completed"]
        md += ["", f"Completed strict turns: {c['n']} ({c['reached']} with the entry crossed): uncovered at the entry {100 * c['p_unc_entry']:.0f} %, "
               f"3 s before it {100 * c['p_unc_entry_m3']:.0f} %; turn zeros at t_f - 3 s {100 * c['p_unc_zero_m3']:.0f} % "
               f"(zeros - completed {100 * d['diff']:+.0f} pp [{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]); geometric: completed at the entry {100 * c['gp_unc_entry']:.0f} %, "
               f"3 s before {100 * c['gp_unc_entry_m3']:.0f} %, turn zeros {100 * c['gp_unc_zero_m3']:.0f} %. Per set (slot): "
               + "; ".join(f"{k} n {v['n']}, entry {100 * v['p_unc_entry']:.0f} %, - 3 s {100 * v['p_unc_m3']:.0f} %" for k, v in c["per_set"].items()) + "."]
    m, dz = TZ.first_off_m3 > 0, TZ.gnear_m3 - TZ.first_off_m3
    md += ["", f"Slot labelling (Amendment 2): where the near route is missing, the first served slot says the route starts {dz[m].median():.1f} m (median; "
           f"p90 {dz[m].quantile(0.9):.1f}) nearer than it does along the route ({int((dz[m] >= 5).sum())} / {int(m.sum())} turn zeros >= 5 m)"
           + (f"; completed turns at the entry {(D.gnear_e0 - D.first_off_e0)[D.first_off_e0 > 0].median():.1f} m "
              f"({int(((D.gnear_e0 - D.first_off_e0) >= 5).sum())} / {int((D.first_off_e0 > 0).sum())} >= 5 m)" if len(D) else "") + "."]
    L = res["legs"]
    md += ["", f"Road legs at t_f - 3 s ({L['n']} turn zeros): far segment (s + 60 m) on another leg (>= 45 deg) {L['far_other_leg']}, of the {L['T_n']} T zeros "
           f"{L['T_far_other_leg']}; two opposite turns within 80 m {L['double_opp']}; a turn entry inside the first 40 m (the message gap) {L['gap_turn']}. "
           f"Zeros without a turn (not in the read): {res['non_turn_zeros']}.", "",
           "Per turn zero (t_f - 3 s):", "",
           "| set | scene | primary | class | shortcut (ego / log dev deg) | cov 0-20 / 20-40 (geo) | first valid (m, lat, deg) | driven (m) | far leg dpsi | double | gap turn |",
           "|:--|:--|:--|:--|:--|:--|:--|--:|--:|:--|:--|"]
    for _, r in TZ.sort_values(["set", "cls", "scene"]).iterrows():
        md.append(f"| {r.set} | {r.scene[-36:][:8] if r.scene.startswith('clipgt') else r.scene[-16:]} | {r.primary} | {r.cls} | {r.shortcut} ({r.ego_dev_deg:+.0f} / {r.log_dev_deg:+.0f}) | "
                  f"{r.cov_0_20_m3:.1f} / {r.cov_20_40_m3:.1f} ({r.gcov_0_20_m3:.1f} / {r.gcov_20_40_m3:.1f}) | {r.first_off_m3:.0f}, {r.first_lat_m3:+.1f}, {r.first_brg_m3:+.0f} | {r.drv_m3:.0f} | {r.far_dpsi:+.0f} | "
                  f"{'yes' if r.double_opp else ''} | {'yes' if r.gap_turn else ''} |")
    (out / "progress.md").write_text("\n".join(md) + "\n")


def fig_bev(a, zj, Z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PF, TX = _imports()
    picks = a.picks.split(",")
    jobs = {j[1]["scene"].split("-")[1] if j[1]["scene"].startswith("clipgt") else j[1]["scene"]: j for j in zj if j[0] == a.fig_set}
    fig, axs = plt.subplots(2, 3, figsize=(16, 10.5))
    for ax, p in zip(axs.flat, picks):
        name, o, tax, rm, _ = jobs[p]
        e = Ego(o, rm, PF, TX)
        r = Z[(Z.set == name) & (Z.scene == o["scene"])].iloc[0]
        tf, t3 = r.t_f, max(r.t_f - 3e6, e.t[0])
        for poly in o.get("fig", {}).get("road", []):
            ax.fill(poly[:, 0], poly[:, 1], color="#e6e6e6", lw=0)
        for poly in o.get("fig", {}).get("holes", []):
            ax.fill(poly[:, 0], poly[:, 1], color="white", lw=0)
        for ln in o.get("fig", {}).get("lanes", []):
            ax.plot(ln[:, 0], ln[:, 1], color="#c8c8c8", lw=0.5)
        ax.plot(e.R[:, 0], e.R[:, 1], "--", color="#4a7fc1", lw=1.2, label="route (full, never sent at once)")
        ax.plot(e.LP[:, 0], e.LP[:, 1], color="#3a9a5b", lw=1.6, label="log path")
        k = e.t <= tf + 1e6
        ax.plot(e.P[k, 0], e.P[k, 1], color="#d9622b", lw=2.0, label="driven path")
        i3 = e.i_at(t3)
        s3 = e.st[i3]
        offs = s3 + OFFS
        allp = e.route_xy(offs)
        X, of = e.mem_poly(t3)
        gc = e.geo_cov(t3)
        ok = (offs >= gc.get("g_lo", np.inf)) & (offs <= gc.get("g_hi", -np.inf))
        ax.scatter(allp[~ok, 0], allp[~ok, 1], s=22, facecolors="none", edgecolors="#b03a3a", lw=0.9, label="route -6..72 m not in memory (t_f - 3 s)", zorder=4)
        ax.scatter(X[:, 0], X[:, 1], s=16, color="#1f4e9c", label="memory valid (t_f - 3 s)", zorder=5)
        ax.plot(*e.P[i3], marker="^", ms=10, color="black", zorder=6)
        ax.plot(*e.P[e.i_at(tf)], marker="X", ms=11, color="#b03a3a", zorder=6)
        c = np.r_[e.P[: e.i_at(tf) + 1], X] if len(X) else e.P[: e.i_at(tf) + 1]
        lo, hi = c.min(0) - 12, c.max(0) + 12
        m = max(hi - lo) / 2
        ctr = (lo + hi) / 2
        ax.set_xlim(ctr[0] - m, ctr[0] + m)
        ax.set_ylim(ctr[1] - m, ctr[1] + m)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{p}  {r.primary} -> {r.cls}{'  shortcut' if r.shortcut is True else ''}\n"
                     f"cov 0-20 m slot {r.cov_0_20_m3:.1f} / geo {r.gcov_0_20_m3:.1f}, 20-40 m {r.cov_20_40_m3:.1f} / {r.gcov_20_40_m3:.1f}, driven {r.drv_m3:.0f} m (t_f - 3 s)", fontsize=9)
    axs.flat[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle("Reading A: RSG-L18-s0 PAI 281 turn zeros. Triangle = ego at t_f - 3 s, X = first zero event. Blue dots = the route memory the model "
                 "was fed at t_f - 3 s; red rings = route stations -6..72 m ahead that the memory did not hold", fontsize=10)
    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rmem")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fig", default="")
    p.add_argument("--fig-set", default="pai")
    p.add_argument("--picks", default="9f393c8a,2466266e,382a7247,de9db5dc,1a69f757,fce18b62")
    a = ap.parse_args()
    {"rmem": rmem_extract, "report": report}[a.cmd](a)
