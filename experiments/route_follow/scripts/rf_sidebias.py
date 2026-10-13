#!/usr/bin/env python3
"""SIDE-BIAS (research/route-follow/side-bias.md): where does RSGboth's left / right turn bias come from (left late and wide, right early and
inside, research/route-follow/rsgboth-ledger.md)? Training target, controller tracking or row balance. CPU only, read only.

  pool  (box, repo .venv)  rf_sidebias.py pool --rows L18RB0 --out DIR      -> DIR/pool_rows.csv.gz (one line per pool row)
        The on-policy pool RSGboth trains on (on1_L18RB0 = RB0-F-s0 driving onpol-train, 18 decisions). Per row, in the row frame, against
        the human's logged path (meta gpath, s0 - 15 .. s0 + 70 m): the state's offset e0, the imitation target `fut` (on_rows Amendment 2:
        the human path at the ego's arc + D_k, moved along the path normal by e(D) = (e0 + (tan psi + e0 / Lr) D) exp(-D / Lr)), the log's
        own future `logfut` at the same times and the served plan of RB0 at that state, at 1 / 2 / 3 / 4 s.
  held  (box, repo .venv)  rf_sidebias.py held --set LABEL=<rf_tax pkl> ... --out DIR   -> DIR/held_<LABEL>.csv.gz (one line per decision)
        Held 460 @18 reads: per decision the served plan (drive.jsonl `poses` in the `anchor` frame) and the controller trace (what the car
        did) at 0.5 / 1 / 2 s, both against the human's logged path (rf_tax pkl `log`) and against each other in the anchor frame.
  report (Mac, .venv)      rf_sidebias.py report --res DIR --figs FIGDIR       -> DIR/tables.md, DIR/*.csv, FIGDIR/sidebias.png

Definitions
  path       human's logged path (pool: gpath; held: the log's rear-axle positions, points closer than 0.2 m merged, continued 40 m straight).
  turn       on the path, with the ego's station se: windows [a, a + 30 m], a in [se - 30, se + 20] (ego inside the window or <= 20 m before
             it), heading change h(a + 30) - h(a) (1 m sampling, 3 m chord headings); the window of largest |change|; turn if >= 60 deg,
             side L (counter-clockwise) / R; bin `in` (ego inside the window) / `pre` (0-20 m before it). straight: the largest |change| < 10 deg.
             Defined on the human path (the reference the target is built on), not the lane centre: the ledger showed the two differ in turns.
  lat_in     signed lateral offset to the path, inside negative (left turn: -left offset; right turn: +left offset). On straight rows the
             plain left-positive offset (lat_l) is kept for the global left / right drift.
  lag        heading lag (deg) = (path heading at the foot - pose yaw) x side: + = turned less than the path (late / wide), - = ahead (early).
  lead       arc of the target pose minus arc of the log's pose at the same time (m, + = target ahead of the human).
  push2      fut lat_in - served lat_in at 2 s: the lateral direction the imitation term pulls the plan at that row (- = inward).
  e0 bin     state offset e0 (inside-signed): inside < -0.5 m, on, outside > 0.5 m.
  held d*    driven minus plan in the anchor frame at the same horizon: dlat_in (inside-signed lateral), dyaw_in (deg, + = turned more than the
             plan); r* the same with the driven displacement taken in the controller's own frame at t0 (removes the anchor - controller
             offset at t0, p95 0.27 m); tracking bias = a side-dependent mean of these. plan_lat_in / drv_lat_in = each against the human path.
             Straight decisions: left + throughout. Decisions whose t0 + 2 s lies beyond the controller trace are dropped.
  CI         log-clustered bootstrap 95 % (jevdrive.stats), log = scene id without its token.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

TURN_DEG, STRAIGHT_DEG, WIN, PRE, BACK, EXT = 60.0, 10.0, 30.0, 20.0, 30.0, 40.0
POOL_H = (1, 3, 5, 7)            # fut / served indices of 1 / 2 / 3 / 4 s (0.5 s steps from 0.5 s)
HELD_H = (0.5, 1.0, 2.0)
EBIN = 0.5


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))


class Poly:
    """Polyline (M, 2) -> arcs, chord headings; project(points) -> arc, lateral (left +), heading at the foot."""

    def __init__(self, P):
        P = np.asarray(P, float)
        P = P[np.isfinite(P).all(1)]
        keep = [0]
        for i in range(1, len(P)):
            if np.hypot(*(P[i] - P[keep[-1]])) > 0.2:
                keep.append(i)
        self.P = P[keep]
        d = np.diff(self.P, axis=0)
        self.L = np.hypot(*d.T)
        self.s = np.r_[0, np.cumsum(self.L)]
        self.d = d
        self.h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))

    def heading(self, s):
        """Chord heading over +-1.5 m at arcs s (unwrapped)."""
        s = np.atleast_1d(s)
        return np.interp(s, 0.5 * (self.s[1:] + self.s[:-1]), self.h)

    def project(self, X, lo=None, hi=None):
        X = np.atleast_2d(np.asarray(X, float))
        a, d, L2 = self.P[:-1], self.d, np.maximum(self.L ** 2, 1e-12)
        w = np.clip(((X[:, None] - a) * d).sum(-1) / L2, 0, 1)
        f = a + w[..., None] * d
        dist = np.hypot(*(X[:, None] - f).transpose(2, 0, 1))
        s = self.s[:-1] + w * np.sqrt(L2)
        if lo is not None:
            dist = np.where((s >= lo) & (s <= hi), dist, np.inf)
        j = dist.argmin(1)
        r = np.arange(len(X))
        cr = d[j, 0] * (X[:, 1] - a[j, 1]) - d[j, 1] * (X[:, 0] - a[j, 0])
        return s[r, j], np.sign(cr) * dist[r, j], self.heading(s[r, j])

    def turn(self, se):
        """-> side (+1 L / -1 R / 0), change deg (signed), window start a, bin ('in' / 'pre' / ''), straight flag."""
        lo, hi = max(se - BACK, self.s[0] + 1.5), min(se + PRE, self.s[-1] - 1.5 - WIN)
        if hi < lo:
            return 0, np.nan, np.nan, "", False
        a = np.arange(lo, hi + 1e-6, 1.0)
        dh = np.degrees(self.heading(a + WIN) - self.heading(a))
        i = int(np.abs(dh).argmax())
        if abs(dh[i]) >= TURN_DEG:
            return int(np.sign(dh[i])), float(dh[i]), float(a[i]), ("in" if a[i] <= se else "pre"), False
        return 0, float(dh[i]), float(a[i]), "", bool(abs(dh[i]) < STRAIGHT_DEG)


def side_metrics(poly, X, side, lo, hi):
    """Poses X (n, 3) map / row frame -> lat_in (or lat_l when side == 0), lag (deg), arc."""
    s, lat, h = poly.project(X[:, :2], lo, hi)
    sg = side if side else -1
    return -sg * lat, np.degrees(wrap(h - X[:, 2])) * (side if side else 1), s


# ---------------------------------------------------------------- pool (box)
def cmd_pool(a):
    import pandas as pd
    R = data_dir() / "runs/onpol_nuplan/rows" / a.rows
    m = np.load(R / "meta.npz", allow_pickle=True)
    t = np.load(data_dir() / "runs/op_parity/cache" / f"on1_{a.rows}" / "tab.npz", allow_pickle=True)
    assert (m["names"] == t["names"]).all()
    fut, srv, lf = t["fut"].astype(float), t["served"].astype(float), m["logfut"].astype(float)
    out = []
    for r in range(len(fut)):
        G = Poly(m["gpath"][r])
        se, _, _ = G.project(np.zeros((1, 2)))
        se = float(se[0])
        side, dh, a0, b, st = G.turn(se)
        if not (side or st):
            continue
        sg = side if side else -1
        row = dict(name=str(m["names"][r]), scene=str(m["scene"][r]), k=int(m["k"][r]), side="L" if side > 0 else ("R" if side < 0 else "S"),
                   bin=b, dh=dh, d_entry=a0 - se, speed=float(m["speed"][r]), e0_in=-sg * float(m["e0"][r]), psi_in=-sg * float(np.degrees(m["psi_e"][r])),
                   Lr=float(m["Lr"][r]), D8=float(m["D"][r, -1]), D8_log=float(m["D_reg"][r, -1]))
        lo, hi = se - 5, se + 60
        fl, fg, fs = side_metrics(G, fut[r, list(POOL_H)], side, lo, hi)
        sl, sgl, _ = side_metrics(G, srv[r, list(POOL_H)], side, lo, hi)
        ll, _, ls = side_metrics(G, lf[r, [2, 4, 6, 8]], side, lo, hi) if np.isfinite(lf[r, [2, 4, 6, 8]]).all() else (np.full(4, np.nan),) * 3
        for j, hz in enumerate((1, 2, 3, 4)):
            row |= {f"fut_lat{hz}": fl[j], f"srv_lat{hz}": sl[j], f"log_lat{hz}": ll[j], f"fut_lag{hz}": fg[j], f"srv_lag{hz}": sgl[j],
                    f"fut_lead{hz}": fs[j] - ls[j]}
        out.append(row)
    df = pd.DataFrame(out)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    df.to_csv(o / "pool_rows.csv.gz", index=False, float_format="%.4f")
    print(len(fut), "rows;", df.side.value_counts().to_dict(), "->", o / "pool_rows.csv.gz", flush=True)


# ---------------------------------------------------------------- held (box)
def drive_recs(run):
    S = {}
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "drive":
            S.setdefault(r["session"], []).append((int(r["t0"]), np.asarray(r["anchor"], float), np.asarray(r["poses"], float)))
    return S


def held_one(job):
    label, o = job
    C, lg = o.get("ctrl"), o.get("log")
    if C is None or lg is None:
        return []
    recs = drive_recs(o["run"]).get(o["rid"], [])
    P = np.asarray(lg["P"], float)[:, :2]
    tail = P[-1] + EXT * np.array([np.cos(lg["Y"][-1]), np.sin(lg["Y"][-1])])
    G = Poly(np.r_[P, tail[None]])
    tc = np.asarray(C["t"], float)
    cyaw = np.unwrap(np.asarray(C["yaw"], float))
    cx, cy = np.asarray(C["x"], float), np.asarray(C["y"], float)
    out, hint = [], None
    for k, (t0, an, poses) in enumerate(sorted(recs, key=lambda x: x[0])):
        if t0 + 1e6 * HELD_H[-1] > tc[-1]:
            break
        lo, hi = (None, None) if hint is None else (hint - 10, hint + 30)
        se, _, _ = G.project(an[None, :2], lo, hi)
        se = hint = float(se[0])
        side, dh, a0, b, st = G.turn(se)
        if not (side or st):
            continue
        sg = side if side else -1
        c, s_ = np.cos(an[2]), np.sin(an[2])
        pm = np.c_[an[0] + c * poses[:, 0] - s_ * poses[:, 1], an[1] + s_ * poses[:, 0] + c * poses[:, 1], an[2] + poses[:, 2]]
        hs = np.array(HELD_H)
        ti = t0 + 1e6 * hs
        dv = np.c_[np.interp(ti, tc, cx), np.interp(ti, tc, cy), np.interp(ti, tc, cyaw)]
        pp = np.array([np.interp(h, np.arange(1, 9) * 0.5, pm[:, q]) for h in hs for q in range(3)]).reshape(len(hs), 3)
        a_err = float(np.hypot(np.interp(t0, tc, cx) - an[0], np.interp(t0, tc, cy) - an[1]))
        row = dict(label=label, scene=o["scene"], k=k, side="L" if side > 0 else ("R" if side < 0 else "S"), bin=b, dh=dh, d_entry=a0 - se,
                   score=o.get("score"), anchor_err=a_err, v0=float(np.hypot(np.interp(t0, tc, C["vx"]), np.interp(t0, tc, C["vy"]))))
        pl, pg, _ = side_metrics(G, pp, side, se - 5, se + 40)
        dl, dg, _ = side_metrics(G, dv, side, se - 5, se + 40)
        e0 = np.array([np.interp(t0, tc, cx), np.interp(t0, tc, cy), np.interp(t0, tc, cyaw)])   # controller state at t0
        ce, se_ = np.cos(e0[2]), np.sin(e0[2])
        for j, h in enumerate(hs):
            q = dv[j, :2] - pp[j, :2]                                      # driven - plan, in the anchor frame
            dy = -s_ * q[0] + c * q[1]
            r_ = dv[j, :2] - e0[:2]                                        # driven displacement in the controller's t0 frame vs the plan pose
            kk = int(round(h / 0.5)) - 1
            ry = -se_ * r_[0] + ce * r_[1] - poses[kk, 1]
            row |= {f"plan_lat{h}": pl[j], f"drv_lat{h}": dl[j], f"plan_lag{h}": pg[j], f"drv_lag{h}": dg[j],
                    f"dlat{h}": -sg * dy, f"dyaw{h}": float(np.degrees(wrap(dv[j, 2] - pp[j, 2]))) * (side if side else 1),
                    f"rlat{h}": -sg * ry, f"ryaw{h}": float(np.degrees(wrap(dv[j, 2] - e0[2] - poses[kk, 2]))) * (side if side else 1)}
        out.append(row)
    return out


def cmd_held(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    for spec in a.set:
        label, f = spec.split("=", 1)
        rolls = pickle.load(open(f, "rb"))["rolls"]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for rs in ex.map(held_one, [(label, x) for x in rolls], chunksize=8) for r in rs]
        df = pd.DataFrame(rows)
        df.to_csv(o / f"held_{label}.csv.gz", index=False, float_format="%.4f")
        print(label, len(rolls), "rollouts,", len(df), "decisions", df.side.value_counts().to_dict(), "anchor_err p95 %.3f" % df.anchor_err.quantile(0.95), flush=True)


# ---------------------------------------------------------------- report (Mac)
def boot(x, g):
    from jevdrive import stats
    x, g = np.asarray(x, float), np.asarray(g)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan, int(ok.sum())
    r = stats.bootstrap(x[ok], g[ok])
    return r["mean"], r["lo"], r["hi"], r["n"]


def logof(s):
    return np.array([x.rsplit("-", 1)[0] for x in np.asarray(s, str)])


def ci(x, g, f=".2f"):
    m, lo, hi, n = boot(x, g)
    return f"{m:+{f}} [{lo:+{f}}, {hi:+{f}}]" if np.isfinite(m) else "-"


def cmd_report(a):
    import pandas as pd
    o = Path(a.res)
    md = ["# SIDE-BIAS tables (rf_sidebias.py report)", ""]
    P = pd.read_csv(o / "pool_rows.csv.gz")
    P["log"] = logof(P.scene)
    P["ebin"] = np.where(P.e0_in < -EBIN, "inside", np.where(P.e0_in > EBIN, "outside", "on"))
    for hz in (1, 2, 3, 4):
        P[f"push{hz}"] = P[f"fut_lat{hz}"] - P[f"srv_lat{hz}"]
    T = P[P.side != "S"]
    # ---- 1 pool targets
    md += ["## 1. Pool on1_L18RB0: target `fut` vs the human's own future, per side (turn rows, lat inside -)", "",
           f"rows in the pool with a turn (>= {TURN_DEG:.0f} deg / {WIN:.0f} m on the human path, ego inside or <= {PRE:.0f} m before): "
           f"L {int((T.side == 'L').sum())} / R {int((T.side == 'R').sum())}; straight (< {STRAIGHT_DEG:.0f} deg) {int((P.side == 'S').sum())}", ""]
    hdr = "| side | bin | rows | e0_in | " + " | ".join(f"fut lat {h} s" for h in (1, 2, 3, 4)) + " | log lat 2 s | fut lead 2 s | fut lag 2 s | served lat 2 s | push 2 s |"
    md += [hdr, "|" + ":--|" * 14]
    tab1 = []
    for sd in "LR":
        for b in ("all", "inside", "on", "outside"):
            X = T[(T.side == sd) & ((T.ebin == b) | (b == "all"))]
            cells = [ci(X[f"fut_lat{h}"], X.log) for h in (1, 2, 3, 4)]
            md.append(f"| {sd} | {b} | {len(X)} | {ci(X.e0_in, X.log)} | " + " | ".join(cells) +
                      f" | {ci(X.log_lat2, X.log)} | {ci(X.fut_lead2, X.log)} | {ci(X.fut_lag2, X.log, '.1f')} | {ci(X.srv_lat2, X.log)} | {ci(X.push2, X.log)} |")
            tab1.append(dict(side=sd, ebin=b, rows=len(X), **{f"{c}": X[c].mean() for c in
                        ["e0_in", "fut_lat1", "fut_lat2", "fut_lat3", "fut_lat4", "log_lat2", "fut_lead2", "fut_lag2", "srv_lat2", "srv_lag2", "push2", "Lr"]}))
    pd.DataFrame(tab1).to_csv(o / "pool_side_bins.csv", index=False, float_format="%.4f")
    # blend symmetry: fut_lat / e0 (decay ratio) and Lr by side and offset sign
    md += ["", "Blend geometry (rows with |e0| > 0.5 m): decay ratio fut_lat(t) / e0_in and Lr, by side and by which side of the path the state is", "",
           "| side | state | rows | ratio 1 s | ratio 2 s | ratio 4 s | Lr (m) | psi_in (deg) |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for sd in "LR":
        for b in ("inside", "outside"):
            X = T[(T.side == sd) & (T.ebin == b)]
            r = {h: X[f"fut_lat{h}"] / X.e0_in for h in (1, 2, 4)}
            md.append(f"| {sd} | {b} | {len(X)} | {r[1].median():.2f} | {r[2].median():.2f} | {r[4].median():.2f} | {X.Lr.median():.1f} | {ci(X.psi_in, X.log, '.1f')} |")
    # ---- 3 balance
    S = P[P.side == "S"]
    md += ["", "## 3. Row balance in the pool and the straight-row drift (lat left +)", "",
           f"turn rows L : R = {int((T.side == 'L').sum())} : {int((T.side == 'R').sum())} (scenes {T[T.side == 'L'].scene.nunique()} : {T[T.side == 'R'].scene.nunique()}); "
           f"straight rows {len(S)} ({len(S) / len(P) * 100:.0f} % of the rows classified; pool total in pool_rows header). "
           f"Turn change |deg| median L {T[T.side == 'L'].dh.abs().median():.0f} / R {T[T.side == 'R'].dh.abs().median():.0f}.", "",
           "| set | rows | e0 (left +) | fut lat 2 s | fut lat 4 s | served lat 2 s | push 2 s | push 4 s |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    # side_metrics on straight rows returns lat_in with sg = -1 -> +left: that is lat_l
    md.append(f"| straight | {len(S)} | {ci(S.e0_in, S.log)} | {ci(S.fut_lat2, S.log)} | {ci(S.fut_lat4, S.log)} | {ci(S.srv_lat2, S.log)} | "
              f"{ci(S.push2, S.log)} | {ci(S.push4, S.log)} |")
    for sd in "LR":
        X = T[T.side == sd]
        sgn = 1 if sd == "R" else -1                                        # lat_in -> lat_l
        md.append(f"| turn {sd} (as left +) | {len(X)} | {ci(sgn * X.e0_in, X.log)} | {ci(sgn * X.fut_lat2, X.log)} | {ci(sgn * X.fut_lat4, X.log)} | "
                  f"{ci(sgn * X.srv_lat2, X.log)} | {ci(sgn * X.push2, X.log)} | {ci(sgn * X.push4, X.log)} |")
    # ---- 2 / 4 held
    labels = sorted(f.name[5:-7] for f in o.glob("held_*.csv.gz"))
    H = {lb: pd.read_csv(o / f"held_{lb}.csv.gz") for lb in labels}
    md += ["", "## 2 / 4. Held 460 @18: served plan vs what the car did (turn decisions, inside -; d = driven - plan in the anchor frame)", "",
           "| label | side | dec | plan lat 1 s | drv lat 1 s | plan lat 2 s | drv lat 2 s | dlat 0.5 s | dlat 2 s | rlat 2 s | plan lag 2 s | drv lag 2 s | dyaw 2 s (+ more turn) | ryaw 2 s |",
           "|" + ":--|" * 14]
    tab2 = []
    for lb in labels:
        D = H[lb]
        D["log"] = logof(D.scene)
        for sd in "LR":
            X = D[D.side == sd]
            md.append(f"| {lb} | {sd} | {len(X)} | {ci(X['plan_lat1.0'], X.log)} | {ci(X['drv_lat1.0'], X.log)} | {ci(X['plan_lat2.0'], X.log)} | "
                      f"{ci(X['drv_lat2.0'], X.log)} | {ci(X['dlat0.5'], X.log)} | {ci(X['dlat2.0'], X.log)} | {ci(X['rlat2.0'], X.log)} | {ci(X['plan_lag2.0'], X.log, '.1f')} | "
                      f"{ci(X['drv_lag2.0'], X.log, '.1f')} | {ci(X['dyaw2.0'], X.log, '.1f')} | {ci(X['ryaw2.0'], X.log, '.1f')} |")
            tab2.append(dict(label=lb, side=sd, dec=len(X), **{c: X[c].mean() for c in X.columns if c[:4] in ("plan", "drv_", "dlat", "dyaw", "rlat", "ryaw")}))
        X = D[D.side == "S"]
        md.append(f"| {lb} | straight (left +) | {len(X)} | {ci(X['plan_lat1.0'], X.log)} | {ci(X['drv_lat1.0'], X.log)} | {ci(X['plan_lat2.0'], X.log)} | "
                  f"{ci(X['drv_lat2.0'], X.log)} | {ci(X['dlat0.5'], X.log)} | {ci(X['dlat2.0'], X.log)} | {ci(X['rlat2.0'], X.log)} | | | | |")
        tab2.append(dict(label=lb, side="S", dec=len(X), **{c: X[c].mean() for c in X.columns if c[:4] in ("plan", "drv_", "dlat", "dyaw", "rlat", "ryaw")}))
    pd.DataFrame(tab2).to_csv(o / "held_side.csv", index=False, float_format="%.4f")
    # paired by scene: RSGboth - RSG and RSGboth - RB0, plan lat 2 s per side (group mean over seeds)
    md += ["", "Plan lat 2 s, paired by scene (mean over the label's seeds per scene, then the difference), turn decisions:", "",
           "| pair | side | scenes | plan lat 2 s | drv lat 2 s | dlat 2 s | rlat 2 s |", "|:--|:--|:--|:--|:--|:--|:--|"]
    grp = {"RSGboth": [x for x in labels if x.startswith("RSGboth")], "RSG": [x for x in labels if x.startswith("RSG-")], "RB0": [x for x in labels if x.startswith("RB0")]}

    def per_scene(g, sd, c):
        return pd.concat([H[x][H[x].side == sd].groupby("scene")[c].mean() for x in grp[g]], axis=1).mean(1)
    for A, B in (("RSGboth", "RSG"), ("RSGboth", "RB0"), ("RSG", "RB0")):
        if not grp[A] or not grp[B]:
            continue
        for sd in "LRS":
            cells = []
            for c in ("plan_lat2.0", "drv_lat2.0", "dlat2.0", "rlat2.0"):
                d = (per_scene(A, sd, c) - per_scene(B, sd, c)).dropna()
                cells.append(ci(d.values, logof(d.index)))
            md.append(f"| {A} - {B} | {sd} | {len(d)} | " + " | ".join(cells) + " |")
    (o / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if a.figs:
        fig(P, H, grp, Path(a.figs))


def fig(P, H, grp, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs.mkdir(parents=True, exist_ok=True)
    f, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharey="row")
    T = P[P.side != "S"]
    hz = np.array([1, 2, 3, 4])
    col = {"RSGboth": "#d95f02", "RSG": "#1b9e77", "RB0": "#7570b3"}
    for i, sd in enumerate("LR"):
        X = T[T.side == sd]
        a = ax[i, 0]
        for b, ls in (("inside", ":"), ("on", "-"), ("outside", "--")):
            Y = X[X.ebin == b]
            a.plot(np.r_[0, hz], np.r_[Y.e0_in.mean(), [Y[f"fut_lat{h}"].mean() for h in hz]], "k" + ls, label=f"target fut, state {b} (n={len(Y)})")
            a.plot(hz, [Y[f"srv_lat{h}"].mean() for h in hz], ls, color=col["RB0"], alpha=0.8, label=f"RB0 served, state {b}")
        a.plot(np.r_[0, hz], np.r_[0, [X[f"log_lat{h}"].mean() for h in hz]], color="0.6", lw=3, alpha=0.5, label="human log")
        a.set_title(f"pool on1_L18RB0, {'left' if sd == 'L' else 'right'} turns: target vs human")
        a = ax[i, 1]
        a.plot(np.r_[0, hz], np.r_[X.e0_in.mean(), [X[f"fut_lat{h}"].mean() for h in hz]], "k-", label="target fut (all rows)")
        a.plot(hz, [X[f"srv_lat{h}"].mean() for h in hz], color=col["RB0"], label="RB0 served")
        a.plot(hz, [X[f"push{h}"].mean() for h in hz], "k--", label="push = fut - served")
        a.axhline(0, color="0.6", lw=3, alpha=0.5)
        a.set_title("pool: mean over all turn rows (inside -)")
        a = ax[i, 2]
        hh = np.array(HELD_H)
        for g, labs in grp.items():
            if not labs:
                continue
            D = pd_concat([H[x] for x in labs])
            D = D[D.side == sd]
            a.plot(hh, [D[f"plan_lat{h}"].mean() for h in hh], "-o", color=col[g], label=f"{g} plan")
            a.plot(hh, [D[f"drv_lat{h}"].mean() for h in hh], "--s", color=col[g], label=f"{g} driven")
        a.axhline(0, color="0.6", lw=3, alpha=0.5)
        a.set_title("held 460: served plan vs driven (inside -)")
    for a in ax.flat:
        a.set_xlabel("horizon (s)")
        a.legend(fontsize=6.5)
        a.grid(alpha=0.3)
    for a in ax[:, 0]:
        a.set_ylabel("lateral offset to human path (m, inside -)")
    f.tight_layout()
    f.savefig(figs / "sidebias.png", dpi=130)
    print("->", figs / "sidebias.png")


def pd_concat(xs):
    import pandas as pd
    return pd.concat(xs, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("pool")
    q.add_argument("--rows", default="L18RB0")
    q.add_argument("--out", required=True)
    q = s.add_parser("held")
    q.add_argument("--set", nargs="+", required=True, help="LABEL=<rf_tax extract pkl>")
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=4)
    q = s.add_parser("report")
    q.add_argument("--res", required=True)
    q.add_argument("--figs", default="")
    a = p.parse_args()
    dict(pool=cmd_pool, held=cmd_held, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
