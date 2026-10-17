#!/usr/bin/env python3
"""ALIGN-AUDIT (results/align_audit.md; diagnosis only: no training, no serving change): does the route memory the served driver feeds
(lib/rmem.py, the projected layout of RSG-L18 / RTB1ARO-L18) run ahead of the car at U-turns and junctions, by how much, and where did that
plausibly cause a zero? CPU only; stored reads and stored training features are read, never written.

  extract (box, AlpaSim venv)  rm_align.py extract --set NAME=GLOB [NAME=GLOB ..] --out DIR [--limit N] [--workers N] [--scene S] [--chunk N]
          [--dump]
          every rollout of the reads (GLOB = run dirs): the drive records, AlpaSim's route rebuilt at the logged anchors in decision order
          (rm_build.regen: RouteGeneratorMap, MAP, 40 m, scene data of the run's own config), the memory of the same messages offline in both
          layouts (rm_build.feats_of = the served projected layout, gate against the logged `rmem`; feats_of_o = the ordered, progress-anchored
          layout of decision 302), the evaluator's 2 Hz event series, the controller trace at the failure
          -> DIR/<NAME>.dec.parquet (one row per decision), DIR/<NAME>.roll.parquet (one per rollout), DIR/<NAME>.geo.pkl (route, path, both
          memories and plans of the U-turn, early-memory and zero rollouts, for figures); --dump: DIR/<NAME>.msg.pkl, the regenerated messages,
          anchors, decision times and route R of every rollout (the input of `anchor`, which needs no AlpaSim)
  nav     (box, AlpaSim venv)  rm_align.py nav --out DIR [--data navtrain_full.s0of12 ..] [--limit N]
          the NAVSIM training rows rebuilt exactly as rm_build nav builds them (decision k of a log-following 18-decision rollout; checked
          against the stored feature rows of runs/route_mem/feat), the same lead / early / fold per row against the generator's polyline
          -> DIR/nav_<data>.parquet. The on-policy pool rows (on1_L18RB0, rm_build pool = the served Tracker) are `extract` on RB0-F-s0's
          rollouts.
  msgs    (box, AlpaSim venv)  rm_align.py msgs --out DIR      the driver messages (rollout.asl) of every zero rollout with an `ahead` decision
          in the 3 s before its failure (+ RSG's score-1 rerun of 42c7526f as a control) -> DIR/msgs/<group>/<scene>.pkl, targets.json
  replay  (box, envs/op-train, one GPU: a pool job)  rm_align.py replay --out DIR [--ckpts ..]
          open loop on the logged inputs: each target through each checkpoint with the memory as served, in the ordered layout's slots, and
          empty -> DIR/replay/*.pkl, identity.json (own checkpoint, served memory, against the stored plans)
  report  (Mac, .venv)  rm_align.py report --dir DIR --res RESULTS_DIR --fig PNG
          RESULTS_DIR/align.json + align.md (the tables of align_audit.md) and the BEV figure of the worked example

Definitions (fixed before any output was read):
  route    R = the generator's whole local polyline after the rollout's last decision (every extension it triggered), stations S; heading h(s) =
           chord heading over s +- 2 m
  ego arc  s_e = the anchor's station on R, tracked: decision 0 the global nearest point, then the nearest within [s_prev - 5, s_prev + 40]
  lead     a valid memory slot o (its arc label, m) holds point p (rear-axle frame -> local by the anchor); s_p = p's station on R (global nearest;
           d_p = its distance to R); lead(o) = (s_p - s_e) - o; > 0 = the slot shows route that lies further along than its label says (the memory
           runs ahead of the car). first = the smallest valid o; near = valid o <= +20 m; ahead = some near slot with lead >= 10 m
  msg lead the newest message's first waypoint (route0) at station s_0: s_0 - s_e - L0 (AlpaSim's own projection; 0 unless the simulator snapped)
  fold     the largest backward step (m of route arc) between consecutive points of rmem.merge's polyline: > 0 when a newer batch was appended
           from a point that is not beyond the memory's end (its last segment points back toward it), so the polyline doubles back
  early    the memory has turned while the road has not: a near slot o and the next valid slot both with |h_mem - h(s_e)| >= 30 deg while
           |h(s_e + o) - h(s_e)| < 15 deg at their labels
  pull     signed lateral (left +) of a near slot's point from the route point at its label s_e + o, in the route frame there (mean over the early
           slots, else over the near slots); plan10 = the served plan's lateral from the route at s_e + 10 m (ao_e1's rule)
  geometry the largest heading range of R over the rollout's stretch [s_e(first), s_e(last) + 40 m]: U-turn >= 135 deg, turn 45-135, bend 15-45,
           straight < 15; turn_d = the first label 0 .. 60 m where the route has turned >= 30 deg from h(s_e) (and still has 2 m on)
  failure  zero type and time t_f: the earliest of the flagged rollout metrics (collision_at_fault -> first collision_front / lateral step, offroad,
           left_corridor_laterally) on the 2 Hz series; lat_f = the lateral from the route of the point 3 m ahead of the controller's rear
           axle at t_f (the car's front: a cut corner reads inside); a decision is "before
           the failure" when t_f - 3 s <= t0 <= t_f
  pulled   a zero whose last early decision before t_f has sign(pull) = sign(plan10) = sign(lat_f): the plan went to the memory's side, and the car
           failed on that side
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

REPO = Path(os.environ.get("JEV_REPO") or Path(__file__).resolve().parents[3])      # JEV_REPO: a scratch copy of this file run against a checkout
sys.path[:0] = [str(p) for p in (REPO, REPO / "lib", REPO / "experiments/route_mem/lib", REPO / "experiments/route_mem/scripts",
                                 REPO / "experiments/anchor_offset/scripts", REPO / "experiments/alpasim/scripts", REPO / "experiments/alpasim/lib",
                                 REPO / "experiments/eyes_vs_head/scripts")]
import rmem as RM  # noqa: E402

OFFS = RM.OFFS
NEAR_M, AHEAD_M, TURN_MEM, TURN_ROAD, BEFORE_S = 20.0, 10.0, 30.0, 15.0, 3.0
EV_NAMES = ("offroad", "collision_front", "collision_lateral", "collision_any", "left_corridor_laterally", "wrong_lane")


def wrap(x):
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi


class Route:
    """Polyline R (m, 2) -> stations, chord heading, vectorised projection."""

    def __init__(self, R):
        R = np.asarray(R, float)[:, :2]
        R = R[np.isfinite(R).all(1)]
        keep = np.r_[True, np.hypot(*np.diff(R, axis=0).T) > 1e-6]
        self.R = R[keep]
        seg = np.diff(self.R, axis=0)
        self.L = np.hypot(*seg.T)
        self.u = seg / self.L[:, None]
        self.S = np.r_[0.0, np.cumsum(self.L)]

    def xy(self, s):
        s = np.clip(np.asarray(s, float), 0.0, self.S[-1])
        return np.stack([np.interp(s, self.S, self.R[:, 0]), np.interp(s, self.S, self.R[:, 1])], -1)

    def hdg(self, s):
        s = np.asarray(s, float)
        if self.S[-1] > 4.0:                                  # beyond either end: the end segment's direction
            s = np.clip(s, 2.0, self.S[-1] - 2.0)
        d = self.xy(s + 2.0) - self.xy(s - 2.0)
        return np.arctan2(d[..., 1], d[..., 0])

    def proj(self, P, lo=-np.inf, hi=np.inf):
        """P (n, 2) -> station (n,), signed lateral (left +) (n,), distance (n,), the segments restricted to [lo, hi] when any lies there."""
        P = np.asarray(P, float).reshape(-1, 2)
        a = self.R[:-1]
        ok = (self.S[1:] >= lo) & (self.S[:-1] <= hi)
        if not ok.any():
            ok[:] = True
        a, u, L, s0 = a[ok], self.u[ok], self.L[ok], self.S[:-1][ok]
        t = np.clip(((P[:, None, :] - a[None]) * u[None]).sum(-1), 0.0, L[None])
        q = a[None] + t[..., None] * u[None]
        D = np.hypot(*(P[:, None, :] - q).transpose(2, 0, 1))
        i = D.argmin(1)
        n = np.arange(len(P))
        v = P - a[i]
        return s0[i] + t[n, i], u[i, 0] * v[:, 1] - u[i, 1] * v[:, 0], D[n, i]


def lead_block(route, a, s_e, f):
    """One memory point block (N_PT, NF) at anchor a and ego arc s_e -> dict of the slot-level quantities (module docstring)."""
    v = f[:, 0] > 0.5
    o = OFFS[v]
    out = dict(n_valid=int(v.sum()), first=np.nan, first_true=np.nan, lead_first=np.nan, lead_near=np.nan, lead_near_min=np.nan, near_n=0,
               ahead=False, early=False, early_o=np.nan, pull=np.nan, dmax=np.nan, mem_turn_near=np.nan)
    if not v.any():
        return out, None
    P = RM.to_local(a, f[v, 1:3] * 20.0)
    hm = np.arctan2(f[v, 4], f[v, 3]) + a[2]
    st, _, dist = route.proj(P)
    lead = st - s_e - o
    near = o <= NEAR_M
    h0 = float(route.hdg(s_e))
    mt = np.degrees(np.abs(wrap(hm - h0)))
    rt = np.degrees(np.abs(wrap(route.hdg(s_e + o) - h0)))
    cand = (mt >= TURN_MEM) & (rt < TURN_ROAD)
    pair = cand[:-1] & cand[1:] & (np.diff(o) <= 4.0) & near[:-1]
    q = route.xy(s_e + o)
    ht = route.hdg(s_e + o)
    dl = -(P[:, 0] - q[:, 0]) * np.sin(ht) + (P[:, 1] - q[:, 1]) * np.cos(ht)
    out.update(first=float(o[0]), first_true=float(st[0] - s_e), lead_first=float(lead[0]), dmax=float(dist.max()), near_n=int(near.sum()))
    if near.any():
        out.update(lead_near=float(lead[near].max()), lead_near_min=float(lead[near].min()), ahead=bool((lead[near] >= AHEAD_M).any()),
                   mem_turn_near=float(mt[near].max()), pull=float(dl[near].mean()))
    if pair.any():
        j = int(np.argmax(pair))
        sel = cand & near
        out.update(early=True, early_o=float(o[j]), pull=float(dl[sel].mean()))
    return out, dict(o=o, P=P.astype(np.float32), lead=lead.astype(np.float32), hm=hm.astype(np.float32))


def turn_d(route, s_e):
    o = np.arange(0.0, 62.01, 2.0)
    rt = np.degrees(np.abs(wrap(route.hdg(s_e + o) - route.hdg(s_e))))
    c = (rt >= 30.0) & (s_e + o <= route.S[-1])
    c[:-1] &= c[1:]
    c[-1] = False
    if c[:31].any():
        return float(o[np.argmax(c[:31])])
    return 60.0 if s_e + 60.0 <= route.S[-1] else np.nan


def plan_lat10(route, a, s_e, poses):
    """Served plan (8, 3) rear-axle frame -> its lateral from the route at s_e + 10 m (the last pose's when the plan is shorter) and its 2 s
    heading error against the route."""
    poses = np.asarray(poses, float).reshape(-1, 3)
    Pm = RM.to_local(a, poses[:, :2])
    st, la, _ = route.proj(np.vstack([a[:2], Pm]), s_e - 10.0, s_e + 80.0)
    st = np.maximum.accumulate(st)
    l10 = float(la[-1] if s_e + 10.0 > st[-1] else np.interp(s_e + 10.0, st, la))
    i2 = min(3, len(poses) - 1)
    return l10, float(np.degrees(wrap(a[2] + poses[i2, 2] - route.hdg(st[i2 + 1]))))


# ---------------------------------------------------------------- extract (box, AlpaSim venv)
def read_run(run):
    """-> (summary rollouts, {session: drive records sorted by t0, one per decision time})."""
    S = json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]
    rec = {}
    f = Path(run) / "driver-logs/drive.jsonl"
    if f.exists():
        for line in open(f):
            if '"kind": "drive"' not in line:
                continue
            r = json.loads(line)
            if r.get("infer") is False:
                continue
            rec.setdefault(r["session"], {})[int(r["t0"])] = r
    return S, {k: [v[t] for t in sorted(v)] for k, v in rec.items()}


def series(run, scene, rid):
    import polars as pl
    f = glob.glob(f"{run}/rollouts/{scene}/{rid}/metrics.parquet")
    if not f:
        return {}
    m = pl.read_parquet(f[0]).filter(pl.col("name").is_in(list(EV_NAMES))).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    return {n: np.array(sorted(zip(x["timestamps_us"].to_list(), [np.nan if v is None else v for v in x["values"].to_list()])), float).reshape(-1, 2)
            for (n,), x in m.group_by("name")}


def first_t(ser, names):
    ts = [s[s[:, 1] > 0, 0].min() for n in names if (s := ser.get(n)) is not None and len(s) and (s[:, 1] > 0).any()]
    return float(min(ts)) if ts else np.nan


def ctrl_front(run, rid, t, ahead=3.0):
    """The controller state nearest t -> the point `ahead` m in front of the rear axle (local frame), or None."""
    import csv
    f = Path(run) / f"controller/alpasim_controller_{rid}.csv"
    if not f.exists() or not np.isfinite(t):
        return None
    rows = list(csv.reader(open(f)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    if not len(A):
        return None
    i = int(np.argmin(np.abs(A[:, h["timestamp_us"]] - t)))
    qx, qy, qz, qw = (A[i, h[k]] for k in ("qx", "qy", "qz", "qw"))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return A[i, h["x"]] + ahead * np.cos(yaw), A[i, h["y"]] + ahead * np.sin(yaw)


def extract_unit(u):
    """(set, run, cfg, [(scene, [(rid, score, metrics, records)])], dump) -> (decision rows, rollout rows, geo dict, message dumps); one run dir
    per unit, so a worker builds its scene loader once per unit; a rollout that fails is kept as a rollout row with `err`."""
    import logging
    import traceback
    logging.disable(logging.WARNING)
    name, run, cfg, items, dump = u
    D, Rr, G, Mg = [], [], {}, {}
    for scene, rid, score, met, recs in ((sc, *x) for sc, rolls in items for x in rolls):
        try:
            rows, ro, g, m = one_rollout(name, run, cfg, scene, rid, score, met, recs, dump)
        except Exception:  # noqa: BLE001
            Rr.append(dict(set=name, scene=scene, rid=rid, err=traceback.format_exc()[-600:]))
            continue
        D += rows
        Rr.append(ro)
        if g is not None:
            G[rid] = g
        if m is not None:
            Mg[rid] = m
    return D, Rr, G, Mg


def one_rollout(name, run, cfg, scene, rid, score, met, recs, dump=False):
    """One rollout -> (decision rows, rollout row, geo dict or None, message dump or None) (module docstring). dump: the regenerated messages
    (W (K, 20, 2) rig frame, NaN padded; anchors A (K, 3); t0 (K,) us) and the route R (the truth polyline), for `anchor`."""
    import rm_build as RB
    Fu = {}
    W, A = RB.regen(scene, [dict(r, k=i) for i, r in enumerate(recs)], cfg, Fu)
    Fp, Fo = RB.feats_of(W, A), RB.feats_of_o(W, A)
    route = Route(Fu[max(Fu)])
    ser = series(run, scene, rid)
    flag = dict(coll=float(met.get("collision_at_fault") or 0) > 0, off=float(met.get("offroad") or 0) > 0,
                cor=float(met.get("left_corridor_laterally") or 0) > 0)
    tz = dict(coll=first_t(ser, ("collision_front", "collision_lateral")) if flag["coll"] else np.nan,
              off=first_t(ser, ("offroad",)) if flag["off"] else np.nan, cor=first_t(ser, ("left_corridor_laterally",)) if flag["cor"] else np.nan)
    if flag["coll"] and not np.isfinite(tz["coll"]):
        tz["coll"] = first_t(ser, ("collision_any",))
    zt = min(((t, k) for k, t in tz.items() if np.isfinite(t)), default=(np.nan, ""))
    hint, rows, geo, Bm = None, [], [], []
    for i, r in enumerate(recs):
        wp = W[i][np.isfinite(W[i]).all(1)]
        if len(wp):
            Bm.append(RM.to_local(A[i], wp))
        a = np.asarray(r["anchor"], float)
        s_e, lat_e, _ = route.proj(a[:2][None], *((-np.inf, np.inf) if hint is None else (hint - 5.0, hint + 40.0)))
        s_e, lat_e = float(s_e[0]), float(lat_e[0])
        hint = s_e
        served = r.get("rmem")
        fs = (np.asarray(served, np.float32)[:RM.DIM] if served is not None else Fp[i]).reshape(RM.N_PT, RM.NF)
        fp, fo = Fp[i].reshape(RM.N_PT, RM.NF), RM.pts(Fo[i])
        vv = (fs[:, 0] > 0.5) | (fp[:, 0] > 0.5)
        g_serve = float(np.abs(fs[vv, 1:3] - fp[vv, 1:3]).max() * 20.0) if served is not None and vv.any() else np.nan
        m_serve = bool(((fs[:, 0] > 0.5) == (fp[:, 0] > 0.5)).all()) if served is not None else None
        ls, gs = lead_block(route, a, s_e, fs)
        lo, go = lead_block(route, a, s_e, fo)
        r0 = r.get("route0")
        msg = np.nan
        if r0 is not None and np.isfinite(np.asarray(r0, float)).all():
            s0, _, d0 = route.proj(RM.to_local(a, np.asarray(r0, float)[None]))
            msg = float(s0[0] - s_e - RM.L0) if d0[0] < 1.0 else np.nan
        l10, dpsi2 = plan_lat10(route, a, s_e, r["poses"]) if r.get("poses") is not None else (np.nan, np.nan)
        t0 = float(r["t0"])
        row = dict(set=name, scene=scene, rid=rid, k=i, t0=t0, s_e=s_e, lat_e=lat_e, dpsi_e=float(np.degrees(wrap(a[2] - route.hdg(s_e)))),
                   msg_lead=msg, g_serve=g_serve, m_serve=m_serve, fold=fold_m(route, RM.merge(Bm)), turn_d=turn_d(route, s_e), mem_turn_d=float(RM.turn_dist(fs.reshape(-1))),
                   plan10=l10, plan_dpsi2=dpsi2, cmd=r.get("cmd"), before=bool(np.isfinite(zt[0]) and zt[0] - BEFORE_S * 1e6 <= t0 <= zt[0]),
                   **{k: v for k, v in ls.items()}, **{f"o_{k}": v for k, v in lo.items()})
        rows.append(row)
        geo.append(dict(k=i, a=a.astype(np.float32), s_e=s_e, served=gs, ordered=go, poses=np.asarray(r.get("poses") or np.zeros((0, 3)), np.float32)))
    st0, st1 = rows[0]["s_e"], max(x["s_e"] for x in rows) + 40.0
    grid = np.arange(st0, min(st1, route.S[-1]) + 0.01, 1.0)
    H = np.unwrap(route.hdg(grid)) if len(grid) > 1 else np.zeros(1)
    rng = float(np.degrees(H.max() - H.min()))
    geo_cls = "uturn" if rng >= 135 else "turn" if rng >= 45 else "bend" if rng >= 15 else "straight"
    lat_f = np.nan
    if np.isfinite(zt[0]):
        p = ctrl_front(run, rid, zt[0])
        if p is not None:
            lat_f = float(route.proj(np.array([p]), rows[-1]["s_e"] - 10.0, rows[-1]["s_e"] + 60.0)[1][0])
    early_b = [x for x in rows if x["early"] and x["before"]]
    n_pre = sum(x["early"] for x in rows if not np.isfinite(zt[0]) or x["t0"] <= zt[0])          # exposure before the outcome
    last = early_b[-1] if early_b else None
    pulled = bool(last is not None and np.isfinite(lat_f) and np.sign(last["pull"]) == np.sign(last["plan10"]) == np.sign(lat_f))
    sc_ = np.nan if score is None else float(score)
    ro = dict(set=name, scene=scene, rid=rid, score=sc_, zero=bool(sc_ == 0.0), ztype=zt[1], t_f=zt[0],
              lat_f=lat_f, wrong_lane=float(met.get("wrong_lane") or 0) > 0, t_wl=first_t(ser, ("wrong_lane",)), geo=geo_cls, turn_range=rng,
              turn_net=float(np.degrees(H[-1] - H[0])), n_dec=len(rows), n_ahead=sum(x["ahead"] for x in rows), n_early=sum(x["early"] for x in rows),
              n_early_before=len(early_b), n_early_pre=n_pre, pulled=pulled, n_empty=sum(x["n_valid"] == 0 and x["o_n_valid"] > 0 for x in rows), lead_first_max=float(np.nanmax([x["lead_first"] for x in rows] + [-np.inf])),
              lead_near_max=float(np.nanmax([x["lead_near"] for x in rows] + [-np.inf])), msg_lead_max=float(np.nanmax(np.abs([x["msg_lead"] for x in rows] + [0.0]))),
              o_lead_near_max=float(np.nanmax([x["o_lead_near"] for x in rows] + [-np.inf])), n_o_early=sum(x["o_early"] for x in rows),
              g_serve_max=float(np.nanmax([x["g_serve"] for x in rows] + [0.0])), fold_max=float(max(x["fold"] for x in rows)), dist_m=float(met.get("dist_traveled_m") or np.nan),
              gt_dist_m=float(met.get("gt_dist_traveled_m") or np.nan), progress=float(met.get("progress_rel") or np.nan))
    keep = geo_cls == "uturn" or ro["zero"] or ro["n_early"] or ro["wrong_lane"]
    ks = sorted(W)
    m = dict(scene=scene, W=np.stack([W[k] for k in ks]), A=np.stack([A[k] for k in ks]), t0=np.array([float(recs[k]["t0"]) for k in ks]),
             R=route.R) if dump else None
    return rows, ro, (dict(scene=scene, route=route.R.astype(np.float32), S=route.S.astype(np.float32), dec=geo, roll=ro) if keep else None), m


def cmd_extract(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_mem", "align-extract", config=vars(a)) as run:
        for sp in ("alpasim_nuplan/onpol-held", "alpasim_nuplan/onpol-all", "alpasim_nuplan/onpol-train"):
            run.use_split(splits.load(sp))
        for spec in a.set:
            name, pat = spec.split("=", 1)
            units, n_roll = [], 0
            for run_dir in sorted(glob.glob(pat)):
                S, recs = read_run(run_dir)
                cfg = str(Path(run_dir) / "generated-user-config-0.yaml")
                by = {}
                for r in S:
                    rr = recs.get(r["rollout_id"])
                    if rr:
                        by.setdefault(r["clipgt_id"], []).append((r["rollout_id"], r.get("score"), r.get("metrics") or {}, rr))
                its = [(sc, v) for sc, v in sorted(by.items()) if a.scene in sc]
                units += [(name, run_dir, cfg, its[i:i + a.chunk], a.dump) for i in range(0, len(its), a.chunk)]
            units = units[: a.limit or None]
            res = par.pmap(extract_unit, units, run=run, workers=a.workers or n_cpus(), desc=name)
            D, Rr, G, Mg = [], [], {}, {}
            for v in res.values:
                if v is not None:
                    D += v[0]
                    Rr += v[1]
                    G.update(v[2])
                    Mg.update(v[3])
            pd.DataFrame(D).to_parquet(out / f"{name}.dec.parquet")
            pd.DataFrame(Rr).to_parquet(out / f"{name}.roll.parquet")
            pickle.dump(G, open(out / f"{name}.geo.pkl", "wb"), protocol=4)
            if a.dump:                                       # the regenerated messages of every rollout (`anchor` reads them; no AlpaSim there)
                pickle.dump(Mg, open(out / f"{name}.msg.pkl", "wb"), protocol=4)
            n_roll = len(Rr)
            dd = pd.DataFrame(D)
            rec = dict(units=len(units), errors=len(res.errors), rollouts=n_roll, decisions=len(dd),
                       g_serve_max=float(dd.g_serve.max()) if len(dd) else None, m_serve=float(dd.m_serve.dropna().astype(float).mean()) if len(dd) else None,
                       msg_lead_abs_gt1=float((dd.msg_lead.abs() > 1.0).mean()) if len(dd) else None)
            run.info(f"{name}: {json.dumps(rec)}")
            run.summary[f"set_{name}"] = rec


# ---------------------------------------------------------------- nav (NAVSIM training rows, AlpaSim venv)
def nav_unit(u):
    """One log of a navtrain shard: rm_build.nav_one's rows rebuilt (the route generator on the log track from decision 0, one message per
    decision at the logged poses, the projected memory at the row's frame), checked against the stored feature row, then lead_block against
    the generator's polyline -> list of row dicts."""
    import logging
    import zlib
    import ap2_route as AR
    import rm_build as RB
    from alpasim_runtime.route_generator import RouteGeneratorMap
    logging.disable(logging.WARNING)
    split_dir, log, items, fstore = u
    fr, pos = AR.load_log(split_dir, log)
    vm, org = AR.city_map(fr.loc)
    out = []
    for (r, tok), fst in zip(items, fstore):
        i = pos.get(tok)
        if i is None:
            continue
        k = min(zlib.crc32(f"rm|{tok}".encode()) % RB.K_MAX, i)
        P = RB.track(fr, i - k)
        if P is None:
            continue
        off = np.r_[org, P[0, 2]]
        try:
            g = RouteGeneratorMap((P - off).astype(np.float32), vm, route_start_offset_m=AR.OFFSET_M)
            B = []
            for jj in range(k + 1):
                f_ = i - k + jj
                wp = AR.route_at(g, fr.p[f_] - off, fr.yaw[f_])
                wp = wp[np.isfinite(wp).all(1)]
                if len(wp):
                    B.append(RM.to_local(np.r_[fr.p[f_, :2] - org, fr.yaw[f_]], wp))
        except Exception:  # noqa: BLE001  (off-map / fold-back scenes fail in AlpaSim as well; rm_build skips them too)
            continue
        a = np.r_[fr.p[i, :2] - org, fr.yaw[i]]
        f = RM.memory(B, a).reshape(RM.N_PT, RM.NF)
        route = Route(np.asarray(g.route_polyline_in_local.points, float)[:, :2])
        s_e, lat_e, _ = route.proj(a[:2][None])
        ls, _ = lead_block(route, a, float(s_e[0]), f)
        vv = (f[:, 0] > 0.5) | (fst[:, 0] > 0.5)
        fut = float(np.degrees(wrap(fr.yaw[i + 8] - fr.yaw[i]))) if i + 8 < len(fr.yaw) else np.nan
        out.append(dict(row=int(r), token=tok, k=int(k), s_e=float(s_e[0]), lat_e=float(lat_e[0]), turn_d=turn_d(route, float(s_e[0])),
                        g_store=float(np.abs(f[vv, 1:3] - fst[vv, 1:3]).max() * 20.0) if vv.any() else 0.0, fold=fold_m(route, RM.merge(B)),
                        fut_dyaw4=fut, **ls))
    return out


def fold_m(route, M):
    """Merged memory polyline M (local frame) -> the largest backward step (m of route arc) between consecutive points: > 0 when rmem.merge
    appended a newer batch from a point that is not beyond its end (the polyline folds back on itself)."""
    if M is None or len(M) < 2:
        return 0.0
    st = route.proj(M)[0]
    return float(max(0.0, -np.diff(st).min()))


def cmd_nav(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import data_dir, n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    F, CR = data_dir() / "runs/route_mem/feat", data_dir() / "runs/op_parity/cache"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_mem", "align-nav", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        for data in a.data:
            tab, z = np.load(CR / data / "tab.npz"), np.load(F / f"{data}.npz")
            names, logs = tab["names"], tab["log"]
            assert (z["names"] == names).all(), data
            fz = z["f"].astype(np.float32).reshape(-1, RM.N_PT, RM.NF)
            split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
            by = {}
            for r, (t, lg) in enumerate(zip(names.tolist(), logs.tolist())):
                if z["ok"][r]:
                    by.setdefault(lg, []).append((r, t))
            jobs = [(split_dir, lg, it, fz[[r for r, _ in it]]) for lg, it in sorted(by.items(), key=lambda x: -len(x[1]))][: a.limit or None]
            res = par.pmap(nav_unit, jobs, run=run, workers=a.workers or n_cpus(), desc=data)
            df = pd.DataFrame([x for v in res.values if v for x in v])
            df["dev"] = z["dev"][df.row.values] if "dev" in z.files else False
            df.to_parquet(out / f"nav_{data}.parquet")
            rec = dict(rows=len(df), errors=len(res.errors), g_store_ok=float((df.g_store <= 0.05).mean()), ahead=float(df.ahead.mean()), early=float(df.early.mean()),
                       fold=float((df.fold > 1.0).mean()), lead_first_p90=float(df.lead_first.quantile(0.9)))
            run.info(f"{data}: {json.dumps(rec)}")
            run.summary[f"rows_{data}"] = rec


# ---------------------------------------------------------------- msgs (AlpaSim venv) / replay (op-train env, one GPU: a pool job)
SETRUNS = {"rsg_held": "runs/route_mem/read/RSG-L18-s0/c*/run", "v2_held": "runs/route_mem/read/RTB1ARO-L18-s0/c*/run",
           "rsg_pub": "runs/route_follow/guard/read10/RSG-L18-s0/b*", "v2_pub": "runs/wait_cap/next/cont/wide/pub/RTB1ARO-L18-s0/b*",
           "a_pub": "runs/wait_cap/next/cont/wide/pub/RTB1O-L18-s0/b*"}
OWN = {"rsg_held": "RSG-L18-s0", "v2_held": "RTB1ARO-L18-s0", "rsg_pub": "RSG-L18-s0", "v2_pub": "RTB1ARO-L18-s0", "a_pub": "RTB1O-L18-s0",
       "ctl_pub": "RSG-L18-s0"}
CONTROL = ("ctl_pub", "runs/wait_cap/next/zvid2/pub-RSG-L18-s0", "42c7526f")    # RSG's score-1 rerun of the worked example (its asl kept)
MODES = ("served", "ordered", "empty")


def replay_targets(out):
    """-> [(group, run dir, scene, rid)]: every zero rollout of the five reads with an `ahead` decision in the 3 s before its failure and its
    rollout.asl kept, plus CONTROL."""
    import pandas as pd
    from jevdrive.common import data_dir
    T = []
    for s, pat in SETRUNS.items():
        d, r = pd.read_parquet(Path(out) / f"{s}.dec.parquet"), pd.read_parquet(Path(out) / f"{s}.roll.parquet")
        hit = set(d[d.before & d.ahead].rid)
        for _, x in r[r.zero & r.rid.isin(hit)].iterrows():
            for D in glob.glob(str(data_dir() / pat)):
                if glob.glob(f"{D}/rollouts/{x.scene}/{x.rid}/rollout.asl"):
                    T.append((s, D, x.scene, x.rid))
    g, D, key = CONTROL
    for p in glob.glob(str(data_dir() / D / "rollouts" / f"*{key}*" / "*" / "rollout.asl")):
        T.append((g, str(data_dir() / D), Path(p).parents[1].name, Path(p).parent.name))
    return T


def _msgs(job):
    import c1_extract as X
    return X.one_msgs(job)


def cmd_msgs(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("route_mem", "align-msgs", config=vars(a)) as run:
        T = replay_targets(a.out)
        for g, D, sc, rid in T:
            (Path(a.out) / "msgs" / g).mkdir(parents=True, exist_ok=True)
        res = par.pmap(_msgs, [(D, sc, str(Path(a.out) / "msgs" / g)) for g, D, sc, rid in T], run=run, workers=min(16, len(T)), desc="msgs")
        (Path(a.out) / "msgs" / "targets.json").write_text(json.dumps(T, indent=1))
        run.info(f"{len(T)} rollouts: {[(g, sc[-16:-8]) for g, _, sc, _ in T]}; {len(res.errors)} errors")
        run.summary.update(targets=len(T), errors=len(res.errors))


class _Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def cmd_replay(a):
    """Every target's driver messages through each checkpoint, served memory three ways (MODES): served = the Tracker as served; ordered = the
    same messages in the ordered layout's point block (rmem.memory_o, the d302 progress anchor; the same 40 x 5 slots); empty = zeros.
    -> DIR/replay/<group>__<scene8>__<ckpt>.pkl {mode: [per decision: now, anchor, k, t0, poses, rmem]}; identity (own checkpoint, served)
    against the stored plans -> DIR/replay/identity.json."""
    import tempfile
    import torch
    for p in (REPO / "experiments/op_parity/scripts", REPO / "experiments/route_follow/scripts"):
        sys.path.insert(0, str(p))
    import sh30_driver as D
    from jevdrive.run import Run
    od = Path(a.out) / "replay"
    od.mkdir(parents=True, exist_ok=True)
    T = json.loads((Path(a.out) / "msgs" / "targets.json").read_text())
    D.RMEM = RM                                          # what sh30_driver.main() sets (nx_turns.replay): a Session gets a route tracker
    orig = RM.Tracker.features

    def feats(self, t0, pose_of, anchor):
        if MODE["m"] == "empty":
            return np.zeros(RM.DIM, np.float32)
        if MODE["m"] == "ordered":
            self.order = True
            try:
                return np.asarray(orig(self, t0, pose_of, anchor), np.float32)[:RM.DIM]
            finally:
                self.order = False
        return orig(self, t0, pose_of, anchor)
    MODE = {"m": "served"}
    RM.Tracker.features = feats
    pb, ctx = D.egodriver_pb2, _Ctx()
    ident = {}
    with Run("route_mem", "align-replay", config=vars(a)) as run:
        for ck in a.ckpts:
            core = D.C.Core(ck, "cuda", "backwarp", 1.0, "gpu")
            assert core.rm and not getattr(core, "rm_order", False), f"{ck}: a projected-layout ROUTE-MEM checkpoint"
            RM.SERVE_ORDER = False
            last = {}
            o_plan = core.plan

            def plan(*x, _o=o_plan, **k):
                last["o"] = _o(*x, **k)
                return last["o"]
            core.plan = plan
            for g, run_dir, scene, rid in T:
                msgs = pickle.load(open(Path(a.out) / "msgs" / g / f"{scene}.pkl", "rb"))
                res = {}
                for m in MODES:
                    MODE["m"] = m
                    logd = Path(tempfile.mkdtemp())
                    drv = D.Driver(core, logd, 0, False)
                    rec, uuid = [], None
                    for kind, raw in msgs:
                        if kind == "driver_session_request":
                            req = pb.DriveSessionRequest.FromString(raw)
                            uuid = req.session_uuid
                            drv.start_session(req, ctx)
                        elif kind == "driver_camera_image":
                            drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
                        elif kind == "driver_ego_trajectory":
                            drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
                        elif kind == "route_request":
                            drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
                        elif kind == "driver_request":
                            req = pb.DriveRequest.FromString(raw)
                            try:
                                drv.drive(req, ctx)
                            except RuntimeError as e:
                                run.info(f"{g} {scene[-16:-8]} {ck} {m} t {req.time_now_us}: {e}")
                                continue
                            rec.append(dict(now=int(req.time_now_us), poses=np.asarray(last["o"]["poses"], np.float32).copy(),
                                            rmem=None if "rmem" not in last["o"] else np.asarray(last["o"]["rmem"], np.float32).copy()))
                    drv.sessions.pop(uuid, None)
                    drv.out.flush()
                    drives = [json.loads(x) for x in open(logd / "drive.jsonl") if '"kind": "drive"' in x]
                    assert len(drives) == len(rec), f"{g} {ck} {m}: {len(drives)} records for {len(rec)} decisions"
                    for r, dd in zip(rec, drives):
                        r.update(anchor=np.asarray(dd["anchor"], float), k=int(dd["k"]), t0=int(dd["t0"]))
                    res[m] = rec
                if ck == OWN[g]:                             # identity: the served replay against the stored drive records
                    st = {int(r["t0"]): r for r in read_run(run_dir)[1].get(rid, [])}
                    dd = [float(np.hypot(*(np.asarray(st[r["t0"]]["poses"], float)[7, :2] - r["poses"][7, :2]))) for r in res["served"] if r["t0"] in st]
                    ident[f"{g}__{scene[-16:-8]}"] = dict(n=len(dd), share_4s_lt_10cm=float(np.mean(np.array(dd) < 0.10)) if dd else None,
                                                          max_m=float(max(dd)) if dd else None)
                    run.info(f"identity {g} {scene[-16:-8]} {ck}: {ident[f'{g}__{scene[-16:-8]}']}")
                pickle.dump(res, open(od / f"{g}__{scene[-16:-8]}__{ck}.pkl", "wb"), protocol=4)
            del core
            torch.cuda.empty_cache()
        (od / "identity.json").write_text(json.dumps(ident, indent=1))
        run.summary.update(targets=len(T), identity=ident)


# ---------------------------------------------------------------- report (Mac)
SETS = (("rsg_held", "RSG-L18-s0", "held 460 @18"), ("v2_held", "RTB1ARO-L18-s0 (V2)", "held 460 @18"),
        ("rsg_pub", "RSG-L18-s0", "public 1491 @10"), ("v2_pub", "RTB1ARO-L18-s0 (V2)", "public 1491 @10"),
        ("pool_rb0", "RB0-F-s0 (pool on1_L18RB0)", "train 1020 @18"), ("a_pub", "RTB1O-L18-s0 (arm a, the video)", "public 1491 @10"))


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if len(x) else float("nan")


def boot_rr(y, x, groups, B=10000, seed=0):
    """Risk ratio P(y | x) / P(y | not x) with a scene-cluster bootstrap CI."""
    y, x, groups = np.asarray(y, bool), np.asarray(x, bool), np.asarray(groups)
    u, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    a1, n1 = np.bincount(inv, y & x, len(u)), np.bincount(inv, x, len(u))
    a0, n0 = np.bincount(inv, y & ~x, len(u)), np.bincount(inv, ~x, len(u))
    rr = lambda w: (w @ a1 / max(w @ n1, 1e-9)) / max(w @ a0 / max(w @ n0, 1e-9), 1e-9)  # noqa: E731
    pt = rr(np.ones(len(u)))
    idx = rng.integers(0, len(u), (B, len(u)))
    bs = np.array([rr(np.bincount(i, minlength=len(u)).astype(float)) for i in idx])
    return float(pt), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def replay_table(d, dec, res):
    """Replay read-out: for each target and checkpoint, at the target's decisions that were `ahead` in the 3 s before its failure (the stored
    rollout's flags; the control: its own ahead decisions), the plan's lateral at s_e + 10 m and 2 s heading error against the route, signed
    toward the memory's side (sign of `pull`), with the memory served / in the ordered layout / empty."""
    import pandas as pd
    rd = Path(d) / "replay"
    if not (rd / "identity.json").exists():
        return []
    T = json.loads((Path(d) / "msgs" / "targets.json").read_text())
    geo = {s: pickle.load(open(Path(d) / f"{s}.geo.pkl", "rb")) for s in {"rsg_pub" if g == "ctl_pub" else g for g, *_ in T}}
    rows = []
    for g, _, scene, rid in T:
        s = "rsg_pub" if g == "ctl_pub" else g
        D = dec[s]
        if g == "ctl_pub":                                      # the rerun is not the stored rollout: its own decisions, the stored route
            gg = next(v for v in geo[s].values() if v["scene"] == scene)
            flags = None
        else:
            gg = geo[s][rid]
            Dr = D[D.rid == rid].set_index("k")
            flags = Dr[Dr.before & Dr.ahead]
        route = Route(gg["route"])
        for f in sorted(rd.glob(f"{g}__{scene[-16:-8]}__*.pkl")):
            ck = f.stem.split("__")[-1]
            R = pickle.load(open(f, "rb"))
            out = {}
            for m, recs in R.items():
                v = []
                for i, r in enumerate(recs):
                    a = r["anchor"]
                    s_e = float(route.proj(a[:2][None], *((-np.inf, np.inf) if i == 0 else (v[-1][1] - 5.0, v[-1][1] + 40.0)))[0][0])
                    l10, dp = plan_lat10(route, a, s_e, r["poses"])
                    fs = (np.zeros((RM.N_PT, RM.NF), np.float32) if r["rmem"] is None else np.asarray(r["rmem"], np.float32)[:RM.DIM].reshape(RM.N_PT, RM.NF))
                    lb, _ = lead_block(route, a, s_e, fs) if m == "served" else ({}, None)
                    v.append((i, s_e, l10, dp, lb.get("ahead", False), lb.get("pull", np.nan)))
                out[m] = pd.DataFrame(v, columns=["k", "s_e", "plan10", "dpsi2", "ahead", "pull"]).set_index("k")
            S0 = out["served"]
            ks = list(flags.index) if flags is not None else list(S0.index[S0.ahead])
            ks = [k for k in ks if k in S0.index]
            if not ks:
                continue
            sg = np.sign(S0.loc[ks, "pull"].values)
            row = dict(group=g, scene=scene[-16:-8], ckpt=ck, own=ck == OWN[g], n=len(ks))
            for m in MODES:
                if m in out:
                    X = out[m].loc[ks]
                    row[f"{m}_p10"] = float(np.nanmean(sg * X.plan10.values))
                    row[f"{m}_dpsi"] = float(np.nanmean(sg * X.dpsi2.values))
            rows.append(row)
    res["replay"] = rows
    res["replay_identity"] = json.loads((rd / "identity.json").read_text())
    L = ["", "## T6 Open-loop replay: the same logged inputs with the memory served, re-anchored (ordered layout's slots) or empty", "",
         "At each target's decisions that were `ahead` in the 3 s before its failure: the plan's lateral from the route at s_e + 10 m (m) and its "
         "2 s heading error (deg), both signed toward the side of the misplaced memory (+ = toward it). Identity: own checkpoint, served memory, "
         "4 s plan point against the stored plan.", "",
         "| target | checkpoint | decisions | served p10 | ordered p10 | empty p10 | served dpsi | ordered dpsi | empty dpsi |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        L.append(f"| {r['group']} {r['scene']} | {r['ckpt']}{' (own)' if r['own'] else ''} | {r['n']} | {r.get('served_p10', np.nan):+.2f} | "
                 f"{r.get('ordered_p10', np.nan):+.2f} | {r.get('empty_p10', np.nan):+.2f} | {r.get('served_dpsi', np.nan):+.1f} | "
                 f"{r.get('ordered_dpsi', np.nan):+.1f} | {r.get('empty_dpsi', np.nan):+.1f} |")
    idt = res["replay_identity"]
    L += ["", "Identity (own checkpoint, served): " + "; ".join(f"{k} {v['share_4s_lt_10cm']:.2f} of {v['n']} within 10 cm" for k, v in idt.items() if v["n"])]
    return L


def cmd_report(a):
    import pandas as pd
    d = Path(a.dir)
    res, L = {}, []
    dec = {s: pd.read_parquet(d / f"{s}.dec.parquet") for s, _, _ in SETS if (d / f"{s}.dec.parquet").exists()}
    rol = {s: pd.read_parquet(d / f"{s}.roll.parquet") for s, _, _ in SETS if (d / f"{s}.roll.parquet").exists()}
    lab = {s: (t, b) for s, t, b in SETS}

    # gates
    L += ["## Gates", "", "served = offline: the logged `rmem` against the rebuild from the regenerated messages (max point distance, mask agreement); "
          "msg lead: AlpaSim's own route anchor against the tracked ego arc; fold: rmem.merge's polyline doubles back by > 1 m (decisions, rollouts).", "",
          "| set | rollouts (errors) | decisions | served = offline max m | mask agree | msg lead > 1 m | fold decisions (rollouts) | served empty, ordered not |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for s, D in dec.items():
        R = rol[s]
        ne = int(R.err.notna().sum()) if "err" in R else 0
        g = dict(rollouts=int(len(R) - ne), errors=ne, decisions=int(len(D)), g_serve_max=float(D.g_serve.max()),
                 m_serve=float(D.m_serve.dropna().astype(float).mean()) if D.m_serve.notna().any() else None,
                 msg_gt1=float((D.msg_lead.abs() > 1.0).mean()), msg_gt1_n=int((D.msg_lead.abs() > 1.0).sum()),
                 fold_dec=int((D.fold > 1.0).sum()), fold_roll=int((R.fold_max > 1.0).sum()) if "fold_max" in R else 0,
                 empty=int(((D.n_valid == 0) & (D.o_n_valid > 0)).sum()))
        res.setdefault(s, {})["gate"] = g
        L.append(f"| {lab[s][0]}, {lab[s][1]} | {g['rollouts']} ({ne}) | {g['decisions']} | {g['g_serve_max']:.1e} | "
                 f"{'-' if g['m_serve'] is None else round(g['m_serve'], 4)} | {g['msg_gt1']:.4f} ({g['msg_gt1_n']}) | {g['fold_dec']} ({g['fold_roll']}) | {g['empty']} |")

    # T1: how far ahead, by decision geometry
    L += ["", "## T1 Lead of the served memory (projected layout) against the car's route arc", "",
          "lead = true route offset of the slot's point minus the slot's label (m); first = the first valid slot; near = slots <= +20 m.", "",
          "| set | stratum | decisions | near slots valid | first lead median | p90 | >= 5 m | >= 10 m | >= 30 m | ahead (near >= 10 m) | early turn |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s, D in dec.items():
        R = rol[s].set_index("rid")
        D = D.assign(geo=R.loc[D.rid, "geo"].values)
        D["stratum"] = np.where(D.turn_d.le(42), "turn within 42 m", np.where(D.turn_d.le(60), "turn 44-60 m", "no turn in 60 m"))
        for nm, g in [("all", D)] + [(f"route turn ahead: {k}", g) for k, g in D.groupby("stratum")] + [(f"rollout geometry: {k}", g) for k, g in D.groupby("geo")]:
            x = g.lead_first
            r = dict(n=int(len(g)), near=float((g.near_n > 0).mean()), med=q(x, 0.5), p90=q(x, 0.9), ge5=float((x >= 5).mean()), ge10=float((x >= 10).mean()),
                     ge30=float((x >= 30).mean()), ahead=float(g.ahead.mean()), early=float(g.early.mean()), n_ahead=int(g.ahead.sum()), n_early=int(g.early.sum()))
            res[s].setdefault("t1", {})[nm] = r
            L.append(f"| {lab[s][0]}, {lab[s][1]} | {nm} | {r['n']} | {r['near']:.3f} | {r['med']:.1f} | {r['p90']:.1f} | {r['ge5']:.3f} | {r['ge10']:.3f} | {r['ge30']:.3f} | "
                     f"{r['ahead']:.3f} ({r['n_ahead']}) | {r['early']:.3f} ({r['n_early']}) |")

    # T1b: by decision index (the near range fills as the car drives into the memory)
    L += ["", "## T1b First-slot lead by decision index (all sets pooled per benchmark)", "", "| benchmark | k | decisions | near valid | first lead p90 | >= 10 m | early |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for bench, ss in (("held 460 @18", ("rsg_held", "v2_held")), ("public 1491 @10", ("rsg_pub", "v2_pub"))):
        if not all(x in dec for x in ss):
            continue
        Dk = pd.concat([dec[x] for x in ss])
        for k, g in Dk.groupby("k"):
            r = dict(n=int(len(g)), near=float((g.near_n > 0).mean()), p90=q(g.lead_first, 0.9), ge10=float((g.lead_first >= 10).mean()), early=int(g.early.sum()))
            res.setdefault("t1b", {}).setdefault(bench, {})[int(k)] = r
            L.append(f"| {bench} | {k} | {r['n']} | {r['near']:.2f} | {r['p90']:.1f} | {r['ge10']:.3f} | {r['early']} |")

    # T2: the ordered layout (decision 302) on the same decisions
    L += ["", "## T2 The same decisions in the ordered layout (d302, `memory_o`, progress anchor)", "",
          "| set | decisions | first lead median | p90 | |lead| > 1 m | ahead | early turn | of these on a folded merge | AlpaSim msg lead > 1 m |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s, D in dec.items():
        x = D.o_lead_first
        bad = x.abs() > 1.0
        r = dict(med=q(x, 0.5), p90=q(x, 0.9), bad=int(bad.sum()), ahead=int(D.o_ahead.sum()), early=int(D.o_early.sum()),
                 bad_fold=int((bad & (D.fold > 1.0)).sum()), msg=int((D.msg_lead.abs() > 1).sum()))
        res[s]["t2"] = r
        L.append(f"| {lab[s][0]}, {lab[s][1]} | {len(D)} | {r['med']:.2f} | {r['p90']:.2f} | {r['bad']} | {r['ahead']} | {r['early']} | {r['bad_fold']} | {r['msg']} |")

    # T3: zeros and the early memory
    L += ["", "## T3 Zeros against early-turned memory", "",
          "with early = an early-turn decision before the failure (any decision for a rollout without one); early-before = within 3 s before the "
          "failure; pulled = plan and failure on the memory's side (module docstring).", "",
          "ahead-before (looser) = a decision with a near slot >= 10 m ahead within 3 s before the failure; side = that decision's pull, the plan and "
          "the failure on one side. RR = zero rate with early / without, scene-bootstrap CI; '-' where a cell has no zero.", "",
          "| set | geometry | rollouts | zeros | rollouts with early | zeros with early-before | pulled | zeros with ahead-before | side | zero rate with / without early | RR [CI] |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for s, R in rol.items():
        R = R[R.n_dec.notna()] if "err" in R else R
        D = dec[s]
        loose = {}
        for _, x in R[R.zero].iterrows():
            dd = D[(D.rid == x.rid) & D.before & D.ahead]
            if len(dd):
                l_ = dd.iloc[-1]
                loose[x.rid] = bool(np.isfinite(x.lat_f) and np.sign(l_.pull) == np.sign(l_.plan10) == np.sign(x.lat_f))
        R = R.assign(ahead_b=R.rid.isin(list(loose)), side=R.rid.map(loose).fillna(False).astype(bool))
        res[s]["loose"] = {R.loc[R.rid == k, "scene"].iloc[0]: v for k, v in loose.items()}
        for nm, g in [("all", R)] + [(k, g) for k, g in R.groupby("geo")]:
            e = g.n_early_pre > 0
            zr1 = float(g.zero[e].mean()) if e.any() else float("nan")
            zr0 = float(g.zero[~e].mean()) if (~e).any() else float("nan")
            ok = bool((g.zero & e).any() and (g.zero & ~e).any())
            rr = boot_rr(g.zero.values, e.values, g.scene.values) if ok else (np.nan, np.nan, np.nan)
            r = dict(n=int(len(g)), zeros=int(g.zero.sum()), with_early=int(e.sum()), zero_early_before=int((g.zero & (g.n_early_before > 0)).sum()),
                     pulled=int((g.zero & g.pulled).sum()), ahead_before=int(g.ahead_b.sum()), side=int(g.side.sum()), zr1=zr1, zr0=zr0, rr=rr)
            res[s].setdefault("t3", {})[nm] = r
            rrs = f"{rr[0]:.2f} [{rr[1]:.2f}, {min(rr[2], 99):.2f}]" if ok else "-"
            L.append(f"| {lab[s][0]}, {lab[s][1]} | {nm} | {r['n']} | {r['zeros']} | {r['with_early']} | {r['zero_early_before']} | {r['pulled']} | "
                     f"{r['ahead_before']} | {r['side']} | {zr1:.3f} / {zr0:.3f} | {rrs} |")

    # T4: the scenes (U-turns, early-before zeros, pulled zeros)
    L += ["", "## T4 Scenes: U-turn rollouts and every zero with early memory before the failure", "",
          "| scene | set | geometry | score | zero | wrong lane | early decisions (before) | max near lead m | ordered max near lead m | pulled |", "|:--|:--|:--|--:|:--|:-:|--:|--:|--:|:-:|"]
    rows = []
    for s, R in rol.items():
        for _, x in R[(R.geo == "uturn") | (R.zero & (R.n_early_before > 0))].iterrows():
            rows.append((x.scene[-16:-8], s, x))
    for sc, s, x in sorted(rows, key=lambda t: (t[0], t[1])):
        L.append(f"| {sc} | {s} | {x.geo} ({x.turn_range:.0f} deg) | {x.score:.2f} | {x.ztype or '-'} | {'y' if x.wrong_lane else '-'} | {x.n_early} ({x.n_early_before}) | "
                 f"{x.lead_near_max:.1f} | {x.o_lead_near_max:.1f} | {'y' if x.pulled else '-'} |")
    res["t4"] = [dict(scene=x.scene, set=s, geo=x.geo, turn=float(x.turn_range), score=float(x.score), ztype=x.ztype, wrong_lane=bool(x.wrong_lane), n_early=int(x.n_early),
                      n_early_before=int(x.n_early_before), lead_near_max=float(x.lead_near_max), o_lead_near_max=float(x.o_lead_near_max), pulled=bool(x.pulled)) for _, s, x in rows]

    # T5: training rows
    L += ["", "## T5 Training rows against serving (same rule, same truth)", "",
          "NAVSIM rows rebuilt as rm_build nav builds them (store = share of rows within 0.05 m of the stored float16 feature row); pool = the on-policy rows of "
          "on1_L18RB0 (RB0-F-s0's rollouts, rm_build pool = the served Tracker). label turns = the logged 4 s yaw change >= 30 deg on the early rows.", "",
          "| rows | n | rebuilt = stored | near slots valid | first lead median | p90 | >= 10 m | ahead | early turn | fold | early rows whose label turns |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    tr = sorted(d.glob("nav_navtrain_full.s*of12.parquet"))
    groups = [("NAVSIM navtrain", pd.concat([pd.read_parquet(f) for f in tr]) if tr else None)]
    groups += [(f"{lab[s][0]}, {lab[s][1]}", dec[s]) for s in dec]
    for nm, T in groups:
        if T is None:
            continue
        lt = int(((T.fut_dyaw4.abs() >= 30) & T.early).sum()) if "fut_dyaw4" in T else -1
        r = dict(n=int(len(T)), store=float((T.g_store <= 0.05).mean()) if "g_store" in T else float("nan"), near=float((T.near_n > 0).mean()), med=q(T.lead_first, 0.5),
                 p90=q(T.lead_first, 0.9), ge10=float((T.lead_first >= 10).mean()), ahead=float(T.ahead.mean()), early=float(T.early.mean()), n_early=int(T.early.sum()),
                 fold=float((T.fold > 1.0).mean()), label_turns=lt)
        res.setdefault("t5", {})[nm] = r
        L.append(f"| {nm} | {r['n']} | {r['store']:.4f} | {r['near']:.3f} | {r['med']:.1f} | {r['p90']:.1f} | {r['ge10']:.3f} | {r['ahead']:.4f} | "
                 f"{r['early']:.4f} ({r['n_early']}) | {r['fold']:.4f} | {'-' if lt < 0 else lt} |")

    L += replay_table(d, dec, res)
    Path(a.res).mkdir(parents=True, exist_ok=True)
    (Path(a.res) / "align.json").write_text(json.dumps(res, indent=1, default=float))
    (Path(a.res) / "align.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.fig:
        fig_example(d, a.fig, a.example)


def fig_example(d, png, example):
    """BEV of the worked example: the route, the car's path, the served memory window (projected layout) and the ordered window at three
    decisions, coloured by the slot label; the true station of the first slot written next to it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setname, key = example.split(":", 1)
    G = pickle.load(open(Path(d) / f"{setname}.geo.pkl", "rb"))
    g = next(v for v in G.values() if key in v["scene"])
    R, dec = g["route"], g["dec"]
    ks = sorted({0, len(dec) // 2, len(dec) - 1})
    fig, axs = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 5.2), constrained_layout=True)
    A = np.array([x["a"] for x in dec])
    S = g["S"]
    sel = (S >= dec[0]["s_e"] - 10.0) & (S <= dec[-1]["s_e"] + 90.0)
    pts = [A[:, :2], R[sel]] + [x[b]["P"] for x in dec for b in ("served", "ordered") if x[b] is not None]
    allp = np.vstack(pts)
    c, r = (allp.min(0) + allp.max(0)) / 2, (allp.max(0) - allp.min(0)).max() / 2 + 5.0
    lo, hi = c - r, c + r
    for ax, k in zip(np.atleast_1d(axs), ks):
        x = dec[k]
        ax.plot(R[:, 0], R[:, 1], color="#bbbbbb", lw=6, solid_capstyle="round", zorder=1, label="route (AlpaSim, rebuilt)")
        ax.plot(A[:, 0], A[:, 1], color="#444444", lw=1.5, zorder=2, label="car (decision anchors)")
        a = x["a"]
        ax.scatter([a[0]], [a[1]], marker=(3, 0, np.degrees(a[2]) - 90), s=160, color="#111111", zorder=5)
        if len(x["poses"]):
            pl = RM.to_local(a, x["poses"][:, :2])
            ax.plot(np.r_[a[0], pl[:, 0]], np.r_[a[1], pl[:, 1]], color="#d62728", lw=2, zorder=6, label="served plan (4 s)")
        for blk, mk, lab_ in (("served", "o", "served memory (projected)"), ("ordered", "s", "ordered layout (d302)")):
            b = x[blk]
            if b is None:
                continue
            sc = ax.scatter(b["P"][:, 0], b["P"][:, 1], c=b["o"], cmap="viridis", vmin=-6, vmax=72, marker=mk, s=28 if mk == "o" else 14,
                            edgecolors="k" if mk == "o" else "none", linewidths=0.4, zorder=4 if mk == "o" else 3, label=lab_)
        b = x["served"]
        if b is not None:
            ax.annotate(f"slot {b['o'][0]:+.0f} m, really {b['o'][0] + b['lead'][0]:+.0f} m", b["P"][0], textcoords="offset points", xytext=(6, 8), fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=7)
        ax.set_aspect("equal")
        ax.set_title(f"decision {k}: car at route arc {x['s_e']:.0f} m", fontsize=10)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.tick_params(labelsize=7)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower left")
    fig.colorbar(sc, ax=axs, shrink=0.7, label="slot label (m of route arc from the car)")
    fig.suptitle(f"{g['scene'][-16:]} ({setname}): score {g['roll']['score']:.2f}, {g['roll']['ztype'] or 'no zero'}", fontsize=11)
    fig.savefig(png, dpi=130)
    print("figure ->", png)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--set", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--workers", type=int, default=0)
    e.add_argument("--scene", default="", help="only scenes whose id contains this (a check)")
    e.add_argument("--chunk", type=int, default=16, help="scenes of one run dir per unit")
    e.add_argument("--dump", action="store_true", help="also write DIR/<NAME>.msg.pkl: the regenerated messages and the route per rollout")
    t = sp.add_parser("nav")
    t.add_argument("--data", nargs="+", default=[f"navtrain_full.s{i}of12" for i in range(12)])
    t.add_argument("--out", required=True)
    t.add_argument("--limit", type=int, default=0, help="logs per shard (a check)")
    t.add_argument("--workers", type=int, default=0)
    m = sp.add_parser("msgs")
    m.add_argument("--out", required=True)
    rp = sp.add_parser("replay")
    rp.add_argument("--out", required=True)
    rp.add_argument("--ckpts", nargs="+", default=["RSG-L18-s0", "RTB1ARO-L18-s0", "RTB1O-L18-s0"])
    r = sp.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--res", required=True)
    r.add_argument("--fig", default="")
    r.add_argument("--example", default="v2_pub:42c7526f")
    a = p.parse_args()
    dict(extract=cmd_extract, nav=cmd_nav, msgs=cmd_msgs, replay=cmd_replay, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
