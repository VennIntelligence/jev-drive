#!/usr/bin/env python3
"""ROUTE-NIGHT counterfactual pool `on1_CF` (research/next-round/route-night.md section 2 "反事实行", section 4 stage 1 B): the camera tokens of a
navtrain row paired with the route memory of ANOTHER legal route, labelled with that route (route hinge, dead zone 0.5 m) and the row's own
road raster; no imitation, no agent hinge. CPU only. Three stages, two venvs (the lane graph needs nuPlan's map API, the route generator
AlpaSim's runtime); the hand-off is one pickle per shard.

  geom   (envs/navsim2)   --data navtrain_full.s2of12 ..   per token, corridor/lib/lanegraph.py on the logged poses (corr_geom.Matcher, decision 240):
                          junction  the driven node paths within 40 m of the ego's lane and every other path whose exit heading at 40 m differs by
                                    >= 45 deg from all driven ones (and turns <= 120 deg off the ego heading); up to 2 per token, largest first ->
                                    counterfactual centreline = that node path from the ego's projection, 110 m (straightest successor beyond)
                          lc        the matched sequence makes a lateral (lane-change) move within 4 s and the logged pose at +4 s is >= 2 m off the
                                    centreline of the lane run at t0 -> counterfactual centreline = that run, 110 m ("stay in lane")
                          -> $DATA_DIR/runs/route_follow/cf/geom/<data>.pkl: list of dict(row, token, log, cls 'j0' / 'j1' / 'lc', cl (m, 2) global xy
                             at 1 m, dh (deg, junction exit heading gap), branch (m of arc to the first node off the driven path))
  build  (AlpaSim venv)   --data ..                         per counterfactual: k = crc32("rm|" + token) % 18 clipped to the frames before (rm_build nav);
                          the fake recorded track = the logged poses of frames i - k .. i (cubic spline to 0.1 s, as rm_build.track) followed by the
                          counterfactual centreline from 1 m beyond the ego's projection (z of frame i); RouteGeneratorMap on it (the exit centreline
                          IS the track, so the generator cannot fall back to the straightest successor before 110 m); where its lane matching
                          snaps to a neighbour lane (AlpaSim's find_consistent_upcoming_lane_id keeps the longest-lived of parallel candidates
                          within 4 m), RouteGeneratorRecorded on the same track (their resampling / trimming, no map projection; gen = rec); route messages j = 0..k at the
                          logged poses (ap2_route.route_at), carried into frame i -> rmem features; label = route_polyline_in_local in frame i, arc
                          -10 .. +90 m around the ego's projection every 2 m (51 points, NaN outside); cmd = rf_cf.route_cmd of the message at the
                          decision (navigation.command_from_route). ok also needs the generated route within 1.5 m of the lane-graph centreline
                          over 0 .. 60 m (the generator snapped to the intended lanes); beyond 60 m the label is cut where it leaves
                          that line. The real route (rm_build nav's log track, i.e. the navtrain row's own memory) at the same poses: ok also needs
                          the merged memory windows to differ by >= 2 m somewhere (rm_build's track ends 19 - k frames after the token, so the real
                          route often extends straight where the log turns or changes lane later: such a counterfactual equals the real row);
                          equal-arc gaps of the newest / first message and of the merged memory window are kept as diagnostics
                          -> $DATA_DIR/runs/route_follow/cf/rows/<data>.npz
  pack   (AlpaSim venv)   --data .. [--name CF]            pool files (experiments/onpol_nuplan/plans/on_pool_format.md), split by log: logs holding a
                          token of navsim/op-parity-full-dev -> on1_<name>dev (G-swap only, never trained), all others -> on1_<name>:
                          cache/on1_<name>/{tab, agent (ok False), road (the token's P2H10 raster from runs/body1/labels_road: its frame is the
                          token's rear-axle frame = the row frame, so no `off`), route}.npz, cache/on1_<name>@warp/{front.npy, teacher.npz} (row
                          subset of the shard's @warp files, di / pi unchanged), runs/route_mem/feat/on1_<name>.npz (names, f, k, ok);
                          tab = the shard's tab rows with names on:cf:<token>:<j0|j1|lc>, ego[:, 1:4] and cmd[:, -1] replaced by the counterfactual
                          command, fut = the logged future (finite; not read in `self` mode). Gates -> $DATA_DIR/runs/route_follow/cf/gates.json
                          (+ experiments/route_follow/results/cf_gates.json), BEV figures experiments/route_follow/figs/cf_bev_*.png:
                          G-cf1  50 random pool rows: median of the per-row max equal-arc gap between the counterfactual and the real message the
                                 driver receives at the decision (j = k) >= 2 m (first message j = 0 and merged memory reported as well)
                          G-cf2  >= 95 % of the label polyline samples inside the raster extent have sdf_road > 0 (bilinear)
                          G-cf3  6 rows (3 junction, 3 lane change) drawn for a human look

  NAV=$DATA_DIR/envs/navsim2/bin/python; P=$DATA_DIR/third_party/alpasim/.venv/bin/python
  $NAV experiments/route_follow/scripts/rf_cfpool.py geom --data navtrain_full.s2of12 && $P .. build --data navtrain_full.s2of12 && $P .. pack --data navtrain_full.s2of12
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "lib"), str(R / "experiments/route_mem/lib"), str(R / "experiments/route_mem/scripts"), str(R / "experiments/alpasim/scripts"),
                str(R / "experiments/alpasim/lib"), str(R / "experiments/corridor/lib"), str(R / "experiments/corridor/scripts"), str(R / "experiments/route_follow/scripts")]
from jevdrive.common import data_dir  # noqa: E402

D = data_dir()
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
CR = D / "runs/op_parity/cache"
OUT = D / "runs/route_follow/cf"
FEAT = D / "runs/route_mem/feat"
ROAD = D / "runs/body1/labels_road/navtrain_s01234567891011.npz"
DEV = "navsim/op-parity-full-dev"
K_MAX, WINDOW, MIN_DH, MAX_TURN, AHEAD, LC_MIN, SNAP = 18, 40.0, 45.0, 120.0, 110.0, 2.0, 1.5
M_PT, S_BACK, S_STEP, DZ = 51, -10.0, 2.0, 0.5
DIFF = 2.0                                                              # m, min max-gap of the counterfactual to the real merged memory
I0, I60 = 5, 35                                                         # label indices of arc 0 m and +60 m (ok needs the generated route on the line there)


def split_dir(log):
    return "test" if (D / "datasets/navsim/navsim_logs/test" / f"{log}.pkl").exists() else "trainval"


def by_log(data):
    tab = np.load(CR / data / "tab.npz")
    by = {}
    for r, (t, lg) in enumerate(zip(tab["names"].tolist(), tab["log"].tolist())):
        by.setdefault(lg, []).append((r, t))
    return by


# ---------------------------------------------------------------- geom (navsim2: lane graph)
def geom_one(u):
    import corr_geom as CG
    import lanegraph as LG
    log, items = u
    fr, xy, yaw = CG.log_frames(split_dir(log), log)
    g = LG.get(fr[0]["map_location"])
    M = CG.Matcher(g, xy, yaw)
    pos = {f["token"]: i for i, f in enumerate(fr)}
    out = []

    def dense(Rxy):                                         # 0.25 m centreline -> 1 m
        return np.ascontiguousarray(Rxy[::4])

    for row, tok in items:
        i = pos.get(tok)
        if i is None:
            continue
        st, fpath, _ = M.driven(i)
        if fpath is None:
            continue
        nodes = [nd for _, nd in fpath]
        base = dict(row=row, token=tok, log=log)
        # lane change: a lateral move within 4 s and >= 2 m off the t0 lane run at +4 s
        runs = g.runs(nodes)
        if len(runs) > 1 and fpath[runs[1][0]][0] <= 8 and i + 8 < len(xy):
            Rxy, _, _ = g.centreline(runs[0][1], xy[i], AHEAD)
            if len(Rxy) > 40:
                ln = LG.Line(Rxy)
                s8, d8 = ln.frenet(xy[i + 8][None])
                _, d0 = ln.frenet(xy[i][None])
                if 0 < s8[0] < ln.L and abs(d8[0] - d0[0]) >= LC_MIN:
                    out.append(base | dict(cls="lc", cl=dense(Rxy), dh=np.nan, branch=np.nan))
        # junction: alternative exits (corr_geom.supply_log's paths, decision 240's >= 45 deg rule)
        seq, _ = g.sequence(nodes)
        s0 = float(g.line[seq[0]].frenet(xy[i][None])[0][0])
        route = {str(x) for x in (fr[i].get("roadblock_ids") or [])}
        own = {}
        for p, _ in g.paths_ahead(seq[0], s0, WINDOW):
            own.setdefault(p[-1], (p, g.heading_at(p, s0, WINDOW)))
        own = list(own.values())
        drv = [(p, h) for p, h in own if all(seq[k] == p[k] for k in range(min(len(seq), len(p))))]
        if len(drv) > 1 and route:
            drv = [(p, h) for p, h in drv if all(g.group[k] in route for k in p)] or drv
        if not drv:
            continue
        dh = np.array([h for _, h in drv])
        alts = []
        for p, h in own:
            if any(p is q for q, _ in drv):
                continue
            gap = float(np.degrees(np.abs(LG.wrap(h - dh)).min()))
            if gap >= MIN_DH and abs(float(np.degrees(LG.wrap(h - yaw[i])))) <= MAX_TURN:
                alts.append((gap, p))
        for j, (gap, p) in enumerate(sorted(alts, key=lambda x: -x[0])[:2]):
            Rxy, _, _ = g.centreline(p, xy[i], AHEAD)
            if len(Rxy) < 4 * 60:
                continue
            dp = drv[0][0]
            n = next((m for m in range(len(p)) if m >= len(dp) or p[m] != dp[m]), len(p))
            br = sum(g.line[k].L for k in p[:n]) - s0
            out.append(base | dict(cls=f"j{j}", cl=dense(Rxy), dh=gap, branch=float(br)))
    return out


def cmd_geom(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("route_follow", "cfpool-geom", config=vars(a)) as run:
        for data in a.data:
            dst = OUT / "geom" / f"{data}.pkl"
            if dst.exists() and not a.force:
                run.info(f"{data}: exists")
                continue
            jobs = sorted(by_log(data).items(), key=lambda x: -len(x[1]))[: a.limit or None]
            res = par.pmap(geom_one, jobs, run=run, workers=a.workers or None, desc=f"geom {data}")
            rows = [r for v in res.values if v for r in v]
            if res.errors:
                run.info(f"{data}: {res.summary()}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(rows, open(dst, "wb"))
            c = {k: sum(r["cls"] == k for r in rows) for k in ("j0", "j1", "lc")}
            run.info(f"{data}: {len(jobs)} logs, {c}, tokens {len({r['token'] for r in rows})}, failed logs {len(res.errors)}")
            run.summary[data] = dict(logs=len(jobs), failed=len(res.errors), **c)


# ---------------------------------------------------------------- build (AlpaSim: route generator + memory)
def spline(fr, i0, i1):
    """Logged positions of frames i0 .. i1 at 0.1 s (rm_build.track's spline) -> (m, 3), or None when the frames are not 0.5 s apart."""
    from scipy.interpolate import CubicSpline
    if i1 == i0:
        return fr.p[i0:i0 + 1].copy()
    t = fr.t[i0:i1 + 1] - fr.t[i0]
    if np.abs(np.diff(t) - 0.5).max() > 0.05:
        return None
    return CubicSpline(t, fr.p[i0:i1 + 1])(np.arange(int(round(t[-1] / 0.1)) + 1) * 0.1)


def arc_window(Me):
    """Polyline (M, 2) in the row frame -> (M_PT, 2) samples at arc -10 .. +90 m around the origin's projection (NaN outside), seglen (M_PT - 1,)."""
    P = np.full((M_PT, 2), np.nan, np.float32)
    d = np.diff(Me, axis=0)
    L = np.hypot(*d.T)
    keep = np.r_[True, L > 1e-6]
    Me = Me[keep]
    d = np.diff(Me, axis=0)
    L = np.hypot(*d.T)
    if len(L) == 0:
        return P, np.zeros(M_PT - 1, np.float32)
    cum = np.r_[0.0, np.cumsum(L)]
    u = d / L[:, None]
    t = np.clip(-(Me[:-1] * u).sum(1), 0.0, L)
    j = int(np.argmin(np.hypot(*(Me[:-1] + t[:, None] * u).T)))
    s = cum[j] + t[j] + S_BACK + S_STEP * np.arange(M_PT)
    ok = (s >= 0) & (s <= cum[-1])
    P[ok, 0], P[ok, 1] = np.interp(s[ok], cum, Me[:, 0]), np.interp(s[ok], cum, Me[:, 1])
    sl = np.nan_to_num(np.hypot(*np.diff(P, axis=0).T)).astype(np.float32)
    return P, sl


def pl_dist(Q, M):
    """Distance of points Q (n, 2) to polyline M (m, 2)."""
    a, ab = M[:-1], np.diff(M, axis=0)
    ap = Q[:, None] - a[None]
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.hypot(*(ap - t[..., None] * ab).transpose(2, 0, 1)).min(1)


def msg_gap(A, B):
    """Max distance between equal-index waypoints valid in both (equal arc length from the same pose), nan if none."""
    m = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    return float(np.hypot(*(A[m] - B[m]).T).max()) if m.any() else np.nan


def build_one(u):
    import logging
    import ap2_route as AR
    import rmem as RM
    import rm_build as RB
    from alpasim_runtime.route_generator import RouteGeneratorMap, RouteGeneratorRecorded
    from rf_cf import route_cmd
    logging.disable(logging.WARNING)
    log, items = u
    fr, pos = AR.load_log(split_dir(log), log)
    vm, org = AR.city_map(fr.loc)
    res = []
    for it in items:
        tok, i = it["token"], pos.get(it["token"])
        if i is None:
            continue
        k = min(zlib.crc32(f"rm|{tok}".encode()) % K_MAX, i)
        H = spline(fr, i - k, i)
        if H is None:
            continue
        cl = it["cl"]
        a = np.r_[fr.p[i, :2] - org, fr.yaw[i]]
        C0 = RM.in_ego(a, cl - org)                                     # centreline in the row frame, 1 m samples from the ego's projection
        Ce = C0[1:]                                                     # the fake track continues 1 m beyond the projection
        tr = np.r_[H, np.c_[RM.to_local(a, Ce) + org, np.full(len(Ce), fr.p[i, 2])]]
        off = np.r_[org, tr[0, 2]]
        rec = dict(row=it["row"], token=tok, log=log, cls=it["cls"], k=k, dh=it["dh"], branch=it["branch"], ok=False)
        got = None
        for kind, Gen in (("map", RouteGeneratorMap), ("rec", RouteGeneratorRecorded)):  # Recorded = the same resampling on our own track
            try:
                g = Gen((tr - off).astype(np.float32), vm, route_start_offset_m=AR.OFFSET_M) if kind == "map" else \
                    Gen((tr - off).astype(np.float32), route_start_offset_m=AR.OFFSET_M)
                W = [AR.route_at(g, fr.p[i - k + j] - off, fr.yaw[i - k + j]) for j in range(k + 1)]
            except Exception:                                           # noqa: BLE001  generator sanity checks (fold-back, off-map)
                continue
            Le = RM.in_ego(a, np.asarray(g.route_polyline_in_local.points, float)[:, :2])
            poly, _ = arc_window(Le)
            dd = np.full(M_PT, np.inf)
            ahead = np.isfinite(poly[:, 0]) & (np.arange(M_PT) >= I0)
            if ahead.any():
                dd[ahead] = pl_dist(poly[ahead], C0)
            snap = float(dd[I0:I60 + 1].max())
            got = kind, W, poly, dd, snap
            if snap <= SNAP:
                break
        if got is None:
            res.append(rec)
            continue
        kind, W, poly, dd, snap = got
        B = [RM.to_local(np.r_[fr.p[i - k + j, :2] - org, fr.yaw[i - k + j]], w[np.isfinite(w).all(1)]) for j, w in enumerate(W) if np.isfinite(w).all(1).any()]
        f = RM.memory(B, a).astype(np.float32)
        bad = np.flatnonzero(dd[I60 + 1:] > SNAP)                       # beyond +60 m: cut the label where the generator leaves the lane-graph line
        if len(bad):
            poly[I60 + 1 + bad[0]:] = np.nan
        sl = np.nan_to_num(np.hypot(*np.diff(poly, axis=0).T)).astype(np.float32)
        fin = np.isfinite(poly[:, 0])
        rec.update(gen=kind, f=f, poly=poly, seglen=sl, cov20=bool(RM.summary(f)["c20"]), cmd=int(route_cmd(W[-1][None])[0]), snap=snap,
                   ok=bool(snap <= SNAP and fin.sum() >= 30 and len(B)), msg_k=W[-1], msg_0=W[0])
        # the real route (rm_build nav's log track) at the same poses: diagnostics and figures
        P = RB.track(fr, i - k)
        try:
            gr = RouteGeneratorMap((P - np.r_[org, P[0, 2]]).astype(np.float32), vm, route_start_offset_m=AR.OFFSET_M)
            offr = np.r_[org, P[0, 2]]
            Wr = [AR.route_at(gr, fr.p[i - k + j] - offr, fr.yaw[i - k + j]) for j in range(k + 1)]
            Br = [RM.to_local(np.r_[fr.p[i - k + j, :2] - org, fr.yaw[i - k + j]], w[np.isfinite(w).all(1)]) for j, w in enumerate(Wr) if np.isfinite(w).all(1).any()]
            fr_ = RM.memory(Br, a).reshape(RM.N_PT, RM.NF)
            fc = f.reshape(RM.N_PT, RM.NF)
            both = (fr_[:, 0] > 0.5) & (fc[:, 0] > 0.5)
            rec.update(f_real=fr_.reshape(-1), real=arc_window(RM.in_ego(a, np.asarray(gr.route_polyline_in_local.points, float)[:, :2]))[0], gap_k=msg_gap(W[-1], Wr[-1]),
                       gap_0=msg_gap(W[0], Wr[0]), gap_mem=float(np.hypot(*((fc[both, 1:3] - fr_[both, 1:3]) * 20).T).max()) if both.any() else np.nan)
            rec["ok"] = rec["ok"] and not rec["gap_mem"] < DIFF          # the real row's memory already shows this route (its track ended early)
        except Exception:                                               # noqa: BLE001  no real route: the navtrain row has no memory either
            pass
        res.append(rec)
    return res


def cmd_build(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("route_follow", "cfpool-build", config=vars(a)) as run:
        for data in a.data:
            dst = OUT / "rows" / f"{data}.npz"
            if dst.exists() and not a.force:
                run.info(f"{data}: exists")
                continue
            G = pickle.load(open(OUT / "geom" / f"{data}.pkl", "rb"))
            by = {}
            for r in G:
                by.setdefault(r["log"], []).append(r)
            res = par.pmap(build_one, sorted(by.items(), key=lambda x: -len(x[1]))[: a.limit or None], run=run, workers=a.workers or None, desc=f"build {data}")
            if res.errors:
                run.info(f"{data}: {res.summary()}")
            rows = [r for v in res.values if v for r in v]
            n = len(rows)
            nan = lambda *s: np.full((n,) + s, np.nan, np.float32)  # noqa: E731
            Z = dict(row=np.zeros(n, np.int32), token=np.array([r["token"] for r in rows]), log=np.array([r["log"] for r in rows]),
                     cls=np.array([r["cls"] for r in rows]), gen=np.array([r.get("gen", "") for r in rows]), k=np.array([r["k"] for r in rows], np.int8), ok=np.array([r["ok"] for r in rows]),
                     dh=np.array([r["dh"] for r in rows], np.float32), branch=np.array([r["branch"] for r in rows], np.float32),
                     f=np.zeros((n, 200), np.float16), f_real=np.zeros((n, 200), np.float16), poly=nan(M_PT, 2), seglen=np.zeros((n, M_PT - 1), np.float32), cov20=np.zeros(n, bool),
                     cmd=np.full(n, 3, np.int8), snap=nan(), real=nan(M_PT, 2), gap_k=nan(), gap_0=nan(), gap_mem=nan(), msg_k=nan(20, 2), msg_0=nan(20, 2))
            for m, r in enumerate(rows):
                Z["row"][m] = r["row"]
                for key in ("f", "f_real", "poly", "seglen", "cov20", "cmd", "snap", "real", "gap_k", "gap_0", "gap_mem", "msg_k", "msg_0"):
                    if key in r:
                        Z[key][m] = r[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, **Z)
            rec = {c: dict(n=int((Z["cls"] == c).sum()), ok=int((Z["ok"] & (Z["cls"] == c)).sum())) for c in ("j0", "j1", "lc")}
            rec["snap_med"] = float(np.nanmedian(Z["snap"][np.isfinite(Z["snap"])])) if np.isfinite(Z["snap"]).any() else None
            run.info(f"{data}: {n} rows {json.dumps(rec)} failed logs {len(res.errors)}")
            run.summary[data] = rec


# ---------------------------------------------------------------- pack (pool files, gates, figures)
def sdf_at(sdf, P):
    """Bilinear sdf (128, 96) at row-frame points P (n, 2) (cell centres x = -8 + 0.5 (i + 0.5), y = -24 + 0.5 (j + 0.5)); nan outside."""
    fi, fj = (P[:, 0] + 8.0) / 0.5 - 0.5, (P[:, 1] + 24.0) / 0.5 - 0.5
    ok = np.isfinite(fi) & (fi >= 0) & (fi <= 127) & (fj >= 0) & (fj <= 95)
    out = np.full(len(P), np.nan)
    i0, j0 = np.floor(fi[ok]).astype(int).clip(0, 126), np.floor(fj[ok]).astype(int).clip(0, 94)
    wi, wj = fi[ok] - i0, fj[ok] - j0
    s = sdf.astype(np.float32)
    out[ok] = (s[i0, j0] * (1 - wi) * (1 - wj) + s[i0 + 1, j0] * wi * (1 - wj) + s[i0, j0 + 1] * (1 - wi) * wj + s[i0 + 1, j0 + 1] * wi * wj)
    return out


def write_pool(name, Z, sel, datas, road, run):
    """Rows `sel` of the concatenated build rows Z (with Z['data'] the shard index) -> cache/on1_<name>{,@warp} + feat/on1_<name>.npz."""
    n = len(sel)
    pdir, wdir = CR / f"on1_{name}", CR / f"on1_{name}@warp"
    pdir.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    names = np.array([f"on:cf:{t}:{c}" for t, c in zip(Z["token"][sel], Z["cls"][sel])])
    assert len(set(names.tolist())) == n
    tabs = [dict(np.load(CR / d / "tab.npz")) for d in datas]
    keys = [k for k in tabs[0] if k not in ("names", "off")]
    tb = {k: np.zeros((n,) + tabs[0][k].shape[1:], tabs[0][k].dtype) for k in keys if k != "log"}
    tb["log"] = Z["log"][sel]
    front = np.lib.format.open_memmap(wdir / "front.npy", mode="w+", dtype=np.float16, shape=(n, 8, 32, 512))
    t_out, t_plan, di, pi = np.zeros((n, 1086), np.float32), np.zeros((n, 33, 15), np.float32), None, None
    for s, d in enumerate(datas):
        m = np.flatnonzero(Z["data"][sel] == s)
        if not len(m):
            continue
        src = Z["row"][sel][m]
        assert (tabs[s]["names"][src] == Z["token"][sel][m]).all()
        for k in keys:
            if k != "log":
                tb[k][m] = tabs[s][k][src]
        o = np.argsort(src)
        F = np.load(CR / f"{d}@warp" / "front.npy", mmap_mode="r")
        for c in range(0, len(m), 512):
            front[m[o[c:c + 512]]] = F[src[o[c:c + 512]]]
        T = np.load(CR / f"{d}@warp" / "teacher.npz")
        t_out[m], t_plan[m] = T["out"][src], T["plan"][src]
        if di is None:
            di, pi = T["di"], T["pi"]
        assert (T["di"] == di).all() and (T["pi"] == pi).all()
    front.flush()
    del front
    cmd = Z["cmd"][sel].astype(int)
    oh = np.zeros((n, 4), np.float32)
    oh[np.arange(n), cmd] = 1.0
    tb["ego"][:, 1:4] = oh[:, :3]                                       # parity_adapter.ego_features: cmd[..., :3], unknown -> all zero
    tb["cmd"][:, -1] = oh
    assert np.isfinite(tb["fut"]).all() and "off" not in tb
    np.savez(pdir / "tab.npz", names=names, **tb)
    np.savez(wdir / "teacher.npz", out=t_out, plan=t_plan, di=di, pi=pi)
    np.savez(pdir / "agent.npz", tokens=names, ok=np.zeros(n, bool), box=np.zeros((n, 9, 32, 5), np.float32), valid=np.zeros((n, 9, 32), bool))
    rpos = {t: j for j, t in enumerate(road["tokens"].tolist())}
    ri = np.array([rpos.get(t, -1) for t in Z["token"][sel].tolist()])
    sdf = np.zeros((n, 128, 96), np.float16)
    sdf[ri >= 0] = road["sdf"][ri[ri >= 0]]
    rok = (ri >= 0) & road["ok"][np.maximum(ri, 0)]
    np.savez(pdir / "road.npz", tokens=names, ok=rok, sdf=sdf)
    np.savez(pdir / "route.npz", tokens=names, ok=Z["ok"][sel], poly=Z["poly"][sel].astype(np.float32), seglen=Z["seglen"][sel].astype(np.float32),
             dz=np.full(n, DZ, np.float32), cov20=Z["cov20"][sel])
    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez(FEAT / f"on1_{name}.npz", names=names, f=Z["f"][sel].astype(np.float16), k=Z["k"][sel], ok=Z["ok"][sel])
    run.info(f"on1_{name}: {n} rows, {len(set(tb['log'].tolist()))} logs, road ok {rok.mean():.3f}, classes "
             f"{ {c: int((Z['cls'][sel] == c).sum()) for c in ('j0', 'j1', 'lc')} }, cmd LSRU {np.bincount(cmd, minlength=4).tolist()}")
    return names, sdf


def figure(path, Z, i, sdf, fut):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rmem as RM
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    yy, xx = np.meshgrid(-24 + 0.5 * (np.arange(96) + 0.5), -8 + 0.5 * (np.arange(128) + 0.5))
    ax.contourf(-yy, xx, sdf.astype(np.float32), levels=[0, 1e3], colors=["#dddddd"])
    ax.contour(-yy, xx, sdf.astype(np.float32), levels=[0], colors=["#888888"], linewidths=0.6)
    if np.isfinite(Z["real"][i]).any():
        ax.plot(-Z["real"][i][:, 1], Z["real"][i][:, 0], "-", color="#1f77b4", lw=2, label="real route (log track)")
    ax.plot(-Z["poly"][i][:, 1], Z["poly"][i][:, 0], "-", color="#d62728", lw=2, label="counterfactual label")
    m = Z["f"][i].astype(np.float32).reshape(RM.N_PT, RM.NF)
    v = m[:, 0] > 0.5
    ax.plot(-m[v, 2] * 20, m[v, 1] * 20, ".", color="#d62728", ms=4, label="cf memory (input)")
    ax.plot(np.r_[0, -fut[:, 1]], np.r_[0, fut[:, 0]], "o-", color="#2ca02c", ms=3, lw=1.2, label="logged future 4 s")
    ax.plot([0], [0], "k^", ms=8)
    ax.set_xlim(-24, 24), ax.set_ylim(-8, 56), ax.set_aspect("equal")
    ax.set_xlabel("-y (m, right +)"), ax.set_ylabel("x (m)")
    ax.set_title(f"{Z['token'][i][:10]} {Z['cls'][i]} k={Z['k'][i]} cmd={'LSRU'[Z['cmd'][i]]}\n"
                 f"gap@decision {Z['gap_k'][i]:.1f} m, dh {Z['dh'][i]:.0f} deg, branch {Z['branch'][i]:.0f} m", fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_pack(a):
    import shutil
    from jevdrive.data import splits
    from jevdrive.run import Run
    dvs = splits.load(DEV)
    with Run("route_follow", f"cfpool-pack-{a.name}", config=vars(a)) as run:
        run.use_split(dvs)
        zs = [dict(np.load(OUT / "rows" / f"{d}.npz")) for d in a.data]
        Z = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
        Z["data"] = np.concatenate([np.full(len(z["row"]), s) for s, z in enumerate(zs)])
        futs = [np.load(CR / d / "tab.npz")["fut"] for d in a.data]
        fut = np.stack([futs[s][r] for s, r in zip(Z["data"], Z["row"])]) if len(Z["row"]) else np.zeros((0, 8, 3))
        dev_logs = set()
        for d in a.data:
            t = np.load(CR / d / "tab.npz")
            dev_logs |= set(t["log"][dvs.mask(t["names"])].tolist())
        is_dev = np.isin(Z["log"], list(dev_logs))
        good = Z["ok"] & np.isfinite(fut).all((1, 2))
        run.info(f"build rows {len(good)}, ok {Z['ok'].sum()}, finite fut and ok {good.sum()}, dev-log rows {(good & is_dev).sum()}")
        road = np.load(ROAD)
        road = dict(tokens=road["tokens"], ok=road["ok"], sdf=road["sdf"])
        tr, dv = np.flatnonzero(good & ~is_dev), np.flatnonzero(good & is_dev)
        _, sdf_tr = write_pool(a.name, Z, tr, a.data, road, run)
        if len(dv):
            write_pool(f"{a.name}dev", Z, dv, a.data, road, run)
        # gates on the training pool
        rng = np.random.default_rng(0)
        g50 = rng.choice(len(tr), min(50, len(tr)), replace=False)
        gk, g0, gm = (Z[k][tr][g50] for k in ("gap_k", "gap_0", "gap_mem"))
        cls = Z["cls"][tr]
        inside, cnt = [], {}
        for m in range(len(tr)):
            s = sdf_at(sdf_tr[m], Z["poly"][tr[m]])
            s = s[np.isfinite(s)]
            inside.append(s > 0)
        ins = np.concatenate(inside) if inside else np.zeros(0, bool)
        for c in ("j0", "j1", "lc"):
            sel = np.flatnonzero(cls == c)
            cnt[c] = float(np.concatenate([inside[m] for m in sel]).mean()) if len(sel) else None
        fin = lambda x: x[np.isfinite(x)]  # noqa: E731
        G = {"rows": {"train": int(len(tr)), "dev": int(len(dv)), **{c: int((cls == c).sum()) for c in ("j0", "j1", "lc")},
                      "tokens": int(len(set(Z["token"][tr].tolist()))), "logs": int(len(set(Z["log"][tr].tolist())))},
             "build": {"rows": int(len(good)), "ok": int(Z["ok"].sum()), "ok_by_class": {c: [int((Z["ok"] & (Z["cls"] == c)).sum()), int((Z["cls"] == c).sum())]
                                                                                        for c in ("j0", "j1", "lc")}},
             "G_cf1": {"n": int(len(g50)), "median_gap_msg_at_decision": float(np.median(fin(gk))) if len(fin(gk)) else None,
                       "median_gap_first_msg_j0": float(np.median(fin(g0))) if len(fin(g0)) else None,
                       "median_gap_merged_memory": float(np.median(fin(gm))) if len(fin(gm)) else None,
                       "share_ge_2m_at_decision": float((fin(gk) >= 2).mean()) if len(fin(gk)) else None,
                       "all_rows_median_gap_at_decision": {c: float(np.nanmedian(Z["gap_k"][tr][cls == c])) if (cls == c).any() else None for c in ("j0", "j1", "lc")}},
             "G_cf2": {"samples": int(len(ins)), "inside": float(ins.mean()) if len(ins) else None, "by_class": cnt,
                       "rows_all_inside": float(np.mean([x.all() for x in inside])) if inside else None},
             "generator_by_class": {c: {gk: int(((cls == c) & (Z["gen"][tr] == gk)).sum()) for gk in ("map", "rec")} for c in ("j0", "j1", "lc")},
             "cov20": float(Z["cov20"][tr].mean()) if len(tr) else None, "cmd_LSRU": np.bincount(Z["cmd"][tr].astype(int), minlength=4).tolist(),
             "snap_p95": float(np.quantile(Z["snap"][tr], 0.95)) if len(tr) else None, "data": a.data}
        same = []                                                       # our real memory vs rm_build nav's stored row features (frame / rule check)
        for s, d in enumerate(a.data):
            fz = FEAT / f"{d}.npz"
            if fz.exists():
                z = np.load(fz)
                m = np.flatnonzero(Z["data"] == s)
                m = m[z["ok"][Z["row"][m]] & np.isfinite(Z["gap_mem"][m])]
                assert (z["names"][Z["row"][m]] == Z["token"][m]).all()
                same.append(np.abs(z["f"][Z["row"][m]].astype(np.float32) - Z["f_real"][m].astype(np.float32)).max(1))
        same = np.concatenate(same) if same else np.zeros(0)
        G["real_memory_vs_rm_build"] = dict(n=int(len(same)), share_le_1e2=float((same <= 1e-2).mean()) if len(same) else None,
                                            max=float(same.max()) if len(same) else None)
        G["G_cf1"]["pass"] = bool(G["G_cf1"]["median_gap_msg_at_decision"] is not None and G["G_cf1"]["median_gap_msg_at_decision"] >= 2.0)
        G["G_cf2"]["pass"] = bool(G["G_cf2"]["inside"] is not None and G["G_cf2"]["inside"] >= 0.95)
        figs = R / "experiments/route_follow/figs"
        figs.mkdir(parents=True, exist_ok=True)
        pick = [m for c in ("j0", "lc") for m in rng.permutation(np.flatnonzero(cls == c))[:3]]
        G["G_cf3"] = []
        for q, m in enumerate(pick):
            p = figs / f"cf_bev_{q}.png"
            figure(p, {k: v[tr] for k, v in Z.items()}, m, sdf_tr[m], fut[tr[m]])
            G["G_cf3"].append(dict(fig=str(p.relative_to(R)), token=str(Z["token"][tr[m]]), cls=str(cls[m])))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "gates.json").write_text(json.dumps(G, indent=1))
        (R / "experiments/route_follow/results").mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT / "gates.json", R / "experiments/route_follow/results/cf_gates.json")
        run.summary.update(G)
        print(json.dumps(G, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("geom", "build"):
        p = sub.add_parser(c)
        p.add_argument("--data", nargs="+", required=True), p.add_argument("--workers", type=int, default=0), p.add_argument("--limit", type=int, default=0)
        p.add_argument("--force", action="store_true")
    p = sub.add_parser("pack")
    p.add_argument("--data", nargs="+", required=True), p.add_argument("--name", default="CF")
    a = ap.parse_args()
    dict(geom=cmd_geom, build=cmd_build, pack=cmd_pack)[a.cmd](a)
