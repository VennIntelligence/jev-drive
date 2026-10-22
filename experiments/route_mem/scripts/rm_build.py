#!/usr/bin/env python3
"""ROUTE-MEM row features (plans/2026-10-12-route-mem-prereg.md): the route memory (lib/rmem.py) of every training / dev row, rebuilt offline
with AlpaSim's own RouteGenerator, as the served driver would have held it. AlpaSim's venv, CPU only.

  nav    --data navtrain_full.s0of12 ..   NAVSIM rows as decision k of an 18-decision rollout that follows the log: k = crc32("rm|" + token) % 18
                                          (clipped to the frames before the token), the scene's route = RouteGeneratorMap on the log track from
                                          decision 0 over 9.5 s (20 frames at 2 Hz, cubic spline to 0.1 s, as ap2_route.py's 5.5 s track; fewer where
                                          the log ends), one route message per decision j = 0..k at the logged pose j, carried by the logged poses
                                          -> feat/<data>.npz: names, f (n, 200) f16, k, ok, n_msg, fut_d (n, 8) lateral distance of the logged
                                          future poses to the merged route (nan where not covered), and for the dev rows (navsim/op-parity-full-dev)
                                          jdist (m along the route to the first junction within 60 m, nan = none; ev.Geo's rule) and turn40 (deg)
  yr     --data navtrain_full.s0of12 ..   the yaw-rate rows (cache yr1_<data>: src_row, off) from nav's features, moved into each row's perturbed frame
  pool   --name L18YS                     on-policy rows ($DATA_DIR/runs/onpol_nuplan/rows/<name>/meta.npz): per rollout the route regenerated at every
                                          decision's driver anchor (drive.jsonl) by RouteGenerator.create (MAP, 40 m) on the scene's after160 data
                                          source (ao_e1's way, decision 266), in decision order -> feat/on1_<name>.npz (names, f, k, route0_d),
                                          and the route hinge's labels (experiments/onpol_nuplan/lib/on_route.py) -> cache/on1_<name>/route.npz:
                                          tokens = names, ok, poly (n, 51, 2) = the generator's full route polyline (route_polyline_in_local right
                                          after that decision's generate_route) in the row frame (the anchor), from 10 m behind the anchor's
                                          projection to 90 m ahead every 2 m, NaN outside the polyline; seglen (n, 50) (NaN where an end is NaN);
                                          dz 1.0; cov20 = the memory feature's +20 m point valid
  served --run DIR [--tag LABEL]          a read run whose driver logged its served memory (drive.jsonl `rmem`): the same offline rebuild at the
                                          logged anchors, per decision max |served - offline| over the points valid in either, mask agreement
                                          -> served/<label>.json (gate G-serve) and served/<label>.parquet
  cover  --runs LABEL=DIR ..              per decision of read runs with `rmem`: coverage summary (n valid, nearest valid offset, +20 m covered)
  order  --data navtrain_full.s0of12 .. [--pool L18RB0 ..] [--cf on1_CF on1_CFdev]
                                          route v2 (--rm-order): the ordered, progress-anchored memory (rmem.memory_o) of every row, rebuilt
                                          from the same route messages as `nav` / `pool` (NAVSIM rows: messages at frames i - k .. i, the
                                          newest at the row; pool rows: rm_build regen at the decision anchors; yr1_<data>: the source row's
                                          block moved into the perturbed frame and re-scored; --cf: rf_cfpool build --order's rows/<data>.o.npz)
                                          -> feat/<name>.o.npz (names, f (n, DIM_O) f16, ok, trav)
  anchor --data D .. [--pool P ..] [--cat NAME=SRC,..] [--merge NAME=POOL:prefix,..] [--limit N]
                                          the anchored layout (rmem.memory_a, rm_train --rm-anchor; results/anchor_check.md): the same
                                          messages as `nav` (frames i - k .. i at the logged poses, empty messages kept) and `pool` (regen
                                          at the decision anchors); yr1_<data> as in `order`; --cat: a pool whose rows (on:<token>) are rows
                                          of the sources (wt.py / df.py pack: every pool row's old-layout row must equal its source row);
                                          --merge: an rf_pait.py merge of pools (names and old-layout rows checked) -> feat/<name>.a.npz
                                          (names, f (n, DIM_O) f16, ok, trav); --limit: runs/route_mem/anchor_sample/ (a check only).
                                          Every file a base retrain on the anchored layout reads (box, AlpaSim venv, one CPU pool job):
      PYTHONHASHSEED=0 $AP rm_build.py anchor --force --workers 64 --data navtrain_full.s{0..11}of12 idx_waitcap_trA idx_waitcap_trB \
          idx_waitcap_trC idx_selfiter_trA idx_selfiter_trB idx_selfiter_trC --pool L18RB0 L18RTB1 \
          --cat on1_WAIT=idx_waitcap_trA,idx_waitcap_trB,idx_waitcap_trC on1_NR1TV=idx_selfiter_trA,idx_selfiter_trB,idx_selfiter_trC \
          --merge on1_L18NR1=on1_L18RB0:rb0,on1_L18RTB1:rtb1
                                          (navtrain shards + their yr1_ rows, the KD pool on1_WAIT, the trainval pool on1_NR1TV, the L18
                                          pools on1_L18RB0 / on1_L18RTB1 / on1_L18NR1). PYTHONHASHSEED: AlpaSim's map route search (the NAVSIM
                                          rows' generator) depends on Python's hash seed; 6 of 277 sampled rows change with it (results/
                                          anchor_check.md), so a fixed seed makes the build repeatable (pmap's forked workers share it)
  win    --data D .. [--pool P ..] [--cat NAME=SRC,..] [--merge NAME=POOL:prefix,..] [--limit N]
                                          the window layout (rmem.memory_w + compose_w, rm_train --rm-win; never trained, the 10-22 base
                                          retrain was dropped: round9 results/route_check): `anchor`'s messages, poses and arc, the slots at win_offs(v) with v
                                          = the row's fed speed (rmem.speed_of of the Store tab's ego column 4; yr1_ rows: their source row's,
                                          asserted equal), memory_a's scalars and the heading command of the fixed-grid block -> feat/<name>.w.npz
                                          (names, f (n, DIM_W) f16: the route part zeroed where not ok, the command kept, ok, trav, v, f0 (n, DIM)
                                          f16: the fixed-grid block, for the gate against feat/<name>.a.npz). Same command line as `anchor`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "lib"), str(R / "experiments/route_mem/lib"), str(R / "experiments/alpasim/scripts"), str(R / "experiments/alpasim/lib"),
                str(R / "experiments/anchor_offset/scripts"), str(R / "experiments/eyes_vs_head/scripts")]
import rmem as RM  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

OUT = data_dir() / "runs/route_mem"
CR = data_dir() / "runs/op_parity/cache"
K_MAX, N_TRACK = 18, 20
LAB = -10.0 + 2.0 * np.arange(51)                    # route label offsets (m) from the ego's projection (on_route.py, M = 51)
T0, DT = 17_000, 500_000


def seg_lat(M, q):
    """Distance of point q to polyline M where the nearest point is interior (else nan)."""
    d = M[1:] - M[:-1]
    t = ((q - M[:-1]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12)
    tc = np.clip(t, 0, 1)
    D = np.hypot(*(M[:-1] + tc[:, None] * d - q).T)
    i = int(np.argmin(D))
    inner = not ((i == 0 and t[i] < 0) or (i == len(d) - 1 and t[i] > 1))
    return float(D[i]) if inner else np.nan


def track(fr, i0: int):
    """Log track from frame i0 over up to N_TRACK frames (>= 2) at 0.1 s -> (m, 3) world positions, or None."""
    from scipy.interpolate import CubicSpline
    n = min(N_TRACK, len(fr.t) - i0)
    if i0 < 0 or n < 2:
        return None
    t = fr.t[i0:i0 + n] - fr.t[i0]
    if np.abs(np.diff(t) - 0.5).max() > 0.05:
        return None
    return CubicSpline(t, fr.p[i0:i0 + n])(np.arange(int(round(t[-1] / 0.1)) + 1) * 0.1)


def nav_one(u):
    """One log: (split dir, log, [(row, token)], dev tokens) -> per-row arrays."""
    import logging
    import ap2_route as AR
    from alpasim_runtime.route_generator import RouteGeneratorMap
    logging.disable(logging.WARNING)
    split_dir, log, items, dev = u[:4]
    order = len(u) > 4 and u[4]
    anc = len(u) > 5 and u[5]                               # the anchored layout (rmem.memory_a): every message and its pose, empty ones kept
    win = u[6] if len(u) > 6 else None                      # the window layout: {row: fed speed} (rmem.memory_w)
    fr, pos = AR.load_log(split_dir, log)
    vm, org = AR.city_map(fr.loc)
    n = len(items)
    F = np.zeros((n, RM.DIM), np.float32)
    FO, TV = np.zeros((n, RM.DIM_O), np.float32), np.zeros(n, np.float32)
    FW, F0 = np.zeros((n, RM.DIM), np.float32), np.zeros((n, RM.DIM), np.float32)
    K, OK, NM = np.zeros(n, np.int8), np.zeros(n, bool), np.zeros(n, np.int8)
    FD, JD, TU = np.full((n, 8), np.nan, np.float32), np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    geo = None
    for m, (_, tok) in enumerate(items):
        i = pos.get(tok)
        if i is None:
            continue
        k = min(zlib.crc32(f"rm|{tok}".encode()) % K_MAX, i)
        P = track(fr, i - k)
        if P is None:
            continue
        off = np.r_[org, P[0, 2]]
        try:
            g = RouteGeneratorMap((P - off).astype(np.float32), vm, route_start_offset_m=AR.OFFSET_M)
        except Exception:                                   # noqa: BLE001  (off-map / fold-back scenes fail in AlpaSim as well)
            continue
        B, last = [], None
        BA, PA = [], []
        try:
            for j in range(k + 1):
                f = i - k + j
                a = np.r_[fr.p[f, :2] - org, fr.yaw[f]]
                wp = AR.route_at(g, fr.p[f] - off, fr.yaw[f])
                wp = wp[np.isfinite(wp).all(1)]
                if len(wp):
                    B.append(RM.to_local(a, wp))
                    last = a
                if anc:
                    BA.append(RM.to_local(a, wp) if len(wp) else np.zeros((0, 2)))
                    PA.append(a)
        except Exception:                                   # noqa: BLE001
            continue
        a = np.r_[fr.p[i, :2] - org, fr.yaw[i]]
        F[m], K[m], OK[m], NM[m] = RM.memory(B, a), k, True, len(B)
        if order:                                           # route v2: the ordered memory (rmem.memory_o), the messages' poses = frames i - k .. i
            TV[m] = RM.travelled(fr.p[i - k:i + 1, :2])
            FO[m] = RM.memory_o(B, a, last if last is not None else a, TV[m])
        if anc:                                             # the anchored memory: messages at frames i - k .. i at their logged poses
            TV[m] = RM.travelled([p[:2] for p in PA], a)
            FO[m] = RM.memory_a(BA, PA, a)
            if win is not None:                             # the window layout: the same messages, the slots at the row's fed speed
                FW[m], F0[m], _ = RM.memory_w(BA, PA, a, win[items[m][0]])
        Mm = RM.merge(B)
        if Mm is not None and len(Mm) >= 2:
            Me = RM.in_ego(a, Mm)
            for t in range(8):                              # the logged future (frames i + 1 .. i + 8) against the merged memory
                if i + 1 + t < len(fr.t):
                    FD[m, t] = seg_lat(Me, RM.in_ego(a, (fr.p[i + 1 + t, :2] - org)[None])[0])
        if tok in dev:                                      # junction distance and turn along the route ahead of the ego (dev rows)
            import ev as EV
            from types import SimpleNamespace
            Rl = np.asarray(g.route_polyline_in_local.points, float)[:, :2]
            S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Rl, axis=0).T))]
            q = a[:2]
            d = Rl[1:] - Rl[:-1]
            tt = np.clip(((q - Rl[:-1]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
            j0 = int(np.argmin(np.hypot(*(Rl[:-1] + tt[:, None] * d - q).T)))
            s0 = S[j0] + tt[j0] * np.hypot(*d[j0])
            at = lambda s: np.array([np.interp(s, S, Rl[:, 0]), np.interp(s, S, Rl[:, 1])])  # noqa: E731
            hd = lambda s: np.arctan2(*(at(s + 1.0) - at(s - 1.0))[::-1])  # noqa: E731
            TU[m] = np.degrees(abs(EV.wrap(hd(s0 + 40.0) - hd(s0))))
            if geo is None:
                lo, hi = fr.p[:, :2].min(0) - org - 150, fr.p[:, :2].max(0) - org + 150
                geo = EV.Geo(SimpleNamespace(map=vm), (lo, hi))
            for s in np.arange(0.0, 60.01, 2.0):
                if s0 + s > S[-1]:
                    break
                p = at(s0 + s)
                if geo.junction(p[0], p[1]):
                    JD[m] = s
                    break
    if win is not None:
        return np.array([r for r, _ in items]), FW, F0, OK, TV
    if order or anc:
        return np.array([r for r, _ in items]), FO, OK, TV
    return np.array([r for r, _ in items]), F, K, OK, NM, FD, JD, TU


def cmd_nav(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    dvs = splits.load("navsim/op-parity-full-dev")
    with Run("route_mem", "build-nav", config=vars(a)) as run:
        run.use_split(dvs)
        for data in a.data:
            out = OUT / "feat" / f"{data}.npz"
            if out.exists() and not a.force:
                run.info(f"{data}: exists")
                continue
            tab = np.load(CR / data / "tab.npz")
            names, logs = tab["names"], tab["log"]
            dev = set(names[dvs.mask(names)].tolist())
            split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
            by = {}
            for r, (t, lg) in enumerate(zip(names.tolist(), logs.tolist())):
                by.setdefault(lg, []).append((r, t))
            jobs = [(split_dir, lg, it, dev & {t for _, t in it}) for lg, it in sorted(by.items(), key=lambda x: -len(x[1]))][: a.limit or None]
            res = par.pmap(nav_one, jobs, run=run, workers=a.workers or n_cpus(), desc=data)
            res.raise_if_failed()
            n = len(names)
            F = np.zeros((n, RM.DIM), np.float16)
            K, OK, NM = np.zeros(n, np.int8), np.zeros(n, bool), np.zeros(n, np.int8)
            FD, JD, TU = np.full((n, 8), np.nan, np.float32), np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
            for rows, f, k, ok, nm, fd, jd, tu in res.values:
                F[rows], K[rows], OK[rows], NM[rows], FD[rows], JD[rows], TU[rows] = f, k, ok, nm, fd, jd, tu
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, names=names, f=F, k=K, ok=OK, n_msg=NM, fut_d=FD, jdist=JD, turn40=TU, dev=dvs.mask(names))
            sm = RM.summary(F.astype(np.float32))
            rec = dict(rows=n, ok=float(OK.mean()), c20=float(sm["c20"][OK].mean()), first_med=float(np.nanmedian(sm["first"][OK])),
                       fut_d_med=float(np.nanmedian(FD)), fut_d_cov=float(np.isfinite(FD).mean()), dev_rows=int(dvs.mask(names).sum()))
            run.info(f"{data}: {json.dumps(rec)}")
            run.summary[data] = rec


def cmd_yr(a):
    from jevdrive.run import Run
    with Run("route_mem", "build-yr", config=vars(a)) as run:
        for data in a.data:
            z, y = np.load(OUT / "feat" / f"{data}.npz"), np.load(CR / f"yr1_{data}" / "tab.npz")
            assert (z["names"][y["src_row"]] == y["names"]).all()
            f = RM.transform_rows(z["f"][y["src_row"]].astype(np.float32), y["off"]).astype(np.float16)
            np.savez(OUT / "feat" / f"yr1_{data}.npz", names=y["names"], f=f, ok=z["ok"][y["src_row"]], k=z["k"][y["src_row"]])
            run.info(f"yr1_{data}: {len(f)} rows")


def label(Rl, a):
    """Full route polyline Rl (m, 2) in the local frame and the row's anchor -> (51, 2) row-frame points at LAB along the polyline from the
    anchor's projection (rmem.window's projection, extrapolated before / beyond the ends), NaN where the offset is off the polyline."""
    P = np.full((len(LAB), 2), np.nan, np.float32)
    M = RM.in_ego(a, np.asarray(Rl, np.float64)[:, :2])
    M = M[np.isfinite(M).all(1)]
    if len(M) < 2:
        return P
    d = np.diff(M, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 1e-6
    if not keep.any():
        return P
    M, d, L = np.vstack([M[:1], M[1:][keep]]), d[keep], L[keep]
    u = d / L[:, None]
    cum = np.r_[0.0, np.cumsum(L)]
    t = np.clip(-(M[:-1] * u).sum(1), 0.0, L)
    i = int(np.argmin(np.hypot(*(M[:-1] + t[:, None] * u).T)))
    s_e = cum[i] + t[i]
    if i == 0 and t[0] <= 0.0:
        s_e = float(-(M[0] * u[0]).sum())
    elif i == len(L) - 1 and t[i] >= L[i]:
        s_e = cum[-1] + float(-(M[-1] * u[-1]).sum())
    s = s_e + LAB
    ok = (s >= 0.0) & (s <= cum[-1])
    j = np.clip(np.searchsorted(cum, s[ok], side="right") - 1, 0, len(L) - 1)
    P[ok] = M[j] + (s[ok] - cum[j])[:, None] * u[j]
    return P


def regen(scene, recs, cfg, full=None):
    """One rollout's drive records (sorted by k) -> {k: (20, 2) rig-frame route waypoints (prepare_for_policy, NaN padded)}, {k: anchor};
    full (dict) if given gets {k: (m, 2) the generator's whole route polyline in the local frame right after decision k's generate_route}."""
    import logging
    from ao_e1 import loader
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    logging.disable(logging.WARNING)
    ds = loader(str(cfg)).get_data_source(scene)
    tr = ds.rig.trajectory
    P, Tt = np.asarray(tr.positions, float), np.asarray(tr.timestamps_us, float)
    g = RouteGenerator.create(P, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
    W, A = {}, {}
    for r in recs:
        x, y, yaw = r["anchor"]
        pose = Pose(np.array([x, y, np.interp(r["t0"], Tt, P[:, 2])]), np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]))
        W[r["k"]] = np.asarray(RouteGenerator.prepare_for_policy(g.generate_route(int(r["t0"]), pose)).waypoints, float)[:, :2]
        if full is not None:
            full[r["k"]] = np.asarray(g.route_polyline_in_local.points, float)[:, :2].copy()
        A[r["k"]] = np.asarray(r["anchor"], float)
    return W, A


def feats_of(W, A):
    """Regenerated messages and anchors -> {k: (DIM,) features at decision k} (the driver's Tracker on the same inputs)."""
    out, B = {}, []
    for k in sorted(W):
        wp = W[k][np.isfinite(W[k]).all(1)]
        if len(wp):
            B.append(RM.to_local(A[k], wp))
        out[k] = RM.memory(B, A[k]).astype(np.float32)
    return out


def feats_of_o(W, A):
    """feats_of for the ordered layout: {k: (DIM_O,)} (messages at the decision anchors: the newest one's pose = its anchor; trav = chords of
    the anchors), the served Tracker(order=True) on the same inputs."""
    out, B, last, Q = {}, [], None, []
    for k in sorted(W):
        wp = W[k][np.isfinite(W[k]).all(1)]
        Q.append(A[k][:2])
        if len(wp):
            B.append(RM.to_local(A[k], wp))
            last = A[k]
        out[k] = RM.memory_o(B, A[k], last if last is not None else A[k], RM.travelled(Q))
    return out


def feats_of_a(W, A):
    """feats_of for the anchored layout: {k: (DIM_O,)} (rmem.memory_a; messages at the decision anchors, empty ones kept), the served
    Tracker(anchor=True) on the same inputs."""
    out, B, P = {}, [], []
    for k in sorted(W):
        wp = W[k][np.isfinite(W[k]).all(1)]
        B.append(RM.to_local(A[k], wp) if len(wp) else np.zeros((0, 2)))
        P.append(A[k])
        out[k] = RM.memory_a(B, P, A[k])
    return out


def feats_of_w(W, A, V):
    """feats_of_a for the window layout: {k: (Fw, F0, trav)} at the decisions of V ({k: fed speed}), the served Tracker(anchor, win) on the
    same inputs."""
    out, B, P = {}, [], []
    for k in sorted(W):
        wp = W[k][np.isfinite(W[k]).all(1)]
        B.append(RM.to_local(A[k], wp) if len(wp) else np.zeros((0, 2)))
        P.append(A[k])
        if k in V:
            out[k] = RM.memory_w(B, P, A[k], V[k])
    return out


def pool_one(job):
    run, scene, ks, cfg = job[:4]
    from ao_e1 import drive_records
    recs = drive_records(run)[scene]
    Fu = {}
    W, A = regen(scene, recs, cfg, Fu)
    if len(job) > 6 and job[6] is not None:                 # the window layout only: job[6] = {k: fed speed}
        return feats_of_w(W, A, job[6])
    if len(job) > 5 and job[5]:                             # the anchored layout only
        F = feats_of_a(W, A)
        return {k: F[k] for k in ks}
    if len(job) > 4 and job[4]:                             # route v2 ordered layout only
        F = feats_of_o(W, A)
        return {k: F[k] for k in ks}
    F = feats_of(W, A)
    r0 = {r["k"]: (float(np.hypot(*(W[r["k"]][0] - np.asarray(r["route0"])))) if r.get("route0") is not None and np.isfinite(W[r["k"]][0]).all() else np.nan)
          for r in recs}
    return {k: (F[k], r0.get(k, np.nan), A[k], label(Fu[k], A[k])) for k in ks}


def cmd_pool(a):
    import ev as EV
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    m = np.load(data_dir() / "runs/onpol_nuplan/rows" / a.name / "meta.npz", allow_pickle=True)
    names = m["names"]
    with Run("route_mem", f"build-pool-{a.name}", config=vars(a)) as run:
        by = {}
        for i, (rn, sc, k) in enumerate(zip(m["run"].tolist(), m["scene"].tolist(), m["k"].tolist())):
            by.setdefault((rn, sc), []).append((i, int(k)))
        keys = sorted(by)[: a.limit or None]
        res = par.pmap(pool_one, [(rn, sc, [k for _, k in by[rn, sc]], EV.CFG) for rn, sc in keys], run=run, workers=a.workers or n_cpus(), desc="rollouts")
        res.raise_if_failed()
        F, D, ok = np.zeros((len(names), RM.DIM), np.float16), np.full(len(names), np.nan, np.float32), np.zeros(len(names), bool)
        da = np.full(len(names), np.nan)
        poly = np.full((len(names), len(LAB), 2), np.nan, np.float32)
        for key, v in zip(keys, res.values):
            for i, k in by[key]:
                F[i], D[i], ok[i], poly[i] = v[k][0], v[k][1], True, v[k][3]
                dd = v[k][2] - np.asarray(m["anchor"][i], float)
                da[i] = max(np.abs(dd[:2]).max(), abs((dd[2] + np.pi) % (2 * np.pi) - np.pi))                  # the row's own frame = the drive record's anchor
        assert np.nanmax(da) < 1e-3, f"pool rows and drive records disagree on the anchor: {np.nanmax(da)}"
        np.savez(OUT / "feat" / f"on1_{a.name}.npz", names=names, f=F, k=m["k"], ok=ok, route0_d=D)
        sm = RM.summary(F.astype(np.float32))
        seg = np.linalg.norm(np.diff(poly, axis=1), axis=-1).astype(np.float32)
        rok = ok & (np.isfinite(poly).all(-1).sum(1) >= 2)
        if (CR / f"on1_{a.name}").is_dir():                            # the pool dir (ol_chain strip) holds the trainer's files
            np.savez(CR / f"on1_{a.name}" / "route.npz", tokens=names, ok=rok, poly=poly, seglen=seg, dz=np.ones(len(names), np.float32),
                     cov20=np.asarray(sm["c20"], bool) & ok)
        fin = np.isfinite(poly[rok, :, 0])
        rec = dict(rows=len(names), ok=float(ok.mean()), route0_share_005=float((D[ok] <= 0.05).mean()), route0_med=float(np.nanmedian(D)),
                   c20=float(sm["c20"][ok].mean()), first_med=float(np.nanmedian(sm["first"][ok])), route_ok=float(rok.mean()),
                   route_pts_med=float(np.median(fin.sum(1))) if len(fin) else 0.0, route_dir=str(CR / f"on1_{a.name}"),
                   route_written=bool((CR / f"on1_{a.name}").is_dir()), route_lat0_med=float(np.nanmedian(np.abs(poly[rok, 5, 1]))) if rok.any() else np.nan)
        run.info(json.dumps(rec))
        run.summary.update(rec)


def served_one(job):
    run, scene = job
    from ao_e1 import drive_records
    import ev as EV
    recs = [r for r in drive_records(run)[scene] if "rmem" in r]
    if not recs:
        return []
    W, A = regen(scene, recs, EV.CFG)
    F = feats_of(W, A)
    FO = feats_of_o(W, A) if len(recs[0]["rmem"]) == RM.DIM_O else None
    rows = []
    for r in recs:
        s = np.asarray(r["rmem"], np.float32)
        o = F[r["k"]]
        sc = {}
        if FO is not None:                                  # route v2 ordered layout: rmem.memory_o on the regenerated messages
            o = FO[r["k"]]
            sc = dict(d_turn_served=float(s[-3]), d_turn_offline=float(o[-3]), trav_d=float(abs(s[-1] - o[-1]) * 100), dturn_same=bool(abs(s[-3] - o[-3]) < 1e-3 and s[-2] == o[-2]))
        s, o = RM.pts(s), RM.pts(o)
        vs, vo = s[:, 0] > 0.5, o[:, 0] > 0.5
        both = vs & vo
        d = np.hypot(*((s[both, 1:3] - o[both, 1:3]) * 20.0).T) if both.any() else np.zeros(0)
        rows.append(dict(scene=scene, k=r["k"], n_served=int(vs.sum()), n_offline=int(vo.sum()), mask_same=bool((vs == vo).all()),
                         d_max=float(d.max()) if len(d) else 0.0, d_med=float(np.median(d)) if len(d) else 0.0, **sc))
    return rows


def cmd_served(a):
    import pandas as pd
    from ao_e1 import drive_records
    from jevdrive import par
    from jevdrive.run import Run
    lab = a.tag or Path(a.run).parent.name
    with Run("route_mem", f"served-{lab}", config=vars(a)) as run:
        res = par.pmap(served_one, [(a.run, sc) for sc in sorted(drive_records(a.run))], run=run, workers=a.workers or 8, desc="scenes")
        res.raise_if_failed()
        D = pd.DataFrame([x for v in res.values for x in v])
        (OUT / "served").mkdir(parents=True, exist_ok=True)
        D.to_parquet(OUT / "served" / f"{lab}.parquet", index=False)
        g = dict(scenes=int(D.scene.nunique()), decisions=len(D), mask_same=float(D.mask_same.mean()), d_max_le_005=float((D.d_max <= 0.05).mean()),
                 d_max_max=float(D.d_max.max()), n_served_mean=float(D.n_served.mean()))
        g["pass"] = bool(g["mask_same"] >= 0.99 and g["d_max_le_005"] >= 0.99)
        (OUT / "served" / f"{lab}.json").write_text(json.dumps(g, indent=1))
        run.summary.update(g)
        run.info(json.dumps(g))


def write_order(name, names, f, ok, trav, run, suffix=".o", root=None):
    f = np.asarray(f, np.float32).copy()
    f[~np.asarray(ok, bool)] = 0
    (root or OUT / "feat").mkdir(parents=True, exist_ok=True)
    np.savez((root or OUT / "feat") / f"{name}{suffix}.npz", names=names, f=f.astype(np.float16), ok=np.asarray(ok, bool), trav=np.asarray(trav, np.float32))
    sc, P = f[ok, -3:], RM.pts(f[ok])
    rec = dict(rows=len(names), ok=float(np.mean(ok)), dturn_known=float(sc[:, 1].mean()) if len(sc) else None,
               dturn_lt60=float((sc[:, 1].astype(bool) & (sc[:, 0] < 1)).mean()) if len(sc) else None,
               trav_med=float(np.median(np.asarray(trav)[ok])) if np.any(ok) else None,
               near20_empty=float((~(P[:, RM.I0:RM.I20 + 1, 0] > 0.5).any(1)).mean()) if np.any(ok) else None)
    run.info(f"{name}{suffix}: {json.dumps(rec)}")
    run.summary[name if suffix == ".o" else name + suffix] = rec
    return rec


def cmd_order(a):
    """Route v2: the ordered (progress-anchored) memory of every row, rebuilt from the route messages as `nav` / `pool` build them."""
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    import ev as EV
    with Run("route_mem", "build-order", config=vars(a)) as run:
        for data in a.data:
            z = np.load(OUT / "feat" / f"{data}.npz")
            tab = np.load(CR / data / "tab.npz")
            names, logs = tab["names"], tab["log"]
            assert np.array_equal(z["names"], names)
            split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
            by = {}
            for r, (t, lg) in enumerate(zip(names.tolist(), logs.tolist())):
                by.setdefault(lg, []).append((r, t))
            jobs = [(split_dir, lg, it, set(), True) for lg, it in sorted(by.items(), key=lambda x: -len(x[1]))]
            res = par.pmap(nav_one, jobs, run=run, workers=a.workers or n_cpus(), desc=f"order {data}")
            res.raise_if_failed()
            FO, OK, TV = np.zeros((len(names), RM.DIM_O), np.float32), np.zeros(len(names), bool), np.zeros(len(names), np.float32)
            for rows, fo, ok, tv in res.values:
                FO[rows], OK[rows], TV[rows] = fo, ok, tv
            assert np.array_equal(OK, z["ok"]), f"{data}: ok differs from the nav build"
            write_order(data, names, FO, OK, TV, run)
            yz = OUT / "feat" / f"yr1_{data}.npz"
            if yz.exists():                                 # yaw-rate rows: the source row's ordered block moved into the perturbed frame, re-scored
                y, yt = np.load(yz), np.load(CR / f"yr1_{data}" / "tab.npz")
                assert np.array_equal(y["names"], yt["names"])
                blk = RM.transform_rows(FO[yt["src_row"], :RM.DIM], yt["off"])
                write_order(f"yr1_{data}", y["names"], RM.to_order(blk, TV[yt["src_row"]]), OK[yt["src_row"]], TV[yt["src_row"]], run)
        for name in a.pool:
            z = np.load(OUT / "feat" / f"on1_{name}.npz")
            m = np.load(data_dir() / "runs/onpol_nuplan/rows" / name / "meta.npz", allow_pickle=True)
            assert np.array_equal(m["names"], z["names"])
            by = {}
            for i, (rn, sc, k) in enumerate(zip(m["run"].tolist(), m["scene"].tolist(), m["k"].tolist())):
                by.setdefault((rn, sc), []).append((i, int(k)))
            keys = sorted(by)
            res = par.pmap(pool_one, [(rn, sc, [k for _, k in by[rn, sc]], EV.CFG, True) for rn, sc in keys], run=run, workers=a.workers or n_cpus(), desc="pool order")
            res.raise_if_failed()
            FO = np.zeros((len(z["names"]), RM.DIM_O), np.float32)
            for key, v in zip(keys, res.values):
                for i, k in by[key]:
                    FO[i] = v[k]
            write_order(f"on1_{name}", z["names"], FO, z["ok"], FO[:, -1] * 100.0, run)
        if a.cf:                                            # rf_cfpool build --order rows -> the pools' ordered features
            Zo = {}
            for f in sorted((data_dir() / "runs/route_follow/cf/rows").glob("*.o.npz")):
                t = np.load(f)
                Zo.update({(tk, c): (fo, tv) for tk, c, fo, tv in zip(t["token"].tolist(), t["cls"].tolist(), t["f"], t["trav"])})
            for name in a.cf:
                z = np.load(OUT / "feat" / f"{name}.npz")
                key = [tuple(n.split(":")[2:4]) for n in z["names"].tolist()]
                miss = [k for k in key if k not in Zo]
                assert not miss, f"{name}: {len(miss)} rows without an ordered cf row"
                write_order(name, z["names"], np.stack([Zo[k][0] for k in key]).astype(np.float32), z["ok"], np.array([Zo[k][1] for k in key], np.float32), run)


def cmd_anchor(a):
    """The anchored memory (rmem.memory_a) of every row (module docstring `anchor`), from the route messages `nav` / `pool` build them from;
    --limit N: the first N logs / rollouts only, written to runs/route_mem/anchor_sample/ (a check; never a training file)."""
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    import ev as EV
    root = OUT / ("anchor_sample" if a.limit else "feat")
    with Run("route_mem", "build-anchor", config=vars(a)) as run:
        for data in a.data:
            if (root / f"{data}.a.npz").exists() and not a.force:
                run.info(f"{data}.a: exists")
                continue
            z, tab = np.load(OUT / "feat" / f"{data}.npz"), np.load(CR / data / "tab.npz")
            names, logs = tab["names"], tab["log"]
            assert np.array_equal(z["names"], names)
            split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
            by = {}
            for r, (t, lg) in enumerate(zip(names.tolist(), logs.tolist())):
                by.setdefault(lg, []).append((r, t))
            jobs = [(split_dir, lg, it, set(), False, True) for lg, it in sorted(by.items(), key=lambda x: -len(x[1]))][: a.limit or None]
            res = par.pmap(nav_one, jobs, run=run, workers=a.workers or n_cpus(), desc=f"anchor {data}")
            res.raise_if_failed()
            FA, OK, TV = np.zeros((len(names), RM.DIM_O), np.float32), np.zeros(len(names), bool), np.zeros(len(names), np.float32)
            for rows, fa, ok, tv in res.values:
                FA[rows], OK[rows], TV[rows] = fa, ok, tv
            assert a.limit or np.array_equal(OK, z["ok"]), f"{data}: ok differs from the nav build"
            write_order(data, names, FA, OK, TV, run, ".a", root)
            yz = OUT / "feat" / f"yr1_{data}.npz"
            if yz.exists():                                 # yaw-rate rows: the source row's block moved into the perturbed frame, re-scored
                y, yt = np.load(yz), np.load(CR / f"yr1_{data}" / "tab.npz")
                assert np.array_equal(y["names"], yt["names"])
                blk = RM.transform_rows(FA[yt["src_row"], :RM.DIM], yt["off"])
                write_order(f"yr1_{data}", y["names"], RM.to_order(blk, TV[yt["src_row"]]), OK[yt["src_row"]], TV[yt["src_row"]], run, ".a", root)
        for name in a.pool:                                 # rm_build pool rows (the messages regenerated at the decision anchors)
            z = np.load(OUT / "feat" / f"on1_{name}.npz")
            m = np.load(data_dir() / "runs/onpol_nuplan/rows" / name / "meta.npz", allow_pickle=True)
            assert np.array_equal(m["names"], z["names"])
            by = {}
            for i, (rn, sc, k) in enumerate(zip(m["run"].tolist(), m["scene"].tolist(), m["k"].tolist())):
                by.setdefault((rn, sc), []).append((i, int(k)))
            keys = sorted(by)[: a.limit or None]
            res = par.pmap(pool_one, [(rn, sc, [k for _, k in by[rn, sc]], EV.CFG, False, True) for rn, sc in keys], run=run,
                           workers=a.workers or n_cpus(), desc=f"anchor on1_{name}")
            res.raise_if_failed()
            FA, ok = np.zeros((len(z["names"]), RM.DIM_O), np.float32), np.zeros(len(z["names"]), bool)
            for key, v in zip(keys, res.values):
                for i, k in by[key]:
                    FA[i], ok[i] = v[k], z["ok"][i]
            write_order(f"on1_{name}", z["names"], FA, ok, FA[:, -1] * 100.0, run, ".a", root)
        for spec in a.cat:                                  # NAME=SRC,..: a pool whose rows ("on:<token>") are rows of the sources (wt.py / df.py pack)
            name, srcs = spec.split("=", 1)
            z = np.load(OUT / "feat" / f"{name}.npz")
            at, F, TR = {}, {}, {}
            for src in srcs.split(","):
                za, zp = np.load(root / f"{src}.a.npz"), np.load(OUT / "feat" / f"{src}.npz")
                assert np.array_equal(za["names"], zp["names"]), src
                at.update({t: (src, i) for i, t in enumerate(zp["names"].astype(str).tolist())})
                F[src], TR[src] = (za["f"], za["ok"], zp["f"]), za["trav"]
            ix = [at.get(t.split(":")[-1]) for t in z["names"].astype(str).tolist()]
            assert all(x is not None for x in ix), f"{name}: {sum(x is None for x in ix)} rows without a source row"
            assert all(np.array_equal(F[s_][2][i], z["f"][r]) for r, (s_, i) in enumerate(ix)), f"{name}: a pool row != its source row (old layout)"
            write_order(name, z["names"], np.stack([F[s_][0][i] for s_, i in ix]), np.array([F[s_][1][i] for s_, i in ix]) & z["ok"],
                        np.array([TR[s_][i] for s_, i in ix]), run, ".a", root)
        for spec in a.merge:                                # NAME=POOL:prefix,..: an rf_pait.py merge (rows renamed on:<prefix>:<token>, in order)
            name, parts = spec.split("=", 1)
            P = [x.split(":") for x in parts.split(",")]
            z = np.load(OUT / "feat" / f"{name}.npz")
            Za, Zp = [np.load(root / f"{p}.a.npz") for p, _ in P], [np.load(OUT / "feat" / f"{p}.npz") for p, _ in P]
            ren = np.concatenate([[f"on:{pre}:{t[3:]}" for t in zp["names"].astype(str).tolist()] for zp, (_, pre) in zip(Zp, P)])
            assert np.array_equal(ren, z["names"].astype(str)) and np.array_equal(np.concatenate([zp["f"] for zp in Zp]), z["f"]), f"{name}: not the merge of {parts}"
            write_order(name, z["names"], np.concatenate([x["f"] for x in Za]), np.concatenate([x["ok"] for x in Za]),
                        np.concatenate([x["trav"] for x in Za]), run, ".a", root)


def write_win(name, names, FW, F0, ok, trav, v, run, root=None):
    """Window layout file (module docstring `win`): compose_w of the blocks; the route part zeroed where not ok (the command kept)."""
    ok = np.asarray(ok, bool)
    f = RM.compose_w(FW, F0, trav)
    f[~ok, :RM.DIM_O] = 0
    (root or OUT / "feat").mkdir(parents=True, exist_ok=True)
    np.savez((root or OUT / "feat") / f"{name}.w.npz", names=names, f=f.astype(np.float16), ok=ok, trav=np.asarray(trav, np.float32),
             v=np.asarray(v, np.float32), f0=np.asarray(F0, np.float16))
    P = RM.pts(f[ok, :RM.DIM_O])
    lab = np.where(f[:, RM.DIM_O:].max(1) > 0.5, f[:, RM.DIM_O:].argmax(1), 3)
    rec = dict(rows=len(names), ok=float(ok.mean()), v_med=float(np.median(v)) if len(v) else None,
               near_first_empty=float((~(P[:, :, 0] > 0.5)[:, :N_NEAR]).all(1).mean()) if ok.any() else None,
               cmd_lr=float(np.isin(lab, [0, 2]).mean()), cmd_unknown=float((lab == 3).mean()))
    run.info(f"{name}.w: {json.dumps(rec)}")
    run.summary[name + ".w"] = rec
    return rec


N_NEAR = 10                                                 # write_win's summary: the first 10 slots (the near third of the window)


def cmd_win(a):
    """The window layout (rmem.memory_w + compose_w) of every row (module docstring `win`): `anchor`'s messages, the row's fed speed."""
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    import ev as EV
    root = OUT / ("win_sample" if a.limit else "feat")
    spd = lambda d: np.array([RM.speed_of(x) for x in np.load(CR / d / "tab.npz")["ego"][:, 4]], np.float64)   # noqa: E731
    with Run("route_mem", "build-win", config=vars(a)) as run:
        for data in a.data:
            if (root / f"{data}.w.npz").exists() and not a.force:
                run.info(f"{data}.w: exists")
                continue
            z, tab = np.load(OUT / "feat" / f"{data}.npz"), np.load(CR / data / "tab.npz")
            names, logs = tab["names"], tab["log"]
            assert np.array_equal(z["names"], names)
            V = spd(data)
            split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
            by = {}
            for r, (t, lg) in enumerate(zip(names.tolist(), logs.tolist())):
                by.setdefault(lg, []).append((r, t))
            jobs = [(split_dir, lg, it, set(), False, True, {r: V[r] for r, _ in it}) for lg, it in sorted(by.items(), key=lambda x: -len(x[1]))][: a.limit or None]
            res = par.pmap(nav_one, jobs, run=run, workers=a.workers or n_cpus(), desc=f"win {data}")
            res.raise_if_failed()
            n = len(names)
            FW, F0, OK, TV = np.zeros((n, RM.DIM), np.float32), np.zeros((n, RM.DIM), np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
            for rows, fw, f0, ok, tv in res.values:
                FW[rows], F0[rows], OK[rows], TV[rows] = fw, f0, ok, tv
            assert a.limit or np.array_equal(OK, z["ok"]), f"{data}: ok differs from the nav build"
            write_win(data, names, FW, F0, OK, TV, V, run, root)
            yz = OUT / "feat" / f"yr1_{data}.npz"
            if yz.exists():                                 # yaw-rate rows: the source row's blocks moved into the perturbed frame, re-scored
                y, yt = np.load(yz), np.load(CR / f"yr1_{data}" / "tab.npz")
                assert np.array_equal(y["names"], yt["names"])
                sr = yt["src_row"]
                assert np.array_equal(yt["ego"][:, 4], tab["ego"][sr, 4]), f"yr1_{data}: fed speed differs from the source row's"
                write_win(f"yr1_{data}", y["names"], RM.transform_rows(FW[sr], yt["off"]), RM.transform_rows(F0[sr], yt["off"]), OK[sr], TV[sr],
                          V[sr], run, root)
        for name in a.pool:                                 # rm_build pool rows (the messages regenerated at the decision anchors)
            z = np.load(OUT / "feat" / f"on1_{name}.npz")
            m = np.load(data_dir() / "runs/onpol_nuplan/rows" / name / "meta.npz", allow_pickle=True)
            assert np.array_equal(m["names"], z["names"])
            V = spd(f"on1_{name}")
            assert len(V) == len(z["names"])
            by = {}
            for i, (rn, sc, k) in enumerate(zip(m["run"].tolist(), m["scene"].tolist(), m["k"].tolist())):
                by.setdefault((rn, sc), []).append((i, int(k)))
            keys = sorted(by)[: a.limit or None]
            res = par.pmap(pool_one, [(rn, sc, [k for _, k in by[rn, sc]], EV.CFG, False, False, {k: V[i] for i, k in by[rn, sc]}) for rn, sc in keys],
                           run=run, workers=a.workers or n_cpus(), desc=f"win on1_{name}")
            res.raise_if_failed()
            n = len(z["names"])
            FW, F0, ok, TV = np.zeros((n, RM.DIM), np.float32), np.zeros((n, RM.DIM), np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
            for key, v in zip(keys, res.values):
                for i, k in by[key]:
                    FW[i], F0[i], TV[i] = v[k]
                    ok[i] = z["ok"][i]
            write_win(f"on1_{name}", z["names"], FW, F0, ok, TV, V, run, root)
        for spec in a.cat:                                  # NAME=SRC,..: a pool whose rows ("on:<token>") are rows of the sources (wt.py / df.py pack)
            name, srcs = spec.split("=", 1)
            z = np.load(OUT / "feat" / f"{name}.npz")
            Vp = spd(name)
            at, F = {}, {}
            for src in srcs.split(","):
                zw, zp = np.load(root / f"{src}.w.npz"), np.load(OUT / "feat" / f"{src}.npz")
                assert np.array_equal(zw["names"], zp["names"]), src
                at.update({t: (src, i) for i, t in enumerate(zp["names"].astype(str).tolist())})
                F[src] = (zw, zp["f"])
            ix = [at.get(t.split(":")[-1]) for t in z["names"].astype(str).tolist()]
            assert all(x is not None for x in ix), f"{name}: {sum(x is None for x in ix)} rows without a source row"
            assert all(np.array_equal(F[s_][1][i], z["f"][r]) for r, (s_, i) in enumerate(ix)), f"{name}: a pool row != its source row (old layout)"
            g = lambda k: np.stack([F[s_][0][k][i] for s_, i in ix])  # noqa: E731
            assert np.array_equal(g("v").astype(np.float32), Vp.astype(np.float32)), f"{name}: fed speed differs from the source rows'"
            ok = g("ok") & z["ok"]
            f = g("f")
            f[~ok, :RM.DIM_O] = 0
            np.savez(root / f"{name}.w.npz", names=z["names"], f=f, ok=ok, trav=g("trav"), v=g("v"), f0=g("f0"))
            run.info(f"{name}.w: {len(f)} rows from {srcs}")
        for spec in a.merge:                                # NAME=POOL:prefix,..: an rf_pait.py merge (rows renamed on:<prefix>:<token>, in order)
            name, parts = spec.split("=", 1)
            P = [x.split(":") for x in parts.split(",")]
            z = np.load(OUT / "feat" / f"{name}.npz")
            Zw, Zp = [np.load(root / f"{p}.w.npz") for p, _ in P], [np.load(OUT / "feat" / f"{p}.npz") for p, _ in P]
            ren = np.concatenate([[f"on:{pre}:{t[3:]}" for t in zp["names"].astype(str).tolist()] for zp, (_, pre) in zip(Zp, P)])
            assert np.array_equal(ren, z["names"].astype(str)) and np.array_equal(np.concatenate([zp["f"] for zp in Zp]), z["f"]), f"{name}: not the merge of {parts}"
            np.savez(root / f"{name}.w.npz", names=z["names"], **{k: np.concatenate([x[k] for x in Zw]) for k in ("f", "ok", "trav", "v", "f0")})
            run.info(f"{name}.w: merge of {parts}")


def cmd_cover(a):
    import pandas as pd
    from ao_e1 import drive_records
    rows = []
    for spec in a.runs:
        lab, dirs = spec.split("=", 1)
        for d in dirs.split(","):
            for sc, rr in drive_records(d).items():
                for r in rr:
                    if "rmem" in r:
                        s = RM.summary(np.asarray(r["rmem"], np.float32))
                        rows.append(dict(label=lab, scene=sc, k=r["k"], n=int(s["n"]), first=float(s["first"]), c20=bool(s["c20"])))
    D = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(a.out, index=False)
    print(D.groupby("label").agg(decisions=("k", "size"), c20=("c20", "mean"), n=("n", "mean")).to_string())


def cmd_same(a):
    """Plans of two runs on the same scenes (drive.jsonl poses, 1e-4 m): share of decisions whose plan points within 4 s differ by <= 0.03 m
    (docs/long-runs.md: fp16 plan equality) -> printed and written to served/same_<label>.json."""
    from ao_e1 import drive_records
    A, B = drive_records(a.a), drive_records(a.b)
    d = []
    for sc in sorted(set(A) & set(B)):
        pb = {r["k"]: np.asarray(r["poses"], float) for r in B[sc]}
        for r in A[sc]:
            if r["k"] in pb:
                d.append(float(np.abs(np.asarray(r["poses"], float)[:, :2] - pb[r["k"]][:, :2]).max()))
    d = np.array(d)
    g = dict(decisions=len(d), share_le_003=float((d <= 0.03).mean()), max=float(d.max()), median=float(np.median(d)))
    (OUT / "served").mkdir(parents=True, exist_ok=True)
    (OUT / "served" / f"same_{a.label}.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("nav")
    p.add_argument("--data", nargs="+", required=True), p.add_argument("--workers", type=int, default=0), p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("yr")
    p.add_argument("--data", nargs="+", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--name", required=True), p.add_argument("--workers", type=int, default=0), p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("served")
    p.add_argument("--run", required=True), p.add_argument("--tag", default=""), p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("cover")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--out", required=True)
    p = sub.add_parser("order")
    p.add_argument("--data", nargs="*", default=[]), p.add_argument("--pool", nargs="*", default=[]), p.add_argument("--cf", nargs="*", default=[])  # noqa: E501
    p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("same")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--label", required=True)
    p = sub.add_parser("anchor")
    p.add_argument("--data", nargs="*", default=[]), p.add_argument("--pool", nargs="*", default=[]), p.add_argument("--cat", nargs="*", default=[])
    p.add_argument("--merge", nargs="*", default=[]), p.add_argument("--workers", type=int, default=0), p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="first N logs / rollouts, written to runs/route_mem/anchor_sample (a check)")
    p = sub.add_parser("win")
    p.add_argument("--data", nargs="*", default=[]), p.add_argument("--pool", nargs="*", default=[]), p.add_argument("--cat", nargs="*", default=[])
    p.add_argument("--merge", nargs="*", default=[]), p.add_argument("--workers", type=int, default=0), p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="first N logs / rollouts, written to runs/route_mem/win_sample (a check)")
    a = ap.parse_args()
    dict(same=cmd_same, order=cmd_order, nav=cmd_nav, yr=cmd_yr, pool=cmd_pool, served=cmd_served, cover=cmd_cover, anchor=cmd_anchor,
         win=cmd_win)[a.cmd](a)
