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
                                          source (ao_e1's way, decision 266), in decision order -> feat/on1_<name>.npz (names, f, k, route0_d)
  served --run DIR [--tag LABEL]          a read run whose driver logged its served memory (drive.jsonl `rmem`): the same offline rebuild at the
                                          logged anchors, per decision max |served - offline| over the points valid in either, mask agreement
                                          -> served/<label>.json (gate G-serve) and served/<label>.parquet
  cover  --runs LABEL=DIR ..              per decision of read runs with `rmem`: coverage summary (n valid, nearest valid offset, +20 m covered)
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
    split_dir, log, items, dev = u
    fr, pos = AR.load_log(split_dir, log)
    vm, org = AR.city_map(fr.loc)
    n = len(items)
    F = np.zeros((n, RM.DIM), np.float32)
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
        B = []
        try:
            for j in range(k + 1):
                f = i - k + j
                a = np.r_[fr.p[f, :2] - org, fr.yaw[f]]
                wp = AR.route_at(g, fr.p[f] - off, fr.yaw[f])
                wp = wp[np.isfinite(wp).all(1)]
                if len(wp):
                    B.append(RM.to_local(a, wp))
        except Exception:                                   # noqa: BLE001
            continue
        a = np.r_[fr.p[i, :2] - org, fr.yaw[i]]
        F[m], K[m], OK[m], NM[m] = RM.memory(B, a), k, True, len(B)
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


def regen(scene, recs, cfg):
    """One rollout's drive records (sorted by k) -> {k: (20, 2) rig-frame route waypoints (prepare_for_policy, NaN padded)}, {k: anchor}."""
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


def pool_one(job):
    run, scene, ks, cfg = job
    from ao_e1 import drive_records
    recs = drive_records(run)[scene]
    W, A = regen(scene, recs, cfg)
    F = feats_of(W, A)
    r0 = {r["k"]: (float(np.hypot(*(W[r["k"]][0] - np.asarray(r["route0"])))) if r.get("route0") is not None and np.isfinite(W[r["k"]][0]).all() else np.nan)
          for r in recs}
    return {k: (F[k], r0.get(k, np.nan), A[k]) for k in ks}


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
        for key, v in zip(keys, res.values):
            for i, k in by[key]:
                F[i], D[i], ok[i] = v[k][0], v[k][1], True
                dd = v[k][2] - np.asarray(m["anchor"][i], float)
                da[i] = max(np.abs(dd[:2]).max(), abs((dd[2] + np.pi) % (2 * np.pi) - np.pi))                  # the row's own frame = the drive record's anchor
        assert np.nanmax(da) < 1e-3, f"pool rows and drive records disagree on the anchor: {np.nanmax(da)}"
        np.savez(OUT / "feat" / f"on1_{a.name}.npz", names=names, f=F, k=m["k"], ok=ok, route0_d=D)
        sm = RM.summary(F.astype(np.float32))
        rec = dict(rows=len(names), ok=float(ok.mean()), route0_share_005=float((D[ok] <= 0.05).mean()), route0_med=float(np.nanmedian(D)),
                   c20=float(sm["c20"][ok].mean()), first_med=float(np.nanmedian(sm["first"][ok])))
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
    rows = []
    for r in recs:
        s = np.asarray(r["rmem"], np.float32).reshape(RM.N_PT, RM.NF)
        o = F[r["k"]].reshape(RM.N_PT, RM.NF)
        vs, vo = s[:, 0] > 0.5, o[:, 0] > 0.5
        both = vs & vo
        d = np.hypot(*((s[both, 1:3] - o[both, 1:3]) * 20.0).T) if both.any() else np.zeros(0)
        rows.append(dict(scene=scene, k=r["k"], n_served=int(vs.sum()), n_offline=int(vo.sum()), mask_same=bool((vs == vo).all()),
                         d_max=float(d.max()) if len(d) else 0.0, d_med=float(np.median(d)) if len(d) else 0.0))
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
    p = sub.add_parser("same")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--label", required=True)
    a = ap.parse_args()
    dict(same=cmd_same, nav=cmd_nav, yr=cmd_yr, pool=cmd_pool, served=cmd_served, cover=cmd_cover)[a.cmd](a)
