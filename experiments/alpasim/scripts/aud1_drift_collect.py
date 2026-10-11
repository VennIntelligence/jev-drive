#!/usr/bin/env python3
"""AUD1-drift: per-decision drift of the ego from the logged path, out of the STORED 1491-scene public rollouts (no rerun).

  aud1_drift_collect.py --out DIR [--jobs N]      (box, AlpaSim venv: numpy, polars, alpasim_runtime for the scene loader)

Sources per rollout (rollout.asl files are pruned on the box, so nothing here reads them):
  rollouts/<scene>/<rollout_id>/metrics.parquet   the evaluator's own per-step series at 2 Hz (t = 0, 0.017 s, then every 0.5 s to 5.017 s),
                                                  untruncated (RemoveTimestepsAfterEvent only acts at aggregation): dist_to_gt_trajectory,
                                                  lateral_dist_to_gt_trajectory
  controller/alpasim_controller_<rollout_id>.csv  the simulated ego rear-axle pose at 10 Hz in the rollout's local frame (matches the driver's
                                                  `anchor` and the logged EGO actor pose minus the rear-axle -> AABB-centre offset)
  aggregate/results-summary.json                  scene score and the per-rollout scored metrics (same as aud1_collect.py)
  scene loader (data_source.rig.trajectory)       the logged rear-axle path in the same local frame (identical to the asl's
                                                  ego_rig_recorded_ground_truth_trajectory, checked); privileged, used as a ruler only
Decision index k = 1..10: the ego state at t_k = 0.017 s + 0.5 s * k, i.e. after k executed decisions (k = 10 is the scene end).
Writes DIR/per_scene_k.parquet (driver, scene, k, ...) and DIR/per_scene.csv (driver, scene, summary metrics), DIR/coverage.json.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/alpasim"
T0, DT, K = 17000, 500000, 10
SUMM = ("dist_to_gt_trajectory", "dist_traveled_m", "gt_dist_traveled_m", "offroad_or_collision_at_fault", "collision_at_fault", "offroad",
        "left_corridor_laterally")


def dirs():
    o3 = json.loads((RA / "ot3/a/manifest.json").read_text())
    cf = json.loads((RA / "cf1/a/manifest.json").read_text())
    c0c = json.loads((RA / "c0c/manifest.json").read_text())
    d = {k: o3.get(k, []) + o3.get(k + "@new", []) + cf.get(k + "@new", []) + cf.get(k + "@fresh", [])
         for k in ("APY10m10-AB-s0", "APY10m10-AB-s1", "P2H10-F-s0", "P2H10-F-s1", "YR10m10-F-s0", "YR10m10-F-s1")}
    d |= {k: c0c[k] for k in ("SH30-F-s0", "AP2-AB-s0")}
    d["WA-JEPA"] = c0c["WA-JEPA (reference)"]
    return d


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def path_geom(g, p):
    """Logged rear-axle path g (n, 4: t, x, y, yaw) and a point p -> (signed lateral offset, left positive; tangent heading at the
    nearest point of the polyline). Densified to 0.1 m so the tangent is the local segment direction."""
    xy = g[:, 1:3]
    seg = np.diff(xy, axis=0)
    L = np.hypot(*seg.T)
    keep = L > 1e-3
    if not keep.any():   # logged car does not move: heading of the log itself
        return float(np.hypot(*(p - xy[0]))), float(g[0, 3])
    a, s = xy[:-1][keep], seg[keep]
    t = np.clip(((p - a) * s).sum(1) / (s * s).sum(1), 0, 1)
    q = a + t[:, None] * s
    d = np.hypot(*(p - q).T)
    i = int(d.argmin())
    h = float(np.arctan2(s[i, 1], s[i, 0]))
    n = np.array([-np.sin(h), np.cos(h)])
    return float((p - q[i]) @ n), h


def one(a):
    """(driver, run dir, scene, rollout_id, logged path) -> rows for k = 0..10."""
    drv, D, scene, rid, g = a
    import polars as pl
    m = pl.read_parquet(f"{D}/rollouts/{scene}/{rid}/metrics.parquet").select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    ser = {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list()))
           for n, x in ((n[0], x) for n, x in m.filter(pl.col("name").is_in(["dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory"]))
                        .group_by("name"))}
    cf = Path(D) / f"controller/alpasim_controller_{rid}.csv"
    c = np.loadtxt(cf, delimiter=",", skiprows=1, usecols=(0, 1, 2, 6, 7), ndmin=2) if cf.exists() else None
    rows = []
    for k in range(K + 1):
        t = T0 + DT * k
        r = dict(driver=drv, scene=scene, k=k, t_us=t, dist_gt=ser.get("dist_to_gt_trajectory", {}).get(t),
                 lat_gt=ser.get("lateral_dist_to_gt_trajectory", {}).get(t))
        if c is not None and g is not None:
            j = np.flatnonzero(c[:, 0] == t)
            if len(j):
                x, y, qz, qw = c[j[0], 1:5]
                yaw = 2 * np.arctan2(qz, qw)
                lat, h = path_geom(g, np.array([x, y]))
                hlog_t = np.interp(t, g[:, 0], np.unwrap(g[:, 3]))
                r |= dict(x=x, y=y, yaw=yaw, lat_signed=lat, head_err=float(np.degrees(wrap(yaw - h))),
                          head_err_time=float(np.degrees(wrap(yaw - hlog_t))))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=min(48, os.cpu_count() or 8))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = dirs()
    # scene -> rollout per driver, first run dir wins (aud1_collect.py's rule)
    pick, summ, cov = {}, [], {}
    for drv, ds in D.items():
        seen = {}
        for d in ds:
            S = json.loads((Path(d) / "aggregate/results-summary.json").read_text())["rollouts"]
            for r in S:
                s = r["clipgt_id"]
                if s in seen:
                    continue
                seen[s] = 1
                mm = r.get("metrics") or {}
                rid = glob.glob(f"{d}/rollouts/{s}/*/metrics.parquet")
                summ.append(dict(driver=drv, scene=s, run=d, score=r["score"], status=r.get("status"),
                                 rollout_id=rid[0].split("/")[-2] if rid else None, **{n: mm.get(n) for n in SUMM}))
                if rid:
                    pick.setdefault(drv, []).append((d, s, rid[0].split("/")[-2]))
        cov[drv] = dict(scenes=len(seen), with_parquet=len(pick.get(drv, [])), run_dirs=len(ds))
    scenes = sorted({s for v in pick.values() for _, s, _ in v})
    # logged rear-axle paths through the runtime's scene loader (one loader; the generated config of the first run dir)
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import build_scene_loader
    from omegaconf import OmegaConf
    first = next(iter(D.values()))[0]
    L = build_scene_loader(OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(f"{first}/generated-user-config-0.yaml")))
    G, miss = {}, []
    for s in scenes:
        try:
            tr = L.get_data_source(s).rig.trajectory
            G[s] = np.c_[np.asarray(tr.timestamps_us, np.float64), np.asarray(tr.positions)[:, :2], np.asarray(tr.yaws)]
        except Exception as e:  # noqa: BLE001
            miss.append((s, repr(e)[:200]))
    cov["_logged_paths"] = dict(scenes=len(scenes), loaded=len(G), missing=miss[:20])
    print(json.dumps(cov, indent=1), flush=True)
    jobs = [(drv, d, s, rid, G.get(s)) for drv, v in pick.items() for d, s, rid in v]
    import pandas as pd
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for rr in ex.map(one, jobs, chunksize=16) for r in rr]
    P = pd.DataFrame(rows)
    P.to_parquet(out / "per_scene_k.parquet", index=False)
    pd.DataFrame(summ).to_csv(out / "per_scene.csv", index=False)
    cov["_ctrl_rows_with_pose"] = {drv: int(x.yaw.notna().sum()) for drv, x in P.groupby("driver")}
    (out / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(json.dumps(cov["_ctrl_rows_with_pose"], indent=1), len(P), "rows")


if __name__ == "__main__":
    main()
