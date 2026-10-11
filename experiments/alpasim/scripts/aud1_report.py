#!/usr/bin/env python3
"""AUD1 read-out from aud1_collect.py's per_scene.csv (any python with numpy + pandas):

  aud1_report.py <aud1 dir> [calib]   (calib: only task2_calib_by_vehicle.csv)
  aud1_report.py <aud1 dir>     -> <dir>/task1_runlevel.csv, task1_ci.csv, task1_by_zero.csv, task2_calib_by_vehicle.csv,
                                   task3_strata.csv, and the markdown tables on stdout
Run-level metrics follow src/eval/aggregation/processing.py (see aud1_collect.py); CIs are a log-clustered bootstrap (10 000 draws).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DRV = ["APY10m10-AB-s0", "P2H10-F-s0", "WA-JEPA"]
B = 10000


def runlevel(d, k):
    ok = d[f"{k}|dist_traveled_m"].notna()
    x = d[ok]
    dt, af = x[f"{k}|dist_traveled_m"].sum(), x[f"{k}|offroad_or_collision_at_fault"].sum()
    return dict(score=d[f"{k}|score"].mean(), at_fault_km=dt / af / 1000 if af else np.inf, gt_m=x[f"{k}|dist_to_gt_trajectory"].mean())


def boot(d, k, rng):
    logs = d.log.unique()
    g = {l: d.index[d.log == l].values for l in logs}
    out = []
    for _ in range(B):
        idx = np.concatenate([g[l] for l in rng.choice(logs, len(logs))])
        out.append(list(runlevel(d.loc[idx], k).values()))
    return np.percentile(np.array(out), [2.5, 97.5], 0)


def md(df, fl=3):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(":--" if df[c].dtype == object else "--:" for c in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(f"{v:.{fl}f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(L)


def main():
    o = Path(sys.argv[1])
    d = pd.read_csv(o / "per_scene.csv")
    rng = np.random.default_rng(0)
    # task 1
    rows = []
    for k in DRV:
        r = runlevel(d, k)
        ci = boot(d, k, rng)
        ok = d[f"{k}|dist_traveled_m"].notna()
        g = d.loc[ok, f"{k}|dist_to_gt_trajectory"]
        rows.append({"driver": k, "avg scene score": r["score"], "score CI": f"[{ci[0, 0]:.3f}, {ci[1, 0]:.3f}]",
                     "at-fault dist km": r["at_fault_km"], "km CI": f"[{ci[0, 1]:.3f}, {ci[1, 1]:.3f}]", "GT dist m": r["gt_m"],
                     "GT CI": f"[{ci[0, 2]:.3f}, {ci[1, 2]:.3f}]", "GT median m": g.median(), "share GT >= 4 m": (g >= 4).mean(),
                     "at-fault events": int(d.loc[ok, f"{k}|offroad_or_collision_at_fault"].sum()), "clips": int(ok.sum()),
                     "mean dist travelled m": d.loc[ok, f"{k}|dist_traveled_m"].mean()})
    t1 = pd.DataFrame(rows)
    t1.to_csv(o / "task1_runlevel.csv", index=False)
    print(md(t1))
    # per-scene GT distance by outcome, and the share of far scenes that would give the private 2.288 m
    rows = []
    for k in DRV:
        ok = d[f"{k}|dist_traveled_m"].notna()
        x = d[ok]
        g, z = x[f"{k}|dist_to_gt_trajectory"], x[f"{k}|score"] == 0
        far = g >= 4
        lo, hi = g[~far].mean(), g[far].mean()
        rows.append({"driver": k, "GT m, score > 0": g[~z].mean(), "GT m, score 0": g[z].mean(), "GT m, scenes < 4 m": lo,
                     "GT m, scenes >= 4 m": hi, "local share >= 4 m": far.mean(), "share >= 4 m giving 2.288": (2.288 - lo) / (hi - lo),
                     "at-fault share local": x[f"{k}|offroad_or_collision_at_fault"].mean(),
                     "at-fault share giving 0.132 km at local dist/clip": x[f"{k}|dist_traveled_m"].mean() / 132.0})
    t1b = pd.DataFrame(rows)
    t1b.to_csv(o / "task1_by_outcome.csv", index=False)
    print("\n" + md(t1b))
    # task 3 strata
    d["v0 bin"] = pd.cut(d.v0, [-1, 1, 5, 10, 99], labels=["<1", "1-5", "5-10", ">10"]).astype(str)
    d["command"] = d.cmd_ap2.map({0: "left", 1: "straight", 2: "right"}).fillna("none")
    ay = d.route0_y.abs()
    d["route0 |y|"] = pd.cut(ay, [-1, 2, 10, 99], labels=["<2 m", "2-10 m", ">10 m"]).astype(str)
    d["route0 x"] = pd.cut(d.route0_x, [-99, 0, 20, 35, 99], labels=["behind (<0)", "beside / near (0-20)", "20-35", ">=35"]).astype(str)
    d["route waypoints"] = np.where(d.n_wp0 >= 10, "10", np.where(d.n_wp0.isna(), "none", "<10"))
    d["log path m"] = pd.cut(d["APY10m10-AB-s0|gt_dist_traveled_m"], [-1, 5, 20, 40, 60, 999], labels=["<5", "5-20", "20-40", "40-60", ">60"]).astype(str)
    d["a0x bin"] = pd.cut(d.a0x, [-99, -1, 1, 99], labels=["braking <-1", "-1..1", "accel >1"]).astype(str)
    d["fy/fx"] = pd.cut(d.fy / d.fx, [0, .9, .97, 1.03, 9], labels=["<0.90", "0.90-0.97", "0.97-1.03", ">1.03"]).astype(str)
    d["vehicle "] = d.vehicle
    rows = []
    for s in ["v0 bin", "command", "route0 |y|", "route0 x", "route waypoints", "log path m", "a0x bin", "fy/fx", "vehicle "]:
        for lev, x in d.groupby(s):
            r = {"stratum": s.strip(), "level": lev, "n": len(x)}
            for k in DRV:
                r[f"{k} mean"] = x[f"{k}|score"].mean()
                r[f"{k} zeros"] = int((x[f"{k}|score"] == 0).sum())
            r["APY - WA-JEPA"] = r["APY10m10-AB-s0 mean"] - r["WA-JEPA mean"]
            rows.append(r)
    t3 = pd.DataFrame(rows)
    t3.to_csv(o / "task3_strata.csv", index=False)
    print("\n" + md(t3))


if __name__ == "__main__" and len(sys.argv) == 2:
    main()


def calib_by_vehicle(o):
    """Task 2: CAM_F0 calibration spread per vehicle (driver-logged K, R, t of every scene that started a session)."""
    d = pd.read_csv(o / "per_scene.csv").dropna(subset=["fx"])
    R = np.stack(d.R.map(json.loads).map(np.array))
    t = np.stack(d.t.map(json.loads).map(np.array))
    key = [np.round(np.r_[r.ravel(), fx, fy, cx, cy], 4).tobytes() for r, fx, fy, cx, cy in zip(R, d.fx, d.fy, d.cx, d.cy)]
    d = d.assign(key=key, yaw=np.degrees(np.arctan2(R[:, 1, 2], R[:, 0, 2])), pitch=np.degrees(np.arcsin(-R[:, 2, 2])),
                 tx=t[:, 0], tz=t[:, 2], fyfx=d.fy / d.fx, rkey=[np.round(r, 3).tobytes() for r in R])
    g = d.groupby("vehicle").agg(scenes=("scene", "size"), logs=("log", "nunique"), calib_keys=("key", "nunique"), rotations_3dp=("rkey", "nunique"),
                                 fx_min=("fx", "min"), fx_max=("fx", "max"), fy_min=("fy", "min"), fy_max=("fy", "max"), yaw_min=("yaw", "min"),
                                 yaw_max=("yaw", "max"), pitch_min=("pitch", "min"), pitch_max=("pitch", "max"), tx_min=("tx", "min"), tx_max=("tx", "max"),
                                 tz_min=("tz", "min"), tz_max=("tz", "max")).sort_values("scenes", ascending=False).reset_index()
    g.to_csv(o / "task2_calib_by_vehicle.csv", index=False)
    print(f"\nscenes {len(d)}, calib keys {d.key.nunique()}, rotations (3 dp) {d.rkey.nunique()}, vehicles {d.vehicle.nunique()}, logs {d.log.nunique()}")
    print(md(g, 2))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "calib":
    calib_by_vehicle(Path(sys.argv[1]))
