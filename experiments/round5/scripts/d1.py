#!/usr/bin/env python3
"""ROUND5 D1: where the remaining PAI points are and which a training-data change can reach (diagnosis, stored reads only, no GPU).

  extract  (box, CPU)  d1.py extract --read NAME=<read root> ... --out <dir>
           per read: every inference decision of the driver logs (runs/*/driver/drive.jsonl) -> <dir>/dec_<NAME>.pkl
           {scene: dict(t0, v0, pm (n, 8, 3) model plan, ps (n, 8, 3) served plan, v_set, v_model, won, cut, lead_d, lead_vl, lead_src)}
  mapcov   (box, alpasim env)  d1.py mapcov --cfg-glob '<roll>/*/runs/*/sim/generated-user-config-0.yaml' --out <csv> [--scenes <txt>]
           per NuRec scene: lanes / road areas in the scene's vector map (the evaluator's offroad surface), share of the human log's
           footprint inside it, and the time to build a 128 x 96 signed-distance raster (the nuPlan road.npz layout) at sampled log poses
  budget | offroad | launch | overshoot  (Mac)  the four D1 read-outs from the stored facts / rf_tax extracts / decision pickles
           -> experiments/round5/results/ (tables appended to d1_budget.md by `report`)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- extract (box)
def _one_log(f):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(f):
        if '"infer": true' not in line:
            continue
        r = json.loads(line)
        if r.get("kind") != "drive":
            continue
        fx = r.get("fix") or {}
        ac, ld = fx.get("acc") or {}, fx.get("lead") or {}
        pm = r.get("poses_model") or r["poses"]
        S = out[r["scene"]]
        S["t0"].append(int(r["t0"]))
        S["v0"].append(float(fx.get("v0", r["ego"][4] * 10)))
        S["pm"].append(np.asarray(pm, np.float32)[:8, :3])
        S["ps"].append(np.asarray(r["poses"], np.float32)[:8, :3])
        S["v_set"].append(float(ac.get("v_set") if ac.get("v_set") is not None else np.nan))
        S["v_model"].append(float(ac.get("v_model") if ac.get("v_model") is not None else np.nan))
        S["won"].append(str(ac.get("won", "")))
        S["cut"].append(float(ld.get("cut") or 0.0))
        S["lead_d"].append(float(ld["d"]) if ld.get("d") is not None else np.nan)
        S["lead_vl"].append(float(ld["vl"]) if ld.get("vl") is not None else np.nan)
        S["lead_src"].append(str(ld.get("src", "")))
    return {s: {k: np.asarray(v) for k, v in d.items()} for s, d in out.items()}


def cmd_extract(a):
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    for spec in a.read:
        name, root = spec.split("=", 1)
        fs = sorted(glob.glob(f"{root}/runs/*/driver/drive.jsonl")) or sorted(glob.glob(f"{root}/runs/*/sim/driver-logs/drive.jsonl"))
        with ProcessPoolExecutor(min(16, len(fs))) as ex:
            parts = list(ex.map(_one_log, fs))
        D = {}
        for p in parts:
            for s, d in p.items():
                if s in D:  # a scene in two chunks (should not happen): keep the longer
                    if len(d["t0"]) <= len(D[s]["t0"]):
                        continue
                D[s] = d
        for s, d in D.items():
            i = np.argsort(d["t0"], kind="stable")
            D[s] = {k: v[i] for k, v in d.items()}
        pickle.dump(D, open(o / f"dec_{name}.pkl", "wb"), protocol=4)
        print(name, len(fs), "logs", len(D), "scenes", sum(len(d["t0"]) for d in D.values()), "decisions", flush=True)


# ---------------------------------------------------------------- map coverage of NuRec scenes (box, alpasim env)
def _mapcov(job):
    scene, cfg, nras = job
    sys.path[:0] = [str(REPO / "experiments/route_follow/scripts")]
    import rf_tax as RT
    import shapely
    t = time.time()
    try:
        ds = RT.loader(cfg).get_data_source(scene)
    except Exception as e:  # noqa: BLE001
        return dict(scene=scene, err=f"load {type(e).__name__}: {str(e)[:80]}")
    t_load = time.time() - t
    tr = ds.rig.trajectory
    P, Y = np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
    vm = ds.map
    n_lanes = len(vm.lanes) if vm is not None else 0
    try:
        n_ra = len(vm.get_road_areas_within(np.r_[P.mean(0), 0.0], 500.0))
    except Exception:  # noqa: BLE001
        n_ra = -1
    if not n_lanes:
        return dict(scene=scene, n_lanes=0, n_road_areas=n_ra, t_load=t_load)
    t = time.time()
    M = RT.Map(vm, P)
    t_map = time.time() - t
    vc = ds.rig.vehicle_config
    rear, front, hw = float(vc.aabb_x_offset_m), float(vc.aabb_x_offset_m + vc.aabb_x_m), float(vc.aabb_y_m) / 2
    c, s = np.cos(Y), np.sin(Y)
    corners = [(front, hw), (front, -hw), (rear, -hw), (rear, hw)]
    fp = np.stack([np.c_[P[:, 0] + c * u - s * v, P[:, 1] + s * u + c * v] for u, v in corners], 1)  # (T, 4, 2)
    mg = M.margin(fp.reshape(-1, 2)).reshape(len(P), 4).min(1)
    # the nuPlan road.npz raster at nras log poses: cell centres x -8 + 0.5 (i + 0.5), y -24 + 0.5 (j + 0.5) in the pose frame
    gx, gy = np.meshgrid(-8 + 0.5 * (np.arange(128) + 0.5), -24 + 0.5 * (np.arange(96) + 0.5), indexing="ij")
    idx = np.linspace(0, len(P) - 1, nras).astype(int) if nras else []
    t = time.time()
    for i in idx:
        X = P[i, 0] + c[i] * gx - s[i] * gy
        Yw = P[i, 1] + s[i] * gx + c[i] * gy
        _ = M.margin(np.c_[X.ravel(), Yw.ravel()])
    t_ras = (time.time() - t) / max(len(idx), 1)
    return dict(scene=scene, n_lanes=n_lanes, n_road_areas=n_ra, area_m2=float(M.U.area), log_in=float((mg >= 0).mean()),
                log_margin_min=float(mg.min()), log_m=float(np.hypot(*np.diff(P, axis=0).T).sum()), t_load=t_load, t_map=t_map,
                t_raster=t_ras)


def cmd_mapcov(a):
    import pandas as pd
    import re
    want = {x.split()[0] for x in open(a.scenes) if x.strip() and not x.startswith("#")} if a.scenes else None
    cfg = {}
    for f in sorted(glob.glob(a.cfg_glob)):
        for s in re.findall(r"scene_id: (clipgt-[0-9a-f-]+)", open(f).read()):
            cfg.setdefault(s, f)
    sc = sorted(s for s in cfg if want is None or s in want)
    print(len(cfg), "scenes in the configs,", len(sc), "kept", flush=True)
    jobs = [(s, cfg[s], a.raster_sample if i % a.raster_every == 0 else 0) for i, s in enumerate(sc)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_mapcov, jobs, chunksize=1))
    D = pd.DataFrame(rows)
    D.to_csv(a.out, index=False)
    print(D.describe(include="all").T.to_string())


# ---------------------------------------------------------------- Mac read-outs live in d1_mac.py
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("extract")
    q.add_argument("--read", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q = sp.add_parser("mapcov")
    q.add_argument("--scenes", default="")
    q.add_argument("--cfg-glob", required=True, help="sim generated-user-config-0.yaml files whose scene lists cover the scenes")
    q.add_argument("--raster-every", type=int, default=10, help="time the raster on every n-th scene")
    q.add_argument("--out", required=True)
    q.add_argument("--raster-sample", type=int, default=20)
    q.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    dict(extract=cmd_extract, mapcov=cmd_mapcov)[a.cmd](a)


if __name__ == "__main__":
    main()
