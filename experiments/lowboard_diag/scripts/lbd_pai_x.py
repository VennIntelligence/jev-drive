#!/usr/bin/env python3
"""LOWDIAG, PAI half, extraction (plans/2026-10-10-lowdiag-prereg.md). Reads finished PAI stacks on the GPU box, writes compact pickles.
Runs with AlpaSim's own venv (the .asl reader and the scene loader live there); reads only:

  cd $DATA_DIR/third_party/alpasim && PYTHONPATH=src/utils .venv/bin/python <this file> --stacks <dir>/ab_*_s0 ... --out <dir> [--jobs N]

Per stack -> <out>/<stack name>/
  lite.pkl   every rollout of the stack (pruned ones too): summary row, the scorer's per-step metric series (metrics.parquet), the
             controller trace, the driver's own drive.jsonl records (served plan `poses`, the adapter's plan `poses_model`, `fix`, command)
  logs.pkl   rollouts whose rollout.asl was kept (the zero-score ones): c1_extract.one_log (simulated ego / actor poses, logged object
             trajectories with the logged ego first, driver requests / returns, ego observations, routes) + rec / summary / start
  map.pkl    privileged, labels only: lanes (centre, edges) and road-edge polylines (x, y, z) around the logged ego path, in the rollout's
             local frame, from the runtime's scene loader (the offroad scorer reads the same road edges)
  msgs/      serialized driver-side messages of the kept rollouts in arrival order (front wide camera only): input of lbd_pai_replay.py
"""
import argparse
import asyncio
import csv
import glob
import json
import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "alpasim" / "scripts"))
import c1_extract as X  # noqa: E402

CAM = "camera_front_wide_120fov"
KIND = {"driver_session_request", "driver_camera_image", "driver_ego_trajectory", "route_request", "driver_request"}


def one_msgs(a) -> str:
    run, scene, out = a

    async def go():
        return [(k, m.SerializeToString()) async for k, m in X._read(X.asl_of(run, scene), KIND) if k != "driver_camera_image" or m.camera_image.logical_id == CAM]
    pickle.dump(asyncio.run(go()), open(f"{out}/{scene}.pkl", "wb"), protocol=4)
    return scene


def lite(run: str) -> dict:
    import polars as pl
    out = {}
    for r in json.loads(Path(f"{run}/aggregate/results-summary.json").read_text())["rollouts"]:
        s, o = r["clipgt_id"], dict(summary=r)
        pq = glob.glob(f"{run}/rollouts/{s}/*/metrics.parquet")
        if pq:
            m = pl.read_parquet(pq[0]).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
            o["metrics"] = {n[0]: np.array([g["timestamps_us"].to_list(), g["values"].to_list()]).T for n, g in m.group_by("name")}
            o["session"] = Path(pq[0]).parent.name
            c = Path(run) / f"controller/alpasim_controller_{o['session']}.csv"
            if c.exists():
                rows = list(csv.reader(open(c)))
                o["ctrl_cols"], o["ctrl"] = rows[0], np.array(rows[1:], np.float64)
        out[s] = o
    for line in open(f"{run}/driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "drive" and r["scene"] in out:
            out[r["scene"]].setdefault("rec", []).append({k: r.get(k) for k in ("now", "t0", "k", "cmd", "route0", "ego", "anchor", "poses", "poses_model", "fix", "session")})
    for line in open(f"{run}/driver-logs/start.jsonl"):
        r = json.loads(line)
        if r.get("scene") in out:
            out[r["scene"]]["start"] = r
    for o in out.values():                                           # a scene may be served twice (retry): keep the session the scorer read
        if "rec" in o and o.get("session"):
            mine = [r for r in o["rec"] if r["session"] == o["session"]]
            o["rec"] = sorted(mine or o["rec"], key=lambda r: r["k"])
    return out


def maps(run: str, scenes: list, logs: dict, radius: float) -> dict:
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import build_scene_loader
    from omegaconf import OmegaConf
    loader = build_scene_loader(OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(f"{run}/generated-user-config-0.yaml")))
    out = {}
    for s in scenes:
        ds = loader.get_data_source(s)
        vm, g = ds.map, logs[s]["logged"][0]["traj"]
        c = np.r_[g[len(g) // 2, 1:3], 0.0]
        lanes = [dict(id=ln.id, center=np.asarray(ln.center.points[:, :2], np.float32),
                      left=None if ln.left_edge is None else np.asarray(ln.left_edge.points[:, :2], np.float32),
                      right=None if ln.right_edge is None else np.asarray(ln.right_edge.points[:, :2], np.float32),
                      next=sorted(ln.next_lanes), adj=sorted(ln.adj_lanes_left | ln.adj_lanes_right)) for ln in vm.get_lanes_within(c, radius)]
        edges = [np.asarray(e.polyline.points[:, :3], np.float32) for e in vm.get_road_edges_within(c, radius)]
        out[s] = dict(lanes=lanes, edges=edges)
        ds._map = None
        getattr(loader._provider, "_cache", {}).pop(s, None)               # the nuPlan provider caches scenes; the PAI one has no cache
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stacks", nargs="+", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8), ap.add_argument("--radius", type=float, default=150.0), ap.add_argument("--lite-only", action="store_true")
    a = ap.parse_args()
    for run in (r.rstrip("/") for r in a.stacks):
        o = Path(a.out) / Path(run).name
        if (o / "DONE").exists():
            continue
        (o / "msgs").mkdir(parents=True, exist_ok=True)
        L = lite(run)
        pickle.dump(L, open(o / "lite.pkl", "wb"), protocol=4)
        kept = [] if a.lite_only else sorted(s for s in L if glob.glob(f"{run}/rollouts/{s}/*/rollout.asl") and "metrics" in L[s])
        if kept:
            with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:       # polars is loaded in the parent: no fork
                logs = dict(ex.map(X.one_log, [(run, s) for s in kept]))
                list(ex.map(one_msgs, [(run, s, str(o / "msgs")) for s in kept]))
            for s, v in logs.items():
                v.update(summary=L[s]["summary"], rec=L[s].get("rec", []), start=L[s].get("start"))
            pickle.dump(logs, open(o / "logs.pkl", "wb"), protocol=4)
            pickle.dump(maps(run, kept, logs, a.radius), open(o / "map.pkl", "wb"), protocol=4)
        (o / "DONE").write_text(f"{len(L)} rollouts, {len(kept)} with logs\n")
        print(Path(run).name, len(L), "rollouts,", len(kept), "with logs", flush=True)


if __name__ == "__main__":
    main()
