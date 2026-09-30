"""op-adapt r2 package S, NAVSIM side (runs in a navsim env: PYTHONPATH=<repo> $DATA_DIR/envs/navsim{1,2}/bin/python).

  extract  --cache v1_navtrain --tokens <txt>   per-token geometry for S_jev (navsim1 env for the v1 cache):
           R/maps/nav/<cache>/<token>.npz: rear-axle pose, speed, WKB of the drivable polygons (ROADBLOCK, INTERSECTION,
           DRIVABLE_AREA, CARPARK_AREA), lane / lane-connector polygons with an on-route flag, intersections, the PDM centre
           line, and every tracked object's box on the 10 Hz grid (observation[t], t = 0 ... 40)
  v5       --cache v2_navtest                   V5 part a (navsim2 env): for the devkit's own PDM-Closed trajectory
           (metric_cache.trajectory), the constant-velocity agent and the human log, simulate with the devkit's
           PDMSimulator, take the devkit PDMScorer's raw driving_direction_compliance (before the human filter) and our
           op_adapt_score.ddc on the same simulated states with the same map -> R/checks/V5a.{json,parquet}
Workers: --workers (default 24), pinned by the caller (taskset -c 0-23).
"""
import argparse
import json
import lzma
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import op_adapt_score as S  # noqa: E402

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "runs" / "op_adapt_r2"
MC = DATA / "runs" / "navsim" / "metric_cache"


def load_mc(path):
    with open(path, "rb") as f:
        return pickle.loads(lzma.decompress(f.read()))


def cache_paths(cache: str) -> dict:
    import csv
    meta = next((MC / cache / "metadata").glob("*.csv"))
    with open(meta) as f:
        rows = [r[0] for r in csv.reader(f)][1:]
    return {Path(p).parent.name: p for p in rows}


def geometry(mc):
    """(drivable, lanes, lane_route, intersection) shapely lists from the cached PDMDrivableMap."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    dm = mc.drivable_area_map
    types, toks, geo = dm._map_types, dm._tokens, dm._geometries
    route = set(mc.route_lane_ids)
    drv = [g for g, t in zip(geo, types) if t in (L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA)]
    lanes = [(g, k in route) for g, t, k in zip(geo, types, toks) if t in (L.LANE, L.LANE_CONNECTOR)]
    inter = [g for g, t in zip(geo, types) if t == L.INTERSECTION]
    return drv, [g for g, _ in lanes], np.array([r for _, r in lanes], bool), inter


KIND = {"VEHICLE": S.VEH, "PEDESTRIAN": S.PED, "BICYCLE": S.CYC}


def _extract_one(args):
    import shapely
    cache, token, path = args
    out = R / "maps" / "nav" / cache / f"{token}.npz"
    if out.exists():
        return token, "exists"
    try:
        mc = load_mc(path)
        drv, lanes, lane_route, inter = geometry(mc)
        ra = mc.ego_state.rear_axle
        obs = mc.observation
        toks = []
        per_t = []
        for t in range(S.NT):
            om = obs[t]
            d = {}
            for k, g in zip(om.tokens, om._geometries):
                if obs.red_light_token in k:
                    continue
                c = np.asarray(g.exterior.coords)[:4]
                fl, rl, rr = c[0], c[1], c[2]
                d[k] = ((fl + rr) / 2, np.arctan2(*(fl - rl)[::-1]), np.hypot(*(fl - rl)) / 2, np.hypot(*(rl - rr)) / 2)
                if k not in toks:
                    toks.append(k)
            per_t.append(d)
        N = len(toks)
        c = np.zeros((S.NT, N, 2))
        h = np.zeros((S.NT, N))
        hl, hw = np.zeros(N), np.zeros(N)
        valid = np.zeros((S.NT, N), bool)
        for t, d in enumerate(per_t):
            for j, k in enumerate(toks):
                if k in d:
                    c[t, j], h[t, j], hl[j], hw[j] = d[k]
                    valid[t, j] = True
        sp = np.hypot(*np.moveaxis(np.gradient(c, S.DT, axis=0), -1, 0)) * valid
        kind = np.array([KIND.get(obs.unique_objects[k].tracked_object_type.name, S.STATIC) if k in obs.unique_objects else S.STATIC
                         for k in toks], np.int8)
        cl = np.array([[p.x, p.y] for p in mc.centerline.discrete_path])
        def wkb(name, gs):                                   # concatenated WKB + offsets ('S' arrays would strip trailing NULs)
            b = [bytes(x) for x in shapely.to_wkb(gs)] if len(gs) else []
            return {f"{name}_wkb": np.frombuffer(b"".join(b), np.uint8), f"{name}_off": np.cumsum([0] + [len(x) for x in b]).astype(np.int64)}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pose=np.array([ra.x, ra.y, ra.heading]), v0=float(mc.ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
                            **wkb("drivable", drv), **wkb("lanes", lanes), lane_route=lane_route, **wkb("intersection", inter), centerline=cl,
                            agent_c=c, agent_h=h, agent_hl=hl, agent_hw=hw, agent_valid=valid, agent_speed=sp, agent_kind=kind)
        tmp.replace(out)
        return token, "ok"
    except Exception as e:  # noqa: BLE001
        return token, repr(e)[:300]


def extract(cache: str, tokens_file: str, workers: int):
    paths = cache_paths(cache)
    toks = [t.strip() for t in open(tokens_file) if t.strip()]
    jobs = [(cache, t, paths[t]) for t in toks if t in paths]
    t0, bad = time.time(), []
    with Pool(workers) as p:
        for i, (t, st) in enumerate(p.imap_unordered(_extract_one, jobs, chunksize=16)):
            if st not in ("ok", "exists"):
                bad.append((t, st))
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s bad={len(bad)}", flush=True)
    res = {"cache": cache, "tokens": len(toks), "found": len(jobs), "errors": len(bad), "examples": bad[:5], "wall_s": time.time() - t0}
    (R / "maps" / "nav" / cache).mkdir(parents=True, exist_ok=True)
    (R / "maps" / "nav" / cache / f"extract_{Path(tokens_file).stem}.json").write_text(json.dumps(res, indent=1))
    print(res)


# ---------------------------------------------------------------- V5 part a

_SIM = {}


def _devkit():
    if not _SIM:
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        ps = TrajectorySampling(num_poses=40, interval_length=0.1)
        _SIM.update(ps=ps, sim=PDMSimulator(ps), scorer=PDMScorer(ps, PDMScorerConfig(human_penalty_filter=True)))
    return _SIM


def _v5_one(args):
    token, path = args
    try:
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
        d = _devkit()
        mc = load_mc(path)
        ego = mc.ego_state
        t0 = ego.time_point
        v = float(np.hypot(ego.dynamic_car_state.rear_axle_velocity_2d.x, ego.dynamic_car_state.rear_axle_velocity_2d.y))
        cv = Trajectory(np.array([[(i + 1) * 0.5 * v, 0.0, 0.0] for i in range(8)], np.float32), TrajectorySampling(time_horizon=4, interval_length=0.5))
        trajs = {"pdm": mc.trajectory, "cv": transform_trajectory(cv, ego)}
        if mc.human_trajectory is not None:
            trajs["human"] = transform_trajectory(mc.human_trajectory, ego)
        pdm_states = get_trajectory_as_array(mc.trajectory, d["ps"], t0)
        drv, lanes, lane_route, inter = geometry(mc)
        pm = S.PolyMap(drv, lanes, lane_route, inter)
        rows = []
        for name, tr in trajs.items():
            ref = get_trajectory_as_array(tr, d["ps"], t0)
            both = {"ref": np.stack([pdm_states, ref]), "sim": d["sim"].simulate_proposals(np.stack([pdm_states, ref]), ego)}
            for how, st in both.items():
                res = d["scorer"].score_proposals(st, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)[1]
                s = st[1]
                hd = s[:, 2]
                c = s[:, :2] + S.EGO["nav"].rc * np.stack([np.cos(hd), np.sin(hd)], -1)
                ar = pm.areas(np.zeros((1, len(c), 4, 2)), c[None], s[None, :, :2], hd[None])
                ours, D = S.ddc({"c": c[None]}, ar)
                rows.append({"token": token, "traj": name, "states": how, "devkit": float(res["driving_direction_compliance"].iloc[0]),
                             "ours": float(ours[0]), "D": float(D[0]), "path_m": float(np.hypot(*np.diff(s[:, :2], axis=0).T).sum())})
        return rows
    except Exception as e:  # noqa: BLE001
        return [{"token": token, "error": repr(e)[:300]}]


def v5(cache: str, workers: int, limit: int | None = None):
    import pandas as pd
    paths = cache_paths(cache)
    jobs = sorted(paths.items())[:limit] if limit else sorted(paths.items())
    t0 = time.time()
    rows = []
    with Pool(workers) as p:
        for i, part in enumerate(p.imap_unordered(_v5_one, jobs, chunksize=8)):
            rows += part
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    t = pd.DataFrame(rows)
    (R / "checks").mkdir(parents=True, exist_ok=True)
    t.to_parquet(R / "checks" / "V5a.parquet", index=False)
    ok = t[t["error"].isna()] if "error" in t else t
    agree = float((ok[ok.states == "ref"].devkit == ok[ok.states == "ref"].ours).mean())
    res = {"check": "V5a", "line": "our DDC == devkit raw DDC (three levels) on >= 99 % of (token, trajectory)", "cache": cache,
           "tokens": len(jobs), "rows": int(len(ok)), "errors": int(len(t) - len(ok)), "agreement": agree, "pass": agree >= 0.99,
           "by_traj": {f"{a}/{b}": {"n": int(len(g)), "agree": float((g.devkit == g.ours).mean()), "devkit_1": float((g.devkit == 1).mean()),
                                    "devkit_05": float((g.devkit == 0.5).mean()), "median_path_m": float(g.path_m.median())}
                       for (a, b), g in ok.groupby(["states", "traj"])},
           "confusion": ok.groupby(["states", "devkit", "ours"]).size().rename("n").reset_index().to_dict("records"), "wall_s": time.time() - t0,
           "note": "states=ref: the trajectory's own interpolated states (S_jev does no LQR tracking; the registered comparison); "
                   "states=sim: after the devkit PDMSimulator, which in this devkit build runs away on many tokens (km-long paths)"}
    (R / "checks" / "V5a.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("extract", "v5"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tokens")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    extract(a.cache, a.tokens, a.workers) if a.step == "extract" else v5(a.cache, a.workers, a.limit)
