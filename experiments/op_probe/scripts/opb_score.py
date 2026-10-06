"""op_probe per-token navtest scoring of arbitrary plans with the devkit's own pdm_score (v2 navtest metric cache, run_pdm_score.py's
simulator / scorer / reactive traffic, as experiments/skill_pack/scripts/offroad_replay_cf.py does for navhard). envs/navsim2, CPU:

  $DATA_DIR/envs/navsim2/bin/python experiments/op_probe/scripts/opb_score.py --poses f.npz [--keys a b] [--tokens t.txt] --out o.csv

f.npz: `tokens` (N,) and one or more (N, 8, 3) pose arrays (rear axle at t0, 0.5 .. 4 s); every key is scored on the selected tokens.
Per (key, token): the v2 sub-scores, `score` = the per-token EPDMS without extended comfort (EC needs the neighbouring frame), and two DAC
diagnostics: `raw_out` (the raw plan, linearly interpolated to 0.1 s with the ego footprint, leaves the scorer's drivable polygons: no
tracker) and `out_depth` (largest distance of a footprint corner outside those polygons along the LQR-simulated states, m).
"""
import argparse
import glob
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
for k, v in dict(NUPLAN_MAP_VERSION="nuplan-maps-v1.0", NUPLAN_MAPS_ROOT=str(D / "datasets/navsim/maps"),
                 OPENSCENE_DATA_ROOT=str(D / "datasets/navsim"), NAVSIM_EXP_ROOT=str(D / "runs/navsim/eval"),
                 NAVSIM_DEVKIT_ROOT=str(D / "third_party/navsim"), OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="1",
                 OMP_NUM_THREADS="1").items():
    os.environ.setdefault(k, v)
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
MCACHE = D / "runs/navsim/metric_cache/v2_navtest"
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort"]
T_POSE, T_DENSE = np.arange(1, 9) * 0.5, np.arange(0, 41) * 0.1
_W = {}


def _init(poses_file, keys):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    GlobalHydra.instance().clear()
    with initialize_config_module("navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose("default_run_pdm_score", overrides=["train_test_split=navtest", f"metric_cache_path={MCACHE}", "experiment_name=op_probe"])
    sim, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    z = np.load(poses_file)
    row = {t: i for i, t in enumerate(z["tokens"].tolist())}
    _W.update(sim=sim, scorer=scorer, policy=instantiate(cfg.traffic_agents_policy.reactive, sim.proposal_sampling),
              samp=sim.proposal_sampling, P={k: z[k] for k in keys}, row=row,
              cp={Path(p).parent.name: p for p in glob.glob(str(MCACHE / "*/*/*/metric_cache.pkl"))})


def _dense(p8):
    P = np.vstack([[0, 0, 0], p8])
    t = np.r_[0, T_POSE]
    h = np.unwrap(P[:, 2])
    return np.stack([np.interp(T_DENSE, t, P[:, 0]), np.interp(T_DENSE, t, P[:, 1]), np.interp(T_DENSE, t, h)], -1)


def _corners(mc, states):
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    return state_array_to_coords_array(states[None], mc.ego_state.car_footprint.vehicle_parameters)[0]   # (41, 5, 2), last = centre


def _out_depth(mc, cor, idc, area_u):
    import shapely
    ins = mc.drivable_area_map.points_in_polygons(cor[None, :, :-1, :])[idc].any(0)[0]     # (41, 4)
    if ins.all():
        return False, 0.0
    pts = shapely.points(cor[:, :-1][~ins])
    return True, float(shapely.distance(area_u, pts).max())


def work(token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from shapely.ops import unary_union
    W = _W
    with lzma.open(W["cp"][token], "rb") as f:
        mc = pickle.load(f)
    am = mc.drivable_area_map
    idc = am.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    area_u = unary_union([am._geometries[k] for k in idc])
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    out = []
    for k, P in W["P"].items():
        p8 = np.asarray(P[W["row"][token]], np.float64)
        row, st = pdm_score(metric_cache=mc, model_trajectory=Trajectory(p8), future_sampling=W["samp"], simulator=W["sim"],
                            scorer=W["scorer"], traffic_agents_policy=W["policy"])
        r = row.iloc[0] if hasattr(row, "iloc") else row
        res = {"key": k, "token": token, **{m: float(r[m]) for m in SUBS}}
        X = np.array([res[m] for m in SUBS])
        res["score"] = float(np.prod(X[:4]) * (5 * X[4] + 5 * X[5] + 2 * X[6] + 2 * X[7]) / 14)
        d = _dense(p8)                                                     # raw plan, ego frame -> global states
        g = np.zeros((41, st.shape[-1]))
        g[:, 0], g[:, 1], g[:, 2] = o[0] + c * d[:, 0] - s * d[:, 1], o[1] + s * d[:, 0] + c * d[:, 1], o[2] + d[:, 2]
        res["raw_out"], res["raw_depth"] = _out_depth(mc, _corners(mc, g), idc, area_u)
        lq = _out_depth(mc, _corners(mc, np.asarray(st)), idc, area_u)
        res["lqr_out"], res["out_depth"] = lq
        out.append(res)
    return out


def main():
    import pandas as pd
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--tokens", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    z = np.load(a.poses)
    keys = a.keys or [k for k in z.files if z[k].ndim == 3 and z[k].shape[1:] == (8, 3)]
    toks = [t.strip() for t in open(a.tokens)] if a.tokens else z["tokens"].tolist()
    toks = [t for t in toks if t]
    with Run("op_probe", f"score-{Path(a.out).stem}", config=vars(a) | {"keys": keys, "n": len(toks)}) as run:
        import multiprocessing as mp
        t0 = time.time()
        rows = []
        with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.poses, keys)) as pool:
            for k, part in enumerate(pool.imap_unordered(work, toks, chunksize=4)):
                rows.extend(part)
                if (k + 1) % 500 == 0:
                    run.status(f"{k + 1}/{len(toks)} tokens, {time.time() - t0:.0f} s")
        df = pd.DataFrame(rows)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        g = df.groupby("key")[["drivable_area_compliance", "score", "raw_out", "lqr_out"]].mean()
        run.summary.update(n=len(toks), keys=keys, wall_s_scoring=time.time() - t0, means=g.to_dict())
        run.info(f"{len(toks)} tokens x {len(keys)} keys in {time.time() - t0:.0f} s\n{g}")


if __name__ == "__main__":
    main()
