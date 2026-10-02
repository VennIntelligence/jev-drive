"""Shared helpers for the navhard stage-2 off-road diagnosis (experiments/skill_pack/plans/2026-10-03-navhard-offroad-diagnosis-plan.md).

Runs in the navsim2 env (nuplan + navsim devkit, numpy 1.23). The scoring objects are built from the devkit's own hydra
configs, the per-token scoring is the devkit's `pdm_score` and the two-stage aggregation is the devkit's
`create_scene_aggregators` / `compute_final_scores` / `calculate_individual_mapping_scores`, so the official numbers are
reproduced and checked in `offroad_replay_cf.py`.
"""
import glob
import lzma
import os
import pickle
from pathlib import Path

import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(D / "datasets/navsim"))
os.environ.setdefault("NAVSIM_EXP_ROOT", str(D / "runs/navsim/eval"))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(D / "third_party/navsim"))
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")     # the box's OpenBLAS kernel bug (navsim_zs_score.sh)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

IDX = D / "runs/navsim_zs/index/navhard_two_stage_slim.pkl"
MCACHE = D / "runs/navsim/metric_cache/v2_navhard_two_stage"
NATIVE_POSES = D / "runs/op_lb/lb_navhard/preds/gimm-cinque__base.npz"
N4_POSES = D / "runs/skill_pack/raise/n4/navhard_n4.npz"
NATIVE_PLANS = D / "runs/op_lb/lb_navhard/plans/gimm@cinque.npz"
NATIVE_CSV = "v2_navhard_two_stage_opi_lb_navhard_gimm-cinque__base"
N4_CSV = "v2_navhard_two_stage_sp_n4_navhard"
OUT = D / "runs/skill_pack/offroad_diag"
CMDS = ["left", "straight", "right", "unknown"]
T_POSE = np.arange(1, 9) * 0.5          # pose file times (s)
T_DENSE = np.arange(0, 41) * 0.1        # simulator times (s)


def index():
    """Slim navhard index as a list (order = tokens.txt of the lb_navhard run)."""
    return pickle.load(open(IDX, "rb"))


def eval_csv(name):
    import pandas as pd
    p = sorted(glob.glob(str(D / "runs/navsim/eval" / name / "*/*.csv")))[-1]
    return pd.read_csv(p), p


def cache_paths():
    return {Path(p).parent.name: p for p in glob.glob(str(MCACHE / "*/*/*/metric_cache.pkl"))}


def load_cache(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def poses_by_token(path):
    z = np.load(path)
    return dict(zip(z["tokens"].tolist(), z["poses"].astype(np.float64)))


def setup_scoring():
    """simulator, scorer, reactive traffic policy, reactive_all_mapping, proposal sampling (as run_pdm_score.py)."""
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    GlobalHydra.instance().clear()
    with initialize_config_module("navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose("default_run_pdm_score", overrides=["train_test_split=navhard_two_stage", f"metric_cache_path={MCACHE}",
                                                        "experiment_name=offroad_diag"])
    sim, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    policy = instantiate(cfg.traffic_agents_policy.reactive, sim.proposal_sampling)
    mapping = {}
    for orig, prev, pairs in cfg.train_test_split.reactive_all_mapping:
        mapping[(orig, prev)] = [tuple(p) for p in pairs]
    return sim, scorer, policy, mapping, sim.proposal_sampling


def to_ego(mc, arr):
    """Global (.., 3) x, y, heading -> ego frame at t0 (rear axle)."""
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    d = arr[..., :2] - o[:2]
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    h = (arr[..., 2] - o[2] + np.pi) % (2 * np.pi) - np.pi
    return np.stack([x, y, h], -1)


def pdm_ref_ego(mc, sampling):
    """PDM reference in the ego frame on the 0.1 s grid (41, 3)."""
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    st = get_trajectory_as_array(mc.trajectory, sampling, mc.ego_state.time_point)
    return to_ego(mc, st[:, :3])


def dense_from_poses(p8):
    """8 poses (0.5 .. 4 s) + the origin -> dense (41, 3) by linear interpolation (heading unwrapped), as the scorer's
    InterpolatedTrajectory does."""
    P = np.vstack([[0, 0, 0], p8])
    t = np.r_[0, T_POSE]
    h = np.unwrap(P[:, 2])
    return np.stack([np.interp(T_DENSE, t, P[:, 0]), np.interp(T_DENSE, t, P[:, 1]), np.interp(T_DENSE, t, h)], -1)


def poses_from_dense(dense):
    return dense[np.rint(T_POSE / 0.1).astype(int)].copy()


def simulate(sim, mc, p8):
    """Official LQR + bicycle states (41, 11) of an 8-pose ego-frame trajectory."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: F401
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    tr = transform_trajectory(Trajectory(np.asarray(p8, np.float64)), mc.ego_state)
    arr = get_trajectory_as_array(tr, sim.proposal_sampling, mc.ego_state.time_point)
    return sim.simulate_proposals(arr[None], mc.ego_state)[0]
