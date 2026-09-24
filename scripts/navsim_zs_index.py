#!/usr/bin/env python
"""Freeze the per-token model inputs of a NAVSIM split for the zero-shot exam (todos/2026-09-24-zeroshot-exam/navsim.md).

Runs in the official devkit venv (envs/navsim2) and uses its SceneLoader, so tokens, history frames and ego statuses
are exactly what a NAVSIM agent receives: 4 history frames @2 Hz (num_history_frames of the split's scene filter),
ego pose / velocity / acceleration / driving command in the current rear-axle frame (AgentInput), plus the absolute
paths and calibration (sensor2lidar rotation, intrinsics, distortion) of the 8 cameras of every history frame.
Synthetic (navhard stage-two) scenes read their camera dicts from the synthetic scene pickles.

    envs/navsim2/bin/python scripts/navsim_zs_index.py navtest [navhard_two_stage ...]
Output: $DATA_DIR/runs/navsim_zs/index/<split>.pkl, a list of dicts (see `entry`).
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_nav = Path(os.environ["DATA_DIR"]) / "datasets/navsim"   # devkit env (docs/navsim.md), before navsim is imported
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(_nav))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(_nav / "maps"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
from jevdrive.navsim_zs import cams_of, root  # noqa: E402


def entry(token, stage, agent_input, cam_dicts, sensor_root, meta) -> dict:
    es = agent_input.ego_statuses
    return {"token": token, "stage": stage, **meta,
            "pose": np.stack([e.ego_pose for e in es]).astype(np.float32),              # (4, 3) x, y, yaw
            "vel": np.stack([e.ego_velocity for e in es]).astype(np.float32),           # (4, 2) body frame
            "acc": np.stack([e.ego_acceleration for e in es]).astype(np.float32),       # (4, 2)
            "cmd": np.stack([np.asarray(e.driving_command) for e in es]).astype(np.int8),  # (4, 4) one-hot
            "cams": [cams_of(d, sensor_root) for d in cam_dicts]}                       # 4 x {cam: calib + path}


def build(split: str) -> list:
    from navsim.common.dataloader import SceneLoader
    import navsim
    cfg_dir = Path(navsim.__file__).parent / "planning/script/config/common/train_test_split"
    split_cfg = OmegaConf.load(cfg_dir / f"{split}.yaml")
    sf_cfg = OmegaConf.load(cfg_dir / "scene_filter" / f"{split_cfg.defaults[0]['scene_filter']}.yaml")
    scene_filter = instantiate(sf_cfg)
    base = Path(os.environ["OPENSCENE_DATA_ROOT"])
    real_root = base / "sensor_blobs" / split_cfg.data_split
    syn_root = base / "navhard_two_stage" / "sensor_blobs"
    loader = SceneLoader(base / "navsim_logs" / split_cfg.data_split, real_root, scene_filter,
                         synthetic_sensor_path=syn_root,
                         synthetic_scenes_path=base / "navhard_two_stage" / "synthetic_scene_pickles")
    nh = scene_filter.num_history_frames
    out = []
    for tok in loader.tokens_stage_one:
        frames = loader.scene_frames_dicts[tok]
        meta = {"log_name": frames[nh - 1]["log_name"], "synthetic": False, "map": frames[nh - 1]["map_location"],
                "timestamp": int(frames[nh - 1]["timestamp"])}
        out.append(entry(tok, "one", loader.get_agent_input_from_token(tok), [f["cams"] for f in frames[:nh]],
                         real_root, meta))
    for tok, (path, log_name) in loader.synthetic_scenes.items():
        with open(path, "rb") as f:
            sd = pickle.load(f)
        meta = {"log_name": log_name, "synthetic": True, "timestamp": int(sd["frames"][nh - 1]["timestamp"]),
                "map": sd["scene_metadata"]["map_name"],
                "original": sd["scene_metadata"].get("corresponding_original_scene")}
        out.append(entry(tok, "two", loader.get_agent_input_from_token(tok),
                         [fr["camera_dict"] for fr in sd["frames"][:nh]], syn_root, meta))
    return out


def human_future(split: str):
    """Logged future (8 poses @2 Hz, current rear-axle frame) of every stage-one token, for ADE and figures only
    (never an input): $DATA_DIR/runs/navsim_zs/index/<split>_future.npz."""
    from navsim.common.dataloader import SceneLoader
    import navsim
    cfg_dir = Path(navsim.__file__).parent / "planning/script/config/common/train_test_split"
    split_cfg = OmegaConf.load(cfg_dir / f"{split}.yaml")
    sf = OmegaConf.load(cfg_dir / "scene_filter" / f"{split_cfg.defaults[0]['scene_filter']}.yaml")
    sf.include_synthetic_scenes = False
    base = Path(os.environ["OPENSCENE_DATA_ROOT"])
    loader = SceneLoader(base / "navsim_logs" / split_cfg.data_split, base / "sensor_blobs" / split_cfg.data_split,
                         instantiate(sf))
    toks = loader.tokens_stage_one
    fut = np.stack([loader.get_scene_from_token(t).get_future_trajectory(8).poses for t in toks]).astype(np.float32)
    np.savez(root("index") / f"{split}_future.npz", tokens=np.array(toks), poses=fut)
    print(f"{split}: future of {len(toks)} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="+")
    ap.add_argument("--future", action="store_true", help="dump the logged future instead of the inputs")
    a = ap.parse_args()
    for split in a.splits:
        if a.future:
            human_future(split)
            continue
        idx = build(split)
        missing = sum(not Path(c["CAM_F0"]["path"]).exists() for e in idx for c in e["cams"])
        with open(root("index") / f"{split}.pkl", "wb") as f:
            pickle.dump(idx, f, protocol=4)
        n2 = sum(e["stage"] == "two" for e in idx)
        print(f"{split}: {len(idx)} tokens ({len(idx) - n2} stage one, {n2} stage two), missing CAM_F0 files {missing}")


if __name__ == "__main__":
    main()
