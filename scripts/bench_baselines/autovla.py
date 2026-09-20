"""Latency of AutoVLA as released (ucla-mobility/AutoVLA @ ba34eed), batch 1, on one sample.

AutoVLA ships only a NAVSIM agent, so this is a minimal single-sample entry point: it builds the
feature dict `AutoVLA.predict()` expects (the same one `AutoVLAAgentFeatureBuilder` produces) and calls
their model unchanged. `models.utils.score` is stubbed out because importing it pulls in navsim and
nuplan-devkit, which only the GRPO reward uses.

  $DATA_DIR/envs/autovla/bin/python scripts/bench_baselines/autovla.py [--iters 200] [--only <tag>]

Input: the WOD-E2E demo scene shipped with Qwen-Drive (front / front-left / front-right, 4 frames each at 2 Hz),
the same scene used for the other baselines. Ego speed and acceleration come from that scene's logged history.
Timed span = `AutoVLA.predict()`: JPEG decode + resize (qwen_vl_utils, CPU), Qwen2.5-VL vision tower, prefill,
autoregressive decode of the optional chain-of-thought plus the action tokens, and detokenizing them to a
trajectory. Modes: `cot` is the released adaptive prompt (the RFT model decides whether to reason),
`nocot` is their fast-thinking prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import ast

import numpy as np
import torch
import yaml

DATA = Path(os.environ["DATA_DIR"])
SRC = DATA / "third_party" / "autovla"
FRAMES = DATA / "third_party" / "qwen-drive" / "data" / "demo"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))
from _bench import bench  # noqa: E402

# models.autovla imports the GRPO reward, which needs navsim + nuplan-devkit; inference never calls it.
stub = types.ModuleType("models.utils.score")
stub.PDM_Reward = stub.TrajectorySampling = stub.Trajectory = object
sys.modules["models.utils.score"] = stub
from models.autovla import AutoVLA  # noqa: E402

p = argparse.ArgumentParser()
ckpt = sorted((DATA / "cache/huggingface/hub/models--Zewei-Zhou--AutoVLA/snapshots").glob("*/AutoVLA_PDMS_89.ckpt"))
p.add_argument("--checkpoint", default=str(ckpt[0]) if ckpt else None, required=not ckpt)
p.add_argument("--base-model", default=str(DATA / "models/Qwen2.5-VL-3B-Instruct"))
p.add_argument("--config", default=str(SRC / "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
p.add_argument("--scene-index", type=int, default=1)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--iters", type=int, default=200)
p.add_argument("--only", default=None)
args = p.parse_args()

config = yaml.safe_load(open(args.config))
config["model"]["pretrained_model_path"] = args.base_model
config["model"]["codebook_cache_path"] = str(SRC / config["model"]["codebook_cache_path"])

# The demo scene: 3 views x 4 frames at 2 Hz, plus the ego state that scene logged.
scene = [json.loads(line) for line in open(FRAMES / "planning_scenes.jsonl")][args.scene_index]
images = [c["image"] for m in scene["messages"] for c in m["content"] if "image" in c]
assert len(images) == 12, images
trajectory = ast.literal_eval(scene["trajectory"])
ego = trajectory["ego_status"]
command = ("go straight", "turn left", "turn right")[int(trajectory["nav_command"])]
features = {
    "images": {"front_camera": [str(FRAMES / i) for i in images[0:4]],
               "front_left_camera": [str(FRAMES / i) for i in images[4:8]],
               "front_right_camera": [str(FRAMES / i) for i in images[8:12]]},
    # The NAVSIM feature builder passes the ego velocity and acceleration vectors; predict() takes their norm.
    "vehicle_velocity": ego["ego_velocity"],
    "vehicle_acceleration": ego["ego_acceleration"],
    "driving_command": command,
    "sensor_data_path": None,
}
token = ast.literal_eval(scene["meta_info"])["token"]
print(f"scene {token}: v {np.linalg.norm(ego['ego_velocity']):.2f} m/s, "
      f"a {np.linalg.norm(ego['ego_acceleration']):.2f} m/s^2, command {command!r}")

meta = {"source": "ucla-mobility/AutoVLA@ba34eed, Zewei-Zhou/AutoVLA AutoVLA_PDMS_89.ckpt (NAVSIM, RFT/GRPO)",
        "dtype": "bfloat16", "base_model": "Qwen2.5-VL-3B-Instruct", "licence": "UCLA Academic Software License",
        "changed_vs_pins": "torch 2.4.0 -> 2.8.0+cu128, torchvision 0.19.0 -> 0.23.0, python 3.9 -> 3.12 (sm_120)",
        "generation": config["inference"]["sample"], "scene": token, "driving_command": command}

model = AutoVLA(config, device="cuda")
state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)["state_dict"]
missing, unexpected = model.load_state_dict({k.replace("autovla.", ""): v for k, v in state.items()}, strict=False)
print(f"loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
del state
model.eval()

for tag, use_cot in (("cot", True), ("nocot", False)):
    if args.only and tag != args.only:
        continue
    model.use_cot = use_cot
    with torch.no_grad():
        traj, text = model.predict(features)
        print(f"[{tag}] trajectory {tuple(traj.shape)}, endpoint {np.round(traj[-1].cpu().numpy(), 2).tolist()}\n"
              f"  output: {text!r}")
        bench("autovla", tag, lambda: model.predict(features), warmup=args.warmup, iters=args.iters,
              meta={**meta, "use_cot": use_cot, "example_output": text},
              extra=lambda r: {"output_chars": len(r[1]), "traj_points": int(r[0].shape[0])})
