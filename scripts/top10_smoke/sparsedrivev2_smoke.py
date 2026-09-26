"""SparseDriveV2 (NAVSIM v1 checkpoint) single-sample smoke on one real navtest token.

Runs the release's own feature builder + test-mode pipeline and model forward, compares the chosen
trajectory to the logged future, and times the forward per docs/baselines.md (batch 1, 5 warmup + 20 timed).
Run on the box from the repo root ~/data/third_party/sparsedrivev2 with env envs/sparsedrivev2 and
OPENBLAS_CORETYPE=Haswell.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import default_collate

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
NAV = DATA / "datasets/navsim"
W = DATA / "models/sparsedrivev2"
CKPT = W / "sparsedrive_navsimv1_92p2.ckpt"
LOG = sys.argv[1] if len(sys.argv) > 1 else "2021.06.03.12.02.06_veh-35_01100_01227"  # a navtest log
MIN_SPEED = 3.0  # pick the first scene of the log whose ego moves faster than this (m/s)

# NAVSIM v1 eval settings from scripts/evaluation/run_pdm_score_navtest_v1.sh
cfg = SparseDriveConfig(
    bkb_path=str(W / "resnet34.bin"),
    path_anchor=str(W / "kmeans/path_1024.npy"),
    velocity_anchor=str(W / "kmeans/velocity_256.npy"),
    trajectory_anchor=str(W / "kmeans/trajectory_1024_256.npz"),
    dataset_version="v1",
    metrics=["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
             "time_to_collision_within_bound", "comfort", "ego_progress"],
    velocity_filter_num=[64, 20],
)
agent = SparseDriveAgent(cfg, lr=1e-4, checkpoint_path=str(CKPT))
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]  # Lightning ckpt, not weights-only
agent.load_state_dict({k.replace("agent.", ""): v for k, v in sd.items()}, strict=True)
agent.eval()

nav_split = yaml.safe_load(open("navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
assert LOG in nav_split["log_names"], f"{LOG} not in navtest"
tokens = nav_split.get("tokens")
sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                 log_names=[LOG], tokens=tokens)
loader = SceneLoader(data_path=NAV / "navsim_logs/test", original_sensor_path=NAV / "sensor_blobs/test",
                     scene_filter=sf, sensor_config=agent.get_sensor_config())
fb = agent.get_feature_builders()[0]
for token in loader.tokens:
    scene = loader.get_scene_from_token(token)
    ego = scene.get_agent_input().ego_statuses[-1]
    if np.linalg.norm(ego.ego_velocity) > MIN_SPEED:
        break

# feature building + test-mode pipeline (CPU), as in their dataset.__getitem__
t0 = time.perf_counter()
agent_input = scene.get_agent_input()
feats = fb.compute_features(agent_input)
feats, _, _ = fb.pipeline(feats, {}, token, test_mode=True)
t_pre = (time.perf_counter() - t0) * 1e3
batch = default_collate([feats])

gpu = torch.device("cuda")
agent.to(gpu)
def to(x):
    if torch.is_tensor(x): return x.to(gpu)
    if isinstance(x, dict): return {k: to(v) for k, v in x.items()}
    if isinstance(x, list): return [to(v) for v in x]
    return x
batch = to(batch)
cam = batch["camera_feature"]
shapes = {"imgs": list(cam["imgs"].shape), "projection_mat": list(cam["projection_mat"].shape),
          "status_feature": list(batch["status_feature"].shape)}

def fwd():
    b = {"camera_feature": dict(cam), "status_feature": batch["status_feature"]}  # forward writes feature_maps into it
    return agent.forward(b, {})[0]["trajectory"]

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    for _ in range(5): fwd()
    ts = []
    for _ in range(20):
        torch.cuda.synchronize(); t = time.perf_counter()
        traj = fwd()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
peak = torch.cuda.max_memory_allocated() / 2**30
pred = traj[0].float().cpu().numpy()                     # (8, 3): x fwd, y left, heading; ego frame, 2 Hz, 4 s
gt = scene.get_future_trajectory(num_trajectory_frames=8).poses
ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean())
fde = float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))
ts = np.array(ts)
out = {
    "model": "SparseDriveV2 navsimv1 92.2 PDMS", "ckpt": str(CKPT), "log": LOG, "token": token,
    "ego_velocity": ego.ego_velocity.tolist(), "ego_acceleration": ego.ego_acceleration.tolist(),
    "driving_command": ego.driving_command.tolist(),
    "input_shapes": shapes, "cams": list(cfg.cams),
    "latency_ms": {"mean": ts.mean(), "p50": np.percentile(ts, 50), "p95": np.percentile(ts, 95)},
    "timed_span": "agent.forward on GPU-resident preprocessed batch: ResNet-34+FPN on 3 cams, 2-layer path/velocity "
                  "scoring decoder, deformable aggregation, trajectory re-scoring + argmax; fp32; shared GPU",
    "preprocess_ms_cpu_once": t_pre, "peak_vram_gb": peak, "gpu": torch.cuda.get_device_name(),
    "pred_traj": pred.tolist(), "gt_traj": gt.tolist(), "ade_m": ade, "fde_m": fde,
    "pred_end_heading_deg": float(np.degrees(pred[-1, 2])), "gt_end_heading_deg": float(np.degrees(gt[-1, 2])),
    "torch": torch.__version__, "cuda": torch.version.cuda,
}
od = DATA / "runs/top10_smoke/sparsedrivev2" / time.strftime("%Y%m%d-%H%M%S")
od.mkdir(parents=True, exist_ok=True)
json.dump(out, open(od / "summary.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["token", "ego_velocity", "driving_command", "input_shapes", "latency_ms",
                                     "peak_vram_gb", "ade_m", "fde_m", "pred_end_heading_deg", "gt_end_heading_deg"]},
                 default=float, indent=1))
print("pred", np.round(pred, 2).tolist()); print("gt  ", np.round(gt, 2).tolist()); print("->", od)
