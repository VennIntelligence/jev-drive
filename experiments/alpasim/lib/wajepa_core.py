"""WA-JEPA (AFARI-Research/WA-JEPA @ bec2966, released weights) as a causal 2 Hz planner: its shipped NAVSIM agent
(eval/navsim_agent.py WorldModelNavsimAgent + WorldModelFeatureBuilder, configs/wa_jepa_navsim_epdms.yaml, the path that scored navtest
EPDMS 91.71 in our devkit) called through `agent.compute_trajectory(AgentInput)` for ONE decision. Nothing of theirs is modified; this module
only assembles the NAVSIM AgentInput the agent reads. experiments/alpasim/lib/wajepa_driver.py wraps it as an AlpaSim EgodriverService.

Per decision at t0, from the keyframes at t0 - 1.5 / 1.0 / 0.5 / 0 s (as many as exist; NAVSIM's 2 Hz grid = AlpaSim's camera rate):
  cameras  CAM_L0 / CAM_F0 / CAM_R0 / CAM_B0 RGB (their config's camera_names), decoded like their builder's file path (cv2 BGR -> RGB)
  ego      4 rear-axle poses (x, y, yaw), t0 velocity / acceleration (vx, vy, ax, ay), NAVSIM one-hot command [L, S, R, unknown]
  output   their 8 poses at 0.5 .. 4 s in the t0 rear-axle frame (fp32, 4-step flow, config flow seed re-applied per call)

Cold start (fewer than 4 keyframes; NAVSIM always has 4):
  repeat (default) the missing frames AND poses are copies of the oldest ones: the rule of WA-JEPA's own closed-loop client
           (close_loop/hugsim_planner.py `_history_indices`: "indices clamp to 0, i.e. the oldest frame is repeated")
  cv       the missing frames are copies of the oldest frame, the missing poses the oldest state run backwards at constant body velocity
           and yaw rate (the pose rule of experiments/alpasim/lib/sh30_core.py)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("WAJ_REPO") or Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "third_party/wajepa")
CKPT = Path(os.environ.get("WAJ_CKPT") or Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "models/wajepa/model_state_dict.pt")
CFG = REPO / os.environ.get("WAJ_CFG", "configs/wa_jepa_navsim_epdms.yaml")
OVERRIDES = ["model.require_pretrained=false", "model.vjepa2_ckpt=null"]   # as experiments/top10/lib/top10_t2/wajepa_run.py
CAMS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")
T_KEY = np.array([-1.5, -1.0, -0.5, 0.0])
COLD = ("repeat", "cv")
HW = (256, 512)


def decode(jpeg: bytes) -> np.ndarray:
    """JPEG bytes -> RGB uint8 resized to the model input with INTER_AREA: what their builder does to a file path (cv2.imread, BGR -> RGB,
    cv2.resize INTER_AREA). The builder resizes again; a same-size cv2.resize is a copy, so the tensor it builds is unchanged."""
    import cv2
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG decode failed")
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (HW[1], HW[0]), interpolation=cv2.INTER_AREA)


def back_extrapolate(pose, vel, yaw_rate: float, cold: str):
    """pose (m, 3) x, y, yaw in the t0 frame of the m <= 4 newest keys (oldest first), vel (2,) body velocity of the oldest key -> (4, 3)."""
    pose = np.asarray(pose, np.float64).reshape(-1, 3)
    e = 4 - len(pose)
    P = np.zeros((4, 3))
    P[e:] = pose
    if cold == "repeat":
        P[:e] = pose[0]
        return P
    w, (x0, y0, a0), v = float(yaw_rate), pose[0], np.asarray(vel, np.float64)
    c0, s0 = np.cos(a0), np.sin(a0)
    for j in range(e):
        dt = T_KEY[j] - T_KEY[e]
        s, c = (np.sin(w * dt) / w, (1 - np.cos(w * dt)) / w) if abs(w) > 1e-6 else (dt, 0.0)
        dx, dy = s * v[0] - c * v[1], c * v[0] + s * v[1]
        P[j] = [x0 + c0 * dx - s0 * dy, y0 + s0 * dx + c0 * dy, a0 + w * dt]
    return P


class Core:
    def __init__(self, dev: str = "cuda", cold: str = "repeat", amp: bool = False):
        assert cold in COLD, cold
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import torch
        from eval.navsim_agent import WorldModelNavsimAgent
        from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus
        self.torch, self.cold, self.dev, self.amp = torch, cold, torch.device(dev), amp
        self.AgentInput, self.Camera, self.Cameras, self.EgoStatus = AgentInput, Camera, Cameras, EgoStatus
        self.agent = WorldModelNavsimAgent(config_path=str(CFG), checkpoint_path=str(CKPT), device=dev, config_overrides=OVERRIDES)
        self.agent.initialize()
        names = [c.upper() for c in self.agent._camera_names]
        assert tuple(names) == CAMS, f"config camera_names {names} != {CAMS}"
        assert tuple(self.agent._resize_to) == HW and self.agent._num_history_image_frames == 4, (self.agent._resize_to,)
        self.tag = f"WA-JEPA-{CKPT.stem}-{CFG.stem}-{'bf16' if amp else 'fp32'}"

    def agent_input(self, frames, pose, vel, acc, cmd, yaw_rate: float = 0.0):
        """frames: m <= 4 dicts CAM -> RGB (oldest first, the last at t0); pose (m, 3) in the t0 frame; vel (m, 2) body velocities;
        acc (2,), cmd (4,) at t0 -> AgentInput with 4 history steps (cold rule applied), the poses actually fed."""
        m = len(frames)
        P = back_extrapolate(pose, np.asarray(vel)[0], yaw_rate, self.cold)
        F = [frames[0]] * (4 - m) + list(frames)
        empty = self.Camera()
        cams = [self.Cameras(cam_f0=self.Camera(image=f["CAM_F0"]), cam_l0=self.Camera(image=f["CAM_L0"]), cam_r0=self.Camera(image=f["CAM_R0"]),
                             cam_b0=self.Camera(image=f["CAM_B0"]), cam_l1=empty, cam_l2=empty, cam_r1=empty, cam_r2=empty) for f in F]
        v_t0 = np.asarray(vel, np.float32)[-1]
        ego = [self.EgoStatus(ego_pose=P[j], ego_velocity=v_t0, ego_acceleration=np.asarray(acc, np.float32),
                              driving_command=np.asarray(cmd, np.int64)) for j in range(4)]
        return self.AgentInput(ego_statuses=ego, cameras=cams, lidars=[]), P

    def plan(self, frames, pose, vel, acc, cmd, yaw_rate: float = 0.0) -> dict:
        """One decision -> poses (8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in the t0 frame, the fed history, stage times (ms)."""
        t0 = time.perf_counter()
        ai, P = self.agent_input(frames, pose, vel, acc, cmd, yaw_rate)
        t1 = time.perf_counter()
        with self.torch.autocast(self.dev.type, dtype=self.torch.bfloat16, enabled=self.amp):
            poses = np.asarray(self.agent.compute_trajectory(ai).poses, np.float64)
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.dev)
        t2 = time.perf_counter()
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise RuntimeError(f"WA-JEPA returned poses of shape {poses.shape}, finite {np.isfinite(poses).all()}")
        return {"poses": poses, "hist": P, "ms": {"prep_in": 1e3 * (t1 - t0), "infer": 1e3 * (t2 - t1)}}
