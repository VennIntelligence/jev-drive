"""HUGSIM FIFO client for the SimScale camera-only NAVSIM models (AlpaSim e2e_challenge samples), run as shipped.

Models (third-party code and weights untouched; this file is the whole adapter, env envs/alpasim-ltf):
  ss-ltf  SimScale LTF            third_party/alpasim/e2e_challenge/sample_submission_simscale_navsim_transfuser    LtfPolicy
  ss-dd   SimScale DiffusionDrive third_party/alpasim/e2e_challenge/sample_submission_simscale_navsim_diffusiondrive DiffusionDrivePolicy
Both read CAM_L0 / CAM_F0 / CAM_R0 at 1920 x 1080 (their own preprocessing crops L0 / R0 to the central 1088 columns and 28 rows
top and bottom, stitches, resizes to 2048 x 512), a NAVSIM command one-hot and ego-frame velocity / acceleration, and return 8
poses (x forward, y left, heading) at 0.5 s.

What this client does, step for step (HUGSIM's own LTF client third_party/hugsim_deps/NAVSIM/ltf_e2e.py + hugsim/dataparser.py is
the template; deviations marked):
  cameras   HUGSIM renders a nuScenes rig at 800 x 450: CAM_FRONT -> CAM_F0, CAM_FRONT_LEFT -> CAM_L0, CAM_FRONT_RIGHT -> CAM_R0
            (the shipped dataparser's map), each resized to 1920 x 1080 with cv2.resize (bilinear, as the shipped dataparser).
            HUGSIM rgb is RGB uint8 HWC, which the policies take as is.
  command   DEVIATION (the point of this adapter): the shipped dataparser hard-codes straight (command[1] = 1, the real command
            commented out). Here HUGSIM's per-step info['command'] (0 = right, 1 = left, 2 = straight) -> NAVSIM index
            (0 = left, 1 = straight, 2 = right) via [2, 0, 1], the map WA-JEPA's shipped HUGSIM client uses.
  ego       velocity (info['ego_velo'], 0), acceleration (info['accelerate'], 0) in the ego frame, as WA-JEPA's client and
            AlpaSim's driver (rig frame). DEVIATION: the shipped dataparser rotates both by the steering angle (as if it were a yaw).
  noise     ss-dd: per-request noise seed 0 and noise_index = step (AlpaSim's driver.py pattern: session seed + a step counter),
            so a scenario's diffusion noise is reproducible.
  output    (x, y)_lidar = (-y, x)_ego, (8, 2) float64, what closed_loop.py expects (same as both shipped clients).
  failures  never send None (closed_loop.py crashes on it): a failed step sends a zero (braking) plan and is counted in
            planner_stats.json; per-step inputs and plans go to ss_steps.jsonl.

Launched by experiments/hugsim/scripts/simscale_e2e.sh (`zsh <launcher> <cuda> <output_dir>`, ad slot `ss`).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
E2E = D / "third_party/alpasim/e2e_challenge"
MODELS = {
    "ss-ltf": (E2E / "sample_submission_simscale_navsim_transfuser", "navsim_transfuser_challenge.policy", "LtfPolicy",
               D / "models/alpasim_ltf/ltf_sim_navtest.ckpt"),
    "ss-dd": (E2E / "sample_submission_simscale_navsim_diffusiondrive", "navsim_diffusiondrive_challenge.policy", "DiffusionDrivePolicy",
              D / "models/alpasim_dd/diffusiondrive_sim_navhard.ckpt"),
}
CAM_MAP = {"CAM_L0": "CAM_FRONT_LEFT", "CAM_F0": "CAM_FRONT", "CAM_R0": "CAM_FRONT_RIGHT"}
COMMAND_MAP = [2, 0, 1]                  # HUGSIM 0 = right / 1 = left / 2 = straight -> NAVSIM 0 = left / 1 = straight / 2 = right
NOISE_SEED = 0


def log(msg: str) -> None:
    print(f"[ss {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scalar(x) -> float:
    return float(np.asarray(x).ravel()[0])


def wait_pipes(obs: str, plan: str, timeout: float = 3600.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(obs) and os.path.exists(plan):
            return True
        time.sleep(0.5)
    return False


def drain(obs: str, plan: str) -> None:
    """After a fatal load error: answer every step with a braking plan until 'Done', so closed_loop.py does not block."""
    while True:
        try:
            with open(obs, "rb") as p:
                raw = p.read()
            if not raw or isinstance(pickle.loads(raw), str):
                return
            with open(plan, "wb") as p:
                p.write(pickle.dumps(np.zeros((8, 2))))
        except Exception:
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    obs_pipe, plan_pipe = os.path.join(a.output, "obs_pipe"), os.path.join(a.output, "plan_pipe")
    stats = dict(model=a.model, n_steps=0, n_failures=0, failures=[], command_hist={}, latency_ms=[], started=time.strftime("%F %T"))
    stats_f = Path(a.output) / "planner_stats.json"
    try:
        import cv2
        import torch
        root, mod, cls, ckpt = MODELS[a.model]
        sys.path.insert(0, str(root))
        m = __import__(mod, fromlist=[cls])
        t0 = time.time()
        policy = getattr(m, cls)(str(ckpt), device="cuda")            # shipped defaults: fp16 autocast, warm-up on
        stats["model_load_s"] = round(time.time() - t0, 1)
        log(f"{a.model} loaded from {ckpt} in {stats['model_load_s']} s; cameras {m.CAMERA_IDS}")
    except Exception:
        stats["fatal"] = traceback.format_exc()
        log("model failed to load:\n" + stats["fatal"])
        stats_f.write_text(json.dumps(stats, indent=1))
        if wait_pipes(obs_pipe, plan_pipe):
            drain(obs_pipe, plan_pipe)
        return 1
    if not wait_pipes(obs_pipe, plan_pipe):
        log("FIFOs never appeared")
        return 1
    steps = open(Path(a.output) / "ss_steps.jsonl", "w")
    while True:
        with open(obs_pipe, "rb") as p:
            raw = p.read()
        if not raw:
            break
        data = pickle.loads(raw)
        if isinstance(data, str):
            log(f"received {data!r}, finishing")
            break
        obs, info = data
        t0 = time.time()
        try:
            raw_cmd = int(np.asarray(info["command"]).ravel()[0])
            cmd = np.zeros(4, np.float32)
            cmd[COMMAND_MAP[raw_cmd]] = 1.0
            v, acc = scalar(info["ego_velo"]), scalar(info["accelerate"])
            imgs = {k: cv2.resize(np.ascontiguousarray(obs["rgb"][c]), (1920, 1080)) for k, c in CAM_MAP.items()}
            kw = dict(noise_seed=NOISE_SEED, noise_index=stats["n_steps"]) if a.model == "ss-dd" else {}
            req = m.InferenceInput(images=imgs, command_one_hot=cmd, velocity_xy=np.array([v, 0.0], np.float32),
                                   acceleration_xy=np.array([acc, 0.0], np.float32), **kw)
            traj = policy.predict_batch([req])[0].trajectory.astype(np.float64)       # (8, 3) x forward, y left, heading
            plan = np.ascontiguousarray(np.stack([-traj[:, 1], traj[:, 0]], 1))
            stats["command_hist"][str(raw_cmd)] = stats["command_hist"].get(str(raw_cmd), 0) + 1
            steps.write(json.dumps(dict(step=stats["n_steps"], hugsim_command=raw_cmd, navsim_command=COMMAND_MAP[raw_cmd], v=v, a=acc,
                                        traj_ego=np.round(traj, 3).tolist())) + "\n")
        except Exception:
            tb = traceback.format_exc()
            stats["n_failures"] += 1
            stats["failures"].append(dict(step=stats["n_steps"], traceback=tb[-2000:]))
            log(f"step {stats['n_steps']} failed, braking plan sent:\n{tb}")
            plan = np.zeros((8, 2))
        stats["latency_ms"].append(round((time.time() - t0) * 1000, 1))
        with open(plan_pipe, "wb") as p:
            p.write(pickle.dumps(plan))
        stats["n_steps"] += 1
    steps.close()
    stats["finished"] = time.strftime("%F %T")
    lat = stats.pop("latency_ms")
    stats["latency_ms_median"] = float(np.median(lat)) if lat else None
    stats["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    stats_f.write_text(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
