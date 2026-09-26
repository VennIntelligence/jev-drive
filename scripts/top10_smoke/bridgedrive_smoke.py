"""BridgeDrive (LEAD/TFv6 post-train, arXiv 2509.23589) single-sample smoke on real CARLA sensor input.

Two roles in one file:
  1. Leaderboard agent (loaded by scripts/b2d_run.py --agent this_file): IS the author's BridgeDrive
     SensorAgent, so the leaderboard attaches the author's own rig and BridgeDrive drives the route as shipped.
     It logs the hero pose every tick and, every `CAPTURE_EVERY` ticks, dumps the exact tensors fed to the
     network (after the author's GPS filter, JPEG round trip, LiDAR accumulation/rasterisation, radar prep).
  2. Offline smoke (python this_file --capture-dir D --model-dir M --out O): loads one captured frame, runs the
     author's ClosedLoopInference 5 warmup + 20 timed calls, and compares the predicted waypoints with the path
     the car actually drove next (closed loop under BridgeDrive itself, so ADE is self-consistency, not an expert log).
Env: envs/bridgedrive, PYTHONPATH=<lead>, LEAD_CLOSED_LOOP_CONFIG as in the author's eval_bench2drive_bridgedrive.sh.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CAPTURE_EVERY, CAPTURE_FROM = 10, 60


def _patch_author_config():
    """Two runtime shims, no source edits. (1) TrainingConfig.gpu_name raises on any GPU outside the author's list
    (ours: RTX PRO 6000 Blackwell); it only feeds training-time mixed-precision switches, so an unknown card maps to
    "" (the author's own no-GPU value). (2) The released config_closed_loop.py reads self.debug_mode in every
    produce_* property but never defines it (AttributeError at the first run_step); define it False, as its
    '# Shu' branches intend for evaluation."""
    import lead.inference.config_closed_loop as cc
    import lead.training.config_training as ct
    cc.ClosedLoopConfig.debug_mode = False
    get = ct.TrainingConfig.gpu_name.fget

    def safe(self):
        try:
            return get(self)
        except Exception:
            return ""
    ct.TrainingConfig.gpu_name = property(safe)


def _cpu(v):
    return v.detach().cpu() if isinstance(v, torch.Tensor) else v


# ----------------------------------------------------------------------------------------- role 1: agent
try:
    from lead.inference.sensor_agent_bridgedrive import SensorAgent
    _patch_author_config()

    def get_entry_point():
        return "CaptureAgent"

    class CaptureAgent(SensorAgent):
        def setup(self, path_to_conf_file, *a):
            from unittest import mock
            with mock.patch("shutil.which", return_value="/bin/true"):  # setup demands ffmpeg for videos we never make
                super().setup(path_to_conf_file, *a)
            self.cap = Path(os.environ["BD_CAPTURE_DIR"])
            self.cap.mkdir(parents=True, exist_ok=True)
            self.pose_f = open(self.cap / "poses.jsonl", "w")
            fwd = self.closed_loop_inference.forward

            def hooked(data):  # keep the exact network input of this tick
                self._last = (data, fwd(data))
                return self._last[1]
            self.closed_loop_inference.forward = hooked
            self._last = None

        def run_step(self, input_data, timestamp, *a):
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            if self.step == -1:  # raw rig as delivered by CARLA, once
                (self.cap / "rig.json").write_text(json.dumps({
                    "sensors": self.sensors(),
                    "raw_shapes": {k: list(np.shape(v[1])) for k, v in input_data.items()}}, indent=1, default=str))
            self._last = None
            ctrl = super().run_step(input_data, timestamp, *a)
            hero = CarlaDataProvider.get_hero_actor()
            tf, v = hero.get_transform(), hero.get_velocity()
            self.pose_f.write(json.dumps({"step": self.step, "t": timestamp, "x": tf.location.x, "y": tf.location.y,
                                          "yaw": tf.rotation.yaw, "speed": float(np.hypot(v.x, v.y)),
                                          "steer": ctrl.steer, "throttle": ctrl.throttle, "brake": ctrl.brake}) + "\n")
            self.pose_f.flush()
            if self._last is not None and self.step >= CAPTURE_FROM and self.step % CAPTURE_EVERY == 0:
                data, pred = self._last
                d = {k: _cpu(v) for k, v in data.items()}
                d["rgb"] = d["rgb"].round().clamp(0, 255).to(torch.uint8)  # JPEG-decoded, lossless as uint8
                live = {k: _cpu(getattr(pred, k)) for k in ("pred_future_waypoints", "pred_route",
                                                            "pred_target_speed_scalar", "pred_target_speed_distribution")}
                live.update(steer=pred.steer, throttle=pred.throttle, brake=pred.brake)
                torch.save({"step": self.step, "t": timestamp, "data": d, "live": live},
                           self.cap / f"frame_{self.step:05d}.pth")
            return ctrl

        def destroy(self, *a):  # the author's destroy only compresses videos (ffmpeg); none are produced
            self.pose_f.close()
except ImportError:  # offline role without leaderboard on the path
    pass


# ----------------------------------------------------------------------------------------- role 2: offline
def to_ego(poses, p0):
    """World (CARLA, left-handed) -> ego frame of p0: x forward, y right."""
    yaw = np.deg2rad(p0["yaw"])
    d = np.stack([poses[:, 0] - p0["x"], poses[:, 1] - p0["y"]], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame", default="", help="frame_XXXXX.pth; default: first frame at >= 4 m/s with 4 s of future")
    a = p.parse_args()
    from lead.expert.config_expert import ExpertConfig
    from lead.inference.closed_loop_inference_bridgedrive import ClosedLoopInference
    from lead.inference.config_closed_loop import ClosedLoopConfig
    from lead.training.config_training import TrainingConfig
    _patch_author_config()

    cap, out = Path(a.capture_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    poses = [json.loads(l) for l in open(cap / "poses.jsonl")]
    by_step = {r["step"]: r for r in poses}
    fps = 20
    if a.frame:
        f = cap / a.frame
    else:
        cands = sorted(cap.glob("frame_*.pth"))
        ok = [c for c in cands if by_step.get(int(c.stem[6:]), {}).get("speed", 0) >= 4.0
              and int(c.stem[6:]) + 4 * fps <= poses[-1]["step"]]
        f = (ok or cands)[0]
    rec = torch.load(f, weights_only=False)
    step = rec["step"]

    cfg_c = ClosedLoopConfig()
    cfg_t = TrainingConfig(json.load(open(Path(a.model_dir) / "config.json")))
    cfg_t.step_num = cfg_c.step_num
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    cli = ClosedLoopInference(config_training=cfg_t, config_closed_loop=cfg_c, config_expert=ExpertConfig(),
                              model_path=a.model_dir, device=dev, prefix="model")
    data = {k: (v.to(dev, torch.float32) if isinstance(v, torch.Tensor) else v) for k, v in rec["data"].items()}

    outs, times = [], []
    with torch.inference_mode():
        for i in range(25):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pr = cli.forward(data)
            torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1e3)
            outs.append(pr)
    t = np.array(times[5:])
    pr = outs[-1]
    wp = pr.pred_future_waypoints[0].cpu().numpy()
    route = pr.pred_route[0].cpu().numpy()
    wp_spread = float(np.max([np.abs(o.pred_future_waypoints[0].cpu().numpy() - wp).max() for o in outs]))

    # logged future (what the car drove next), sampled at the waypoint times
    dt_wp = cfg_t.waypoints_spacing / fps  # waypoint spacing in seconds
    fut_steps = [step + round((k + 1) * dt_wp * fps) for k in range(len(wp))]
    fut = np.array([[by_step[s]["x"], by_step[s]["y"]] for s in fut_steps if s in by_step])
    gt = to_ego(fut, by_step[step]) if len(fut) else np.zeros((0, 2))
    n = len(gt)
    ade = float(np.linalg.norm(wp[:n] - gt, axis=1).mean()) if n else None
    fde = float(np.linalg.norm(wp[n - 1] - gt[-1])) if n else None
    head = lambda xy: float(np.degrees(np.arctan2(xy[1], xy[0])))

    shapes = {k: list(v.shape) if hasattr(v, "shape") else str(v) for k, v in rec["data"].items()}
    s = {
        "model": "BridgeDrive m2_k60 (liushu-ethz/BridgeDrive, model_BridgeDrive_m2_k60_0030.pth)",
        "sample": {"capture_dir": str(cap), "frame": f.name, "step": step, "sim_t": rec["t"],
                   "ego_speed_mps": by_step[step]["speed"]},
        "input_shapes": shapes,
        "command_onehot": rec["data"]["command"].tolist(), "next_command_onehot": rec["data"]["next_command"].tolist(),
        "target_points_ego": {k: rec["data"][k].tolist() for k in ("target_point_previous", "target_point", "target_point_next")},
        "latency_ms": {"mean": float(t.mean()), "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
                       "n_timed": len(t), "n_warmup": 5, "batch": 1,
                       "span": "ClosedLoopInference.forward: TFv6 backbone + BEV/det/depth/sem heads + BridgeDrive "
                               f"diffusion-bridge head ({cfg_c.step_num} ODE steps) + ensemble decode + both PID "
                               "controllers; inputs already on GPU, sensor preprocessing excluded; shared GPU"},
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "output": {
            "waypoints_ego_xy_m": wp.round(3).tolist(), "waypoint_dt_s": dt_wp, "horizon_s": dt_wp * len(wp),
            "route_ego_xy_m": route.round(3).tolist(),
            "target_speed_mps": float(pr.pred_target_speed_scalar.item()),
            "target_speed_distribution": pr.pred_target_speed_distribution[0].cpu().numpy().round(4).tolist(),
            "target_speed_classes": list(cfg_t.target_speed_classes),
            "control": {"steer": pr.steer, "throttle": pr.throttle, "brake": pr.brake,
                        "modalities": [cfg_c.steer_modality, cfg_c.throttle_modality, cfg_c.brake_modality]},
            "frame": "ego vehicle frame, x forward, y right (CARLA), metres",
            "max_abs_diff_over_25_calls_m": wp_spread,
        },
        "live_prediction_same_tick": {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in rec["live"].items()},
        "sanity": {"logged_future_ego_xy_m": gt.round(3).tolist(), "ade_m": ade, "fde_m": fde,
                   "pred_heading_deg_last_wp": head(wp[-1]), "logged_heading_deg_last": head(gt[-1]) if n else None,
                   "note": "logged future = path BridgeDrive itself drove after this tick (closed loop)"},
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    }
    if (cap / "rig.json").exists():
        s["rig"] = json.load(open(cap / "rig.json"))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("sample", "latency_ms", "peak_vram_gb", "sanity")}, indent=1))
    print("waypoints", wp.round(2).tolist(), "\nroute", route.round(2).tolist(), "\ntarget speed", s["output"]["target_speed_mps"])


if __name__ == "__main__":
    main()
