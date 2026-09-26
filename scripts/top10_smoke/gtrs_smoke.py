"""ZTRS (NVlabs Hydra family, woxihuanjiangguo/ZTRS) single-sample smoke on one real NAVSIM navtest token.

Builds the agent from the upstream ztrs_vov.yaml with the navhard inference overrides from docs/ztrs_inference.md
(8192-entry vocabulary), builds features with its own HydraFeatureBuilder, times agent.forward (batch 1, 5 warmup
+ 20 timed) and compares the selected vocabulary trajectory with the logged future.
Run on the box from ~/data/third_party/ztrs with env ~/data/envs/gtrs:
  OPENBLAS_CORETYPE=Haswell NAVSIM_DEVKIT_ROOT=$PWD CUDA_VISIBLE_DEVICES=<gpu> $DATA_DIR/envs/gtrs/bin/python jev_smoke.py [vocab]
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
NAV = DATA / "datasets/navsim"
W = DATA / "models/gtrs"
CKPT = W / "ztrs_vov.ckpt"
VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 8192  # docs/ztrs_inference.md uses traj_final/8192.npy
LOG = "2021.06.03.12.02.06_veh-35_01100_01227"  # first navtest log
MIN_SPEED = 3.0  # first scene of the log whose ego moves faster than this (m/s)
OUT = DATA / "runs/top10_smoke/gtrs" / time.strftime("%Y%m%d-%H%M%S")


def build_agent():
    cfg = OmegaConf.load(REPO / "navsim/planning/script/config/common/agent/ztrs_vov.yaml")
    cfg.pdm_gt_path = None  # 30 GB training-only label pickle
    cfg.checkpoint_path = str(CKPT)
    cfg.config.vocab_path = str(REPO / f"traj_final/{VOCAB}.npy")
    cfg.config.vov_ckpt = str(W / "dd3d_det_final.pth")  # V2-99 init, overwritten by the checkpoint
    agent = instantiate(cfg)
    agent.initialize()
    return agent.cuda().eval()


def pick_scene(sensor_config):
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[LOG])
    loader = SceneLoader(NAV / "navsim_logs/test", NAV / "sensor_blobs/test", sf, sensor_config=sensor_config)
    for token in loader.tokens:
        scene = loader.get_scene_from_token(token)
        if np.linalg.norm(scene.get_agent_input().ego_statuses[-1].ego_velocity) > MIN_SPEED:
            return token, scene


def time_it(fn, n_warm=5, n=20):
    ts = []
    with torch.no_grad():
        for i in range(n_warm + n):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts[n_warm:])
    return out, dict(mean=float(ts.mean()), p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    agent = build_agent()
    token, scene = pick_scene(agent.get_sensor_config())
    agent_input = scene.get_agent_input()
    t0 = time.perf_counter()
    feats = agent.get_feature_builders()[0].compute_features(agent_input)
    t_feat = (time.perf_counter() - t0) * 1e3
    # camera_feature: [prev, current] stitched l0|f0|r0 at 512x2048; status_feature: [current, prev] (cmd4, v2, a2)
    batch = {"camera_feature": [c[None].cuda() for c in feats["camera_feature"]],
             "status_feature": [s[None].cuda() for s in feats["status_feature"]]}

    torch.cuda.reset_peak_memory_stats()
    out, lat = time_it(lambda: agent.forward(batch))
    peak = torch.cuda.max_memory_allocated() / 2**20
    # as shipped, forward also re-runs the model on the previous frame (ec_target, output only used by the loss);
    # the selected trajectory does not depend on it (no_cond=True), so time the current-frame-only path too
    agent._config.ec_target = False
    out2, lat_cur = time_it(lambda: agent.forward(batch))
    agent._config.ec_target = True
    assert torch.equal(out["selected_indices"], out2["selected_indices"])

    pred10 = out["trajectory"][0].float().cpu().numpy()  # (40, 3) x, y, heading, 10 Hz, 4 s, ego rear-axle frame
    pred = pred10[4::5]  # 2 Hz to match the NAVSIM log
    gt = scene.get_future_trajectory(8).poses
    ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean())
    fde = float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))
    v = np.asarray(agent_input.ego_statuses[-1].ego_velocity)
    ade_cv = float(np.linalg.norm(np.stack([v * 0.5 * (i + 1) for i in range(8)]) - gt[:, :2], axis=1).mean())
    vocab = out["trajectory_vocab"].float().cpu().numpy()[:, 4::5]  # (V, 8, 3)
    vade = np.linalg.norm(vocab[..., :2] - gt[None, :, :2], axis=-1).mean(-1)
    sel = int(out["selected_indices"][0])
    heads = {k: float(out[k][0, sel].sigmoid()) for k in ["no_at_fault_collisions", "drivable_area_compliance",
             "driving_direction_compliance", "traffic_light_compliance", "time_to_collision_within_bound",
             "ego_progress", "lane_keeping", "history_comfort"]}
    hdg = lambda p: float(np.degrees(np.arctan2(p[-1, 1], p[-1, 0])))
    es = agent_input.ego_statuses[-1]
    summary = dict(
        model="ZTRS (ztrs_vov, V2-99)", checkpoint=str(CKPT), vocab=f"traj_final/{VOCAB}.npy", token=token, log=LOG,
        inputs=dict(camera_feature=[list(c.shape) for c in batch["camera_feature"]],
                    status_feature=[list(s.shape) for s in batch["status_feature"]]),
        cameras="cam_l0|cam_f0|cam_r0 stitched (l0/r0 cropped 416 px each side, 28 px top/bottom), resized to 512x2048, "
                "frames t-0.5 s and t (the t-0.5 s one only feeds the ec_target pass)",
        ego_velocity=list(map(float, es.ego_velocity)), ego_acceleration=list(map(float, es.ego_acceleration)),
        driving_command=list(map(int, es.driving_command)),
        latency_ms=dict(as_shipped=lat, current_frame_only=lat_cur,
                        span="agent.forward on GPU-resident features, fp32, batch 1: V2-99 backbone + 3-layer decoder + "
                             f"scoring of all {VOCAB} vocabulary entries + argmax; as_shipped includes a second full "
                             "pass on the previous frame (ec_target)",
                        feature_build_ms_cpu=t_feat, gpu_shared_with_other_jobs=True),
        peak_vram_mib=peak, gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        output=dict(horizon_s=4.0, hz=10, frame="ego rear-axle at current frame, x fwd, y left, heading rad",
                    trajectory_10hz=pred10.tolist(), trajectory_2hz=pred.tolist(), logged_future_2hz=gt.tolist()),
        ade_m=ade, fde_m=fde, ade_const_vel_m=ade_cv,
        vocab_oracle_ade_m=float(vade.min()), selected_index=sel, selected_vocab_ade_rank=int((vade < vade[sel]).sum()),
        selected_head_probs=heads, heading_of_travel_deg=dict(pred=hdg(pred), log=hdg(gt)),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ["token", "inputs", "ego_velocity", "driving_command", "latency_ms",
                      "peak_vram_mib", "ade_m", "fde_m", "ade_const_vel_m", "vocab_oracle_ade_m",
                      "selected_vocab_ade_rank", "selected_head_probs", "heading_of_travel_deg"]}, indent=1))
    print("pred", np.round(pred, 2).tolist()); print("log ", np.round(gt, 2).tolist()); print("->", OUT)


if __name__ == "__main__":
    main()
