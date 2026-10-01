"""DrivoR (valeoai/DrivoR) single-sample smoke on one real NAVSIM navtest token.

Builds features with DrivoR's own feature builder, runs the NAVSIM-v1 checkpoint (drivor_Nav1_25epochs.pth,
eval overrides from the upstream README), times the model forward (batch 1, 5 warmup + 20 timed) and
compares the selected trajectory with the logged future.
Run on the box from ~/data/third_party/drivor with env ~/data/envs/drivor.
"""
import json, os, time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
REPO = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "models/drivor"
CKPT = MODELS / "drivor_Nav1_25epochs.pth"
OUT = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "runs/top10_smoke/drivor" / time.strftime("%Y%m%d-%H%M%S")

# NAVSIM-v1 eval overrides, copied from the upstream README (run_pdm_score_multi_gpu.py command)
NAV1 = dict(proposal_num=64, refiner_ls_values=0.0, one_token_per_traj=True, refiner_num_heads=1, tf_d_model=256,
            tf_d_ffn=1024, area_pred=False, agent_pred=False, ref_num=4, noc=1, dac=1, ddc=0.0, ttc=5, ep=5, comfort=2)


def build_agent():
    cfg = OmegaConf.load(REPO / "navsim/planning/script/config/common/agent/drivoR.yaml")
    cfg.config.update(NAV1)
    cfg.config.image_backbone.focus_front_cam = False
    cfg.config.image_backbone.model_weights = str(MODELS / "vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")
    cfg.checkpoint_path, cfg.scheduler_args, cfg.batch_size = str(CKPT), None, 1
    agent = instantiate(cfg)
    agent.initialize()
    return agent.cuda().eval()


def pick_scene(sensor_config):
    nt = OmegaConf.load(REPO / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    nt = OmegaConf.to_container(nt)
    nt.pop("_target_"), nt.pop("_convert_")
    nt["log_names"] = nt["log_names"][:1]  # one log pickle is enough
    loader = SceneLoader(DATA / "navsim_logs/test", DATA / "sensor_blobs/test", SceneFilter(**nt), sensor_config)
    token = loader.tokens[0]
    return token, loader.get_scene_from_token(token)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    agent = build_agent()
    token, scene = pick_scene(agent.get_sensor_config())
    agent_input = scene.get_agent_input()
    t0 = time.perf_counter()
    feats = agent.get_feature_builders()[0].compute_features(agent_input)
    t_feat = (time.perf_counter() - t0) * 1e3
    batch = {k: v[None].float().cuda() for k, v in feats.items()}

    torch.cuda.reset_peak_memory_stats()
    ts = []
    with torch.no_grad():
        for i in range(25):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = agent.forward(batch)
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts[5:])
    peak = torch.cuda.max_memory_allocated() / 2**20

    pred = out["trajectory"][0].cpu().numpy()  # (8, 3) x, y, heading, ego rear-axle frame
    gt = scene.get_future_trajectory(8).poses  # (8, 3) same frame, 0.5 s steps
    ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean())
    fde = float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))
    # constant-velocity reference for scale
    v = np.asarray(agent_input.ego_statuses[-1].ego_velocity)
    cv = np.stack([v * 0.5 * (i + 1) for i in range(8)])
    ade_cv = float(np.linalg.norm(cv - gt[:, :2], axis=1).mean())
    hdg = lambda p: float(np.degrees(np.arctan2(p[-1, 1], p[-1, 0])))
    props = out["proposals"][0].cpu().numpy()
    oracle = float(np.linalg.norm(props[..., :2] - gt[None, :, :2], axis=-1).mean(-1).min())

    es = agent_input.ego_statuses[-1]
    summary = dict(
        model="DrivoR", checkpoint=str(CKPT), token=token, log=scene.scene_metadata.log_name,
        inputs={k: list(v.shape) for k, v in batch.items()},
        ego_status_last=batch["ego_status"][0, -1].tolist(), driving_command=list(map(int, es.driving_command)),
        cameras="f0,b0,l0,r0 (order in the image tensor), current frame only",
        latency_ms=dict(mean=float(ts.mean()), p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)),
                        span="agent.forward on GPU (backbone+decoder+scorer+argmax), features preloaded on GPU, fp32, batch 1",
                        feature_build_ms_cpu=t_feat, gpu_shared_with_other_jobs=True),
        peak_vram_mib=peak,
        output=dict(horizon_s=4.0, hz=2, frame="ego rear-axle at current frame, x fwd, y left, heading rad",
                    trajectory=pred.tolist(), logged_future=gt.tolist()),
        ade_m=ade, fde_m=fde, ade_const_vel_m=ade_cv, best_of_64_proposals_ade_m=oracle,
        heading_of_travel_deg=dict(pred=hdg(pred), log=hdg(gt)),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ["token", "inputs", "driving_command", "latency_ms", "peak_vram_mib", "ade_m",
                                              "fde_m", "ade_const_vel_m", "best_of_64_proposals_ade_m", "heading_of_travel_deg"]}, indent=1))
    print("pred", np.round(pred, 2).tolist()); print("log ", np.round(gt, 2).tolist()); print("->", OUT)


if __name__ == "__main__":
    main()
