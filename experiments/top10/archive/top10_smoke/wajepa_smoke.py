"""WA-JEPA single-sample smoke on one real NAVSIM navtest token (top-10 install check).

Runs the repo's own NAVSIM adapter (eval/navsim_agent.py: SceneLoader -> WorldModelFeatureBuilder ->
predict_trajectory), exactly as eval/navsim_export_trajectory_cache.py does, and compares the plan to the
logged future. Run from the WA-JEPA checkout on the box:
  OPENBLAS_CORETYPE=Haswell CUDA_VISIBLE_DEVICES=<gpu> $DATA_DIR/envs/wajepa/bin/python jev_smoke.py
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
DATA = Path(os.environ["DATA_DIR"])
NAVSIM = Path(os.environ.get("OPENSCENE_DATA_ROOT", DATA / "datasets/navsim"))
CFG = REPO / "configs/wa_jepa_navsim_epdms.yaml"
CKPT = DATA / "models/wajepa/model_state_dict.pt"
FILTER = DATA / "third_party/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
# The full model state dict (encoder included) is restored strictly, so the V-JEPA 2.1 init is not needed
# (same as configs/wa_jepa_hugsim.yaml).
OVERRIDES = ["model.require_pretrained=false", "model.vjepa2_ckpt=null"]


def main():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from eval.navsim_agent import WorldModelNavsimAgent

    out = DATA / "runs/top10_smoke/wajepa" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    agent = WorldModelNavsimAgent(config_path=str(CFG), checkpoint_path=str(CKPT), device="cuda",
                                  config_overrides=OVERRIDES)
    t0 = time.time(); agent.initialize(); load_s = time.time() - t0
    model = agent.model
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    n_params = sum(p.numel() for p in model.parameters())

    sf = instantiate(OmegaConf.load(FILTER))
    sf.log_names = sf.log_names[:1]
    loader = SceneLoader(original_sensor_path=NAVSIM / "sensor_blobs/test", data_path=NAVSIM / "navsim_logs/test",
                         scene_filter=sf, sensor_config=agent.get_sensor_config())
    # First token of the first navtest log whose logged future moves > 10 m (heading check is meaningful).
    for token in loader.tokens:
        gt = loader.get_scene_from_token(token).get_future_trajectory(num_trajectory_frames=8).poses
        if np.linalg.norm(gt[-1, :2]) > 10:
            break
    agent_input = loader.get_agent_input_from_token(token)

    builder = agent.get_feature_builders()[0]
    t0 = time.time(); feats = builder.compute_features(agent_input); feat_ms = (time.time() - t0) * 1e3
    feats = {k: v.unsqueeze(0).cuda() for k, v in feats.items()}
    shapes = {k: list(v.shape) for k, v in feats.items()}

    def bench(amp):  # amp=False: NAVSIM eval path (fp32); amp=True: HUGSIM path (fp32 weights, bf16 autocast)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for _ in range(5):
                model.predict_trajectory(feats)
            ts = []
            for _ in range(20):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pred = model.predict_trajectory(feats)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        return pred.squeeze(0).float().cpu().numpy(), dict(
            mean=float(ts.mean()), p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)),
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))

    pred, lat_fp32 = bench(False)
    pred_bf16, lat_bf16 = bench(True)
    traj = agent.compute_trajectory(agent_input).poses  # full adapter path, same seed -> same plan
    ade = float(np.linalg.norm(traj[:, :2] - gt[:, :2], axis=1).mean())
    fde = float(np.linalg.norm(traj[-1, :2] - gt[-1, :2]))
    hdg = lambda p: float(np.degrees(np.arctan2(p[-1, 1], p[-1, 0])))
    summary = dict(
        model="WA-JEPA", checkpoint=str(CKPT), config=str(CFG), overrides=OVERRIDES, token=token,
        log=sf.log_names[0], gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        param_dtypes=dtypes, n_params=n_params, load_s=round(load_s, 1), feature_build_ms=round(feat_ms, 1),
        input_shapes=shapes,
        navigation_command=int(feats["navigation_command"]), ego_status=feats["ego_status"][0].tolist(),
        latency_span="model.predict_trajectory on pre-built features (V-JEPA ViT-L encode 4 hist x 4 cams + "
                     "4 flow steps), batch 1, 5 warmup + 20 timed, GPU shared with other jobs",
        latency_fp32_navsim_path=lat_fp32, latency_bf16_autocast_hugsim_path=lat_bf16,
        ade_bf16_autocast_m=float(np.linalg.norm(pred_bf16[:, :2] - gt[:, :2], axis=1).mean()),
        output=dict(horizon_s=4.0, hz=2, frame="ego at current frame, x forward, y left, heading rad",
                    trajectory=traj.tolist(), timed_call_equals_adapter=bool(np.allclose(pred, traj, atol=1e-3))),
        logged_future=gt.tolist(), ade_m=ade, fde_m=fde, heading_end_pred_deg=hdg(traj), heading_end_gt_deg=hdg(gt),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("logged_future",)}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
