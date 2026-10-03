"""Real vs render, HUGSIM nuScenes scenes: render each scene's 3DGS at its own recorded camera poses (meta_data.json camtoworld,
intrinsics and dynamic-object poses: the training views, the renderer's best case), the three front cameras x 180 frames, as
experiments/hugsim/archive/render_check.py renders them. Runs in envs/hugsim from the HUGSIM repo root:
  cd $DATA_DIR/third_party/HUGSIM && CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/hugsim/bin/python \
     $DATA_DIR/jev-drive/experiments/leaderboard_audit/scripts/rvr_hugsim_render.py scene-0010 scene-0013 ...
Writes $DATA_DIR/runs/real_vs_render/hugsim/<scene>/render.npy (180, 3, 450, 800, 3) uint8, same layout as real.npy.
--env: render instead as the closed-loop env would along the logged trajectory (experiments/hugsim/archive/pairs_render.Scene: ego on
the env's ground height, yaw only, camera = ego @ v2front @ inv(v2c) @ cam_rect, i.e. 0.3 m lower and level, simulator intrinsics, no
recorded dynamic objects) -> render_env.npy. No real counterpart: it shows what the closed loop adds on top of the training-view render."""
import json, os, sys
from glob import glob
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from scene.obj_model import ObjModel  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
HS = D / "datasets/hugsim/scenes/nuscenes"
OUT = D / "runs/real_vs_render/hugsim"
CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def load_scene(d):
    cfg = OmegaConf.load(d / "cfg.yaml")
    g = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
    params, _ = torch.load(d / "scene.pth", weights_only=False)
    g.restore(params, None)
    dyn = {}
    for f in glob(str(d / "dynamic_*.pth")):
        iid = Path(f).stem.split("_", 1)[1]
        dyn[iid] = ObjModel(cfg.model.sh_degree, feat_mutable=False)
        params, _ = torch.load(f, weights_only=False)
        dyn[iid].restore(list(params), None)
    bg = torch.tensor([1, 1, 1] if cfg.model.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    return g, dyn, bg


def main_env(names):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hugsim/archive"))
    import pairs_render as PR
    for name in names:
        o = OUT / name
        if (o / "render_env.npy").exists():
            continue
        S = PR.Scene("nuscenes", name)
        frames = json.load(open(HS / name / "meta_data.json"))["frames"]
        P = np.array([f["camtoworld"] for f in frames if "/CAM_FRONT/" in f["rgb_path"]], float)
        arr = np.lib.format.open_memmap(o / "render_env.npy.tmp", "w+", np.uint8, (len(P), 3, 450, 800, 3))
        for i, c2w in enumerate(P):
            a, b, th = c2w[0, 3], c2w[2, 3], np.arctan2(c2w[0, 2], c2w[2, 2])
            cw = S.c2ws(a, b, th)
            for c, cam in enumerate(CAMS):
                arr[i, c] = PR.to_rgb(S.render(cam, cw[cam]))
        arr.flush()
        del arr
        os.replace(o / "render_env.npy.tmp", o / "render_env.npy")
        print("rendered env", name, flush=True)
        del S
        torch.cuda.empty_cache()


def main():
    if sys.argv[1] == "--env":
        return main_env(sys.argv[2:])
    for name in sys.argv[1:]:
        o = OUT / name
        o.mkdir(parents=True, exist_ok=True)
        if (o / "render.npy").exists():
            continue
        g, dyn, bg = load_scene(HS / name)
        frames = json.load(open(HS / name / "meta_data.json"))["frames"]
        by = {c: [f for f in frames if f"/{c}/" in f["rgb_path"]] for c in CAMS}
        arr = np.lib.format.open_memmap(o / "render.npy.tmp", "w+", np.uint8, (len(by[CAMS[0]]), 3, 450, 800, 3))
        for c, cam in enumerate(CAMS):
            for i, f in enumerate(by[cam]):
                dd = {k: torch.tensor(v, dtype=torch.float32).cuda() for k, v in f.get("dynamics", {}).items()}
                view = Camera(K=np.array(f["intrinsics"]), c2w=np.array(f["camtoworld"]), width=f["width"], height=f["height"],
                              image=np.zeros((f["height"], f["width"], 3)), image_name="", timestamp=f.get("timestamp", -1), dynamics=dd)
                with torch.no_grad():
                    img = render(viewpoint=view, prev_viewpoint=None, pc=g, dynamic_gaussians=dyn, unicycles={}, bg_color=bg)["render"]
                arr[i, c] = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        arr.flush()
        del arr
        os.replace(o / "render.npy.tmp", o / "render.npy")
        print("rendered", name, flush=True)
        del g, dyn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
