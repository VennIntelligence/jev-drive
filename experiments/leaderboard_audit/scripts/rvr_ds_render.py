"""Real vs render, HUGSIM Waymo / PandaSet / KITTI-360 scenes: render each scene's 3DGS at its own recorded camera poses, intrinsics and
dynamic-object poses (training views, the renderer's best case), as rvr_hugsim_render.py does for nuScenes. Runs in envs/hugsim from the HUGSIM repo root:
  cd $DATA_DIR/third_party/HUGSIM && CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/hugsim/bin/python \
     $DATA_DIR/jev-drive/experiments/leaderboard_audit/scripts/rvr_ds_render.py <dataset> scene ...
Writes $DATA_DIR/runs/real_vs_render/<dataset>/<scene>/render.npy (n, ncam, H, W, 3) uint8, same layout as real.npy (cameras as rvr_ds_real.py)."""
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
HS = D / "datasets/hugsim/scenes"
CAMS = {"waymo": ("cam_1", "cam_2", "cam_3"), "pandaset": ("front_camera", "front_left_camera", "front_right_camera"), "kitti360": ("cam_0",)}


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


def main():
    ds = sys.argv[1]
    cams = CAMS[ds]
    for name in sys.argv[2:]:
        o = D / "runs/real_vs_render" / ds / name
        o.mkdir(parents=True, exist_ok=True)
        if (o / "render.npy").exists():
            continue
        g, dyn, bg = load_scene(HS / ds / name)
        frames = json.load(open(HS / ds / name / "meta_data.json"))["frames"]
        by = {c: [f for f in frames if f"/{c}/" in f["rgb_path"]] for c in cams}
        f0 = by[cams[0]][0]
        arr = np.lib.format.open_memmap(o / "render.npy.tmp", "w+", np.uint8, (len(by[cams[0]]), len(cams), f0["height"], f0["width"], 3))
        for c, cam in enumerate(cams):
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
        print("rendered", ds, name, flush=True)
        del g, dyn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
