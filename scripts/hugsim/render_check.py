"""Render-correctness and speed check for a ported HUGSIM scene (run in the hugsim env, from the HUGSIM repo).

Renders the exported scene at the recorded camera poses of its own meta_data.json (with the recorded
dynamic-object poses) and compares against the original camera images, e.g. the sample_data copy of the
same scene. Held-out frames follow HUGSIM's own nuScenes split (idx % 30 >= 24). Also times the renderer:
single 800x450 views and the full 6-camera observation, and records peak VRAM.

  cd $DATA_DIR/third_party/HUGSIM && CUDA_VISIBLE_DEVICES=1 $DATA_DIR/envs/hugsim/bin/python \
    $DATA_DIR/jev-drive/scripts/hugsim/render_check.py --scene <exported scene dir> --images <raw scene dir> --out <dir>
"""
import argparse
import csv
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
from imageio.v2 import imread

sys.path.insert(0, os.getcwd())  # HUGSIM repo root
from omegaconf import OmegaConf  # noqa: E402

from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from scene.obj_model import ObjModel  # noqa: E402


def load_scene(scene_dir):
    cfg = OmegaConf.load(os.path.join(scene_dir, 'cfg.yaml'))
    g = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
    params, _ = torch.load(os.path.join(scene_dir, 'scene.pth'), weights_only=False)
    g.restore(params, None)
    dyn = {}
    for f in glob(os.path.join(scene_dir, 'dynamic_*.pth')):
        iid = Path(f).stem.split('_', 1)[1]
        dyn[iid] = ObjModel(cfg.model.sh_degree, feat_mutable=False)
        params, _ = torch.load(f, weights_only=False)
        dyn[iid].restore(list(params), None)
    bg = torch.tensor([1, 1, 1] if cfg.model.white_background else [0, 0, 0], dtype=torch.float32, device='cuda')
    return g, dyn, bg


def camera(frame, with_dyn):
    h, w = frame['height'], frame['width']
    dyn = {k: torch.tensor(v, dtype=torch.float32).cuda() for k, v in frame.get('dynamics', {}).items()} if with_dyn else {}
    return Camera(K=np.array(frame['intrinsics']), c2w=np.array(frame['camtoworld']), width=w, height=h,
                  image=np.zeros((h, w, 3)), image_name='', timestamp=frame.get('timestamp', -1), dynamics=dyn)


def psnr(a, b):
    return float(-10 * np.log10(np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='exported scene dir (scene.pth, cfg.yaml, meta_data.json)')
    ap.add_argument('--images', required=True, help='raw scene dir whose frames rgb_path points into')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=1, help='evaluate every n-th frame')
    ap.add_argument('--fig_frames', type=int, nargs='*', default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    g, dyn, bg = load_scene(args.scene)
    n_gs = g.get_full_xyz.shape[0] if g.ground_model is not None else g.get_xyz.shape[0]
    frames = json.load(open(os.path.join(args.scene, 'meta_data.json')))['frames']
    kw = dict(pc=g, dynamic_gaussians=dyn, unicycles={}, bg_color=bg)

    def rgb(cam):
        with torch.no_grad():
            img = render(viewpoint=cam, prev_viewpoint=None, **kw)['render']
        return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    rows = []
    for idx in range(0, len(frames), args.stride):
        f = frames[idx]
        gt = imread(os.path.join(args.images, f['rgb_path'].replace('./', '')))[..., :3]
        pred = rgb(camera(f, with_dyn=True))
        rows.append({'idx': idx, 'cam': f['rgb_path'].split('/')[-2], 'split': 'test' if idx % 30 >= 24 else 'train',
                     'psnr': round(psnr(pred, gt), 3)})
        if args.fig_frames and idx in args.fig_frames:
            np.savez_compressed(out / f'frame_{idx:05d}.npz', pred=pred, gt=gt)
    with open(out / 'render_psnr.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # speed: one 800x450 view, and the 6 cameras of one timestep, as the env renders them (sequentially)
    by_t = {}
    for f in frames:
        by_t.setdefault(f['timestamp'], []).append(f)
    six = next(v for v in by_t.values() if len(v) == 6)
    cams6 = [camera(f, with_dyn=True) for f in six]
    for _ in range(10):
        rgb(cams6[0])
    torch.cuda.synchronize()
    n = 200
    t0 = time.perf_counter()
    for i in range(n):
        with torch.no_grad():
            render(viewpoint=cams6[i % 6], prev_viewpoint=None, **kw)
    torch.cuda.synchronize()
    t_view = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(20):
        [rgb(c) for c in cams6]
    t_obs6 = (time.perf_counter() - t0) / 20

    test = [r['psnr'] for r in rows if r['split'] == 'test']
    train = [r['psnr'] for r in rows if r['split'] == 'train']
    summary = {'scene': args.scene, 'n_gaussians': int(n_gs), 'n_dynamic': len(dyn),
               'psnr_test_mean': round(float(np.mean(test)), 3) if test else None, 'n_test': len(test),
               'psnr_train_mean': round(float(np.mean(train)), 3), 'n_train': len(train),
               'ms_per_view_gpu': round(t_view * 1e3, 3), 'fps_view': round(1 / t_view, 1),
               'ms_per_6cam_obs_with_readback': round(t_obs6 * 1e3, 2),
               'peak_vram_gb_torch': round(torch.cuda.max_memory_allocated() / 1e9, 3),
               'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}
    json.dump(summary, open(out / 'render_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
