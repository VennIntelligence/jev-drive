"""factor_wm: metric depth of every packed model frame of a clip set (envs/depth, one GPU): UniDepth v2 ViT-L/14 with the openpilot model-frame
intrinsics (road f 910, wide f 455 on 512 x 256), forward distance (= camera z in the level calibrated frame), scaled per clip and view so that
the road surface matches the road plane at the camera height (median of plane / predicted depth over the near road patch of every frame).
Output: $DATA_DIR/runs/factor_wm/clips/<set>/depth.npy (n, NF, 2, 128, 256) float16 + depth_scale.json.

  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/depth/bin/python experiments/factor_wm/scripts/fw_depth.py g0a g0b [--limit 2]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

PATCH = {"road": (slice(80, 256), slice(156, 356)), "wide": (slice(180, 256), slice(156, 356))}


def plane_lam(view, cam_z):
    r = I._rays(view)
    down = -r[..., 2]
    lam = np.where(down > 1e-6, cam_z / np.maximum(down, 1e-6), np.inf)
    return np.minimum(lam, I.D_FAR / np.linalg.norm(r, axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from unidepth.models import UniDepthV2
    m = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda").eval()
    root = Path(os.environ.get("FW_ROOT") or data_dir() / "runs" / "factor_wm")
    for name in a.sets:
        d = root / "clips" / name
        out_p = d / ("depth.npy" if not a.limit else "depth_smoke.npy")
        if out_p.exists():
            print(name, "exists")
            continue
        imgs = np.load(d / "imgs.npy", mmap_mode="r")
        cam = np.load(d / "tab.npz", allow_pickle=True)["cam"]
        n = a.limit or len(imgs)
        nf = imgs.shape[1]
        out = np.zeros((n, nf, 2, 128, 256), np.float16)
        scales = []
        t0 = time.time()
        for c in range(n):
            sc = []
            for k, view in enumerate(("road", "wide")):
                K = torch.as_tensor(I.OP_K[view], dtype=torch.float32, device="cuda")
                rgb = (I.to_rgb(torch.from_numpy(np.asarray(imgs[c, :, k])).cuda()) * 255).clamp(0, 255).to(torch.uint8)   # (nf, 3, 256, 512)
                D = []
                with torch.inference_mode():
                    for b in range(nf):                                                    # UniDepth's camera takes one image per K
                        D.append(m.infer(rgb[b: b + 1], K[None].clone())["depth"][:, 0].float())
                D = torch.cat(D)                                                                   # (nf, 256, 512)
                pl = torch.as_tensor(plane_lam(view, float(cam[c][2])), dtype=torch.float32, device="cuda")
                ys, xs = PATCH[view]
                ok = (pl[ys, xs] > 4) & (pl[ys, xs] < 30)
                ratio = (pl[ys, xs][None] / D[:, ys, xs].clamp(min=0.5))[:, ok]
                s = float(ratio.median(dim=1).values.median().clamp(0.5, 2.0))
                sc.append(s)
                D = (D * s).clamp(1.0, 120.0)
                out[c, :, k] = F.avg_pool2d(D[:, None], 2)[:, 0].cpu().numpy().astype(np.float16)
            scales.append(sc)
            if c % 10 == 0:
                print(f"{name}: {c + 1}/{n} clips, {time.time() - t0:.0f} s, scale road/wide {sc[0]:.2f}/{sc[1]:.2f}", flush=True)
        tmp = out_p.with_suffix(".tmp.npy")
        np.save(tmp, out)
        tmp.replace(out_p)
        sc = np.array(scales)
        json.dump(dict(n=n, road=dict(median=float(np.median(sc[:, 0])), p10=float(np.percentile(sc[:, 0], 10)), p90=float(np.percentile(sc[:, 0], 90))),
                       wide=dict(median=float(np.median(sc[:, 1])), p10=float(np.percentile(sc[:, 1], 10)), p90=float(np.percentile(sc[:, 1], 90))),
                       per_clip=sc.tolist()), open(d / ("depth_scale.json" if not a.limit else "depth_scale_smoke.json"), "w"), indent=1)
        print(f"{name}: done {n} clips in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
