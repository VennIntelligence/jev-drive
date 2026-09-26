"""I3 at 10 Hz with CAM_BACK (todos/2026-09-26-top10-intersection.md, [T2] 10:05 entry, choice 1): the same scenes,
worlds and actor tracks as processed/hugsim_pairs, re-rendered with I3's own renderer (scripts/hugsim/pairs_render.py)
at 0.1 s instead of 0.2 s and with CAM_BACK added to the three front cameras. Nothing is recomputed: the actor tracks,
the render window and the world validity are read from the original meta.json.

Output: processed/hugsim_pairs_10hz/scenes/<key>/<world>/cams/<front,front_left,front_right,back>/<frame>.jpg with
frame = 2 x 10 Hz index (20 Hz ticks, like the original), frames.jsonl per world and meta.json (the original one plus
`rate_dt`, `t_render_10hz`, the CAM_BACK calibration and the check below).

Check: every front-camera JPEG at a 5 Hz time (even 10 Hz index) is compared with the original file: bytes equal, and
when not, the max |pixel difference| after decoding. The run log reports both per scene.

  cd $DATA_DIR/third_party/HUGSIM && CUDA_VISIBLE_DEVICES=4 $DATA_DIR/envs/hugsim/bin/python \
      $REPO/scripts/hugsim/pairs_render_10hz.py [--keys k ...] [--shard i n]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairs_render as R  # noqa: E402  (imports HUGSIM from the cwd, like the original renderer)
from pairs_render import H, RunLog  # noqa: E402
from sim.utils.sim_utils import fov2focal, load_camera_cfg  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DT = 0.1
CAMS = H.CAMS + ("CAM_BACK",)
KEYS = H.CAM_KEYS + ("back",)
SRC, DST = "hugsim_pairs", "hugsim_pairs_10hz"


def root(name, *parts) -> Path:
    return H.data_dir() / "processed" / name / Path(*parts)


def add_back(S: R.Scene):
    """CAM_BACK with the same construction as Scene.__init__ uses for the front cameras."""
    cam_params, _, rect = load_camera_cfg(OmegaConf.load(f"configs/sim/{S.ds}_camera.yaml"))
    p = cam_params["CAM_BACK"]
    K = np.eye(4)
    K[0, 0] = fov2focal(p["intrinsic"]["fovx"], p["intrinsic"]["W"])
    K[1, 1] = fov2focal(p["intrinsic"]["fovy"], p["intrinsic"]["H"])
    K[0, 2], K[1, 2] = p["intrinsic"]["cx"], p["intrinsic"]["cy"]
    S.cams["CAM_BACK"] = (K, cam_params["CAM_FRONT"]["v2c"] @ np.linalg.inv(p["v2c"]) @ rect, p["intrinsic"]["W"],
                          p["intrinsic"]["H"])


def render_scene(key: str, pool, rl) -> dict:
    meta = json.loads((root(SRC, "scenes", key) / "meta.json").read_text())
    S = R.Scene(meta["dataset"], meta["scene"])
    add_back(S)
    L = H.Logged(S.dir, meta["dataset"])
    tr5 = np.array(meta["t_render"])
    n = int(round((tr5[-1] - tr5[0]) / DT)) + 1
    t10 = np.round(tr5[0] + DT * np.arange(n), 4)
    t_sim = np.array(meta["t_sim"])
    k_sim = np.searchsorted(t_sim, t10 - 1e-6)
    ea, eb, eth = L.pose_at(t10)
    ego = [R.rt2pose(np.array([0.0, eth[k], 0.0]), np.array([ea[k], S.env_height(ea[k], eb[k]), eb[k]]))
           for k in range(n)]
    out = root(DST, "scenes", key)
    futs, cmp_jobs, t_gpu = [], [], 0.0
    for w, m in meta["worlds"].items():
        if not m.get("rendered"):
            continue
        if w != "minus":
            S.load_actor(m["asset"])
        d = out / w
        for c in KEYS:
            (d / "cams" / c).mkdir(parents=True, exist_ok=True)
        lines = []
        for k in range(n):
            actor = None
            if w != "minus":
                a, b, th, _, y = m["track"][k_sim[k]]
                actor = (m["asset"], H.b2w(a, b, th, y))
            files = {}
            for c, ck in zip(CAMS, KEYS):
                g0 = time.perf_counter()
                rgb = R.to_rgb(S.render(c, ego[k] @ S.cams[c][1], actor))
                t_gpu += time.perf_counter() - g0
                f = f"cams/{ck}/{2 * k:07d}.jpg"
                files[ck] = f
                futs.append(pool.submit(cv2.imwrite, str(d / f), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 95]))
                if ck != "back" and k % 2 == 0:
                    cmp_jobs.append((d / f, root(SRC, "scenes", key, w, "cams", ck, f"{2 * k:07d}.jpg")))
            lines.append(json.dumps({"frame": 2 * k, "t": float(t10[k]), "files": files}))
        (d / "frames.jsonl").write_text("\n".join(lines) + "\n")
    for f in futs:
        assert f.result(), "jpeg write failed"
    same, maxd = 0, 0
    for new, old in cmp_jobs:
        if new.read_bytes() == old.read_bytes():
            same += 1
        else:
            maxd = max(maxd, int(np.abs(cv2.imread(str(new)).astype(int) - cv2.imread(str(old)).astype(int)).max()))
    chk = {"front_frames_compared": len(cmp_jobs), "bytes_identical": same, "max_abs_pixel_diff": maxd}
    meta.update(rate_dt=DT, t_render_10hz=t10.tolist(), check_vs_5hz=chk,
                timing_10hz={"gpu_s": round(t_gpu, 2), "views": len(futs)})
    meta["cams"]["CAM_BACK"] = {"K": S.cams["CAM_BACK"][0].tolist(), "c2front": S.cams["CAM_BACK"][1].tolist()}
    (out / "meta.json").write_text(json.dumps(meta, default=R._js))
    del S
    torch.cuda.empty_cache()
    return chk | {"views": len(futs), "gpu_s": round(t_gpu, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"))
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    keys = a.keys or sorted(p.parent.name for p in root(SRC, "scenes").glob("*/meta.json"))
    keys = keys[a.shard[0]::a.shard[1]]
    rl = RunLog("top10_t2", f"i3-10hz-{a.shard[0]}")
    rl.log.info("%d scenes -> %s", len(keys), root(DST))
    from tqdm import tqdm
    with ThreadPoolExecutor(4) as pool:
        for key in tqdm(keys, desc=f"shard {a.shard[0]}/{a.shard[1]}", unit="scene"):
            if a.skip_done and (root(DST, "scenes", key) / "meta.json").exists():
                continue
            t0 = time.perf_counter()
            r = render_scene(key, pool, rl)
            rl.log.info("%s: %.1f s, %s", key, time.perf_counter() - t0, r)
            rl.event("scene_done", key=key, wall_s=round(time.perf_counter() - t0, 2), **r)
    rl.close()


if __name__ == "__main__":
    main()
