"""WA-JEPA inference on a T2 request file (todos/2026-09-26-top10-intersection.md, [T2] 10:05 entry, choice 3).

Request: see scripts/top10_t2/drivor_run.py (img (n, 16) time-major x [l0, f0, r0, b0], '' = black; hist (n, 4, 3);
ego (n, 4) = vx, vy, ax, ay; cmd (n,) NAVSIM order). Features are built exactly as the repo's NAVSIM feature builder
(eval/navsim_agent.py WorldModelFeatureBuilder): PIL-decoded RGB, cv2 INTER_AREA to 256 x 512, [-1, 1];
ego_status = [command one-hot, vx, vy, ax, ay]; history_trajectory relative to the current pose. One sample per call
(batch 1), as both of the repo's adapters do: predict_trajectory draws its flow noise from a generator re-seeded per
call with shape (batch, ...), so batching would give every sample different noise and paired x+ / x- frames
different draws. --amp (default on) is the repo's HUGSIM adapter precision (fp32 weights, bf16 autocast);
--no-amp is its NAVSIM path (fp32). Output npz: keys, traj (n, 8, 3), rear axle, 0.5 s steps.

  --check N   N navtest tokens: runner (fp32 and bf16) against the repo's own agent.compute_trajectory (fp32).
  --shard I N every N-th request from I (several processes on one card); merge with --merge.

Run from ~/data/third_party/wajepa with env ~/data/envs/wajepa.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
REPO = Path.cwd()
sys.path.insert(0, str(REPO))
CFG = REPO / "configs/wa_jepa_navsim_epdms.yaml"
CKPT = DATA / "models/wajepa/model_state_dict.pt"
OVERRIDES = ["model.require_pretrained=false", "model.vjepa2_ckpt=null"]
H, W = 256, 512


def build_agent():
    from eval.navsim_agent import WorldModelNavsimAgent
    agent = WorldModelNavsimAgent(config_path=str(CFG), checkpoint_path=str(CKPT), device="cuda",
                                  config_overrides=OVERRIDES)
    agent.initialize()
    return agent


def image(path: str) -> torch.Tensor:
    if path:
        img = cv2.resize(np.array(Image.open(path)), (W, H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
    return torch.full((3, H, W), -1.0)


def _prep(path: str) -> np.ndarray:
    return cv2.resize(np.array(Image.open(path)), (W, H), interpolation=cv2.INTER_AREA)


def build_cache(paths, d: Path, workers: int):
    """Every unique image decoded and resized once (the same ops as `image`) into a uint8 memmap: history frames are
    shared between neighbouring requests, so this does ~3x less JPEG work on a CPU-starved box. Reused if present."""
    from multiprocessing import Pool
    from tqdm import tqdm
    d.mkdir(parents=True, exist_ok=True)
    u = np.unique([p for p in paths if p])
    if (d / "done").exists() and (np.load(d / "paths.npy") == u).all():
        return u, np.load(d / "cache.npy", mmap_mode="r")
    mm = np.lib.format.open_memmap(d / "cache.npy", "w+", np.uint8, (len(u), H, W, 3))
    with Pool(workers) as pool:
        for i, a in enumerate(tqdm(pool.imap(_prep, u, chunksize=16), total=len(u), desc="cache", mininterval=30)):
            mm[i] = a
    mm.flush()
    np.save(d / "paths.npy", u)
    (d / "done").touch()
    return u, np.load(d / "cache.npy", mmap_mode="r")


class Req(torch.utils.data.Dataset):
    def __init__(self, z, idx, cache=None):
        self.z, self.idx = z, idx
        self.row = {p: i for i, p in enumerate(cache[0])} if cache is not None else None
        self.mm = cache[1] if cache is not None else None

    def img(self, p: str) -> torch.Tensor:
        if self.mm is None or not p:
            return image(p)
        t = torch.from_numpy(np.array(self.mm[self.row[p]])).permute(2, 0, 1).float() / 255.0
        return t.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i, z = self.idx[j], self.z
        imgs = torch.stack([self.img(str(p)) for p in z["img"][i]]).reshape(4, 4, 3, H, W)
        cmd = int(z["cmd"][i])
        ego = np.r_[np.eye(4)[min(cmd, 3)], z["ego"][i]].astype(np.float32)
        return {"history_images": imgs, "history_trajectory": torch.tensor(z["hist"][i], dtype=torch.float32),
                "navigation_command": torch.tensor(cmd), "ego_speed": torch.tensor(float(np.linalg.norm(z["ego"][i][:2]))),
                "ego_status": torch.from_numpy(ego), "trajectory_interval_s": torch.tensor(0.5)}


@torch.no_grad()
def run(model, z, idx, amp: bool, workers=4, cache=None) -> np.ndarray:
    from tqdm import tqdm
    dl = torch.utils.data.DataLoader(Req(z, idx, cache), batch_size=1, num_workers=workers, pin_memory=True,
                                     prefetch_factor=4 if workers else None)
    out = []
    for f in tqdm(dl, desc="wajepa", unit="sample", mininterval=10):
        f = {k: v.cuda(non_blocking=True) for k, v in f.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out.append(model.predict_trajectory(f).float().cpu().numpy())
    return np.concatenate(out)


def check(agent, n: int, out: Path):
    from datasets.ego_trajectory_utils import convert_absolute_to_relative_se2_array
    from hydra.utils import instantiate
    from nuplan.common.actor_state.state_representation import StateSE2
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    sf = instantiate(OmegaConf.load(DATA / "third_party/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
    sf.log_names = sf.log_names[:8]
    loader = SceneLoader(original_sensor_path=root / "sensor_blobs/test", data_path=root / "navsim_logs/test",
                         scene_filter=sf, sensor_config=agent.get_sensor_config())
    toks = loader.tokens[::max(1, len(loader.tokens) // n)][:n]
    ref, req = [], {"img": [], "hist": [], "ego": [], "cmd": []}
    for tok in toks:
        ai = loader.get_agent_input_from_token(tok)
        ref.append(agent.compute_trajectory(ai).poses)
        frames = loader.scene_frames_dicts[tok][:len(ai.cameras)][-4:]
        req["img"].append([str(root / "sensor_blobs/test" / fr["cams"][k]["data_path"]) for fr in frames
                           for k in ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")])
        poses = np.array([[e.ego_pose[0], e.ego_pose[1], e.ego_pose[2]] for e in ai.ego_statuses[-4:]])
        req["hist"].append(convert_absolute_to_relative_se2_array(StateSE2(*poses[-1]), poses))
        es = ai.ego_statuses[-1]
        req["ego"].append(np.r_[es.ego_velocity, es.ego_acceleration])
        req["cmd"].append(int(np.argmax(es.driving_command)))
    z = {k: np.array(v) for k, v in req.items()}
    ref = np.array(ref)
    idx = np.arange(len(toks))
    fp32, bf16 = run(agent.model, z, idx, False, 2), run(agent.model, z, idx, True, 2)
    ade = lambda a, b: float(np.linalg.norm(a[..., :2] - b[..., :2], axis=-1).mean())  # noqa: E731
    res = {"n": len(toks), "fp32_max_abs_diff_m": float(np.abs(fp32 - ref)[..., :2].max()), "fp32_ade_to_ref_m": ade(fp32, ref),
           "bf16_ade_to_fp32_m": ade(bf16, ref), "bf16_max_abs_diff_m": float(np.abs(bf16 - ref)[..., :2].max())}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge", nargs="*", help="shard outputs to merge into --out")
    ap.add_argument("--cache", help="dir of the decoded-image memmap (built on first use; --cache-only builds and exits)")
    ap.add_argument("--cache-only", action="store_true")
    a = ap.parse_args()
    if a.merge:
        zs = [np.load(p) for p in a.merge]
        idx = np.concatenate([z["idx"] for z in zs])
        o = np.argsort(idx)
        np.savez_compressed(a.out, keys=np.concatenate([z["keys"] for z in zs])[o],
                            traj=np.concatenate([z["traj"] for z in zs])[o])
        return
    if a.cache_only:
        with np.load(a.request) as f:
            build_cache(f["img"].ravel(), Path(a.cache), a.workers)
        return
    agent = build_agent()
    if a.check:
        o = DATA / "runs/top10_t2/checks"
        o.mkdir(parents=True, exist_ok=True)
        check(agent, a.check, o / "wajepa_check.json")
        return
    with np.load(a.request) as f:                 # materialise: DataLoader workers must not share the zip handle
        z = {k: f[k] for k in f.files}
    idx = np.arange(len(z["keys"]))[a.shard[0]::a.shard[1]]
    t0 = time.time()
    cache = build_cache(z["img"].ravel(), Path(a.cache), a.workers) if a.cache else None
    traj = run(agent.model, z, idx, not a.no_amp, a.workers, cache)
    np.savez_compressed(a.out, keys=z["keys"][idx], idx=idx, traj=traj)
    print(f"{len(traj)} plans in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
