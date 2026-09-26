"""DrivoR inference on a T2 request file (todos/2026-09-26-top10-intersection.md, [T2] 10:05 entry, choice 2).

Request (npz, written by jevdrive/top10_t2.py): keys (n,), img (n, 16) paths, time-major t-1.5 ... t x cameras
[l0, f0, r0, b0] ('' = black image), hist (n, 4, 3), ego (n, 4) = vx, vy, ax, ay, cmd (n,) NAVSIM order
(0 left, 1 straight, 2 right, 3 unknown). DrivoR reads the current frame only ([f0, b0, l0, r0], its feature
builder's order) and the last ego status [pose 0, v, a, command one-hot]; images go through the builder's own resize
and normalisation (PIL resize to 1148 x 672, ImageNet mean / std). Output npz: keys, traj (n, 8, 3), rear axle,
0.5 s steps, x fwd, y left, heading.

  --check N   N real navtest tokens: this runner's tensors / plans against DrivoR's own feature builder + forward.

Run from ~/data/third_party/drivor with env ~/data/envs/drivor.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
REPO = Path.cwd()
MODELS = DATA / "models/drivor"
NAV1 = dict(proposal_num=64, refiner_ls_values=0.0, one_token_per_traj=True, refiner_num_heads=1, tf_d_model=256,
            tf_d_ffn=1024, area_pred=False, agent_pred=False, ref_num=4, noc=1, dac=1, ddc=0.0, ttc=5, ep=5, comfort=2)
SIZE = (1148, 672)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
SLOTS = (1, 3, 0, 2)          # request camera order [l0, f0, r0, b0] -> DrivoR's [f0, b0, l0, r0]


def build_agent():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(REPO / "navsim/planning/script/config/common/agent/drivoR.yaml")
    cfg.config.update(NAV1)
    cfg.config.image_backbone.focus_front_cam = False
    cfg.config.image_backbone.model_weights = str(MODELS / "vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")
    cfg.checkpoint_path, cfg.scheduler_args, cfg.batch_size = str(MODELS / "drivor_Nav1_25epochs.pth"), None, 1
    agent = instantiate(cfg)
    agent.initialize()
    return agent.cuda().eval()


def image(path: str) -> torch.Tensor:
    """The feature builder's camera path, from a file ('' = black)."""
    if path:
        im = Image.fromarray(np.asarray(Image.open(path).convert("RGB"))).resize(SIZE)
        a = np.asarray(im, dtype=np.float32) / 255.0
    else:
        a = np.zeros((SIZE[1], SIZE[0], 3), np.float32)
    return torch.from_numpy((a - MEAN) / STD).permute(2, 0, 1)


def ego_vec(ego, cmd) -> torch.Tensor:
    return torch.tensor(np.r_[0.0, 0.0, 0.0, ego, np.eye(4)[int(cmd)]], dtype=torch.float32)


class Req(torch.utils.data.Dataset):
    def __init__(self, z):
        self.img, self.ego, self.cmd = z["img"], z["ego"], z["cmd"]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        cur = self.img[i][12:16]
        return torch.stack([image(str(cur[s])) for s in SLOTS]), ego_vec(self.ego[i], self.cmd[i])[None]


@torch.no_grad()
def run(agent, z, bs=16, workers=6) -> np.ndarray:
    from tqdm import tqdm
    dl = torch.utils.data.DataLoader(Req(z), batch_size=bs, num_workers=workers, pin_memory=True)
    out = []
    for img, ego in tqdm(dl, desc="drivor", unit="batch"):
        o = agent.forward({"image": img.cuda(non_blocking=True), "ego_status": ego.cuda(non_blocking=True)})
        out.append(o["trajectory"].float().cpu().numpy())
    return np.concatenate(out)


def check(agent, n: int, out: Path):
    """Runner path against DrivoR's own builder + forward on n navtest tokens (first log, every 10th token)."""
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    nt = OmegaConf.to_container(OmegaConf.load(REPO / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
    nt.pop("_target_"), nt.pop("_convert_")
    nt["log_names"] = nt["log_names"][:8]
    root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    loader = SceneLoader(root / "navsim_logs/test", root / "sensor_blobs/test", SceneFilter(**nt), agent.get_sensor_config())
    toks = loader.tokens[::max(1, len(loader.tokens) // n)][:n]
    fb = agent.get_feature_builders()[0]
    ref, req, dimg = [], {"img": [], "ego": [], "cmd": []}, 0.0
    for tok in toks:
        ai = loader.get_agent_input_from_token(tok)
        f = fb.compute_features(ai)
        with torch.no_grad():
            ref.append(agent.forward({k: v[None].float().cuda() for k, v in f.items()})["trajectory"][0].cpu().numpy())
        fr = loader.scene_frames_dicts[tok][len(ai.cameras) - 1]["cams"]
        paths = [str(root / "sensor_blobs/test" / fr[k]["data_path"]) for k in ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")]
        es = ai.ego_statuses[-1]
        req["img"].append([""] * 12 + paths)
        req["ego"].append(np.r_[es.ego_velocity, es.ego_acceleration])
        req["cmd"].append(int(np.argmax(es.driving_command)))
        mine = torch.stack([image(paths[s]) for s in SLOTS])
        dimg = max(dimg, float((mine - f["image"]).abs().max()))
    z = {k: np.array(v) for k, v in req.items()}
    got = run(agent, z, bs=8, workers=2)
    ref = np.array(ref)
    single = run(agent, {k: v[:1] for k, v in z.items()}, bs=1, workers=0)
    res = {"n": len(toks), "max_abs_image_diff": dimg, "max_abs_traj_diff_m": float(np.abs(got - ref)[..., :2].max()),
           "batch8_vs_batch1_first_m": float(np.abs(got[:1] - single)[..., :2].max()),
           "ade_to_ref_mean_m": float(np.linalg.norm(got[..., :2] - ref[..., :2], axis=-1).mean())}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def chunked(fn, n: int, out: str, size: int = 1024) -> np.ndarray:
    """fn(slice) per chunk of `size` requests, each saved atomically under <out>.part/, so a killed run resumes at the
    first missing chunk."""
    part = Path(out + ".part")
    part.mkdir(parents=True, exist_ok=True)
    res = []
    for c0 in range(0, n, size):
        f = part / f"{c0:07d}.npz"
        if not f.exists():
            np.savez(part / "tmp.npz", traj=fn(slice(c0, c0 + size)))
            os.replace(part / "tmp.npz", f)
        res.append(np.load(f)["traj"])
    return np.concatenate(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    agent = build_agent()
    if a.check:
        o = DATA / "runs/top10_t2/checks"
        o.mkdir(parents=True, exist_ok=True)
        check(agent, a.check, o / "drivor_check.json")
        return
    with np.load(a.request) as f:
        z = {k: f[k] for k in f.files}
    t0 = time.time()
    traj = chunked(lambda sl: run(agent, {k: v[sl] for k, v in z.items()}, a.bs, a.workers), len(z["keys"]), a.out)
    np.savez_compressed(a.out, keys=z["keys"], traj=traj)
    print(f"{len(traj)} plans in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
