"""DrivoR on a request under several ego-status arms, the image backbone run once per batch (night queue 3, Q5;
todos/2026-09-26-night-queue-3.md, [D] 16:50 Q5 entry).

DrivoRModel.forward reads the images only through `image_backbone(img, scene_embeds)`; the ego status enters the
trajectory tokens and the scorer. The backbone output of a batch is computed once and handed to every arm's forward
by a stand-in module that returns it, so each arm runs the model's own forward unchanged; --check compares this with
plain forwards. Images and batching are scripts/top10_t2/drivor_run.py's (feature-builder resize / normalisation, bs 16).

Inputs: the T2 request (img (n, 16); DrivoR reads the current frame, slots 12..15) plus an arms file (names (A,),
ego8 (A, n, 8) = vx, vy, ax, ay, command one-hot). DrivoR's ego vector = [pose 0 (3), vx, vy, ax, ay, one-hot].
Output npz: names, keys, traj (A, n, 8, 3). Run from ~/data/third_party/drivor with env ~/data/envs/drivor.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "top10_t2"))
import drivor_run as DR  # noqa: E402


class Cached(torch.nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.feat = feat

    def forward(self, img, scene_tokens):
        return self.feat


class Imgs(torch.utils.data.Dataset):
    def __init__(self, img):
        self.img = img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        cur = self.img[i][12:16]
        return torch.stack([DR.image(str(cur[s])) for s in DR.SLOTS]), i


def ego_vec(e8: np.ndarray) -> torch.Tensor:
    """(B, 8) -> (B, 1, 11)."""
    return torch.from_numpy(np.concatenate([np.zeros((len(e8), 3)), e8], 1).astype(np.float32))[:, None]


@torch.no_grad()
def run(agent, img, ego8, bs=16, workers=6, reuse=True) -> np.ndarray:
    from tqdm import tqdm
    m = agent._drivor_model
    A = len(ego8)
    out = np.zeros((A, len(img), 8, 3), np.float32)
    dl = torch.utils.data.DataLoader(Imgs(img), batch_size=bs, num_workers=workers, pin_memory=True)
    bb = m.image_backbone
    for x, i in tqdm(dl, desc="drivor-arms", unit="batch", mininterval=30):
        x, i = x.cuda(non_blocking=True), i.numpy()
        if reuse:
            m.image_backbone = Cached(bb(x, m.scene_embeds.repeat(len(x), 1, 1, 1)))
        try:
            for a in range(A):
                o = agent.forward({"image": x, "ego_status": ego_vec(ego8[a, i]).cuda()})
                out[a, i] = o["trajectory"].float().cpu().numpy()
        finally:
            m.image_backbone = bb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("arms")
    ap.add_argument("--out")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check", type=int, default=0, help="first N requests: reuse path vs plain forwards, timings")
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()
    with np.load(a.request) as f:
        img, keys = f["img"], f["keys"]
    with np.load(a.arms) as f:
        arms = {k: f[k] for k in f.files}
    agent = DR.build_agent()
    if a.check:
        sl = slice(0, a.check)
        t0 = time.time()
        ref = run(agent, img[sl], arms["ego8"][:, sl], a.bs, a.workers, reuse=False)
        t1 = time.time()
        got = run(agent, img[sl], arms["ego8"][:, sl], a.bs, a.workers)
        t2 = time.time()
        res = {"n": a.check, "arms": int(len(arms["names"])), "max_abs_diff_m": float(np.abs(got - ref)[..., :2].max()),
               "bitwise_equal": bool((got == ref).all()), "s_per_sample_plain": (t1 - t0) / a.check,
               "s_per_sample_reuse": (t2 - t1) / a.check}
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return
    part = Path(a.out + ".part")
    part.mkdir(parents=True, exist_ok=True)
    t0, res = time.time(), []
    for c0 in range(0, len(keys), a.chunk):
        f = part / f"{c0:07d}.npz"
        if not f.exists():
            sl = slice(c0, c0 + a.chunk)
            np.savez(part / "tmp.npz", traj=run(agent, img[sl], arms["ego8"][:, sl], a.bs, a.workers))
            os.replace(part / "tmp.npz", f)
        res.append(np.load(f)["traj"])
    traj = np.concatenate(res, 1)
    np.savez(a.out, names=arms["names"], keys=keys, traj=traj)
    print(f"{traj.shape[1]} requests x {traj.shape[0]} arms in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
