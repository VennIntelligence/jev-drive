"""WA-JEPA on a request under several ego-status arms, the image encoder run once per sample (night queue 3, Q5;
todos/2026-09-26-night-queue-3.md, [D] 16:50 Q5 entry).

predict_trajectory (models/multiview_causal_future_jepa.py) encodes the four history frames of the four cameras
(`_encode_scene_context`), then runs the flow sampler, whose only other inputs are ego_status and history_trajectory
and whose generator is re-seeded on every call. `encode` + `flow` below are that method's statements, split after
the context encoding, so one encoding serves every arm; each arm's flow re-creates the generator exactly as a separate
call would. The third-party code is not modified. --check compares this path with separate predict_trajectory calls.

Inputs: the T2 request (keys, img (n, 16) time-major t-1.5 ... t x [l0, f0, r0, b0]) plus an arms file: names (A,),
ego8 (A, n, 8) = vx, vy, ax, ay, command one-hot (4; all zero allowed), hist (A, n, 4, 3). WA-JEPA's ego_status is
[one-hot, vx, vy, ax, ay] (its NAVSIM feature builder). Output npz: names, keys, traj (A, n, 8, 3), rear axle, 0.5 s.
Precision: --amp (default) = fp32 weights + bf16 autocast (T2's exam path); --no-amp = fp32 (its NAVSIM path).
Run from ~/data/third_party/wajepa with env ~/data/envs/wajepa.
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
import wajepa_run as WR  # noqa: E402  (build_agent, image, H, W)


class Imgs(torch.utils.data.Dataset):
    def __init__(self, img, idx):
        self.img, self.idx = img, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        return torch.stack([WR.image(str(p)) for p in self.img[i]]).reshape(4, 4, 3, WR.H, WR.W), i


def ego_status(e8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.r_[e8[4:8], e8[:4]].astype(np.float32))[None]


@torch.no_grad()
def encode(model, history: torch.Tensor) -> dict:
    f = model.prepare_batch({"history_images": history})
    history = model._ensure_multiview(f["history_images"])
    bsz, _, views, ch, h, w = history.shape
    future = torch.zeros(bsz, model.num_future_frames, views, ch, h, w, device=history.device, dtype=history.dtype)
    clip = model._make_full_clip(history, future)
    masks = model.mask_sampler.full_future_mask(bsz, device=clip.device)
    idx, masks_x = masks.indices[0], masks.masks_x[0]
    view_idx = (idx.unsqueeze(1) * model.num_cameras + torch.arange(model.num_cameras, device=clip.device).unsqueeze(0)).reshape(-1)
    ctx = model._encode_scene_context(clip.index_select(0, view_idx), masks_x.index_select(0, idx), idx.numel())
    cti = model._history_indices_from_masks(masks_x.index_select(0, idx), idx.numel())
    fcs = model._future_mask_condition(bsz, device=clip.device, dtype=ctx.dtype)
    return {"ctx": ctx, "cti": cti, "fcs": fcs, "idx": idx, "bsz": bsz, "dev": clip.device}


@torch.no_grad()
def flow(model, E: dict, ego: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
    ad = model.driving_condition_adapter
    p = next(model.predictor.parameters())
    b = ad.prepare_batch({"ego_status": ego, "history_trajectory": hist}, dtype=p.dtype, device=p.device)
    dc = ad.inference_conditions(b)
    ctx, dev, bsz = E["ctx"], E["dev"], E["bsz"]
    gen, sdev = model._make_inference_generator(dev)
    scene = model._randn_for_inference((bsz, model.predictor.num_scene_tokens, model.scene_projector.scene_dim), device=dev,
                                       dtype=ctx.dtype, generator=gen, sample_device=sdev) * model.flow_inference_noise_scale
    gc = dc.select(E["idx"])
    traj = ad.initial_inference_trajectory(model, bsz, device=dev, dtype=ctx.dtype, generator=gen, sample_device=sdev)
    ti = ad.prepare_inference_inputs(model, gc, dtype=ctx.dtype, device=dev)
    steps = max(model.flow_num_inference_steps, 1)
    dt = 1.0 / float(steps)
    for step in range(steps):
        t_value = min(float(step) / float(steps), 1.0 - 1e-4)
        t_cont = torch.full((bsz,), t_value, device=dev, dtype=ctx.dtype)
        ti["noisy_trajectory"] = traj
        pred_scene, pred_traj = model.predictor(context_scene=ctx, context_token_indices=E["cti"], noisy_future_scene=scene,
                                                future_condition_scene=E["fcs"], t_cont=t_cont, trajectory_inputs=ti)
        denom = (1.0 - t_cont).clamp_min(1e-3)
        scene = scene + dt * (pred_scene - scene) / denom.view(-1, 1, 1)
        traj = traj + dt * (pred_traj - traj) / denom.view(-1, 1, 1)
    return ad.inference_output(model, traj)


@torch.no_grad()
def flow_batched(model, E: dict, ego: torch.Tensor, hist: torch.Tensor, A: int) -> torch.Tensor:
    """flow() for B samples x A arms in one pass (rows arm-major: arm a, sample b at a * B + b). Every separate call
    re-seeds the generator, so every sample and arm starts from the same noise: it is drawn once at batch size 1, exactly
    as a batch-1 call draws it, and broadcast. Only the batch size of the predictor differs from separate calls."""
    ad = model.driving_condition_adapter
    p = next(model.predictor.parameters())
    b = ad.prepare_batch({"ego_status": ego, "history_trajectory": hist}, dtype=p.dtype, device=p.device)
    dc = ad.inference_conditions(b)
    ctx0, dev, B = E["ctx"], E["dev"], E["bsz"]
    n = A * B
    ctx = ctx0.repeat(A, *([1] * (ctx0.ndim - 1)))
    cti = E["cti"].repeat(A, *([1] * (E["cti"].ndim - 1)))
    fcs = model._future_mask_condition(n, device=dev, dtype=ctx.dtype)
    gen, sdev = model._make_inference_generator(dev)
    scene = model._randn_for_inference((1, model.predictor.num_scene_tokens, model.scene_projector.scene_dim), device=dev,
                                       dtype=ctx.dtype, generator=gen, sample_device=sdev) * model.flow_inference_noise_scale
    traj = ad.initial_inference_trajectory(model, 1, device=dev, dtype=ctx.dtype, generator=gen, sample_device=sdev)
    scene, traj = scene.expand(n, *scene.shape[1:]).contiguous(), traj.expand(n, *traj.shape[1:]).contiguous()
    gc = dc.select(torch.arange(n, device=dev))
    ti = ad.prepare_inference_inputs(model, gc, dtype=ctx.dtype, device=dev)
    steps = max(model.flow_num_inference_steps, 1)
    dt = 1.0 / float(steps)
    for step in range(steps):
        t_value = min(float(step) / float(steps), 1.0 - 1e-4)
        t_cont = torch.full((n,), t_value, device=dev, dtype=ctx.dtype)
        ti["noisy_trajectory"] = traj
        pred_scene, pred_traj = model.predictor(context_scene=ctx, context_token_indices=cti, noisy_future_scene=scene,
                                                future_condition_scene=fcs, t_cont=t_cont, trajectory_inputs=ti)
        denom = (1.0 - t_cont).clamp_min(1e-3)
        scene = scene + dt * (pred_scene - scene) / denom.view(-1, 1, 1)
        traj = traj + dt * (pred_traj - traj) / denom.view(-1, 1, 1)
    return ad.inference_output(model, traj)


def run_batched(model, img, arms: dict, idx, amp: bool, workers: int, bs: int) -> np.ndarray:
    from tqdm import tqdm
    A = len(arms["names"])
    out = np.zeros((A, len(idx), 8, 3), np.float32)
    dl = torch.utils.data.DataLoader(Imgs(img, idx), batch_size=bs, num_workers=workers, pin_memory=True,
                                     prefetch_factor=4 if workers else None)
    j0 = 0
    for h, i in tqdm(dl, desc="wajepa-arms", unit="batch", mininterval=30):
        h, i = h.cuda(non_blocking=True), i.numpy()
        ego = torch.cat([torch.cat([ego_status(e) for e in arms["ego8"][a, i]]) for a in range(A)]).cuda()
        hist = torch.from_numpy(arms["hist"][:, i].reshape(-1, 4, 3).astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = flow_batched(model, encode(model, h), ego, hist, A)
        out[:, j0:j0 + len(i)] = y.float().cpu().numpy().reshape(A, len(i), 8, 3)
        j0 += len(i)
    return out


def run(model, img, arms: dict, idx, amp: bool, workers: int, separate: bool = False, bs: int = 0) -> np.ndarray:
    if bs:
        return run_batched(model, img, arms, idx, amp, workers, bs)
    """(A, len(idx), 8, 3). separate=True: one full predict_trajectory call per arm (the reference path)."""
    from tqdm import tqdm
    A = len(arms["names"])
    out = np.zeros((A, len(idx), 8, 3), np.float32)
    dl = torch.utils.data.DataLoader(Imgs(img, idx), batch_size=1, num_workers=workers, pin_memory=True,
                                     prefetch_factor=4 if workers else None)
    for j, (h, i) in enumerate(tqdm(dl, desc="wajepa-arms", unit="sample", mininterval=30)):
        h, i = h.cuda(non_blocking=True), int(i)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            E = None if separate else encode(model, h)
            for a in range(A):
                ego = ego_status(arms["ego8"][a, i]).cuda()
                hist = torch.from_numpy(arms["hist"][a, i].astype(np.float32))[None].cuda()
                if separate:
                    f = {"history_images": h, "history_trajectory": hist, "ego_status": ego,
                         "trajectory_interval_s": torch.tensor([0.5], device="cuda")}
                    y = model.predict_trajectory(f)
                else:
                    y = flow(model, E, ego, hist)
                out[a, j] = y.float().cpu().numpy()[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("arms")
    ap.add_argument("--out")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", type=int, default=0, help="first N requests: reuse path vs separate calls, timings")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--bs", type=int, default=0, help="> 0: B samples x all arms per predictor pass (flow_batched)")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    with np.load(a.request) as f:
        img, keys = f["img"], f["keys"]
    with np.load(a.arms) as f:
        arms = {k: f[k] for k in f.files}
    agent = WR.build_agent()
    model = agent.model
    amp = not a.no_amp
    if a.check:
        idx = np.arange(a.check)
        t0 = time.time()
        ref = run(model, img, arms, idx, amp, a.workers, separate=True)
        t1 = time.time()
        got = run(model, img, arms, idx, amp, a.workers)
        t2 = time.time()
        res = {"n": int(a.check), "arms": int(len(arms["names"])), "amp": amp,
               "max_abs_diff_m": float(np.abs(got - ref)[..., :2].max()), "bitwise_equal": bool((got == ref).all()),
               "s_per_sample_separate": (t1 - t0) / a.check, "s_per_sample_reuse": (t2 - t1) / a.check}
        if a.bs:
            t3 = time.time()
            bat = run(model, img, arms, idx, amp, a.workers, bs=a.bs)
            d = np.linalg.norm(bat[..., :2] - ref[..., :2], axis=-1)
            res.update({"batched_bs": a.bs, "batched_max_abs_diff_m": float(np.abs(bat - ref)[..., :2].max()),
                        "batched_mean_disp_m": float(d.mean()), "batched_p99_disp_m": float(np.percentile(d, 99)),
                        "s_per_sample_batched": (time.time() - t3) / a.check})
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return
    idx = np.arange(len(keys))[a.shard[0]::a.shard[1]]
    part = Path(a.out + ".part")
    part.mkdir(parents=True, exist_ok=True)
    t0, res = time.time(), []
    for c0 in range(0, len(idx), a.chunk):
        f = part / f"{c0:07d}.npz"
        if not f.exists():
            np.savez(part / "tmp.npz", traj=run(model, img, arms, idx[c0:c0 + a.chunk], amp, a.workers, bs=a.bs))
            os.replace(part / "tmp.npz", f)
        res.append(np.load(f)["traj"])
    traj = np.concatenate(res, 1)
    np.savez(a.out, names=arms["names"], keys=keys[idx], idx=idx, traj=traj)
    print(f"{traj.shape[1]} requests x {traj.shape[0]} arms in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
