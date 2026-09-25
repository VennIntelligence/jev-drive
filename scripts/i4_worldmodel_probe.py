#!/usr/bin/env python3
"""I4: does commaai/worldmodel-4B respond to the ego action? One context clip, several constant actions.

The world model (comma.ai, HF commaai/worldmodel-4B) is run exactly as shipped through the Runtime of
openpilot.distill/rl/server.py; this script only prepares inputs the way openpilot.distill/rl/env.py does
(Physics: curvature [1/m], accel [m/s^2] -> relative pose of the target frame) and measures the outputs.

  run   (env $DATA_DIR/envs/comma-wm): encode a comma1M clip, roll out every arm for --steps 5 Hz frames,
        batched across arms, time generation, save latents / decoded frames / metrics to a run dir.
  plot  (env jevdrive): side-by-side figure and metric curves from a run dir.

Arms: straight (two noise seeds -> null), left and right curvature, hard brake. All arms share the same
initial noise seed except the null twin, so a difference to `straight` is the action effect alone.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARMS = {  # name: (curvature 1/m, accel m/s^2, noise seed); env convention: +curvature = right (WASD "D")
    "straight": (0.0, 0.0, 0),
    "straight_seed1": (0.0, 0.0, 1),
    "left": (-0.01, 0.0, 0),
    "right": (0.01, 0.0, 0),
    "brake": (0.0, -4.0, 0),
}


def load_clip(runtime, seg_dir: Path, first: int, n: int, gpu_id: int):
    """Encode 5 Hz frames [first, first+n) of a comma1M segment (comma 3/3X cameras) like rl.env.Episode."""
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    from openpilot.common.transformations.model import medmodel_intrinsics, sbigmodel_intrinsics
    from safetensors.numpy import load_file
    import cv2
    from helpers.video_helpers import calibration_view_eulers, calibration_warp_matrix, decode_frames
    from rl.env import FRAME_SKIP

    cams = DEVICE_CAMERAS[("tici", "ar0231")]  # 1928x1208 road cameras, same intrinsics as ox03c10
    frame_info = load_file(seg_dir / "frame_info.safetensors")
    loc = load_file(seg_dir / "localizer.safetensors")
    speeds = np.linalg.norm(loc["frame_states"][::FRAME_SKIP, 7:10], axis=1)
    view_euler = calibration_view_eulers(loc["rpy"])
    wanted = [(first + i) * FRAME_SKIP for i in range(n)]
    views = []
    for cam, src_k, dst_k in (("fcamera", cams.narrow_road.intrinsics, medmodel_intrinsics),
                              ("ecamera", cams.wide_road.intrinsics, sbigmodel_intrinsics)):
        m = calibration_warp_matrix(src_k, dst_k, view_euler)
        m[:2] *= 0.5
        dec = decode_frames(seg_dir / f"{cam}.hevc", frame_info[f"{cam}/index"], set(wanted), gpu_id=gpu_id)
        views.append(np.stack([cv2.warpPerspective(dec[w], m, (256, 128), borderMode=cv2.BORDER_REPLICATE)
                               for w in wanted]))
    frames = np.concatenate(views, axis=-1)  # (n, 128, 256, 6): narrow RGB | wide RGB
    return runtime.encode(frames), frames, speeds[first:first + n]


def run(a):
    import torch
    from huggingface_hub import hf_hub_download
    import rl.server as srv
    from rl.env import FPS, Physics
    from jevdrive.runlog import RunLog

    # Glue only: point the shipped Runtime at the VAE we downloaded with --local-dir.
    vae_dir = Path(a.vae_dir)
    srv.hf_hub_download = lambda repo, name: str(vae_dir / name) if (vae_dir / name).exists() else hf_hub_download(repo, name)

    rl = RunLog("reactivity-i4", a.tag)
    rl.info(f"args {vars(a)}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    rt = srv.Runtime(Path(a.model), "commaai/vit-ae-2x-f8c32", "cuda")
    rl.info(f"runtime loaded in {time.perf_counter() - t0:.1f}s; history={rt.history_frames} "
            f"future={rt.future_frames} dtype={rt.model_dtype}")
    H, F, K = rt.history_frames, rt.future_frames, a.steps
    first = a.start - H
    lat, frames, speeds = load_clip(rt, Path(a.segment), first, H + K + F, a.gpu_id)
    rl.info(f"clip {Path(a.segment).name} 5Hz frames {first}..{first + H + K + F}, speed at start {speeds[H - 1]:.1f} m/s")

    names = list(ARMS)
    B = len(names)
    ctx = lat[:H].clone()[None].repeat(B, 1, 1, 1, 1)
    future = lat[H + K:H + K + F].clone()[None].repeat(B, 1, 1, 1, 1)  # logged anchors at the rollout end
    future_fidx = torch.arange(H + K, H + K + F)
    phys = [Physics(float(speeds[H - 1])) for _ in names]
    gens = {n: torch.Generator(device="cpu").manual_seed(ARMS[n][2]) for n in names}
    out_lat, plans, ego = [], [], []
    step_ms = []
    for k in range(K):
        noise = torch.stack([torch.randn(ctx.shape[2:], generator=gens[n]) for n in names])[:, None]
        latents = torch.cat((future, ctx, noise.to(ctx)), 1)
        pos = latents.new_zeros((B, latents.shape[1], 3))
        eul = torch.zeros_like(pos)
        mask = torch.ones(latents.shape[:2], dtype=torch.int64)
        e = []
        for i, n in enumerate(names):
            dx, dy, yaw = phys[i].step(ARMS[n][0], ARMS[n][1])
            pos[i, -1, 0], pos[i, -1, 1], eul[i, -1, 2] = dx, dy, yaw
            e.append((dx, dy, yaw, phys[i].speed))
        mask[:, -1] = 0
        fidxs = torch.cat((future_fidx - k, torch.arange(H + 1)))[None].repeat(B, 1)
        dev = rt.device
        torch.cuda.synchronize(); t = time.perf_counter()
        o = rt.model.generate(latents=latents.to(dev, rt.model_dtype).clone(), augments_pos_ref_augment=pos.to(dev, rt.model_dtype),
                              ref_augment_from_augments_euler=eul.to(dev, rt.model_dtype), pose_mask=mask.to(dev),
                              fidxs=fidxs.to(dev), steps=a.sampling_steps, num_prefill_frames=latents.shape[1] - 1,
                              dtype=rt.model_dtype, inference_schedule="linear", cfg=a.cfg)
        torch.cuda.synchronize(); step_ms.append((time.perf_counter() - t) * 1e3)
        new = o["latents"][:, 0].float().cpu()
        ctx = torch.cat((ctx[:, 1:], new[:, None].to(ctx)), 1)
        out_lat.append(new)
        p = o["plan"].float().cpu()
        plans.append(p[:, : p.shape[1] // 2].reshape(B, -1, 15).numpy())
        ego.append(e)
        rl.event("step_end", step=k, ms=step_ms[-1])
    out_lat = torch.stack(out_lat, 1)  # (B, K, 32, 16, 32)
    dec = np.stack([rt.decode(out_lat[i].to(rt.device)) for i in range(B)])
    if dec.shape[-1] != 6:  # decoder returns NCHW -> (B, K, 128, 256, 6)
        dec = np.ascontiguousarray(np.moveaxis(dec, 2, -1))
    rl.info(f"decoded {dec.shape}")

    # Timing: batch 1 and batch B (per generated frame), decode, at the run's steps / cfg.
    def time_gen(bs, reps=5):
        L = torch.cat((future[:1], ctx[:1], noise[:1].to(ctx)), 1).repeat(bs, 1, 1, 1, 1).to(rt.device, rt.model_dtype)
        kw = dict(augments_pos_ref_augment=pos[:1].repeat(bs, 1, 1).to(rt.device, rt.model_dtype),
                  ref_augment_from_augments_euler=eul[:1].repeat(bs, 1, 1).to(rt.device, rt.model_dtype),
                  pose_mask=mask[:1].repeat(bs, 1).to(rt.device), fidxs=fidxs[:1].repeat(bs, 1).to(rt.device),
                  num_prefill_frames=L.shape[1] - 1, dtype=rt.model_dtype, inference_schedule="linear", cfg=a.cfg)
        res = {}
        for steps in (15, 30):
            rt.model.generate(latents=L.clone(), steps=steps, **kw)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(reps):
                rt.model.generate(latents=L.clone(), steps=steps, **kw)
            torch.cuda.synchronize()
            res[steps] = (time.perf_counter() - t) * 1e3 / reps
        return res
    timing = {f"gen_ms_b{bs}": time_gen(bs) for bs in (1, 8, 32)}
    x = out_lat[0, :1].to(rt.device)
    rt.decode(x); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        rt.decode(x)
    torch.cuda.synchronize()
    timing["decode_ms_b1"] = (time.perf_counter() - t) * 100
    timing["rollout_step_ms_batch_arms"] = float(np.median(step_ms[1:]))
    timing["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
    timing["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    rl.info(f"timing {json.dumps(timing)}")

    np.savez_compressed(rl.dir / "rollout.npz", names=np.array(names), latents=out_lat.numpy().astype(np.float16),
                        decoded=dec, context_frames=frames, plans=np.stack(plans, 1), ego=np.array(ego).transpose(1, 0, 2),
                        speeds=speeds)
    metrics = measure(names, out_lat.numpy(), dec, frames[H - 1])
    json.dump({"segment": Path(a.segment).name, "start_5hz": a.start, "steps": K, "sampling_steps": a.sampling_steps,
               "cfg": a.cfg, "arms": ARMS, "timing": timing, "metrics": metrics}, open(rl.dir / "results.json", "w"), indent=1)
    rl.info(f"results -> {rl.dir}")
    rl.event("end")
    rl.close()


def flow_stats(prev, cur):
    """Mean horizontal flow (px, +right) and radial expansion (px) between two 128x256 RGB frames (wide camera)."""
    import cv2
    g0, g1 = (cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in (prev, cur))
    fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = g0.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.stack((xx - w / 2, yy - h * 0.45), -1)
    r /= np.linalg.norm(r, axis=-1, keepdims=True) + 1e-6
    return float(fl[..., 0].mean()), float((fl * r).sum(-1).mean())


def measure(names, lat, dec, last_ctx):
    """Per step: latent RMS difference to `straight` (in units of latent std), pixel MAE, flow statistics."""
    ref = names.index("straight")
    std = lat.std()
    m = {}
    for i, n in enumerate(names):
        d_lat = np.sqrt(((lat[i] - lat[ref]) ** 2).mean(axis=(1, 2, 3))) / std
        d_pix = np.abs(dec[i].astype(np.float32) - dec[ref].astype(np.float32)).mean(axis=(1, 2, 3))
        seq = np.concatenate((last_ctx[None], dec[i]))
        fl = np.array([flow_stats(seq[t][..., 3:], seq[t + 1][..., 3:]) for t in range(len(dec[i]))])
        m[n] = {"lat_rms_vs_straight": d_lat.tolist(), "pix_mae_vs_straight": d_pix.tolist(),
                "flow_x_wide": fl[:, 0].tolist(), "flow_expand_wide": fl[:, 1].tolist()}
    return m


def plot(a):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from jevdrive.plots import COL, PAGE, STYLE, OKABE_ITO, save
    d = Path(a.run_dir)
    z = np.load(d / "rollout.npz")
    res = json.load(open(d / "results.json"))
    names = list(z["names"])
    dec, ctx = z["decoded"], z["context_frames"]
    H = ctx.shape[0] - res["steps"] - 5
    show = ["straight", "left", "right", "brake"]
    ts = [0, 4, 9, min(14, res["steps"] - 1)]
    ts = sorted(set(t for t in ts if t < res["steps"]))
    out = Path(a.out_dir)
    color = {"straight": "#7F7F7F", "straight_seed1": "#000000", "left": "#0072B2", "right": "#D55E00", "brake": "#009E73"}
    label = {"straight": "straight", "straight_seed1": "straight, seed 2 (null)", "left": r"left, $\kappa$=0.01 m$^{-1}$",
             "right": r"right, $\kappa$=0.01 m$^{-1}$", "brake": r"brake, $a$=$-4$ m s$^{-2}$"}
    with mpl.rc_context(STYLE):
        cols = 1 + len(ts)
        fig, axes = plt.subplots(len(show), cols, figsize=(PAGE, PAGE * len(show) / cols * 0.5 + 0.1),
                                 gridspec_kw=dict(wspace=0.03, hspace=0.06))
        for r, n in enumerate(show):
            i = names.index(n)
            for c in range(cols):
                ax = axes[r, c]
                img = ctx[H - 1][..., 3:] if c == 0 else dec[i, ts[c - 1]][..., 3:]
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if r == 0:
                    ax.set_title("context (t=0)" if c == 0 else f"t=+{(ts[c - 1] + 1) / 5:.1f} s", fontsize=8)
                if c == 0:
                    ax.set_ylabel(label[n], fontsize=7)
        fig.savefig(out / "i4_worldmodel_actions.png", dpi=200)
        plt.close(fig)

        m = res["metrics"]
        t = (np.arange(res["steps"]) + 1) / 5
        fig, axes = plt.subplots(1, 3, figsize=(PAGE, 1.7))
        for n in names:
            kw = dict(color=color[n], label=label[n], ls="--" if n == "straight_seed1" else "-")
            if n != "straight":
                axes[0].plot(t, m[n]["pix_mae_vs_straight"], **kw)
            axes[1].plot(t, m[n]["flow_x_wide"], **kw)
            axes[2].plot(t, m[n]["flow_expand_wide"], **kw)
        axes[0].set_ylabel("pixel MAE vs straight (0-255)")
        axes[1].set_ylabel("mean horizontal flow (px/frame)")
        axes[2].set_ylabel("radial expansion flow (px/frame)")
        for ax in axes:
            ax.set_xlabel("time after context (s)")
            ax.grid(True)
        axes[1].axhline(0, color="k", lw=0.4)
        h, l = axes[1].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(l))
        fig.tight_layout()
        fig.savefig(out / "i4_worldmodel_metrics.png", dpi=250)
        fig.savefig(d / "i4_worldmodel_metrics.pdf")
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--distill", default=None, help="openpilot.distill checkout (default $DATA_DIR/models/comma/openpilot.distill)")
    r.add_argument("--model", default=None, help="model.fp8_nvfp4.torchpackage")
    r.add_argument("--vae-dir", default=None)
    r.add_argument("--segment", required=True, help="comma1M segment dir with fcamera/ecamera.hevc")
    r.add_argument("--start", type=int, default=100, help="first generated 5 Hz frame index")
    r.add_argument("--steps", type=int, default=15, help="generated 5 Hz frames per arm")
    r.add_argument("--sampling-steps", type=int, default=15)
    r.add_argument("--cfg", type=float, default=2.0)
    r.add_argument("--gpu-id", type=int, default=0)
    r.add_argument("--tag", default="probe")
    q = sub.add_parser("plot")
    q.add_argument("run_dir")
    q.add_argument("--out-dir", default=str(REPO / "research/figs"))
    a = p.parse_args()
    if a.cmd == "run":
        import os
        base = Path(os.environ["DATA_DIR"]) / "models/comma"
        a.distill = a.distill or str(base / "openpilot.distill")
        a.model = a.model or str(base / "worldmodel-4B/model.fp8_nvfp4.torchpackage")
        a.vae_dir = a.vae_dir or str(base / "vit-ae-2x-f8c32")
        sys.path.insert(0, a.distill)
        run(a)
    else:
        plot(a)
