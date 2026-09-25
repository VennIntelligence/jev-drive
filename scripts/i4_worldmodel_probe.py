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

ARMS = {  # name: (curvature 1/m, accel m/s^2); env convention: +curvature = right turn (WASD "D")
    "straight": (0.0, 0.0),
    "left": (-0.01, 0.0),
    "right": (0.01, 0.0),
    "brake": (0.0, -4.0),
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
    results = {"steps": a.steps, "sampling_steps": a.sampling_steps, "cfg": a.cfg, "seeds": a.seeds, "arms": ARMS,
               "clips": []}
    for c, (seg, start) in enumerate(zip(a.segment, a.start)):
        clip = rollout_clip(rt, rl, Path(seg), start, a)
        np.savez_compressed(rl.dir / f"rollout_{c}.npz", **{k: v for k, v in clip.items() if k != "metrics"})
        results["clips"].append({"segment": Path(seg).name, "start_5hz": start, "speed0": float(clip["speeds"][rt.history_frames - 1]),
                                 "step_ms": clip["step_ms"], "metrics": clip["metrics"]})
    if a.timing:
        results["timing"] = timing(rt, clip["timing_inputs"], a.cfg)
    results["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
    results["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    rl.info(f"timing {json.dumps(results.get('timing'))} peak alloc {results['peak_alloc_gb']:.1f} GB")
    json.dump(results, open(rl.dir / "results.json", "w"), indent=1)
    rl.info(f"results -> {rl.dir}")
    rl.event("end")
    rl.close()


def rollout_clip(rt, rl, seg: Path, start: int, a):
    """Roll out every arm x seed from one context clip, batched; returns latents, decoded frames, plans, metrics."""
    import torch
    from rl.env import Physics

    H, F, K = rt.history_frames, rt.future_frames, a.steps
    lat, frames, speeds = load_clip(rt, seg, start - H, H + K + F, a.gpu_id)
    rl.info(f"clip {seg.name} 5Hz frames {start - H}..{start + K + F}, speed at start {speeds[H - 1]:.1f} m/s")
    names = [f"{arm}/s{sd}" for arm in ARMS for sd in range(a.seeds)]
    B, dev, dt = len(names), rt.device, rt.model_dtype
    ctx = lat[:H].clone()[None].repeat(B, 1, 1, 1, 1)
    future = lat[H + K:H + K + F].clone()[None].repeat(B, 1, 1, 1, 1)  # logged anchors at the rollout end
    future_fidx = torch.arange(H + K, H + K + F)
    phys = [Physics(float(speeds[H - 1])) for _ in names]
    gens = [torch.Generator().manual_seed(int(n.split("/s")[1])) for n in names]  # same seed -> same noise across arms
    out_lat, plans, ego, step_ms = [], [], [], []
    for k in range(K):
        noise = torch.stack([torch.randn(ctx.shape[2:], generator=g) for g in gens])[:, None]
        latents = torch.cat((future, ctx, noise.to(ctx)), 1)
        pos = latents.new_zeros((B, latents.shape[1], 3))
        eul = torch.zeros_like(pos)
        mask = torch.ones(latents.shape[:2], dtype=torch.int64, device=dev)
        e = []
        for i, n in enumerate(names):
            dx, dy, yaw = phys[i].step(*ARMS[n.split("/")[0]])
            pos[i, -1, 0], pos[i, -1, 1], eul[i, -1, 2] = dx, dy, yaw
            e.append((dx, dy, yaw, phys[i].speed))
        mask[:, -1] = 0  # 0 = pose given (the target frame); 1 = masked (anchors and history, as in rl/env.py)
        fidxs = torch.cat((future_fidx - k, torch.arange(H + 1)))[None].repeat(B, 1).to(dev)
        inputs = dict(latents=latents.to(dev, dt), augments_pos_ref_augment=pos.to(dev, dt),
                      ref_augment_from_augments_euler=eul.to(dev, dt), pose_mask=mask, fidxs=fidxs)
        torch.cuda.synchronize(); t = time.perf_counter()
        o = rt.model.generate(**{**inputs, "latents": inputs["latents"].clone()}, steps=a.sampling_steps,
                              num_prefill_frames=latents.shape[1] - 1, dtype=dt, inference_schedule="linear", cfg=a.cfg)
        torch.cuda.synchronize(); step_ms.append((time.perf_counter() - t) * 1e3)
        new = o["latents"][:, 0]
        ctx = torch.cat((ctx[:, 1:], new[:, None].to(ctx)), 1)
        out_lat.append(new.float().cpu())
        p = o["plan"].float().cpu()
        plans.append(p[:, : p.shape[1] // 2].reshape(B, -1, 15).numpy())
        ego.append(e)
        rl.event("step_end", clip=seg.name, step=k, ms=step_ms[-1])
    out_lat = torch.stack(out_lat, 1)  # (B, K, 32, 16, 32)
    dec = np.stack([rt.decode(out_lat[i].to(dev)) for i in range(B)])
    if dec.shape[-1] != 6:  # decoder returns NCHW -> (B, K, 128, 256, 6)
        dec = np.ascontiguousarray(np.moveaxis(dec, 2, -1))
    ego = np.array(ego).transpose(1, 0, 2)
    plans = np.stack(plans, 1)
    return dict(names=np.array(names), latents=out_lat.numpy().astype(np.float16), decoded=dec, context_frames=frames,
                plans=plans, ego=ego, speeds=speeds, step_ms=step_ms,
                metrics=measure(names, out_lat.numpy(), dec, frames[H - 1], plans, ego),
                timing_inputs={k: v[:1] for k, v in inputs.items()})


def timing(rt, inputs, cfg):
    """ms per generate() call at batch 1 / 8 / 32 and 15 / 30 denoising steps, plus VAE decode."""
    import torch
    res = {}
    for bs in (1, 8, 32):
        kw = {k: v.repeat(bs, *([1] * (v.dim() - 1))) for k, v in inputs.items()}
        for steps in (15, 30):
            call = lambda: rt.model.generate(**{**kw, "latents": kw["latents"].clone()}, steps=steps,
                                             num_prefill_frames=kw["latents"].shape[1] - 1, dtype=rt.model_dtype,
                                             inference_schedule="linear", cfg=cfg)
            call(); torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            res[f"gen_ms_b{bs}_s{steps}"] = (time.perf_counter() - t) * 200
    x = inputs["latents"][:1, -2]
    rt.decode(x); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        rt.decode(x)
    torch.cuda.synchronize()
    res["decode_ms_b1"] = (time.perf_counter() - t) * 100
    return res


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


def measure(names, lat, dec, last_ctx, plans, ego):
    """Per arm/seed: per-step latent RMS / pixel MAE / flow differences to `straight` with the same noise seed,
    the null being `straight` with another seed; plan yaw rate and speed read from the world model's plan head."""
    std = lat.std()
    m = {}
    for i, n in enumerate(names):
        arm, sd = n.split("/")
        ref = names.index(f"straight/{sd}") if arm != "straight" else names.index("straight/s0")
        if arm == "straight" and sd == "s0":
            ref = names.index("straight/s1") if "straight/s1" in names else ref
        seq = np.concatenate((last_ctx[None], dec[i]))
        fl = np.array([flow_stats(seq[t][..., 3:], seq[t + 1][..., 3:]) for t in range(len(dec[i]))])
        m[n] = {"ref": names[ref],
                "lat_rms_vs_ref": (np.sqrt(((lat[i] - lat[ref]) ** 2).mean(axis=(1, 2, 3))) / std).tolist(),
                "pix_mae_vs_ref": np.abs(dec[i].astype(np.float32) - dec[ref].astype(np.float32)).mean(axis=(1, 2, 3)).tolist(),
                "flow_x_wide": fl[:, 0].tolist(), "flow_expand_wide": fl[:, 1].tolist(),
                "plan_yaw_rate0": plans[i, :, 0, 14].tolist(), "plan_v0": plans[i, :, 0, 3].tolist(),
                "cmd_yaw_per_frame": ego[i, :, 2].tolist(), "cmd_speed": ego[i, :, 3].tolist()}
    return m


WIDE_FL = 227.5  # px, wide view focal length at the model's 256x128 input (sbigmodel intrinsics / 2)
COLORS = {"straight": "#7F7F7F", "null": "#000000", "left": "#0072B2", "right": "#D55E00", "brake": "#009E73"}
LABELS = {"straight": "straight", "null": "straight, other seed (null)", "left": r"left, $\kappa$=0.01 m$^{-1}$",
          "right": r"right, $\kappa$=0.01 m$^{-1}$", "brake": r"brake, $a$=$-4$ m s$^{-2}$"}


def summarize(res):
    """Per arm, arrays (clip*seed, step) of the metrics; `null` = straight seeds >= 1 against straight seed 0."""
    out = {}
    for clip in res["clips"]:
        for n, m in clip["metrics"].items():
            arm, sd = n.split("/")
            key = "null" if arm == "straight" and sd != "s0" else arm
            ref = clip["metrics"][m["ref"]]
            row = {k: np.array(v) for k, v in m.items() if k != "ref"}
            row["dflow_x"] = row["flow_x_wide"] - np.array(ref["flow_x_wide"])
            row["dflow_expand"] = row["flow_expand_wide"] - np.array(ref["flow_expand_wide"])
            for k, v in row.items():
                out.setdefault(key, {}).setdefault(k, []).append(v)
    return {a: {k: np.stack(v) for k, v in d.items()} for a, d in out.items()}


def plot(a):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from jevdrive.plots import PAGE, STYLE
    d = Path(a.run_dir)
    res = json.load(open(d / "results.json"))
    z = np.load(d / f"rollout_{a.clip}.npz")
    names, dec, ctx = list(z["names"]), z["decoded"], z["context_frames"]
    K = res["steps"]
    H = ctx.shape[0] - K - 5
    ts = [t for t in (0, 4, 9, 14) if t < K]
    out = Path(a.out_dir)
    with mpl.rc_context(STYLE):
        arms = list(ARMS)
        cols = 1 + len(ts)
        fig, axes = plt.subplots(len(arms), cols, figsize=(PAGE, PAGE / cols * 0.5 * len(arms) + 0.15),
                                 gridspec_kw=dict(wspace=0.03, hspace=0.05))
        for r, arm in enumerate(arms):
            i = names.index(f"{arm}/s0")
            for c in range(cols):
                ax = axes[r, c]
                ax.imshow(ctx[H - 1][..., 3:] if c == 0 else dec[i, ts[c - 1]][..., 3:])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title("last context frame" if c == 0 else f"generated, +{(ts[c - 1] + 1) / 5:.1f} s", fontsize=8)
                if c == 0:
                    ax.set_ylabel(arm, fontsize=8)
        fig.savefig(out / f"i4_worldmodel_actions{'' if a.clip == 0 else f'_clip{a.clip}'}.png", dpi=140)
        plt.close(fig)

        S = summarize(res)
        t = (np.arange(K) + 1) / 5
        fig, axes = plt.subplots(1, 3, figsize=(PAGE, 1.8))

        def band(ax, x, y, key, **kw):
            mu, sd = y.mean(0), y.std(0)
            ax.plot(x, mu, color=COLORS[key], label=LABELS[key], **kw)
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[key], alpha=0.15, lw=0)

        for key in ("null", "left", "right", "brake"):
            band(axes[0], t, S[key]["dflow_x"], key)
        for key, sign in (("left", 1), ("right", -1)):  # yaw of -psi (left) moves the scene +WIDE_FL*psi px (right)
            exp = -WIDE_FL * S[key]["cmd_yaw_per_frame"].mean(0)
            axes[0].plot(t, exp, color=COLORS[key], ls=":", lw=0.8)
        axes[0].set_ylabel(r"$\Delta$ horizontal flow vs straight (px/frame)")
        for key in ("straight", "left", "right"):
            band(axes[1], t, S[key]["plan_yaw_rate0"], key)
            axes[1].plot(t, S[key]["cmd_yaw_per_frame"].mean(0) * 5, color=COLORS[key], ls=":", lw=0.8)
        axes[1].set_ylabel("plan-head yaw rate (rad/s)")
        for key in ("straight", "brake"):
            band(axes[2], t, S[key]["plan_v0"], key)
            axes[2].plot(t, S[key]["cmd_speed"].mean(0), color=COLORS[key], ls=":", lw=0.8)
        axes[2].set_ylabel("plan-head speed (m/s)")
        for ax in axes:
            ax.set_xlabel("time after context (s)")
            ax.grid(True)
        h, l = [], []
        for ax in axes:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh); l.append(ll)
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(l))
        fig.tight_layout()
        fig.savefig(out / "i4_worldmodel_metrics.png", dpi=250)
        fig.savefig(d / "i4_worldmodel_metrics.pdf")
        plt.close(fig)

    rows = []
    for key in ("null", "left", "right", "brake", "straight"):
        s = S[key]
        rows.append({"arm": key, "n": len(s["pix_mae_vs_ref"]),
                     **{f"pix_mae_{k + 1}": s["pix_mae_vs_ref"][:, k].mean() for k in (0, 4, K - 1)},
                     **{f"lat_rms_{k + 1}": s["lat_rms_vs_ref"][:, k].mean() for k in (0, 4, K - 1)},
                     "dflow_x_3_end": s["dflow_x"][:, 3:].mean(), "dflow_x_3_end_sd": s["dflow_x"][:, 3:].mean(1).std(),
                     "expected_dflow_x": float(-WIDE_FL * s["cmd_yaw_per_frame"].mean()),
                     "dexpand_5_end": s["dflow_expand"][:, 5:].mean(),
                     "plan_yaw_rate": s["plan_yaw_rate0"].mean(), "cmd_yaw_rate": s["cmd_yaw_per_frame"].mean() * 5,
                     "plan_v0_end": s["plan_v0"][:, -1].mean(), "cmd_v_end": s["cmd_speed"][:, -1].mean()})
    import csv
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(open(d / "summary.csv").read())
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--distill", default=None, help="openpilot.distill checkout (default $DATA_DIR/models/comma/openpilot.distill)")
    r.add_argument("--model", default=None, help="model.fp8_nvfp4.torchpackage")
    r.add_argument("--vae-dir", default=None)
    r.add_argument("--segment", nargs="+", required=True, help="comma1M segment dirs with fcamera/ecamera.hevc")
    r.add_argument("--start", type=int, nargs="+", required=True, help="first generated 5 Hz frame index, per segment")
    r.add_argument("--seeds", type=int, default=4, help="noise seeds per arm (seed 1.. of straight = null)")
    r.add_argument("--timing", action="store_true")
    r.add_argument("--steps", type=int, default=15, help="generated 5 Hz frames per arm")
    r.add_argument("--sampling-steps", type=int, default=15)
    r.add_argument("--cfg", type=float, default=2.0)
    r.add_argument("--gpu-id", type=int, default=0)
    r.add_argument("--tag", default="probe")
    q = sub.add_parser("plot")
    q.add_argument("run_dir")
    q.add_argument("--out-dir", default=str(REPO / "research/figs"))
    q.add_argument("--clip", type=int, default=0, help="clip shown in the image grid")
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
