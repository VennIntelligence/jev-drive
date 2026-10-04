"""Stage 3 (jevdrive env, one leased card): the shipped WL-2 latent world model (arms B and A, seeds 0-4, no retraining) on the WOD anchors.

  vjepa   V-JEPA 2 `mean` of every 5 Hz frame of the 10 segments, as the WL data: 3 cameras (FRONT, FRONT_LEFT, FRONT_RIGHT) x the frame and the 3
          before it (5 Hz), squashed to 256 x 256 -> runs/wm_vs_reproj/vjepa/<seg>.npy (n, 3072) fp16
  predict  per anchor and arm (nominal + perturbations of wm_common.arms): history z = [openpilot `temporal` of the real stream (wm_op G rows) |
           V-JEPA mean | s = (e_y, e_psi, v, a_prev, omega_prev)] for the 8 history steps, future actions (a, omega) of the pose sequence, source flag 1
           (as the WL-2 fork readouts), the checkpoint's own standardisation -> 10 predicted steps; saved: the openpilot block (512) and, for B, s.
           -> runs/wm_vs_reproj/w/<anchor>.npz
Route for e_y / e_psi: the real ego path smoothed with a 1.5 s Gaussian (WL used the CARLA route polyline). CARLA signs: omega and e_psi right +, e_y left +.
  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/jevdrive/bin/python experiments/world_model/scripts/wm_w.py vjepa|predict
"""
import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
import wm_common as C  # noqa: E402

VJ = C.RUN / "vjepa"
WOUT = C.RUN / "w"
ACT_SCALE = np.array([3.0, 0.5], np.float32)
V_SCALE = 10.0
SEEDS = range(5)
ARMS = ("B", "A")


def cmd_vjepa(a):
    import torch
    from PIL import Image
    from jevdrive import features as F
    VJ.mkdir(parents=True, exist_ok=True)
    fx = F.VJepaFeatures(frames=4)
    for f in sorted((C.RUN / "prep").glob("*.npz")):
        out = VJ / f"{f.stem}.npy"
        if out.exists():
            continue
        d = pickle.load(open(C.SF / f"{f.stem}.pkl", "rb"))
        sel = np.load(f)["frame"]
        tf = {}                                                                # (frame, cam) -> transformed tensor

        def get(k, c):
            if (k, c) not in tf:
                tf[(k, c)] = fx.tf(Image.open(io.BytesIO(d["frames"][sel[k]]["jpg"][c])).convert("RGB"))
            return tf[(k, c)]
        clips = [torch.stack([get(max(k - 3 + q, 0), c) for q in range(4)]) for k in range(len(sel)) for c in (1, 2, 3)]
        feats = []
        for lo in range(0, len(clips), 48):
            feats.append(fx(torch.stack(clips[lo: lo + 48]))["mean"].half().cpu().numpy())
        np.save(out, np.concatenate(feats).reshape(len(sel), -1))
        print("vjepa", f.stem[:12], flush=True)
        del d, tf, clips


def load_arm(arm, seed, dev="cuda"):
    import torch
    from experiments.world_model.archive import wl_model as M
    from jevdrive.common import data_dir
    p = sorted((data_dir() / "runs" / "wl2" / "model" / arm / f"seed{seed}").glob("*/model.pt"))[-1]
    ck = torch.load(p, map_location="cpu")
    model = M.build(ck["mu"].numel()).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in ck["state"].items()})
    return model.eval(), ck["mu"].to(dev).float(), ck["sd"].to(dev).float()


def route_track(pose):
    """Smoothed real path -> (x, y, heading) per frame, for e_y (left +) and e_psi (CARLA sign: right +)."""
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(pose[:, :2], 7.5, axis=0, mode="nearest")
    t = np.gradient(sm, axis=0)
    return sm, np.arctan2(t[:, 1], t[:, 0])


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def cmd_predict(a):
    import torch
    out_dir = WOUT
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {(arm, s): load_arm(arm, s) for arm in ARMS for s in SEEDS}
    files = sorted((C.RUN / "op").glob("*.npz"))
    for f in files:
        out = out_dir / f.name
        if out.exists():
            continue
        z = np.load(f)
        an = json.loads(str(z["anchor"]))
        prep = np.load(C.RUN / "prep" / f"{an['seg']}.npz")
        pose, v = prep["pose"], prep["v"]
        i, H, K = an["i"], C.HIST, C.K
        vj = np.load(VJ / f"{an['seg']}.npy").astype(np.float32)
        sm, th = route_track(pose)
        t = np.arange(i - H + 1, i + 1)
        ey = (pose[t, 0] - sm[t, 0]) * -np.sin(th[t]) + (pose[t, 1] - sm[t, 1]) * np.cos(th[t])
        epsi = -wrap(pose[t, 2] - th[t])
        a_prev = (v[t] - v[t - 1]) / C.DT
        w_prev = -wrap(pose[t, 2] - pose[t - 1, 2]) / C.DT
        S = np.stack([ey, epsi, v[t], a_prev, w_prev], 1).astype(np.float32)
        zh_raw = np.concatenate([z["G/z"][:H], vj[t]], 1)                      # (8, 3584)
        hact = torch.as_tensor(np.stack([a_prev / ACT_SCALE[0], w_prev / ACT_SCALE[1], v[t] / V_SCALE], 1), dtype=torch.float32, device="cuda")
        P = {k.split("/")[1]: z[k] for k in z.files if k.startswith("pose/")}
        vf = v[i: i + K + 1]
        res = {}
        for name, Q in P.items():
            fact = torch.as_tensor(C.w_actions(Q, vf) / ACT_SCALE, device="cuda")
            for (arm, s), (model, mu, sd) in models.items():
                raw = np.concatenate([zh_raw, S], 1) if arm == "B" else zh_raw
                zh = ((torch.as_tensor(raw, device="cuda").float() - mu) / sd)[None]
                one = lambda n: torch.ones((1, n), dtype=torch.long, device="cuda")  # noqa: E731
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    pred = model(zh, hact[None], fact[None], one(H), one(K)).float()[0]
                pred = (pred * sd + mu).cpu().numpy()
                res.setdefault(f"{arm}/{name}/op", []).append(pred[:, :512])
                if arm == "B":
                    res.setdefault(f"{arm}/{name}/s", []).append(pred[:, -5:])
        np.savez(out.with_suffix(".tmp.npz"), anchor=z["anchor"], **{k: np.stack(x) for k, x in res.items()})
        out.with_suffix(".tmp.npz").replace(out)
        print("predicted", f.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("vjepa", "predict"))
    a = ap.parse_args()
    {"vjepa": cmd_vjepa, "predict": cmd_predict}[a.cmd](a)
