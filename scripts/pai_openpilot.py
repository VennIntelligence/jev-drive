#!/usr/bin/env python
"""Reverse check: openpilot on PhysicalAI-AV, Alpamayo's home domain (todos/2026-09-24-zeroshot-exam/
nuscenes-physicalai.md). The 31 clips of the Alpamayo smoke run, t0 = 5.1 s, GT = the official loader's 64-step
future (0.1 ... 6.4 s, t0 rig frame at the rear axle), the same conventions as Alpamayo's minADE 0.738 m.

  frames  (Alpamayo venv: physical_ai_av) camera_front_wide_120fov on a 20 Hz clock from t0 - 5.0 s to t0,
          rendered into openpilot's road / wide model frames with the clip's own f-theta calibration (calib frame
          = the rig axes), + GT and history -> $DATA_DIR/runs/pai_op/frames/<clip>.npz
  run     (openpilot venv) zero state, step every frame, plan at t0 -> rear-axle points at 0.1 ... 6.4 s
  score   (any venv) ADE / FDE vs GT, paired with Alpamayo's per-clip samples from the NAVSIM adapter ablation
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

T0 = 5_100_000
DT = 50_000                      # modeld's 20 Hz clock
N_STEPS = 101                    # t0 - 5.0 s ... t0: fills the 4.8 s feature context (the clip starts ~5.1 s before t0)
ALP_T = np.arange(1, 65) * 0.1
MODELS = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}
ACTION_T = (0.275, 0.525)
CAM = "camera_front_wide_120fov"


def root(*p) -> Path:
    d = data_dir() / "runs" / "pai_op" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def calib(avdi, clip: str) -> dict:
    import pandas as pd
    from jevdrive.alpamayo.data import cache_dir
    snap = next((cache_dir() / "datasets--nvidia--PhysicalAI-Autonomous-Vehicles" / "snapshots").iterdir())
    ch = avdi.get_clip_chunk(clip)
    i = pd.read_parquet(snap / f"calibration/camera_intrinsics/camera_intrinsics.chunk_{ch:04d}.parquet").loc[(clip, CAM)]
    e = pd.read_parquet(snap / f"calibration/sensor_extrinsics/sensor_extrinsics.chunk_{ch:04d}.parquet").loc[(clip, CAM)]
    return {"cx": i.cx, "cy": i.cy, "fw": [i[f"fw_poly_{k}"] for k in range(5)], "w": int(i.width), "h": int(i.height),
            "R": G.quat_to_matrix((e.qx, e.qy, e.qz, e.qw)), "xyz": np.array([e.x, e.y, e.z])}


def ftheta_index(cal: dict) -> tuple[list, float]:
    """Nearest-pixel gather indices of the road / wide model frames in the f-theta image, and their coverage."""
    out, cov = [], []
    for k in ("road", "wide"):
        r = G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H) @ cal["R"]           # rig -> camera (OpenCV)
        th = np.arccos(np.clip(r[..., 2], -1, 1))
        rho = np.maximum(np.hypot(r[..., 0], r[..., 1]), 1e-12)
        rad = np.polynomial.polynomial.polyval(th, cal["fw"])
        u, v = cal["cx"] + rad * r[..., 0] / rho, cal["cy"] + rad * r[..., 1] / rho
        ok = (u >= -.5) & (u <= cal["w"] - .5) & (v >= -.5) & (v <= cal["h"] - .5)
        cov.append(float(ok.mean()))
        x = np.clip(np.rint(u), 0, cal["w"] - 1).astype(np.int64)
        y = np.clip(np.rint(v), 0, cal["h"] - 1).astype(np.int64)
        out.append((y * cal["w"] + x).ravel())
    return out, min(cov)


def pack(ycc, idx):
    p = ycc.reshape(-1, 3)[idx].reshape(256, 512, 3)
    Y = p[..., 0]
    uv = np.rint(p[..., 1:].reshape(128, 2, 256, 2, 2).astype(np.float32).mean((1, 3))).astype(np.uint8)
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv[..., 0], uv[..., 1]])


def cmd_frames(a, log):
    import cv2
    from PIL import Image
    from jevdrive.alpamayo import data as D
    avdi = D.interface()
    clips = json.loads((D.cache_dir() / "clips.json").read_text())
    for c in clips:
        cal = calib(avdi, c)
        ix, cov = ftheta_index(cal)
        cam = avdi.get_clip_feature(c, getattr(avdi.features.CAMERA, CAM.upper()), maybe_stream=False)
        ts = T0 - DT * np.arange(N_STEPS - 1, -1, -1, dtype=np.int64)
        imgs, fts = cam.decode_images_from_timestamps(ts)
        ycc = [np.asarray(Image.fromarray(im).convert("YCbCr")) for im in imgs]   # BT.601 full range, as JPEG
        frames = np.stack([np.stack([pack(y, ix[0]), pack(y, ix[1])]) for y in ycc])
        d = D.load_clip(c, avdi)
        gt = d["ego_future_xyz"][0, 0, :, :2].numpy()
        hx = d["ego_history_xyz"][0, 0].numpy()
        np.savez(root("frames") / f"{c}.npz", frames=frames, t=ts, fts=np.asarray(fts, np.int64), gt=gt, hist=hx,
                 cam_xyz=cal["xyz"], coverage=cov)
        if a.preview and c in clips[:4]:   # native front-wide at t0 next to what openpilot sees (luma of road / wide)
            from jevdrive.openpilot.frames import unpack_luma
            nat = cv2.resize(imgs[-1], (768, 432))
            op = np.concatenate([unpack_luma(frames[-1, 0]), unpack_luma(frames[-1, 1])], 0)
            op = cv2.cvtColor(cv2.resize(op, (432 * 512 // 512, 432)), cv2.COLOR_GRAY2RGB)
            cv2.imwrite(str(log.dir / f"{c}_views.jpg"), np.concatenate([nat, op], 1)[..., ::-1])
        dt = (np.asarray(fts) - ts) * 1e-3
        log.info(f"{c}: coverage {cov:.3f}, frame - step time {dt.min():+.1f} ... {dt.max():+.1f} ms, "
                 f"speed {np.linalg.norm(hx[-1] - hx[-2]) / 0.1:.1f} m/s")


def cmd_run(a, log):
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    files = sorted(root("frames").glob("*.npz"))
    out = {}
    for m, backend in MODELS.items():
        mod = OPModel(m, backend)
        preds = {}
        t1 = time.time()
        for f in files:
            z = np.load(f)
            mod.reset()
            for fr in z["frames"]:
                raw = mod.step(fr, desire=np.zeros(8, np.float32), traffic=(1, 0), action_t=ACTION_T)
            v = float(np.linalg.norm(z["hist"][-1, :2] - z["hist"][-2, :2]) / 0.1)
            o = decode(raw, mod.slices, v, ACTION_T)
            p = np.stack([o["plan_pos"][:, 0], -o["plan_pos"][:, 1]], -1)
            psi = -np.asarray(o["plan_yaw"], np.float64)
            dv = z["cam_xyz"][:2]
            c, s = np.cos(psi), np.sin(psi)
            rear = dv + p - np.stack([c * dv[0] - s * dv[1], s * dv[0] + c * dv[1]], -1)
            preds[f.stem] = np.stack([np.interp(ALP_T, T_IDXS, rear[:, k]) for k in range(2)], -1)
        out[m] = preds
        log.info(f"{m}: {len(files)} clips in {time.time() - t1:.0f} s")
    clips = sorted(out["small"])
    np.savez(root() / "preds.npz", clips=np.array(clips), **{m: np.stack([out[m][c] for c in clips]) for m in out})


def cmd_score(a, log):
    import pandas as pd
    P = np.load(root() / "preds.npz")
    clips = list(P["clips"])
    gt = np.stack([np.load(root("frames") / f"{c}.npz")["gt"] for c in clips])
    hist = np.stack([np.load(root("frames") / f"{c}.npz")["hist"] for c in clips])
    v0 = np.linalg.norm(hist[:, -1, :2] - hist[:, -2, :2], axis=1) / 0.1
    cv = np.stack([v0[:, None] * ALP_T, 0 * ALP_T[None].repeat(len(clips), 0)], -1)   # straight at current speed
    alp = pd.read_csv(Path(a.alpamayo_csv))
    alp = alp[alp.config == "native"]
    rows, per = [], {}

    def metr(pred):  # (n, 64, 2)
        e = np.linalg.norm(pred - gt, axis=-1)
        return {"ade64": e.mean(1), "ade40": e[:, :40].mean(1), "ade30": e[:, :30].mean(1), "fde64": e[:, -1]}
    for m in ("small", "cinque", "lebowski"):
        per[f"openpilot {m}"] = metr(P[m])
    per["cv (straight, current speed)"] = metr(cv)
    g = alp.groupby("clip")
    a64 = g.ade64.mean().reindex(clips).to_numpy()
    per["Alpamayo 1.5, one sample (mean of 6)"] = {"ade64": a64, "ade40": g.ade40.mean().reindex(clips).to_numpy()}
    per["Alpamayo 1.5, ORACLE minADE_6"] = {"ade64": g.ade64.min().reindex(clips).to_numpy(),
                                             "ade40": g.ade40.min().reindex(clips).to_numpy()}
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(clips), (10000, len(clips)))
    ref = per["Alpamayo 1.5, one sample (mean of 6)"]["ade64"]
    for name, d in per.items():
        r = {"row": name, "n": len(clips)}
        for k, v in d.items():
            bs = v[B].mean(1)
            r[k], r[k + "_lo"], r[k + "_hi"] = v.mean(), *np.percentile(bs, [2.5, 97.5])
        dd = (d["ade64"] - ref)[B].mean(1)
        r["d_ade64_vs_alp"], r["d_lo"], r["d_hi"] = (d["ade64"] - ref).mean(), *np.percentile(dd, [2.5, 97.5])
        r["median_ade64"] = float(np.median(d["ade64"]))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(root() / "results.csv", index=False)
    np.savez(root() / "per_clip.npz", clips=np.array(clips), gt=gt, cv=cv, **{m: P[m] for m in ("small", "cinque", "lebowski")},
             alp_mean=a64)
    log.info("\n" + df.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("frames", "run", "score"))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--alpamayo-csv", default=str(data_dir() / "runs/navsim_zs/alpamayo_adapt/20260924-172401/adapt.csv"))
    a = ap.parse_args()
    log = RunLog("pai_op", a.cmd)
    {"frames": cmd_frames, "run": cmd_run, "score": cmd_score}[a.cmd](a, log)
    log.event("end")
