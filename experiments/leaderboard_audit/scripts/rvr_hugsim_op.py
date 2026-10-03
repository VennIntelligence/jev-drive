"""Real vs render, HUGSIM nuScenes scenes: openpilot (Cinque) on paired real / rendered frame sequences through the closed-loop adapter
(hugsim_zs.OpenpilotFrames with calibs(nuscenes_camera.yaml, cam_rect)). Pre-registration:
experiments/leaderboard_audit/plans/2026-10-04-real-vs-render-prereg.md.

  fit     image-side fixes fitted on the calibration scenes (scene[::4]): sharpening to the real high-frequency share, per-channel
          colour affine, additive noise to the real noise level -> fixes.json; plus per-arm image statistics -> imgstats.json
  stream  continuous 20 Hz rollouts from a zero state over each scene (every arm), raw heads at every new-frame step -> <scene>/stream.npz
          (--env: the closed-loop-rig render only -> <scene>/stream_env.npz)
  probe   history-yaw probes (G at 1 and 10 deg/s, launch gain L) every 12 frames from frame 18 -> probe_<scene>.npz

envs/openpilot on a leased card:
  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/openpilot/bin/python experiments/leaderboard_audit/scripts/rvr_hugsim_op.py stream|probe [scenes ...]
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from jevdrive import hugsim_zs as Z  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
HS = D / "datasets/hugsim/scenes/nuscenes"
OUT = D / "runs/real_vs_render/hugsim"
CAMYAML = D / "third_party/HUGSIM/configs/sim/nuscenes_camera.yaml"
CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
STREAM_ARMS = ("real", "render", "render_sharp", "render_stat", "render_all", "real_sharp", "real_all")
PROBE_ARMS = ("real", "render", "render_all")
DT = 0.05


def scenes():
    return sorted(p.name for p in OUT.iterdir() if (p / "render.npy").exists() and (p / "real.npy").exists())


def calib_scenes():
    return scenes()[::4]


# ---------------------------------------------------------------- adapter and fixes

def cam_params():
    import yaml
    from scipy.spatial.transform import Rotation
    cfg = yaml.safe_load(CAMYAML.read_text())["cams"]
    out = {}
    for c, p in cfg.items():
        v2c = np.eye(4)
        v2c[:3, :3] = Rotation.from_euler("XYZ", p["extrinsics"]["v2c_rot"], degrees=True).as_matrix()
        v2c[:3, 3] = p["extrinsics"]["v2c_trans"]
        it = dict(p["intrinsics"])
        it["fovx"], it["fovy"] = np.radians(it["fovx"]), np.radians(it["fovy"])
        out[c] = {"intrinsic": it, "v2c": v2c}
    return out


def adapter():
    return Z.OpenpilotFrames(Z.calibs(cam_params(), Z.rect_matrix(str(CAMYAML))), CAMS)


def hf_share(g, r0=0.25):
    """Share of spectral power (DC excluded) above r0 cycles/px (sc_hugsim_probe.hf)."""
    g = g.astype(np.float64)
    F = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))))) ** 2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(g.shape[0])), np.fft.fftshift(np.fft.fftfreq(g.shape[1])), indexing="ij")
    return float(F[np.hypot(fy, fx) > r0].sum() / F.sum())


def noise_sigma(g):
    """Immerkaer (1996) fast noise estimate of a grey image (std of additive white noise)."""
    import cv2
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    r = cv2.filter2D(g.astype(np.float64), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def grey(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def apply_fix(rgb, kind, P, seed=0):
    """rgb (H, W, 3) uint8 -> fixed uint8. kind: none | sharp | stat | all (stat = colour affine + noise; all = stat then sharp)."""
    import cv2
    if kind == "none":
        return rgb
    x = rgb.astype(np.float32)
    if kind in ("stat", "all"):
        x = x * np.asarray(P["gain"], np.float32) + np.asarray(P["bias"], np.float32)
    if kind in ("sharp", "all"):
        b = cv2.GaussianBlur(x, (0, 0), P["sigma"])
        x = x + P["amount"] * (x - b)
    if kind in ("stat", "all") and P["noise"] > 0:
        x = x + np.random.default_rng(seed).normal(0, P["noise"], x.shape[:2])[..., None].astype(np.float32)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def arm_src(arm):
    if arm == "env":
        return "render_env", "none"
    src, _, kind = arm.partition("_")
    return src, kind or "none"


def frame_rgb(arrs, arm, k, P):
    src, kind = arm_src(arm)
    return {c: apply_fix(np.asarray(arrs[src][k, j]), kind, P, seed=k * 3 + j) for j, c in enumerate(CAMS)}


def load(scene):
    return {s: np.load(OUT / scene / f"{s}.npy", mmap_mode="r") for s in ("real", "render", "render_env")
            if (OUT / scene / f"{s}.npy").exists()}


def meta(scene):
    m = json.load(open(HS / scene / "meta_data.json"))
    fr = [f for f in m["frames"] if "/CAM_FRONT/" in f["rgb_path"]]
    T = np.array([f["timestamp"] for f in fr])
    P = np.array([np.asarray(f["camtoworld"])[:3, 3] for f in fr])
    d = np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1)
    v = np.r_[d[0], d] / np.r_[T[1] - T[0], np.diff(T)]
    v = np.convolve(np.pad(v, 3, mode="edge"), np.ones(7) / 7, "valid")     # 0.6 s box filter: the sweep timing alternates
    return T, v


# ---------------------------------------------------------------- fit

def cmd_fit(a):
    import cv2
    stats = {}
    S = calib_scenes()
    fr = range(6, 180, 10)
    R = {"real": [], "render": []}
    for s in S:
        A = load(s)
        for k in fr:
            for src in R:
                R[src].append(np.asarray(A[src][k, 0]))
    real, rend = np.stack(R["real"]).astype(np.float64), np.stack(R["render"]).astype(np.float64)
    mr, sr = real.reshape(-1, 3).mean(0), real.reshape(-1, 3).std(0)
    mo, so = rend.reshape(-1, 3).mean(0), rend.reshape(-1, 3).std(0)
    gain, bias = sr / so, mr - mo * sr / so
    hf_real = np.mean([hf_share(grey(x)) for x in R["real"]])
    n_real = np.mean([noise_sigma(grey(x)) for x in R["real"]])
    best = None
    for sigma in (0.7, 1.0, 1.5, 2.0):
        for amount in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            P = dict(gain=[1, 1, 1], bias=[0, 0, 0], noise=0.0, sigma=sigma, amount=amount)
            hf = np.mean([hf_share(grey(apply_fix(x, "sharp", P))) for x in R["render"][::3]])
            if best is None or abs(hf - hf_real) < best[0]:
                best = (abs(hf - hf_real), sigma, amount, hf)
    _, sigma, amount, hf_fit = best
    n_rend = np.mean([noise_sigma(grey(apply_fix(x, "stat", dict(gain=gain, bias=bias, noise=0.0)))) for x in R["render"]])
    noise = float(np.sqrt(max(0.0, n_real ** 2 - n_rend ** 2)))
    P = dict(gain=gain.tolist(), bias=bias.tolist(), sigma=sigma, amount=amount, noise=noise, calib_scenes=S,
             hf_real=hf_real, hf_render_sharp=hf_fit, noise_real=n_real, noise_render=n_rend)
    (OUT.parent / "fixes.json").write_text(json.dumps(P, indent=1))
    print(json.dumps(P, indent=1))
    # image statistics per arm over all scenes (CAM_FRONT, every 10th frame)
    for arm in ("real", "render", "render_sharp", "render_stat", "render_all"):
        rows = []
        for s in scenes():
            A = load(s)
            for k in fr:
                x = frame_rgb(A, arm, k, P)["CAM_FRONT"]
                g = grey(x)
                rows.append([hf_share(g), cv2.Laplacian(g, cv2.CV_64F).var(), noise_sigma(g), *x.reshape(-1, 3).mean(0), *x.reshape(-1, 3).std(0)])
        r = np.array(rows)
        stats[arm] = dict(zip(["hf", "lap", "noise", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"], np.median(r, 0).round(4).tolist()))
    psnr = {}
    for s in scenes():
        A = load(s)
        ps = [float(-10 * np.log10(np.mean((np.asarray(A["real"][k, 0], np.float64) - np.asarray(A["render"][k, 0], np.float64)) ** 2) / 255 ** 2)) for k in fr]
        psnr[s] = float(np.median(ps))
    stats["psnr_front_by_scene"] = psnr
    (OUT.parent / "imgstats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- model

def keep_layout(m):
    heads = sorted(((q, s) for q, s in m.slices.items() if q not in ("hidden_state", "pad")), key=lambda x: x[1].start)
    keep = np.concatenate([np.arange(s.start, s.stop) for _, s in heads])
    hs, o = {}, 0
    for q, s in heads:
        hs[q] = o
        o += s.stop - s.start
    return keep, hs


def model():
    from jevdrive.openpilot.model import OPModel
    return OPModel("cinque", "trt", cache=D / "runs/real_vs_render/trt_cache")


_W = {}


def _winit(scene, P):
    _W.update(op=adapter(), A=load(scene), P=P)


def _pack(job):
    """(arm, k, yaw_deg) -> packed model frames (2, 6, 128, 256)."""
    arm, k, yaw = job
    op = _W["op"]
    return op.pack(frame_rgb(_W["A"], arm, k, _W["P"]), op.rot_index(yaw) if abs(yaw) >= 0.05 else None)


def cmd_stream(a):
    arms, fn = (("env",), "stream_env.npz") if a.env else (STREAM_ARMS, "stream.npz")
    P = json.loads((OUT.parent / "fixes.json").read_text())
    m = model()
    keep, hs = keep_layout(m)
    for s in a.scenes or scenes():
        f = OUT / s / fn
        if f.exists() or (a.env and not (OUT / s / "render_env.npy").exists()):
            continue
        T, v = meta(s)
        n = len(T)
        steps = np.arange(int(T[-1] / DT) + 1) * DT
        kk = np.searchsorted(T, steps + 1e-6, side="right") - 1
        first = np.r_[True, kk[1:] != kk[:-1]]                       # the step at which frame kk becomes the newest
        H = {}
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=_winit, initargs=(s, P)) as ex:
            for arm in arms:
                packed = list(ex.map(_pack, [(arm, k, 0.0) for k in range(n)], chunksize=8))
                m.reset()
                out = np.full((n, len(keep)), np.nan, np.float32)
                for j, k in enumerate(kk):
                    raw = m.step(packed[k])
                    if first[j]:
                        out[k] = raw[keep]
                H[arm] = out
        np.savez(f, **H, T=T, v=v, info=json.dumps(dict(heads_slices=hs, arms=arms)))
        print(s, "stream", f"{time.time() - t0:.0f}s", flush=True)


def window(T, k, sec):
    """20 Hz steps over the last `sec` seconds ending at frame k: (frame index, time before t_k) per step, oldest first."""
    ts = T[k] - np.arange(int(round(sec / DT)), -1, -1) * DT
    j = np.clip(np.searchsorted(T, ts + 1e-6, side="right") - 1, 0, None)
    return j, T[k] - ts


def probe_jobs(arm, T, k):
    """The probe variants of one frame: (name, [(arm, frame, yaw_deg)] per step)."""
    jw, back = window(T, k, 1.5)
    V = {"normal": [(arm, int(j), 0.0) for j in jw]}
    for w in (1.0, 10.0):
        for sgn, nm in ((1, "L"), (-1, "R")):
            # a left turn at w deg/s: the camera at time t_k - b pointed w * b to the right of now -> virtual yaw -w * b (left-positive)
            V[f"g{w:g}{nm}"] = [(arm, int(j), round(-sgn * w * b, 1)) for j, b in zip(jw, back)]
    for sgn, nm in ((1, "L"), (-1, "R")):
        static = [(arm, k, round(-sgn * 1.0, 1))] * 100
        ramp = [(arm, k, round(-sgn * (1.0 - (i + 1) / 20), 1)) for i in range(20)]
        V[f"launch{nm}"] = static + ramp
    return V


def heading3(raw, sl):
    plan = raw[sl["plan"]:sl["plan"] + 990][:495].reshape(33, 15)
    t = 10 * (np.arange(33) / 32) ** 2
    return float(-np.degrees(np.interp(3.0, t, plan[:, 11])))       # left-positive


def cmd_probe(a):
    P = json.loads((OUT.parent / "fixes.json").read_text())
    m = model()
    keep, hs = keep_layout(m)
    sl = {q: s.start for q, s in m.slices.items()}
    for s in a.scenes or scenes():
        f = OUT / s / "probe.npz"
        if f.exists():
            continue
        T, v = meta(s)
        ks = list(range(18, len(T), 12))
        names = None
        R = {}
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=_winit, initargs=(s, P)) as ex:
            for arm in PROBE_ARMS:
                rows = []
                for k in ks:
                    V = probe_jobs(arm, T, k)
                    names = list(V)
                    uniq = sorted({j for steps in V.values() for j in steps})
                    pk = dict(zip(uniq, ex.map(_pack, uniq, chunksize=4)))
                    row = []
                    for nm in names:
                        m.reset()
                        for j in V[nm]:
                            raw = m.step(pk[j])
                        row.append(heading3(raw, sl))
                    rows.append(row)
                R[arm] = np.array(rows, np.float32)
        np.savez(f, **R, k=np.array(ks), v=v[ks], names=np.array(names))
        print(s, "probe", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "stream", "probe"])
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--env", action="store_true", help="stream: the closed-loop-rig render (render_env.npy) only -> stream_env.npz")
    a = ap.parse_args()
    {"fit": cmd_fit, "stream": cmd_stream, "probe": cmd_probe}[a.cmd](a)
