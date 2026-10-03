"""Real vs render for HUGSIM Waymo / PandaSet / KITTI-360 scenes: openpilot (Cinque) on the paired real / rendered sequences through the
closed-loop adapter (hugsim_zs.OpenpilotFrames). The same protocol as rvr_hugsim_op.py (nuScenes), unchanged except where the dataset forces it:
  - the source rate is 10 Hz (nuScenes 12 Hz): frames >= 3 s = frame 30; probe frames every 10 frames (1 s) from frame 15 (1.5 s)
  - camera calibration: the recorded training intrinsics and relative camera poses of the scene (meta_data.json), with the dataset's cam_rect
    from configs/sim/<ds>_camera.yaml (nuScenes: the sim's yaml intrinsics equal the training ones, here they do not)
  - cameras: waymo cam_1/2/3 -> CAM_FRONT/FRONT_LEFT/FRONT_RIGHT, pandaset front/front_left/front_right, kitti360 front only
  - arms: real and render only (no image-side fixes, no closed-loop-rig arm)
envs/openpilot on a leased card:
  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/openpilot/bin/python experiments/leaderboard_audit/scripts/rvr_ds_op.py <dataset> stream|probe [scenes ...]
Reads $DATA_DIR/runs/real_vs_render/<ds>/<scene>/{real,render}.npy, writes stream.npz / probe.npz there."""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).resolve().parent)]
import rvr_hugsim_op as H  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402

D = H.D
HS = D / "datasets/hugsim/scenes"
RUN = D / "runs/real_vs_render"
CAMS = {"waymo": ("cam_1", "cam_2", "cam_3"), "pandaset": ("front_camera", "front_left_camera", "front_right_camera"), "kitti360": ("cam_0",)}
MCAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
DT = H.DT
SRC_DT = 0.1
ARMS = ("real", "render")


def scenes(ds):
    return sorted(p.name for p in (RUN / ds).iterdir() if (p / "render.npy").exists() and (p / "real.npy").exists())


def meta(ds, scene):
    """(T seconds, speed m/s, frames of the first camera, by camera) of a scene."""
    m = json.load(open(HS / ds / scene / "meta_data.json"))
    by = {c: [f for f in m["frames"] if f"/{c}/" in f["rgb_path"]] for c in CAMS[ds]}
    fr = by[CAMS[ds][0]]
    ts = np.array([f["timestamp"] for f in fr], float)
    T = np.arange(len(fr)) * SRC_DT if ds != "pandaset" else ts - ts[0]
    P = np.array([np.asarray(f["camtoworld"])[:3, 3] for f in fr])
    d = np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1)
    v = np.r_[d[0], d] / np.r_[T[1] - T[0], np.diff(T)]
    v = np.convolve(np.pad(v, (3, 2), mode="edge"), np.ones(6) / 6, "valid")      # 0.6 s box filter
    return T, v, by


def cam_params(ds, scene):
    """Recorded (training) intrinsics and camera rig of frame 0 in the adapter's cam_params layout. The front camera is the ego frame:
    v2c_front = I, v2c_cam = inv(c2w_cam) @ c2w_front, so that E = v2front @ inv(v2c_cam) = c2w_front^-1 c2w_cam (hugsim_zs.calibs)."""
    _, _, by = meta(ds, scene)
    out = {}
    c0 = np.asarray(by[CAMS[ds][0]][0]["camtoworld"], float)
    for c, mc in zip(CAMS[ds], MCAMS):
        f = by[c][0]
        K = np.asarray(f["intrinsics"], float)
        W, Hh = f["width"], f["height"]
        out[mc] = {"intrinsic": dict(W=W, H=Hh, cx=K[0, 2], cy=K[1, 2], fovx=2 * np.arctan(W / (2 * K[0, 0])), fovy=2 * np.arctan(Hh / (2 * K[1, 1]))),
                   "v2c": np.linalg.inv(np.asarray(f["camtoworld"], float)) @ c0}
    return out


def adapter(ds, scene):
    cal = Z.calibs(cam_params(ds, scene), Z.rect_matrix(str(D / f"third_party/HUGSIM/configs/sim/{ds}_camera.yaml")), MCAMS[:len(CAMS[ds])])
    return Z.OpenpilotFrames(cal, MCAMS[:len(CAMS[ds])])


_W = {}


def _winit(ds, scene):
    _W.update(op=adapter(ds, scene), A={s: np.load(RUN / ds / scene / f"{s}.npy", mmap_mode="r") for s in ARMS}, n=len(CAMS[ds]))


def _pack(job):
    arm, k, yaw = job
    op = _W["op"]
    rgb = {c: np.asarray(_W["A"][arm][k, j]) for j, c in enumerate(MCAMS[:_W["n"]])}
    return op.pack(rgb, op.rot_index(yaw) if abs(yaw) >= 0.05 else None)


def cmd_stream(a):
    m = H.model()
    keep, hs = H.keep_layout(m)
    for s in a.scenes or scenes(a.ds):
        f = RUN / a.ds / s / "stream.npz"
        if f.exists():
            continue
        T, v, _ = meta(a.ds, s)
        n = len(T)
        steps = np.arange(int(T[-1] / DT) + 1) * DT
        kk = np.searchsorted(T, steps + 1e-6, side="right") - 1
        first = np.r_[True, kk[1:] != kk[:-1]]
        Hd = {}
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=_winit, initargs=(a.ds, s)) as ex:
            for arm in ARMS:
                packed = list(ex.map(_pack, [(arm, k, 0.0) for k in range(n)], chunksize=8))
                m.reset()
                out = np.full((n, len(keep)), np.nan, np.float32)
                for j, k in enumerate(kk):
                    raw = m.step(packed[k])
                    if first[j]:
                        out[k] = raw[keep]
                Hd[arm] = out
        np.savez(f, **Hd, T=T, v=v, info=json.dumps(dict(heads_slices=hs, arms=ARMS, coverage=_cov(a.ds, s))))
        print(a.ds, s, "stream", f"{time.time() - t0:.0f}s", flush=True)


def _cov(ds, s):
    op = adapter(ds, s)
    return {"coverage": op.coverage, "src_frac": op.src_frac}


def cmd_probe(a):
    m = H.model()
    keep, hs = H.keep_layout(m)
    sl = {q: s.start for q, s in m.slices.items()}
    for s in a.scenes or scenes(a.ds):
        f = RUN / a.ds / s / "probe.npz"
        if f.exists():
            continue
        T, v, _ = meta(a.ds, s)
        ks = list(range(15, len(T), 10))
        names, R = None, {}
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=_winit, initargs=(a.ds, s)) as ex:
            for arm in ARMS:
                rows = []
                for k in ks:
                    V = H.probe_jobs(arm, T, k)
                    names = list(V)
                    uniq = sorted({j for steps in V.values() for j in steps})
                    pk = dict(zip(uniq, ex.map(_pack, uniq, chunksize=4)))
                    row = []
                    for nm in names:
                        m.reset()
                        for j in V[nm]:
                            raw = m.step(pk[j])
                        row.append(H.heading3(raw, sl))
                    rows.append(row)
                R[arm] = np.array(rows, np.float32)
        np.savez(f, **R, k=np.array(ks), v=v[ks], names=np.array(names))
        print(a.ds, s, "probe", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", choices=list(CAMS))
    ap.add_argument("cmd", choices=["stream", "probe"])
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"stream": cmd_stream, "probe": cmd_probe}[a.cmd](a)
