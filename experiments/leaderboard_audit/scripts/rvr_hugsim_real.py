"""Real vs render, HUGSIM nuScenes scenes: the real camera frames behind each reconstructed scene, in HUGSIM's own frame order.
HUGSIM's nuScenes scene `scene-XXXX` holds 180 frames per camera at nominal 12 Hz (source_path scene-XXXX_0_180), but not every
nuScenes sweep: on scene-0383 (whose reconstruction inputs ship with HUGSIM) frame i is sweep i up to i = 105 and sweep i-1 / i-2 later
(found by image match, PSNR 51 dB). So frames are matched by pose: HUGSIM frame i's global camera position (inv(inv_pose) @ camtoworld)
against every nuScenes sample_data of that camera in the log (ego_pose @ calibrated_sensor), nearest in position within a monotone
time window; `check` compares this with the image match on scene-0383.
Images are the nuScenes JPEGs resized 1600x900 -> 800x450 (INTER_AREA), the size HUGSIM trains and renders at.
Project venv, CPU: .venv/bin/python experiments/leaderboard_audit/scripts/rvr_hugsim_real.py [check|build]
Writes $DATA_DIR/runs/real_vs_render/hugsim/<scene>/real.npy (180, 3, 450, 800, 3) uint8 [FRONT, FRONT_LEFT, FRONT_RIGHT] + files.json."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
NU = D / "datasets/nuscenes"
HS = D / "datasets/hugsim/scenes/nuscenes"
OUT = D / "runs/real_vs_render/hugsim"
CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
N = 180


def scenes():
    return sorted(p.name for p in HS.iterdir() if p.is_dir() and p.name.startswith("scene-"))


def index():
    t = NU / "v1.0-trainval"
    sc = {x["name"]: x for x in json.load(open(t / "scene.json"))}
    lg = {x["token"]: x["logfile"] for x in json.load(open(t / "log.json"))}
    sm = {x["token"]: x["timestamp"] for x in json.load(open(t / "sample.json"))}
    return {n: (lg[sc[n]["log_token"]], sm[sc[n]["first_sample_token"]]) for n in scenes()}


def pose_index(names):
    """{(log, cam): [(timestamp, filename, global camera xyz)]} for the logs of the given scenes (sample_data.json is 1.3 GB)."""
    t = NU / "v1.0-trainval"
    idx = index()
    logs = {idx[n][0] for n in names}
    lg = {x["token"]: x["logfile"] for x in json.load(open(t / "log.json"))}
    cs = {x["token"]: x for x in json.load(open(t / "calibrated_sensor.json"))}
    sd = [x for x in json.load(open(t / "sample_data.json")) if x["fileformat"] == "jpg" and
          any(x["filename"].split("/")[1] == c for c in CAMS) and x["filename"].split("/")[2].split("__")[0] in logs]
    need = {x["ego_pose_token"] for x in sd}
    ep = {x["token"]: x for x in json.load(open(t / "ego_pose.json")) if x["token"] in need}
    from scipy.spatial.transform import Rotation as Rot
    out = {}
    for x in sd:
        c, e = cs[x["calibrated_sensor_token"]], ep[x["ego_pose_token"]]
        Re = Rot.from_quat(np.r_[e["rotation"][1:], e["rotation"][0]]).as_matrix()
        xyz = Re @ np.asarray(c["translation"]) + np.asarray(e["translation"])
        log, cam = x["filename"].split("/")[2].split("__")[0], x["filename"].split("/")[1]
        out.setdefault((log, cam), []).append((x["timestamp"], str(NU / x["filename"]), xyz))
    return {k: sorted(v, key=lambda r: r[0]) for k, v in out.items()}


def match(name, cam, pidx):
    """HUGSIM frame i -> nuScenes file, by nearest global camera position, searched forward from the previous match."""
    m = json.load(open(HS / name / "meta_data.json"))
    to_g = np.linalg.inv(np.asarray(m["inv_pose"]))
    P = np.array([(to_g @ np.asarray(f["camtoworld"]))[:3, 3] for f in m["frames"] if f"/{cam}/" in f["rgb_path"]])
    log, t0 = index_cache()[name]
    rows = pidx[(log, cam)]
    X = np.array([r[2] for r in rows])
    k0 = next(i for i, r in enumerate(rows) if r[0] >= t0 - 40000)
    out, err, prev = [], [], k0 - 3
    for p in P:
        lo = max(0, prev)
        d = np.linalg.norm(X[lo:lo + 12] - p, axis=1)
        j = lo + int(np.argmin(d))
        out.append(rows[j][1]), err.append(float(d.min()))
        prev = j
    return out, np.array(err)


_IDX = {}


def index_cache():
    if not _IDX:
        _IDX.update(index())
    return _IDX


def files(log, t0, cam, listing, offset=0):
    fs = sorted((int(f.rsplit("__", 1)[1][:-4]), d, f) for d, f in listing[cam] if f.startswith(log + "__"))
    k = next(i for i, (ts, _, _) in enumerate(fs) if ts >= t0 - 40000)       # cameras fire within ~40 ms of the sample time
    k = max(0, k + offset)
    return [str(NU / d / cam / f) for _, d, f in fs[k:k + N]]


def load(path):
    import cv2
    im = cv2.imread(path)[..., ::-1]
    return cv2.resize(im, (800, 450), interpolation=cv2.INTER_AREA)


def listing():
    return {c: [(d, f) for d in ("samples", "sweeps") for f in os.listdir(NU / d / c)] for c in CAMS}


def psnr(a, b):
    return float(-10 * np.log10(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) / 255 ** 2))


def check():
    import cv2
    pidx = pose_index(["scene-0383"])
    ref = D / "datasets/hugsim/sample_data/data/scene-0383/images"
    for cam in CAMS:
        fl, err = match("scene-0383", cam, pidx)
        ps = [psnr(load(fl[i]), cv2.imread(str(ref / cam / f"{i:05d}.jpg"))[..., ::-1]) for i in range(0, 180, 6)]
        print(cam, "pose err m median/max %.3f / %.3f" % (np.median(err), err.max()), "PSNR min / median", round(min(ps), 1), round(float(np.median(ps)), 1))


def build_one(args):
    name, log, t0, fl = args
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    if (d / "real.npy").exists():
        return name
    arr = np.lib.format.open_memmap(d / "real.npy.tmp", "w+", np.uint8, (N, 3, 450, 800, 3))
    for c, cam in enumerate(CAMS):
        for i, f in enumerate(fl[cam]):
            arr[i, c] = load(f)
    arr.flush()
    del arr
    os.replace(d / "real.npy.tmp", d / "real.npy")
    (d / "files.json").write_text(json.dumps(fl))
    return name


def build():
    idx = index_cache()
    pidx = pose_index(list(idx))
    jobs = []
    for n, (log, t0) in idx.items():
        fl, errs = {}, {}
        for c in CAMS:
            fl[c], e = match(n, c, pidx)
            errs[c] = [float(np.median(e)), float(e.max())]
        print(n, "pose err median / max", errs, flush=True)
        (OUT / n).mkdir(parents=True, exist_ok=True)
        (OUT / n / "match_err.json").write_text(json.dumps(errs))
        jobs.append((n, log, t0, fl))
    with ProcessPoolExecutor(20) as ex:
        for n in ex.map(build_one, jobs):
            print("done", n, flush=True)


if __name__ == "__main__":
    {"check": check, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
