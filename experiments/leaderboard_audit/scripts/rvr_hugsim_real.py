"""Real vs render, HUGSIM nuScenes scenes: the real camera frames behind each reconstructed scene, in HUGSIM's own frame order.
HUGSIM's nuScenes scene `scene-XXXX` holds 180 frames per camera at 12 Hz from the scene's first key frame (source_path
scene-XXXX_0_180); frame i = the i-th CAM_* image (samples + sweeps, by timestamp) at or after the first sample. The rule is checked
against the one scene whose reconstruction inputs ship with HUGSIM (sample_data scene-0383) by PSNR over offsets -2..2.
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
    idx, lst = index(), listing()
    log, t0 = idx["scene-0383"]
    ref = D / "datasets/hugsim/sample_data/data/scene-0383/images/CAM_FRONT"
    for off in (-2, -1, 0, 1, 2):
        fs = files(log, t0, "CAM_FRONT", lst, off)
        ps = [psnr(load(fs[i]), cv2.imread(str(ref / f"{i:05d}.jpg"))[..., ::-1]) for i in (0, 50, 120, 179)]
        print("offset", off, np.round(ps, 2))


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
    idx, lst = index(), listing()
    jobs = []
    for n, (log, t0) in idx.items():
        fl = {c: files(log, t0, c, lst) for c in CAMS}
        if min(len(v) for v in fl.values()) < N:
            print("short", n, {c: len(v) for c, v in fl.items()})
            continue
        jobs.append((n, log, t0, fl))
    with ProcessPoolExecutor(20) as ex:
        for n in ex.map(build_one, jobs):
            print("done", n, flush=True)


if __name__ == "__main__":
    {"check": check, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
