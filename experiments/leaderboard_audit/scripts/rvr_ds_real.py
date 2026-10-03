"""Real vs render, HUGSIM Waymo / PandaSet / KITTI-360 scenes: the real camera frames behind each reconstructed scene, in HUGSIM's frame order,
resized to the training resolution of the scene (meta_data width x height; INTER_AREA). Same protocol as rvr_hugsim_real.py (nuScenes).
Cameras (HUGSIM names): waymo cam_1 = FRONT, cam_2 = FRONT_LEFT, cam_3 = FRONT_RIGHT (from the recorded relative poses); pandaset front_camera,
front_left_camera, front_right_camera; kitti360 cam_0 only (cam_1 is the parallel stereo partner, cam_2/3 are side fisheyes).
Frame i of a scene = Waymo tfrecord frame i / PandaSet NN = i / KITTI-360 frame (start + i).
Run with $DATA_DIR/envs/waymo/bin/python (opencv-python-headless, numpy, protobuf):
  rvr_ds_real.py <dataset> [check|build] [scenes ...]
Writes $DATA_DIR/runs/real_vs_render/<dataset>/<scene>/real.npy (n, ncam, H, W, 3) uint8."""
import json, os, struct, sys
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
HS = D / "datasets/hugsim/scenes"
RAW = D / "datasets/rvr_real"
CAMS = {"waymo": ("cam_1", "cam_2", "cam_3"), "pandaset": ("front_camera", "front_left_camera", "front_right_camera"), "kitti360": ("cam_0",)}
WAYMO_NAME = {"cam_1": 1, "cam_2": 2, "cam_3": 3}                    # dataset_pb2.CameraName FRONT, FRONT_LEFT, FRONT_RIGHT


def scenes(ds):
    return sorted(p.name for p in (HS / ds).iterdir() if p.is_dir() and p.name != ds)


def meta_wh(ds, scene):
    m = json.load(open(HS / ds / scene / "meta_data.json"))
    out = {}
    for f in m["frames"]:
        c = f["rgb_path"].split("/")[-2]
        out.setdefault(c, []).append((f["width"], f["height"]))
    return {c: (v[0], len(v)) for c, v in out.items()}


def resize(im_bgr, wh):
    import cv2
    return cv2.resize(im_bgr[..., ::-1], wh, interpolation=cv2.INTER_AREA)


def tfrecord(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                return
            n = struct.unpack("<Q", h[:8])[0]
            data = f.read(n)
            f.read(4)
            yield data


def waymo_frames(scene, n_frames, whs):
    import cv2
    sys.path.insert(0, str(D / "envs/waymo/gen"))
    from waymo_open_dataset import dataset_pb2
    seg = json.load(open(RAW / "waymo/segments.json"))[scene.split("_")[0]].rsplit("/", 1)[1]
    arr = np.zeros((n_frames, 3, whs[1], whs[0], 3), np.uint8)
    for i, rec in zip(range(n_frames), tfrecord(RAW / "waymo" / seg)):
        fr = dataset_pb2.Frame()
        fr.ParseFromString(rec)
        im = {x.name: x.image for x in fr.images}
        for j, c in enumerate(CAMS["waymo"]):
            arr[i, j] = resize(cv2.imdecode(np.frombuffer(im[WAYMO_NAME[c]], np.uint8), cv2.IMREAD_COLOR), whs)
    return arr


def panda_frames(scene, n_frames, whs):
    import cv2
    arr = np.zeros((n_frames, 3, whs[1], whs[0], 3), np.uint8)
    for j, c in enumerate(CAMS["pandaset"]):
        for i in range(n_frames):
            arr[i, j] = resize(cv2.imread(str(RAW / "pandaset" / scene / c / f"{i:02d}.jpg")), whs)
    return arr


def kitti_frames(scene, n_frames, whs):
    import cv2
    a = int(scene.split("_")[1])
    arr = np.zeros((n_frames, 1, whs[1], whs[0], 3), np.uint8)
    for i in range(n_frames):
        arr[i, 0] = resize(cv2.imread(str(RAW / "kitti360/cam_0" / f"{a + i:010d}.png")), whs)
    return arr


def build_one(ds, scene):
    out = D / "runs/real_vs_render" / ds / scene
    if (out / "real.npy").exists():
        return None
    mw = meta_wh(ds, scene)
    c0 = CAMS[ds][0]
    whs, n = mw[c0]
    arr = {"waymo": waymo_frames, "pandaset": panda_frames, "kitti360": kitti_frames}[ds](scene, n, whs)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "real.npy.tmp.npy", arr)
    os.replace(out / "real.npy.tmp.npy", out / "real.npy")
    return arr.shape


if __name__ == "__main__":
    ds, cmd, names = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "build", sys.argv[3:]
    for s in names or scenes(ds):
        try:
            print(ds, s, build_one(ds, s), flush=True)
        except FileNotFoundError as e:
            print(ds, s, "MISSING", e, flush=True)
