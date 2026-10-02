#!/usr/bin/env python
"""Export recorded HUGSIM episodes for the visual review page. No inference, no rendering, CPU only.

Reads, per case, what the recorded run left on disk: video.mp4 (2x3 camera grid, one frame per 0.25 s step; the
CAM_FRONT cell is cut out), infos.pkl (ego pose, speed, actor boxes per step), eval.json (HD-Score terms),
sim.log (end reason), ground.ply / scene.ply (the simulator's ground and non-ground scene points, rasterised into a
top-down map) and the scene's ground_param.pkl (recorded route). Writes <out>/<case>/f_XXX.jpg and <out>/<case>.json.

    CUDA_VISIBLE_DEVICES= $DATA_DIR/envs/hugsim/bin/python experiments/hugsim/scripts/export_review_cases.py
"""
import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
R = D / "runs"
O = R / "openloop_visual_review_20261002" / "hugsim"
EX, BASE, VAL = R / "hugsim-exam/scored-op", R / "hugsim-exam/scored-base", R / "infra-accept/hugsim-val"
END = [("Collision with background", "bg_collision"), ("Collision with foreground", "fg_collision"),
       ("Far from preset trajectory", "off_route"), ("Complete", "complete")]
CELL, N_FRAMES = 0.25, 28


def rd(tag, scenario, sc):
    _, mode, k = scenario.rsplit("-", 2)
    root, ad = (VAL, "pre") if tag.startswith("preset") else (BASE, tag.split("-")[0]) if tag[:3] in ("ltf", "cv-") \
        else (EX, "zs")
    return root / tag / ad / f"{sc}_{mode}_{k}"


# (case id, dataset, scenario, scene directory, primary tag, overlay tags); the primary run supplies the frames
CASES = [
    ("hugsim-01", "nuscenes", "scene-0166-easy-00", "scene-0166", "cinque-fixed", ["cinque-official"]),
    ("hugsim-02", "nuscenes", "scene-0166-easy-00", "scene-0166", "cinque-official", ["cinque-fixed"]),
    ("hugsim-03", "nuscenes", "scene-0013-medium-00", "scene-0013", "cinque-fixed", ["ltf-fixed"]),
    ("hugsim-04", "nuscenes", "scene-0411-medium-00", "scene-0411", "cinque-fixed", ["ltf-fixed"]),
    ("hugsim-05", "waymo", "scene-113792265837-easy-00", "113792265837_0_200", "cinque-fixed", ["ltf-fixed"]),
    ("hugsim-06", "waymo", "scene-100613054308-hard-00", "100613054308_0_200", "cinque-fixed", ["ltf-fixed"]),
    ("hugsim-07", "nuscenes", "scene-0138-extreme-00", "scene-0138", "cinque-fixed", ["ltf-fixed"]),
    ("hugsim-08", "waymo", "scene-144248042870-extreme-00", "144248042870_0_200", "lebowski-fixed", ["cinque-fixed"]),
    ("hugsim-09", "kitti360", "scene-570_770-easy-00", "0000_570_770", "preset-official", ["preset-fixed2", "preset-ideal"]),
    ("hugsim-10", "kitti360", "scene-570_770-easy-00", "0000_570_770", "preset-fixed2", ["preset-official", "preset-ideal"]),
]


def route(ds, scene):
    p = D / "datasets/hugsim/scenes" / ds / scene / "ground_param.pkl"
    if p.exists():
        cam = pickle.load(open(p, "rb"))[0]
    else:
        with zipfile.ZipFile(D / "datasets/hugsim/scenes" / ds / f"{scene}.zip") as z:
            name = next(n for n in z.namelist() if n.endswith("ground_param.pkl"))
            cam = pickle.load(io.BytesIO(z.read(name)))[0]
    return np.asarray(cam)[:, :3, 3]            # camera x (right), y (down), z (forward)


def run(d):
    infos = pickle.load(open(d / "infos.pkl", "rb"))
    ev = json.load(open(d / "eval.json"))
    txt = (d / "sim.log").read_text(errors="replace")
    end = next((e for k, e in END if k in txt), "max_steps" if len(infos) >= 400 else "other")
    det = [ev["details"][k] for k in sorted(ev["details"], key=float)]
    return {"run_dir": str(d), "end": end, "n_steps": len(infos),
            "eval": {k: v for k, v in ev.items() if k != "details"},
            "pos": [[round(float(i["ego_pos"][0]), 3), round(float(i["ego_pos"][2]), 3)] for i in infos],
            "yaw_box": [round(float(i["ego_box"][6]), 4) for i in infos],
            "v": [round(float(np.ravel(i["ego_velo"])[0]), 3) for i in infos],
            "cmd": [int(i["command"]) for i in infos],
            "objs": [[[round(float(np.ravel(x)[0]), 3) for x in b] for b in i["obj_boxes"]] for i in infos],
            "terms": {k: [float(s[k]) for s in det] for k in ("nc", "dac", "ttc", "c", "pdms")},
            "terms_t": [float(k) for k in sorted(ev["details"], key=float)]}, infos


def topdown(d, rt, paths):
    """Counts of ground / scene points per 0.25 m cell; scene points only within the ego box height band."""
    allp = np.concatenate([rt[:, [0, 2]]] + paths)
    lo, hi = allp.min(0) - 25, allp.max(0) + 25
    shape = np.ceil((hi - lo) / CELL).astype(int)
    tree = cKDTree(rt[:, [0, 2]])
    out = {}
    for name in ("ground", "scene"):
        p = np.asarray(o3d.io.read_point_cloud(str(d / f"{name}.ply")).points)
        p = p[((p[:, [0, 2]] >= lo) & (p[:, [0, 2]] < hi)).all(1)]
        if name == "scene":                     # ego box reaches from the camera 1.5 m down (y is down)
            ycam = rt[tree.query(p[:, [0, 2]], workers=8)[1], 1]
            p = p[(p[:, 1] > ycam) & (p[:, 1] < ycam + 1.5)]
        ij = ((p[:, [0, 2]] - lo) / CELL).astype(int)
        h = np.zeros(shape, np.uint16)
        np.add.at(h, (ij[:, 0], ij[:, 1]), 1)
        out[name] = h
    return lo, out


def main():
    cv2.setNumThreads(8)
    O.mkdir(parents=True, exist_ok=True)
    for cid, ds, scenario, scene, tag, others in CASES:
        rt = route(ds, scene)
        d = rd(tag, scenario, scene)
        runs = {}
        runs[tag], infos = run(d)
        for t in others:
            runs[t], _ = run(rd(t, scenario, scene))
        lo, maps = topdown(d, rt, [np.asarray(r["pos"]) for r in runs.values()])
        (O / cid).mkdir(exist_ok=True)
        # uint8 PNG: ground = 255 where any ground point; scene = clipped count (>= 4 points per cell draws as obstacle)
        cv2.imwrite(str(O / cid / "ground.png"), ((maps["ground"] > 0) * 255).astype(np.uint8))
        cv2.imwrite(str(O / cid / "scene.png"), np.clip(maps["scene"], 0, 255).astype(np.uint8))
        cap = cv2.VideoCapture(str(d / "video.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last = min(n, len(infos)) - 1
        idx = sorted(set(np.linspace(0, last, min(N_FRAMES, last + 1)).round().astype(int).tolist()))
        frames = []
        for k in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = cap.read()
            if not ok:
                continue
            h, w = f.shape[0] // 2, f.shape[1] // 3
            cv2.imwrite(str(O / cid / f"f_{k:03d}.jpg"), cv2.resize(f[:h, w:2 * w], (480, 270), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            frames.append(k)
        json.dump({"id": cid, "dataset": ds, "scenario": scenario, "scene": scene, "primary": tag, "runs": runs,
                   "route": np.round(rt[:, [0, 2]], 3).tolist(), "map_origin": lo.round(3).tolist(), "cell": CELL,
                   "video_frames": n, "frames": frames, "video": str(d / "video.mp4")}, open(O / f"{cid}.json", "w"))
        print(cid, scenario, tag, runs[tag]["end"], runs[tag]["n_steps"], "video frames", n, "exported", len(frames), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
