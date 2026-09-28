"""op-adapt data (todos/2026-09-28-op-adapt.md): nuScenes pedestrian-corridor labels and per-scene trunk caches.

  labels   every trainval keyframe (processed/nusc_zs/trainval_index.pkl): GT pedestrians / cyclists with at least one
           lidar point, the corridor of decisions 44 / 53 (logged future path from the origin, extended along its last
           heading to 30 m; box centre or any corner with |d| <= 1.5 m and 0 < s <= 30 m), a wide corridor (4 m, 40 m)
           and the distance of the nearest corridor pedestrian -> processed/op_adapt/nusc_labels.parquet
  cache    per scene, openpilot's nuScenes protocol (scripts/nusc_zs_openpilot.py: 20 Hz clock, 10 steps per
           keyframe interval, each step fed the latest CAM_FRONT frame, road + wide rendered from CAM_FRONT): the
           stage-3 output (`permute_73`, fp16 (1024, 8, 16)) of every even step, i.e. of the image pair
           (frame at step s-4, frame at step s). Every even step s >= 0 is a sample whose 9-frame context is the
           even steps s, s-4, .., s-32 (zero hidden before the start) -> processed/op_adapt/nusc/<scene>.npz
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import data_dir, dataroot

STEPS = 10                                     # 20 Hz steps per 0.5 s keyframe interval
HALF_W, REACH, WIDE_W, WIDE_R = 1.5, 30.0, 4.0, 40.0


def root(*parts) -> Path:
    d = data_dir() / "processed" / "op_adapt" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- labels
def _path(fut: np.ndarray, reach: float, ds: float = 0.25) -> np.ndarray:
    """Origin + future points, extended along the last heading to `reach` m, resampled every ds m."""
    p = np.r_[[[0.0, 0.0]], fut[np.isfinite(fut).all(1)][:, :2]]
    seg = np.diff(p, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    good = ln > 0.05
    h = seg[good][-1] / ln[good][-1] if good.any() else np.array([1.0, 0.0])
    p, ln = np.r_[p[:1], p[1:][good]], ln[good]
    left = reach - ln.sum()
    if left > 0:
        p, ln = np.r_[p, [p[-1] + h * left]], np.r_[ln, left]
    s = np.r_[0.0, np.cumsum(ln)]
    q = np.arange(0, reach + ds / 2, ds)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)


def corridor(fut: np.ndarray, xy: np.ndarray, yaw: np.ndarray, lw: np.ndarray, half_w: float, reach: float) -> np.ndarray:
    """(n,) bool: box centre or a corner within |d| <= half_w of the path and 0 < s <= reach (s by nearest path point)."""
    if not len(xy):
        return np.zeros(0, bool)
    path = _path(fut, reach)
    c, s_ = np.cos(yaw), np.sin(yaw)
    pts = [xy] + [np.c_[xy[:, 0] + a * lw[:, 0] / 2 * c - b * lw[:, 1] / 2 * s_,
                        xy[:, 1] + a * lw[:, 0] / 2 * s_ + b * lw[:, 1] / 2 * c] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    P = np.concatenate(pts)
    d2 = ((P[:, None, :] - path[None]) ** 2).sum(-1)
    i = d2.argmin(1)
    ok = (np.sqrt(d2[np.arange(len(P)), i]) <= half_w) & (i > 0)
    return ok.reshape(5, -1).any(0)


def labels() -> "pd.DataFrame":
    import pandas as pd
    from . import nuscenes_zs as Z
    idx = Z.load_index("trainval")
    tab = lambda n: json.loads((dataroot() / "v1.0-trainval" / f"{n}.json").read_text())  # noqa: E731
    cat = {c["token"]: c["name"] for c in tab("category")}
    inst = {i["token"]: cat[i["category_token"]] for i in tab("instance")}
    attr = {a["token"]: a["name"] for a in tab("attribute")}
    by_s = defaultdict(list)
    for a in tab("sample_annotation"):
        c = inst[a["instance_token"]]
        ped = c.startswith("human.pedestrian.")
        cyc = c in ("vehicle.bicycle", "vehicle.motorcycle") and any(attr[t] == "cycle.with_rider" for t in a["attribute_tokens"])
        if (ped or cyc) and a["num_lidar_pts"] >= 1:
            by_s[a["sample_token"]].append((ped, a["translation"], a["rotation"], a["size"]))
    rows = []
    for e in idx["samples"]:
        sc = idx["scenes"][e["scene"]]
        k = int(np.abs(sc["pose_t"] - e["t0"]).argmin())
        R, t = sc["pose_R"][k], sc["pose_xyz"][k]
        yaw_e = np.arctan2(R[1, 0], R[0, 0])
        a = by_s.get(e["token"], [])
        g_r = e.get("gt_rear")
        fut = np.asarray(g_r, np.float64).reshape(-1, 2) if g_r is not None and len(g_r) else np.zeros((0, 2))
        r = {"token": e["token"], "scene": e["scene"], "split": e["split"], "t0": e["t0"], "valid": bool(e.get("valid", False)),
             "n_vru": len(a), "ped_corr": False, "ped_wide": False, "vru_corr": False, "vru_wide": False, "ped_dist": np.nan}
        if a:
            ped = np.array([x[0] for x in a])
            g = np.array([x[1] for x in a])
            xy = ((g - t) @ R)[:, :2]
            q = np.array([x[2] for x in a])
            yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)) - yaw_e
            lw = np.array([[x[3][1], x[3][0]] for x in a])            # nuScenes size = (w, l, h)
            cn = corridor(fut, xy, yaw, lw, HALF_W, REACH)
            wd = corridor(fut, xy, yaw, lw, WIDE_W, WIDE_R)
            r.update(ped_corr=bool((cn & ped).any()), ped_wide=bool((wd & ped).any()), vru_corr=bool(cn.any()),
                     vru_wide=bool(wd.any()),
                     ped_dist=float(np.hypot(*xy[cn & ped].T).min()) if (cn & ped).any() else np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(root() / "nusc_labels.parquet", index=False)
    return df


# ---------------------------------------------------------------- per-scene step plan and rendering
def scene_plan(idx: dict, name: str) -> dict:
    """scripts/nusc_zs_openpilot.scene_plan on the trainval index: step times, CAM_FRONT frame per step, keyframes."""
    from . import nuscenes_zs as Z
    sc = idx["scenes"][name]
    keys = [e for e in idx["samples"] if e["scene"] == name]
    kt = np.array([e["t0"] for e in keys], np.float64)
    t = [kt[0]] + [kt[k - 1] + j * (kt[k] - kt[k - 1]) / STEPS for k in range(1, len(kt)) for j in range(1, STEPS + 1)]
    fr = [Z.latest_frame(sc, "CAM_FRONT", x + 1) for x in t]
    return {"t": np.array(t), "frame": np.array(fr), "key_step": np.arange(len(kt)) * STEPS,
            "tokens": [e["token"] for e in keys],
            "traffic": (0.0, 1.0) if sc["location"].startswith("singapore") else (1.0, 0.0)}


_ctx = {}


def _init():
    from . import nuscenes_zs as Z
    _ctx["idx"] = Z.load_index("trainval")


def render_scene(name: str):
    """Worker: packed (n_unique, 2, 6, 128, 256) frames of one scene, and the step -> unique-row map."""
    import sys
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import nusc_zs_openpilot as NZ
    idx = _ctx["idx"]
    plan = scene_plan(idx, name)
    sc = idx["scenes"][name]
    ix, cov = NZ.op_index(sc)
    uniq, inv = np.unique(plan["frame"], return_inverse=True)
    out = np.empty((len(uniq), 2, 6, 128, 256), np.uint8)
    for r, f in enumerate(uniq):
        im = Image.open(dataroot() / sc["cams"]["CAM_FRONT"]["path"][f])
        im.draft("YCbCr", im.size)
        ycc = np.asarray(im.convert("YCbCr"))
        out[r, 0], out[r, 1] = NZ.pack(ycc, ix[0]), NZ.pack(ycc, ix[1])
    return name, plan, out, inv, cov
