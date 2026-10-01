"""op-adapt data (fc65452:todos/2026-09-28-op-adapt.md): nuScenes pedestrian-corridor labels and per-scene trunk caches.

  labels   every trainval keyframe (processed/nusc_zs/trainval_index.pkl): GT pedestrians / cyclists with at least one
           lidar point, the corridor of decisions 44 / 53 (logged future path from the origin, extended along its last
           heading to 30 m; box centre or any corner with |d| <= 1.5 m and 0 < s <= 30 m), a wide corridor (4 m, 40 m)
           and the distance of the nearest corridor pedestrian -> processed/op_adapt/nusc_labels.parquet
  labels_wod / labels_nav   the real-data half of the next (sim + real) B round, same corridor definitions:
           WOD train (10 Hz streams of processed/drive_backbones/op_plan_trainval.json, train sequences only): WOD-E2E
           has no 3D boxes, so the labels come from the YOLO26x detections of real_transfer/yolo/wod_train (score >
           0.25, flat-ground lift gx, gy in the rear-axle frame, a 0.7 x 0.7 m footprint around the ground point) and
           the logged 5 s future (waymo_e2e/future.npy); a pedestrian / cyclist that the lift could not place makes
           the frame `uncertain`. Samples are the even frame numbers (context stride 2 = 5 Hz; 82 % of the YOLO frames are even), labels where YOLO ran.
           NAVSIM navtrain: GT boxes of the e3 agent cache and navtrain_future (decisions 44 / 53), one sample per token
           on the NAVSIM 2 Hz sample-and-hold protocol (scripts/navsim_zs_openpilot.py; decision 36: off-protocol for
           openpilot, kept as its own dataset tag).
           dist_bin: nearest corridor pedestrian (else wide-corridor pedestrian) in 0-10 / 10-20 / 20-30 / 30+ m.
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
             "n_vru": len(a), "ped_corr": False, "ped_wide": False, "vru_corr": False, "vru_wide": False, "ped_dist": np.nan,
             "ped_dist_wide": np.nan}
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
                     ped_dist=float(np.hypot(*xy[cn & ped].T).min()) if (cn & ped).any() else np.nan,
                     ped_dist_wide=float(np.hypot(*xy[wd & ped].T).min()) if (wd & ped).any() else np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(root() / "nusc_labels.parquet", index=False)
    return df


DIST_BINS = (0.0, 10.0, 20.0, 30.0, np.inf)
PED_LW = (0.7, 0.7)            # footprint put around a detection's ground point (WOD, no box)


def dist_bin(df):
    """-1 = no pedestrian in the wide corridor; else the bin of the nearest corridor (else wide) pedestrian."""
    d = df.ped_dist.fillna(df.ped_dist_wide).to_numpy()
    b = np.digitize(d, DIST_BINS[1:-1])
    return np.where(np.isfinite(d), b, -1)


def _flags(fut, xy, yaw, lw, ped):
    cn = corridor(fut, xy, yaw, lw, HALF_W, REACH)
    wd = corridor(fut, xy, yaw, lw, WIDE_W, WIDE_R)
    dist = np.hypot(*xy.T) if len(xy) else np.zeros(0)
    return {"ped_corr": bool((cn & ped).any()), "ped_wide": bool((wd & ped).any()), "vru_corr": bool(cn.any()),
            "vru_wide": bool(wd.any()), "ped_dist": float(dist[cn & ped].min()) if (cn & ped).any() else np.nan,
            "ped_dist_wide": float(dist[wd & ped].min()) if (wd & ped).any() else np.nan}


def _wod_worker(job):
    rows = []
    for fid, fut, xy, ped, unl in job:
        r = {"frame_id": fid, **_flags(fut, xy, np.zeros(len(xy)), np.tile(PED_LW, (len(xy), 1)), ped),
             "uncertain": bool(unl)}
        rows.append(r)
    return rows


def labels_wod(workers: int = 16, limit_seq: int = 0):
    """processed/op_adapt/wod_train_labels.parquet (one row per YOLO frame on an even stream offset) and
    wod_train_plan.json (streams: key, names = the even offsets, targets = positions of the labelled names)."""
    import pandas as pd
    from multiprocessing import Pool
    from . import drive_backbones as DB
    y = data_dir() / "processed/real_transfer/yolo/wod_train"
    fr = pd.read_parquet(y / "frames.parquet")
    seqs = sorted(fr.sequence.unique())[: limit_seq or None]
    fr = fr[fr.sequence.isin(seqs)]
    plan = json.loads((DB.root() / DB.plan_name("trainval")).read_text())
    streams, keep = [], set()
    for i, st in enumerate(plan["streams"]):
        if st["sequence"] not in set(seqs):
            continue
        names = [n for n in st["names"] if int(n.rsplit("-", 1)[1]) % 2 == 0]    # YOLO ran mostly on multiples of 4
        streams.append({"key": f"{i:04d}_{st['sequence']}", "names": names})
        keep |= set(names)
    fr = fr[fr.frame_id.isin(keep)]
    d = pd.read_parquet(y / "dets.parquet", columns=["frame_id", "prompt", "score", "gx", "gy", "lift_ok"])
    d = d[d.prompt.isin(["pedestrian", "cyclist"]) & d.frame_id.isin(set(fr.frame_id))]
    fut = np.load(data_dir() / "processed/waymo_e2e/future.npy", mmap_mode="r")
    by = d.groupby("frame_id")
    jobs, chunk = [], []
    for fid, row in zip(fr.frame_id, fr.row):
        g = by.get_group(fid) if fid in by.groups else d.iloc[:0]
        ok = g.lift_ok.to_numpy(bool)
        chunk.append((fid, np.asarray(fut[row][:, :2], np.float64), g[["gx", "gy"]].to_numpy(float)[ok],
                      (g.prompt.to_numpy() == "pedestrian")[ok], (~ok).any()))
        if len(chunk) == 512:
            jobs.append(chunk)
            chunk = []
    jobs.append(chunk)
    with Pool(workers) as p:
        rows = [r for part in p.imap(_wod_worker, jobs) for r in part]
    lab = pd.DataFrame(rows).merge(fr[["frame_id", "sequence"]], on="frame_id")
    lab["dist_bin"] = dist_bin(lab)
    have = set(lab.frame_id)
    for st in streams:
        st["targets"] = [j for j, n in enumerate(st["names"]) if n in have]
    streams = [st for st in streams if st["targets"]]
    tag = f"_limit{limit_seq}" if limit_seq else ""
    lab.to_parquet(root() / f"wod_train_labels{tag}.parquet", index=False)
    (root() / f"wod_train_plan{tag}.json").write_text(json.dumps({"streams": streams}))
    return lab, streams


def _nav_worker(job):
    return [{"token": tok, **_flags(fut, b[:, :2].astype(np.float64), b[:, 6].astype(np.float64),
                                    b[:, 3:5].astype(np.float64), c == 1)} for tok, fut, b, c in job]


def labels_nav(workers: int = 16, limit: int = 0):
    """processed/op_adapt/navtrain_labels.parquet: one row per navtrain token (GT boxes x, y, z, l, w, h, yaw)."""
    import pickle
    import pandas as pd
    from multiprocessing import Pool
    from . import navsim_zs as Z
    with open(data_dir() / "runs/elicitation/e3-cache/navtrain_agents.pkl", "rb") as f:
        ag = pickle.load(f)
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        toks, fut = f["tokens"].astype(str), f["poses"]
    sel = [i for i, t in enumerate(toks) if t in ag][: limit or None]
    def vru(t):                      # pedestrians (1) and bicycles (2) of the e3 cache classes
        m = np.isin(ag[t]["cls"], (1, 2))
        return ag[t]["boxes"][m], ag[t]["cls"][m]
    jobs = [[(toks[i], fut[i][:, :2].astype(np.float64), *vru(toks[i])) for i in sel[a:a + 512]]
            for a in range(0, len(sel), 512)]
    with Pool(workers) as p:
        rows = [r for part in p.imap(_nav_worker, jobs) for r in part]
    lab = pd.DataFrame(rows)
    lab["map"] = [ag[t]["map"] for t in lab.token]
    lab["dist_bin"] = dist_bin(lab)
    lab.to_parquet(root() / f"navtrain_labels{'_limit%d' % limit if limit else ''}.parquet", index=False)
    return lab


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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("nusc", "wod", "nav"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.what == "nusc":
        df = labels()
    elif a.what == "wod":
        df, st = labels_wod(a.workers)
        print(f"{len(st)} streams, {sum(len(s['names']) for s in st)} slots, uncertain {df.uncertain.mean():.3f}")
    else:
        df = labels_nav(a.workers)
    print(len(df), df[["ped_corr", "ped_wide", "vru_corr", "vru_wide"]].mean().round(4).to_dict(),
          df.dist_bin.value_counts().sort_index().to_dict())
