"""Fusion diagnostics Q4: how much SAM 3.1 sees, and how well its ground-contact points land in BEV.
Pre-registration and the clarifications this module implements: todos/2026-09-25-fusion-diagnostics.md (Q4, and the
[Q4] lines of the deviation log). Detection itself is `jevdrive.sam_detect` (envs/sam3); everything here is CPU.

  lists     image lists for sam_detect: P5 observation frames x 3 cameras, nuScenes val keyframes x 3 front
            cameras, WOD front images (479 rater frames + the p2p3_v1 subset) -> processed/fusion_diag/lists/
  gt        per image: every GT object with its class, hazard flag, BEV reference point (the footprint point nearest
            to the camera, clarification (3)), bottom centre, and its projection into that camera
  lift      detections' ground-contact pixels -> ego BEV on the flat ground z = 0
  match     per (image, class) Hungarian on BEV distance, gate max(2 m, 0.1 x distance)
  report    recall / precision / BEV error by class x distance bin x weather; the hazard reading (i)
  check     the pre-registered 16-frame coordinate check (a)(b) (and (d) on 4 nuScenes frames)

Frames: ego = x forward, y left, z up. P5: origin at the rear axle on the ground (the WOD vehicle frame the P4 / P5
camera rig is written in); nuScenes: its ego frame (rear axle, ground). Camera calibration records are WOD style:
intrinsic [fu fv cu cv k1 k2 p1 p2 k3], extrinsic = vehicle_from_camera with the camera x along the optical axis,
y left, z up; nuScenes cameras are converted into that form (zero distortion).
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
CAMS = ("front", "front_left", "front_right")
REAR_AXLE_X = -1.388633220            # actor-relative rear axle of the hero (scripts/p4_carla_agent.py)
EMERGENCY = ("vehicle.carlamotors.firetruck", "vehicle.ford.ambulance", "vehicle.dodge.charger_police",
             "vehicle.dodge.charger_police_2020")
CLASSES = ("pedestrian", "cyclist", "vehicle", "cone", "debris", "emergency vehicle")
GT_CLASSES_P5 = ("pedestrian", "vehicle", "emergency vehicle")
DIST_BINS = (0, 10, 20, 40, 80)
FACTOR_PX = 20                         # p5_pairs.PX_ACTOR
MAX_LIFT = 80.0
SCORE = 0.5
NUSC_CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
OPENCV_TO_WOD = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])   # columns: OpenCV cam x, y, z in WOD cam axes


def root(*p) -> Path:
    d = data_dir() / "processed" / "fusion_diag" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ================================================================ camera geometry (numpy)

def project(pts: np.ndarray, calib: dict):
    """Ego-frame points (n, 3) -> pixel u, v and validity (in front, inside the image)."""
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = (float(x) for x in calib["intrinsic"])
    E = np.asarray(calib["extrinsic"], np.float64).reshape(4, 4)
    p = (pts - E[:3, 3]) @ E[:3, :3]                        # ego -> camera (x optical, y left, z up)
    fwd = p[:, 0]
    safe = np.where(fwd > 1e-6, fwd, 1.0)
    x, y = -p[:, 1] / safe, -p[:, 2] / safe
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u, v = fu * xd + cu, fv * yd + cv
    ok = (fwd > 0.1) & (u >= -0.5) & (u <= calib["width"] - 0.5) & (v >= -0.5) & (v <= calib["height"] - 0.5)
    return u, v, ok


def lift(u: np.ndarray, v: np.ndarray, calib: dict, ground_z: float = 0.0):
    """Pixels -> ego-frame ground points (n, 2) and validity (ray hits the ground ahead, within MAX_LIFT)."""
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = (float(x) for x in calib["intrinsic"])
    xd, yd = (u - cu) / fu, (v - cv) / fv
    x, y = xd.copy(), yd.copy()
    for _ in range(20):                                      # invert the distortion by fixed-point iteration
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad
        y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad
    E = np.asarray(calib["extrinsic"], np.float64).reshape(4, 4)
    d = np.stack([np.ones_like(x), -x, -y], 1) @ E[:3, :3].T  # camera ray -> ego
    t = E[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (ground_z - t[2]) / d[:, 2]
    g = t[None, :2] + s[:, None] * d[:, :2]
    ok = np.isfinite(s) & (s > 0) & (np.hypot(g[:, 0], g[:, 1]) <= MAX_LIFT) & np.isfinite(u) & np.isfinite(v)
    return g, ok


def nearest_on_box(c: np.ndarray, yaw: np.ndarray, ext: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Footprint rectangle (centre c (n, 2), yaw (n,), half extents ext (n, 2)) -> its point nearest to q (2,)."""
    cs, sn = np.cos(yaw), np.sin(yaw)
    d = q[None] - c
    lx, ly = cs * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + cs * d[:, 1]
    lx, ly = np.clip(lx, -ext[:, 0], ext[:, 0]), np.clip(ly, -ext[:, 1], ext[:, 1])
    return c + np.stack([cs * lx - sn * ly, sn * lx + cs * ly], 1)


# ================================================================ P5

def p5_calib() -> dict:
    c = json.loads((data_dir() / "processed" / "carla_p5" / "op_plan.json").read_text())["calib"]
    return {cam: c[str(i + 1)] for i, cam in enumerate(CAMS)}


def p5_list() -> pd.DataFrame:
    """Observation frames (pair and null, both worlds) x 3 cameras; the current frame of each camera's 4-frame clip."""
    d = data_dir() / "processed" / "carla_p5"
    t = pd.read_parquet(d / "index.parquet")
    t = t[t.role == "obs"].reset_index(drop=True)
    obs = pd.read_parquet(d / "obs.parquet")
    fpx = obs.set_index("fn_plus").factor_px
    fpx = fpx[~fpx.index.duplicated()]
    rows = []
    for r in t.itertuples():
        f = list(r.files)
        assert len(f) == 12 and f[3].endswith(f"/front/{r.frame:07d}.jpg"), f[3]
        for i, cam in enumerate(CAMS):
            rows.append({"key": f"{r.frame_name}|{cam}", "path": f[4 * i + 3], "frame_name": r.frame_name, "cam": cam,
                         "adir": f[3].rsplit("/cams/", 1)[0], "frame": r.frame, "world": r.world, "base_id": r.base_id,
                         "seed": r.seed, "k": r.k, "sun_altitude": r.sun_altitude, "precipitation": r.precipitation})
    out = pd.DataFrame(rows)
    out["factor_px"] = out.frame_name.map(fpx).where(out.world == "plus")
    pairs = pd.read_csv(d / "pairs.csv", dtype={"base_id": str})
    out["family"] = out.base_id.map(pairs.drop_duplicates("base_id").set_index("base_id").family)
    return out


def _p5_gt_attempt(args) -> pd.DataFrame:
    adir, frames, calib = args
    adir = Path(adir)
    pose = {}
    with open(adir / "pose.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["frame"] in frames:
                pose[r["frame"]] = (r["x"], r["y"], r["z"], r["yaw"])
    a = np.load(adir / "actors.npz")
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    meta = json.loads((adir / "meta.json").read_text())
    hero = meta.get("hero_id")
    hz = {h["id"] for h in json.loads((adir / "hidden.json").read_text())} if (adir / "hidden.json").exists() else set()
    sel = np.isin(a["frame"], list(frames))
    fr, ids, xyz, yaw = a["frame"][sel], a["id"][sel], a["xyz"][sel], a["yaw"][sel]
    out = []
    for f in np.unique(fr):
        if f not in pose:
            continue
        hx, hy, hz_, hyaw = pose[f]
        m = (fr == f) & (ids != hero)
        k = [kinds.get(str(int(i))) for i in ids[m]]
        keep = np.array([x is not None for x in k])
        if not keep.any():
            continue
        idm, P, Y = ids[m][keep], xyz[m][keep], yaw[m][keep]
        k = [x for x in k if x is not None]
        bb = np.array([x[2] for x in k], np.float64)                     # loc xyz, extent xyz (actor local)
        ay = np.radians(Y)
        # footprint centre in the world (CARLA left-handed), bottom z
        cx = P[:, 0] + np.cos(ay) * bb[:, 0] - np.sin(ay) * bb[:, 1]
        cy = P[:, 1] + np.sin(ay) * bb[:, 0] + np.cos(ay) * bb[:, 1]
        bz = P[:, 2] + bb[:, 2] - bb[:, 5]
        # world -> hero actor local (left-handed) -> ego (rear axle, right-handed)
        th = np.radians(hyaw)
        dx, dy = cx - hx, cy - hy
        lx, ly = np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy
        ex, ey = lx - REAR_AXLE_X, -ly
        eyaw = -(ay - th)
        tid = [x[0] for x in k]
        cls = np.array(["pedestrian" if t.startswith("walker.") else "vehicle" for t in tid])
        for cam, cal in calib.items():
            cpos = np.asarray(cal["extrinsic"], np.float64).reshape(4, 4)[:3, 3]
            ref = nearest_on_box(np.stack([ex, ey], 1), eyaw, bb[:, 3:5], cpos[:2])
            z = bz - hz_
            u, v, ok = project(np.c_[ref, z], cal)
            ub, vb, okb = project(np.c_[ex, ey, z], cal)
            out.append(pd.DataFrame({"adir": str(adir), "frame": int(f), "cam": cam, "id": idm.astype(np.int64),
                                     "type_id": tid, "cls": cls, "emergency": np.isin(tid, EMERGENCY),
                                     "hazard": np.isin(idm, list(hz)), "x": ref[:, 0], "y": ref[:, 1], "xc": ex, "yc": ey,
                                     "dz": z, "u": u, "v": v, "in_img": ok, "ub": ub, "vb": vb, "in_img_c": okb}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def p5_gt(lst: pd.DataFrame, workers: int = 12) -> pd.DataFrame:
    """GT objects per (attempt, frame, camera) for every image of the list; keyed like the list."""
    calib = p5_calib()
    jobs = [(adir, set(g.frame.astype(int)), calib) for adir, g in lst.groupby("adir")]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_p5_gt_attempt, jobs, chunksize=4))
    gt = pd.concat([p for p in parts if len(p)], ignore_index=True)
    key = lst.set_index(["adir", "frame", "cam"]).key
    gt["key"] = key.reindex(pd.MultiIndex.from_frame(gt[["adir", "frame", "cam"]])).to_numpy()
    gt = gt[gt.key.notna()].reset_index(drop=True)
    gt["dist"] = np.hypot(gt.x, gt.y)
    gt["dist_c"] = np.hypot(gt.xc, gt.yc)
    return gt


# ================================================================ detections -> BEV, matching

def load_dets(det_dir, score: float = SCORE) -> pd.DataFrame:
    parts = sorted(Path(det_dir).glob("part-*.parquet"))
    d = pd.concat([pd.read_parquet(p, columns=["key", "prompt", "score", "x0", "y0", "x1", "y1", "area", "cu", "cv"])
                   for p in parts], ignore_index=True)
    return d[d.score > score].reset_index(drop=True)


def lift_dets(d: pd.DataFrame, cal_key: np.ndarray, calibs: dict) -> pd.DataFrame:
    """Ground-contact pixels -> ego BEV. cal_key: per detection row, the key into `calibs` (camera / sample_data)."""
    d = d.copy()
    gx, gy, ok = np.full(len(d), np.nan), np.full(len(d), np.nan), np.zeros(len(d), bool)
    cu, cv = d.cu.to_numpy(float), d.cv.to_numpy(float)
    for ck, idx in pd.Series(cal_key).groupby(cal_key).indices.items():
        g, o = lift(cu[idx], cv[idx], calibs[ck])
        gx[idx], gy[idx], ok[idx] = g[:, 0], g[:, 1], o
    d["gx"], d["gy"], d["lift_ok"] = gx, gy, ok
    d["gdist"] = np.hypot(gx, gy)
    return d


def gate(dist: np.ndarray) -> np.ndarray:
    return np.maximum(2.0, 0.1 * dist)


def match(dets: pd.DataFrame, gt: pd.DataFrame, cls_det: str, gt_mask: np.ndarray, ref=("x", "y")) -> tuple:
    """Hungarian per image between detections of prompt `cls_det` (lift_ok) and the GT rows in `gt_mask`.
    Returns (gt index -> matched det index or -1, det index -> matched gt index or -1, distances of matches)."""
    from scipy.optimize import linear_sum_assignment
    D = dets[(dets.prompt == cls_det) & dets.lift_ok]
    G = gt[gt_mask]
    g_m = pd.Series(-1, index=G.index)
    d_m = pd.Series(-1, index=D.index)
    dd = pd.Series(np.nan, index=G.index)
    dg = D.groupby("key").indices
    for key, gi in G.groupby("key").indices.items():
        di = dg.get(key)
        if di is None:
            continue
        gp = G.iloc[gi][list(ref)].to_numpy(float)
        dp = D.iloc[di][["gx", "gy"]].to_numpy(float)
        C = np.hypot(gp[:, None, 0] - dp[None, :, 0], gp[:, None, 1] - dp[None, :, 1])
        lim = gate(np.hypot(gp[:, 0], gp[:, 1]))[:, None]
        cost = np.where(C <= lim, C, 1e6)
        r, c = linear_sum_assignment(cost)
        good = cost[r, c] < 1e6
        r, c = r[good], c[good]
        g_m.iloc[gi[r]] = D.index[di[c]]
        d_m.iloc[di[c]] = G.index[gi[r]]
        dd.iloc[gi[r]] = C[r, c]
    return g_m, d_m, dd


def dist_bin(d) -> pd.Categorical:
    return pd.cut(d, DIST_BINS, right=False, labels=[f"{a}-{b} m" for a, b in zip(DIST_BINS[:-1], DIST_BINS[1:])])


# ================================================================ the 16-frame coordinate check

def pick_check(lst: pd.DataFrame, gt: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """16 x+ images with a clearly visible hazard: 8 pedestrian, 8 vehicle; per class 4 front + 2 per side camera,
    spread over 5-40 m (front: factor_px >= 200; side cameras: the hazard's reference point inside the image)."""
    rng = np.random.default_rng(seed)
    h = gt[gt.hazard & gt.in_img & (gt.dist >= 5) & (gt.dist <= 40)].merge(
        lst[["key", "world", "factor_px", "family"]], on="key")
    h = h[h.world == "plus"]
    h = h[(h.cam != "front") | (h.factor_px >= 200)]
    out = []
    for cls in ("pedestrian", "vehicle"):
        for cam, n in (("front", 4), ("front_left", 2), ("front_right", 2)):
            c = h[(h.cls == cls) & (h.cam == cam)].drop_duplicates("key")
            if not len(c):
                continue
            b = pd.cut(c.dist, np.linspace(5, 40, n + 1), include_lowest=True)
            pick = [g.sample(1, random_state=int(rng.integers(1 << 31))) for _, g in c.groupby(b, observed=True)]
            got = pd.concat(pick) if pick else c.iloc[:0]
            if len(got) < n:
                got = pd.concat([got, c[~c.key.isin(got.key)].sample(min(n - len(got), len(c) - len(got)),
                                                                     random_state=seed)])
            out.append(got.head(n))
    return pd.concat(out, ignore_index=True)


def check_ab(sel: pd.DataFrame, gt: pd.DataFrame, dets: pd.DataFrame) -> pd.DataFrame:
    """(a) projected GT reference point inside the matched mask's box, pixel error to the contact point;
    (b) BEV error of the lifted contact point. One row per selected hazard."""
    rows = []
    for r in sel.itertuples():
        g = gt[(gt.key == r.key)]
        cls = r.cls
        gm, _, dd = match(dets[dets.key == r.key], g, cls, (g.cls == cls).to_numpy())
        gi = g.index[g.id == r.id][0]
        di = gm.loc[gi]
        row = {"key": r.key, "cls": cls, "cam": r.cam, "dist": r.dist, "family": r.family, "matched": di >= 0}
        if di >= 0:
            d = dets.loc[di]
            row |= {"in_box": bool(d.x0 - 2 <= g.loc[gi, "u"] <= d.x1 + 2 and d.y0 - 2 <= g.loc[gi, "v"] <= d.y1 + 2),
                    "px_err": float(np.hypot(d.cu - g.loc[gi, "u"], d.cv - g.loc[gi, "v"])),
                    "bev_err": float(dd.loc[gi]), "bev_err_centre": float(np.hypot(d.gx - g.loc[gi, "xc"], d.gy - g.loc[gi, "yc"])),
                    "score": float(d.score)}
        else:   # nearest detection of the class, for diagnosis
            c = dets[(dets.key == r.key) & (dets.prompt == cls)]
            if len(c):
                j = np.argmin(np.hypot(c.gx - g.loc[gi, "x"], c.gy - g.loc[gi, "y"]))
                row |= {"nearest_det_bev": float(np.hypot(c.gx.iloc[j] - g.loc[gi, "x"], c.gy.iloc[j] - g.loc[gi, "y"])),
                        "nearest_det_px": float(np.hypot(c.cu.iloc[j] - g.loc[gi, "u"], c.cv.iloc[j] - g.loc[gi, "v"]))}
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================ nuScenes val

def _nusc_tables(names):
    import orjson
    from .common import dataroot
    return {n: pd.DataFrame(orjson.loads((dataroot() / "v1.0-trainval" / f"{n}.json").read_bytes())) for n in names}


def _quat_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def nusc_cls(cat: str, attrs: set) -> list[str]:
    """Clarification (4): nuScenes category -> our classes (possibly two, for emergency vehicles)."""
    if cat.startswith("human.pedestrian."):
        return ["pedestrian"]
    if cat == "vehicle.bicycle":
        return ["cyclist"] if "cycle.with_rider" in attrs else []
    if cat.startswith("vehicle.emergency."):
        return ["vehicle", "emergency vehicle"]
    if cat.split(".")[0] == "vehicle":
        return ["vehicle"]
    return {"movable_object.trafficcone": ["cone"], "movable_object.debris": ["debris"]}.get(cat, [])


def nusc_build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(image list, GT rows, calibration records by sample_data token) for the 3 front cameras of every val keyframe."""
    from .common import dataroot
    from .nuscenes_index import scene_splits
    t = _nusc_tables(["scene", "sample", "sample_data", "calibrated_sensor", "sensor", "ego_pose", "sample_annotation",
                      "instance", "category", "attribute"])
    split = scene_splits("v1.0-trainval")
    scene = t["scene"].assign(split=t["scene"].name.map(split))
    val = scene[scene.split == "val"]
    smp = t["sample"][t["sample"].scene_token.isin(val.token)][["token", "scene_token"]]
    cs = t["calibrated_sensor"].merge(t["sensor"][["token", "channel"]].rename(columns={"token": "sensor_token"}))
    sd = t["sample_data"].merge(cs[["token", "channel", "translation", "rotation", "camera_intrinsic"]]
                                .rename(columns={"token": "calibrated_sensor_token", "translation": "c_t", "rotation": "c_q"}))
    sd = sd[sd.channel.isin(NUSC_CAMS) & sd.is_key_frame & sd.sample_token.isin(smp.token)]
    ep = t["ego_pose"].rename(columns={"token": "ego_pose_token", "translation": "e_t", "rotation": "e_q"})
    sd = sd.merge(ep[["ego_pose_token", "e_t", "e_q"]]).merge(smp.rename(columns={"token": "sample_token"}))
    sd = sd.merge(val[["token", "name", "description"]].rename(columns={"token": "scene_token", "name": "scene"}))
    desc = sd.description.str.lower()
    lst = pd.DataFrame({"key": sd.token.to_numpy(), "path": [str(dataroot() / f) for f in sd.filename],
                        "cam": sd.channel.to_numpy(), "sample_token": sd.sample_token.to_numpy(), "scene": sd.scene.to_numpy(),
                        "night": desc.str.contains("night").to_numpy(), "rain": desc.str.contains("rain").to_numpy()})
    calibs = {}
    for r in sd.itertuples():
        K = np.asarray(r.camera_intrinsic, np.float64)
        E = np.eye(4)
        E[:3, :3] = _quat_R(r.c_q) @ OPENCV_TO_WOD.T
        E[:3, 3] = r.c_t
        calibs[r.token] = {"intrinsic": [K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0, 0], "extrinsic": E.ravel().tolist(),
                           "width": int(r.width), "height": int(r.height)}
    cat = t["instance"].merge(t["category"][["token", "name"]].rename(columns={"token": "category_token", "name": "cat"}))
    ann = t["sample_annotation"][t["sample_annotation"].sample_token.isin(smp.token)].merge(
        cat[["token", "cat"]].rename(columns={"token": "instance_token"}))
    attr = dict(zip(t["attribute"].token, t["attribute"].name))
    ann["cls"] = [nusc_cls(c, {attr[a] for a in at}) for c, at in zip(ann.cat, ann.attribute_tokens)]
    ann = ann[ann.cls.map(len) > 0].explode("cls")
    rows = []
    by_sample = ann.groupby("sample_token")
    for r in sd.itertuples():
        if r.sample_token not in by_sample.groups:
            continue
        a = by_sample.get_group(r.sample_token)
        Re = _quat_R(r.e_q)
        c = (np.stack(a.translation.to_numpy()) - np.asarray(r.e_t)) @ Re          # global -> ego (at camera time)
        yaw_e = np.arctan2(Re[1, 0], Re[0, 0])
        q = np.stack(a.rotation.to_numpy())
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)) - yaw_e
        size = np.stack(a["size"].to_numpy())                                        # w, l, h
        cal = calibs[r.token]
        cpos = np.asarray(cal["extrinsic"]).reshape(4, 4)[:3, 3]
        ref = nearest_on_box(c[:, :2], yaw, size[:, [1, 0]] / 2, cpos[:2])
        bz = c[:, 2] - size[:, 2] / 2
        u, v, ok = project(np.c_[ref, bz], cal)
        rows.append(pd.DataFrame({"key": r.token, "cam": r.channel, "id": a.instance_token.to_numpy(), "cls": a.cls.to_numpy(),
                                  "cat": a.cat.to_numpy(), "vis": a.visibility_token.astype(int).to_numpy(),
                                  "x": ref[:, 0], "y": ref[:, 1], "xc": c[:, 0], "yc": c[:, 1], "dz": bz, "u": u, "v": v,
                                  "in_img": ok}))
    gt = pd.concat(rows, ignore_index=True)
    gt["dist"] = np.hypot(gt.x, gt.y)
    gt["hazard"], gt["emergency"] = False, gt.cls == "emergency vehicle"
    return lst, gt, calibs


# ================================================================ WOD front images (for Q2b)

def wod_list() -> pd.DataFrame:
    from . import waymo as W
    from .waymo_ladder import subset_path
    df = W.load_index()
    names = W.frame_names(df)
    s = pd.read_parquet(subset_path())
    r = df.iloc[s.row.to_numpy()]
    assert (names[s.row.to_numpy()] == s.frame_name.to_numpy()).all()
    return pd.DataFrame({"key": s.frame_name.to_numpy(), "path": "", "shard": [str(W.shard_dir() / x) for x in r.shard],
                         "off": r.front_off.to_numpy(), "len": r.front_len.to_numpy(), "sequence": s.sequence.to_numpy(),
                         "rater": s.rater.to_numpy()})


# ================================================================ reports

def recall_table(gt: pd.DataFrame, g_m: pd.Series, dd: pd.Series, by: list, name: str) -> pd.DataFrame:
    t = gt.assign(hit=(g_m.reindex(gt.index) >= 0), err=dd.reindex(gt.index))
    out = t.groupby(by, observed=True).agg(n=("hit", "size"), recall=("hit", "mean"),
                                            bev_err_med=("err", "median"),
                                            bev_err_p90=("err", lambda x: x.quantile(0.9))).reset_index()
    return out.assign(reading=name)


def precision_table(d: pd.DataFrame, d_m: pd.Series, by: list, name: str) -> pd.DataFrame:
    t = d.assign(tp=(d_m.reindex(d.index) >= 0))
    return t.groupby(by, observed=True).agg(n=("tp", "size"), precision=("tp", "mean")).reset_index().assign(reading=name)


def evaluate(lst: pd.DataFrame, gt: pd.DataFrame, dets: pd.DataFrame, classes, weather) -> dict:
    """Recall (ii) on in-image GT <= 40 m, precision against in-image GT <= 80 m, BEV error; per class x distance x
    weather. `weather`: key -> weather label. Also the bottom-centre side reading (clarification 3).
    An optional boolean column `eval` restricts the recall rows (nuScenes visibility) while every GT object stays in
    the matching pool, so a detection of a poorly visible object is not counted as a false positive."""
    if "eval" not in gt:
        gt = gt.assign(eval=True)
    gt = gt.assign(dbin=dist_bin(gt.dist), weather=gt.key.map(weather))
    dets = dets.assign(dbin=dist_bin(dets.gdist), weather=dets.key.map(weather))
    rec, prec, cen = [], [], []
    for cls in classes:
        is_c = (gt.cls == cls) if cls != "emergency vehicle" else gt.emergency
        pool = (is_c & gt.in_img & (gt.dist <= MAX_LIFT) & (gt.dz.abs() < 8)).to_numpy()
        g_m, d_m, dd = match(dets, gt, cls, pool)
        ev = gt[pool & (gt.dist < 40).to_numpy() & gt["eval"].to_numpy()]
        for by in (["dbin"], ["weather"], ["dbin", "weather"]):
            rec.append(recall_table(ev.assign(cls=cls), g_m, dd, ["cls", *by], "recall (ii), <= 40 m, in image"))
        rec.append(recall_table(ev.assign(cls=cls, all="all"), g_m, dd, ["cls", "all"], "recall (ii), <= 40 m, in image"))
        D = dets[(dets.prompt == cls) & dets.lift_ok]
        for by in (["dbin"], ["weather"]):
            prec.append(precision_table(D.assign(cls=cls), d_m, ["cls", *by], "precision, lifted <= 80 m"))
        prec.append(precision_table(D.assign(cls=cls, all="all"), d_m, ["cls", "all"], "precision, lifted <= 80 m"))
        g2, _, dd2 = match(dets, gt, cls, pool, ref=("xc", "yc"))
        cen.append(recall_table(ev.assign(cls=cls, all="all"), g2, dd2, ["cls", "all"], "side: bottom-centre reference"))
    return {"recall": pd.concat(rec, ignore_index=True), "precision": pd.concat(prec, ignore_index=True),
            "centre": pd.concat(cen, ignore_index=True)}


def hazard_reading(lst: pd.DataFrame, gt: pd.DataFrame, dets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reading (i): hazard actors on x+ front frames with factor_px >= FACTOR_PX (front detections only)."""
    L = lst.set_index("key")
    g = gt[gt.hazard & (gt.cam == "front")].copy()
    g["world"], g["factor_px"] = g.key.map(L.world), g.key.map(L.factor_px)
    g["family"], g["day"] = g.key.map(L.family), g.key.map(L.sun_altitude) >= 0
    g["rain"] = g.key.map(L.precipitation) > 30
    g = g[(g.world == "plus") & (g.factor_px >= FACTOR_PX) & (g.dz.abs() < 8)]
    rows, per = [], []
    for cls in ("pedestrian", "vehicle"):
        m = (g.cls == cls).to_numpy()
        g_m, _, dd = match(dets, g, cls, m)
        h = g[m].assign(hit=g_m[g[m].index] >= 0, err=dd[g[m].index], dbin=dist_bin(g[m].dist))
        per.append(h)
        for name, sub in (("all", h), ("<= 30 m, day", h[(h.dist <= 30) & h.day]), ("<= 20 m", h[h.dist <= 20]),
                          ("night", h[~h.day]), ("rain", h[h.rain])):
            rows.append({"cls": cls, "scope": name, "n": len(sub), "frames": sub.key.nunique(),
                         "recall": sub.hit.mean() if len(sub) else np.nan,
                         "bev_err_med": sub.err.median(), "bev_err_p90": sub.err.quantile(0.9)})
        for (fam, b), sub in h.groupby(["family", "dbin"], observed=True):
            rows.append({"cls": cls, "scope": f"{fam} {b}", "n": len(sub), "frames": sub.key.nunique(),
                         "recall": sub.hit.mean(), "bev_err_med": sub.err.median(), "bev_err_p90": sub.err.quantile(0.9)})
    return pd.DataFrame(rows), pd.concat(per, ignore_index=True)


def boot_recall(h: pd.DataFrame, group: str = "base_id", b: int = 2000, seed: int = 0) -> tuple[float, float]:
    codes, uniq = pd.factorize(h[group])
    num = np.bincount(codes, h.hit.astype(float), len(uniq))
    den = np.bincount(codes, minlength=len(uniq)).astype(float)
    idx = np.random.default_rng(seed).integers(len(uniq), size=(b, len(uniq)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


# ================================================================ CLI

def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("lists", "check_pick", "check_ab", "p5_report", "nusc_report"))
    ap.add_argument("--dets", help="sam_detect output directory")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    rl = RunLog("fusion_diag", "q4", a.step)
    rl.event("start", args=vars(a))
    L = root("lists")
    if a.step == "lists":
        p5 = p5_list()
        p5.to_parquet(L / "p5.parquet", index=False)
        gt = p5_gt(p5, a.workers)
        gt.to_parquet(L / "p5_gt.parquet", index=False)
        rl.info(f"p5: {len(p5)} images, {len(gt)} GT rows, hazards in front x+ with factor_px >= {FACTOR_PX}: "
                f"{int((gt.hazard & (gt.cam == 'front')).sum())}")
        nl, ng, nc = nusc_build()
        nl.to_parquet(L / "nusc.parquet", index=False)
        ng.to_parquet(L / "nusc_gt.parquet", index=False)
        (L / "nusc_calib.json").write_text(json.dumps(nc))
        rl.info(f"nuscenes: {len(nl)} images, {len(ng)} GT rows; ground z of GT boxes (median dz, <= 30 m, vis >= 3): "
                f"{ng[(ng.dist <= 30) & (ng.vis >= 3)].dz.median():.3f}")
        w = wod_list()
        w.to_parquet(L / "wod.parquet", index=False)
        rl.info(f"wod: {len(w)} front images ({int(w.rater.sum())} rater frames)")
    elif a.step == "check_pick":
        p5, gt = pd.read_parquet(L / "p5.parquet"), pd.read_parquet(L / "p5_gt.parquet")
        sel = pick_check(p5, gt)
        sel.to_parquet(L / "check16_sel.parquet", index=False)
        p5[p5.key.isin(sel.key)].drop_duplicates("key").to_parquet(L / "check16.parquet", index=False)
        rl.info("check selection:\n" + sel[["key", "cls", "cam", "dist", "family", "factor_px"]].to_string())
    elif a.step == "check_ab":
        p5, gt, sel = (pd.read_parquet(L / f) for f in ("p5.parquet", "p5_gt.parquet", "check16_sel.parquet"))
        cal = p5_calib()
        d = load_dets(a.dets)
        d = lift_dets(d, d.key.str.split("|").str[1].to_numpy(), cal)
        g = gt[gt.key.isin(sel.key)]
        r = check_ab(sel, g, d)
        r.to_csv(rl.dir / "check_ab.csv", index=False)
        m = r[r.matched]
        rl.info("check (a)(b):\n" + r.to_string() + f"\nmatched {len(m)}/{len(r)}; (a) in box {m.in_box.mean():.2f}, "
                f"px err median {m.px_err.median():.1f}; (b) BEV err median (<= 20 m) {m[m.dist <= 20].bev_err.median():.2f} m")
    elif a.step == "p5_report":
        p5, gt = pd.read_parquet(L / "p5.parquet"), pd.read_parquet(L / "p5_gt.parquet")
        d = load_dets(a.dets)
        d = lift_dets(d, d.key.str.split("|").str[1].to_numpy(), p5_calib())
        wx = p5.set_index("key")
        weather = np.where(wx.sun_altitude < 0, "night", np.where(wx.precipitation > 30, "rain", "day"))
        ev = evaluate(p5, gt, d, GT_CLASSES_P5, pd.Series(weather, index=wx.index))
        haz, per = hazard_reading(p5, gt, d)
        per["base_id"] = per.key.map(wx.base_id)
        ci = {cls: boot_recall(per[per.cls == cls]) for cls in ("pedestrian", "vehicle")}
        haz["ci"] = haz.apply(lambda r: ci[r.cls] if r.scope == "all" else None, axis=1)
        for k, t in {**ev, "hazard": haz}.items():
            t.to_csv(rl.dir / f"p5_{k}.csv", index=False)
            rl.info(f"P5 {k}:\n" + t.to_string())
        per.to_parquet(rl.dir / "p5_hazard_frames.parquet", index=False)
    elif a.step == "nusc_report":
        nl, ng = pd.read_parquet(L / "nusc.parquet"), pd.read_parquet(L / "nusc_gt.parquet")
        cal = json.loads((L / "nusc_calib.json").read_text())
        d = load_dets(a.dets)
        d = lift_dets(d, d.key.to_numpy(), cal)
        nx = nl.set_index("key")
        weather = pd.Series(np.where(nx.night, "night", np.where(nx.rain, "rain", "day")), index=nx.index)
        for vis in (3, 2):
            ev = evaluate(nl, ng.assign(eval=ng.vis >= vis), d, CLASSES, weather)
            for k, t in ev.items():
                t.to_csv(rl.dir / f"nusc_vis{vis}_{k}.csv", index=False)
                rl.info(f"nuScenes vis >= {vis} {k}:\n" + t.to_string())
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
