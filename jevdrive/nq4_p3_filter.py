"""nq4 P3 exam-item filter (proposed pre-registration amendment, descriptive only; the registered gate stands).

A P3 pair (x+ = re-render, x- = the same re-render with the corridor pedestrians deleted) is a valid "should react"
item only at frames where the correct ego action differs between the two worlds. The +-4 m / 30 m corridor used for
selection is wider than that: it admits pedestrians on the kerb, pedestrians beyond a lead vehicle, pedestrians the
camera never shows, and pedestrians the logged driver ignored. Rules (todos/2026-09-26-night-queue-4.md, P section,
proposed amendment 2026-09-28), all on WOD v2 labels, evaluated per 10 Hz frame i and pedestrian track p:

  path      the logged ego path in the world frame (whole segment, extended straight); S, L = arc and signed lateral
            offset of an object centre on it; d = S_obj - S_ego (from the rear-axle origin)
  in_lane   |L| <= LANE and FRONT <= d <= FRONT + REACH (in front of the bumper, inside the ego lane, <= 30 m)
  conflict  p is in_lane at some frame of [i, i + ENTER_S] and ahead at i, and the time to reach it at the current speed
            (floored at V_FLOOR) is <= TTR_S
  lead      a vehicle or cyclist in the ego lane between ego and p at i (|L| <= LANE, FRONT <= d_v <= d_p - 1):
            the lead's problem, an experienced driver follows the lead; such frames are not items
  seen      p carries a human FRONT-camera label (camera_box associated with its laser id) of height >= H_MIN native px
            in every frame of the SEEN_S before and including i (noticeable over consecutive frames, not one still)
  react     conflict & ~lead & seen (a "should react" frame)
  responded the logged driver slowed: max speed over [t_c - 1 s, t_c] minus min speed over [t_c, t_c + 3 s]
            >= max(DV_MIN, DV_FRAC * that max), t_c = first react frame of the event
  psnr      (post-reconstruction, rendered scenes only) x+ against the log inside the reacted pedestrian's projected
            front-camera box, mean over its react frames, >= PSNR_MIN

  python -m jevdrive.nq4_p3_filter events --out <dir>    # all WOD v2 segments -> events.parquet, frames of scenes 0-9
  python -m jevdrive.nq4_p3_filter scenes --out <dir>    # scenes 0-9: frame flags, per-pedestrian box PSNR, readouts
  python -m jevdrive.nq4_p3_filter summary --out <dir> --sens <dir>/sens/*   # funnel, survivors, response evidence
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger, n_cpus
from . import nq4_p3_select as SEL

log = get_logger(__name__)
V2 = SEL.V2
L_ = "[LiDARBoxComponent]."
C_ = "[CameraBoxComponent]."
K_ = "[CameraCalibrationComponent]."
HZ = 10
LANE, FRONT, REACH, TTR_S, V_FLOOR, ENTER_S = 1.75, 4.0, 30.0, 4.0, 3.0, 2.0
H_MIN, SEEN_S = 40.0, 1.0
DV_MIN, DV_FRAC, RESP_PRE, RESP_POST = 1.0, 0.2, 1.0, 3.0
PSNR_MIN = 20.0
LEAD_TYPES = (1, 4)                    # vehicle, cyclist
WIN_PRE, WIN_POST = 3.0, 2.0           # registered render window
F0_RANGE, MIN_SPEED, MAX_DEL = (3.0, 16.5), 2.0, 5
SCORED = np.arange(-24, 21, 2)         # scored frames of a rendered scene, log frames relative to f0 (5 Hz, -2.4..+2.0 s)


# ---------------------------------------------------------------- loading
def load(split: str, seg: str) -> dict:
    pose = pd.read_parquet(V2 / split / "vehicle_pose" / f"{seg}.parquet").sort_values("key.frame_timestamp_micros")
    ts = pose["key.frame_timestamp_micros"].to_numpy()
    T = np.stack(pose["[VehiclePoseComponent].world_from_vehicle.transform"].to_numpy()).reshape(-1, 4, 4)
    fidx = pd.Series(np.arange(len(ts)), index=ts)
    b = pd.read_parquet(V2 / split / "lidar_box" / f"{seg}.parquet")
    b = b[b[L_ + "type"].isin((1, 2, 4))]
    box = pd.DataFrame({"frame": fidx.reindex(b["key.frame_timestamp_micros"]).to_numpy(), "track": b["key.laser_object_id"].to_numpy(),
                        "type": b[L_ + "type"].to_numpy(), "npts": b[L_ + "num_lidar_points_in_box"].to_numpy(),
                        **{k: b[L_ + c].to_numpy() for k, c in (("x", "box.center.x"), ("y", "box.center.y"), ("z", "box.center.z"),
                                                                ("sx", "box.size.x"), ("sy", "box.size.y"), ("sz", "box.size.z"),
                                                                ("hd", "box.heading"), ("vx", "speed.x"), ("vy", "speed.y"),
                                                                ("ax", "acceleration.x"), ("ay", "acceleration.y"))}}).dropna(subset=["frame"])
    box["frame"] = box.frame.astype(int)
    cb = pd.read_parquet(V2 / split / "camera_box" / f"{seg}.parquet")
    cb = cb[cb["key.camera_name"] == 1]
    cv = cb[cb[C_ + "type"] == 1]                    # vehicle labels: no 2D-3D association in WOD, matched by IoU later
    vlab = pd.DataFrame({"frame": fidx.reindex(cv["key.frame_timestamp_micros"]).to_numpy(),
                         **{k: cv[C_ + c].to_numpy() for k, c in (("u", "box.center.x"), ("v", "box.center.y"),
                                                                  ("w", "box.size.x"), ("h", "box.size.y"))}}).dropna(subset=["frame"])
    vlab["frame"] = vlab.frame.astype(int)
    cb = cb[cb[C_ + "type"] == 2]
    asc = pd.read_parquet(V2 / split / "camera_to_lidar_box_association" / f"{seg}.parquet")
    cb = cb.merge(asc, on=["key.frame_timestamp_micros", "key.camera_name", "key.camera_object_id"])
    lab = pd.DataFrame({"frame": fidx.reindex(cb["key.frame_timestamp_micros"]).to_numpy(), "track": cb["key.laser_object_id"].to_numpy(),
                        "lab_h": cb[C_ + "box.size.y"].to_numpy(), "lab_w": cb[C_ + "box.size.x"].to_numpy(),
                        "lab_u": cb[C_ + "box.center.x"].to_numpy(), "lab_v": cb[C_ + "box.center.y"].to_numpy()}).dropna(subset=["frame"])
    lab["frame"] = lab.frame.astype(int)
    lab = lab.sort_values("lab_h").drop_duplicates(["frame", "track"], keep="last")
    cc = pd.read_parquet(V2 / split / "camera_calibration" / f"{seg}.parquet").set_index("key.camera_name").loc[1]
    cal = {"f": (cc[K_ + "intrinsic.f_u"], cc[K_ + "intrinsic.f_v"]), "c": (cc[K_ + "intrinsic.c_u"], cc[K_ + "intrinsic.c_v"]),
           "ext": np.asarray(cc[K_ + "extrinsic.transform"], float).reshape(4, 4), "wh": (int(cc[K_ + "width"]), int(cc[K_ + "height"]))}
    st = pd.read_parquet(V2 / split / "stats" / f"{seg}.parquet", columns=["[StatsComponent].time_of_day"])
    return {"seg": seg, "split": split, "t": (ts - ts[0]) / 1e6, "T": T, "box": box, "lab": lab, "vlab": vlab, "cal": cal,
            "day": bool((st["[StatsComponent].time_of_day"] == "Day").all())}


def project_box(r, cal: dict, scale: float = 1.0):
    """Axis-aligned image rectangle (u0, v0, u1, v1) of a vehicle-frame 3D box in the FRONT camera (pinhole, no
    distortion), clipped to the image; None when it is behind the camera or off the image."""
    c, s = np.cos(r.hd), np.sin(r.hd)
    o = np.array([[a, b, h] for a in (-.5, .5) for b in (-.5, .5) for h in (-.5, .5)]) * [r.sx, r.sy, r.sz]
    p = o @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [r.x, r.y, r.z]
    q = (np.linalg.inv(cal["ext"]) @ np.c_[p, np.ones(8)].T).T[:, :3]          # Waymo camera frame: x fwd, y left, z up
    q = q[q[:, 0] > 0.1]
    if len(q) == 0:
        return None
    u = (cal["c"][0] - cal["f"][0] * q[:, 1] / q[:, 0]) * scale
    v = (cal["c"][1] - cal["f"][1] * q[:, 2] / q[:, 0]) * scale
    W, H = cal["wh"][0] * scale, cal["wh"][1] * scale
    u0, v0, u1, v1 = max(u.min(), 0), max(v.min(), 0), min(u.max(), W), min(v.max(), H)
    return None if u0 >= u1 or v0 >= v1 else (u0, v0, u1, v1)


# ---------------------------------------------------------------- per-frame features
def speed(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    v = np.zeros(len(t))
    v[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) / (t[2:] - t[:-2])
    v[0], v[-1] = v[1], v[-2]
    return v


def path_coords(xy: np.ndarray, yaw: np.ndarray):
    """Logged path polyline (extended 20 m back, 150 m ahead), its arc length, and the ego's own arc per frame."""
    back = xy[0] - 20 * np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    ahead = xy[-1] + 150 * np.array([np.cos(yaw[-1]), np.sin(yaw[-1])])
    P = np.vstack([back, xy, ahead])
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return P, cum, cum[1:-1]


def project(P: np.ndarray, cum: np.ndarray, W: np.ndarray, s_lo: float, s_hi: float):
    """Arc S and signed lateral L (left +) of world points W on the polyline segments whose arc overlaps [s_lo, s_hi]."""
    k = np.flatnonzero((cum[1:] >= s_lo) & (cum[:-1] <= s_hi))
    a, ab = P[k], P[k + 1] - P[k]
    n2 = (ab ** 2).sum(1)
    ok = n2 > 1e-6
    a, ab, n2, k = a[ok], ab[ok], n2[ok], k[ok]
    rel = W[:, None, :] - a[None]
    u = np.clip((rel * ab[None]).sum(-1) / n2[None], 0, 1)
    dist = np.linalg.norm(rel - u[..., None] * ab[None], axis=-1)
    j = dist.argmin(1)
    r = np.arange(len(W))
    ln = np.sqrt(n2[j])
    S = cum[k[j]] + u[r, j] * ln
    L = (ab[j, 0] * rel[r, j, 1] - ab[j, 1] * rel[r, j, 0]) / ln
    return S, L, np.arctan2(ab[j, 1], ab[j, 0])


def run_len(flag: np.ndarray) -> np.ndarray:
    """Length of the run of True ending at each index (0 where False)."""
    out = np.zeros(len(flag), int)
    c = 0
    for i, f in enumerate(flag):
        c = c + 1 if f else 0
        out[i] = c
    return out


def frame_features(D: dict) -> tuple[pd.DataFrame, dict]:
    """Per (frame, pedestrian track) features and per-frame ego quantities."""
    t, T, box = D["t"], D["T"], D["box"]
    n = len(t)
    xy, yaw = T[:, :2, 3], np.arctan2(T[:, 1, 0], T[:, 0, 0])
    v = speed(t, xy)
    P, cum, Se = path_coords(xy, yaw)
    rows = []
    for i, g in box.groupby("frame"):
        W = (T[i] @ np.c_[g[["x", "y", "z"]].to_numpy(), np.ones(len(g))].T).T[:, :2]
        S, L, _ = project(P, cum, W, Se[i] - 10, Se[i] + FRONT + REACH + 60)
        rows.append(g.assign(S=S, L=L, d=S - Se[i]))
    ob = pd.concat(rows, ignore_index=True)
    ped = ob[ob.type == 2].copy()
    veh = ob[ob.type.isin(LEAD_TYPES) & (ob.L.abs() <= LANE) & (ob.d >= FRONT)]
    near_lead = veh.groupby("frame").d.min()                               # nearest in-lane vehicle / cyclist ahead
    ped["lead"] = ped.d - 1.0 >= ped.frame.map(near_lead).fillna(np.inf)
    ped["in_lane"] = (ped.L.abs() <= LANE) & (ped.d >= FRONT) & (ped.d <= FRONT + REACH)
    ped["ttr"] = (ped.d - FRONT) / np.maximum(v[ped.frame], V_FLOOR)
    # registered +-4 m / 30 m corridor (the deletion set is every track inside it during the render window)
    local = [np.zeros((0, 2))] * n
    fr = {int(f): g for f, g in ped.groupby("frame")}
    for f, g in fr.items():
        local[f] = g[["x", "y"]].to_numpy()
    cm = SEL.corridor_mask(xy, yaw, local)
    ped["corridor"] = False
    for f, g in fr.items():
        ped.loc[g.index, "corridor"] = cm[f]
    ped = ped.merge(D["lab"], on=["frame", "track"], how="left")
    ped["labelled"] = ped.lab_h.notna()
    ped["vis"] = ped.lab_h.fillna(0) >= H_MIN
    ped = ped.sort_values(["track", "frame"]).reset_index(drop=True)
    E = int(ENTER_S * HZ)
    seen_n = int(round(SEEN_S * HZ))
    enter, seen = np.zeros(len(ped), bool), np.zeros(len(ped), bool)
    for tr, g in ped.groupby("track"):
        full_lane = np.zeros(n + E + 1, bool)
        full_lane[g.frame.to_numpy()] = g.in_lane.to_numpy()
        csum = np.r_[0, np.cumsum(full_lane)]
        f = g.frame.to_numpy()
        enter[g.index] = csum[f + E + 1] - csum[f] > 0
        full_vis = np.zeros(n, bool)
        full_vis[f] = g.vis.to_numpy()
        seen[g.index] = run_len(full_vis)[f] >= seen_n
    ped["conflict"] = enter & (ped.d >= FRONT) & (ped.d <= FRONT + REACH) & (ped.ttr <= TTR_S)
    ped["seen"] = seen
    ped["lane2"] = enter & (ped.d >= FRONT)                                # in the lane now or within ENTER_S, ahead
    ped["react"] = ped.conflict & ~ped.lead & ped.seen
    return ped, {"v": v, "t": t, "n": n}


def response(v: np.ndarray, i: int) -> dict:
    """Logged slow-down around frame i: max speed over [i - 1 s, i] minus min speed over [i, i + 3 s]."""
    a, b = max(i - int(RESP_PRE * HZ), 0), min(i + int(RESP_POST * HZ), len(v) - 1)
    vp, vm = float(v[a:i + 1].max()), float(v[i:b + 1].min())
    return {"v_pre": vp, "v_min_post": vm, "dv": vp - vm, "post_s": (b - i) / HZ,
            "responded": bool(vp - vm >= max(DV_MIN, DV_FRAC * vp))}


# ---------------------------------------------------------------- amendment 2: frame categories
CATS = ("react", "react_unconfirmed", "stopped_hold", "behind_lead", "not_yet_visible", "stopped_other", "gray_far", "clean")


def released(ob: pd.DataFrame, v: np.ndarray, tr, f: int, n: int) -> bool:
    """Stopped ego held by object tr at frame f: the ego moves off (>= 1 m/s) within 2 s after tr leaves the lane."""
    lane = set(ob[(ob.track == tr) & ob.in_lane].frame)
    clear = next((x for x in range(f, n) if x not in lane), n)
    go = next((x for x in range(f, n) if v[x] >= 1.0), n)
    return clear < n and go < n and 0 <= (go - clear) / HZ <= 2


def object_classes(ob: pd.DataFrame, v: np.ndarray, n: int) -> pd.Series:
    """Amendment-2 class of each (frame, deleted object) row; ob carries lane2, conflict, lead, seen, in_lane."""
    mov = v[ob.frame.to_numpy()] >= MIN_SPEED
    raw = mov & ob.conflict.to_numpy() & ~ob.lead.to_numpy() & ob.seen.to_numpy()
    resp = np.zeros(len(ob), bool)
    for tr, g in ob[raw].groupby("track"):                 # driver response per run of react frames
        f = g.frame.to_numpy()
        run = np.cumsum(np.r_[True, np.diff(f) > 2])
        for r in np.unique(run):
            resp[ob.index.get_indexer(g.index[run == r])] = response(v, int(f[run == r][0]))["responded"]
    stop = ~mov & ob.lane2.to_numpy()
    ok = stop & ~ob.lead.to_numpy() & ob.seen.to_numpy()
    hold = np.zeros(len(ob), bool)
    for j in np.flatnonzero(ok):
        hold[j] = released(ob, v, ob.track.iat[j], int(ob.frame.iat[j]), n)
    lane2, conf, lead, seen = (ob[c].to_numpy() for c in ("lane2", "conflict", "lead", "seen"))
    cls = np.select([raw & resp, raw, ok & hold, (conf | lane2) & lead, mov & conf & ~lead & ~seen, stop, mov & lane2 & ~conf],
                    list(CATS[:7]), "clean")
    return pd.Series(cls, index=ob.index)


def frame_category(classes: pd.Series, frames: pd.Series) -> dict:
    """frame -> the highest-priority class among the deleted objects present (clean when none is)."""
    rank = {c: i for i, c in enumerate(CATS)}
    out = {}
    for f, c in zip(frames, classes):
        if f not in out or rank[c] < rank[out[f]]:
            out[f] = c
    return out


# ---------------------------------------------------------------- amendment 2: vehicle events
VEH_CATS = ("cut_in", "lead_brake", "obstacle", "crossing", "oncoming")


def _iou(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def vehicle_features(D: dict) -> tuple[pd.DataFrame, dict]:
    """Per (frame, vehicle track): path S / L / d, heading and velocity relative to the path, ego-frame x / y,
    TTC, lane flags, 'another vehicle between', FRONT-label visibility (IoU >= 0.3 match)."""
    t, T, box = D["t"], D["T"], D["box"]
    n = len(t)
    xy, yaw = T[:, :2, 3], np.arctan2(T[:, 1, 0], T[:, 0, 0])
    v = speed(t, xy)
    P, cum, Se = path_coords(xy, yaw)
    ob = box[box.type.isin(LEAD_TYPES)].copy()
    S, L, th = np.zeros(len(ob)), np.zeros(len(ob)), np.zeros(len(ob))
    for i, idx in ob.groupby("frame").indices.items():
        g = ob.iloc[idx]
        W = (T[i] @ np.c_[g[["x", "y", "z"]].to_numpy(), np.ones(len(g))].T).T[:, :2]
        S[idx], L[idx], th[idx] = project(P, cum, W, Se[i] - 10, Se[i] + FRONT + REACH + 60)
    f = ob.frame.to_numpy()
    ob["S"], ob["L"], ob["d"] = S, L, S - Se[f]
    yw = yaw[f]
    ob["rel"] = np.degrees(np.abs(np.angle(np.exp(1j * (yw + ob.hd.to_numpy() - th)))))
    c, s_ = np.cos(yw - th), np.sin(yw - th)                  # vehicle-frame vectors -> along the path tangent
    ob["v_long"] = c * ob.vx.to_numpy() - s_ * ob.vy.to_numpy()
    ob["a_long"] = c * ob.ax.to_numpy() - s_ * ob.ay.to_numpy()
    ob["spd"] = np.hypot(ob.vx, ob.vy)
    ob["in_lane"] = (ob.L.abs() <= LANE) & (ob.d >= FRONT) & (ob.d <= FRONT + REACH)
    ob["ttc"] = (ob.d - FRONT - ob.sx / 2) / np.maximum(v[f] - ob.v_long, 0.5)
    ob["lane0"] = (ob.y.abs() <= LANE) & (ob.x >= FRONT) & (ob.x <= FRONT + REACH)
    ob["ttc0"] = (ob.x - FRONT - ob.sx / 2) / np.maximum(v[f], 0.5)
    # another vehicle / cyclist between ego and this one, on the path lane and on the straight lane
    between, between0 = np.zeros(len(ob), bool), np.zeros(len(ob), bool)
    for i, idx in ob.groupby("frame").indices.items():
        g = ob.iloc[idx]
        dl = g.d.to_numpy()[g.in_lane.to_numpy()]
        x0 = g.x.to_numpy()[g.lane0.to_numpy()]
        between[idx] = [(dl < d - 1).any() for d in g.d.to_numpy()]
        between0[idx] = [(x0 < x - 1).any() for x in g.x.to_numpy()]
    ob["between"], ob["between0"] = between, between0
    # FRONT-camera visibility by greedy IoU matching of projected 3D boxes to the vehicle labels
    lab_h = np.full(len(ob), np.nan)
    vl = {fr: g[["u", "v", "w", "h"]].to_numpy() for fr, g in D["vlab"].groupby("frame")}
    for i, idx in ob.groupby("frame").indices.items():
        if i not in vl:
            continue
        lb = np.c_[vl[i][:, 0] - vl[i][:, 2] / 2, vl[i][:, 1] - vl[i][:, 3] / 2, vl[i][:, 0] + vl[i][:, 2] / 2, vl[i][:, 1] + vl[i][:, 3] / 2]
        cand = []
        for j in idx:
            r = ob.iloc[j]
            if r.x < 1 or r.x > 80:
                continue
            bx = project_box(r, D["cal"])
            if bx:
                for k, l in enumerate(lb):
                    iou = _iou(bx, l)
                    if iou >= 0.3:
                        cand.append((iou, j, k))
        used_j, used_k = set(), set()
        for iou, j, k in sorted(cand, reverse=True):
            if j not in used_j and k not in used_k:
                used_j.add(j), used_k.add(k)
                lab_h[j] = vl[i][k, 3]
    ob["lab_h"] = lab_h
    ob = ob.sort_values(["track", "frame"]).reset_index(drop=True)
    E, seen_n = int(ENTER_S * HZ), int(round(SEEN_S * HZ))
    lane2, seen = np.zeros(len(ob), bool), np.zeros(len(ob), bool)
    for tr, idx in ob.groupby("track").indices.items():
        g = ob.iloc[idx]
        fr = g.frame.to_numpy()
        full = np.zeros(n + E + 1, bool)
        full[fr] = g.in_lane.to_numpy()
        cs = np.r_[0, np.cumsum(full)]
        lane2[idx] = cs[fr + E + 1] - cs[fr] > 0
        vis = np.zeros(n, bool)
        vis[fr] = np.nan_to_num(g.lab_h.to_numpy()) >= H_MIN
        seen[idx] = run_len(vis)[fr] >= seen_n
    ob["lane2"] = lane2 & (ob.d >= FRONT)
    ob["seen"] = seen
    ob["conflict"] = ob.lane2 & (ob.d <= FRONT + REACH) & (ob.ttc <= TTR_S)
    ob["lead"] = ob.between
    return ob, {"v": v, "t": t, "n": n, "yaw": yaw, "xy": xy}


def lateral_dev(xy: np.ndarray, yaw: np.ndarray, i: int, horizon: int) -> tuple[float, float]:
    """Max |lateral offset| of the logged path over the next `horizon` frames from the straight line of pose i, and the
    heading change over the same span (deg)."""
    j = min(i + horizon, len(xy) - 1)
    d = xy[i:j + 1] - xy[i]
    lat = -np.sin(yaw[i]) * d[:, 0] + np.cos(yaw[i]) * d[:, 1]
    return float(np.abs(lat).max()), float(np.degrees(abs(np.angle(np.exp(1j * (yaw[j] - yaw[i]))))))


def _vehicle_segment(job: tuple[str, str]) -> dict:
    split, seg = job
    D = load(split, seg)
    res = {"seg": seg, "split": split, "day": D["day"], "events": [], "base": []}
    if not D["box"].type.isin(LEAD_TYPES).any():
        return res
    ob, eg = vehicle_features(D)
    v, t, n, yaw, xy = eg["v"], eg["t"], eg["n"], eg["yaw"], eg["xy"]
    for tr, idx in ob.groupby("track").indices.items():
        g = ob.iloc[idx]
        fr = g.frame.to_numpy()
        byf = pd.Series(np.arange(len(g)), index=fr)
        # obstacle: a stopped vehicle in the straight lane of the current heading, the ego moving and not turning
        obst = (g.lane0 & (g.spd < 1.0) & (g.ttc0 <= TTR_S)).to_numpy() & (v[fr] >= MIN_SPEED)
        starts = []
        if obst.any():
            for i in fr[obst]:
                if lateral_dev(xy, yaw, int(i), int(RESP_POST * HZ))[1] < 15:
                    starts.append(("obstacle", int(i)))
                    break
        conf = g.conflict.to_numpy()
        if conf.any():
            f_c = fr[conf]
            for fc in f_c[np.r_[True, np.diff(f_c) > 2]]:
                r = g.iloc[byf[fc]]
                prev = g[(g.frame >= fc - 30) & (g.frame <= fc - 10)]
                if 45 <= r.rel <= 135:
                    cat = "crossing"
                elif r.rel > 135:
                    cat = "oncoming"
                elif len(prev) >= 10 and ((prev.L.abs() > LANE) & (prev.L.abs() <= 3 * LANE)).all():
                    cat = "cut_in"
                elif len(g[(g.frame >= fc - 30) & (g.frame <= fc) & g.in_lane]) >= 24 and \
                        g[(g.frame >= fc - 10) & (g.frame <= fc)].a_long.min() <= -3.0:
                    cat = "lead_brake"
                elif len(g[(g.frame >= fc - 30) & (g.frame <= fc) & g.in_lane]) >= 24:
                    cat = "lead_follow"
                else:
                    cat = "other"
                starts.append((cat, int(fc)))
        for cat, fc in starts:
            r = g.iloc[byf[fc]]
            rs = response(v, fc)
            lat, turn = lateral_dev(xy, yaw, fc, int(4 * HZ))
            between = bool(r.between0) if cat == "obstacle" else bool(r.between)
            resp = rs["responded"] or (cat == "obstacle" and lat >= 1.0 and turn < 15)
            res["events"].append({"seg": seg, "track": tr, "cat": cat, "f0": fc, "t0": t[fc], "v0": v[fc], "d": r.d, "x": r.x, "L": r.L,
                                  "y": r.y, "rel": r.rel, "ttc": r.ttc0 if cat == "obstacle" else r.ttc, "spd": r.spd, "a_long": r.a_long,
                                  "between": between, "seen": bool(r.seen), "lab_h": r.lab_h, "dv": rs["dv"], "decel": rs["responded"],
                                  "lat_dev": lat, "turn": turn, "responded": bool(resp), "post_s": rs["post_s"],
                                  "ok_window": bool(F0_RANGE[0] <= t[fc] <= F0_RANGE[1]), "ok_speed": bool(v[fc] >= MIN_SPEED)})
    cand = np.flatnonzero((v >= MIN_SPEED) & (np.arange(n) + RESP_POST * HZ < n) & (np.arange(n) >= 10))
    for i0 in cand[::20]:
        res["base"].append({"seg": seg, "frame": int(i0), "v0": v[i0], **response(v, int(i0))})
    return res


def vehicles(out: Path, workers: int):
    jobs = [(sp, p.stem) for sp in ("training", "validation") for p in sorted((V2 / sp / "lidar_box").glob("*.parquet"))]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_vehicle_segment, jobs, chunksize=4))
    ev = pd.DataFrame([e for r in res for e in r["events"]])
    base = pd.DataFrame([b for r in res for b in r["base"]])
    seg = pd.read_csv(out / "segments.csv").set_index("seg")
    ev["day"] = ev.seg.map(seg.day)
    ev["valid"] = (ev.cat.isin(VEH_CATS) & ev.ok_speed & ~ev.between & ev.seen & ev.responded & ev.ok_window & ev.day)
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "vehicle_events.csv", index=False)
    q66 = set(seg.index[seg.qualifies])
    ped = pd.read_csv(out / "survivors.csv")
    rows = []
    for cat in (*VEH_CATS, "lead_follow", "other"):
        e = ev[ev.cat == cat]
        m = e.ok_speed & e.ok_window & e.day
        rows.append({"category": cat, "events": len(e), "events_moving_window_day": int(m.sum()),
                     "no_vehicle_between": int((m & ~e.between).sum()), "seen": int((m & ~e.between & e.seen).sum()),
                     "responded": int((m & ~e.between & e.seen & e.responded).sum()),
                     "response_rate_given_seen": float(e[m & ~e.between & e.seen].responded.mean()) if (m & ~e.between & e.seen).any() else np.nan,
                     "segments_valid": e[e.valid].seg.nunique() if cat in VEH_CATS else np.nan,
                     "segments_valid_in_ped_pool66": len(set(e[e.valid].seg) & q66) if cat in VEH_CATS else np.nan})
    rows.append({"category": "any valid vehicle event", "events": int(ev.valid.sum()), "segments_valid": ev[ev.valid].seg.nunique(),
                 "segments_valid_in_ped_pool66": len(set(ev[ev.valid].seg) & q66)})
    rows.append({"category": "pedestrian (amendment 1, B strict)", "segments_valid": int(ped.in_B.sum())})
    rows.append({"category": "random moving frames (baseline)", "events": len(base), "response_rate_given_seen": float(base.responded.mean())})
    F = pd.DataFrame(rows)
    F.to_csv(out / "vehicle_funnel.csv", index=False)
    log.info("vehicle funnel:\n%s", F.to_string())


# ---------------------------------------------------------------- events over all segments
def registered_targets(seg: str, ped: pd.DataFrame, t: np.ndarray, v: np.ndarray) -> list[dict]:
    """Every registered target event of a segment (nq4_p3_select.select_scene rules (1)-(3), not only the crc32 pick)."""
    inside = ped[ped.corridor]
    out = []
    for tr, g in inside.groupby("track"):
        f = np.sort(g.frame.to_numpy())
        for f0 in f[np.r_[True, np.diff(f) > 1]]:
            if not F0_RANGE[0] <= t[f0] <= F0_RANGE[1] or v[f0] < MIN_SPEED:
                continue
            q = ped[(ped.track == tr) & (t[ped.frame] >= t[f0] - SEL.VIS_PRE) & (t[ped.frame] <= t[f0] + SEL.VIS_POST)]
            q = q[(q.x >= SEL.VIS_X[0]) & (q.x <= SEL.VIS_X[1]) & (np.abs(np.arctan2(q.y, q.x)) <= SEL.VIS_BEARING)]
            if SEL.longest_run(np.sort(t[q.frame.to_numpy()]), 0.1) < SEL.VIS_MIN_S:
                continue
            win = inside[(t[inside.frame] >= t[f0] - WIN_PRE) & (t[inside.frame] <= t[f0] + WIN_POST)]
            dl = sorted(map(str, win.track.unique()))
            if len(dl) > MAX_DEL:
                continue
            out.append({"track": tr, "f0": int(f0), "delete": dl, "crc": SEL.crc(f"{seg}/{tr}/{f0}")})
    return out


def item_readout(ped: pd.DataFrame, v: np.ndarray, f0: int, delete: list[str], frames: np.ndarray) -> dict:
    """Filter verdict of one pair (anchor f0, deletion set) over the given scored frames."""
    q = ped[ped.track.isin(delete) & ped.frame.isin(frames)]
    rf = np.sort(q[q.react].frame.unique())
    r = {"n_scored": len(frames), "n_react": len(rf),
         "n_conflict": int(q[q.conflict].frame.nunique()), "n_conflict_lead": int(q[q.conflict & q.lead].frame.nunique()),
         "n_conflict_unseen": int(q[q.conflict & ~q.lead & ~q.seen].frame.nunique()),
         "react_tracks": json.dumps(sorted(map(str, q[q.react].track.unique())))}
    if len(rf):
        r.update({"t_c_rel": (rf[0] - f0) / HZ, **response(v, int(rf[0]))})
    else:
        r.update({"t_c_rel": np.nan, "responded": False})
    r["survives"] = bool(len(rf)) and r["responded"]
    return r


def _segment(job: tuple[str, str]) -> dict:
    split, seg = job
    D = load(split, seg)
    res = {"seg": seg, "split": split, "day": D["day"], "n_frames": len(D["t"]), "A": [], "B": [], "ev": []}
    if not (D["box"].type == 2).any():
        return res
    ped, eg = frame_features(D)
    t, v, n = eg["t"], eg["v"], eg["n"]
    # label / projection sanity: FRONT labels over pedestrians whose 3D box centre is in the FRONT frustum at <= 40 m
    fr = ped[(ped.x > 4) & (ped.x <= 40) & (np.abs(np.arctan2(ped.y, ped.x)) <= np.radians(22))]
    res["frustum_labelled"] = float(fr.labelled.mean()) if len(fr) else np.nan
    # A: registered targets (all of them), readout on their scored 5 Hz frames inside the registered window
    A = []
    for tg in registered_targets(seg, ped, t, v):
        A.append({"seg": seg, "variant": "A", "track": tg["track"], "f0": tg["f0"], "t0": t[tg["f0"]], "v0": v[tg["f0"]],
                  "n_delete": len(tg["delete"]), "delete": json.dumps(tg["delete"]), "crc": tg["crc"],
                  **item_readout(ped, v, tg["f0"], tg["delete"], tg["f0"] + SCORED)})
    # B: re-anchored items, f0 := first react frame of a track's react run; registered rules applied at that anchor
    B = []
    corr = ped[ped.corridor]
    for tr, g in ped[ped.react].groupby("track"):
        f = np.sort(g.frame.to_numpy())
        for fc in f[np.r_[True, np.diff(f) > 2]]:
            win = corr[(t[corr.frame] >= t[fc] - WIN_PRE) & (t[corr.frame] <= t[fc] + WIN_POST)]
            dl = sorted(map(str, win.track.unique()))
            rs = response(v, int(fc))
            # stopped / creeping ego: was this pedestrian what held it? release gap = first frame the ego moves off
            # (>= 1 m/s) minus the first frame the pedestrian is out of the lane, both after the anchor
            lane = set(ped[(ped.track == tr) & ped.in_lane].frame)
            clear = next((f for f in range(int(fc), n) if f not in lane), n)
            go = next((f for f in range(int(fc), n) if v[f] >= 1.0), n)
            rs["release_gap"] = (go - clear) / HZ if go < n and clear < n else np.nan
            B.append({"seg": seg, "variant": "B", "track": tr, "f0": int(fc), "t0": t[fc], "v0": v[fc], "n_delete": len(dl),
                      "delete": json.dumps(dl), "deletable": tr in dl, "crc": SEL.crc(f"{seg}/{tr}/{fc}"),
                      "ok_window": bool(F0_RANGE[0] <= t[fc] <= F0_RANGE[1]), "ok_speed": bool(v[fc] >= MIN_SPEED),
                      "ok_crowd": len(dl) <= MAX_DEL, **rs,
                      "n_react": int(ped[ped.track.isin(dl) & ped.frame.isin(fc + SCORED) & ped.react].frame.nunique())})
    # evidence for the human-response rule: the same slow-down metric by pedestrian category, anchored at the first
    # frame of the category (conflict with / without a lead, corridor-only); and a random-frame baseline
    ev = []
    for tr, g in ped.groupby("track"):
        c = g[g.conflict]
        if len(c):
            i0 = int(c.frame.iat[0])
            cat = "conflict_lead" if bool(c.lead.iat[0]) else ("conflict_seen" if bool(c.seen.iat[0]) else "conflict_unseen")
        elif g.corridor.any():
            i0, cat = int(g[g.corridor].frame.iat[0]), "corridor_only"
        else:
            continue
        if v[i0] >= MIN_SPEED and i0 + RESP_POST * HZ < n:
            ev.append({"seg": seg, "track": tr, "cat": cat, "frame": i0, "v0": v[i0], **response(v, i0)})
    busy = np.zeros(n, bool)
    for f in ped[ped.corridor].frame.unique():
        busy[max(f - 30, 0):f + 31] = True
    cand = np.flatnonzero(~busy & (v >= MIN_SPEED) & (np.arange(n) + RESP_POST * HZ < n) & (np.arange(n) >= 10))
    for i0 in cand[::20]:
        ev.append({"seg": seg, "track": "", "cat": "no_ped_baseline", "frame": int(i0), "v0": v[i0], **response(v, int(i0))})
    res.update(A=A, B=B, ev=ev)
    return res


def events(out: Path, workers: int):
    jobs = [(sp, p.stem) for sp in ("training", "validation") for p in sorted((V2 / sp / "lidar_box").glob("*.parquet"))]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_segment, jobs, chunksize=4))
    sel = pd.read_csv(Path(__file__).resolve().parents[1] / "research/results/nq4/p3/selection_wod.csv")
    seg = pd.DataFrame([{k: v for k, v in r.items() if k not in ("A", "B", "ev")} for r in res])
    A = pd.DataFrame([a for r in res for a in r["A"]])
    B = pd.DataFrame([b for r in res for b in r["B"]])
    ev = pd.DataFrame([e for r in res for e in r["ev"]])
    # the recomputed registered targets must reproduce the registered selection (crc32 pick, f0, deletion set)
    q = sel[sel.n_targets > 0].set_index("name")
    pick = A.sort_values("crc").groupby("seg").head(1).set_index("seg")
    ntg = A.groupby("seg").size()
    bad = [s for s in q.index if s not in pick.index or pick.loc[s, "track"] != q.loc[s, "target_track"]
           or pick.loc[s, "f0"] != int(q.loc[s, "f0"]) or json.loads(pick.loc[s, "delete"]) != json.loads(q.loc[s, "delete_tracks"])
           or ntg.get(s, 0) != q.loc[s, "n_targets"]]
    assert not bad and len(pick) == len(q), f"registered targets not reproduced: {bad[:5]} ({len(pick)} vs {len(q)})"
    seg = seg.merge(sel[["name", "qualifies", "picked"]].rename(columns={"name": "seg"}), on="seg")
    out.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out / "segments.csv", index=False)
    A.to_csv(out / "items_A.csv", index=False)
    B.to_csv(out / "items_B.csv", index=False)
    ev.to_csv(out / "response_evidence.csv", index=False)
    log.info("segments %d, A items %d, B items %d, evidence rows %d; registered targets reproduced for %d segments",
             len(seg), len(A), len(B), len(ev), len(q))


# ---------------------------------------------------------------- the ten rendered scenes
def scenes(out: Path, exam_dir: Path, native: Path):
    from PIL import Image
    sc = json.loads((data_dir() / "runs/nq4/p3/scenes.json").read_text())["targets"]
    frames, peds, items = [], [], []
    for tg in sc:
        D = load("validation" if tg["split"] == "validation" else "training", tg["segment"])
        ped, eg = frame_features(D)
        f0, dl, fr = tg["f0"], tg["delete_tracks"], tg["f0"] + SCORED
        sd = data_dir() / "processed/nq4_p3/scenes" / tg["key"]
        q = ped[ped.track.isin(dl) & ped.frame.isin(fr)]
        # per deleted pedestrian and scored frame: flags, label size and box PSNR of x+ against the log (render scale 0.5)
        for f in fr:
            im = {w: np.asarray(Image.open(sd / w / "cams/front" / f"{2 * f:07d}.jpg"), np.float64) for w in ("real", "plus", "minus")}
            for r in q[q.frame == f].itertuples():
                row = {"scene": tg["key"], "frame": int(f), "t_rel": (f - f0) / HZ, "track": r.track, "S": r.S, "L": r.L, "d": r.d,
                       "ttr": r.ttr, "in_lane": r.in_lane, "lead": r.lead, "labelled": r.labelled, "lab_h": r.lab_h, "npts": r.npts,
                       "seen": r.seen, "conflict": r.conflict, "react": r.react, "corridor": r.corridor}
                bx = project_box(r, D["cal"], 0.5)
                if bx:
                    u0, v0, u1, v1 = int(bx[0]), int(bx[1]), int(np.ceil(bx[2])), int(np.ceil(bx[3]))
                    e = (im["plus"][v0:v1, u0:u1] - im["real"][v0:v1, u0:u1]) / 255
                    dm = np.abs(im["plus"][v0:v1, u0:u1] - im["minus"][v0:v1, u0:u1]).max(-1)
                    row.update(box_h=v1 - v0, box_px=(u1 - u0) * (v1 - v0),
                               psnr_box=float(-10 * np.log10(max((e ** 2).mean(), 1e-10))), del_frac=float((dm > 8).mean()))
                peds.append(row)
        rf = set(q[q.react].frame)
        qa = ped[ped.track.isin(dl)]
        qa = qa.assign(cls=object_classes(qa, eg["v"], eg["n"]))
        qs = qa[qa.frame.isin(fr)]
        fcat = frame_category(qs.cls, qs.frame)
        for f in fr:
            h = q[q.frame == f]
            frames.append({"scene": tg["key"], "frame": int(f), "sfx": f"{2 * f:07d}", "t_rel": round((f - f0) / HZ, 1), "v": eg["v"][f],
                           "react": f in rf, "conflict": bool(h.conflict.any()), "lead": bool((h.conflict & h.lead).any()),
                           "unseen": bool((h.conflict & ~h.lead & ~h.seen).any()),
                           # react = item (actions should differ); gray = a deleted pedestrian in the ego lane but not a
                           # react frame (farther than TTR_S, behind a lead or not yet seen); clean = every deleted
                           # pedestrian off the ego lane (actions should agree: a deletion null)
                           "cls": "react" if f in rf else "gray" if bool(h.in_lane.any() or h.conflict.any()) else "clean",
                           "cat": fcat.get(f, "clean")})
        items.append({"scene": tg["key"], "segment": tg["segment"], "f0": f0, "v0": tg["v0"], "n_delete": len(dl),
                      **item_readout(ped, eg["v"], f0, dl, fr)})
        # every 10 Hz frame's labels of the deleted tracks, for the review clips (scripts/p3/ds.py clip)
        cl = {str(f): {r.track: {"d": r.d, "L": r.L, "ttr": r.ttr, "in_lane": bool(r.in_lane), "lead": bool(r.lead),
                                 "lab_h": None if r.lab_h != r.lab_h else float(r.lab_h), "seen": bool(r.seen), "react": bool(r.react)}
                       for r in g.itertuples()} for f, g in ped[ped.track.isin(dl)].groupby("frame")}
        (out / "clip_labels").mkdir(parents=True, exist_ok=True)
        (out / "clip_labels" / f"{tg['key']}.json").write_text(json.dumps({"v": eg["v"].round(3).tolist(), "frames": cl}))
    P, F, I = pd.DataFrame(peds), pd.DataFrame(frames), pd.DataFrame(items)
    pr = P[P.react].groupby("scene").psnr_box.mean()
    I["psnr_react"] = I.scene.map(pr)
    I["psnr_ok"] = I.psnr_react >= PSNR_MIN
    I["survives_all"] = I.survives & I.psnr_ok
    # readouts (registered flip rule, I3 tau) on frame subsets; ridge_late is the registered examinee, native descriptive
    key = ["base_id", "sfx"]
    lab = F.rename(columns={"scene": "base_id"})[key + ["react", "conflict", "lead", "unseen", "cls", "cat"]]
    lab = lab.merge(I.rename(columns={"scene": "base_id"})[["base_id", "survives", "survives_all"]], on="base_id")
    taus = pd.read_csv(data_dir() / "runs/elicitation/i3-exam/20260926-012841/flip_rates.csv").query("scope == 'pooled'").set_index("examinee").tau_model
    sources = [("ridge_late", pd.read_parquet(exam_dir / "frames_scored.parquet"), "ridge_late op-{m} temporal")]
    if native.exists():
        sources.append(("native plan (borrowed tau)", pd.read_parquet(native), "{m}"))
    rows = []
    for src, fs, pat in sources:
        fs = fs.merge(lab, on=key)
        assert len(fs) == len(F), (src, len(fs), len(F))
        subsets = {"all frames (registered)": fs.react | ~fs.react,
                   "react frames, surviving items": fs.react & fs.survives,
                   "react frames, surviving items, psnr ok": fs.react & fs.survives_all,
                   "react frames, all items": fs.react,
                   "non-react frames, surviving items": ~fs.react & fs.survives,
                   "non-react frames, all items": ~fs.react,
                   "clean frames (every deleted pedestrian off the ego lane)": fs.cls == "clean",
                   "gray frames (a deleted pedestrian in the lane, not react)": fs.cls == "gray",
                   **{f"category {c}": fs.cat == c for c in CATS},
                   "conflict frames with a lead": fs.lead,
                   "conflict frames, pedestrian not yet seen": fs.unseen}
        for m in ("cinque", "lebowski"):
            col, tau = pat.format(m=m), float(taus[f"ridge_late op-{m} temporal"])
            for name, msk in subsets.items():
                g = fs[msk.to_numpy()]
                dn, dp = g[f"{col}|null_d"].abs(), g[f"{col}|pair_d"]
                rows.append({"source": src, "model": m, "subset": name, "frames": len(g), "scenes": g.base_id.nunique(),
                             "null_ff": int(((dn >= tau) & (dn > 0)).sum()), "pair_moved": int(((dp.abs() >= tau) & (dp != 0)).sum()),
                             "pair_slower": int((dp <= -tau).sum()), "pair_d_median": float(dp.median()) if len(g) else np.nan})
    R = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "scenes_ped_frames.csv", index=False)
    F.to_csv(out / "scenes_frames.csv", index=False)
    I.to_csv(out / "scenes_items.csv", index=False)
    R.to_csv(out / "scenes_readout.csv", index=False)
    log.info("items:\n%s", I.drop(columns=["segment"]).to_string())
    log.info("readout:\n%s", R.to_string())


# ---------------------------------------------------------------- summary tables (CPU, reads the events outputs)
def _survivors(d: Path, seg: pd.DataFrame) -> dict:
    q66 = set(seg.index[seg.qualifies])
    A, B = pd.read_csv(d / "items_A.csv"), pd.read_csv(d / "items_B.csv")
    B["day"] = B.seg.map(seg.day)
    core = B.day & B.ok_window & B.deletable
    stop_bind = (B.v0 < MIN_SPEED) & B.get("release_gap", pd.Series(np.nan, index=B.index)).between(0, 2)
    sets = {"A: registered anchor, react frame + responded (of the 66)": A[A.survives & A.seg.isin(q66)].seg,
            "B: re-anchored, all registered rules + responded": B[core & B.ok_speed & B.ok_crowd & B.responded].seg,
            "B without the speed rule, stopped items need release gap in [0, 2] s": B[core & B.ok_crowd & ((B.ok_speed & B.responded) | stop_bind)].seg,
            "B without the crowd rule": B[core & B.ok_speed & B.responded].seg,
            "B without speed and crowd rules (no response rule)": B[core].seg,
            "B ceiling: any react event, daytime": B[B.day].seg}
    return {k: sorted(set(v)) for k, v in sets.items()}


def summary(out: Path, sens: list[Path]):
    seg = pd.read_csv(out / "segments.csv").set_index("seg")
    order = seg[seg.qualifies].sort_values("seg", key=lambda s: s.map(SEL.crc)).index.tolist()     # pool scene k
    rows, ids = [], {}
    for d in [out, *sens]:
        tag = "base" if d == out else d.name
        for k, v in _survivors(d, seg).items():
            rows.append({"config": tag, "rule set": k, "segments": len(v), "in_qualifying66": sum(s in order for s in v)})
            if tag == "base":
                ids[k] = v
    pd.DataFrame(rows).to_csv(out / "funnel.csv", index=False)
    base = sorted(set(ids["A: registered anchor, react frame + responded (of the 66)"]) |
                  set(ids["B: re-anchored, all registered rules + responded"]))
    pd.DataFrame({"seg": base, "pool_scene": [order.index(s) if s in order else -1 for s in base],
                  "in_A": [s in ids["A: registered anchor, react frame + responded (of the 66)"] for s in base],
                  "in_B": [s in ids["B: re-anchored, all registered rules + responded"] for s in base]}).sort_values("pool_scene").to_csv(out / "survivors.csv", index=False)
    ev = pd.read_csv(out / "response_evidence.csv")
    ev.groupby("cat").agg(n=("responded", "size"), responded=("responded", "mean"), dv_median=("dv", "median"),
                          v0_median=("v0", "median")).to_csv(out / "response_by_category.csv")
    log.info("funnel:\n%s", pd.DataFrame(rows).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("events", "scenes", "summary", "vehicles"))
    ap.add_argument("--out", type=Path, default=data_dir() / "runs/nq4/p3/filter")
    ap.add_argument("--workers", type=int, default=0, help="events: worker processes (default min(8, n_cpus()))")
    ap.add_argument("--exam-dir", type=Path, default=data_dir() / "runs/nq4/p3-exam/20260927-232404")
    ap.add_argument("--native", type=Path, default=data_dir() / "runs/nq4/p3/formal10/native/native_frames.parquet")
    ap.add_argument("--sens", type=Path, nargs="*", default=[], help="summary: sensitivity output dirs")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="sensitivity runs only: override a rule constant (LANE, TTR_S, SEEN_S, H_MIN, ...)")
    a = ap.parse_args()
    for kv in a.set:                   # forked workers inherit the overridden module globals
        k, v = kv.split("=")
        assert k in globals() and isinstance(globals()[k], float), k
        globals()[k] = float(v)
    if a.cmd == "events":
        events(a.out, a.workers or min(8, n_cpus()))
    elif a.cmd == "vehicles":
        vehicles(a.out, a.workers or min(8, n_cpus()))
    elif a.cmd == "scenes":
        scenes(a.out, a.exam_dir, a.native)
    else:
        summary(a.out, a.sens)


if __name__ == "__main__":
    main()
