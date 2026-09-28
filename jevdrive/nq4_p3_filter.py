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
                                                                ("hd", "box.heading"))}}).dropna(subset=["frame"])
    box["frame"] = box.frame.astype(int)
    cb = pd.read_parquet(V2 / split / "camera_box" / f"{seg}.parquet")
    cb = cb[(cb["key.camera_name"] == 1) & (cb[C_ + "type"] == 2)]
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
    return {"seg": seg, "split": split, "t": (ts - ts[0]) / 1e6, "T": T, "box": box, "lab": lab, "cal": cal,
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
    return S, L


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
        S, L = project(P, cum, W, Se[i] - 10, Se[i] + FRONT + REACH + 60)
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
    ped["react"] = ped.conflict & ~ped.lead & ped.seen
    return ped, {"v": v, "t": t, "n": n}


def response(v: np.ndarray, i: int) -> dict:
    """Logged slow-down around frame i: max speed over [i - 1 s, i] minus min speed over [i, i + 3 s]."""
    a, b = max(i - int(RESP_PRE * HZ), 0), min(i + int(RESP_POST * HZ), len(v) - 1)
    vp, vm = float(v[a:i + 1].max()), float(v[i:b + 1].min())
    return {"v_pre": vp, "v_min_post": vm, "dv": vp - vm, "post_s": (b - i) / HZ,
            "responded": bool(vp - vm >= max(DV_MIN, DV_FRAC * vp))}


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
        for f in fr:
            h = q[q.frame == f]
            frames.append({"scene": tg["key"], "frame": int(f), "sfx": f"{2 * f:07d}", "t_rel": round((f - f0) / HZ, 1), "v": eg["v"][f],
                           "react": f in rf, "conflict": bool(h.conflict.any()), "lead": bool((h.conflict & h.lead).any()),
                           "unseen": bool((h.conflict & ~h.lead & ~h.seen).any())})
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
    lab = F.rename(columns={"scene": "base_id"})[key + ["react", "conflict", "lead", "unseen"]]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("events", "scenes"))
    ap.add_argument("--out", type=Path, default=data_dir() / "runs/nq4/p3/filter")
    ap.add_argument("--workers", type=int, default=min(8, n_cpus()))
    ap.add_argument("--exam-dir", type=Path, default=data_dir() / "runs/nq4/p3-exam/20260927-232404")
    ap.add_argument("--native", type=Path, default=data_dir() / "runs/nq4/p3/formal10/native/native_frames.parquet")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="sensitivity runs only: override a rule constant (LANE, TTR_S, SEEN_S, H_MIN, ...)")
    a = ap.parse_args()
    for kv in a.set:                   # forked workers inherit the overridden module globals
        k, v = kv.split("=")
        assert k in globals() and isinstance(globals()[k], float), k
        globals()[k] = float(v)
    events(a.out, a.workers) if a.cmd == "events" else scenes(a.out, a.exam_dir, a.native)


if __name__ == "__main__":
    main()
