"""I3: real-appearance counterfactual pairs rendered from HUGSIM 3DGS scenes along the LOGGED ego trajectory
(todos/2026-09-25-reactivity-program/i3-hugsim-pairs.md).

Every world of a scene is the same open-loop camera stream (the recorded ego poses, no controller); the worlds
differ only in one inserted 3DRealCar actor:
  minus     no actor (the plain scene)
  static    a stopped car in the ego lane where the logged ego reaches at t_c (HUGSIM's "medium" scenario type)
  cutin     a car in the adjacent lane, 10 m ahead, that HUGSIM's AttackPlanner steers at the logged ego from t_e on
  oncoming  a car in the ego lane facing the ego, 3 m/s, AttackPlanner from t_e on (HUGSIM's "extreme" type)
  null      the cut-in car and start, but it keeps its lane at the ego's logged speed: visible, never in the way

This module is imported by the HUGSIM venv (scripts/hugsim/pairs_render.py: actor trajectories, rendering) and by
the jevdrive venv (labels, the P5-shaped index, figures): numpy only at import time.

  select    scene table: every scene on disk with a valid t_c (see `pick_tc`)
  index     per-scene render output -> processed/hugsim_pairs/{index.parquet, past.npy, future.npy, obs.parquet,
            null.parquet, pairs.csv}, the layout of processed/carla_p5 (jevdrive.p5_pairs)
  fig       the validation figure (x+ / x- / |diff| per scene)
"""
import json
import os
import pickle
import zlib
from pathlib import Path

import numpy as np

from .common import data_dir

DATASETS = ("nuscenes", "kitti360", "waymo", "pandaset")
FRONT = {"nuscenes": "CAM_FRONT", "kitti360": "cam_0", "waymo": "cam_1", "pandaset": "front_camera"}
IDX_DT = {"kitti360": 0.1, "waymo": 0.1}          # their meta timestamps are 10 Hz frame indices, not seconds
CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")   # HUGSIM rig names in configs/sim/<ds>_camera.yaml
CAM_KEYS = ("front", "front_left", "front_right")          # P5 / waymo.CAMS order
WORLDS = ("minus", "static", "cutin", "oncoming", "null")
PLUS = ("static", "cutin", "oncoming")

# ---- pre-registered parameters (the sub-doc's pre-registration section is the source of truth)
RATE_DT = 0.2            # camera frame period (5 Hz): P5 clip spacing and openpilot's context step
SIM_DT = 0.05            # actor / label time grid
T_LEAD = 6.0             # render window starts t_c - T_LEAD
T_ENGAGE = 4.0           # AttackPlanner engages at t_e = t_c - T_ENGAGE
T_POST = 1.0             # render window ends t_c + T_POST (capped at the log end)
FUT_NEED = 3.0           # an observation frame needs this much logged future (stop label horizon)
V_MIN = 3.0              # logged speed over [t_c - T_ENGAGE, t_c] must stay above this (m/s)
LANE_W = 3.5             # adjacent-lane offset (m)
CUTIN_AHEAD = 10.0       # cut-in / null car centre ahead of the ego camera (m)
ONCOMING_V = 3.0         # HUGSIM extreme scenarios' attacker speed (m/s)
ATTACK_DT, ATTACK_FREQ = 0.1, 5    # AttackPlanner step and replanning period (steps)
EGO_FRONT, EGO_REAR, EGO_HALF_W = 2.0, 2.0, 1.2   # HUGSIM's 3.0 x 1.6 m ego box grown by 0.5 m / 0.4 m
HORIZON = 4.0            # label: conflict search horizon (s)
A_BRAKE, A_MAX = 3.0, 8.0          # label: stopping profile deceleration, hardest allowed deceleration (m/s^2)
VIS_PX = 100             # an actor is visible in a frame with this many changed pixels over the three cameras
DIFF_THR = 8             # a pixel "changed" when any channel differs by more than this (0-255)
ROAD_R = 0.5             # road coverage: footprint sample within this of a road Gaussian (m)
ROAD_MEAN, ROAD_MIN = 0.7, 0.4     # valid actor path: mean / worst-frame road coverage before the conflict
OBST_PTS = 100           # invalid actor path: more obstacle Gaussians inside its box (HUGSIM's collision count)


def root(*parts) -> Path:
    p = data_dir() / "processed" / os.environ.get("I3_SET", "hugsim_pairs") / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def scenes_dir() -> Path:
    return data_dir() / "datasets" / "hugsim" / "scenes"


def scene_key(ds: str, scene: str) -> str:
    return f"{ds}-{scene}"


def unpack(ds: str, scene: str) -> Path:
    """The scene dir, unpacked from its zip on first use (both release layouts, like scripts/hugsim/zs_run.py)."""
    import shutil
    import zipfile
    d = scenes_dir() / ds / scene
    if not (d / "scene.pth").exists():
        with zipfile.ZipFile(scenes_dir() / ds / f"{scene}.zip") as z:
            z.extractall(scenes_dir() / ds)
        nested = scenes_dir() / ds / ds / scene
        if nested.exists() and not d.exists():
            shutil.move(str(nested), str(d))
    return d


# ---------------------------------------------------------------- logged ego trajectory

def left_of(th):
    """Unit vector to the left of heading th, in the (a, b) = (world x, world z) plane (HUGSIM: forward =
    (sin th, cos th), th positive turning right)."""
    return np.stack([-np.cos(th), np.sin(th)], -1)


def fwd_of(th):
    return np.stack([np.sin(th), np.cos(th)], -1)


class Logged:
    """The recorded front-camera trajectory as HUGSIM's env poses it: (a, b, th), th the yaw about +y."""

    def __init__(self, scene_dir: Path, ds: str):
        m = json.loads((scene_dir / "meta_data.json").read_text())
        fr = sorted((f for f in m["frames"] if f["rgb_path"].split("/")[-2] == FRONT[ds]), key=lambda f: f["timestamp"])
        ts = np.array([f["timestamp"] for f in fr], float)
        self.t = (ts - ts[0]) * IDX_DT.get(ds, 1.0)
        P = np.array([f["camtoworld"] for f in fr], float)
        with open(scene_dir / "ground_param.pkl", "rb") as fh:
            cp, _, cmds = pickle.load(fh)
        if len(cp) != len(P) or np.abs(cp - P).max() > 1e-3:
            raise ValueError("ground_param route disagrees with the recorded front-camera poses")
        self.cmd = np.asarray(cmds)
        self.a, self.b = P[:, 0, 3], P[:, 2, 3]
        self.th = np.unwrap(np.arctan2(P[:, 0, 2], P[:, 2, 2]))
        self.s = np.r_[0.0, np.cumsum(np.hypot(np.diff(self.a), np.diff(self.b)))]
        self.dur = float(self.t[-1])
        self.v_end = (self.s[-1] - np.interp(self.dur - 0.5, self.t, self.s)) / 0.5
        # the path by arc length (standing samples dropped), extended 300 m straight at both ends
        keep = np.r_[True, np.diff(self.s) > 1e-3]
        ps, pa, pb, pth = self.s[keep], self.a[keep], self.b[keep], self.th[keep]
        ext = 300.0
        f0, f1 = fwd_of(pth[0]), fwd_of(pth[-1])
        self.ps = np.r_[ps[0] - ext, ps, ps[-1] + ext]
        self.pa = np.r_[pa[0] - ext * f0[0], pa, pa[-1] + ext * f1[0]]
        self.pb = np.r_[pb[0] - ext * f0[1], pb, pb[-1] + ext * f1[1]]
        self.pth = np.r_[pth[0], pth, pth[-1]]

    def s_at(self, t):
        t = np.asarray(t, float)
        return np.where(t <= self.dur, np.interp(t, self.t, self.s), self.s[-1] + self.v_end * (t - self.dur))

    def v_at(self, t, h: float = 0.25):
        """Speed as the centred arc-length difference over +-h/2 (smooths the 10-12 Hz pose jitter)."""
        t = np.asarray(t, float)
        lo, hi = np.maximum(t - h / 2, 0.0), t + h / 2
        return (self.s_at(hi) - self.s_at(lo)) / (hi - lo)

    def pose_at(self, t):
        """(a, b, th) at times t; beyond the log end, straight on at the last speed."""
        t = np.asarray(t, float)
        a, b, th = np.interp(t, self.t, self.a), np.interp(t, self.t, self.b), np.interp(t, self.t, self.th)
        late = t > self.dur
        if late.any():
            d = self.v_end * (t[late] - self.dur)
            f = fwd_of(self.th[-1])
            a[late], b[late], th[late] = self.a[-1] + d * f[0], self.b[-1] + d * f[1], self.th[-1]
        return a, b, th

    def path(self, s, d=0.0):
        """Point at arc length s on the path, offset d to the left; (a, b, th)."""
        s = np.asarray(s, float)
        a, b, th = np.interp(s, self.ps, self.pa), np.interp(s, self.ps, self.pb), np.interp(s, self.ps, self.pth)
        o = left_of(th) * np.asarray(d, float)[..., None]
        return a + o[..., 0], b + o[..., 1], th

    def command_at(self, t):
        return self.cmd[np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1)]


def pick_tc(L: Logged):
    """Target conflict time t_c: the earliest t on the 0.2 s grid with a full render lead (t - T_LEAD >= 0), enough
    logged future for the labels of the last observation frame (t + FUT_NEED <= log end) and the logged ego moving
    (>= V_MIN m/s) over the whole attack phase [t - T_ENGAGE, t]. None when the scene has no such t."""
    for tc in np.arange(T_LEAD, L.dur - FUT_NEED + 1e-6, RATE_DT):
        if L.v_at(np.arange(tc - T_ENGAGE, tc + 1e-6, 0.25)).min() >= V_MIN:
            return round(float(tc), 2)
    return None


def select(out: Path | None = None):
    """Every scene on disk (zip) with its t_c; the pair set is all scenes with a t_c."""
    import pandas as pd
    rows = []
    for ds in DATASETS:
        for z in sorted((scenes_dir() / ds).glob("*.zip")):
            d = unpack(ds, z.stem)
            row = {"key": scene_key(ds, z.stem), "dataset": ds, "scene": z.stem}
            try:
                L = Logged(d, ds)
            except ValueError as e:
                rows.append({**row, "t_c": None, "excluded": str(e)})
                continue
            tc = pick_tc(L)
            rows.append({**row, "dur": round(L.dur, 2), "path_m": round(float(L.s[-1]), 1),
                         "v_mean": round(float(L.s[-1] / L.dur), 2), "t_c": tc,
                         "excluded": None if tc is not None else "no_tc"})
    t = pd.DataFrame(rows)
    t.to_csv(out or root() / "scenes.csv", index=False)
    return t


# ---------------------------------------------------------------- actors

def asset_for(key: str, family: str, assets: list[str]) -> str:
    """Deterministic 3DRealCar asset per scene and family (the null world reuses the cut-in car)."""
    fam = "cutin" if family == "null" else family
    return assets[zlib.crc32(f"{key}/{fam}".encode()) % len(assets)]


def scenario_assets() -> list[str]:
    """The 3DRealCar assets that HUGSIM's released scenarios use (105), sorted."""
    import yaml
    out = set()
    for f in (data_dir() / "datasets" / "hugsim" / "scenarios").glob("*/*.yaml"):
        for p in yaml.safe_load(f.read_text()).get("plan_list") or []:
            out.add(str(p[5]).split("/")[0])
    have = {p.name for p in (data_dir() / "datasets" / "hugsim" / "3DRealCar").iterdir()}
    return sorted(out & have)


def window(tc: float, L: Logged):
    """Render times (5 Hz) and the actor / label time grid (0.05 s, running HORIZON + 1 s past the window)."""
    w0, w1 = tc - T_LEAD, min(tc + T_POST, L.dur)
    t_render = np.round(w0 + RATE_DT * np.arange(int(round((w1 - w0) / RATE_DT)) + 1), 4)
    t_sim = np.round(w0 + SIM_DT * np.arange(int(round((tc + HORIZON + 1.0 - w0) / SIM_DT)) + 1), 4)
    return t_render, t_sim


def _attack(state0, t_e, t_end, L: Logged):
    """HUGSIM's AttackPlanner from state0 = (a, b, th, v) at t_e, attacking the LOGGED ego future (the shipped
    planner attacks a constant-velocity extrapolation; here the ego's future is known). Its state convention:
    [a, b, yaw, v] with yaw = -th. Returns times and (a, b, th, v) at ATTACK_DT."""
    import torch
    from sim.utils.agent_controller import AttackPlanner
    ap = AttackPlanner(pred_steps=20, ATTACK_FREQ=ATTACK_FREQ)
    st = torch.tensor([state0[0], state0[1], -state0[2], state0[3]], dtype=torch.float64)
    far = torch.full((1, 20, 4), 1e6, dtype=torch.float64)
    ts = np.round(t_e + ATTACK_DT * np.arange(int(round((t_end - t_e) / ATTACK_DT)) + 1), 4)
    out = [st.numpy().copy()]
    for i, t in enumerate(ts[:-1]):
        ea, eb, eth = L.pose_at(t + ATTACK_DT * np.arange(20))
        ego = torch.tensor(np.stack([ea, eb, -eth, L.v_at(t + ATTACK_DT * np.arange(20))], -1))
        nxt = ap.update(state=st.float(), unified_map=None, dt=ATTACK_DT, neighbors=far.float(),
                        attacked_states=ego.float(), new_plan=(i % ATTACK_FREQ == 0))
        st = nxt.double().clone()
        out.append(st.numpy().copy())
    o = np.array(out)
    th = np.unwrap(-o[:, 2])
    return ts, np.stack([o[:, 0], o[:, 1], th, o[:, 3]], -1)


def actor_track(family: str, side: float, tc: float, t_sim: np.ndarray, L: Logged, length: float):
    """(a, b, th, v) of the family's actor on t_sim; side = +1 left / -1 right lane for cutin and null."""
    te = tc - T_ENGAGE
    ego_s = L.s_at(t_sim)
    if family == "static":
        s = L.s_at(tc) + EGO_FRONT + length / 2
        a, b, th = L.path(np.full_like(t_sim, s))
        return np.stack([a, b, th, np.zeros_like(t_sim)], -1)
    if family in ("cutin", "null"):
        a, b, th = L.path(ego_s + CUTIN_AHEAD, side * LANE_W)
        pre = np.stack([a, b, th, L.v_at(t_sim)], -1)
    elif family == "oncoming":
        s_meet = L.s_at(tc) + EGO_FRONT + length / 2
        s = s_meet + ONCOMING_V * (tc - t_sim)
        a, b, th = L.path(s)
        pre = np.stack([a, b, th + np.pi, np.full_like(t_sim, ONCOMING_V)], -1)
    else:
        raise ValueError(family)
    if family == "null":
        return pre
    k = int(np.searchsorted(t_sim, te - 1e-6))
    ta, xa = _attack(pre[k], t_sim[k], t_sim[-1], L)
    post = np.stack([np.interp(t_sim[k:], ta, xa[:, j]) for j in range(4)], -1)
    return np.concatenate([pre[:k], post])


def b2w(a, b, th, y):
    """Actor body-to-world, as sim.utils.plan.planner builds it: rotation about y by (-yaw - pi/2), yaw = -th;
    3DRealCar assets have x along the car, y down with the wheels at y = 0."""
    c, s = np.cos(th - np.pi / 2), np.sin(th - np.pi / 2)
    M = np.eye(4)
    M[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    M[:3, 3] = a, y, b
    return M


# ---------------------------------------------------------------- geometry for labels

def _box(cx, cy, th, front, rear, half_w):
    f, l = fwd_of(th), left_of(th)
    c = np.array([cx, cy])
    return np.array([c + front * f + half_w * l, c + front * f - half_w * l, c - rear * f - half_w * l,
                     c - rear * f + half_w * l])


def boxes_overlap(P, Q) -> bool:
    """Separating-axis test for two convex quads (4, 2)."""
    for R in (P, Q):
        for i in range(4):
            e = R[(i + 1) % 4] - R[i]
            n = np.array([-e[1], e[0]])
            p, q = P @ n, Q @ n
            if p.max() < q.min() or q.max() < p.min():
                return False
    return True


def conflicts(L: Logged, t_sim, track, wl) -> np.ndarray:
    """Per t_sim step: the logged ego box overlaps the actor box at the same time."""
    ea, eb, eth = L.pose_at(t_sim)
    w, l = wl
    return np.array([boxes_overlap(_box(ea[i], eb[i], eth[i], EGO_FRONT, EGO_REAR, EGO_HALF_W),
                                   _box(track[i, 0], track[i, 1], track[i, 2], l / 2, l / 2, w / 2))
                     for i in range(len(t_sim))])


def cv_conflict(L: Logged, t: float, state, wl) -> float | None:
    """The rule's hazard test at time t: the actor extrapolated at constant velocity from its current state
    (a, b, th, v) against the logged ego over [t, t + HORIZON]; the first time the two boxes overlap, else None.
    Only what a camera could see at t enters (the actor's current pose and velocity), not its scripted future."""
    tt = t + SIM_DT * np.arange(int(round(HORIZON / SIM_DT)) + 1)
    a, b, th, v = state[:4]
    f = fwd_of(th)
    d = v * (tt - t)
    track = np.stack([a + d * f[0], b + d * f[1], np.full_like(tt, th)], -1)
    hit = np.flatnonzero(conflicts(L, tt, track, wl))
    return float(tt[hit[0]]) if len(hit) else None


def rule_speed(L: Logged, t0: float, t_conf: float | None, n: int = 101):
    """The rule expert's speed on t0 + SIM_DT * [0, n): the logged speed, capped by a stopping profile (A_BRAKE)
    that ends where the logged ego box first touches the actor (t_conf), never braking harder than A_MAX.
    Without a conflict it is the logged speed."""
    tt = t0 + SIM_DT * np.arange(n)
    vlog = L.v_at(tt)
    if t_conf is None:
        return tt, vlog, L.s_at(tt) - L.s_at(t0)
    s_stop = L.s_at(t_conf) - L.s_at(t0)
    v, s = np.empty(n), np.empty(n)
    v[0], s[0] = min(vlog[0], np.sqrt(2 * A_BRAKE * max(s_stop, 0.0))), 0.0
    for i in range(1, n):
        s[i] = s[i - 1] + v[i - 1] * SIM_DT
        cap = np.sqrt(2 * A_BRAKE * max(s_stop - s[i], 0.0))
        v[i] = max(min(vlog[i], cap), v[i - 1] - A_MAX * SIM_DT, 0.0)
        v[i] = min(v[i], vlog[i])
    return tt, v, s


def ego_frame(L: Logged, t_ref: float, pts_ab):
    """(a, b) points -> the ego frame at t_ref: x forward, y left."""
    a, b, th = L.pose_at(np.array([t_ref]))
    d = np.asarray(pts_ab) - np.array([a[0], b[0]])
    return np.stack([d @ fwd_of(th[0]), d @ left_of(th[0])], -1)


def past_future(L: Logged, t: float, fut_s=None):
    """P4 / WOD-E2E-shaped ego rows at time t (front-camera origin, x forward, y left): past (16, 6) at 0.25 s,
    [x, y, vx, vy, dvx, dvy] with dv the velocity change per 0.25 s step (P4's convention, including WOD's repeat
    of the t0 sample in the t - 0.25 slot); future (20, 2) at 0.25 s. fut_s: optional arc-length profile
    (times, s) replacing the logged one (the rule expert's x+ future); future beyond the log end is NaN."""
    tp = t + 0.25 * np.arange(-15, 1)
    tpc = np.maximum(tp, 0.0)
    pa, pb, _ = L.pose_at(tpc)
    p = ego_frame(L, t, np.stack([pa, pb], -1))
    h = 0.05
    va, vb, _ = L.pose_at(np.minimum(tpc + h, L.dur))
    ua, ub, _ = L.pose_at(np.maximum(tpc - h, 0.0))
    dt = np.minimum(tpc + h, L.dur) - np.maximum(tpc - h, 0.0)
    vel = np.stack([(va - ua) / dt, (vb - ub) / dt], -1)
    _, _, th0 = L.pose_at(np.array([t]))
    vv = np.stack([vel @ fwd_of(th0[0]), vel @ left_of(th0[0])], -1)
    aa = vv - np.r_[vv[:1], vv[:-1]]
    pad = tp < 0
    vv[pad], aa[pad] = 0.0, 0.0
    vv[-2], aa[-2] = vv[-1], aa[-1]
    past = np.concatenate([p, vv, aa], -1)
    tf = t + 0.25 * np.arange(1, 21)
    if fut_s is None:
        fa, fb, _ = L.pose_at(tf)
    else:
        s = L.s_at(t) + np.interp(tf, fut_s[0], fut_s[1])
        fa, fb, _ = L.path(s)
    fut = ego_frame(L, t, np.stack([fa, fb], -1))
    fut[tf > L.dur + 1e-6] = np.nan
    return past, fut


def v2(fut):
    """Speed 2 s ahead, P5's definition: the 1.75 -> 2.0 s displacement over 0.25 s."""
    return float(np.linalg.norm(fut[7] - fut[6]) / 0.25)


def stop3(v0, fut):
    pts = np.concatenate([np.zeros((1, 2)), fut[:12]])
    sp = np.linalg.norm(np.diff(pts, axis=0), axis=1) / 0.25
    return bool(min(sp.min(), v0) < 0.5)


# ---------------------------------------------------------------- index (jevdrive venv)

INTENT = {2: 1, 1: 2, 0: 3}   # HUGSIM command (0 right, 1 left, 2 straight) -> WOD intent (1 straight, 2 left, 3 right)


def _scene_rows(sdir: Path):
    """Index rows, pair frames and null frames of one rendered scene."""
    meta = json.loads((sdir / "meta.json").read_text())
    ds, scene, key, tc = meta["dataset"], meta["scene"], meta["key"], meta["t_c"]
    L = Logged(unpack(ds, scene), ds)
    tr = np.array(meta["t_render"])
    tsim = np.array(meta["t_sim"])
    worlds = {w: m for w, m in meta["worlds"].items() if m.get("rendered")}
    rows, past, fut = [], [], []
    labels = {}          # (world, k) -> dict
    for w, m in worlds.items():
        track = np.array(m["track"]) if w != "minus" else None
        conf = np.array(m["conflict"], bool) if w != "minus" else None
        vis = np.array(m.get("vis_px", [[0] * 3] * len(tr)))
        for k in range(3, len(tr)):
            t = float(tr[k])
            t_conf = t_true = None
            if conf is not None:
                i = int(np.searchsorted(tsim, t - 1e-6))
                t_conf = cv_conflict(L, t, track[i], (m["wlh"][0], m["wlh"][1]))
                hit = np.flatnonzero(conf & (tsim >= t - 1e-6) & (tsim <= t + HORIZON + 1e-6))
                t_true = float(tsim[hit[0]]) if len(hit) else None
            fs = None
            if w in PLUS and t_conf is not None:
                tt, _, ss = rule_speed(L, t, t_conf)
                fs = (tt, ss)
            p, f = past_future(L, t, fs)
            fn = f"{key}-{w}-{4 * k:07d}"
            files = [str(sdir / w / "cams" / c / f"{4 * j:07d}.jpg") for c in CAM_KEYS for j in range(k - 3, k + 1)]
            rows.append({"frame_name": fn, "route_id": f"{key}-{w}", "town": ds, "frame": 4 * k, "cam_index": k,
                         "t": t, "files": files, "base_id": key, "source": "hugsim", "family": w,
                         "world": "minus" if w == "minus" else "null" if w == "null" else "plus", "seed": 0,
                         "intent": INTENT[int(L.command_at(t))], "actor_px": int(vis[k].sum()),
                         "past_padded_s": float(max(0.0, 3.75 - t) // 0.25 * 0.25)})
            past.append(p)
            fut.append(f)
            labels[(w, k)] = {"fn": fn, "t_conf": t_conf, "t_true": t_true, "v0": float(np.linalg.norm(p[-1, 2:4])),
                              "v2": v2(f), "stop": stop3(float(np.linalg.norm(p[-1, 2:4])), f),
                              "px": int(vis[k].sum()), "fut_ok": bool(t + FUT_NEED <= L.dur + 1e-6)}
    frames, nulls, pairs = [], [], []
    for w in PLUS:
        if w not in worlds:
            pairs.append({"base_id": key, "town": ds, "family": w, "seed": 0, "plus": f"{key}-{w}",
                          "minus": f"{key}-minus", "reason": worlds_reason(meta, w), "n_obs": 0})
            continue
        m = worlds[w]
        vis_k = [k for k in range(3, len(tr)) if labels[(w, k)]["px"] >= VIS_PX]
        t_vis = float(tr[vis_k[0]]) if vis_k else None
        conf_t = [float(x) for x in tsim[np.array(m["conflict"], bool)]]
        t_first = conf_t[0] if conf_t else None
        n = 0
        for k in range(3, len(tr)):
            lp, lm = labels[(w, k)], labels[("minus", k)]
            t = float(tr[k])
            if lp["px"] < VIS_PX or not lp["fut_ok"] or (t_first is not None and t >= t_first - 1e-6):
                continue
            frames.append({"base_id": key, "family": w, "seed": 0, "k": k, "t": t, "fn_plus": lp["fn"],
                           "fn_minus": lm["fn"], "v0": lm["v0"], "v2_plus": lp["v2"], "v2_minus": lm["v2"],
                           "stop_plus": lp["stop"], "stop_minus": lm["stop"], "impure_visible": 0,
                           "factor_px": lp["px"], "t_conf": lp["t_conf"], "t_conf_true": lp["t_true"],
                           "ttc": None if lp["t_conf"] is None else lp["t_conf"] - t})
            n += 1
        pairs.append({"base_id": key, "town": ds, "family": w, "seed": 0, "plus": f"{key}-{w}",
                      "minus": f"{key}-minus", "t_trig": tc - T_ENGAGE if w != "static" else None, "t_vis": t_vis,
                      "t_div": np.inf, "t_conflict": t_first, "reason": "ok" if n else
                      ("never_visible" if t_vis is None else "no_frame"), "n_obs": n})
    if "null" in worlds:
        for k in range(3, len(tr)):
            ln, lm = labels[("null", k)], labels[("minus", k)]
            if ln["px"] < VIS_PX or not ln["fut_ok"]:
                continue
            nulls.append({"base_id": key, "family": "null", "seed": 0, "k": k, "t": float(tr[k]),
                          "fn_plus": lm["fn"], "fn_null": ln["fn"], "v2_plus": lm["v2"], "v2_null": ln["v2"],
                          "null_conflict": ln["t_conf"] is not None, "factor_px": ln["px"],
                          "d_expert": ln["v2"] - lm["v2"]})
    return rows, past, fut, frames, nulls, pairs


def worlds_reason(meta, w):
    m = meta["worlds"].get(w, {})
    return m.get("reason", "missing")


def index(workers: int = 4):
    import pandas as pd
    from joblib import Parallel, delayed
    sdirs = sorted(p.parent for p in root("scenes").glob("*/meta.json"))
    out = Parallel(workers)(delayed(_scene_rows)(d) for d in sdirs)
    t = pd.DataFrame([r for o in out for r in o[0]])
    past = np.array([p for o in out for p in o[1]], np.float32)
    fut = np.array([f for o in out for f in o[2]], np.float32)
    obs = pd.DataFrame([f for o in out for f in o[3]])
    null = pd.DataFrame([f for o in out for f in o[4]])
    pairs = pd.DataFrame([f for o in out for f in o[5]])
    obs["d_expert"] = obs.v2_plus - obs.v2_minus
    assert t.frame_name.is_unique
    d = root()
    t["role"] = np.where(t.frame_name.isin(set(obs.fn_plus) | set(obs.fn_minus) | set(null.fn_null)), "obs", "stream")
    t.to_parquet(d / "index.parquet", index=False)
    np.save(d / "past.npy", past)
    np.save(d / "future.npy", fut)
    obs.to_parquet(d / "obs.parquet", index=False)
    null.to_parquet(d / "null.parquet", index=False)
    pairs.to_csv(d / "pairs.csv", index=False)
    return t, obs, null, pairs


FAMILY_LABEL = {"static": "stopped car", "cutin": "cut-in", "oncoming": "oncoming", "null": "null (keeps lane)"}


def fig(specs: list[str], probe: str | None, out: Path, name: str = "i3-hugsim-pairs-validation",
        width_px: int = 360):
    """Front camera x+ / x- / |x+ - x-|, one row per spec "key:world:dt" (dt = frame time - t_c, s), plus an
    optional occlusion-probe row (a parked car put behind scene objects, see pairs_render.occlusion_probe)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from . import plots
    rows = []
    for sp in specs:
        key, w, dt = sp.split(":")
        z = np.load(root("scenes", key) / "fig_frames.npz")
        tag = f"{w}_{float(dt):+.0f}"
        ds, scene = key.split("-", 1)
        rows.append((f"{ds} {scene[:12]}: {FAMILY_LABEL[w]}, $t_c{float(dt):+.0f}$ s", z[f"{tag}_plus"],
                     z[f"{tag}_minus"]))
    if probe:
        z = np.load(root("scenes", probe) / "probe_frames.npz")
        ds, scene = probe.split("-", 1)
        rows.append((f"{ds} {scene[:12]}: occlusion probe (parked car)", z["plus"], z["minus"]))
    ar = rows[0][1].shape[0] / rows[0][1].shape[1]
    with plt.rc_context(plots.STYLE):
        f = plt.figure(figsize=(plots.PAGE, plots.PAGE / 3 * ar * len(rows) + 0.18))
        gs = f.add_gridspec(len(rows), 3, left=0, right=1, bottom=0, top=1 - 0.18 / (plots.PAGE / 3 * ar * len(rows)
                                                                                     + 0.18), wspace=0.01, hspace=0.02)
        for r, (label, p, m) in enumerate(rows):
            h = int(round(width_px * ar))
            small = lambda x: np.asarray(Image.fromarray(x).resize((width_px, h), Image.LANCZOS))  # noqa: E731
            d = np.abs(p.astype(np.int16) - m).max(-1).astype(np.uint8)
            panels = (small(p), small(m), np.asarray(Image.fromarray(d).resize((width_px, h), Image.BOX)))
            for c, im in enumerate(panels):
                a = f.add_subplot(gs[r, c])
                a.imshow(im, cmap="magma" if c == 2 else None, vmin=0, vmax=128 if c == 2 else None)
                a.set_axis_off()
                if r == 0:
                    a.set_title(("$x^+$ (with actor)", "$x^-$ (plain scene)", "$|x^+ - x^-|$, max over RGB")[c], pad=2)
                if c == 0:
                    a.text(0.01, 0.97, label, transform=a.transAxes, va="top", ha="left", fontsize=5.5, color="white",
                           bbox=dict(facecolor="black", alpha=0.55, pad=1.0, edgecolor="none"))
        f.savefig(out / f"{name}.png", dpi=170)
        f.savefig(out / f"{name}.pdf")
        plt.close(f)
    # photographic panels: an adaptive 256-colour palette keeps the PNG under the repo's ~500 KB
    im = Image.open(out / f"{name}.png").convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out / f"{name}.png",
                                                                                               optimize=True)
    return out / f"{name}.png"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("select", "index", "fig"))
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--specs", nargs="*")
    ap.add_argument("--probe")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.step == "select":
        t = select()
        print(t.to_string())
        print(t.groupby("dataset").t_c.apply(lambda s: f"{s.notna().sum()}/{len(s)}"))
    elif a.step == "fig":
        print(fig(a.specs, a.probe, Path(a.out)))
    else:
        t, obs, null, pairs = index()
        print(len(t), "frames;", len(obs), "pair frames;", len(null), "null frames")
        print(pairs.groupby(["family", "reason"]).size())


if __name__ == "__main__":
    main()
