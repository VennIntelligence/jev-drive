"""The scene's logged ego trajectory as a HUGSIM plan: the known-good plan source of the controller acceptance
(todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md). NumPy only (runs in envs/hugsim).

Log. `ground_param.pkl` holds the recorded front-camera poses the simulator uses as its route (RC, "far from preset
trajectory", command); `meta_data.json` holds the same poses per camera with timestamps. The camera whose poses equal
the route gives the times. Waymo and KITTI-360 store frame indices instead of seconds; both log at 10 Hz.

Plan at every step. Project the ego onto the logged path (arc length s0), then run a speed profile from the ego's
current speed towards the logged speed at the arc length reached, with bounded acceleration (a_up, a_down), for 3 s,
and put the waypoints at 0.5 k s on the logged path (straight on past its end). The plan starts at the ego and its
points lie on the log, so an ego off the path gets a plan that leads back onto it, as the logged future would read
from where the car is. The bounded ramp keeps the plan within what the iLQR tracker can do (|a| <= 3 m/s^2): every
scenario starts at 1 m/s while the logs run at 4-15 m/s.
"""
import json
import pickle
from pathlib import Path

import numpy as np

from .hugsim_zs import ego_pose2d, plan_times, world_to_plan

FRAME_HZ = 10.0                                   # Waymo / KITTI-360 logs: timestamp = frame index at 10 Hz


class LoggedPlan:
    def __init__(self, scene_dir, a_up: float = 2.0, a_down: float = 4.0, v_window: float = 0.5):
        scene_dir = Path(scene_dir)
        with open(scene_dir / "ground_param.pkl", "rb") as f:
            poses = np.asarray(pickle.load(f)[0], np.float64)
        frames = json.loads((scene_dir / "meta_data.json").read_text())["frames"]
        by_cam = {}
        for fr in frames:
            by_cam.setdefault(fr["rgb_path"].split("/")[-2], []).append(fr)
        cam = next(c for c, fr in by_cam.items() if len(fr) == len(poses)
                   and np.abs(np.array([f["camtoworld"] for f in fr]) - poses).max() < 1e-6)
        t = np.array([f["timestamp"] for f in by_cam[cam]], np.float64)
        t -= t[0]
        self.frame_index_time = bool(np.median(np.diff(t)) >= 0.5)
        if self.frame_index_time:
            t /= FRAME_HZ
        xz = poses[:, [0, 2], 3]
        keep = np.r_[True, np.linalg.norm(np.diff(xz, axis=0), axis=1) > 1e-3]
        self.t, self.xz = t[keep], xz[keep]
        seg = np.diff(self.xz, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.tan = seg / self.seg_len[:, None]
        self.s = np.r_[0.0, np.cumsum(self.seg_len)]
        # logged speed at each pose: arc length over a centred window of v_window seconds (the 12 Hz nuScenes poses are
        # unevenly spaced, so a per-segment ds/dt is noisy)
        h = v_window / 2
        tq = np.clip(np.c_[self.t - h, self.t + h], self.t[0], self.t[-1])
        ds = np.interp(tq[:, 1], self.t, self.s) - np.interp(tq[:, 0], self.t, self.s)
        self.v = ds / np.maximum(tq[:, 1] - tq[:, 0], 1e-6)
        self.a_up, self.a_down = a_up, a_down
        self.s_last = None

    def summary(self) -> dict:
        return {"length_m": round(float(self.s[-1]), 2), "duration_s": round(float(self.t[-1]), 2),
                "v_median": round(float(np.median(self.v)), 2), "v_max": round(float(self.v.max()), 2),
                "n_poses": int(len(self.s)), "frame_index_time": self.frame_index_time}

    def project(self, p):
        """Nearest point on the logged polyline: (arc length, signed cross-track distance, + = ego right of the path,
        tangent heading theta). Searched within [-5, +30] m of the last projection, so a path that comes back near
        itself cannot make the progress jump."""
        d = p - self.xz[:-1]
        u = np.clip(np.einsum("ij,ij->i", d, self.tan), 0.0, self.seg_len)
        foot = self.xz[:-1] + u[:, None] * self.tan
        dist = np.linalg.norm(p - foot, axis=1)
        s = self.s[:-1] + u
        if self.s_last is not None:
            dist = np.where((s >= self.s_last - 5) & (s <= self.s_last + 30), dist, np.inf)
        i = int(np.argmin(dist))
        tx, tz = self.tan[i]
        right = np.array([tz, -tx])                   # right of the direction of travel (see hugsim_zs.fwd_right)
        self.s_last = float(s[i])
        return float(s[i]), float((p - foot[i]) @ right), float(np.arctan2(tx, tz))

    def at(self, s):
        """World points at arc lengths s, straight on along the last segment past the end."""
        s = np.asarray(s, np.float64)
        pts = np.stack([np.interp(s, self.s, self.xz[:, k]) for k in (0, 1)], -1)
        over = np.maximum(s - self.s[-1], 0.0)
        return pts + over[:, None] * self.tan[-1]

    def __call__(self, info, dt: float = 0.05):
        pos, th = ego_pose2d(info)
        s0, xt, th_log = self.project(pos)
        v = max(float(info["ego_velo"]), 0.0)
        tq = plan_times()
        s, ss = s0, []
        for _ in range(int(round(tq[-1] / dt))):
            vt = float(np.interp(s, self.s, self.v))
            v = max(min(vt, v + self.a_up * dt), v - self.a_down * dt, 0.0)
            s += v * dt
            ss.append(s)
        plan = world_to_plan(self.at(np.asarray(ss)[np.rint(tq / dt).astype(int) - 1]), pos, th)
        dth = (th - th_log + np.pi) % (2 * np.pi) - np.pi
        meta = {"log_s": round(s0, 3), "log_xt": round(xt, 4), "log_dth": round(float(dth), 5),
                "log_v": round(float(np.interp(s0, self.s, self.v)), 3), "log_t": round(float(np.interp(s0, self.s, self.t)), 3)}
        return plan, meta
