"""ROUND6 line B (plans/2026-10-21-recovery-rows.md): onpol_pai's log-replay driver (onp_log_driver.py, the on1_PAIH rollouts) with
ChauffeurNet-style lateral perturbation episodes. The model runs, is fed and is recorded exactly as there; the returned reference is the log
plus the offset E of lib/rec_geom.py (ramp-in over the T_IN s before each episode's t0, driven recovery over T_OUT s after it), so the rendered
frames and the ego history of every decision are those of an ego that really drove the perturbed path.

  REC_EPS   a JSON file {scene id: [[t0_us, e0_m, tan_psi0], ...]} (rollout clock; scripts/rec.py sched writes it). A scene not in the file,
            and every decision whose reference window touches no episode, returns onp_log_driver's reference unchanged (same object).
            Episodes with e0 = psi0 = 0 go through the offset code and return the log bit for bit (G1 of the plan).
  records   every drive record of a listed scene gains `rec` = [E, dE/ds] of the driven offset at time_now (0 outside the windows)
Run as onp_log_driver (pai_box.sh SERVE: DRV_PY=../../round6/lib/rec_driver.py REC_EPS=<file>, + the H read's wizard override
runtime.simulation_config.send_recording_ground_truth=true). ONP_SWITCH / ONP_BRANCH / ONP_CELLS must stay unset.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.normpath(os.path.join(_HERE, "..", "..", "onpol_pai", "lib")), os.path.normpath(os.path.join(_HERE, "..", "..", "alpasim", "lib"))):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.append(_d)
import onp_log_driver as L  # noqa: E402
import rec_geom as RG  # noqa: E402
from onp_log_driver import P, S, common_pb2  # noqa: E402

CAM = os.environ.get("PAI_CAM", "camera_front_wide_120fov")   # = pai_driver.CAM; read by lib/sim_fast.py (ALPASIM_FAST) from this file
EPS = json.load(open(os.environ["REC_EPS"])) if os.environ.get("REC_EPS") else {}
assert not (L.CELLS or L.BRANCH or L.SWITCH or L.SWITCH_ALL is not None), "rec_driver: ONP_CELLS / ONP_BRANCH / ONP_SWITCH must be unset"


class RecDriver(L.LogDriver):
    def log_reference(self, s, now: int):
        ref = super().log_reference(s, now)
        eps = EPS.get(s.scene)
        if isinstance(ref, str) or not eps:
            return ref
        n = int(round(L.HORIZON_S * 1e6 / P.STEP_US))
        ts = now + P.STEP_US * np.arange(n + 1.0)
        if not any(t0 - RG.T_IN <= ts[-1] and ts[0] <= t0 + RG.T_OUT for t0, _, _ in eps):
            ref[1]["rec"] = [0.0, 0.0]
            return ref
        g = s.gt
        t, pos, ax = L.log_arrays(g.trajectory)
        with s.lock:
            pose = s.poses.get(now)
        R, o = S.quat_R(pose.pose.quat), np.array([pose.pose.vec.x, pose.pose.vec.y, pose.pose.vec.z])
        pl, al = pos @ R.T + o, ax @ R.T                     # as onp_log_driver.log_reference
        rf, _ = L.sample(t, pl[:, :2], np.unwrap(np.arctan2(al[:, 1], al[:, 0])), ts)
        ta, Sa = RG.arc(t, pos[:, :2])                       # arc length: frame-free
        E, D1, _ = RG.driven(ts, ta, Sa, eps)
        rf = RG.offset(rf, E, D1)
        traj = common_pb2.Trajectory(poses=[common_pb2.PoseAtTime(timestamp_us=now + P.STEP_US * i, pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=float(x), y=float(y), z=pose.pose.vec.z),
            quat=common_pb2.Quat(w=float(np.cos(a / 2)), x=0.0, y=0.0, z=float(np.sin(a / 2))))) for i, (x, y, a) in enumerate(rf)])
        return traj, {**ref[1], "rec": [round(float(E[0]), 5), round(float(D1[0]), 6)]}

    def get_version(self, req, ctx):
        v = super().get_version(req, ctx)
        v.version_id += "-rec"
        return v


if __name__ == "__main__":
    P.Driver = RecDriver                                     # pai_driver.main builds whatever its module calls Driver
    P.main()
