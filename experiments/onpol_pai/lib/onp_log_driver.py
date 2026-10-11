"""Log-replay variant of experiments/alpasim/lib/pai_driver.py (lane ONPOL-PAI, pass 2): the simulated ego is steered along the scene's
LOGGED path while the model runs, is fed and is recorded exactly as in the normal driver. Imitation rows at on-log states with the PAI
serving input (f-theta camera through pai_core, rendered 10 Hz frames in every slot).

  logged path   `submit_recording_ground_truth`, which the runtime sends before every `drive` once the wizard override
                `runtime.simulation_config.send_recording_ground_truth=true` is set (off in the dev preset): the recording's whole ego
                trajectory (rig poses, absolute timestamps on the rollout clock), expressed in the TRUE rig frame of the step start
                (AlpaSim 0bb4c4b, runtime/events/policy.py). Composed here with the egomotion pose of `time_now` (full 3-D) it is the
                log in the frame the driver answers in, also when an egomotion error model is on
  model         untouched: pai_driver.Driver.drive runs first on every call and writes its usual record (`poses`, `poses_model`, dumps)
  returned      the log from `time_now` on: ONP_HORIZON s (default 4.0, the model plan's horizon) of 10 Hz poses, x / y / yaw in the local
                frame, z of the current pose and a yaw-only quaternion, as navsim_transfuser_challenge.trajectory.build_trajectory_from_plan
                writes them. The first pose is the log at `time_now`, not the ego's pose: an ego that is off the log gets a reference
                that pulls it back (no blending). Past the end of the log (the rollout ends where the log ends, so the last
                ONP_HORIZON s of every scene) the reference continues on the arc of the log's last 0.5 s (constant speed and yaw
                rate): the MPC clamps a short reference at its last pose, which would brake the ego
  hand-over     the runtime itself keeps the ego on the log for the first force_gt_duration_us (1.7 s in the dev preset) and seeds the
                controller's dynamic state from the log at the first closed-loop step; from there the MPC tracks this reference
  drive.jsonl   every drive record gains `ret` ("log"), `ret_n` (poses returned), `ret_ext` (how many of them are extrapolated),
                `log_now` (the log's pose at time_now in the rig frame of time_now, i.e. where the log is as seen from the ego:
                x > 0 = the ego is behind the log, y > 0 = the ego is right of it) and `log_poses` ((8, 3): the log at time_now + 0.5 .. 4.0 s in that frame, the imitation label in the layout of
                `poses`; `log_valid` of them lie inside the log). The close record gains `log_error`

A `drive` without a ground-truth message of its own timestamp is aborted with a gRPC error and counted: this driver never returns the
model's plan. Run through experiments/onpol_pai/scripts/onp_pass2.sh (DRV_PY of pai_run.sh); environment as pai_driver.py plus ONP_HORIZON.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

import pai_driver as P
from pai_driver import common_pb2, egodriver_pb2, grpc, S

HORIZON_S, TAIL_US = float(os.environ.get("ONP_HORIZON", "4.0")), 500_000
TL = threading.local()


class Out:
    """drive.jsonl writer that merges the calling thread's log-replay fields into the record pai_driver writes."""

    def __init__(self, f):
        self.f = f

    def write(self, line: str):
        x = getattr(TL, "extra", None)
        return self.f.write(json.dumps({**json.loads(line), **x}) + "\n" if x else line)


def log_arrays(traj) -> tuple:
    """A common.Trajectory -> t_us (n,), positions (n, 3), unit x axes (n, 3) of the poses, in the message's frame."""
    a = np.array([[p.timestamp_us, p.pose.vec.x, p.pose.vec.y, p.pose.vec.z, p.pose.quat.w, p.pose.quat.x, p.pose.quat.y, p.pose.quat.z]
                  for p in traj.poses], np.float64).reshape(-1, 8)
    w, x, y, z = a[:, 4:].T
    return a[:, 0], a[:, 1:4], np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], 1)


def sample(t, xy, yaw, ts) -> tuple:
    """The path (t, xy, unwrapped yaw) at times ts -> (len(ts), 3) x, y, yaw and the number of extrapolated rows. Inside the log: linear.
    After its end: the arc of the last TAIL_US (constant speed, course rate = yaw rate)."""
    out = np.stack([np.interp(ts, t, xy[:, 0]), np.interp(ts, t, xy[:, 1]), np.interp(ts, t, yaw)], 1)
    ext = ts > t[-1]
    if ext.any():
        ta = max(t[-1] - TAIL_US, t[0])
        d = max((t[-1] - ta) * 1e-6, 1e-3)
        pa = np.array([np.interp(ta, t, xy[:, 0]), np.interp(ta, t, xy[:, 1]), np.interp(ta, t, yaw)])
        v, w = (xy[-1] - pa[:2]) / d, (yaw[-1] - pa[2]) / d
        speed, course, dt = float(np.hypot(*v)), float(np.arctan2(v[1], v[0])) + w * d / 2, (ts[ext] - t[-1]) * 1e-6
        if abs(w) > 1e-4:
            dx, dy = (np.sin(course + w * dt) - np.sin(course)) / w, (np.cos(course) - np.cos(course + w * dt)) / w
        else:
            dx, dy = np.cos(course) * dt, np.sin(course) * dt
        out[ext] = np.stack([xy[-1, 0] + speed * dx, xy[-1, 1] + speed * dy, yaw[-1] + w * dt], 1)
    return out, int(ext.sum())


class LogDriver(P.Driver):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.out = Out(self.out)

    def submit_recording_ground_truth(self, req, ctx):
        s = self._s(req.session_uuid, ctx)
        s.gt = req.ground_truth                              # parsed in drive: one message per step, each the whole clip
        return common_pb2.Empty()

    def log_reference(self, s, now: int):
        """(Trajectory to return, drive.jsonl fields) from the ground-truth message of `now`; a string when it cannot be built."""
        g = getattr(s, "gt", None)
        with s.lock:
            pose = s.poses.get(now)
        if g is None or int(g.timestamp_us) != now:
            return f"no recording ground truth at Drive time {now} (latest {int(g.timestamp_us) if g is not None else None}): " \
                   "runtime.simulation_config.send_recording_ground_truth=true is required"
        if pose is None or len(g.trajectory.poses) < 2:
            return f"no egomotion pose at Drive time {now}" if pose is None else "recording ground truth with fewer than 2 poses"
        t, pos, ax = log_arrays(g.trajectory)
        yaw_r = np.unwrap(np.arctan2(ax[:, 1], ax[:, 0]))
        R, o = S.quat_R(pose.pose.quat), np.array([pose.pose.vec.x, pose.pose.vec.y, pose.pose.vec.z])
        pl, al = pos @ R.T + o, ax @ R.T                     # the log in the local frame the driver answers in
        n = int(round(HORIZON_S * 1e6 / P.STEP_US))
        ref, ext = sample(t, pl[:, :2], np.unwrap(np.arctan2(al[:, 1], al[:, 0])), now + P.STEP_US * np.arange(n + 1.0))
        traj = common_pb2.Trajectory(poses=[common_pb2.PoseAtTime(timestamp_us=now + P.STEP_US * i, pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=float(x), y=float(y), z=pose.pose.vec.z),
            quat=common_pb2.Quat(w=float(np.cos(a / 2)), x=0.0, y=0.0, z=float(np.sin(a / 2))))) for i, (x, y, a) in enumerate(ref)])
        rig, _ = sample(t, pos[:, :2], yaw_r, now + 500_000 * np.arange(9.0))        # label layout of `poses`: rig frame of `now`
        rig[:, 2] -= 2 * np.pi * np.round(rig[0, 2] / (2 * np.pi))
        return traj, {"ret": "log", "ret_n": n + 1, "ret_ext": ext, "log_now": rig[0].round(4).tolist(),
                      "log_poses": rig[1:].round(4).tolist(), "log_valid": int((now + 500_000 * np.arange(1, 9) <= t[-1]).sum())}

    def drive(self, req, ctx):
        s, now = self._s(req.session_uuid, ctx), int(req.time_now_us)
        ref = self.log_reference(s, now)
        s.count.setdefault("log_error", 0)
        if isinstance(ref, str):
            s.count["log_error"] += 1
            ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, ref)
        TL.extra = ref[1]
        try:
            super().drive(req, ctx)                          # the model, its record and its dumps; the plan it returns is dropped
        finally:
            TL.extra = None
        return egodriver_pb2.DriveResponse(trajectory=ref[0])

    def get_version(self, req, ctx):
        v = super().get_version(req, ctx)
        v.version_id += "-logreplay"
        return v


if __name__ == "__main__":
    P.Driver = LogDriver                                     # pai_driver.main builds whatever its module calls Driver
    P.main()
