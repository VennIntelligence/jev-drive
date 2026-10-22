#!/usr/bin/env python
"""Replay finished AlpaSim rollouts (sim/rollouts/<scene>/<rollout>/rollout.asl) through lib/human_driver.py, no GPU and no stack.

The logged session request, camera JPEGs, ego trajectory and route go to the driver's own gRPC methods in log order; before every logged
drive request comes a recording ground-truth message built as the runtime builds it with send_recording_ground_truth (events/policy.py:
the scene's recorded ego trajectory, RolloutMetadata.ego_rig_recorded_ground_truth_trajectory = unbound.gt_ego_trajectory, moved into
the rig frame of the step (positions and the yaw of each log pose); here the ego pose comes from the logged ego trajectory the driver received). The driver then writes the live
dir (state.json with the shadow, front.jpg, camera.json) and its log dir (drive.jsonl, sessions.jsonl, thumbs/<rollout>.jpg) as in a
real run, so the viewer and the gallery can be tested on real scenes. The driver's own plan is not used: the world replays the logged
drive. Paced by HUMAN_REALTIME (1.0 = real time, 0 = as fast as possible). Runs in the driver image (alpasim_grpc, numpy, PIL):

  docker run --rm -v $REPO:$REPO:ro -v $OUT:$OUT -v $RUN:$RUN:ro -e HUMAN_LIVE=$OUT/live -e ALPASIM_DRIVER_LOG_DIR=$OUT/driver \
    -e HUMAN_START=none -e HUMAN_WHEEL_PORT=47809 -e HUMAN_REALTIME=1 --entrypoint python jev-alpasim:p2h10-f-s0-6f3d05d1 \
    $REPO/experiments/alpasim_human/scripts/replay_asl.py <rollout.asl> [...]
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np
from alpasim_grpc.v0 import common_pb2, egodriver_pb2
from alpasim_grpc.v0.logging_pb2 import LogEntry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import human_driver as hd  # noqa: E402


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError("%s: %s" % (code, msg))


def entries(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                return
            yield LogEntry.FromString(f.read(struct.unpack(">L", h)[0]))


def rot(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def replay(drv, asl, ctx):
    gt, gq, ego, uuid, n = None, None, {}, None, 0
    for e in entries(asl):
        k = e.WhichOneof("log_entry")
        m = getattr(e, k)
        if k == "rollout_metadata":
            gtp = m.ego_rig_recorded_ground_truth_trajectory.poses
            gt = np.array([(p.timestamp_us, p.pose.vec.x, p.pose.vec.y, p.pose.vec.z) for p in gtp])
            gq = np.array([rot(p.pose.quat) for p in gtp])
        elif k == "driver_session_request":
            uuid = m.session_uuid
            drv.start_session(m, ctx)
        elif k == "driver_camera_image":
            drv.submit_image_observation(m, ctx)
        elif k == "driver_ego_trajectory":
            drv.submit_egomotion_observation(m, ctx)
            ego.update((p.timestamp_us, p.pose) for p in m.trajectory.poses)
        elif k == "route_request":
            drv.submit_route(m, ctx)
        elif k == "driver_request":
            ts = max([t for t in ego if t <= m.time_now_us], default=None)
            if gt is not None and ts is not None:
                v = ego[ts].vec
                Re = rot(ego[ts].quat)
                P = (gt[:, 1:4] - [v.x, v.y, v.z]) @ Re                        # local -> rig of the step: R^T (p - t)
                Rr = np.einsum("ji,njk->nik", Re, gq)                          # R^T R_gt: the log pose's rotation in the rig frame
                h = np.arctan2(Rr[:, 1, 0], Rr[:, 0, 0]) / 2                   # sent as its yaw (about z), enough for the shadow
                drv.submit_recording_ground_truth(egodriver_pb2.GroundTruthRequest(session_uuid=uuid, ground_truth=egodriver_pb2.GroundTruth(
                    timestamp_us=m.time_now_us, trajectory=common_pb2.Trajectory(poses=[common_pb2.PoseAtTime(
                        timestamp_us=int(t), pose=common_pb2.Pose(vec=common_pb2.Vec3(x=x, y=y, z=z), quat=common_pb2.Quat(w=math.cos(a), z=math.sin(a))))
                        for t, (x, y, z), a in zip(gt[:, 0], P, h)]))), ctx)
            drv.drive(m, ctx)
            n += 1
    drv.close_session(egodriver_pb2.DriveSessionCloseRequest(session_uuid=uuid), ctx)
    return uuid, n, 0 if gt is None else len(gt)


def main():
    drv, ctx = hd.Driver(Path(hd.E("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver"))), Ctx()
    for asl in sys.argv[1:]:
        print("replayed %s: session %s, %d drive calls, %d recorded GT poses" % ((asl,) + replay(drv, asl, ctx)), flush=True)


if __name__ == "__main__":
    main()
