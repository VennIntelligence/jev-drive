"""Synthetic load on a running driver, shaped like the nuPlan track: N concurrent rollouts, each 10 decisions at 2 Hz with 8 JPEGs
(1920 x 1080), one egomotion and one route message per step, then `drive` (docs/alpasim.md, "What a nuPlan-track driver actually
receives"). Checks every response and prints one JSON line: time until the service answered, per-`drive` latency percentiles (client
side, so queueing behind the other rollouts is included), the driver's version. Images are noise over a gradient: decode cost is real,
the plans mean nothing. Runs inside the image (`docker exec <c> python /app/probe.py`, the --network none test) or from any env with
alpasim_grpc.

  python probe.py [--address 127.0.0.1:6789] [--sessions 2] [--rollouts 4] [--steps 10] [--ready-timeout 300] [--pace 0]
"""
import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
from alpasim_grpc.v0 import common_pb2 as C, egodriver_pb2 as E, egodriver_pb2_grpc, sensorsim_pb2 as S
from PIL import Image

CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
STEP = 500_000


def jpeg(seed: int) -> bytes:
    r = np.random.default_rng(seed)
    im = np.linspace(40, 200, 1920)[None, :, None] + r.normal(0, 12, (1080, 1920, 3))
    b = io.BytesIO()
    Image.fromarray(im.clip(0, 255).astype(np.uint8)).save(b, "JPEG", quality=90)
    return b.getvalue()


def start(uuid: str) -> E.DriveSessionRequest:
    cams = []
    for c in CAMS:                                       # CAM_F0 of the public scenes; the drivers read no other camera
        a = S.AvailableCamerasReturn.AvailableCamera(logical_id=c)
        a.intrinsics.logical_id, a.intrinsics.resolution_h, a.intrinsics.resolution_w = c, 1080, 1920
        p = a.intrinsics.opencv_pinhole_param
        p.focal_length_x, p.focal_length_y, p.principal_point_x, p.principal_point_y = 1573.5, 1496.3, 960.0, 560.0
        a.rig_to_camera.vec.x, a.rig_to_camera.vec.y, a.rig_to_camera.vec.z = 1.786, -0.026, 1.523
        q = a.rig_to_camera.quat                         # camera (x right, y down, z forward) -> rig (x forward, y left, z up)
        q.w, q.x, q.y, q.z = 0.5, -0.5, 0.5, -0.5
        cams.append(a)
    V = E.DriveSessionRequest.RolloutSpec.VehicleDefinition
    return E.DriveSessionRequest(session_uuid=uuid, random_seed=0, rollout_spec=E.DriveSessionRequest.RolloutSpec(vehicle=V(available_cameras=cams)))


def rollout(stub, uuid: str, steps: int, imgs: list, pace: float, pool: ThreadPoolExecutor) -> list:
    """One rollout at 8 m/s on a straight line; returns the `drive` latencies (s)."""
    stub.start_session(start(uuid), timeout=60)
    lat = []
    for k in range(steps):
        t = 17_000 + k * STEP
        pose = C.PoseAtTime(timestamp_us=t, pose=C.Pose(vec=C.Vec3(x=8.0 * t * 1e-6), quat=C.Quat(w=1.0)))
        msgs = [(stub.submit_image_observation, E.RolloutCameraImage(session_uuid=uuid, camera_image=E.RolloutCameraImage.CameraImage(
            frame_start_us=max(t - 30_000, 0), frame_end_us=t, image_bytes=imgs[(k + i) % len(imgs)], logical_id=c))) for i, c in enumerate(CAMS)]
        msgs.append((stub.submit_egomotion_observation, E.RolloutEgoTrajectory(
            session_uuid=uuid, trajectory=C.Trajectory(poses=[pose]), dynamic_states=[C.DynamicState(linear_velocity=C.Vec3(x=8.0))])))
        msgs.append((stub.submit_route, E.RouteRequest(session_uuid=uuid, route=E.Route(
            timestamp_us=t, waypoints=[C.Vec3(x=40.0 + 4.2 * i) for i in range(10)] + [C.Vec3(x=float("nan"), y=float("nan"))] * 10))))
        list(pool.map(lambda m: m[0](m[1], timeout=60), msgs))          # concurrent, in no fixed order, as the runtime sends them
        t0 = time.perf_counter()
        r = stub.drive(E.DriveRequest(session_uuid=uuid, time_now_us=t, time_query_us=t + STEP), timeout=120)
        lat.append(time.perf_counter() - t0)
        ts = [p.timestamp_us for p in r.trajectory.poses]
        assert ts and ts[0] == t and ts[-1] >= t + STEP and ts == sorted(set(ts)), f"{uuid} step {k}: bad trajectory timestamps {ts[:3]}..{ts[-1:]}"
        assert all(np.isfinite([p.pose.vec.x, p.pose.vec.y]).all() for p in r.trajectory.poses), f"{uuid} step {k}: non-finite pose"
        time.sleep(pace)
    stub.close_session(E.DriveSessionCloseRequest(session_uuid=uuid), timeout=60)
    return lat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--address", default="127.0.0.1:6789")
    ap.add_argument("--sessions", type=int, default=2, help="concurrent rollouts (the organisers run 2 per replica)")
    ap.add_argument("--rollouts", type=int, default=4, help="rollouts per concurrent slot, one after the other")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--ready-timeout", type=float, default=300)
    ap.add_argument("--pace", type=float, default=0.0, help="sleep between steps (s); 0 = as fast as the driver answers")
    a = ap.parse_args()
    t0 = time.time()
    ch = grpc.insecure_channel(a.address, options=[("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)])
    grpc.channel_ready_future(ch).result(timeout=a.ready_timeout)
    stub = egodriver_pb2_grpc.EgodriverServiceStub(ch)
    ver = stub.get_version(C.Empty(), timeout=60)
    ready = time.time() - t0
    imgs = [jpeg(i) for i in range(4)]
    pool = ThreadPoolExecutor(10 * a.sessions)
    with ThreadPoolExecutor(a.sessions) as ex:
        runs = list(ex.map(lambda i: sum((rollout(stub, f"probe-{i}-{j}", a.steps, imgs, a.pace, pool) for j in range(a.rollouts)), []), range(a.sessions)))
    ms = 1e3 * np.array(sum(runs, []))
    api = ver.grpc_api_version
    print(json.dumps({"version": ver.version_id, "git": ver.git_hash, "api": [api.major, api.minor, api.patch], "ready_wait_s": round(ready, 2),
                      "sessions": a.sessions, "drives": len(ms), "jpeg_kb": round(len(imgs[0]) / 1e3),
                      "drive_ms": {k: round(float(v), 1) for k, v in zip(("p50", "p90", "p99", "max", "mean"),
                                                                        (*np.percentile(ms, [50, 90, 99]), ms.max(), ms.mean()))}}))


if __name__ == "__main__":
    main()
