#!/usr/bin/env python
"""Stand-in agent for Bench2Drive closed-loop cost measurement. See research/carla-efficiency.md.

The leaderboard loads this file by path (`--agent`) and calls `get_entry_point()`. Everything is
configured through the JSON file passed as `--agent-config`:

    {"rig": "front3", "width": 1600, "height": 900, "infer_ms": 129.0,
     "decimate": 1, "overlap": false, "policy": "sleep"}

`policy` decides what the inference stand-in actually does:
  none    no inference at all. Measures the simulator's floor.
  sleep   burn `infer_ms` of wall clock without touching the GPU. Upper bound on throughput for a
          policy of that latency; it does NOT model GPU contention with CARLA's renderer.
  gpu     ask scripts/b2d_policy_server.py (a real frozen Qwen3-VL in envs/jevdrive, one process
          per worker) for features over a unix socket, and wait. This is the honest number.

`decimate` runs the policy every N ticks and holds the last control in between; the cameras get
`sensor_tick = N * 0.05` so the server does not render frames nobody reads. `overlap` additionally
runs inference in a background thread, so the simulator keeps ticking while the policy thinks and
the new control lands one inference later - which is what a real vehicle does anyway.

Everything except `policy: none` is off by default, so the unoptimised path stays reproducible.
Python 3.8: this runs in envs/carla, not in the project env.
"""
import json
import os
import socket
import struct
import sys
import threading
import time

import carla
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.timer import GameTime

DELTA = 0.05  # the leaderboard fixes 20 Hz

# The Bench2Drive evaluation rig, copied from Bench2Drive/tools/data_collect.py so the extrinsics
# and FOVs match the official one. Cost depends on count and resolution, not on pose, but a rig
# that differs from the published one invites the question.
CAMERAS = [
    ("CAM_FRONT", 0.80, 0.0, 1.60, 0.0, 70),
    ("CAM_FRONT_LEFT", 0.27, -0.55, 1.60, -55.0, 70),
    ("CAM_FRONT_RIGHT", 0.27, 0.55, 1.60, 55.0, 70),
    ("CAM_BACK", -2.0, 0.0, 1.60, 180.0, 110),
    ("CAM_BACK_LEFT", -0.32, -0.55, 1.60, -110.0, 70),
    ("CAM_BACK_RIGHT", -0.32, 0.55, 1.60, 110.0, 70),
]
# front3 is our planner's rig; b2d6 is what UniAD/VAD/AD-MLP are evaluated with in Bench2DriveZoo.
RIGS = {"none": 0, "front1": 1, "front3": 3, "b2d6": 6}


# The leaderboard destroys its agent during cleanup, so b2d_route.py cannot read the timings off
# the evaluator afterwards. The agent parks itself here instead.
LAST_AGENT = None


def get_entry_point():
    return "StubAgent"


class PolicyClient(object):
    """Talks to scripts/b2d_policy_server.py over a unix socket: one length-prefixed request with
    the raw BGRA planes, one float reply with the server-side latency. Kept deliberately dumb - we
    are measuring the cost of having a policy in the loop, not designing an RPC protocol."""

    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)

    def infer(self, arrays):
        head = struct.pack("<III", len(arrays), arrays[0].shape[1], arrays[0].shape[0])
        payload = b"".join(a.tobytes() for a in arrays)
        self.sock.sendall(head + struct.pack("<Q", len(payload)) + payload)
        (server_ms,) = struct.unpack("<f", self._recv_exactly(4))
        return server_ms

    def _recv_exactly(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("policy server closed the connection")
            buf += chunk
        return buf


class StubAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        global LAST_AGENT
        LAST_AGENT = self
        self.track = Track.SENSORS
        # Bench2Drive rewrites --agent-config to "<path>+<route name>" before calling setup
        # (leaderboard_evaluator.py:363), so the path it hands over is not a file. Splitting on
        # '+' is the only way to get the config back; an agent that trusts the argument silently
        # runs on its defaults, which is how the first sweep measured the same rig four times.
        path = (path_to_conf_file or "").split("+")[0]
        cfg = {}
        if path and os.path.isfile(path):
            with open(path) as fh:
                cfg = json.load(fh)
        else:
            raise RuntimeError("agent config %r not readable; refusing to run on defaults"
                               % path_to_conf_file)
        self.cfg = cfg
        self.n_cam = RIGS[cfg.get("rig", "front3")]
        self.width = int(cfg.get("width", 1600))
        self.height = int(cfg.get("height", 900))
        self.infer_ms = float(cfg.get("infer_ms", 0.0))
        self.decimate = max(1, int(cfg.get("decimate", 1)))
        self.overlap = bool(cfg.get("overlap", False))
        self.policy = cfg.get("policy", "none")
        self.drive = cfg.get("drive", "straight")

        self._tick = 0
        self._control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        self._client = None
        self._pending = None
        self._lock = threading.Lock()
        # Per-phase wall clock, in seconds, one entry per tick. b2d_route.py merges these with the
        # timings the hooks collect around this call.
        self.timings = {"sensor_wait": [], "infer": [], "agent_total": [], "policy_ticks": 0,
                        "server_infer_ms": []}
        if self.policy == "gpu":
            self._client = PolicyClient(cfg["policy_socket"])

    def sensors(self):
        sensors = []
        for i in range(self.n_cam):
            name, x, y, z, yaw, fov = CAMERAS[i]
            spec = {"type": "sensor.camera.rgb", "x": x, "y": y, "z": z,
                    "roll": 0.0, "pitch": 0.0, "yaw": yaw,
                    "width": self.width, "height": self.height, "fov": fov, "id": name}
            if self.decimate > 1:
                # Do not render frames the policy will never read. This is the whole point of
                # decimation: the saving is server-side rendering plus the transfer, not just the
                # Python work we skip.
                spec["sensor_tick"] = self.decimate * DELTA
            sensors.append(spec)
        sensors += [
            {"type": "sensor.other.imu", "x": -1.4, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": DELTA, "id": "IMU"},
            {"type": "sensor.other.gnss", "x": -1.4, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "GPS"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ]
        return sensors

    # The base class prints one wallclock line per tick and always calls get_data(). We need the
    # per-phase split and the decimation, so this call is reimplemented rather than wrapped.
    def __call__(self):
        t0 = time.perf_counter()
        self._tick += 1
        run_policy = (self._tick - 1) % self.decimate == 0
        t_wait = t_infer = 0.0
        if run_policy:
            t = time.perf_counter()
            input_data = self._read_sensors()
            t_wait = time.perf_counter() - t
            t = time.perf_counter()
            if self.overlap:
                self._submit(input_data)
            else:
                self._control = self.run_step(input_data, GameTime.get_time())
            t_infer = time.perf_counter() - t
            self.timings["policy_ticks"] += 1
        if self.overlap:
            self._collect()
        self.timings["sensor_wait"].append(t_wait)
        self.timings["infer"].append(t_infer)
        self.timings["agent_total"].append(time.perf_counter() - t0)
        return self._control

    def _read_sensors(self):
        """With `sensor_tick` set, the cameras do not report on every frame, so the stock
        `get_data(frame)` - which waits for data tagged with the *current* frame - would block
        forever. Take the newest complete set instead. With decimate == 1 the two agree."""
        if self.decimate == 1:
            return self.sensor_interface.get_data(GameTime.get_frame())
        si = self.sensor_interface
        wanted = set(si._sensors_objects.keys())
        got = {}
        while not wanted.issubset(got.keys()):
            tag, frame, data = si._data_buffers.get(True, si._queue_timeout)
            got[tag] = (frame, data)
        return got

    def _submit(self, input_data):
        with self._lock:
            if self._pending is not None:
                return  # still thinking about the previous frame; skip this one, as a real car would
            self._pending = {"done": False}
            pending = self._pending

        def work():
            try:
                ctrl = self.run_step(input_data, GameTime.get_time())
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread below
                with self._lock:
                    pending["error"] = exc
                    pending["done"] = True
                return
            with self._lock:
                pending["control"] = ctrl
                pending["done"] = True

        threading.Thread(target=work, daemon=True).start()

    def _collect(self):
        done = None
        with self._lock:
            if self._pending is not None and self._pending["done"]:
                done, self._pending = self._pending, None
        if done is None or not done.get("done"):
            return
        # A policy thread that dies quietly is the worst outcome: the route keeps running on the
        # last control and finishes with a plausible-looking time that measured nothing.
        if "error" in done:
            raise RuntimeError("policy failed in the overlap thread: %r" % (done["error"],))
        self._control = done["control"]

    def run_step(self, input_data, timestamp):
        if self.policy == "sleep":
            # A busy wait, not time.sleep: sleep yields the core, which would make the loop look
            # cheaper on CPU than a real policy that holds one.
            end = time.perf_counter() + self.infer_ms / 1e3
            while time.perf_counter() < end:
                pass
        elif self.policy == "gpu":
            arrays = [v[1][:, :, :3] for k, v in sorted(input_data.items()) if k.startswith("CAM_")]
            if arrays:
                self.timings["server_infer_ms"].append(self._client.infer(arrays))
        return self._drive_control(input_data)

    def _drive_control(self, input_data):
        """Keep the car moving so the route actually progresses: a parked ego finishes no route and
        would measure the wrong thing. Not a driving policy - speed hold plus the route heading."""
        speed = 0.0
        if "SPEED" in input_data:
            speed = float(input_data["SPEED"][1]["speed"])
        throttle = 0.5 if speed < 6.0 else 0.0
        steer = 0.0
        if self.drive == "route" and getattr(self, "_global_plan_world_coord", None):
            steer = self._steer_to_route()
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)

    def _steer_to_route(self):
        hero = getattr(self, "hero_actor", None)
        if hero is None:
            self.get_hero()
            hero = getattr(self, "hero_actor", None)
            if hero is None:
                return 0.0
        tf = hero.get_transform()
        here = np.array([tf.location.x, tf.location.y])
        yaw = np.deg2rad(tf.rotation.yaw)
        for wp, _ in self._global_plan_world_coord:
            tgt = np.array([wp.location.x, wp.location.y])
            if np.linalg.norm(tgt - here) > 6.0:
                d = tgt - here
                ang = np.arctan2(d[1], d[0]) - yaw
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                return float(np.clip(ang, -1.0, 1.0))
        return 0.0

    def destroy(self):
        if self._client is not None:
            try:
                self._client.sock.close()
            except OSError:
                pass
