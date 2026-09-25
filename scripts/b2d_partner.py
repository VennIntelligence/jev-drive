"""Shared control for zero-shot models in Bench2Drive: a learned partner (TCP) drives the phases the model structurally
cannot handle, the model drives everything else. Pre-registration: todos/2026-09-24-zeroshot-exam/openpilot-migration.md,
section D3. Python 3.8, runs inside the route subprocess (envs/b2d-tcp: torch + carla).

Partner: the official Bench2DriveZoo TCP agent (team_code/tcp_b2d_agent.py, branch tcp/admlp 8a08b07,
PLANNER_TYPE=only_traj, checkpoint tcp_b2d.ckpt sha256 e6573ff1...) run as shipped, with its own three 1600x900
cameras, its own route planner / target point and its own PID and final throttle cap (environment adaptations in
TCPPartner). Its network runs on every tick on which its control has weight (it drives or is being blended in); on
the other ticks only its route planner advances (TCP has no temporal input, seq_len 1), and each stint starts with
fresh PID windows. Running it on every tick cost ~160 ms per tick (the shipped JPEG round trip of three 1600x900
images, the resize and the forward), which would have made the full run ~4x slower.

Arbiter - observable state only (speed, route geometry and the ego's progress along it, time), never outcomes:
  the partner drives while any of
    standstill latch   set when speed < 0.1 m/s has lasted 0.5 s (so also at the route start); cleared when
                       speed >= 1.0 m/s has lasted 1.0 s. (TCP's shipped throttle cap - throttle <= 0.05 above
                       1.5 m/s, 1.0 m/s while turning - keeps it near 1.5-2 m/s, so a 3 m/s release would never fire.)
    junction zone      the ego is between 15 m before the start and 5 m after the end (route arc length) of a
                       LEFT / RIGHT route command                                            (only if junctions=True)
    model warm-up      the model has not finished its warm-up
    model not ready    (only while the partner still has weight) the model's own control, computed every tick, has
                       braked within the last 0.5 s: control is handed back only to a model that would keep the car
                       moving. Without this, openpilot + Zoo PID took over at TCP's 1.5 m/s, read its own low-speed plan
                       (x@1 s ~1.2 m) as "slower than now", braked to a stop within 0.5 s and handed back: a 3 s cycle
                       (plumbing run, route 2086, before any acceptance or scored run).
  the model drives otherwise ("tcp_only": the partner drives throughout, the reference arm). A change of driver is blended linearly over 0.5 s (10 ticks) on steer, throttle, brake.
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np

STANDSTILL_V, STANDSTILL_S = 0.1, 0.5
RELEASE_V, RELEASE_S = 1.0, 1.0
GO_S = 0.5
ZONE_BEFORE_M, ZONE_AFTER_M = 15.0, 5.0
BLEND_TICKS = 10
LEFT, RIGHT = 1, 2


class RemoteTCP(object):
    """Stands in for the shipped agent's `self.net`: the network forward runs on the TCP server (scripts/b2d_tcp_server.py,
    a CUDA build that knows this GPU; envs/b2d-tcp's torch 2.2 has no sm_120 kernels), process_action / control_pid (with
    their PID state) run here on the CPU copy of the shipped model, exactly as the shipped code calls them."""

    def __init__(self, net, sock_path):
        import socket
        import zeroshot_wire as wire
        self.net, self.wire = net, wire
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_path)

    def __call__(self, img, state, target_point):
        import torch
        self.wire.send(self.sock, {"cmd": "forward"}, {"img": img.numpy().astype(np.float32),
                                                      "state": state.numpy().astype(np.float32),
                                                      "target_point": target_point.numpy().astype(np.float32)})
        _, out = self.wire.recv(self.sock)
        return {k: torch.from_numpy(np.array(v)) for k, v in out.items()}

    def process_action(self, *a):
        return self.net.process_action(*a)

    def control_pid(self, *a):
        return self.net.control_pid(*a)


class TCPPartner(object):
    """The shipped Bench2DriveZoo TCPAgent (setup, tick, run_step, route planner, PIDs, throttle cap: its own code),
    with three environment adaptations: tensors that the shipped code moves to 'cuda' stay on the CPU and the network
    forward runs on the TCP server (RemoteTCP); the checkpoint is loaded to the CPU; its 'bev' camera (a 50 m-high
    top view that it only saves to disk, rejected by the leaderboard's sensor validation) is not spawned and a blank
    image is passed in its place. Its per-tick metric_info log (six simulator RPCs, saved only with SAVE_PATH) is skipped."""

    def __init__(self, ckpt, sock_path, hero=None, zoo_root=None):
        os.environ.setdefault("PLANNER_TYPE", "only_traj")
        os.environ.setdefault("IS_BENCH2DRIVE", "1")
        os.environ.pop("SAVE_PATH", None)
        root = zoo_root or os.environ.get("B2D_ZOO_ROOT") or str(Path(os.environ["DATA_DIR"]) / "third_party/Bench2DriveZoo")
        if root not in sys.path:
            sys.path.insert(0, root)
        import torch
        from leaderboard.autoagents.autonomous_agent import Track
        from leaderboard.envs.sensor_interface import SensorInterface
        from team_code.tcp_b2d_agent import TCPAgent
        orig_to, orig_load, orig_cuda = torch.Tensor.to, torch.load, torch.nn.Module.cuda

        def to_cpu(self_, *args, **kw):
            cpu = lambda d: "cpu" if (isinstance(d, str) and d.startswith("cuda")) or \
                (isinstance(d, torch.device) and d.type == "cuda") else d  # noqa: E731
            return orig_to(self_, *[cpu(x) for x in args], **{k: cpu(v) for k, v in kw.items()})

        torch.Tensor.to = to_cpu                             # this process has no other torch user
        torch.load = lambda f, map_location=None, **kw: orig_load(f, map_location="cpu", **kw)
        torch.nn.Module.cuda = lambda self_, *args, **kw: self_
        try:
            a = TCPAgent.__new__(TCPAgent)                  # the base __init__ only finds the hero; set its fields
            a.track, a._global_plan, a._global_plan_world_coord = Track.SENSORS, None, None
            a.sensor_interface, a.wallclock_t0 = SensorInterface(), None
            a.hero_actor = hero                              # read by the shipped run_step (get_metric_info)
            a.setup(ckpt + "+partner")
        finally:
            torch.load, torch.nn.Module.cuda = orig_load, orig_cuda
        a.net = RemoteTCP(a.net, sock_path)
        a.get_metric_info = dict                             # logging only (six hero RPCs per tick); never read here
        self.fresh_pids = copy.deepcopy((a.net.net.turn_controller, a.net.net.speed_controller))
        self.idle = False
        self.agent = a
        self.specs = [s for s in a.sensors() if s["id"] != "bev"]
        self.tags = [s["id"] for s in self.specs]
        self.bev = np.zeros((512, 512, 4), np.uint8)
        self.last = (0.0, 0.0, 1.0)

    def sensors(self, taken):
        return [dict(s) for s in self.specs if s["id"] not in taken]

    def camera_tags(self):
        return [s["id"] for s in self.specs if s["type"].startswith("sensor.camera")]

    def set_global_plan(self, gps, world):
        self.agent.set_global_plan(gps, world)

    def advance(self, data):
        """A tick on which the partner's control has no weight: only its route planner advances (the shipped tick()
        does this before the network; skipped, the planner would lose the ego after 50 m). Next stint starts with
        fresh PID windows, as at a route start."""
        a = self.agent
        if not a.initialized:
            a._init()
        a._route_planner.run_step(a.gps_to_location(np.asarray(data["GPS"][1], float)[:2]))
        self.idle = True

    def step(self, data, timestamp):
        """data: {tag: (frame, value)} with every partner sensor; returns (throttle, steer, brake)."""
        if not all(t in data for t in self.tags):
            return self.last
        if self.idle:
            net = self.agent.net.net
            net.turn_controller, net.speed_controller = copy.deepcopy(self.fresh_pids)
            self.idle = False
        inp = {t: data[t] for t in self.tags}
        inp["bev"] = (data["SPEED"][0], self.bev)
        c = self.agent.run_step(inp, timestamp)
        self.last = (float(c.throttle), float(c.steer), float(c.brake))
        return self.last


class Arbiter(object):
    def __init__(self, route_xy, route_cmd, junctions=True, partner_only=False):
        xy, cmd = np.asarray(route_xy, float), np.asarray(route_cmd)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        self.zones = []
        i = 0
        while junctions and i < len(cmd):
            if cmd[i] in (LEFT, RIGHT):
                j = i
                while j + 1 < len(cmd) and cmd[j + 1] == cmd[i]:
                    j += 1
                self.zones.append((self.s[i] - ZONE_BEFORE_M, self.s[j] + ZONE_AFTER_M))
                i = j + 1
            else:
                i += 1
        self.latch, self.low_t, self.high_t, self.go_t = True, 0.0, 0.0, 0.0
        self.w = 0.0                                          # weight of the model's control, 0 = partner
        self.partner_only = bool(partner_only)                # reference arm: the partner drives the whole route
        self.driver = "partner"

    def step(self, dt, speed, route_index, model_ready, model_go=True):
        """-> weight of the model's control in [0, 1] and the reason the partner holds (or '').
        model_go: the model's own control (computed every tick, also while the partner drives) does not brake."""
        self.go_t = self.go_t + dt if model_go else 0.0
        if speed < STANDSTILL_V:
            self.low_t += dt
            if self.low_t >= STANDSTILL_S:
                self.latch = True
        else:
            self.low_t = 0.0
        if speed >= RELEASE_V:
            self.high_t += dt
            if self.high_t >= RELEASE_S:
                self.latch = False
        else:
            self.high_t = 0.0
        s_now = self.s[min(route_index, len(self.s) - 1)]
        zone = any(a <= s_now <= b for a, b in self.zones)
        why = "partner_only" if self.partner_only else "warmup" if not model_ready else "standstill" if self.latch else "junction" if zone else ""
        if not why and self.w < 1.0 and self.go_t < GO_S:
            why = "model_not_ready"                           # hand over only to a model that is not braking
        target = 0.0 if why else 1.0
        step = 1.0 / BLEND_TICKS
        self.w = min(target, self.w + step) if target > self.w else max(target, self.w - step)
        self.driver = "model" if self.w >= 1.0 else "partner" if self.w <= 0.0 else "blend"
        return self.w, why

    @staticmethod
    def mix(w, model, partner):
        return tuple(float(w * m + (1.0 - w) * p) for m, p in zip(model, partner))
