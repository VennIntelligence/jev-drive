#!/usr/bin/env python
"""Night queue 3, lane A: one shadow model in a process of its own, fed by scripts/nq3_recorder.py
(todos/2026-09-26-night-queue-3.md, Q1 / Q3 [A] entries).

Why a process per model: the recorder has to carry TFv6 (LEAD cvpr2026) and BridgeDrive (LEAD a41d116, its own `lead`
package) at once, and a shadow never feeds the drive, so its compute need not sit on the simulation's critical path.
The recorder sends every tick's rig data (the author rig, which is sensor for sensor the same for both models) down a
pipe and moves on; this process runs the author's code on it in tick order, as far behind as it needs to be, and
writes the readouts itself. What runs per tick is exactly what the in-process recorders ran:

  tfv6         scripts/p5_pair_agent.py's shadow (P5 / P6 v0): the author's full `run_step` on camera ticks, the
               author's `BaseAgent.tick` in between, the Kalman filter fed the model's own control, no reseed.
               -> tfv6.jsonl, the P5 recorder's format (plus k and tick)
  bridgedrive  scripts/top10_t3_agent.py's shadow (top-10 T3): the same, with the Kalman filter fed the expert's
               control of the previous tick and torch reseeded before every forward. -> bridgedrive.jsonl, T3's format

The simulator handles the authors take in `_init` (hero, world) are stand-ins: with every video / demo output off,
the only live use left is the town name the model input carries (`self._world.get_map().name`), which the recorder
sends. Anything else touching them raises and is counted as a shadow error.

Protocol (stdin / stdout, binary; the process's own stdout is re-pointed to stderr so author prints cannot corrupt it):
frames of <8-byte little-endian length><pickle>. Main -> worker: ("init", dict), ("tick", dict), ("end",). Worker ->
main: ("ready", {sensors, ...}) after init, ("done", summary) after end. A reader thread drains the pipe into a queue so
the recorder's writes never wait on the model.

  python scripts/nq3_shadow.py --model tfv6|bridgedrive      (env: the model's venv, PYTHONPATH at its lead tree)
"""
import argparse
import json
import os
import pickle
import queue
import struct
import sys
import threading
import time

_PROTO_IN = os.fdopen(os.dup(0), "rb", buffering=0)
_PROTO_OUT = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)                                     # author prints and logging go to stderr from here on
sys.stdout = os.fdopen(1, "w", buffering=1)

TFV6_SPEEDS = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)


def send(msg):
    b = pickle.dumps(msg, protocol=5)
    _PROTO_OUT.write(struct.pack("<Q", len(b)) + b)


def recv_exact(n):
    out = bytearray()
    while len(out) < n:
        chunk = _PROTO_IN.read(n - len(out))
        if not chunk:
            raise EOFError
        out += chunk
    return bytes(out)


def recv():
    (n,) = struct.unpack("<Q", recv_exact(8))
    return pickle.loads(recv_exact(n))


class _Stub:
    """Stand-in for the hero / world handles; only the map name is answered."""
    def __init__(self, town):
        self._town = town

    def get_world(self):
        return self

    def get_map(self):
        return self

    @property
    def name(self):
        return self._town

    def __getattr__(self, key):
        raise AttributeError("nq3_shadow stub: %s is not available in the shadow process" % key)


def _imports(model):
    """The author's agent class and the per-model shims the in-process recorders applied."""
    from unittest import mock
    import numpy as np
    import torch
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    import lead.inference.video_recorder as vr
    from lead.common.base_agent import BaseAgent
    # the recorder's handles are stand-ins; VideoRecorder's beartype check wants real carla types, the body does not
    # touch them with every output off
    raw = getattr(vr.VideoRecorder.__init__, "__wrapped__", None)
    if raw is not None:
        vr.VideoRecorder.__init__ = raw
    if model == "bridgedrive":
        import lead.inference.config_closed_loop as _cc
        import lead.training.config_training as _ct
        from lead.inference.sensor_agent_bridgedrive import SensorAgent
        _cc.ClosedLoopConfig.debug_mode = False          # top10_t3_agent._author_shims, verbatim
        get = _ct.TrainingConfig.gpu_name.fget

        def safe(self):
            try:
                return get(self)
            except Exception:
                return ""
        _ct.TrainingConfig.gpu_name = property(safe)
    else:
        from lead.inference.sensor_agent import SensorAgent
    return SensorAgent, BaseAgent, CarlaDataProvider, mock, np, torch


class Shadow:
    def __init__(self, model, init):
        import carla
        from agents.navigation.local_planner import RoadOption
        SensorAgent, self.BaseAgent, CDP, mock, np, self.torch = _imports(model)
        self.np, self.carla, self.model, self.SensorAgent = np, carla, model, SensorAgent
        self.stub = _Stub(init["town"])
        CDP.get_hero_actor = staticmethod(lambda: self.stub)
        self.agent = SensorAgent("127.0.0.1", 0, False)
        a = self.agent
        a.set_global_plan([({"lat": la, "lon": lo, "z": z}, RoadOption(o)) for la, lo, z, o in init["plan_gps"]],
                          [(carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=p, yaw=yw, roll=r)), RoadOption(o))
                           for x, y, z, p, yw, r, o in init["plan_world"]])
        with mock.patch("shutil.which", return_value="/bin/true"):     # setup demands ffmpeg for videos never made
            a.setup(init["model_dir"])
        forward = a.closed_loop_inference.forward
        self.seed = int(init.get("seed", 0))
        self.pred, self.t = None, {"base": [], "cam": [], "forward": []}

        def capture(*args, **kwargs):
            if model == "bridgedrive":
                self.torch.manual_seed(self.seed)
            t0 = time.perf_counter()
            self.pred = forward(*args, **kwargs)
            self.t["forward"].append(time.perf_counter() - t0)
            return self.pred

        a.closed_loop_inference.forward = capture
        self.sensors = a.sensors()
        name = "tfv6.jsonl" if model == "tfv6" else "bridgedrive.jsonl"
        self.fh = open(os.path.join(init["out"], name), "w")
        self.errors, self.ticks, self.cls = 0, 0, None
        if model == "bridgedrive":
            self.cls = list(a.training_config.target_speed_classes)

    def _np(self, v):
        return self.np.round(v[0].detach().float().cpu().numpy(), 4).tolist()

    def tick(self, m):
        a, np, carla = self.agent, self.np, self.carla
        full, data, t = m["cam"], m["data"], m["t"]
        self.pred = None
        self.ticks += 1
        if self.model == "bridgedrive" and m.get("prev_ctrl") is not None:
            th, st, br = m["prev_ctrl"]
            a.control = carla.VehicleControl(throttle=th, steer=st, brake=br)
        t0 = time.perf_counter()
        try:
            if not full and a.initialized:
                self.BaseAgent.tick(a, dict(data), use_kalman_filter=a.training_config.use_kalman_filter_for_gps)
                self.t["base"].append(time.perf_counter() - t0)
                return
            ctrl = self.SensorAgent.run_step(a, dict(data), t)
        except Exception as e:                     # recorded, never allowed to stop the drive
            self.errors += 1
            if self.errors <= 3:
                print("%s shadow failed at frame %d: %r" % (self.model, m["frame"], e), flush=True)
            return
        self.t["cam"].append(time.perf_counter() - t0)
        p = self.pred
        if p is None:                              # the author's first tick only initialises
            return
        keys = ("pred_target_speed_distribution", "pred_target_speed_scalar", "pred_future_waypoints", "pred_route")
        if self.model == "tfv6":
            rec = {"frame": m["frame"], "k": m["k"], "tick": m["tick"]}
            for key in keys:
                v = getattr(p, key, None)
                if v is not None:
                    rec[key] = self._np(v)
            if "pred_target_speed_distribution" in rec:
                rec["v_expect"] = round(float(np.dot(rec["pred_target_speed_distribution"], TFV6_SPEEDS)), 4)
        else:
            fm = a.force_move_post_processor
            rec = {"frame": m["frame"], "k": m["k"], "tick": m["tick"], "step": a.step,
                   "steer": float(ctrl.steer), "throttle": float(ctrl.throttle), "brake": float(ctrl.brake),
                   "stuck_detector": int(fm.stuck_detector), "force_move": int(fm.force_move)}
            for key in keys:
                v = getattr(p, key, None)
                if v is not None:
                    rec[key] = self._np(v)
        self.fh.write(json.dumps(rec) + "\n")

    def summary(self):
        np = self.np
        self.fh.close()
        return {"model": self.model, "ticks": self.ticks, "errors": self.errors, "target_speed_classes": self.cls,
                "ms_mean": {k: round(1e3 * float(np.mean(v)), 2) for k, v in self.t.items() if v},
                "ms_p95": {k: round(1e3 * float(np.percentile(v, 95)), 2) for k, v in self.t.items() if v}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("tfv6", "bridgedrive"), required=True)
    a = ap.parse_args()
    t_start = time.time()
    _imports(a.model)                              # pay the imports while the recorder's world loads
    t_imp = time.time()
    msg = recv()
    assert msg[0] == "init", msg[0]
    sh = Shadow(a.model, msg[1])
    send(("ready", {"sensors": sh.sensors, "import_s": round(t_imp - t_start, 1),
                    "setup_s": round(time.time() - t_imp, 1), "pid": os.getpid()}))
    q = queue.Queue()

    def reader():                                  # drain the pipe at once, so the recorder never waits on us
        try:
            while True:
                m = recv()
                q.put(m)
                if m[0] == "end":
                    return
        except EOFError:
            q.put(("eof",))

    threading.Thread(target=reader, daemon=True).start()
    lag, t_idle = [], 0.0
    while True:
        t0 = time.perf_counter()
        m = q.get()
        t_idle += time.perf_counter() - t0
        if m[0] == "tick":
            lag.append(q.qsize())
            sh.tick(m[1])
            continue
        s = sh.summary()
        s.update(idle_s=round(t_idle, 1), queue_max=max(lag) if lag else 0,
                 queue_mean=round(sum(lag) / len(lag), 1) if lag else 0, wall_s=round(time.time() - t_start, 1))
        if m[0] == "end":
            send(("done", s))
        return


if __name__ == "__main__":
    main()
