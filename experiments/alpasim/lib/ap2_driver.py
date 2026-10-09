"""AP2 as an AlpaSim nuPlan-track driver: sh30_driver.py's EgodriverService (session bookkeeping, CAM_F0 packing, output trajectory, the
no-fallback rule: every `drive` is a real inference or a gRPC error) around ap2_core.Core, the model trained for AlpaSim's inputs.

Differences from the SH30 driver, all on the input side (experiments/alpasim/lib/ap2_inputs.py):
  history   decisions with fewer than 4 keyframes use the cold-start rule the checkpoint was TRAINED with (ap2_core; AP2-AB: back-warped
            first frame, the SH30 driver's patch, now in the training distribution)
  ego       DynamicState as delivered: the model was trained on its per-decision definitions. The states of the rollout's start (time <=
            the first `drive`) arrive rotated by -yaw; they are rotated back when the pose difference says so (sh30_driver's test) and, when
            that test cannot tell (below 1 m/s), by the rule read from AlpaSim's source (event_loop._initialize_ego_trajectories), counted
  command   the shipped route rule (as SH30); the 20 route waypoints go to the model as well when the checkpoint is a route arm

Environment: as sh30_driver.py, with AP2_TAG (run tag under $DATA_DIR/runs/op_parity/runs, required) and AP2_COLD (zero | backwarp;
unset = the trained rule); SH30_MOTION / SH30_MOTION_GATE as in sh30_driver.py.
Run with envs/op-train:  AP2_TAG=AP2-A-s0 python experiments/alpasim/lib/ap2_driver.py
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sh30_driver as D  # noqa: E402  (gRPC stubs, the shipped sample's trajectory / route functions, Session, Driver)
import ap2_core as AC  # noqa: E402
import ap2_inputs as AI  # noqa: E402

LOG = logging.getLogger("ap2")
SLOW = 1.0                                             # m/s: below it the pose-difference test of a start state is inconclusive


class Driver(D.Driver):
    def submit_route(self, req, ctx):
        s = self._s(req.session_uuid, ctx)
        wp = np.array([[w.x, w.y] for w in req.route.waypoints], np.float32).reshape(-1, 2)
        with s.lock:
            s.cmd, s.wp, s.line = D.command_from_route(req.route), wp, D.route_line(wp)
            s.route0 = wp[0].tolist() if len(wp) else None
            s.count["cmd_rule_diff"] = s.count.get("cmd_rule_diff", 0) + int(np.argmax(AI.route_cmd(wp)) != np.argmax(s.cmd))
        return D.common_pb2.Empty()

    def drive(self, req, ctx):
        s, now, tq = self._s(req.session_uuid, ctx), int(req.time_now_us), int(req.time_query_us)
        t_in = time.perf_counter()
        with s.lock:
            s.count["drive"] += 1
            s.t_first = getattr(s, "t_first", now)
            ts = [t for t in s.poses if t <= now and t in s.frames]
            t0 = max(ts) if ts else None
            if t0 is None or now - t0 > D.TOL_US:
                s.count["input_error"] += 1
                ctx.abort(D.grpc.StatusCode.FAILED_PRECONDITION, f"no CAM_F0 frame and ego pose at Drive time {now} (latest pair {t0})")
            keys = [t0]
            while len(keys) < 4 and (k := next((t for t in s.frames if abs(t - (keys[-1] - D.STEP_US)) <= D.TOL_US and t in s.poses), None)):
                keys.append(k)
            keys = keys[::-1]
            anchor, yaw0 = s.poses[t0], D.yaw_from_quat(s.poses[t0].pose.quat)
            R0 = D.rot(yaw0).T
            pose = np.array([[*(R0 @ (D._xy(s.poses[k]) - D._xy(anchor))), D.yaw_from_quat(s.poses[k].pose.quat) - yaw0] for k in keys])
            pose[:, 2] = (pose[:, 2] + np.pi) % (2 * np.pi) - np.pi
            dyn = []
            for k in keys:
                v, a, w, r = s.velocity(k)
                if k <= s.t_first and not r and np.linalg.norm(v) < SLOW:      # a start state the test cannot judge: the source's rule
                    Rz = D.rot(D.yaw_from_quat(s.poses[k].pose.quat))
                    v, a, r = Rz @ v, Rz @ a, True
                    s.count["state_rotated_slow"] = s.count.get("state_rotated_slow", 0) + (k == t0)
                dyn.append((v, a, w, r))
            s.count["state_rotated"] += int(dyn[-1][3])
            s.count["cold"] += len(keys) < 4
            frames, cmd = [s.frames[k] for k in keys], np.asarray(s.cmd, np.float32).copy()
            wp = getattr(s, "wp", None)
        t_q = time.perf_counter()
        try:
            with self.gpu:
                t_g = time.perf_counter()
                mo = self.motion(s)
                o = self.core.plan(frames, pose, np.array([d[0] for d in dyn]), dyn[-1][1], cmd, s.cam["t"], yaw_rate=dyn[0][2], lht=self.lht, wp=wp, motion=mo)
        except Exception as e:
            s.count["inference_error"] += 1
            LOG.exception("inference failed, session %s t %d", req.session_uuid, now)
            ctx.abort(D.grpc.StatusCode.INTERNAL, f"AP2 inference failed: {e!r}")
        s.count["inference"] += 1
        plan = D.make_cached_plan(t0, anchor, o["poses"])
        traj = D.build_trajectory_from_plan(plan, anchor, now, tq)
        t_out = time.perf_counter()
        rec = {"kind": "drive", "t": time.time(), "session": req.session_uuid, "scene": s.scene, "now": now, "t0": t0, "k": s.count["drive"] - 1,
               "n_keys": len(keys), "n_slots": int(o["valid"].sum()), "cmd": int(np.argmax(cmd)), "route0": getattr(s, "route0", None), "motion": mo,
               "n_wp": None if wp is None else int((~np.isnan(wp[:, 0])).sum()), "rotated": [bool(d[3]) for d in dyn], "w": float(dyn[-1][2]),
               "ego": o["ego"][:20].round(5).tolist(), "hist": o["hist"].round(4).tolist(), "anchor": [anchor.pose.vec.x, anchor.pose.vec.y, yaw0],
               "poses": o["poses"].round(4).tolist(), "n_out": len(traj.poses),
               "ms": {**{k: round(v, 2) for k, v in o["ms"].items()}, "prep": round(1e3 * (t_q - t_in), 2), "wait": round(1e3 * (t_g - t_q), 2),
                      "total": round(1e3 * (t_out - t_in), 2)}}
        self.out.write(json.dumps(rec) + "\n")
        if s.n < self.dump:
            np.savez_compressed(self.dir / "dump" / f"s{s.n:02d}_k{rec['k']}.npz", cur=o["cur"], valid=o["valid"], mu=o["mu"], poses=o["poses"],
                                ego=o["ego"], cam_t=s.cam["t"], scene=s.scene, jpeg=np.frombuffer(s.jpeg.get(t0, b""), np.uint8))
        return D.egodriver_pb2.DriveResponse(trajectory=traj)


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    host, port = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0"), int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    log_dir = Path(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver"))
    t0 = time.time()
    core = AC.Core(os.environ["AP2_TAG"], os.environ.get("SH30_DEVICE", "cuda"), os.environ.get("AP2_COLD", ""), float(os.environ.get("SH30_MOTION", "1")))
    z = np.zeros(AC.C.FRAME, np.uint8)
    for m in (1, 2, 3, 4, 4):                           # warm-up: every slot count compiled before the port opens
        core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
    LOG.info("%s (%s, route arm %s) ready in %.1f s, VRAM %.2f GiB", core.tag, core.cold, core.route, time.time() - t0, torch.cuda.max_memory_allocated() / 2**30)
    drv = Driver(core, log_dir, int(os.environ.get("SH30_DUMP", "0")), os.environ.get("SH30_LHT", "0") == "1", os.environ.get("SH30_MOTION_GATE", ""))
    server = D.grpc.server(ThreadPoolExecutor(max_workers=int(os.environ.get("ALPASIM_DRIVER_GRPC_WORKERS", "8"))))
    D.egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(drv, server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    server.start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: server.stop(grace=0.0))
    LOG.info("listening on %s:%d", host, port)
    try:
        server.wait_for_termination()
    finally:
        (log_dir / "vram.json").write_text(json.dumps({"max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                                       "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}))


if __name__ == "__main__":
    main()
