#!/usr/bin/env python
"""Night queue 4 head agent (todos/2026-09-26-night-queue-4.md, G / X, the [F] entries): scripts/b2d_zeroshot_agent.py's
"model": "head" path unchanged (P4 rig, pose-track ego input, 5 Hz, P7), plus

  "folds": {"split": K's route_split.json, "key": "heads" | "q2_dir", "R1": path, "R2": path, "pick": "unseen" | "seen"}
      cross-fitting: the route's base id (official id, or a G variant id 100 b + 90 + code) picks the fold whose readout
      never saw it ("unseen"; "seen" = the one that did; K's rule: unlabelled routes -> R1), and every plan request carries
      {key: that fold's path} (scripts/nq3_cl_server.py loads the fold's heads.npz / Head dir).
  "x": true
      X: the Q2 mode head's output -> jevdrive.nq4_x.XState -> the path P7 gets (bypass: PDM-Lite's lane-shift geometry
      on the CARLA map's neighbouring lane centres; stop / wait: target speed 0; keep: the head's trajectory). The mode
      comes from our own camera features only; nothing reads CarlaDataProvider.active_scenarios. Every plan is dumped to
      x_dump.jsonl and the route geometry to x_route.npz for the offline check (jevdrive.nq4_x check).

_head_plan is b2d_zeroshot_agent's, with the fold meta added to the request and the X step between the server's answer and
the controller; keep the two in step if the parent's changes.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zeroshot_wire as wire  # noqa: E402
from b2d_zeroshot_agent import DELTA, ZeroShotAgent  # noqa: E402
from jevdrive import nq4_x as NX  # noqa: E402


def get_entry_point():
    return "NQ4Agent"


class NQ4Agent(ZeroShotAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.fold_meta, self.x = {}, None
        folds = self.cfg.get("folds")
        rid = os.environ.get("BENCHMARK_ROUTE_ID", "0")
        if folds:
            split = json.load(open(folds["split"]))["routes"]
            fk = NX.fold_pick(split, NX.base_of(rid), folds.get("pick", "unseen"))
            self.fold_meta = {folds["key"]: folds[fk]}
            with open(os.path.join(self.out, "fold.json"), "w") as fh:
                json.dump({"route": rid, "base": NX.base_of(rid), "fold": fk, "pick": folds.get("pick", "unseen"),
                           "meta": self.fold_meta}, fh)
        if self.cfg.get("x") and getattr(self, "route", None) is not None:
            self._init_x()

    def _init_route(self):
        super()._init_route()
        if getattr(self, "cfg", {}).get("x") and getattr(self, "fold_meta", None) is not None and self.x is None:
            self._init_x()

    def _init_x(self):
        """Neighbouring lane centres of every dense route point, as PDM-Lite's shift_route_smoothly reads them."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        cmap = CarlaDataProvider.get_map()
        P, L, R = [], [], []
        for tf, _ in self._dense_plan:
            wp = cmap.get_waypoint(tf.location)
            P.append([tf.location.x, tf.location.y])
            for side, acc in ((wp.get_left_lane() if wp else None, L), (wp.get_right_lane() if wp else None, R)):
                acc.append([side.transform.location.x, side.transform.location.y] if side is not None else [np.nan, np.nan])
        self.x = NX.XState(P, L, R)
        np.savez(os.path.join(self.out, "x_route.npz"), route=np.asarray(P), left=np.asarray(L), right=np.asarray(R))
        self.x_log = open(os.path.join(self.out, "x_dump.jsonl"), "w", buffering=1)

    def _head_plan(self):
        from jevdrive import nq3_cl as CL
        t_start = time.perf_counter()
        k, f, t_frame, jpgs, desire = self.head_pending
        self.head_pending = None
        xy = np.array([p[0] for p in self.pose_track], float)
        yaw = np.array([p[1] for p in self.pose_track], float)
        ra, th = CL.rh_track(xy, yaw)
        it = CL.intent(self.route.xy, self.route.cmd, self.route.s, xy[k], yaw[k])
        ego = CL.ego_input(CL.ego_past(ra, th, k), it)
        every = int(self.cfg.get("dump_every", 0))
        dump = os.path.join(self.out, "frames", "%06d.npz" % f) if every and self.n_plans % every == 0 else ""
        meta = {"cmd": "plan", "arm": self.cfg["arm"], "desire": int(desire), "frame": int(f), "dump": dump}
        meta.update(self.fold_meta)
        arrays = {"jpg%d" % i: b for i, b in enumerate(jpgs)}
        arrays["ego"] = ego
        wire.send(self.sock, meta, arrays)
        info, out = wire.recv(self.sock)
        path = np.asarray(out["path"], float)
        if self.first_set_t is None:
            self.first_set_t = t_frame
        warm = t_frame - self.first_set_t < self.warmup_s - 1e-6
        self.warm_now = warm
        xinfo = None
        if self.x is not None and not warm:
            head_path = path
            path, xinfo = self.x.step(info.get("mode"), head_path, xy[k], float(yaw[k]))
            self.x_log.write(json.dumps({"frame": int(f), "t": t_frame, "mode": info.get("mode"), "warmup": False,
                                         "pose": [float(xy[k, 0]), float(xy[k, 1]), float(yaw[k])],
                                         "traj": np.asarray(head_path).tolist(), "path": np.asarray(path).tolist(),
                                         "x": xinfo}) + "\n")
        accepted = False if warm else self.controller.update(path, t_frame)
        self.n_plans += 1
        ms = 1e3 * (time.perf_counter() - t_start)
        self.timings["plan_ms"].append(ms)
        rec = {"frame": f, "t": t_frame, "k": k, "speed": float(np.hypot(*(ra[k] - ra[k - 1])) / DELTA) if k else 0.0,
               "accepted": accepted, "path": np.round(path, 3).tolist(), "round_trip_ms": round(ms, 1),
               "pose": [float(xy[k, 0]), float(xy[k, 1]), float(yaw[k])], "intent": it, "desire": int(desire),
               "warmup": bool(warm), "dump": dump}
        rec.update(info)
        if xinfo is not None:
            rec["x"] = xinfo
        rec.update(self._truth())
        self.plan_log.write(json.dumps(rec) + "\n")
        return ms

    def destroy(self):
        if getattr(self, "x_log", None) is not None:
            self.x_log.close()
        super().destroy()
