#!/usr/bin/env python
"""Bench2Drive agent for the openpilot closed-loop integration study: a small route follower (the "base") and openpilot
Cinque as a modifier, with several arbitration modes. Plan and registration: todos/2026-09-28-op-closedloop.md;
write-up: research/openpilot-closedloop-integration.md. Python 3.8, route process env (envs/carla or b2d-tcp).

It is scripts/b2d_zeroshot_agent.py (the CL2 openpilot path: native road + wide rig rendered every tick, 5 s warm-up,
rear-axle plan transform, route turn / lane-change desire, P7 controller fed at 5 Hz) with `_plan` replaced. Every
mode runs the same openpilot session on the same frames; only what reaches P7 differs:

  native   openpilot's own plan (CL2 as registered in night queue 3)
  base     the base alone, openpilot only logged (the ablation: openpilot disabled)
  oshadow  diagnosis only, never scored as a candidate: the base path with a privileged speed governor (ground-truth
           lead / walker in the path and red lights on the route), openpilot in shadow
  acc      base path; speed = min(base profile, IDM on openpilot's lead head)             (openpilot chill mode)
  e2e      acc + openpilot's plan as a speed constraint while moving, with a stop latch     (openpilot experimental mode)
  switch   openpilot's own plan (lateral + longitudinal) while rolling outside turn zones; e2e for launches and route turns
           ("zones": false: openpilot's plan also through route turns, i.e. only launches and stop latches are the base's)
  drive    op-drive (todos/2026-09-29-op-drive.md): openpilot drives lateral on lane-follow route segments ("lat": "op";
           executed as P7 tracking its plan path, "lat_exec": "p7", or its desired curvature through the bicycle model
           at 20 Hz, "curv"); the route geometry steers in zones around route commands ("zone_m": {cmd: [before, after]})
           and while openpilot's path and the route diverge (> div_m at div_arc m ahead, back below div_back_m for
           div_hold_s). Longitudinal "lon": "op" = min(set speed and curvature cap, lead-head IDM, openpilot plan while
           rolling), "base" = the set-speed governor alone. "hold": "intent" latches a standstill only when the plan itself
           asked to stop (plan v(5 s) < 0.5 m/s) just before it; released by the plan (x@5 s > release_th for release_s),
           a departing lead (lead head v > lead_go_v) or the driver's resume after latch_max_s of standstill.
  Any mode: "coast_v" > 0 replaces a brake request by coasting below coast_v m/s while the arbitrated profile still
  asks to move (CARLA's MKZ stops dead on a light brake at 1-2 m/s).

The base (observable inputs only: the route every Bench2Drive agent gets, the sensor pose, the speedometer) is the
route geometry (b2d_controller_adapter.RouteAdapter's rejoin path) timed by a speed governor: a set speed ("cruise"),
lateral acceleration <= alat on the path's curvature, accel <= amax, and a stop at the route end. It has no perception.

Config: the b2d_zeroshot_agent keys, plus "arb": {"mode", "cruise", "alat", "amax", "bmax", "twin", "lead_p",
"plan_vmin", "plan_form" ("abs" | "rel"), "plan_gate" ("always" | "brake": only while meta brake_press > brake_th), "meta_k" (meta time slot: 0 = now, 1 = 2 s),
"release" ("none" | "gas" | "planx" | "nobrake"), "release_th", "latch_max_s",
"zone_before_m", "zone_after_m", "cruise_by_route" ({route id: set speed})}. Per plan (every tick) one line in plans.jsonl with openpilot's heads, the base and
arbitration state, and ground-truth context (evaluation only; only mode oshadow reads it for control).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2d_zeroshot_agent as Z  # noqa: E402
import zeroshot_rigs as rigs  # noqa: E402
import zeroshot_wire as wire  # noqa: E402
from b2d_controller_adapter import RouteAdapter, world_to_local  # noqa: E402

TIMES = np.arange(1, 21) * 0.25
REAR_TO_BUMPER = 1.3886 + 2.4508       # MKZ 2020: rear axle -> centre -> front bumper (m)
CAM_TO_BUMPER = REAR_TO_BUMPER - rigs.OP_MOUNT_RIG[0]   # openpilot's lead x is measured from the camera
DT_SIM, HORIZON = 0.05, 5.0
MODES = ("native", "base", "oshadow", "acc", "e2e", "switch", "drive")
DEFAULTS = {"mode": "native", "cruise": 8.0, "alat": 2.0, "amax": 1.5, "bmax": 3.0, "twin": False, "lead_p": 0.5,
            "plan_vmin": 1.0, "plan_form": "abs", "release": "none", "release_th": 0.5, "latch_max_s": 20.0,
            "plan_gate": "always", "brake_th": 0.5, "meta_k": 0, "zones": True, "release_s": 0.0,
            "zone_before_m": 15.0, "zone_after_m": 5.0, "idm_s0": 2.5, "idm_T": 1.2, "idm_b": 2.0,
            "lat": "route", "lat_exec": "p7", "lon": "op", "hold": "any", "lead_go_v": 1.0, "coast_v": 0.0,
            "zone_m": None, "div_m": 1.0, "div_back_m": 0.5, "div_arc": 15.0, "div_hold_s": 1.0}
DRIVE_ZONES = {Z.LEFT: (15.0, 5.0), Z.RIGHT: (15.0, 5.0), Z.STRAIGHT: (5.0, 5.0),
               Z.CHANGE_LEFT: (5.0, 10.0), Z.CHANGE_RIGHT: (5.0, 10.0)}


def get_entry_point():
    return "OpArbAgent"


# ------------------------------------------------------------------------------------------------ profiles (numpy)
def arc(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def place(path, s):
    """Positions at arc lengths s on a polyline (straight extrapolation past its end)."""
    a = arc(path)
    if a[-1] < 1e-3:
        return np.repeat(path[:1], len(s), 0)
    out = np.stack([np.interp(s, a, path[:, k]) for k in range(2)], -1)
    over = s > a[-1]
    if over.any():
        d = path[-1] - path[max(len(path) - 2, 0)]
        d = d / max(np.linalg.norm(d), 1e-6)
        out[over] = path[-1] + (s[over] - a[-1])[:, None] * d
    return out


def governor(path, v0, cruise, alat, amax, bmax, end_stop):
    """Base speed profile: arc length at TIMES from the curvature cap, the set speed and (end_stop) a stop at the path end."""
    a = arc(path)
    grid = np.arange(0.0, max(a[-1], 1.0) + 1e-6, 1.0)
    p = place(path, grid)
    h = np.unwrap(np.arctan2(np.gradient(p[:, 1]), np.gradient(p[:, 0]))) if len(p) > 2 else np.zeros(len(p))
    k = np.abs(np.gradient(h)) if len(p) > 2 else np.zeros(len(p))
    k = np.convolve(k, np.ones(3) / 3, "same")[: len(p)]      # "same" returns max(len, 3) points
    cap = np.minimum(cruise, np.sqrt(alat / np.maximum(k, 1e-4)))
    if end_stop:
        cap[-1] = 0.0
    for i in range(len(cap) - 2, -1, -1):
        cap[i] = min(cap[i], math.sqrt(cap[i + 1] ** 2 + 2 * bmax * (grid[i + 1] - grid[i])))
    return integrate(v0, lambda s, v: np.interp(s, grid, cap, right=cap[-1]), amax, bmax)


def integrate(v0, vcap, amax, bmax):
    s, v, out = 0.0, max(v0, 0.0), []
    n = int(round(HORIZON / DT_SIM))
    for i in range(1, n + 1):
        c = vcap(s, v)
        v = min(max(c, v - bmax * DT_SIM), v + amax * DT_SIM)
        v = max(v, 0.0)
        s += v * DT_SIM
        if i % 5 == 0:
            out.append(s)
    return np.array(out)


def idm(v0, gap0, v_lead, a_lead, cruise, amax, b, s0, T, amin=-6.0):
    """IDM ego arc length at TIMES behind a lead at bumper gap gap0 moving at v_lead with accel a_lead (decays over 1 s)."""
    s, v, sl, vl, out = 0.0, max(v0, 0.0), gap0, max(v_lead, 0.0), []
    n = int(round(HORIZON / DT_SIM))
    for i in range(1, n + 1):
        t = i * DT_SIM
        al = a_lead * math.exp(-t)
        vl = max(vl + al * DT_SIM, 0.0)
        sl += vl * DT_SIM
        gap = max(sl - s, 0.05)
        star = s0 + max(0.0, v * T + v * (v - vl) / (2 * math.sqrt(amax * b)))
        acc = amax * (1 - (v / max(cruise, 0.1)) ** 4 - (star / gap) ** 2)
        v = max(v + max(acc, amin) * DT_SIM, 0.0)
        s = min(s + v * DT_SIM, max(sl - 0.3, s))
        if i % 5 == 0:
            out.append(s)
    return np.array(out)


def plan_arc(path_rear, v_ego, v_plan0, form):
    """openpilot's plan as an arc-length profile: "abs" its own positions; "rel" its speed change relative to its own
    current-speed estimate, applied to the measured speed (removes a vision speed bias)."""
    s = np.maximum.accumulate(arc(np.r_[[[0.0, 0.0]], path_rear])[1:])
    if form == "rel":
        s = np.maximum.accumulate(np.maximum(s + (v_ego - v_plan0) * TIMES, 0.0))
    return s


class OpArbAgent(Z.ZeroShotAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.arb = dict(DEFAULTS, **self.cfg.get("arb", {}))
        by_route = self.arb.get("cruise_by_route") or {}      # matched-speed control: a set speed per route id
        rid = os.environ.get("BENCHMARK_ROUTE_ID", "")
        if rid in by_route:
            self.arb["cruise"] = float(by_route[rid])
        assert self.arb["mode"] in MODES and not self.alpamayo and not self.head and not self.replay, self.arb
        self.latch = self.moved = False
        self.latch_t = self.stop_t = 0.0
        self.binding_t = -1e9
        self.last_src = None
        self.blend_from, self.blend_t = None, -1e9
        self.ctx_frame, self.ctx = None, {}
        self.div_on, self.agree_t, self.intent_t, self.want_go, self.lat_src = False, 0.0, -1e9, True, "route"
        if self.arb["lat_exec"] == "curv":
            self.lateral = "curvature"                         # the parent's tick loop steers from self.curvature
        if self.arb["coast_v"] > 0:
            self.rules = "coast"                               # the parent's post-controller hook -> _k_rules below

    def _init_route(self):
        super()._init_route()
        self.base = RouteAdapter(self.route.xy, 10.0)
        a = dict(DEFAULTS, **self.cfg.get("arb", {}))
        spec = {int(k): v for k, v in a["zone_m"].items()} if a["zone_m"] else DRIVE_ZONES if a["mode"] == "drive" \
            else {Z.LEFT: (a["zone_before_m"], a["zone_after_m"]), Z.RIGHT: (a["zone_before_m"], a["zone_after_m"])}
        zones, cmd, s = [], self.route.cmd, self.route.s
        i = 0
        while i < len(cmd):                                  # command runs -> [start - before, end + after]
            if cmd[i] in spec:
                j = i
                while j + 1 < len(cmd) and cmd[j + 1] == cmd[i]:
                    j += 1
                zones.append((s[i] - spec[cmd[i]][0], s[j] + spec[cmd[i]][1]))
                i = j + 1
            else:
                i += 1
        self.zones = zones if a["zones"] else []
        self.tl_stops = None                                 # ground truth, built lazily (evaluation / oshadow only)

    def in_zone(self):
        s = self.route.s[self.route.i]
        return any(a <= s <= b for a, b in self.zones)

    # ------------------------------------------------------------------------ ground-truth context (evaluation only)
    def _truth(self):
        rec = super()._truth()
        rec["ctx"] = self._ctx()
        return rec

    def _ctx(self):
        from srunner.scenariomanager.timer import GameTime
        frame = int(GameTime.get_frame())
        if frame == self.ctx_frame:
            return self.ctx
        self.ctx_frame, self.ctx = frame, {}
        try:
            self.ctx = self._ctx_now()
        except Exception as exc:  # noqa: BLE001 - evaluation only
            self.ctx = {"err": str(exc)[:80]}
        return self.ctx

    def _ctx_now(self):
        import carla
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider as CDP
        hero = CDP.get_hero_actor()
        tf = hero.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        rear = np.array([tf.location.x, tf.location.y]) + self.rear_offset * np.array([math.cos(yaw), math.sin(yaw)])
        i0 = self.route.i
        win = self.route.xy[max(i0 - 5, 0):i0 + 120]
        ws = self.route.s[max(i0 - 5, 0):i0 + 120]
        s_ego = ws[int(np.argmin(np.linalg.norm(win - rear, axis=1)))]
        out = {"junc": int(CDP.get_map().get_waypoint(tf.location).is_junction)}
        lead, ped = (None, None), (None, None)
        for a in CDP.get_all_actors():
            tid = a.type_id
            if not (tid.startswith("vehicle.") or tid.startswith("walker.")) or a.id == hero.id:
                continue
            loc = a.get_location()
            p = np.array([loc.x, loc.y])
            if np.linalg.norm(p - rear) > 70:
                continue
            j = int(np.argmin(np.linalg.norm(win - p, axis=1)))
            d = float(np.linalg.norm(win[j] - p))
            ds = float(ws[j] - s_ego)
            ext = a.bounding_box.extent
            gap = ds - REAR_TO_BUMPER - ext.x
            if ds <= 0 or gap < -3:
                continue
            vel = a.get_velocity()
            spd = math.hypot(vel.x, vel.y)
            if tid.startswith("vehicle.") and d - ext.y < 1.2 and (lead[0] is None or gap < lead[0]):
                lead = (gap, spd)
            if tid.startswith("walker.") and d < 2.5 and (ped[0] is None or gap < ped[0]):
                ped = (gap, spd)
        if lead[0] is not None:
            out.update(lead_gap=round(lead[0], 2), lead_v=round(lead[1], 2))
        if ped[0] is not None:
            out.update(ped_gap=round(ped[0], 2), ped_v=round(ped[1], 2))
        if self.tl_stops is None:                            # route stop lines of every traffic light, once
            self.tl_stops = []
            for tl in CDP.get_world().get_actors().filter("*traffic_light*"):
                for w in tl.get_stop_waypoints():
                    q = np.array([w.transform.location.x, w.transform.location.y])
                    dd = np.linalg.norm(self.route.xy - q, axis=1)
                    j = int(np.argmin(dd))
                    if dd[j] < 2.0:
                        self.tl_stops.append((float(self.route.s[j]), tl))
        ahead = [(s - s_ego - REAR_TO_BUMPER, tl) for s, tl in self.tl_stops if -REAR_TO_BUMPER < s - s_ego < 80]
        if ahead:
            dist, tl = min(ahead, key=lambda x: x[0])
            st = tl.get_state()
            out.update(tl_dist=round(dist, 2), tl=0 if st == carla.TrafficLightState.Green else
                       1 if st == carla.TrafficLightState.Yellow else 2 if st == carla.TrafficLightState.Red else -1)
        return out

    # ------------------------------------------------------------------------ per plan
    def _plan(self, speed):
        import time
        t_start = time.perf_counter()
        A = self.arb
        f, t_frame, cams, _ = self.cam_sets[-1]
        now_xy, now_yaw = self.poses[-1][1], self.poses[-1][2]
        desire = self.route.desire() if self.cfg.get("desire", True) else Z.DESIRE_NONE
        meta = {"cmd": "plan", "seed": 0, "dump": "", "speed": speed, "t": t_frame, "desire": desire,
                "twin": bool(A["twin"])}
        wire.send(self.sock, meta, dict(cams))
        info, out = wire.recv(self.sock)
        op_path = Z.resample(out["t"], rigs.openpilot_plan_to_rig(out["pos"], out["yaw"], self.op_mount), TIMES)
        if self.first_set_t is None:
            self.first_set_t = t_frame
        warm = t_frame - self.first_set_t < self.warmup_s - 1e-6
        self.warm_now = warm
        dt = DELTA_PLAN = Z.DELTA * self.plan_every
        # ---- base geometry and profile
        self.base.project(now_xy, now_yaw, speed, t_frame)
        try:
            world = self.base._rejoin_path(now_xy, now_yaw)
            bpath = world_to_local(world, now_xy, now_yaw)
        except ValueError:
            bpath = np.array([[0.0, 0.0], [1.0, 0.0]])
        ba = arc(bpath)
        bpath = bpath[: max(int(np.searchsorted(ba, 80.0)) + 1, 2)]
        end_stop = ba[-1] < 80.0
        s_base = governor(bpath, speed, A["cruise"], A["alat"], A["amax"], A["bmax"], end_stop)
        # ---- openpilot longitudinal signals
        lead = np.asarray(out["lead"])[0]                    # lead now: (6 times, x y v a)
        lp = float(np.asarray(out["lead_prob"])[0])
        v_plan = np.asarray(out["vel"], float)
        cons = {"base": s_base}
        mode = A["mode"]
        op_lon = mode in ("acc", "e2e", "switch") or (mode == "drive" and A["lon"] == "op")
        if lp > A["lead_p"] and op_lon:
            cons["lead"] = idm(speed, float(lead[0, 0]) - CAM_TO_BUMPER, float(lead[0, 2]), float(lead[0, 3]),
                               A["cruise"], A["amax"], A["idm_b"], A["idm_s0"], A["idm_T"])
        s_plan = plan_arc(op_path, speed, float(v_plan[0]), A["plan_form"])
        if mode == "oshadow":
            cons.pop("lead", None)
            cons.update(self._oracle_cons(speed))
        use_plan = mode in ("e2e", "switch") or (mode == "drive" and A["lon"] == "op")
        # ---- stop latch (e2e / switch): a stop the plan caused is held until openpilot releases it
        if speed < 0.2:
            self.stop_t += dt
        else:
            self.stop_t = 0.0
        if speed > 1.0:
            self.moved = True
        # the plan constrains while rolling, and keeps constraining a stop it started down to standstill
        mt0 = np.asarray(out["meta"], float)
        gi, bi = 31 + 4 * A["meta_k"], 32 + 4 * A["meta_k"]  # meta gas / brake press at t = 0, 2, 4 ... s (k = 0, 1, 2 ...)
        gate = A["plan_gate"] == "always" or mt0[bi] > A["brake_th"]      # "brake": P(driver brakes) from the meta head
        plan_on = use_plan and not warm and gate and (speed >= A["plan_vmin"] or t_frame - self.binding_t < 1.5)
        if plan_on:
            cons["plan"] = s_plan
            if speed >= A["plan_vmin"] and s_plan[-1] < min(v[-1] for k, v in cons.items() if k != "plan") - 0.5:
                self.binding_t = t_frame                       # refreshed only while rolling: a standstill plan never binds
                if float(np.interp(5.0, out["t"], v_plan)) < 0.5:
                    self.intent_t = t_frame                    # the binding plan itself asks to stop
        rel = None
        caused = t_frame - (self.intent_t if A["hold"] == "intent" else self.binding_t) < (2.0 if A["hold"] == "intent" else 1.5)
        if use_plan and not self.latch and self.moved and self.stop_t > 0 and caused:
            self.latch, self.latch_t, self.rel_t = True, 0.0, 0.0
        if self.latch:
            self.latch_t += dt
            gas = float(mt0[gi])
            x5 = float(op_path[-1, 0])
            rel = {"gas": gas > A["release_th"], "planx": x5 > A["release_th"], "nobrake": mt0[bi] < A["release_th"],
                   "none": False}[A["release"]]
            self.rel_t = self.rel_t + dt if rel else 0.0          # the release signal must hold release_s seconds
            lead_go = A["hold"] == "intent" and lp > A["lead_p"] and float(lead[0, 2]) > A["lead_go_v"]
            why = "signal" if rel and self.rel_t >= A["release_s"] else "lead_go" if lead_go else \
                "timeout" if self.latch_t > A["latch_max_s"] else "rolling" if speed > 1.0 else None
            if why:
                self.latch, rel = False, why
            else:
                cons["latch"] = np.zeros(len(TIMES))
        src = min(cons, key=lambda k: cons[k][-1] + 1e-3 * (k == "base"))
        s_fin = np.maximum.accumulate(np.maximum(np.min(np.stack(list(cons.values())), 0), 0.0))
        # ---- lateral owner (drive): openpilot on lane-follow segments; the route in command zones and on divergence
        div = float(np.linalg.norm(place(np.r_[[[0.0, 0.0]], op_path], np.array([A["div_arc"]]))[0]
                                   - place(bpath, np.array([A["div_arc"]]))[0]))
        if div > A["div_m"]:
            self.div_on, self.agree_t = True, 0.0
        elif self.div_on:
            self.agree_t = self.agree_t + dt if div < A["div_back_m"] else 0.0
            self.div_on = self.agree_t < A["div_hold_s"]
        lat_why = "warm" if warm else "zone" if self.in_zone() else "div" if self.div_on else None
        lat_src = "op" if mode == "drive" and A["lat"] == "op" and lat_why is None else "route"
        geom = np.r_[[[0.0, 0.0]], op_path] if lat_src == "op" and A["lat_exec"] == "p7" else bpath
        arb_path = place(geom, s_fin)
        self.want_go = bool(s_fin[7] - s_fin[3] > 0.5) and not warm   # the profile moves >= 0.5 m/s at 1-2 s
        if mode == "drive":
            self.curvature = float(info["curvature"]) if lat_src == "op" and A["lat_exec"] == "curv" else None
        if mode == "native":
            drive_path, src = op_path, "op"
        elif mode == "switch" and not warm and speed >= 2.0 and not self.in_zone() and not self.latch:
            drive_path, src = op_path, "op"
        else:
            drive_path = arb_path
        key = lat_src if mode == "drive" else src == "op"
        if mode in ("switch", "drive"):                       # 1 s linear hand-over between the two drivers
            if self.last_src is not None and self.last_src != key:
                self.blend_from, self.blend_t = self.last_drive, t_frame
            w = min((t_frame - self.blend_t) / 1.0, 1.0)
            if w < 1.0 and self.blend_from is not None:
                drive_path = w * drive_path + (1 - w) * self.blend_from
        self.last_src, self.last_drive, self.lat_src = key, np.asarray(drive_path, float), lat_src
        if warm:
            accepted = False
        elif self.n_plans % self.ctl_every == 0:
            accepted = self.controller.update(np.asarray(drive_path, float), t_frame)
        else:
            accepted = None
        self.n_plans += 1
        ms = 1e3 * (time.perf_counter() - t_start)
        self.timings["plan_ms"].append(ms)
        r3 = lambda x: np.round(np.asarray(x, float), 3).tolist()  # noqa: E731
        mt = np.asarray(out["meta"], float)
        rec = {"frame": f, "t": t_frame, "v": speed, "warm": warm, "acc": accepted, "desire": desire, "src": src,
               "zone": self.in_zone(), "latch": self.latch, "rel": rel, "ri": int(self.route.i),
               "lat": lat_src, "lat_why": lat_why, "div": round(div, 2), "go": self.want_go,
               "cmd": self.route.next_maneuver([Z.LEFT, Z.RIGHT, Z.STRAIGHT, Z.CHANGE_LEFT, Z.CHANGE_RIGHT]),
               "s": {k: round(float(v[-1]), 2) for k, v in cons.items()}, "s2": {k: round(float(v[7]), 2) for k, v in cons.items()},
               "op_xy": r3(op_path[[3, 7, 11, 19]]), "base_xy": r3(place(bpath, np.array([5.0, 10, 15, 20, 30]))),
               "vplan": r3(np.interp([0, 1, 2, 3, 5], out["t"], v_plan)), "aplan": r3(np.interp([0, 1, 2], out["t"], out["acc"])),
               "act_a": round(float(info["accel"]), 3), "act_k": round(float(info["curvature"]), 5),
               "lead": r3(lead[[0, 1], :]), "lp": r3(out["lead_prob"]), "pose_v": round(float(np.asarray(out["pose"])[0]), 3),
               "gas": r3(mt[31:55:4]), "brk": r3(mt[32:55:4]), "hb3": r3(mt[4:31:6]), "eng": round(float(mt[0]), 3),
               "ds": r3(out["desire_state"]), "dp": r3(np.asarray(out["desire_pred"])[:, :3]), "lane": r3(out["lane_prob"]),
               "ms": round(ms, 1)}
        if A["twin"]:
            tw = Z.resample(out["t"], rigs.openpilot_plan_to_rig(out["twin_pos"], out["twin_yaw"], self.op_mount), TIMES)
            rec.update(tw_xy=r3(tw[[3, 7, 11, 19]]), tw_dp=r3(np.asarray(out["twin_desire_pred"])[:, :3]))
        rec["ctx"] = self._ctx()
        self.plan_log.write(json.dumps(rec) + "\n")
        return ms

    def _k_rules(self, speed, throttle, brake):
        """coast_v: below coast_v m/s a brake request becomes a coast while the arbitrated profile still moves and no
        stop is latched; a profile that stops (lead, plan stop, route end) still brakes."""
        if brake > 0 and speed < self.arb["coast_v"] and self.want_go and not self.latch:
            return 0.0, 0.0, "coast"
        return throttle, brake, None

    def _oracle_cons(self, speed):
        """oshadow: privileged governor from ground truth (lead / walker in the path, red or yellow light ahead)."""
        A, c, out = self.arb, self._ctx(), {}
        idm_ = lambda gap, v: idm(speed, gap, v, 0.0, A["cruise"], A["amax"], A["idm_b"], A["idm_s0"], A["idm_T"])  # noqa: E731
        if "lead_gap" in c:
            out["t_lead"] = idm_(c["lead_gap"], c["lead_v"])
        if "ped_gap" in c and c["ped_gap"] < 25:
            out["t_ped"] = idm_(c["ped_gap"], 0.0)
        if c.get("tl") in (1, 2) and c.get("tl_dist", 99) > max(0.5, speed * speed / 8.0 - 1.0):
            out["t_light"] = idm_(c["tl_dist"] + A["idm_s0"] - 0.5, 0.0)
        return out
