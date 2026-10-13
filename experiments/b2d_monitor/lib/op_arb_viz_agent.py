#!/usr/bin/env python
"""Monitor view for the op_arb openpilot agent: lib/op_arb_agent.OpArbAgent unchanged, plus a live preview
(scripts/drive_runtime/preview.py, read by the openpilot layout of b2d_viewer.py) of what the model sees and gets.

Published per frame (every PUBLISH_EVERY ticks), never affecting control:
  image "wide"  the wide camera warped exactly as the server warps it for the model (zero calibration, SBIGMODEL_K: focal 455,
                principal point (256, 151.8), 512 x 256), rendered at SCALE x the model resolution (same geometry, more pixels)
  state         the request the agent sent to the server (desire, speed, intent, ...), the route command and its distance,
                ego state (speed, yaw rate, accel, steer / throttle / brake), the model outputs (plan, lead, pose speed, engaged
                prob, desire state), the arbitration (lateral owner, speed source, latch), and pixel polylines in that image of
                the benchmark route ahead (flat ground), openpilot's plan (its own 3-D path) and the executed (arbitrated) path.
Config and driving: exactly the op_arb agent's, unless the config names "lon_smooth" (LonSmooth below: longitudinal tracks openpilot's
plan speed with a PI loop through the measured MKZ pedal maps instead of the op_arb min(set, IDM, plan) + latch + 5 s timer path; lateral
unchanged). Env B2D_PREVIEW_DIR = the preview directory (unset: no preview).
Relative paths in the config: "socket" / "parity.socket" are under the monitor dir (env B2D_MON_DIR, default $DATA_DIR/runs/b2d_monitor),
"controller_config" is under the repo root; the resolved config is written to <mon dir>/cfg-resolved/ (resolve_paths below).
"""
import sys as _sys, pathlib as _pl  # dirs of the modules this file imports by bare name (lib/op_arb_agent.py's convention)
_repo = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_repo / _d) for _d in ("lib", "scripts", "experiments/route_mem/lib", "experiments/b2d_monitor/lib")]
import json
import math
import os
import sys
import time

import op_arb_agent as OA  # noqa: E402
from op_arb_agent import Z, rigs, wire, np, world_to_local  # noqa: E402

SCALE, PUBLISH_EVERY = 2, 1
F_MODEL, CX_MODEL, CY_MODEL, W_MODEL, H_MODEL = 455.0, 256.0, 0.5 * (256 + 47.6), 512, 256
DESIRE_NAMES = {0: "none", 1: "turn left", 2: "turn right", 3: "lane change left", 4: "lane change right"}
CMD_NAMES = {Z.LEFT: "turn left", Z.RIGHT: "turn right", Z.STRAIGHT: "straight through junction",
             Z.LANEFOLLOW: "follow lane", Z.CHANGE_LEFT: "lane change left", Z.CHANGE_RIGHT: "lane change right"}
_LAST_SENT = {}
_send = wire.send


def _send_capture(sock, meta, arrays=None):
    if isinstance(meta, dict) and meta.get("cmd") == "plan":
        _LAST_SENT.clear()
        _LAST_SENT.update({key: v for key, v in meta.items() if key not in ("cmd", "dump", "seed")})
    return _send(sock, meta, arrays)


wire.send = _send_capture   # the module object the agent calls; same bytes go out


def get_entry_point():
    return "OpArbVizAgent"


class _Tee(object):
    """plans.jsonl writer that keeps the last record for the preview."""
    def __init__(self, fh):
        self.fh, self.last = fh, None

    def write(self, s):
        try:
            self.last = json.loads(s)
        except ValueError:
            pass
        return self.fh.write(s)

    def __getattr__(self, name):
        return getattr(self.fh, name)


def _wide_index(scale):
    """Flat BGRA pixel index of the CARLA wide render for every model-frame pixel (nearest neighbour, as modeld)."""
    w, h = rigs.OP_CAMERA_WH
    f = rigs.OP_FOCAL["wide"]
    u, v = np.meshgrid((np.arange(W_MODEL * scale) + 0.5) / scale - 0.5, (np.arange(H_MODEL * scale) + 0.5) / scale - 0.5)
    # model pixel -> calib ray (rpy = 0): x right = (u - cx) / F, y down = (v - cy) / F, z forward = 1; camera pixel = K_cam ray
    x = np.clip(np.rint(f * (u - CX_MODEL) / F_MODEL + w / 2), 0, w - 1).astype(np.int64)
    y = np.clip(np.rint(f * (v - CY_MODEL) / F_MODEL + h / 2), 0, h - 1).astype(np.int64)
    return (y * w + x).ravel()


def _project(dev, scale=SCALE):
    """Device-frame points at the camera (x forward, y right, z down) -> pixel polyline of the scaled model frame."""
    dev = np.asarray(dev, float)
    dev = dev[dev[:, 0] > 0.8]
    if not len(dev):
        return []
    u = scale * (F_MODEL * dev[:, 1] / dev[:, 0] + CX_MODEL)
    v = scale * (F_MODEL * dev[:, 2] / dev[:, 0] + CY_MODEL)
    return np.round(np.stack([u, v], -1), 1).tolist()


class LonSmooth(object):
    """B2D port of the AlpaSim FIX1 serving semantics (decision 226, experiments/alpasim/lib/serve_fix.py) for a speed-tracking car:
      (a) speed-continuous: the target is the plan's own speed at T_TRACK = 1.0 s, reached by the constant acceleration
          (v_plan(1 s) - v_ego) / 1 s from the ego's speed (serve_fix.join over 1.0 s), capped by the set speed and the route curvature;
      (b) openpilot's lead path: jevdrive/openpilot/lead_long.LeadPlanner (radard filter of the model's lead head + long MPC) with that
          acceleration as the e2e candidate; the served target is the pointwise min with the MPC's speed solution re-anchored at the ego
          speed: removes speed only, no latch;
      (c2) at standstill (v < 0.5 m/s) shipped Cinque never asks to move, so the launch ramp (launch_a up to the set speed) is served
          unless LeadPlanner.inside reports a lead inside openpilot's following distance (then the plan's stop is served); "c2": "plan"
          (an adapter checkpoint, FIX1 JEV_BASE=2's "the adapter's launch") serves the plan's own profile at standstill instead;
      fallbacks: decision 140's driver resume (jevdrive/openpilot/resume.ResumeRule, registered values) and, last, a timer after
          timer_s of held standstill; every firing is logged ("event").
    Pedals from b2d_controller's measured MKZ maps (throttle 0 coasts at -2.8 m/s^2, 0.4 holds speed, the least brake is -3.2 m/s^2):
    a_req >= -brake_below is throttle only (partial throttle = mild deceleration); stronger requests brake along the brake map; crawl
    (< 2 m/s) decelerations brake gently in proportion; a held standstill brakes. Lateral is untouched."""
    DEFAULT = dict(t_track=1.0, ki=0.05, i_lim=0.5, jerk_up=3.0, jerk_down=6.0, a_min=-6.0, a_max=2.0, cruise=8.0, alat=2.0,
                   b_cap=1.5, brake_below=3.0, launch_a=1.2, v_moving=0.5, timer_s=15.0, resume={}, c2="ramp")
    THROTTLE = np.array([(0., -2.8), (.125, -2.0), (.2, -1.65), (.3, -.7), (.4, 0.), (.5, .6), (.6, 1.5), (.7, 2.3), (.75, 3.)])
    BRAKE = np.array([(0., -2.8), (.05, -3.2), (.25, -3.9), (.45, -4.5), (.75, -6.3), (1., -8.)])

    def __init__(self, cfg, route, cam_to_front):
        from jevdrive.openpilot.resume import ResumeRule
        from jevdrive.openpilot.lead_long import LeadPlanner, T_MPC
        self.p = dict(self.DEFAULT, **(cfg if isinstance(cfg, dict) else {}))
        self.rr, self.T_MPC = ResumeRule(**dict(self.p["resume"])), np.asarray(T_MPC, float)
        self.lead, self.gate = LeadPlanner(cam_to_front), LeadPlanner(cam_to_front)
        xy, n = route.xy, len(route.xy)
        k = np.zeros(n)
        for i in range(n):                                   # three-point curvature over about +-5 m of dense route
            a, b = max(i - 5, 0), min(i + 5, n - 1)
            p0, p1, p2 = xy[a], xy[i], xy[b]
            d01, d12, d02 = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p2 - p0)
            cr = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            k[i] = 2 * abs(cr) / max(d01 * d12 * d02, 1e-6)
        self.v_curve = np.sqrt(self.p["alat"] / np.maximum(k, 1e-4))
        self.route = route
        self.I, self.a_prev, self.t_prev, self.hold_t, self.v_prev, self.force_t = 0.0, None, None, 0.0, None, -1e9
        self.log = {}

    def step(self, now, v, out):
        P = self.p
        dt = 0.05 if self.t_prev is None else max(now - self.t_prev, 1e-3)
        a_ego = 0.0 if self.v_prev is None else (v - self.v_prev) / dt
        self.t_prev, self.v_prev = now, v
        v0 = max(float(v), 0.0)
        t, vel = np.asarray(out["t"], float), np.asarray(out["vel"], float)
        lead = np.asarray(out["lead"], float).ravel()
        prob = np.clip(np.asarray(out["lead_prob"], float).ravel(), 1e-6, 1 - 1e-6)
        logit = np.log(prob / (1 - prob))                    # the server decodes the probability; LeadPlanner takes openpilot's logits
        vp1 = float(np.interp(P["t_track"], t, vel))
        i, s = self.route.i, self.route.s
        j = int(np.searchsorted(s, s[i] + 60.0)) + 1
        v_cap = float(np.min(np.sqrt(self.v_curve[i:j] ** 2 + 2 * P["b_cap"] * (s[i:j] - s[i])))) if j > i else P["cruise"]
        event, mode = None, "plan"
        inside = self.gate.inside(v0, lead, logit, float(vel[0]))
        if v0 < P["v_moving"]:                               # (c2) standstill: the launch unless a lead is inside the following distance
            if inside:
                mode = "held_lead"
            elif P["c2"] == "plan":                          # an adapter checkpoint: its own launch is the plan (FIX1 JEV_BASE=2)
                mode = "adapter_launch"
            else:                                            # shipped Cinque never asks to move: the launch ramp
                vp1, mode = max(vp1, v0 + P["launch_a"] * P["t_track"]), "launch"
        pt = float(np.interp(1.0, t, np.asarray(out["pos"], float)[:, 0]))
        fire, why = self.rr.step(now, v0, float(prob[0]), float(lead[0]), pt)   # decision 140 driver resume (registered values)
        if fire:
            vp1, mode = max(vp1, v0 + P["launch_a"] * P["t_track"]), "resume_rule"
            if why == "fire":
                event = "resume_rule_fire"
        held = v0 < P["v_moving"] and vp1 < 0.3
        self.hold_t = self.hold_t + dt if held else 0.0
        if self.hold_t > P["timer_s"]:                       # last fallback: the timer
            self.force_t, self.hold_t, event = now, 0.0, "timer_fallback_fire"
        if now - self.force_t < 3.0:
            vp1, mode, held = max(vp1, v0 + P["launch_a"] * P["t_track"]), "timer_fallback", False
        v_ref = min(vp1, P["cruise"], v_cap)                # (a) speed-continuous target at 1 s
        a_e2e = (v_ref - v0) / P["t_track"]
        vs, li = self.lead.update(int(now * 1e6), v0, a_ego, lead, logit, float(vel[0]), a_e2e)      # (b) openpilot's lead path
        v_lead = max(v0 + float(np.interp(P["t_track"], self.T_MPC, vs)) - float(vs[0]), 0.0)
        src = "lead" if v_lead < v_ref else ("cruise" if v_ref == P["cruise"] else "curve" if v_ref == v_cap else mode)
        v_tgt = min(v_ref, v_lead)
        base = dict(vp1=round(vp1, 2), v_cap=round(min(v_cap, 99.0), 2), v_lead=round(v_lead, 2), mode=mode, inside=bool(inside),
                    rr=why, event=event, lead_d=li.get("d"))
        if held and v_tgt < 0.3:
            self.I, self.a_prev = 0.0, 0.0
            self.log = dict(base, v_tgt=0.0, src="hold", a_req=None, hold=True, hold_t=round(self.hold_t, 1))
            return 0.0, 0.6 if v0 < 0.3 else 0.3
        e = v_tgt - v0
        self.I = float(np.clip(self.I + P["ki"] * e * dt, -P["i_lim"], P["i_lim"]))
        a = float(np.clip(e / P["t_track"] + self.I, P["a_min"], P["a_max"]))
        if self.a_prev is not None:
            a = float(np.clip(a, self.a_prev - P["jerk_down"] * dt, self.a_prev + P["jerk_up"] * dt))
        self.a_prev = a
        if a >= -P["brake_below"] or (v0 < 0.3 and a > -0.3):
            thr, brk = float(np.interp(a, self.THROTTLE[:, 1], self.THROTTLE[:, 0])), 0.0
            if v0 < 2.0 and a < 0:                           # crawl: engine drag is weak, a gentle proportional brake
                thr, brk = 0.0, float(np.clip(-a / 8.0, 0.0, 0.4))
        else:
            thr, brk = 0.0, float(np.interp(-a, -self.BRAKE[:, 1], self.BRAKE[:, 0]))
        self.log = dict(base, v_tgt=round(v_tgt, 2), src=src, a_req=round(a, 2), hold=False, I=round(self.I, 2))
        return thr, brk


def mon_dir():
    """The monitor's run dir: live/ (preview, picker files), srv/ (sockets, server logs), runs/, cfg-resolved/."""
    return _pl.Path(os.environ.get("B2D_MON_DIR") or _pl.Path(os.environ["DATA_DIR"]) / "runs" / "b2d_monitor")


def resolve_paths(path_to_conf_file):
    """Config file with relative socket / controller paths -> the same config with absolute ones (a new file only if one changed)."""
    src, rest = path_to_conf_file.split("+")[0], path_to_conf_file[len(path_to_conf_file.split("+")[0]):]
    with open(src) as fh:
        cfg = json.load(fh)
    mon, changed = mon_dir(), False
    for d, k, root in ((cfg, "socket", mon), (cfg.get("parity") or {}, "socket", mon), (cfg, "controller_config", _repo)):
        if isinstance(d.get(k), str) and not os.path.isabs(d[k]):
            d[k], changed = str(root / d[k]), True
    if not changed:
        return path_to_conf_file
    out = mon / "cfg-resolved" / ("%s.%d.json" % (_pl.Path(src).stem, os.getpid()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=1))
    return str(out) + rest


class OpArbVizAgent(OA.OpArbAgent):
    def setup(self, path_to_conf_file):
        super().setup(resolve_paths(path_to_conf_file))
        self.plan_log = _Tee(self.plan_log)
        d = os.environ.get("B2D_PREVIEW_DIR")
        self.preview = None
        if d:
            from drive_runtime.preview import LivePreview
            self.preview = LivePreview(d)
            self.wide_idx = _wide_index(SCALE)
        self.v_hist, self.viz_err = [], 0
        self.lon_cfg, self.lon, self.lon_log = self.cfg.get("lon_smooth"), None, None
        if self.lon_cfg is not None:
            self.lon_fh = open(os.path.join(self.out, "lon.jsonl"), "w", buffering=1)
        self.stop_cfg, self.stop_sensor, self.stop_hits, self.stop_event, self.off_t = self.cfg.get("stop_on"), None, [], None, None
        self.rsg = self.cfg.get("rsg")                       # RsgMode below: an SH30 + ROUTE-MEM checkpoint served by experiments/b2d_monitor/scripts/rsg_server.py
        self.rsg_feat = None
        if self.rsg is not None:
            import rmem                                      # experiments/route_mem/lib
            import rsg_pack
            self.rmem, self.rsg_hist, self.rsg_pack = rmem, [], rsg_pack.Packer()
            self.curvature = None                            # lateral = the P7 tracker on the served poses, never the action curvature

    # ---- RSG mode (config "rsg": {"warm_s", "rmem_off"}): the PAI serving path of an SH30 + ROUTE-MEM checkpoint (decision 286)
    def _plan(self, speed):
        """Every camera set (20 Hz): extend the ego history; every 0.2 s of sim time: one decision from scripts/rsg_server.py. Its 8 rear-axle
        poses (0.5 .. 4 s) become the P7 tracker's path (lateral, as AlpaSim's controller tracks the trajectory); LonSmooth (FIX1) reads
        the decision's own plan speed / lead heads through self.op_out. History frame: CARLA's left-handed world as right-handed (x, -y),
        yaw -> -yaw, so the t0 frame is x forward, y left, yaw left positive (NAVSIM's); world_to_local is the same map."""
        if self.rsg is None:
            return super()._plan(speed)
        import time as _t
        t_start = _t.perf_counter()
        f, t_frame, cams, _ = self.cam_sets[-1]
        _, xy, yaw = self.poses[-1]
        vy = ay = 0.0
        if self.rsg.get("ego_vy"):                           # arm B: the hero's body-frame lateral velocity / acceleration (simulator truth),
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider as CDP   # NAVSIM sign (y left)
            h = CDP.get_hero_actor()
            tf, vw, aw = h.get_transform(), h.get_velocity(), h.get_acceleration()
            ry = math.radians(tf.rotation.yaw)
            right = (-math.sin(ry), math.cos(ry))            # CARLA's right vector (left-handed world)
            vy, ay = -(vw.x * right[0] + vw.y * right[1]), -(aw.x * right[0] + aw.y * right[1])
        self.rsg_hist.append((t_frame, float(xy[0]), -float(xy[1]), -float(yaw), float(speed), float(vy), float(ay)))
        self.rsg_hist = self.rsg_hist[-60:]
        if self.first_set_t is None:
            self.first_set_t = t_frame
        warm = t_frame - self.first_set_t < float(self.rsg.get("warm_s", 2.0)) - 1e-6
        self.warm_now = warm
        if int(round((t_frame - self.first_set_t) / 0.05)) % 4:
            return 0.0
        H = np.asarray(self.rsg_hist, float)
        tq = t_frame + np.array([-1.5, -1.0, -0.5, 0.0])
        yw = np.unwrap(H[:, 3])
        X, Y, W, V, VY = (np.interp(tq, H[:, 0], c) for c in (H[:, 1], H[:, 2], yw, H[:, 4], H[:, 5]))
        c, s_ = math.cos(W[-1]), math.sin(W[-1])
        dx, dy = X - X[-1], Y - Y[-1]
        P = np.stack([c * dx + s_ * dy, -s_ * dx + c * dy, (W - W[-1] + np.pi) % (2 * np.pi) - np.pi], -1)
        Vb = np.stack([V, VY], -1)                           # body velocity: speedometer, vy = 0 (B2D parity convention) unless ego_vy
        acc = np.array([(V[-1] - V[-2]) / 0.5, float(H[-1, 6])])
        desire = self.route.desire() if self.cfg.get("desire", True) else Z.DESIRE_NONE
        mode = self.rsg.get("cmd_mode", "route")             # arm C: "flip5" = L / R only within 5 m of the turn command; "straight" = S always
        if mode == "flip5":
            c_, d_ = self.route.next_maneuver([Z.LEFT, Z.RIGHT])
            desire = {Z.LEFT: Z.DESIRE_TURN_LEFT, Z.RIGHT: Z.DESIRE_TURN_RIGHT}[c_] if c_ is not None and d_ <= 5.0 else Z.DESIRE_NONE
        elif mode == "straight":
            desire = Z.DESIRE_NONE
        cmd4 = np.zeros(4, np.float32)
        cmd4[{Z.DESIRE_TURN_LEFT: 0, Z.DESIRE_TURN_RIGHT: 2}.get(desire, 1)] = 1.0
        route_rh = self.route.xy * np.array([1.0, -1.0])
        feat = self.rmem.memory([route_rh], (X[-1], Y[-1], W[-1])).astype(np.float32)
        meta = {"cmd": "plan", "t": round(float(t_frame), 2), "speed": float(speed), "desire": int(desire), "P": np.round(P, 4).tolist(),
                "V": np.round(Vb, 4).tolist(), "acc": np.round(acc, 4).tolist(), "cmd4": cmd4.tolist(), "rmem": np.round(feat, 5).tolist(),
                "rmem_off": bool(self.rsg.get("rmem_off", False))}
        tp = _t.perf_counter()
        frame = self.rsg_pack(cams["OP_ROAD"], cams["OP_WIDE"])  # packed here: 0.4 MB on the socket instead of two 9.3 MB renders
        ms_pack = 1e3 * (_t.perf_counter() - tp)
        wire.send(self.sock, meta, {"FRAME": frame})
        info, out = wire.recv(self.sock)
        poses, mu = np.asarray(out["poses"], float), np.asarray(out["mu"], float)
        tt = np.r_[0.0, 0.5 * np.arange(1, 9)]
        px, py = np.r_[0.0, poses[:, 0]], np.r_[0.0, poses[:, 1]]
        ext = OA.TIMES > 4.0
        path = np.stack([np.interp(OA.TIMES, tt, px), np.interp(OA.TIMES, tt, py)], -1)
        path[ext] = path[~ext][-1] + (OA.TIMES[ext, None] - 4.0) * (np.array([px[-1], py[-1]]) - [px[-2], py[-2]]) / 0.5
        lp = 1.0 / (1.0 + np.exp(-np.asarray(out["lead_prob"], float)))
        lead = np.asarray(out["lead"], float)[:72].reshape(3, 6, 4)
        self.op_out = {"t": np.asarray(out["t"], float), "pos": mu[:, 0:3], "vel": mu[:, 3], "acc": mu[:, 6], "lead": lead, "lead_prob": lp}
        self.rsg_feat, self.last_drive, self.rsg_pose = np.asarray(out["rmem"], float), path, (np.asarray(xy, float).copy(), float(yaw))
        accepted = None if warm else self.controller.update(path, t_frame)
        self.n_plans += 1
        r3 = lambda x: np.round(np.asarray(x, float), 3).tolist()  # noqa: E731
        rec = {"frame": f, "t": t_frame, "v": float(speed), "warm": warm, "acc": accepted, "desire": int(desire), "cmd4": cmd4.tolist(),
               "lat": "rsg-path", "lat_why": "warm" if warm else None, "src": "rsg", "lead": r3(lead[0][[0, 1]]), "lp": r3(lp),
               "vplan": r3(np.interp([0, 1, 2, 3, 5], self.op_out["t"], mu[:, 3])), "poses": r3(poses), "ego": r3(out["ego"]),
               "rmem_n": int((self.rsg_feat.reshape(40, 5)[:, 0] > 0.5).sum()), "rmem_on": info.get("rmem_on"), "valid": info.get("valid"),
               "srv_ms": info.get("ms"), "pack_ms": round(ms_pack, 1), "new": info.get("new"), "check": info.get("check"),
               "ms": round(1e3 * (_t.perf_counter() - t_start), 1)}
        # memory geometry check: the fed points vs the dense leaderboard route at the same arc offsets, through world_to_local
        Fm = self.rsg_feat.reshape(40, 5)
        okm = Fm[:, 0] > 0.5
        loc = OA.world_to_local(self.route.xy, xy, yaw)
        sR = self.route.s
        dd = np.hypot(loc[:, 0], loc[:, 1])
        i0 = int(np.argmin(dd[max(self.route.i - 20, 0):self.route.i + 60])) + max(self.route.i - 20, 0)
        ref = np.stack([np.interp(sR[i0] + self.rmem.OFFS, sR, loc[:, k]) for k in (0, 1)], -1)
        rec["rmem_xy"] = r3(Fm[okm, 1:3] * 20.0)
        rec["rmem_dev"] = round(float(np.max(np.hypot(*(Fm[okm, 1:3] * 20.0 - ref[okm]).T))), 3) if okm.any() else None
        self.plan_log.write(json.dumps(rec, default=OA._json_scalar) + "\n")
        return rec["ms"]

    def _rig_to_dev(self, xy, z=0.0):
        """Rig points (rear axle on the ground, x forward, y left) -> camera device frame."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        m = self.op_mount
        return np.stack([xy[:, 0] - m[0], -(xy[:, 1] - m[1]), np.full(len(xy), m[2] - z)], -1)

    # ---- "stop_on": {"offroute_m": 3.5, "offroute_s": 1.0, "grace_s": 0.5}: end the route at the first collision / off-route event
    # (evaluation only, simulator truth, never used for control): the leaderboard then scores the route up to that point
    def _stop_check(self):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider as CDP
        from srunner.scenariomanager.timer import GameTime
        cfg, now = self.stop_cfg, float(GameTime.get_time())
        hero = CDP.get_hero_actor()
        if hero is None:
            return
        if self.stop_sensor is None:
            bp = CDP.get_world().get_blueprint_library().find("sensor.other.collision")
            self.stop_sensor = CDP.get_world().spawn_actor(bp, OA.Z.carla.Transform(), attach_to=hero)
            self.stop_sensor.listen(lambda e: self.stop_hits.append((float(GameTime.get_time()), e.other_actor.type_id)))
        if self.stop_event is None and self.stop_hits:
            t, what = self.stop_hits[0]
            self.stop_event = {"t": round(t, 2), "kind": "collision", "with": what}
        loc = hero.get_location()
        d = float(np.min(np.hypot(self.route.xy[:, 0] - loc.x, self.route.xy[:, 1] - loc.y)))
        self.off_t = (self.off_t or now) if d > float(cfg.get("offroute_m", 3.5)) else None
        if self.stop_event is None and self.off_t is not None and now - self.off_t >= float(cfg.get("offroute_s", 1.0)):
            self.stop_event = {"t": round(now, 2), "kind": "off_route", "dist_m": round(d, 2)}
        if self.stop_event is not None and now - self.stop_event["t"] >= float(cfg.get("grace_s", 0.5)):
            self.stop_event["progress"] = round(float(self.route.s[self.route.i] / max(self.route.s[-1], 1e-3)), 3)
            with open(os.path.join(self.out, "stop_event.json"), "w") as fh:
                json.dump(self.stop_event, fh)
            raise RuntimeError("stop_on: route ended at the first %s (%s)" % (self.stop_event["kind"], self.stop_event))

    def __call__(self):
        if self.stop_cfg is not None:
            self._stop_check()
        control = super().__call__()
        if self.lon_cfg is not None:
            control = self._lon_smooth(control)
        if self.preview is not None and self.tick % PUBLISH_EVERY == 0:
            try:
                self._publish(control)
            except Exception as e:  # the view must never stop the car
                self.viz_err += 1
                if self.viz_err <= 3:
                    print("preview error: %r" % (e,), file=sys.stderr, flush=True)
        return control

    def _lon_smooth(self, control):
        """Replace throttle / brake by LonSmooth once openpilot drives (after the warm-up); steer is untouched."""
        out = getattr(self, "op_out", None)
        if self.warm_now or not out or not self.poses:
            return control
        if self.lon is None:
            self.lon = LonSmooth(self.lon_cfg, self.route, float(getattr(self, "cam_to_bumper", 2.25)))
        now = self.poses[-1][0]
        v = float(self.v_hist[-1][1]) if self.v_hist else 0.0
        rec = self.plan_log.last or {}
        v = float(self.rsg_hist[-1][4]) if self.rsg is not None and self.rsg_hist else float(rec.get("v", v))
        thr, brk = self.lon.step(now, v, out)
        control = OA.Z.carla.VehicleControl(throttle=float(thr), steer=float(control.steer), brake=float(brk))
        self.control = control
        self.lon_log = dict(self.lon.log, t=round(now, 2), v=round(v, 2), thr=round(thr, 3), brk=round(brk, 3))
        self.lon_fh.write(json.dumps(self.lon_log) + "\n")
        return control

    def _publish(self, control):
        if not self.cam_sets or not self.poses:
            return
        now, xy, yaw = self.poses[-1]
        cams = self.cam_sets[-1][2]
        wide = np.asarray(cams["OP_WIDE"]).reshape(-1, 4)[self.wide_idx]
        img = wide[:, 2::-1].reshape(H_MODEL * SCALE, W_MODEL * SCALE, 3)
        speed = float(getattr(self, "last_v", 0.0))
        rec = self.plan_log.last or {}
        speed = float(rec.get("v", speed))
        # ego state from the agent's own pose filter and speedometer (what the base / controller use)
        self.v_hist.append((now, speed, yaw))
        self.v_hist = self.v_hist[-11:]
        (t0, v0, y0), (t1, v1, y1) = self.v_hist[0], self.v_hist[-1]
        dt = max(t1 - t0, 1e-3)
        yaw_rate = math.degrees((y1 - y0 + math.pi) % (2 * math.pi) - math.pi) / dt
        accel = (v1 - v0) / dt
        # route ahead (benchmark dense route, flat ground) and its distance marks
        i = self.route.i
        s0 = self.route.s[i]
        j = int(np.searchsorted(self.route.s, s0 + 120.0)) + 1
        pts = self.route.xy[i:j]
        local = world_to_local(pts, xy, yaw) if len(pts) else np.zeros((0, 2))
        route_px = _project(self._rig_to_dev(local)) if len(local) else []
        marks = []
        for dm in (10, 20, 40, 80):
            k = int(np.searchsorted(self.route.s, s0 + dm))
            if k < len(self.route.xy):
                p = _project(self._rig_to_dev(world_to_local(self.route.xy[k:k + 1], xy, yaw)))
                if p:
                    marks.append([dm] + p[0])
        out = getattr(self, "op_out", None) or {}
        # openpilot's plan is the device's own trajectory (camera height): drop it by the mount height onto the road
        plan_px = _project(np.asarray(out["pos"], float) + [0.0, 0.0, self.op_mount[2]]) if "pos" in out else []
        drive = getattr(self, "last_drive", None)
        drive_px = _project(self._rig_to_dev(np.r_[[[0.0, 0.0]], np.asarray(drive, float)])) if drive is not None else []
        cmd, dist = self.route.next_maneuver([Z.LEFT, Z.RIGHT, Z.STRAIGHT, Z.CHANGE_LEFT, Z.CHANGE_RIGHT])
        sent = dict(_LAST_SENT)
        lead = rec.get("lead") or [[0, 0, 0, 0]]
        state = {
            "agent": "openpilot", "running": True, "t": time.time(), "route": os.environ.get("BENCHMARK_ROUTE_ID", "?"),
            "step": self.tick, "sim_t": round(now, 2), "warm": bool(rec.get("warm", True)),
            "desire": int(sent.get("desire", 0)), "desire_name": DESIRE_NAMES.get(int(sent.get("desire", 0)), "?"),
            "cmd": CMD_NAMES.get(cmd, "follow lane") if cmd is not None else "follow lane (no command left)",
            "cmd_code": int(cmd) if cmd is not None else 0, "cmd_dist": None if dist is None else round(dist, 1),
            "sent": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sent.items()
                     if not isinstance(v, (list, dict)) and k != "parity_socket"},
            "speed": round(speed, 2), "yaw_rate": round(yaw_rate, 2), "accel": round(accel, 2),
            "steer": round(float(control.steer), 3), "throttle": round(float(control.throttle), 3),
            "brake": round(float(control.brake), 3),
            "lat": rec.get("lat"), "lat_why": rec.get("lat_why"), "lon_src": rec.get("src"),
            "s5": rec.get("s"), "latch": rec.get("latch"), "rel": rec.get("rel"),
            "lead_x": round(float(lead[0][0]), 1), "lead_v": round(float(lead[0][2]), 1),
            "lead_p": round(float((rec.get("lp") or [0])[0]), 2), "pose_v": rec.get("pose_v"), "eng": rec.get("eng"),
            "act_k": rec.get("act_k"), "act_a": rec.get("act_a"), "vplan": rec.get("vplan"),
            "ds": rec.get("ds"), "div": rec.get("div"),
            "route_px": route_px, "route_marks": marks, "plan_px": plan_px, "drive_px": drive_px,
            "parity_ego": [round(float(x), 2) for x in sent.get("parity_ego") or []],
            "phase": self._phase(speed, rec), "lon": self.lon_log, "img_scale": SCALE, **self._rmem_overlay(), "config": os.environ.get("VIZ_CONFIG", ""),
            "model": os.environ.get("VIZ_MODEL", "Cinque shipped"),
            "progress": round(float(self.route.s[self.route.i] / max(self.route.s[-1], 1e-3)), 3),
        }
        self.preview.publish(state, {"wide": img})

    def _phase(self, speed, rec):
        """One line for the top bar: WARM-UP <s left> / DRIVING / STOPPED <reason> (LOADING is the viewer's: no fresh state yet)."""
        if self.first_set_t is None or self.warm_now:
            w = float(self.rsg.get("warm_s", 2.0)) if self.rsg is not None else float(self.cfg.get("warmup_s", 5.0))
            left = w - (self.poses[-1][0] - self.first_set_t) if self.first_set_t is not None and self.poses else w
            return "WARM-UP %.1f s left" % max(left, 0.0)
        if speed >= 0.3:
            return "DRIVING"
        lon = self.lon_log or {}
        mode, src = lon.get("mode"), lon.get("src")
        if self.lon is not None:
            rr = self.lon.rr
            wait = ""
            if rr.state == "idle" and rr.stand_since is not None:
                left = rr.p["t_stand"] - (self.poses[-1][0] - rr.stand_since)
                wait = ", driver resume in %.0f s" % left if left > 0.5 else ", driver resume blocked: lead in view"
            timer = ", timer in %.0f s" % max(self.lon.p["timer_s"] - self.lon.hold_t, 0.0) if self.lon.hold_t > 0 else ""
            if mode == "held_lead":
                return "STOPPED: lead inside the following distance%s%s" % (wait, timer)
            if src == "hold" and mode == "adapter_launch":
                return "STOPPED: the model's plan holds (adapter launch pending%s%s)" % (wait, timer)
            if src == "hold":
                return "STOPPED: the plan stops%s%s" % (wait, timer)
            if src == "lead":
                return "STOPPED: lead MPC"
            return "STOPPED: starting (%s)" % (mode or src)
        if rec.get("latch"):
            return "STOPPED: stop latch (plan asked to stop)"
        return "STOPPED"

    def _rmem_overlay(self):
        """The route memory the model received (RSG mode): pixel polyline of its valid points and the 40-point mask (offsets -6 .. +72 m)."""
        if self.rsg_feat is None:
            return {}
        F = self.rsg_feat.reshape(40, 5)
        ok = F[:, 0] > 0.5
        if not ok.any():
            return {"rmem_px": [], "rmem_mask": [bool(x) for x in ok]}
        (x0, y0), w0 = self.rsg_pose                         # decision pose -> world -> the pose of this frame (no 0.2 s display lag)
        c, s_ = math.cos(w0), math.sin(w0)
        L = F[ok, 1:3] * 20.0
        world = np.stack([x0 + c * L[:, 0] + s_ * L[:, 1], y0 + s_ * L[:, 0] - c * L[:, 1]], -1)   # inverse of world_to_local
        _, xy, yaw = self.poses[-1]
        return {"rmem_px": _project(self._rig_to_dev(world_to_local(world, xy, yaw))), "rmem_mask": [bool(x) for x in ok]}

    def destroy(self):
        if getattr(self, "stop_sensor", None) is not None:
            try:
                self.stop_sensor.stop()
                self.stop_sensor.destroy()
            except Exception:
                pass
            self.stop_sensor = None
        try:
            if getattr(self, "preview", None) is not None:
                self.preview.close()
                self.preview = None
        finally:
            super().destroy()
