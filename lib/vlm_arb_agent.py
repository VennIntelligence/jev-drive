"""Bench2Drive agent: op-drive (frozen openpilot Cinque) plus a VLM slow channel behind an arbitration table R1-R5.

Protocol and registered parameters: experiments/vlm_arb/plans/2026-10-02-vlm-arb.md. Python 3.8 (route process env).

Environment (set per unit by experiments/vlm_arb/scripts/vlm_arb_chain.py):
  VLM_ARM           drive | dslow | jslow | vred | vbyp | vall. drive / dslow open no row (dslow gets its per-route
                    set speed through the config's cruise_by_route).
  VLM_ROWS          comma list that restricts the arm's rows (Phase A gating), e.g. "R1,R4,R5".
  VLM_SHADOW        1: ask the VLM and log its answers even when no VLM row is open (Phase A).
  VLM_SAVE_FRAMES   1: save the wide and road frame of every request as JPEG under <attempt>/vlm_frames.
  VLM_L             seconds of simulation time before an answer may be used (registered: Phase A p95).
  VLM_LIGHT_VARIANT Q_light input variant (lib/vlm_protocol.VARIANTS); a second, Q_light-only request.
  VLM_ORACLE        truth: answers are the ground-truth labels (integration test of the table, debug routes only);
                    stuckred: as truth but Q_light is always red (exercises R5).
  VLM_ENDPOINT      System One URL.

Logs in the attempt dir: vlm_decisions.jsonl, one line per request {"k": "a", t_q, t_eff, gt, ans} and one line per
0.5 s {"k": "s", t, v, rules, ...}; plans.jsonl gets the table state under pc.vlm.
"""
import json
import math
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
REPO = _p.parents[1]
for _d in (str(REPO / "lib"), str(REPO / "scripts")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from op_arb_agent import OpArbAgent, REAR_TO_BUMPER, idm  # noqa: E402
from b2d_privileged_geometry import Privileged, project  # noqa: E402
from vlm_client import VLMClient, jpeg  # noqa: E402


def _js(value):
    """NumPy scalars at the log boundary (a numpy bool in a log line crashed the agent in decision 82)."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("not JSON serializable: %s" % type(value).__name__)


def get_entry_point():
    return "VlmArbAgent"


# Registered arbitration parameters (plan section 2.4). L is replaced by VLM_L once Phase A measured it.
ARB_PARAMS = {"N_jslow_m": 25.0, "v_jslow": 4.5, "K_debounce": 2, "T_stop_sign_s": 2.5, "T_max_stop_s": 25.0,
              "TTL_answer_s": 2.5, "sim_delay_L_s": 0.5, "query_s": 0.5}
ARM_ROWS = {"drive": (), "dslow": (), "jslow": ("R1",), "vred": ("R2", "R3", "R5"), "vbyp": ("R4", "R5"),
            "vall": ("R1", "R2", "R3", "R4", "R5")}
LIGHT_M, SIGN_M = 50.0, 25.0          # truth windows of the oracle answers; the label bins are applied offline


class VlmArbitrationPC:
    """The `pc` hook of OpArbAgent: privileged geometry (bypass path) + the table's stop constraints."""

    def __init__(self, agent, priv):
        self.agent, self.priv = agent, priv

    @property
    def meta(self):
        return dict(self.priv.meta, vlm=self.agent.table_state) if self.priv else {}

    def geometry(self, speed, t, xy, yaw, world, warm):
        self.priv.bypass = bool(self.agent.r4_on)
        return self.priv.geometry(speed, t, xy, yaw, world, warm)

    def constraints(self, s_base, speed, path, t, warm):
        self.priv.constraints(s_base, speed, path, t, warm)      # arm "drive": no control, writes privileged.jsonl
        return self.agent.pending_cons, self.agent.pending_release

    def close(self):
        if self.priv:
            self.priv.close()
            self.priv = None


class VlmArbAgent(OpArbAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        env = os.environ.get
        self.vlm_arm = env("VLM_ARM", "drive")
        rows = set(ARM_ROWS[self.vlm_arm])
        if env("VLM_ROWS"):
            rows &= set(env("VLM_ROWS").split(","))
        self.rows = rows
        self.vlm_oracle = env("VLM_ORACLE", "")
        self.shadow = env("VLM_SHADOW", "0") == "1"
        self.use_vlm = self.shadow or bool(rows & {"R2", "R3", "R4"})
        self.save_frames = env("VLM_SAVE_FRAMES", "0") == "1"
        self.L = float(env("VLM_L", ARB_PARAMS["sim_delay_L_s"]))
        self.light_variant = env("VLM_LIGHT_VARIANT", "base")
        self.client = VLMClient(endpoint=env("VLM_ENDPOINT", "http://127.0.0.1:8080/v1/systemone"))
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.priv = Privileged(self, "drive")           # truth geometry and labels; arm "drive" = it controls nothing
        self.pc = VlmArbitrationPC(self, self.priv)
        self.pending_cons, self.pending_release, self.table_state = {}, False, {}
        self.queue = deque()                            # (t_q, gt, future or answer dict)
        self.answer = None
        self.last_q = self.last_s = -1e9
        K = ARB_PARAMS["K_debounce"]
        self.h_light, self.h_sign, self.h_block = deque(maxlen=K), deque(maxlen=K), deque(maxlen=K)
        self.r2_hold = self.r3_hold = self.r4_on = self.r5 = self.owned_stop = False
        self.r3_done, self.r3_since = set(), None
        self.r4_seen = False
        self.stop_since = None
        self.release_until = -1e9
        self.signs = self.lights = None
        out = Path(self.out)
        self.vlm_log = (out / "vlm_decisions.jsonl").open("w", buffering=1)
        self.frame_dir = out / "vlm_frames"
        if self.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_log.write(json.dumps(dict(k="h", arm=self.vlm_arm, rows=sorted(rows), L=self.L, shadow=self.shadow,
                                           oracle=self.vlm_oracle, light_variant=self.light_variant, params=ARB_PARAMS)) + "\n")

    # ------------------------------------------------------------------ truth (labels and oracle; never a model input)
    def _prepare(self):
        if not self.priv.prepared:
            self.priv.prepare()
        if self.signs is None:
            import carla
            r, self.signs, self.lights = self.route, [], []
            actors = self.priv.world.get_actors()
            for a in actors.filter("traffic.stop"):           # stop signs whose trigger volume lies on the route
                v = a.get_transform().transform(a.trigger_volume.location)
                loc = carla.Location(x=v.x, y=v.y, z=v.z)
                s, lat, tan = project([[loc.x, loc.y]], r.xy)
                f = self.priv.map.get_waypoint(loc).transform.get_forward_vector()
                if abs(float(lat[0])) <= 2.5 and f.x * tan[0][0] + f.y * tan[0][1] > 0.5:
                    self.signs.append((float(s[0]), int(a.id)))
            for a in actors.filter("*traffic_light*"):
                loc = a.get_location()
                self.lights.append((a, np.array([loc.x, loc.y])))

    def _junction(self, ego_s):
        """(distance of the rear axle to the next junction entrance on the route, junction id); inside a junction
        the distance stays within [-1, 0]."""
        r, flags = self.route, self.priv.flags
        ahead = np.flatnonzero(flags & (r.s >= ego_s - 1.0) & (r.s <= ego_s + 100.0))
        if len(ahead) == 0:
            return 999.0, -1
        return float(r.s[ahead[0]]) - ego_s, int(self.priv.jids[ahead[0]])

    def _side(self, start_s):
        """Which adjacent lane the privileged bypass would take for an obstacle starting at start_s."""
        import carla
        r = self.route
        p = r.xy[int(np.searchsorted(r.s, start_s).clip(0, len(r.s) - 1))]
        wp = self.priv.map.get_waypoint(carla.Location(x=float(p[0]), y=float(p[1])))
        fwd = wp.transform.get_forward_vector()
        opts = []
        for order, other in enumerate((wp.get_left_lane(), wp.get_right_lane())):
            if other is None or other.lane_type != carla.LaneType.Driving:
                continue
            loc, u = other.transform.location, other.transform.get_forward_vector()
            off = abs((loc.x - p[0]) * -fwd.y + (loc.y - p[1]) * fwd.x)
            if 2.5 <= off <= 4.5 and abs(loc.z - wp.transform.location.z) <= .75:
                opts.append((bool(u.x * fwd.x + u.y * fwd.y < 0), order))
        return ("left_free", "right_free")[sorted(opts)[0][1]] if opts else "none_free"

    def _truth_labels(self, xy, ego_s, junc_dist):
        import carla
        code = {carla.TrafficLightState.Green: 0, carla.TrafficLightState.Yellow: 1, carla.TrafficLightState.Red: 2}
        c = self._ctx()
        gt = {"tl": c.get("tl"), "tl_dist": c.get("tl_dist"), "tl_id": c.get("tl_id"), "junc_dist": round(junc_dist, 2),
              "lead_gap": c.get("lead_gap"), "lead_v": c.get("lead_v")}
        near = []                                             # every light within 60 m: [id, state, distance]
        for a, p in self.lights:
            d = float(np.linalg.norm(p - xy))
            if d <= 60.0:
                near.append([int(a.id), code.get(a.get_state(), -1), round(d, 1)])
        gt["lights"] = sorted(near, key=lambda x: x[2])[:10]
        ahead = [s - ego_s - REAR_TO_BUMPER for s, _ in self.signs if -2.0 < s - ego_s - REAR_TO_BUMPER < 80.0]
        gt["stop_dist"] = round(min(ahead), 2) if ahead else None
        obs = self.priv.meta.get("obstacles") or []
        if obs:
            gt.update(block="static_block", block_dist=round(obs[0]["start_s"] - ego_s - REAR_TO_BUMPER, 2),
                      side=self._side(obs[0]["start_s"]))
        else:
            lead = c.get("lead_gap") is not None and c["lead_gap"] < 40.0 and c.get("lead_v", 0.0) > 0.5
            gt.update(block="moving_lead" if lead else "clear", block_dist=None, side="none_free")
        return gt

    def _oracle_answer(self, gt):
        d, tl = gt.get("tl_dist"), gt.get("tl")
        light = "no_light"
        if d is not None and d < LIGHT_M and tl in (0, 1, 2):
            light = "green_for_ego" if tl == 0 else "red_or_yellow_for_ego"
        if self.vlm_oracle == "stuckred":
            light = "red_or_yellow_for_ego"
        sd = gt.get("stop_dist")
        return {"ok": True, "latency_ms": 0.0, "oracle": self.vlm_oracle, "Q_light": light,
                "Q_sign": "yes" if sd is not None and sd <= SIGN_M else "no", "Q_block": gt["block"], "Q_side": gt["side"]}

    # ------------------------------------------------------------------ the request (worker thread)
    def _request(self, t_q, frames):
        jpgs = {k: jpeg(v) for k, v in frames.items()}
        if self.save_frames:
            for k, b in jpgs.items():
                (self.frame_dir / ("%08.2f_%s.jpg" % (t_q, k))).write_bytes(b)
        ans = self.client.ask(jpgs, "base")
        if ans["ok"] and self.light_variant != "base":         # Q_light from the diagnosed variant, latencies add up
            alt = self.client.ask(jpgs, self.light_variant, only_light=True)
            ans["latency_ms"] += alt["latency_ms"]
            ans["Q_light_base"] = ans.get("Q_light")
            if alt["ok"]:
                ans.update({k: alt[k] for k in ("Q_light", "Q_light_conf", "Q_light_p") if k in alt})
            else:
                ans.update(ok=False, error=alt.get("error"))
        return ans

    def _ask(self, t, cams, gt):
        if self.vlm_oracle:
            self.queue.append((t, gt, self._oracle_answer(gt)))
            return
        frames = {}
        for tag, name in (("OP_WIDE", "wide"), ("OP_ROAD", "road")):   # CARLA frames are BGRA
            if tag in cams and cams[tag] is not None:
                frames[name] = np.ascontiguousarray(cams[tag][:, :, [2, 1, 0]])
        if frames:
            self.queue.append((t, gt, self.pool.submit(self._request, t, frames)))

    def _arrivals(self, t):
        """Answers become usable at t_q + max(L, measured latency): the simulator waits for the VLM, the car would not."""
        while self.queue and t >= self.queue[0][0] + self.L - 1e-6:
            t_q, gt, a = self.queue[0]
            ans = a if isinstance(a, dict) else a.result()
            t_eff = t_q + max(self.L, ans.get("latency_ms", 0.0) / 1e3)
            if t < t_eff - 1e-6:
                self.queue[0] = (t_q, gt, ans)
                break
            self.queue.popleft()
            ans.update(t_q=t_q, t_eff=t_eff)
            if ans["ok"]:                                      # a failed request is no answer: the last one ages out
                self.answer = ans
                self.h_light.append(ans["Q_light"])
                self.h_sign.append(ans["Q_sign"])
                self.h_block.append((ans["Q_block"], ans["Q_side"]))
            self.vlm_log.write(json.dumps(dict(k="a", t=t, t_q=t_q, t_eff=t_eff, gt=gt, ans=ans), default=_js) + "\n")

    # ------------------------------------------------------------------ the table
    def _table(self, speed, t, junc_dist, jid):
        P, K, A = ARB_PARAMS, ARB_PARAMS["K_debounce"], self.arb
        rows, out, active = self.rows, {}, []
        fresh = bool(self.answer is not None and t - self.answer["t_eff"] <= P["TTL_answer_s"])
        full = lambda h, ok: len(h) == K and all(ok(x) for x in h)     # noqa: E731
        line = junc_dist - REAR_TO_BUMPER - 0.5                        # bumper to the stop position (junction entrance)
        can_stop = line > max(0.5, speed * speed / 8.0 - 1.0) or (speed < 1.0 and line > -1.0)   # as the privileged arm
        stop = lambda: idm(speed, max(0.1, line + A["idm_s0"]), 0.0, 0.0, A["cruise"], A["amax"], A["idm_b"],  # noqa: E731
                           A["idm_s0"], A["idm_T"])
        if speed < 0.2:
            self.stop_since = t if self.stop_since is None else self.stop_since
        else:
            self.stop_since = None
        stood = t - self.stop_since if self.stop_since is not None else 0.0
        release_now = False
        # R5: a held stop older than T_max, or held on an expired answer, is dropped until the car rolls again
        if "R5" in rows and (self.r2_hold or self.r3_hold) and (stood > P["T_max_stop_s"] or not fresh):
            self.r5, self.r2_hold, self.r3_hold, self.r3_since = True, False, False, None
            self.release_until = t + 2.0
            active.append("R5")
        if self.r5 and speed > 1.0:
            self.r5 = False
        # R1: junction speed cap from the map
        cap = A["cruise"]
        if "R1" in rows and -10.0 <= junc_dist <= P["N_jslow_m"]:
            cap = min(cap, P["v_jslow"])
            active.append("R1")
        # R2: red / yellow light for the ego, stop at the junction entrance; only green releases
        if "R2" in rows and not self.r5 and fresh:
            if not self.r2_hold and full(self.h_light, lambda x: x == "red_or_yellow_for_ego") and line < 50.0 and can_stop:
                self.r2_hold = True
            elif self.r2_hold and full(self.h_light, lambda x: x == "green_for_ego"):
                self.r2_hold, self.release_until = False, t + 2.0
        if self.r2_hold and line <= -2.0:                              # past the line: nothing left to hold
            self.r2_hold = False
        if self.r2_hold:
            out["R2"] = stop()
            active.append("R2")
        # R3: stop sign, dwell T_s once per junction; suppressed while R2 holds
        if "R3" in rows and not self.r5 and not self.r2_hold:
            if (not self.r3_hold and fresh and jid not in self.r3_done and full(self.h_sign, lambda x: x == "yes")
                    and line < 25.0 and can_stop):
                self.r3_hold, self.r3_since = True, None
            if self.r3_hold:
                if speed < 0.2 and self.r3_since is None:
                    self.r3_since = t
                if (self.r3_since is not None and t - self.r3_since >= P["T_stop_sign_s"]) or line <= -2.0:
                    self.r3_hold, self.r3_since = False, None
                    self.r3_done.add(jid)
                    self.release_until = t + 2.0
                else:
                    out["R3"] = stop()
                    active.append("R3")
        # R4: bypass of a static block; not while a stop is held, not inside or within N m of a junction; once
        # started it stays on until the privileged geometry has returned to the route
        if "R4" in rows:
            state = self.priv.bypass_state
            if not self.r4_on:
                trig = fresh and full(self.h_block, lambda x: x[0] == "static_block" and x[1] != "none_free")
                if trig and not (self.r2_hold or self.r3_hold) and junc_dist > P["N_jslow_m"]:
                    self.r4_on, self.r4_seen = True, False
            else:
                self.r4_seen = self.r4_seen or state is not None
                done = self.r4_seen and state is None
                idle = not self.r4_seen and fresh and full(self.h_block, lambda x: x[0] != "static_block")
                if done or idle:
                    self.r4_on = False
            if self.r4_on:
                active.append("R4")
        if out and speed < 0.2:
            self.owned_stop = True
        release_now = bool(t < self.release_until and speed < 1.0 and self.owned_stop and not out)
        if speed >= 1.0:
            self.release_until, self.owned_stop = -1e9, False
        self.table_state = dict(rules=active, release=release_now, r5=bool(self.r5), fresh=fresh,
                                line=round(float(line), 2), jid=int(jid), light=list(self.h_light), sign=list(self.h_sign), block=list(self.h_block))
        return out, cap, release_now, active

    # ------------------------------------------------------------------ per plan
    def _plan(self, speed):
        f, t, cams, _ = self.cam_sets[-1]
        xy = np.asarray(self.poses[-1][1], float)
        self._prepare()
        ego_s = float(project([xy], self.route.xy)[0][0])
        junc_dist, jid = self._junction(ego_s)
        if self.use_vlm and t - self.last_q >= ARB_PARAMS["query_s"] - 1e-4:
            self.last_q = t
            self._ask(t, cams, self._truth_labels(xy, ego_s, junc_dist))
        self._arrivals(t)
        cons, cap, release, active = self._table(speed, t, junc_dist, jid)
        self.pending_cons, self.pending_release = cons, release
        if t - self.last_s >= ARB_PARAMS["query_s"] - 1e-4:
            self.last_s = t
            self.vlm_log.write(json.dumps(dict(k="s", t=t, v=round(float(speed), 3), ego_s=round(ego_s, 2),
                                               junc_dist=round(junc_dist, 2), cap=cap, **self.table_state), default=_js) + "\n")
        cruise = self.arb["cruise"]
        self.arb["cruise"] = min(cruise, cap)
        try:
            return super()._plan(speed)
        finally:
            self.arb["cruise"] = cruise

    def destroy(self):
        try:
            self.pool.shutdown(wait=True)
            if getattr(self, "vlm_log", None) and not self.vlm_log.closed:
                self.vlm_log.close()
        finally:
            super().destroy()
