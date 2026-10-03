"""Bench2Drive agent: op-drive (frozen openpilot Cinque) plus a VLM slow channel behind an arbitration table R1-R5.

Protocol and registered parameters: experiments/vlm_arb/plans/2026-10-02-vlm-arb.md. Python 3.8 (route process env).

Environment (set per unit by experiments/vlm_arb/scripts/vlm_arb_chain.py):
  VLM_ARM           drive | dslow | jslow | vred | vred3 | vbyp | vall. drive / dslow open no row (dslow gets its per-route
                    set speed through the config's cruise_by_route).
  VLM_ROWS          comma list that restricts the arm's rows (Phase A gating), e.g. "R1,R4,R5".
  VLM_SHADOW        1: ask the VLM and log its answers even when no VLM row is open (Phase A).
  VLM_SAVE_FRAMES   1: save the wide and road frame of every request as JPEG under <attempt>/vlm_frames.
  VLM_L             seconds of simulation time before an answer may be used (registered: Phase A p95).
  VLM_LIGHT_VARIANT Q_light input variant (lib/vlm_protocol.VARIANTS); a second, Q_light-only request.
  VLM_ORACLE        truth: answers are the ground-truth labels (integration test of the table, debug routes only);
                    stuckred: as truth but Q_light is always red (exercises R5).
  VLM_R2_TARGET     junction (default): R2 stops the car at the next junction entrance on the route (arm vred).
                    stopline: R2 stops it short of the stop line of the traffic light that governs the ego lane at that
                    junction, taken from the map (arm vred2, plans/2026-10-02-pbyp2-vred2.md); same margin as the privileged
                    `pred` (target = bumper-to-line distance - 0.5 m). The light's state is never read for the target.
                    vred3 (plans/2026-10-02-pbyp2-vred2.md section 11): R1 (junction slow-down, approach only: ends at the stop line) + R2 with
                    the stop-line target + R5 + the yellow / commit rule: on the first non-green answer after green the go / stop decision of
                    lib/yellow_rule.py is taken (stop: tentative R2 hold confirmed by the second answer; go: R2 suppressed and the junction cap lifted
                    until the tail has cleared the line); once the front bumper has passed the stop line R2 is never started and R1 is off.
                    Needs VLM_R2_TARGET=stopline and VLM_ROWS=R1,R2,R5.
  vmerge            (plans/2026-10-03-vmerge.md) the merged arm, no CARLA map input to any control decision: junction windows from
                    the official route's command runs (entrance = command start, stop line = entrance - VM["stop_offset_m"]); the
                    light question is asked when openpilot's trigger heads (OP_DET_HEAD, served by op_arb_server) report a light / stop
                    sign, inside a route junction window, or while a stop is held; R2 starts on K = 2 red answers and releases on the
                    cumulative green evidence (results/release_replay.md, cusum >= 5); R3 asks the stop-sign question inside the window
                    and stops once per junction; R1 junction slow-down; the bypass is Privileged arm pbyp3 behind bypass_gate().
                    Needs VLM_BACKEND=qwen. VM_ABL (ablations, one component off): nocusum (release on K = 2 greens), nobyp (no bypass),
                    nor1 (no junction slow-down), nosign (no stop-sign question / R3).
  vmerge3           (plans/2026-10-04-vmerge3.md) switches on top of vmerge (all unset = vmerge unchanged):
                    VM3_BYP=perc   the bypass without privileged input (lib/vm3_perception.PerceivedBypass: openpilot lead, lane lines and
                                   road edges, three radars, the route); the privileged geometry only logs (arm "drive")
                    VM3_REL=1      release check: an R2 cusum release or the end of an R3 dwell waits while the cross-traffic question
                                   (/cross, wide + road frame) says a vehicle is closing on the ego's path; up to VM3["rel_wait_s"]
  VLM_ENDPOINT      System One URL.
  VLM_BACKEND       openjev (default) | qwen: zero-shot Qwen3-VL-4B light reading, one forward pass with option scoring
                    (experiments/vlm_arb/scripts/vlm_qwen_server.py, one server per card on port VLM_BASE_PORT + VLM_GPU,
                    default 8200). It answers Q_light only; Q_sign / Q_block / Q_side are logged as "na" and no row may
                    read them (VLM_ROWS must be R2,R5). `latency_ms` of a qwen answer runs from the moment the request
                    was handed over (frame conversion, JPEG encoding and the wait for a pool thread included) to the answer;
                    `rtt_ms` is the HTTP round trip alone.

Logs in the attempt dir: vlm_decisions.jsonl, one line per request {"k": "a", t_q, t_eff, gt, ans} and one line per
0.5 s {"k": "s", t, v, rules, ...}; plans.jsonl gets the table state under pc.vlm.
"""
import json
import math
import os
import sys
import time
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
from vlm_qwen_client import QwenClient  # noqa: E402
import yellow_rule  # noqa: E402
import vm3_perception as V3  # noqa: E402
import b2d_zeroshot_agent as Z  # noqa: E402


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
ARM_ROWS = {"drive": (), "dslow": (), "jslow": ("R1",), "vred": ("R2", "R3", "R5"), "vred3": ("R1", "R2", "R5"), "vbyp": ("R4", "R5"),
            "vall": ("R1", "R2", "R3", "R4", "R5"), "vmerge": ("R1", "R2", "R3", "R5")}
# vmerge parameters (plans/2026-10-03-vmerge.md, written before its first run)
VM = {"stop_offset_m": 3.3,       # route command start - stop line (route_junction_error.md: dense-plan median +3.26 m, 9 of 12 within 1 m of 3.3)
      "window_before_m": 40.0,    # the junction window opens this far (rear axle) before the command start ...
      "window_after_m": 6.0,      # ... and closes this far past it (inside the junction the distance is clamped to -1)
      "det_latch_s": 3.0,         # a trigger-head firing keeps the light question on for this long
      "cusum_release": 5.0, "cusum_reset_p": 0.9, "cusum_clip": 7.0,   # release_replay.md "cusum >= 5"
      "sign_dwell_s": 2.5,        # R3 dwell (T_stop_sign_s)
      "byp_stand_s": 5.0, "byp_stand_v": 0.3, "byp_lead_x": 40.0, "byp_lead_v": 1.0, "byp_junction_m": 15.0, "byp_red_s": 5.0}   # bypass_timedet.md

# vmerge3 release check (plans/2026-10-04-vmerge3.md, registered before the first run): ask /cross at 2 Hz while a release is near
# (R2 cusum >= ask_cusum, R3 standing at its line) or pending; release only when the last K = 2 answers have P(vehicle_crossing) < thr
VM3 = {"ask_cusum": 2.5, "thr": 0.5, "fresh_s": 1.5, "rel_wait_s": 8.0, "post_s": 4.0}
LIGHT_M, SIGN_M = 50.0, 25.0          # truth windows of the oracle answers; the label bins are applied offline


class VlmArbitrationPC:
    """The `pc` hook of OpArbAgent: privileged geometry (bypass path) + the table's stop constraints."""

    def __init__(self, agent, priv, perc=None):
        self.agent, self.priv, self.perc = agent, priv, perc
        self.amax0 = agent.arb["amax"]

    @property
    def meta(self):
        if not self.priv:
            return {}
        m = dict(self.priv.meta, vlm=self.agent.table_state)
        if self.perc is not None:
            m.update(self.perc.meta)
        return m

    def geometry(self, speed, t, xy, yaw, world, warm):
        byp = bool(self.agent.r4_on) or (self.agent.vm and self.agent.abl != "nobyp")
        if self.perc is None:
            self.priv.bypass = byp
            return self.priv.geometry(speed, t, xy, yaw, world, warm)
        self.priv.bypass = False                              # vmerge3: the privileged geometry only logs the truth
        self.priv.geometry(speed, t, xy, yaw, world, warm)
        if not byp:
            return world
        gate = self.agent.bypass_gate(t) if self.perc.state is None else None
        out = self.perc.geometry(speed, t, xy, yaw, world, warm, gate is None)
        if gate:
            self.perc.meta["suppressed"] = gate
        # go decisively once the pull-out has started (restored by the next call)
        self.agent.arb["amax"] = V3.P["amax_byp"] if self.perc.meta.get("bypass") else self.amax0
        return out

    def constraints(self, s_base, speed, path, t, warm):
        self.priv.constraints(s_base, speed, path, t, warm)      # arm "drive": no control, writes privileged.jsonl
        if self.perc is not None and self.perc.meta.get("hold_stop"):
            return dict(self.agent.pending_cons, byp_hold=np.zeros(len(s_base))), False
        return self.agent.pending_cons, self.agent.pending_release

    def close(self):
        if self.priv:
            self.priv.close()
            self.priv = None
        if self.perc is not None:
            self.perc.close()


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
        self.backend = env("VLM_BACKEND", "openjev")
        self.r2_target = env("VLM_R2_TARGET", "junction")
        assert self.r2_target in ("junction", "stopline"), self.r2_target
        self.v3 = self.vlm_arm == "vred3"
        assert not self.v3 or self.r2_target == "stopline", "vred3 needs the stop-line target"
        self.vm = self.vlm_arm == "vmerge"
        self.abl = env("VM_ABL", "") if self.vm else ""
        assert self.abl in ("", "nocusum", "nobyp", "nor1", "nosign"), self.abl
        if self.abl in ("nor1", "nosign"):
            self.rows = rows = rows - {"nor1": {"R1"}, "nosign": {"R3"}}[self.abl]
        if self.vm:
            assert self.backend == "qwen", "vmerge needs the qwen backend"
            # vmerge-j (plan 2026-10-03-vmerge.md, follow-up): VLM_R2_TARGET=junction stops R2 at the route-command entrance instead of
            # the estimated stop line (R3 keeps the stop line); VM_R5_TMAX replaces R5's T_max. Unset: the vmerge behaviour.
            self.r2_target = env("VLM_R2_TARGET", "stopline")
            self.r5_tmax = float(env("VM_R5_TMAX", ARB_PARAMS["T_max_stop_s"]))
            d = np.load(env("OP_DET_HEAD"))
            self.det_thr = [float(x) for x in d["thr"]]
            self.t_det_l = self.t_det_s = self.t_red = -1e9
            self.cusum, self.n_rel, self.rel_t = 0.0, 0, None
            self.h_sign2, self.sign_answer = deque(maxlen=ARB_PARAMS["K_debounce"]), None
            self.still_since = None
            self.ask_why = ""
            self.rjunc = None
            self.vm_junc, self.det_now, self.last_sq = 999.0, [0.0, 0.0], -1e9
        if self.backend == "qwen" and not self.vm:
            if rows & {"R3", "R4"}:
                raise ValueError("the qwen backend answers Q_light only: VLM_ROWS must not contain R3 / R4")
            self.client = QwenClient(int(env("VLM_BASE_PORT", "8200")) + int(env("VLM_GPU", "0")))
        elif self.backend == "qwen":
            self.client = QwenClient(self._qwen_port())
        else:
            self.client = VLMClient(endpoint=env("VLM_ENDPOINT", "http://127.0.0.1:8080/v1/systemone"))
        self.pool = ThreadPoolExecutor(max_workers=4 if self.backend == "qwen" else 2)
        self.vm3_byp = self.vm and env("VM3_BYP", "") == "perc"
        self.vm3_rel = self.vm and env("VM3_REL", "") == "1"
        self.priv = Privileged(self, "pbyp3" if self.vm and not self.vm3_byp else "drive")    # truth geometry and labels; "drive" = controls nothing
        self.pc = VlmArbitrationPC(self, self.priv, V3.PerceivedBypass(self) if self.vm3_byp else None)
        self.h_cross, self.cross_answer, self.last_cq = deque(maxlen=4), None, -1e9
        self.rel_wait, self.rel_post, self.r6, self.n_cross_wait, self.n_cross_timeout, self.n_rehold = None, None, None, 0, 0, 0
        self.pc.byp_free = self.vm
        self.pending_cons, self.pending_release, self.table_state = {}, False, {}
        self.queue = deque()                            # (t_q, gt, future or answer dict)
        self.answer = None
        self.last_q = self.last_s = -1e9
        K = ARB_PARAMS["K_debounce"]
        self.h_light, self.h_sign, self.h_block = deque(maxlen=K), deque(maxlen=K), deque(maxlen=K)
        self.r2_hold = self.r3_hold = self.r4_on = self.r5 = self.owned_stop = False
        self.r3_done, self.r3_since = set(), None
        self.n_ans = self._seen_ans = 0                 # vred3: answers that arrived / answers the table has looked at
        self.last_green_tq = None                       # frame time of the latest green answer
        self.go_commit, self.go_t, self.go_line_s = False, None, None
        self.r2_tentative, self.tent_n = False, 0
        self.r4_seen = False
        self.stop_since = None
        self.release_until = -1e9
        self.signs = self.lights = self.stop_arcs = None
        out = Path(self.out)
        self.vlm_log = (out / "vlm_decisions.jsonl").open("w", buffering=1)
        self.frame_dir = out / "vlm_frames"
        if self.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_log.write(json.dumps(dict(k="h", arm=self.vlm_arm, rows=sorted(rows), L=self.L, shadow=self.shadow,
                                           oracle=self.vlm_oracle, light_variant=self.light_variant, backend=self.backend, params=ARB_PARAMS,
                                           vm3=dict(byp=self.vm3_byp, rel=self.vm3_rel, P=V3.P, VM3=VM3), qwen=getattr(self.client, "url", None))) + "\n")

    def _qwen_port(self):
        """VLM_PORTS (vmerge3: two servers on one card): take the first free slot, round robin over the servers, by an flock held for the
        life of this route process (released by the OS when it exits); else VLM_BASE_PORT + VLM_GPU."""
        env = os.environ.get
        ports = [int(p) for p in env("VLM_PORTS", "").split(",") if p]
        if not ports:
            return int(env("VLM_BASE_PORT", "8200")) + int(env("VLM_GPU", "0"))
        import fcntl
        d = Path(env("VLM_SLOT_DIR"))
        d.mkdir(parents=True, exist_ok=True)
        for k in range(8):
            for p in ports:
                fh = open(d / ("%d.%d" % (p, k)), "w")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    fh.close()
                    continue
                self._slot_fh = fh
                return p
        return ports[0]

    def sensors(self):
        own = super().sensors()
        if os.environ.get("VLM_ARM") == "vmerge" and (os.environ.get("VM3_BYP", "") == "perc" or os.environ.get("VM3_REL", "") == "1"):
            own += V3.radar_specs()                               # vmerge3: front + two rear-corner radars (Bench2Drive allows 4)
        return own

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
        if self.stop_arcs is None:
            self.stop_arcs = self._stop_arcs()

    def _stop_arcs(self):
        """Route arc positions of the stop lines of the traffic lights that govern a lane of the route (map only: the
        stop waypoints of each light actor lying within 2 m of the route centre line)."""
        r, out = self.route, []
        for a in self.priv.world.get_actors().filter("*traffic_light*"):
            for w in a.get_stop_waypoints():
                q = [[w.transform.location.x, w.transform.location.y]]
                s, lat, _ = project(q, r.xy)
                if abs(float(lat[0])) <= 2.0 and 0.0 < float(s[0]) < float(r.s[-1]):
                    out.append(float(s[0]))
        return sorted(out)

    def _stop_line(self, ego_s, junc_dist):
        """Distance from the front bumper to the stop line of the light at the next junction on the route, or None.
        The line is the last stop line from 15 m before that junction's entrance up to 1 m inside it."""
        if junc_dist >= 999.0:
            return None
        entrance = ego_s + junc_dist
        cand = [s for s in self.stop_arcs if entrance - 15.0 <= s <= entrance + 1.0]
        return None if not cand else max(cand) - ego_s - REAR_TO_BUMPER

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
    def _request(self, t_q, frames, t_sub, q="light"):
        jpgs = {k: jpeg(v) for k, v in frames.items()}
        if self.save_frames and q == "light":
            for k, b in jpgs.items():
                (self.frame_dir / ("%08.2f_%s.jpg" % (t_q, k))).write_bytes(b)
        if self.backend == "qwen" and q in ("sign", "cross"):
            ans = self.client.ask(jpgs, q)
            ans["rtt_ms"] = ans["latency_ms"]
            ans["latency_ms"] = round((time.perf_counter() - t_sub) * 1e3, 1)
            ans["kind"] = q
            return ans
        if self.backend == "qwen":
            ans = self.client.ask(jpgs)
            ans["rtt_ms"] = ans["latency_ms"]
            ans["latency_ms"] = round((time.perf_counter() - t_sub) * 1e3, 1)
            if ans["ok"]:
                ans.update(Q_sign="na", Q_block="na", Q_side="na")
            return ans
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

    def _ask(self, t, cams, gt, q="light"):
        t_sub = time.perf_counter()
        if self.vlm_oracle and q in ("sign", "cross"):
            return
        if self.vlm_oracle:
            self.queue.append((t, gt, self._oracle_answer(gt)))
            return
        frames = {}
        for tag, name in (("OP_WIDE", "wide"), ("OP_ROAD", "road")):   # CARLA frames are BGRA
            if tag in cams and cams[tag] is not None:
                frames[name] = np.ascontiguousarray(cams[tag][:, :, [2, 1, 0]])
        if frames:
            self.queue.append((t, gt, self.pool.submit(self._request, t, frames, t_sub, q)))
            if self.vm:
                self.vlm_log.write(json.dumps(dict(k="q", t_q=t, q=q)) + "\n")

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
            if ans["ok"] and ans.get("kind") == "sign":        # vmerge stop-sign answers have their own history and age
                self.sign_answer = ans
                self.h_sign2.append(ans["Q_sign"])
            elif ans["ok"] and ans.get("kind") == "cross":     # vmerge3 release check
                self.cross_answer = ans
                self.h_cross.append((t_eff, float(ans["Q_cross_p"]["vehicle_crossing"])))
            elif ans["ok"]:                                    # a failed request is no answer: the last one ages out
                self.answer = ans
                self.h_light.append(ans["Q_light"])
                self.n_ans += 1
                if ans["Q_light"] == "green_for_ego":
                    self.last_green_tq = t_q
                if ans["Q_light"] == "red_or_yellow_for_ego":
                    self.t_red = t
                self.h_sign.append(ans["Q_sign"])
                self.h_block.append((ans["Q_block"], ans["Q_side"]))
            self.vlm_log.write(json.dumps(dict(k="a", t=t, t_q=t_q, t_eff=t_eff, gt=gt, ans=ans), default=_js) + "\n")

    # ------------------------------------------------------------------ the table
    def _ylog(self, **kw):
        self.vlm_log.write(json.dumps(dict(k="y", **kw), default=_js) + "\n")

    def _table(self, speed, t, junc_dist, jid, stop_dist=None, ego_s=0.0):
        P, K, A = ARB_PARAMS, ARB_PARAMS["K_debounce"], self.arb
        rows, out, active = self.rows, {}, []
        fresh = bool(self.answer is not None and t - self.answer["t_eff"] <= P["TTL_answer_s"])
        full = lambda h, ok: len(h) == K and all(ok(x) for x in h)     # noqa: E731
        line = junc_dist - REAR_TO_BUMPER - 0.5                        # bumper to the stop position (junction entrance)
        can_stop = line > max(0.5, speed * speed / 8.0 - 1.0) or (speed < 1.0 and line > -1.0)   # as the privileged arm
        stop = lambda ln=line: idm(speed, max(0.1, ln + A["idm_s0"]), 0.0, 0.0, A["cruise"], A["amax"], A["idm_b"],  # noqa: E731
                                   A["idm_s0"], A["idm_T"])
        # R2's own target: the light's stop line (vred2) when the map gives one, else the junction entrance (as vred)
        line2, src2 = line, "junction"
        if self.r2_target == "stopline" and stop_dist is not None:
            line2, src2 = stop_dist - 0.5, "stopline"
        can_stop2 = line2 > max(0.5, speed * speed / 8.0 - 1.0) or (speed < 1.0 and line2 > -1.0)
        GREEN, RED_ = "green_for_ego", "red_or_yellow_for_ego"
        d_line = junc_dist - REAR_TO_BUMPER                           # front bumper to the scorer's line (junction entrance)
        ydec = None
        if self.v3:
            # commit: nothing below starts an R2 stop once the front bumper is past the stop line
            before_line = stop_dist is None or stop_dist > 0.0
            beyond = None if self.go_line_s is None else ego_s + REAR_TO_BUMPER - self.go_line_s
            if self.go_commit and (beyond is None or beyond >= yellow_rule.TAIL_M or t - self.go_t > 10.0):
                self._ylog(t=t, ev="go_end", beyond=None if beyond is None else round(beyond, 2), v=round(float(speed), 2))
                self.go_commit = False
        else:
            before_line = True
        new_ans = self.n_ans != self._seen_ans
        self._seen_ans = self.n_ans
        hl = list(self.h_light)
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
        if self.r5 and "R5" not in active:                             # logged for as long as the fallback is in force
            active.append("R5")
        # R1: junction speed cap from the map; vred3: approach only (off once the front bumper is past the stop line, and during a go)
        cap = A["cruise"]
        past_line = self.v3 and (self.go_commit or (stop_dist is not None and stop_dist <= 0.0))
        if "R1" in rows and -10.0 <= junc_dist <= P["N_jslow_m"] and not past_line:
            cap = min(cap, P["v_jslow"])
            active.append("R1")
        # R2: red / yellow light for the ego, stop at the junction entrance; only green releases
        if self.v3 and self.r2_tentative and self.n_ans > self.tent_n:    # the second answer after a tentative start
            self.r2_tentative = False
            if not hl or hl[-1] != RED_:
                self.r2_hold, self.release_until = False, t + 2.0
                self._ylog(t=t, ev="tentative_cancel", answer=hl[-1] if hl else None, v=round(float(speed), 2))
            else:
                self._ylog(t=t, ev="tentative_confirm", v=round(float(speed), 2))
        if "R2" in rows and not self.r5 and fresh:
            if (self.v3 and new_ans and not self.r2_hold and not self.go_commit and before_line and len(hl) == 2 and hl[-1] == RED_ and hl[-2] == GREEN
                    and line2 < 50.0 and self.last_green_tq is not None):
                # first non-green answer after green: the yellow decision, without waiting for the K = 2 debounce
                age = t - self.last_green_tq
                rem = yellow_rule.remaining_yellow(age)
                ydec = yellow_rule.decide(line2, d_line, float(speed), rem, A["cruise"])
                c = self._ctx()
                if ydec == "go" or not can_stop2:
                    self.go_commit, self.go_t, self.go_line_s = True, t, ego_s + junc_dist
                    ydec = ydec if ydec == "go" else "stop_infeasible_go"
                else:
                    self.r2_hold, self.r2_tentative, self.tent_n = True, True, self.n_ans
                self._ylog(t=t, ev="yellow", decision=ydec, v=round(float(speed), 2), age=round(age, 2), remaining=round(rem, 2), d_stop=round(line2 + 0.5, 2),
                           d_line=round(d_line, 2), can_stop=bool(can_stop2), stop_need=round(yellow_rule.stop_distance(float(speed)), 2),
                           truth=dict(tl=c.get("tl"), tl_dist=c.get("tl_dist"), tl_id=c.get("tl_id")))
            elif (not self.r2_hold and not self.go_commit and before_line and full(self.h_light, lambda x: x == "red_or_yellow_for_ego") and line2 < 50.0
                  and can_stop2):
                self.r2_hold = True
            elif self.r2_hold and full(self.h_light, lambda x: x == "green_for_ego"):
                self.r2_hold, self.release_until = False, t + 2.0
        if self.r2_hold and line2 <= -2.0:                             # past the line: nothing left to hold
            self.r2_hold = False
        if self.r2_hold:
            out["R2"] = stop(line2)
            active.append("R2")
        if self.v3 and self.go_commit and "R1" in active:               # a go decided in this very call: no cap from this step on
            active.remove("R1")
            cap = A["cruise"]
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
                                line=round(float(line), 2), line_r2=round(float(line2), 2), r2_src=src2, go=bool(self.go_commit), tent=bool(self.r2_tentative),
                                d_stop=None if stop_dist is None else round(float(stop_dist), 2), jid=int(jid), light=list(self.h_light), sign=list(self.h_sign), block=list(self.h_block))
        return out, cap, release_now, active

    # ------------------------------------------------------------------ vmerge: route-command junctions, trigger, bypass gate
    def _route_junctions(self):
        """[(entrance s, end s)] of every LEFT / RIGHT / STRAIGHT command run of the official dense route (no map)."""
        if self.rjunc is None:
            r, out, i = self.route, [], 0
            cmd = r.cmd
            while i < len(cmd):
                if cmd[i] in (Z.LEFT, Z.RIGHT, Z.STRAIGHT):
                    j = i
                    while j + 1 < len(cmd) and cmd[j + 1] == cmd[i]:
                        j += 1
                    out.append((float(r.s[i]), float(r.s[j])))
                    i = j + 1
                else:
                    i += 1
            self.rjunc = out
        return self.rjunc

    def _junction_vm(self, ego_s):
        """As _junction (rear axle to the next entrance, clamped to [-1, 0] inside), plus the bumper distance to the estimated
        stop line (entrance - stop_offset_m), from the route commands."""
        for k, (a, b) in enumerate(self._route_junctions()):
            if b >= ego_s - 1.0:
                if a - ego_s > 100.0:
                    break
                d = a - ego_s
                return (d if d >= 0.0 else max(d, -1.0)), k, a - VM["stop_offset_m"] - ego_s - REAR_TO_BUMPER
        return 999.0, -1, None

    def route_junction_at(self, s):
        """An arc position inside (or within 5 m of) a route-command junction (pbyp3: no blocker there)."""
        return any(a - 5.0 <= s <= b + 5.0 for a, b in self._route_junctions())

    def vlm_red_recent(self, t):
        return t - self.t_red < VM["byp_red_s"]

    def bypass_gate(self, t):
        """None when a bypass may start, else the reason it may not (bypass_timedet.md: ego standing >= 5 s with the
        openpilot lead head reporting a slow lead within 40 m; no VLM red answer within 5 s; not near a route junction)."""
        if self.still_since is None or t - self.still_since < VM["byp_stand_s"]:
            return "not_standing"
        o = getattr(self, "op_out", None)
        if o is None:
            return "no_lead_head"
        lead = np.asarray(o["lead"])[0]
        if not (float(np.asarray(o["lead_prob"])[0]) > self.arb["lead_p"] and 0.0 < float(lead[0, 0]) < VM["byp_lead_x"]
                and float(lead[0, 2]) < VM["byp_lead_v"]):
            return "no_static_lead"
        if self.vlm_red_recent(t):
            return "vlm_red"
        if self.vm_junc <= VM["byp_junction_m"]:
            return "junction"
        return None

    def _no_ego_light(self):
        """The last two light answers see no light for the ego (the stop-sign question is asked, and R3 may start, only then:
        the dbg-vmerge-334 smoke run answered stop_sign_for_ego in front of a green light)."""
        return len(self.h_light) == 2 and all(x in ("no_light", "light_for_other_lane") for x in self.h_light)

    def _crossing(self, t):
        """D7: the front radar reported a side vehicle closing in at least 2 of the last 3 checks (0.2 s apart)."""
        h = [x for x in self.h_cross if t - x[0] <= 0.65]
        return sum(1 for x in h if x[1]) >= 2

    def _cross_ok(self, t, who):
        """vmerge3 release check: True when the release may go ahead (always without VM3_REL)."""
        if not self.vm3_rel:
            return True
        if self.rel_wait is None or self.rel_wait[0] != who:
            self.rel_wait = (who, t)
        h = [x for x in self.h_cross if t - x[0] <= 0.65]
        clear = len(h) >= 3 and not any(x[1] for x in h)
        if clear or t - self.rel_wait[1] > VM3["rel_wait_s"]:
            if not clear:
                self.n_cross_timeout += 1
            if t - self.rel_wait[1] > 0.05:
                self.n_cross_wait += 1
            self.rel_wait = None
            return True
        return False

    def _table_vm(self, speed, t, junc_dist, jid, stop_dist):
        P, K, A = ARB_PARAMS, ARB_PARAMS["K_debounce"], self.arb
        rows, out, active = self.rows, {}, []
        fresh = bool(self.answer is not None and t - self.answer["t_eff"] <= P["TTL_answer_s"])
        sfresh = bool(self.sign_answer is not None and t - self.sign_answer["t_eff"] <= P["TTL_answer_s"])
        full = lambda h, ok: len(h) == K and all(ok(x) for x in h)     # noqa: E731
        stop = lambda ln: idm(speed, max(0.1, ln + A["idm_s0"]), 0.0, 0.0, A["cruise"], A["amax"], A["idm_b"],  # noqa: E731
                              A["idm_s0"], A["idm_T"])
        line3 = (stop_dist - 0.5) if stop_dist is not None else junc_dist - REAR_TO_BUMPER - 0.5    # estimated stop line (R3)
        line2 = junc_dist - REAR_TO_BUMPER - 0.5 if self.r2_target == "junction" else line3            # R2 target
        can_stop2 = line2 > max(0.5, speed * speed / 8.0 - 1.0) or (speed < 1.0 and line2 > -1.0)
        can_stop3 = line3 > max(0.5, speed * speed / 8.0 - 1.0) or (speed < 1.0 and line3 > -1.0)
        new_ans = self.n_ans != self._seen_ans
        self._seen_ans = self.n_ans
        if speed < 0.2:
            self.stop_since = t if self.stop_since is None else self.stop_since
        else:
            self.stop_since = None
        stood = t - self.stop_since if self.stop_since is not None else 0.0
        # R5 as registered: a held stop older than T_max, or held on an expired answer, is dropped until the car rolls again
        if "R5" in rows and ((self.r2_hold and (stood > self.r5_tmax or not fresh))
                             or (self.r3_hold and not sfresh and self.r3_since is None)):
            self.r5, self.r2_hold, self.r3_hold, self.r3_since = True, False, False, None
            self.release_until = t + 2.0
            active.append("R5")
        if self.r5 and speed > 1.0:
            self.r5 = False
        if self.r5 and "R5" not in active:
            active.append("R5")
        cap = A["cruise"]
        if "R1" in rows and -10.0 <= junc_dist <= P["N_jslow_m"]:
            cap = min(cap, P["v_jslow"])
            active.append("R1")
        # R2: start on K = 2 red / yellow answers before the estimated stop line, release on cumulative green evidence
        rel = None
        if "R2" in rows and not self.r5 and fresh:
            if (not self.r2_hold and full(self.h_light, lambda x: x == "red_or_yellow_for_ego") and line2 < 50.0 and can_stop2):
                self.r2_hold, self.cusum = True, 0.0
            elif self.r2_hold and self.abl == "nocusum":
                if full(self.h_light, lambda x: x == "green_for_ego"):
                    self.r2_hold, self.release_until, rel = False, t + 2.0, 2.0
            elif self.r2_hold and new_ans:
                pp = self.answer.get("Q_light_p") or {}
                pg, pr = float(pp.get("green_for_ego", 0.0)), float(pp.get("red_or_yellow_for_ego", 0.0))
                if pr >= VM["cusum_reset_p"]:
                    self.cusum = 0.0
                else:
                    c = VM["cusum_clip"]
                    self.cusum = max(0.0, self.cusum + min(c, max(-c, math.log((pg + 1e-3) / (pr + 1e-3)))))
                if self.cusum < VM["cusum_release"] and self.rel_wait is not None and self.rel_wait[0] == "R2":
                    self.rel_wait = None                          # the evidence fell back below the line: no release pending
                if self.cusum >= VM["cusum_release"] and self._cross_ok(t, "R2"):
                    self.r2_hold, self.release_until, rel = False, t + 2.0, round(self.cusum, 2)
            elif self.r2_hold and self.rel_wait is not None and self.rel_wait[0] == "R2" and self._cross_ok(t, "R2"):
                self.r2_hold, self.release_until, rel = False, t + 2.0, round(self.cusum, 2)
        if rel is not None:
            self.n_rel, self.rel_t = self.n_rel + 1, t
            self.rel_post = ("R2", t)
        if self.r2_hold and line2 <= -2.0:
            self.r2_hold = False
        if self.r2_hold:
            out["R2"] = stop(line2)
            active.append("R2")
        # R3: stop sign answered twice inside the junction window: stop at the estimated line, dwell, once per junction
        if "R3" in rows and not self.r5 and not self.r2_hold:
            if (not self.r3_hold and sfresh and jid not in self.r3_done and full(self.h_sign2, lambda x: x == "stop_sign_for_ego")
                    and line3 < 25.0 and can_stop3 and self._no_ego_light()):
                self.r3_hold, self.r3_since = True, None
            if self.r3_hold:
                # the dwell counts only at the target and after the warm-up (V2: on 17280 the hold was served standing at the spawn,
                # 2.2 m short of the line, where the scorer's 4 m proximity test does not see the car; the hold now creeps it up first)
                at_line = line3 <= 1.0 and not getattr(self, "warm_now", True)
                if speed < 0.1 and at_line and self.r3_since is None:
                    self.r3_since = t
                dwell = self.r3_since is not None and t - self.r3_since >= VM["sign_dwell_s"]
                if line3 <= -2.0 or (dwell and self._cross_ok(t, "R3")):
                    self.r3_hold, self.r3_since = False, None
                    self.r3_done.add(jid)
                    self.release_until = t + 2.0
                    self.rel_post = ("R3", t)
                else:
                    out["R3"] = stop(line3)
                    active.append("R3")
        # R6 (vmerge3): within post_s after an R2 / R3 release, two fresh "vehicle_crossing" answers stop the car again at the same line if it
        # still can; it is released by two clear answers or after rel_wait_s
        if self.vm3_rel and not out and not self.r5:
            r6 = getattr(self, "r6", None)
            ln = None
            if self.rel_post is not None and t - self.rel_post[1] <= VM3["post_s"]:
                ln = line2 if self.rel_post[0] == "R2" else line3
            if r6 is None and ln is not None and self._crossing(t) and ln > max(0.5, speed * speed / 8.0 - 1.0) and speed < 5.0:
                self.r6, self.n_rehold = (ln, t, self.rel_post[0]), self.n_rehold + 1
                self.rel_wait = ("R6", t)
            elif r6 is not None:
                ln = line2 if r6[2] == "R2" else line3
                if ln <= -2.0 or self._cross_ok(t, "R6"):
                    self.r6, self.release_until, self.rel_post = None, t + 2.0, None
            if getattr(self, "r6", None) is not None:
                out["R6"] = stop(ln)
                active.append("R6")
        if out and speed < 0.2:
            self.owned_stop = True
        release_now = bool(t < self.release_until and speed < 1.0 and self.owned_stop and not out)
        if speed >= 1.0:
            self.release_until, self.owned_stop = -1e9, False
        self.table_state = dict(rules=active, release=release_now, r5=bool(self.r5), fresh=fresh, sfresh=sfresh, cusum=round(self.cusum, 2), n_rel=self.n_rel, rel_t=self.rel_t,
                                cross=[int(p) for _, p in self.h_cross], rel_wait=self.rel_wait, n_cwait=self.n_cross_wait, n_ctime=self.n_cross_timeout,
                                n_rehold=self.n_rehold,
                                line_r2=round(float(line2), 2), d_stop=None if stop_dist is None else round(float(stop_dist), 2), jid=int(jid),
                                light=list(self.h_light), sign=list(self.h_sign2), ask=self.ask_why, det=self.det_now,
                                byp=self.priv.meta.get("suppressed"))
        return out, cap, release_now, active

    def _plan_vm(self, speed):
        f, t, cams, _ = self.cam_sets[-1]
        xy = np.asarray(self.poses[-1][1], float)
        self._prepare()
        ego_s = float(project([xy], self.route.xy)[0][0])
        junc_dist, jid, stop_dist = self._junction_vm(ego_s)
        self.vm_junc = junc_dist
        if speed < VM["byp_stand_v"] and not (self.vm3_byp and getattr(self, "warm_now", True)):
            # vmerge3: the warm-up hold is the harness's, not a stop (dbg-vm3-334 started a bypass at the spawn behind a queue)
            self.still_since = t if self.still_since is None else self.still_since
        else:
            self.still_since = None
        o = getattr(self, "op_out", None)
        det = [float(x) for x in np.asarray(o["det"])] if o is not None and "det" in o else [0.0, 0.0]
        self.det_now = [round(x, 3) for x in det]
        if det[0] >= self.det_thr[0]:
            self.t_det_l = t
        if det[1] >= self.det_thr[1]:
            self.t_det_s = t
        in_win = -VM["window_after_m"] <= junc_dist <= VM["window_before_m"]
        trig = t - max(self.t_det_l, self.t_det_s) <= VM["det_latch_s"]
        why = [w for w, on in (("det", trig), ("window", in_win), ("hold", self.r2_hold or self.r3_hold)) if on]
        self.ask_why = "+".join(why)
        if why and t - self.last_q >= ARB_PARAMS["query_s"] - 1e-4:
            self.last_q = t
            gt = self._truth_labels(xy, ego_s, junc_dist)
            gt.update(det=self.det_now, why=self.ask_why, jid_vm=jid)
            self._ask(t, cams, gt)
        # the stop-sign question on its own 1 Hz clock, 0.25 s after a light slot (seed-0 round 1: asked together with the light
        # question it doubled the queue, p95 468 ms on the shard with the unsignalised junctions), only before the entrance
        if ("R3" in self.rows and in_win and jid not in self.r3_done and 0.0 <= junc_dist <= 30.0 and self._no_ego_light()
                and t - self.last_q >= 0.25 - 1e-4 and t - self.last_sq >= 1.0 - 1e-4):
            self.last_sq = t
            gt = self._truth_labels(xy, ego_s, junc_dist)
            gt.update(det=self.det_now, why="sign", jid_vm=jid)
            self._ask(t, cams, gt, "sign")
        if self.vm3_rel and t - self.last_cq >= 0.2 - 1e-4 and not getattr(self, "warm_now", True):
            # D7: the release check reads the front radar every 0.2 s (the Qwen question failed its offline line)
            self.last_cq = t
            thr, why_c = V3.cross_threat(V3.radar_returns(self, xy, float(self.poses[-1][2]), float(speed)))
            self.h_cross.append((t, thr))
            if thr and (self.r2_hold or self.r3_hold or self.rel_wait is not None or self.r6 is not None):
                self.vlm_log.write(json.dumps(dict(k="c", t=t, why=why_c, rel_wait=self.rel_wait)) + "\n")
        self._arrivals(t)
        cons, cap, release, active = self._table_vm(speed, t, junc_dist, jid, stop_dist)
        self.pending_cons, self.pending_release = cons, release
        if t - self.last_s >= ARB_PARAMS["query_s"] - 1e-4:
            self.last_s = t
            self.vlm_log.write(json.dumps(dict(k="s", t=t, v=round(float(speed), 3), ego_s=round(ego_s, 2),
                                               junc_dist=round(junc_dist, 2), cap=cap, **self.table_state), default=_js) + "\n")
        cruise = self.arb["cruise"]
        self.arb["cruise"] = min(cruise, cap)
        try:
            return OpArbAgent._plan(self, speed)
        finally:
            self.arb["cruise"] = cruise

    # ------------------------------------------------------------------ per plan
    def _plan(self, speed):
        if self.vm:
            return self._plan_vm(speed)
        f, t, cams, _ = self.cam_sets[-1]
        xy = np.asarray(self.poses[-1][1], float)
        self._prepare()
        ego_s = float(project([xy], self.route.xy)[0][0])
        junc_dist, jid = self._junction(ego_s)
        if self.use_vlm and t - self.last_q >= ARB_PARAMS["query_s"] - 1e-4:
            self.last_q = t
            self._ask(t, cams, self._truth_labels(xy, ego_s, junc_dist))
        self._arrivals(t)
        cons, cap, release, active = self._table(speed, t, junc_dist, jid, self._stop_line(ego_s, junc_dist) if self.r2_target == "stopline" else None, ego_s)
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
