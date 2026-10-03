"""vmerge3 perception: the non-privileged bypass (obstacle, adjacent lane, gap) for lib/vlm_arb_agent.py (Python 3.8).

Plan: experiments/vlm_arb/plans/2026-10-04-vmerge3.md. Inputs, all on-board:
  - openpilot heads (lib/op_arb_agent.py `op_out`, served by experiments/op_closed_loop/archive/op_arb_server.py with OP_LANES=1):
    lead (x, y, v), lead_prob, lane_lines (4 x 33, y right-positive at X_IDXS), lane_lines prob, road_edges (2 x 33);
  - three radars (Bench2Drive allows 4): front (oncoming traffic, the obstacle's extent) and two rear corners (blind-spot monitoring,
    as the BSM signals openpilot's lane-change assist reads on production cars); parsed by leaderboard's sensor_interface to
    rows [depth, azimuth, altitude, velocity];
  - the official route (arc position of the ego from its own pose; the bypass path is the route shifted sideways) and the ego pose.
Nothing here reads a CARLA actor, the map or a scenario. lib/b2d_privileged_geometry.py keeps logging the truth beside it (arm "drive").

Frames: rig = rear axle, x forward, y LEFT (b2d_controller_adapter.world_to_local); CARLA sensor / vehicle = x forward, y RIGHT;
route lateral (b2d_privileged_geometry.project) = RIGHT positive, which is also the sign of `offset` (the bypass shift).
"""
import json
import math
from pathlib import Path

import numpy as np

from b2d_privileged_geometry import project, project_ext, EGO_BACK, EGO_FRONT

REAR_AXLE_X = -1.388633220199954           # rear axle in the vehicle (bounding-box centre) frame
CAM_X = 1.779                              # openpilot camera ahead of the rear axle (zeroshot_rigs.OP_MOUNT_RIG)
X_IDXS = 192.0 * (np.arange(33) / 32.0) ** 2
# radar mounts (vehicle frame, CARLA: y right, yaw clockwise): front bumper, rear corners looking back-outwards
RADARS = {"RADAR_F": dict(x=2.45, y=0.0, z=0.7, yaw=0.0, hfov=70.0),
          "RADAR_RL": dict(x=-2.4, y=-0.8, z=0.7, yaw=-150.0, hfov=120.0),
          "RADAR_RR": dict(x=-2.4, y=0.8, z=0.7, yaw=150.0, hfov=120.0)}
# registered before the first run (plan 2026-10-04-vmerge3.md); not tuned on the 19 evaluation routes
P = dict(lane_x=(8.0, 30.0),       # openpilot lane lines / edges read as the median over this range ahead of the camera
         room_min=2.6,             # an adjacent lane needs this much between the ego lane line and the road edge on that side
         lane_w=(2.6, 4.5),        # plausible ego lane width; outside it the reading is rejected
         obs_len=15.0,             # obstacle extent prior past the lead (Bench2Drive obstacle scenarios: one car ~5 m, construction / accident ~15 m)
         obs_ext_max=40.0,         # front-radar static returns can extend the obstacle up to this far past its start
         obs_ext_gap=8.0,          # ... while consecutive returns are at most this far apart
         lead_y=1.5,               # a lead counts as in the ego lane within this lateral distance
         band=1.6,                 # half width of the target-lane band for radar targets
         moving_v=1.0,             # a radar return moves when its ego-motion-compensated radial speed exceeds this
         t_start=2.5, d_start=3.0,  # start of the pull-out: no closing vehicle behind within d_start + closing x t_start (m)
         t_abort=0.8, d_abort=1.5,  # after it began (shift 5 % - 30 %): only a vehicle within d_abort + closing x t_abort aborts it
         side_mem_s=1.0,           # a target seen within 8 m behind the rear bumper blocks for this long (it may be alongside)
         radar_frames=3)           # radar frames pooled per decision (0.15 s)


def radar_specs():
    return [dict(type="sensor.other.radar", id=k, x=m["x"], y=m["y"], z=m["z"], roll=0.0, pitch=0.0, yaw=m["yaw"],
                 horizontal_fov=m["hfov"], vertical_fov=6.0) for k, m in RADARS.items()]


def radar_rig(tag, pts, v_ego):
    """Radar rows [depth, azimuth, altitude, velocity] -> rig x, y(left), radial velocity (negative = closing), the
    ego-motion-compensated radial velocity of the target itself, and the height of the return above the ground."""
    pts = np.asarray(pts, float).reshape(-1, 4)
    if not len(pts):
        return np.zeros((0, 5))
    m = RADARS[tag]
    d, az, alt, vr = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    yaw = math.radians(m["yaw"])
    ux, uy = np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az)          # sensor frame, y right
    vx, vy = ux * math.cos(yaw) - uy * math.sin(yaw), ux * math.sin(yaw) + uy * math.cos(yaw)   # vehicle frame, y right
    x = m["x"] + d * vx - REAR_AXLE_X
    y = -(m["y"] + d * vy)
    comp = vr + v_ego * vx                       # a static target reads vr = -v_ego * (los . forward)
    z = m["z"] + d * np.sin(alt)
    return np.stack([x, y, vr, comp, z], -1)


def rig_to_world(xy, yaw, pts):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.asarray(pts, float).reshape(-1, 2)
    return np.asarray(xy, float)[None] + pts[:, :1] * np.array([[c, s]]) + pts[:, 1:2] * np.array([[s, -c]])


def lanes_reading(o):
    """openpilot lane lines / road edges -> dict(w, room_l, room_r, ...) or None (median over P["lane_x"] ahead of the camera)."""
    if o is None or "lane_lines" not in o or "road_edges" not in o:
        return None
    sel = (X_IDXS >= P["lane_x"][0]) & (X_IDXS <= P["lane_x"][1])
    ll, re = np.asarray(o["lane_lines"], float)[:, sel], np.asarray(o["road_edges"], float)[:, sel]
    y = np.median(ll, 1)
    e = np.median(re, 1)
    pr = np.asarray(o.get("lane_prob", np.ones(4)), float)
    w = float(y[2] - y[1])
    return dict(y=[round(float(v), 2) for v in y], edge=[round(float(v), 2) for v in e], prob=[round(float(v), 2) for v in pr],
                w=round(w, 2), room_l=round(float(y[1] - e[0]), 2), room_r=round(float(e[1] - y[2]), 2))


class PerceivedBypass:
    """The bypass path of pbyp3 (route shifted by `offset`, smooth ramps), with every privileged input replaced:
    obstacle start = openpilot lead, extent = prior + front-radar static returns, side / offset = openpilot lane lines and road
    edges, gap = radar targets in the target-lane band."""

    def __init__(self, agent):
        self.agent, self.state, self.meta = agent, None, {}
        self.last_close = -1e9
        self.log = (Path(agent.out) / "perc.jsonl").open("w", buffering=1)
        self.last_log = -1e9

    # ------------------------------------------------------------------ inputs
    def radar(self, xy, yaw, v_ego):
        """Pooled radar returns of the last frames, ground returns (< 0.3 m high) dropped: rows (s, lateral, x_rig, y_rig, vr, comp,
        z, tag index)."""
        fr = getattr(self.agent.router, "frames", {})
        if not fr:
            return np.zeros((0, 8))
        last = max(fr)
        out = []
        for k, tag in enumerate(RADARS):
            for f in range(last - P["radar_frames"] + 1, last + 1):
                d = fr.get(f, {}).get(tag)
                if d is None:
                    continue
                r = radar_rig(tag, d[1], v_ego)
                if len(r):
                    out.append(np.c_[r, np.full(len(r), k)])
        r = np.concatenate(out) if out else np.zeros((0, 6))
        r = r[r[:, 4] > 0.3]
        if not len(r):
            return np.zeros((0, 8))
        w = rig_to_world(xy, yaw, r[:, :2])
        s, lat, _ = project_ext(w, self.agent.route.xy)
        return np.c_[s, lat, r]

    def lead(self):
        o = getattr(self.agent, "op_out", None)
        if o is None:
            return None
        ld = np.asarray(o["lead"])[0]
        if float(np.asarray(o["lead_prob"])[0]) > self.agent.arb["lead_p"] and 0.0 < float(ld[0, 0]) < 40.0 \
                and float(ld[0, 2]) < 1.0 and abs(float(ld[0, 1])) <= P["lead_y"]:
            return float(ld[0, 0])
        return None

    # ------------------------------------------------------------------ per plan
    def geometry(self, speed, t, xy, yaw, world, warm, gate_free):
        """gate_free: the agent's bypass_gate passed this step (only needed to start a new bypass)."""
        r = self.agent.route
        ego_s = float(project([xy], r.xy)[0][0])
        lat_ego = float(project_ext([xy], r.xy)[1][0])
        o = getattr(self.agent, "op_out", None)
        lanes = lanes_reading(o)
        tg = self.radar(xy, yaw, speed) if not warm else np.zeros((0, 8))
        self.meta = dict(bypass=False, ego_s=round(ego_s, 3), perc=True)
        lx = self.lead()
        if self.state is not None and ego_s > self.state["end_s"] + 23:
            self.state = None
        if self.state is None and gate_free and not warm:
            self.state = self._new(ego_s, lx, lanes)
        st = self.state
        if st is not None:
            if lx is not None and not st.get("committed"):          # refine the start while the ego still looks at it
                s_lead = ego_s + CAM_X + lx
                if s_lead < st["start_s"] - 1.0:
                    st["start_s"] = s_lead
                st["end_s"] = max(st["end_s"], s_lead + P["obs_len"])
            self._extend(st, tg)
        gap, why = self._gap(st, tg, ego_s, speed, t, lat_ego) if st is not None else (True, None)
        self._log(t, ego_s, speed, lanes, tg, lx, st, gap, why)
        if st is None:
            return world
        frac = lat_ego / st["offset"] if st["offset"] else 0.0
        if frac >= 0.3:
            st["committed"] = True
        self.meta.update(gap_open=gap, gap_why=why, shift_frac=round(frac, 3), committed=bool(st.get("committed")),
                         bypass_state=dict(start_s=st["start_s"], end_s=st["end_s"], offset=st["offset"], borrow=False))
        if not st.get("committed") and not gap:
            st["held"] = st.get("held", 0) + 1
            self.meta["gap_hold"] = True
            return world
        st["started"] = True
        s = r.s
        enter = np.clip((s - (st["start_s"] - 20)) / 15, 0, 1)
        leave = np.clip((s - (st["end_s"] + 8)) / 15, 0, 1)
        smooth = lambda v: v * v * (3 - 2 * v)  # noqa: E731
        weight = smooth(enter) * (1 - smooth(leave))
        tangent = np.gradient(r.xy, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], -1)
        shifted = r.xy + normal * (weight * st["offset"])[:, None]
        sel = shifted[(s >= ego_s - 1) & (s <= ego_s + 85)]
        d = np.array([math.cos(yaw), math.sin(yaw)])
        sel = sel[((sel - xy) * d).sum(-1) > .5]
        if len(sel) < 2:
            return world
        self.meta["bypass"] = True
        return np.r_[np.asarray(xy)[None], sel]

    def _new(self, ego_s, lx, lanes):
        if lx is None:
            self.meta["no_obstacle"] = True
            return None
        if lanes is None or not P["lane_w"][0] <= lanes["w"] <= P["lane_w"][1]:
            self.meta["no_lanes"] = True
            return None
        # side: the side with room for a lane; both: the right (right-hand traffic: the right lane runs the ego's way)
        side = "right" if lanes["room_r"] >= P["room_min"] else "left" if lanes["room_l"] >= P["room_min"] else None
        if side is None:
            self.meta["no_adjacent_lane"] = True
            return None
        start = ego_s + CAM_X + lx
        return dict(start_s=start, end_s=start + P["obs_len"], offset=lanes["w"] * (1.0 if side == "right" else -1.0),
                    side=side, lanes=lanes, t0=None)

    def _extend(self, st, tg):
        """Static front-radar returns in the ego's original lane band beyond the start extend the obstacle (contiguous)."""
        if not len(tg):
            return
        m = (tg[:, 7] == 0) & (np.abs(tg[:, 1]) <= P["band"]) & (np.abs(tg[:, 5]) < P["moving_v"]) \
            & (tg[:, 0] >= st["start_s"] - 1.0) & (tg[:, 0] <= st["start_s"] + P["obs_ext_max"])
        for s_ in np.sort(tg[m, 0]):
            if s_ > st["end_s"] + P["obs_ext_gap"]:
                break
            st["end_s"] = max(st["end_s"], float(s_) + 2.0)

    def _gap(self, st, tg, ego_s, speed, t, lat_ego):
        """(open, reason). Rear radars: moving targets closing in the target-lane band behind; front radar: oncoming targets in
        the band ahead (the band is the target lane: route lateral within P["band"] of `offset`)."""
        frac = lat_ego / st["offset"] if st["offset"] else 0.0
        began = frac >= 0.05
        T, D = (P["t_abort"], P["d_abort"]) if began else (P["t_start"], P["d_start"])
        if not len(tg):
            return (t - self.last_close > P["side_mem_s"]), ("side_memory" if t - self.last_close <= P["side_mem_s"] else None)
        band = np.abs(tg[:, 1] - st["offset"]) <= P["band"]
        moving = np.abs(tg[:, 5]) >= P["moving_v"]
        rear = band & moving & (tg[:, 7] > 0) & (tg[:, 4] < -0.5)
        rear_bumper = ego_s - EGO_BACK
        for s_, vr in zip(tg[rear, 0], tg[rear, 4]):
            d_behind = rear_bumper - s_
            if d_behind < 8.0:
                self.last_close = t
            if d_behind < D + (-vr) * T:
                return False, "behind %.1f m closing %.1f" % (d_behind, -vr)
        if t - self.last_close <= P["side_mem_s"]:
            return False, "side_memory"
        front = band & moving & (tg[:, 7] == 0) & (tg[:, 4] < -(speed + 1.0))
        need = max(0.0, st["end_s"] + 23 - ego_s) / max(speed, 2.0) + 2.0
        for s_, vr in zip(tg[front, 0], tg[front, 4]):
            dist = s_ - ego_s - EGO_FRONT
            if dist < 15.0 or dist / max(-vr, 0.5) < need:
                return False, "oncoming %.1f m closing %.1f" % (dist, -vr)
        return True, None

    def _log(self, t, ego_s, speed, lanes, tg, lx, st, gap, why):
        if t - self.last_log < 0.2 - 1e-6:
            return
        self.last_log = t
        mv = tg[np.abs(tg[:, 5]) >= P["moving_v"]] if len(tg) else tg
        rec = dict(t=round(t, 2), v=round(float(speed), 2), ego_s=round(ego_s, 2), lead=lx, lanes=lanes, gap=gap, why=why,
                   n_radar=int(len(tg)), moving=[[round(float(x), 1) for x in row[[0, 1, 4, 5, 6, 7]]] for row in mv[:40]],
                   state=None if st is None else {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items() if k != "lanes"})
        self.log.write(json.dumps(rec) + "\n")

    def close(self):
        if not self.log.closed:
            self.log.close()
