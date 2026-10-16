"""Serving-side speed profile of the trajectory an AlpaSim driver returns (lane FIX1, plans/2026-10-09-fix1-prereg.md). The path of the
model's plan is never changed; only where along it the returned poses sit. Three independent switches, all off by default:

  JEV_VCONT=<s>  (a) speed-continuous serving. Unset / 0 = off. The drivers return the plan's 0.5 s points resampled linearly, so the
                 returned trajectory starts at the plan's mean speed over its first 0.5 s whatever the ego's speed is. With the switch
                 the speed profile starts at the ego's speed and blends linearly into the plan's own segment speeds over <s> seconds.
                 The value follows from the controller, not from tuning: AlpaSim's nonlinear MPC (src/controller/alpasim_controller,
                 0bb4c4b, run as shipped) has a 20 x 0.1 s horizon and reads the reference only from horizon index idx_start_penalty = 10
                 on, i.e. positions 1.0 .. 2.0 s ahead, with position weights 2 / 1 against an acceleration weight of 0.1. A linear blend
                 over exactly that 1.0 s is the reference a constant acceleration (v_plan - v_ego) / 1 s meets with zero position and
                 speed error at the first tracked index; the unblended reference asks for the position offset (v_plan - v_ego) x 1 s
                 to be removed inside the untracked second (twice the deceleration, then the opposite sign). JEV_VCONT=1.0 is that value.
  JEV_LEAD=1     (b) lead-aware longitudinal limit: openpilot's own path from the model's lead outputs to an acceleration limit, from
                 source (openpilot master ec95db3f, the commit lib/op_ctrl.py emulates; read in tmp/opsrc, not recalled):
                   radard.py           lead probability through the asymmetric first-order filter (rc 0.2 s at 20 Hz), a lead is present
                                       above 0.5; vision-only lead: dRel = x - (camera to front bumper), vLead = v_ego + (lead v - the
                                       model's own ego speed, plan velocity at t = 0), aLeadK = lead a, aLeadTau 0.3; leadOne / leadTwo =
                                       the lead outputs for 0 s and 2 s
                   long_mpc.py         the lead MPC: jerk-controlled triple integrator on T = 10 (i / 12)^2 s, cost 3 on
                                       (x_obstacle - x - d(v)) / (v + 10) with d(v) = v^2 / (2 x 2.5) + 1.45 v + 6 and
                                       x_obstacle = x_lead + v_lead^2 / (2 x 2.5), 200 on the change of acceleration against the previous
                                       solution (fading to 0 between 1 and 2 s), 5 on jerk, slack penalties 1e6 on v >= 0 and
                                       -3.5 <= a <= 2.0 and 100 on the danger zone 0.75 d(v); no lead = a fake lead 50 m ahead at v + 10
                   longitudinal_planner.py   v_desired filter (rc 2 s) integrated with the output, a_target = get_accel_from_plan at
                                       0.2 s, output = clip(min(lead MPC, e2e), -3.5, 2.0) (no cruise candidate: there is no set speed)
                 Same cost, weights, horizon and constraints as openpilot's acados problem, solved here by scipy's SLSQP (the 1e6 slack limits
                 as constraints; warm-started, 8 iterations per planner tick; acados runs one SQP iteration per tick). The planner ticks at 20 Hz
                 as in the car: one `drive` call advances it by (time since the previous call) / 0.05 s ticks with the model's last
                 output held (2 ticks on PAI, 10 on nuPlan), the convention of lib/op_ctrl.py.
                 What is ours (the interface to a position-tracking controller, openpilot has none): the served speed profile is the
                 pointwise minimum of the profile without the switch and the lead MPC's speed solution re-anchored at the ego's speed.
                 The limit only removes speed, it never adds any, and it has no latch: it lifts in the tick the MPC's solution
                 exceeds the plan's again. The "e2e" candidate of the planner is the acceleration of the profile without the switch
                 over the first 1.0 s.
                 Not emulated: radar tracks, the cruise candidate, FCW, LongControl's stopping state (the position tracker holds a
                 reference that does not move).

  JEV_BASE=1|2   (c) the adapter's path with the frozen base model's speed profile (added after decision 218: the adapter's speed behaviour is a
                 function of the ego state only and replaces the base model's lead-triggered braking and standstill hold). The same policy
                 pass is run once more without the adapter bias (that plan is the shipped model's); the adapter's path is then sampled at
                 the base plan's arc lengths. 1 = on every decision. 2 = while moving (ego speed >= 0.5 m/s, decision 218's split); at
                 standstill the adapter's own profile (its launch) is kept unless the lead output reports a lead inside openpilot's
                 following distance (filtered probability > 0.5 and dRel + v_lead^2 / 5 < d(v_ego)), where the base profile is served.
                 Composes with the other two: (a) blends from the ego speed into whichever profile was chosen, (b) limits it.

  JEV_ACC=1      (d) openpilot-style longitudinal (experiments/route_follow/plans/2026-10-16-pai-acc-prereg.md, nuPlan:
                 2026-10-16-nuplan-acc-prereg.md; the PAI and nuPlan drivers pass the route and the session clock; the handoff time is the
                 driver's: ACC_T_HAND on PAI, ACC_T_HAND_NUPLAN in sh30_driver). The path stays the model's (geometry unchanged); the speed
                 profile is re-timed to
                   v_target = min(v_set, v_curve(s)) [, model]  then (b)'s lead limit as before (min), then placed on the path
                 - v_set: the ego speed at the simulator handoff (first call with t_s >= t_hand; the force-GT speed is the human's), held as
                   cruise set speed; at each decision max(v_hand, the current model plan's own 2 .. 4 s speed), so a standstill handoff gets the
                   model's own launch target instead of 0. (A running maximum of the model's speed, the smoke's first version, ratcheted up: the
                   model follows its own faster history; a174faed 2.3 -> 19.1 m/s. The current value has no ratchet: decision 315's g < 1)
                 - v_curve(s): the route the driver receives (AlpaSim route messages, 10 points 42 .. 80 m ahead, carried as memory by
                   rmem.Tracker.polyline as decision 287 did, so a curve stays covered once it has come within 80 m), resampled at 1 m,
                   curvature = heading change of +-ACC_HW m chords over 2 ACC_HW m; v^2 |k| <= ACC_ALAT, a backward pass decelerating at
                   ACC_DCURVE so the limit is reached at the curve; where the route does not reach (the first ~40 m of a session) no limit
                 - profile: from the ego speed, rising at most openpilot's cruise acceleration table A_CRUISE (by speed), falling at most ACC_DMAX
                 - stop guard: when the model's PREVIOUS plan (before any serving change) asks to stop or brake (vhist.py holdlead's two releases:
                   its 2 s arc < ACC_REL_ARC m, or its segment speeds (s1 - s4) / 1.5 s >= ACC_DEC_REL m/s^2), the profile is min(ACC, the
                   model's own profile as (a) serves it); when the model moves again the guard drops and ACC resumes
                 - the path is extended straight along the plan's end heading (ACC_FAR m) when the new profile runs past its end
                 Logged per decision as fix.acc: v_set, v_curve (at the arc reached at 1.5 s, and its minimum over the route ahead), v_lead,
                 v_model (the model's profile at 1.5 s), the winning term at 1.5 s (set | curve | model | lead; `ramp` when the profile is still
                 more than 0.3 m/s from it), guard (stop | brake | ''), route cover (m ahead).

  JEV_GAP=<t_follow>[,<stop m>]   (e) the following-gap calibration of (b)'s lead MPC (experiments/route_follow/plans/2026-10-16-acc-gap-prereg.md;
                 needs JEV_LEAD=1): openpilot's T_FOLLOW (1.45 s) and STOP_DISTANCE (6 m) of the desired distance
                 d(v) = v^2 / 5 + T_FOLLOW v + STOP_DISTANCE replaced by the given values (lead_long.LongMpc(t_follow, stop_distance)); unset = the
                 defaults, bit for bit (acc_check.py check 5). Why: the MPC holds openpilot's law on the model's lead, but the lead head reads a
                 moving lead ~1 m short and ~1 m/s slow (train scenes; x_obstacle = d + v_lead^2 / 5 loses ~0.45 v m), so the true gap sits
                 4-7 m above openpilot's law and the human's (results/acc_gap/diagnosis.md). The prereg's value 0.7,2.7 is fitted on train scenes so
                 that the true steady gap equals the human's (3.4 m + 1.13 s v) under that bias.

No learned component, no simulator ground truth: inputs are the plan, the checkpoint's own lead / lead_prob outputs of the same forward
pass, the ego speed / acceleration the simulator reports and the session's camera and ego box (camera to front bumper).
"""
from __future__ import annotations

import os
import time

import numpy as np

T_TRACK = 1.0                                             # AlpaSim MPC: idx_start_penalty (10) x dt_mpc (0.1 s)
DT, T_END = 0.05, 4.0
TG = np.arange(DT / 2, T_END, DT)                         # profile grid (interval midpoints)
K5 = int(round(0.5 / DT))

# ---- openpilot's lead path lives in jevdrive/openpilot/lead_long.py (moved there unchanged for the HUGSIM agent, lane HLEAD); every name
# this module had is re-exported, so `serve_fix.LeadPlanner`, `serve_fix.T_MPC` ... keep working for body1 / op_parity.
import pathlib as _pl
import sys as _sys

_R = str(_pl.Path(__file__).resolve().parents[3])
if _R not in _sys.path:
    _sys.path.append(_R)
from jevdrive.openpilot.lead_long import (  # noqa: E402, F401
    ACCEL_MAX, ACCEL_MIN, ACTION_T, A_CHANGE_COST, COMFORT_BRAKE, DANGER_ZONE_COST, DT_MDL, GA, GV, GX, H, J_EGO_COST, LEAD_DANGER_FACTOR,
    LEAD_PRESENT, LEAD_PROB_RC, LIMIT_COST, MIN_STABLE_DELAY, MIN_X_LEAD_FACTOR, N, STOP_DISTANCE, T_DIFFS, T_FOLLOW, T_MPC, V_DESIRED_RC,
    V_STANDSTILL, VISION_A_LEAD_TAU, X_EGO_OBSTACLE_COST, LeadPlanner, LongMpc, extrapolate_lead, get_accel_from_plan, obstacle, safe_distance)

V_MOVING = 0.5                                            # JEV_BASE=2: decision 218's moving / standstill split


# ---------------------------------------------------------------- the served profile
def path(poses):
    """Plan poses (8, 3) at 0.5 .. 4 s -> the polyline from the origin, its segment lengths and the plan's speed on TG."""
    p = np.concatenate([np.zeros((1, 3)), np.asarray(poses, np.float64)])
    ds = np.hypot(*np.diff(p[:, :2], axis=0).T)
    return p, ds, (ds / 0.5)[np.minimum((TG / 0.5).astype(int), 7)]


def place(p, ds, v, far_m: float = 60.0):
    """The path sampled where the speed profile v (on TG) has arrived at 0.5 .. 4 s; beyond the plan's end the path continues straight
    (far_m along the plan's end heading)."""
    s = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 1e-6))])
    sn = np.cumsum(np.maximum(v, 0.0) * DT)[K5 - 1::K5]
    far = p[-1] + np.array([far_m * np.cos(p[-1, 2]), far_m * np.sin(p[-1, 2]), 0.0])
    P, S = np.concatenate([p, far[None]]), np.concatenate([s, [s[-1] + far_m]])
    return np.stack([np.interp(sn, S, P[:, 0]), np.interp(sn, S, P[:, 1]), np.interp(sn, S, np.unwrap(P[:, 2]))], 1)


def join(vp, v0: float, tau: float):
    """(a): start at the ego's speed, blend linearly into the plan's segment speeds over tau seconds."""
    return v0 + (vp - v0) * np.minimum(TG / max(tau, 1e-3), 1.0)


# ---------------------------------------------------------------- (d) JEV_ACC
ACC_T_HAND = 1.6                                          # s since the session's first call: AlpaSim's force-GT ends at 1.7 s (PAI)
ACC_T_HAND_NUPLAN = 0.5                                   # nuPlan: force-GT 0.5 s at 0.5 s steps; the call at 0.5 s is the handoff step
ACC_ALAT = 2.4                                            # m/s^2: human median peak lateral acceleration in PAI turns (decision 293)
ACC_DCURVE = 1.5                                          # m/s^2: comfortable deceleration into a curve (backward pass)
ACC_DMAX = 2.5                                            # m/s^2: the profile's own deceleration bound (openpilot COMFORT_BRAKE)
A_CRUISE_V, A_CRUISE = (0.0, 10.0, 25.0, 40.0), (1.6, 1.2, 0.8, 0.6)   # openpilot longitudinal_planner A_CRUISE_MAX_BP / _VALS
ACC_HW = 3.0                                              # m: half chord of the route heading
ACC_KMAX = 0.2                                            # 1/m: curvature clip (openpilot's minimum radius 5 m)
ACC_OFF = 8.0                                             # m: a route farther than this from the ego is not used
ACC_REL_ARC, ACC_DEC_REL = 2.0, 1.0                       # vhist.py REL_ARC / DEC_REL (the holdlead releases)
ACC_FAR = 300.0                                           # m: straight extension of the path beyond the plan's end
T_WIN = 1.5                                               # s: the logged / winning-term time (AlpaSim's MPC tracks 1.0 .. 2.0 s)


def route_limit(M, v_set: float):
    """Route polyline M (n, 2) in the decision's rig frame (or None) -> (s_grid, v_lim) along the arc from the ego's projection, with the
    backward pass, and the covered length ahead; (None, 0.0) when the route is unusable."""
    if M is None:
        return None, 0.0
    M = np.asarray(M, np.float64).reshape(-1, 2)
    M = M[np.isfinite(M).all(1)]
    if len(M) >= 2:
        M = M[np.r_[True, np.hypot(*np.diff(M, axis=0).T) > 1e-6]]
    if len(M) < 2:
        return None, 0.0
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(M, axis=0).T))]
    d, L = np.diff(M, axis=0), np.diff(cum)
    u = d / L[:, None]
    t = np.clip(-(M[:-1] * u).sum(1), 0.0, L)
    q = M[:-1] + t[:, None] * u
    r = np.hypot(*q.T)
    i = int(np.argmin(r))
    s_e = cum[i] + t[i]
    if i == 0 and t[0] <= 0.0:                            # the route starts ahead (the first decisions): s_e < 0
        s_e, r[i] = float(-(M[0] * u[0]).sum()), abs(float(M[0] @ np.array([-u[0, 1], u[0, 0]])))
    if r[i] > ACC_OFF or cum[-1] - max(s_e, 0.0) < 2 * ACC_HW + 1.0:
        return None, max(0.0, float(cum[-1] - s_e))
    sg = np.arange(max(s_e, 0.0), cum[-1] + 1e-9, 1.0)
    pt = lambda x: np.stack([np.interp(x, cum, M[:, 0]), np.interp(x, cum, M[:, 1])], 1)  # noqa: E731
    c0, c1 = pt(sg) - pt(np.maximum(sg - 2 * ACC_HW, 0.0)), pt(np.minimum(sg + 2 * ACC_HW, cum[-1])) - pt(sg)   # chords behind / ahead
    dh = (np.arctan2(c1[:, 1], c1[:, 0]) - np.arctan2(c0[:, 1], c0[:, 0]) + np.pi) % (2 * np.pi) - np.pi
    ok = (np.hypot(*c0.T) > ACC_HW) & (np.hypot(*c1.T) > ACC_HW)          # both chords at least half length (not at the route's ends)
    k = np.where(ok, np.minimum(np.abs(dh) / (2 * ACC_HW), ACC_KMAX), 0.0)
    v = np.sqrt(ACC_ALAT / np.maximum(k, 1e-4))
    for j in range(len(v) - 2, -1, -1):
        v[j] = min(v[j], np.sqrt(v[j + 1] ** 2 + 2 * ACC_DCURVE * (sg[j + 1] - sg[j])))
    return (sg - s_e, v), float(cum[-1] - s_e)


class Acc:
    """(d): the session's set speed and the model's previous plan (guard)."""

    def __init__(self, t_hand: float = ACC_T_HAND):
        self.v_hand, self.arc, self.dec, self.t_hand = None, np.inf, 0.0, float(t_hand)

    def profile(self, poses, v0: float, vm, M, t_s):
        """-> (v on TG, log record). vm = the model's own profile as (a) serves it (on TG)."""
        p = np.r_[np.zeros((1, 2)), np.asarray(poses, np.float64)[:, :2]]
        seg = np.hypot(*np.diff(p, axis=0).T) / 0.5
        if self.v_hand is None and (t_s is None or t_s >= self.t_hand):
            self.v_hand = v0
        vset = max(self.v_hand, float(seg[4:].mean())) if self.v_hand is not None else v0
        guard = "stop" if self.arc < ACC_REL_ARC else "brake" if self.dec >= ACC_DEC_REL else ""
        self.arc, self.dec = float(seg[:4].sum() * 0.5), float((seg[0] - seg[3]) / 1.5)   # read by the next decision
        lim, cover = route_limit(M, vset)
        v, s, vc, out = v0, 0.0, np.full(len(TG), np.inf), np.empty(len(TG))
        for i in range(len(TG)):
            if lim is not None:                           # before the covered part: the backward pass continued; beyond it: no limit
                g0 = lim[0][0]
                vc[i] = (np.sqrt(lim[1][0] ** 2 + 2 * ACC_DCURVE * (g0 - s)) if s < g0 else float(np.interp(s, *lim, right=np.inf)))
            a_up = float(np.interp(v, A_CRUISE_V, A_CRUISE))
            v = float(np.clip(min(vset, vc[i]), max(v - ACC_DMAX * DT, 0.0), v + a_up * DT))
            out[i] = v
            s += v * DT
        if guard:
            out = np.minimum(out, vm)
        w = int(T_WIN / DT)
        info = dict(v_set=round(vset, 3), v_hand=None if self.v_hand is None else round(self.v_hand, 3), guard=guard,
                    v_curve=None if not np.isfinite(vc[w]) else round(float(vc[w]), 3),
                    v_curve_min=None if lim is None else round(float(lim[1][lim[0] >= 0].min()) if (lim[0] >= 0).any() else float(lim[1][-1]), 3),
                    v_model=round(float(vm[w]), 3), cover=round(cover, 1))
        return out, info


def won(info: dict, v15: float, cap15) -> None:
    """The term that sets the served speed at T_WIN (`ramp`: the profile is still > 0.3 m/s from it)."""
    c = {"set": info["v_set"], "curve": info["v_curve"], "model": info["v_model"] if info["guard"] else None, "lead": cap15}
    c = {k: x for k, x in c.items() if x is not None}
    k = min(c, key=c.get)
    info.update(v_lead=None if cap15 is None else round(cap15, 3), v15=round(v15, 3), won=k if abs(v15 - c[k]) <= 0.3 else "ramp", near=k)


class Serve:
    """One per session. __call__(plan output, ego speed, ego acceleration, decision time) -> the poses to return and a log record."""

    def __init__(self, vcont: float = 0.0, lead: bool = False, cam_to_front: float = 2.26, base: int = 0, acc: bool = False,
                 acc_t_hand: float = ACC_T_HAND, gap=None):
        self.vcont, self.base = float(vcont), int(base)
        self.lead = (LeadPlanner(cam_to_front) if gap is None else LeadPlanner(cam_to_front, *gap)) if lead else None   # gap: JEV_GAP
        self.gate = LeadPlanner(cam_to_front) if base == 2 else None      # its probability filter only
        self.acc = Acc(acc_t_hand) if acc else None

    def __call__(self, o: dict, v0: float, a0: float, t_us: int, route=None, t_s=None):
        t = time.perf_counter()
        v0 = max(float(v0), 0.0)
        p, ds, vp = path(o["poses"])
        info = {"v0": round(v0, 3), "vp0": round(float(vp[0]), 3)}
        if self.base:
            vb = path(o["poses_base"])[2]
            held = self.base == 2 and v0 < V_MOVING and self.gate.inside(v0, o["lead"], o["lead_prob"], float(o["mu"][0, 3]))
            use = self.base == 1 or v0 >= V_MOVING or held
            info.update(vb0=round(float(vb[0]), 3), base=bool(use), held=bool(held))
            vp = vb if use else vp
        v = join(vp, v0, self.vcont) if self.vcont > 0 else vp
        if self.acc is not None:                          # (d): the re-timed profile; v (the model's) only under the stop guard
            v, ai = self.acc.profile(o["poses"], v0, v, route, t_s)
        if self.lead is not None:
            a_e2e = (float(np.interp(T_TRACK, TG, v)) - v0) / T_TRACK
            vs, li = self.lead.update(t_us, v0, float(a0), o["lead"], o["lead_prob"], float(o["mu"][0, 3]), a_e2e)
            cap = np.maximum(v0 + np.interp(TG, T_MPC, vs) - vs[0], 0.0)
            li["cut"] = round(float(np.sum(np.maximum(v - cap, 0.0)[:4 * K5]) * DT), 3)      # metres removed from the first 2 s
            li["vm"] = round(float(o["mu"][0, 3]), 3)
            v = np.minimum(v, cap)
            info["lead"] = li
        if self.acc is not None:
            w = int(T_WIN / DT)
            won(ai, float(v[w]), float(cap[w]) if self.lead is not None else None)
            info["acc"] = ai
        if self.vcont <= 0 and self.lead is None and not self.base and self.acc is None:
            return o["poses"], info
        info["ms"] = round(1e3 * (time.perf_counter() - t), 3)
        return place(p, ds, v, ACC_FAR if self.acc is not None else 60.0), info


VCONT, LEAD = float(os.environ.get("JEV_VCONT", "0") or 0), os.environ.get("JEV_LEAD", "0") == "1"
BASE = int(os.environ.get("JEV_BASE", "0") or 0)
ACC = os.environ.get("JEV_ACC", "0") == "1"
_G = (os.environ.get("JEV_GAP", "") or "").strip()
GAP = None if _G in ("", "0") else tuple(float(x) for x in (_G.split(",") + [str(STOP_DISTANCE)])[:2])   # (t_follow s, stop distance m)
assert GAP is None or (LEAD and 0.3 <= GAP[0] <= 3.0 and 0.0 <= GAP[1] <= 10.0), f"JEV_GAP={_G!r}: <t_follow>[,<stop m>], needs JEV_LEAD=1"
NEED_LEAD = LEAD or BASE == 2
ON = VCONT > 0 or LEAD or BASE > 0 or ACC
if LEAD:
    import scipy.optimize  # noqa: F401  (paid at start-up, not inside the first decision)


SUFFIX = (f"-vc{VCONT:g}" if VCONT > 0 else "") + ("-lead" if LEAD else "") + (f"-base{BASE}" if BASE else "") + ("-acc" if ACC else "") + (f"-gap{GAP[0]:g}_{GAP[1]:g}" if GAP else "")


def cam_to_front(vehicle, cam_x: float) -> float:
    """Camera to front bumper along the rig x axis from the session's ego box (RolloutSpec.VehicleDefinition)."""
    return float(vehicle.rig_to_bounding_box.vec.x + vehicle.bounding_box.size_x / 2 - cam_x)


def new(vehicle, cam_x: float, acc_t_hand: float = ACC_T_HAND):
    """The session's Serve, or None when every switch is off (the driver then returns the plan as before, bit for bit).
    acc_t_hand: JEV_ACC's handoff time (s since the session's first call), per track."""
    return Serve(VCONT, LEAD, cam_to_front(vehicle, cam_x), BASE, ACC, acc_t_hand, GAP) if ON else None


def apply(fix, o: dict, v0: float, a0: float, t_us: int, **acc):
    """Driver hook: o["poses"] becomes the served poses, o["poses_model"] keeps the plan; -> the log record (None when off).
    acc (JEV_ACC only): route = the carried route polyline in the decision's rig frame, t_s = s since the session's first call."""
    if fix is None:
        return None
    o["poses_model"] = o["poses"]
    o["poses"], info = fix(o, v0, a0, t_us, **acc)
    return info
