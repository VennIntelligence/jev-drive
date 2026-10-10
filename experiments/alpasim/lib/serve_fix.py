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


def place(p, ds, v):
    """The path sampled where the speed profile v (on TG) has arrived at 0.5 .. 4 s; beyond the plan's end the path continues straight."""
    s = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 1e-6))])
    sn = np.cumsum(np.maximum(v, 0.0) * DT)[K5 - 1::K5]
    far = p[-1] + np.array([60.0 * np.cos(p[-1, 2]), 60.0 * np.sin(p[-1, 2]), 0.0])
    P, S = np.concatenate([p, far[None]]), np.concatenate([s, [s[-1] + 60.0]])
    return np.stack([np.interp(sn, S, P[:, 0]), np.interp(sn, S, P[:, 1]), np.interp(sn, S, np.unwrap(P[:, 2]))], 1)


def join(vp, v0: float, tau: float):
    """(a): start at the ego's speed, blend linearly into the plan's segment speeds over tau seconds."""
    return v0 + (vp - v0) * np.minimum(TG / max(tau, 1e-3), 1.0)


class Serve:
    """One per session. __call__(plan output, ego speed, ego acceleration, decision time) -> the poses to return and a log record."""

    def __init__(self, vcont: float = 0.0, lead: bool = False, cam_to_front: float = 2.26, base: int = 0):
        self.vcont, self.lead, self.base = float(vcont), LeadPlanner(cam_to_front) if lead else None, int(base)
        self.gate = LeadPlanner(cam_to_front) if base == 2 else None      # its probability filter only

    def __call__(self, o: dict, v0: float, a0: float, t_us: int):
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
        if self.lead is not None:
            a_e2e = (float(np.interp(T_TRACK, TG, v)) - v0) / T_TRACK
            vs, li = self.lead.update(t_us, v0, float(a0), o["lead"], o["lead_prob"], float(o["mu"][0, 3]), a_e2e)
            cap = np.maximum(v0 + np.interp(TG, T_MPC, vs) - vs[0], 0.0)
            li["cut"] = round(float(np.sum(np.maximum(v - cap, 0.0)[:4 * K5]) * DT), 3)      # metres removed from the first 2 s
            li["vm"] = round(float(o["mu"][0, 3]), 3)
            v = np.minimum(v, cap)
            info["lead"] = li
        if self.vcont <= 0 and self.lead is None and not self.base:
            return o["poses"], info
        info["ms"] = round(1e3 * (time.perf_counter() - t), 3)
        return place(p, ds, v), info


VCONT, LEAD = float(os.environ.get("JEV_VCONT", "0") or 0), os.environ.get("JEV_LEAD", "0") == "1"
BASE = int(os.environ.get("JEV_BASE", "0") or 0)
NEED_LEAD = LEAD or BASE == 2
ON = VCONT > 0 or LEAD or BASE > 0
if LEAD:
    import scipy.optimize  # noqa: F401  (paid at start-up, not inside the first decision)


SUFFIX = (f"-vc{VCONT:g}" if VCONT > 0 else "") + ("-lead" if LEAD else "") + (f"-base{BASE}" if BASE else "")


def cam_to_front(vehicle, cam_x: float) -> float:
    """Camera to front bumper along the rig x axis from the session's ego box (RolloutSpec.VehicleDefinition)."""
    return float(vehicle.rig_to_bounding_box.vec.x + vehicle.bounding_box.size_x / 2 - cam_x)


def new(vehicle, cam_x: float):
    """The session's Serve, or None when both switches are off (the driver then returns the plan as before, bit for bit)."""
    return Serve(VCONT, LEAD, cam_to_front(vehicle, cam_x), BASE) if ON else None


def apply(fix, o: dict, v0: float, a0: float, t_us: int):
    """Driver hook: o["poses"] becomes the served poses, o["poses_model"] keeps the plan; -> the log record (None when off)."""
    if fix is None:
        return None
    o["poses_model"] = o["poses"]
    o["poses"], info = fix(o, v0, a0, t_us)
    return info
