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

# ---- openpilot ec95db3f, VERBATIM values: opendbc car/interfaces.py, radard.py, long_mpc.py, longitudinal_planner.py, drive_helpers.py
ACCEL_MIN, ACCEL_MAX = -3.5, 2.0
DT_MDL = 0.05
LEAD_PROB_RC, LEAD_PRESENT = 0.2, 0.5
VISION_A_LEAD_TAU, _LEAD_ACCEL_TAU = 0.3, 1.5
N = 12
T_MPC = 10.0 * (np.arange(N + 1) / N) ** 2
T_DIFFS = np.diff(T_MPC, prepend=[0.0])
H = np.diff(T_MPC)
X_EGO_OBSTACLE_COST, J_EGO_COST, A_CHANGE_COST, DANGER_ZONE_COST, LIMIT_COST = 3.0, 5.0, 200.0, 100.0, 1e6
LEAD_DANGER_FACTOR, COMFORT_BRAKE, STOP_DISTANCE, MIN_X_LEAD_FACTOR, T_FOLLOW = 0.75, 2.5, 6.0, 0.5, 1.45
V_DESIRED_RC = 2.0
ACTION_T, MIN_STABLE_DELAY = 0.15 + DT_MDL, 0.3           # longitudinalActuatorDelay + DT_MDL
V_STANDSTILL = 0.1
V_MOVING = 0.5                                            # JEV_BASE=2: decision 218's moving / standstill split


def _maps():
    """Node states of the triple integrator under piecewise-constant jerk are affine in the jerk vector: x = x_free + Gx j, ..."""
    Gx, Gv, Ga = (np.zeros((N + 1, N)) for _ in range(3))
    for i, h in enumerate(H):
        Ga[i + 1] = Ga[i]
        Ga[i + 1, i] += h
        Gv[i + 1] = Gv[i] + h * Ga[i]
        Gv[i + 1, i] += h * h / 2
        Gx[i + 1] = Gx[i] + h * Gv[i] + h * h / 2 * Ga[i]
        Gx[i + 1, i] += h ** 3 / 6
    return Gx, Gv, Ga


GX, GV, GA = _maps()
_WS = np.r_[H, 1.0]                                        # acados scales a stage cost by its shooting interval; the terminal cost is unscaled
_SQ_OBS = np.sqrt(_WS * X_EGO_OBSTACLE_COST)
_SQ_J, _SQ_DZ = np.sqrt(H * J_EGO_COST), np.sqrt(H * DANGER_ZONE_COST)
_SQ_A = np.sqrt(H * A_CHANGE_COST * np.interp(T_MPC[:-1], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0]))
_C = np.concatenate([GV[1:-1], GA[1:-1], -GA[1:-1]])     # v >= 0, a >= ACCEL_MIN, a <= ACCEL_MAX at the stages after the fixed first one


def safe_distance(v):
    """VERBATIM long_mpc.get_safe_obstacle_distance at T_FOLLOW (standard personality)."""
    return v * v / (2 * COMFORT_BRAKE) + T_FOLLOW * v + STOP_DISTANCE


def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    """VERBATIM LongitudinalMpc.extrapolate_lead."""
    a_traj = a_lead * np.exp(-a_lead_tau * (T_MPC ** 2) / 2.0)
    v_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_traj), 0.0, 1e8)
    return x_lead + np.cumsum(T_DIFFS * v_traj), v_traj


def obstacle(lead, v_ego):
    """LongitudinalMpc.process_lead + the stopped-equivalence distance: lead = None | (dRel, vLead, aLeadK, aLeadTau) -> x_obstacle (13,)."""
    x, v, a, tau = lead if lead is not None else (50.0, v_ego + 10.0, 0.0, _LEAD_ACCEL_TAU)
    x = np.clip(x, MIN_X_LEAD_FACTOR * (v_ego + v) * (v_ego - v) / (-ACCEL_MIN * 2), 1e8)
    xl, vl = extrapolate_lead(x, np.clip(v, 0.0, 1e8), np.clip(a, -10.0, 5.0), tau)
    return xl + vl * vl / (2 * COMFORT_BRAKE)


class LongMpc:
    """openpilot's lead MPC (long_mpc.py gen_long_ocp) in the 12 jerks: the same least-squares cost; the three limits that carry the slack
    weight 1e6 there (v >= 0, ACCEL_MIN <= a <= ACCEL_MAX) are linear in the jerks and enter as constraints of scipy's SLSQP (the
    danger-zone slack, weight 100, stays a one-sided penalty). Warm-started from the previous tick; 8 iterations reproduce the
    converged closed-loop gap to a few 0.1 m on the toy cases at 0.4-1 ms per tick; when the limits cannot all hold from the current state
    (standing with a braking command) the solve is repeated with v >= 0 as the slack penalty."""

    def __init__(self, dt: float = DT_MDL, iters: int = 8):
        from scipy.optimize import minimize
        self.dt, self.iters, self.ftol, self._min = dt, iters, 1e-6, minimize
        self.reset()

    def reset(self):
        self.j, self.v0 = np.zeros(N), 0.0
        self.a_prev = np.zeros(N + 1)
        self.v_sol, self.a_sol = np.zeros(N + 1), np.zeros(N + 1)

    def states(self, j, v0, a0):
        return v0 * T_MPC + a0 * T_MPC ** 2 / 2 + GX @ j, v0 + a0 * T_MPC + GV @ j, a0 + GA @ j

    def cost(self, j, v0, a0, xo, wa, soft_v: bool = False):
        """-> 0.5 |r|^2 and its gradient."""
        x, v, a = self.states(j, v0, a0)
        den = np.maximum(v + 10.0, 1.0)
        dd, d, gap = v / COMFORT_BRAKE + T_FOLLOW, safe_distance(v), xo - x
        e, ez = (gap - d) / den, (gap[:-1] - LEAD_DANGER_FACTOR * d[:-1]) / den[:-1]
        act = ez < 0
        de_v = (-dd * den - (gap - d)) / den ** 2
        dz_v = (-LEAD_DANGER_FACTOR * dd[:-1] * den[:-1] - (gap[:-1] - LEAD_DANGER_FACTOR * d[:-1])) / den[:-1] ** 2
        r = [_SQ_OBS * e, wa * (a[:-1] - self.a_prev[:-1]), _SQ_J * j, _SQ_DZ * ez * act]
        J = [_SQ_OBS[:, None] * (-GX / den[:, None] + de_v[:, None] * GV), wa[:, None] * GA[:-1], np.diag(_SQ_J),
             (_SQ_DZ * act)[:, None] * (-GX[:-1] / den[:-1, None] + dz_v[:, None] * GV[:-1])]
        if soft_v:                                          # the fallback: v >= 0 as openpilot's slack penalty
            neg = v[:-1] < 0
            r.append(np.sqrt(H * LIMIT_COST) * v[:-1] * neg), J.append((np.sqrt(H * LIMIT_COST) * neg)[:, None] * GV[:-1])
        r, J = np.concatenate(r), np.concatenate(J)
        return 0.5 * r @ r, J.T @ r

    def update(self, v, a, leads, prev_accel_constraint: bool = True):
        """One planner tick: set_weights / set_cur_state / update / run of LongitudinalMpc. leads: two of None | (dRel, vLead, aLeadK,
        aLeadTau). -> v_solution, a_solution (13,)."""
        if abs(self.v0 - v) > 2.0:                          # set_cur_state: the previous solution is no initial guess any more
            self.j = np.zeros(N)
        self.v0 = v
        xo = np.minimum(obstacle(leads[0], v), obstacle(leads[1], v))
        wa = _SQ_A * float(prev_accel_constraint)
        nv = N - 1
        b = np.concatenate([-(v + a * T_MPC[1:-1]), np.full(nv, ACCEL_MIN - a), np.full(nv, a - ACCEL_MAX)])
        opt = {"maxiter": self.iters, "ftol": self.ftol}
        res = self._min(self.cost, self.j, args=(v, a, xo, wa), jac=True, method="SLSQP", options=opt,
                        constraints=[{"type": "ineq", "fun": lambda j: _C @ j - b, "jac": lambda j: _C}])
        j = res.x
        if not np.all(np.isfinite(j)) or (_C @ j - b).min() < -1e-3:      # the limits cannot all hold from this state: accel limits only
            Ca, ba = _C[nv:], b[nv:]
            res = self._min(self.cost, np.zeros(N), args=(v, a, xo, wa, True), jac=True, method="SLSQP", options={"maxiter": 40, "ftol": 1e-9},
                            constraints=[{"type": "ineq", "fun": lambda j: Ca @ j - ba, "jac": lambda j: Ca}])
            j = res.x
            if not np.all(np.isfinite(j)):
                self.reset()
                return np.full(N + 1, v), np.zeros(N + 1)
        _, vs, as_ = self.states(j, v, a)
        self.j, self.v_sol, self.a_sol = j, vs, as_
        self.a_prev = np.interp(T_MPC + self.dt, T_MPC, as_)
        return vs, as_


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL):
    """VERBATIM drive_helpers.get_accel_from_plan."""
    v_now, a_now = speeds[0], accels[0]
    if action_t < MIN_STABLE_DELAY:
        v_target = v_now + (action_t / MIN_STABLE_DELAY) * (np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
        v_target = np.interp(action_t, t_idxs, speeds)
    return 2 * (v_target - v_now) / action_t - a_now


class LeadPlanner:
    """radard's vision lead + LongitudinalPlanner.update, one instance per session."""

    def __init__(self, cam_to_front: float):
        self.front, self.mpc = float(cam_to_front), LongMpc()
        self.prob, self.t_us = [0.0, 0.0], None
        self.v_des = self.a_out = 0.0

    def inside(self, v_ego: float, lead, lead_prob, model_v_ego: float) -> bool:
        """A lead inside openpilot's following distance now (one filter step on this output; used without the MPC by JEV_BASE=2)."""
        mu, p = np.asarray(lead, np.float64)[:72].reshape(3, 6, 4), 1 / (1 + np.exp(-float(lead_prob[0])))
        alpha = DT_MDL / (LEAD_PROB_RC + DT_MDL)
        self.prob0 = p if p > getattr(self, "prob0", 0.0) else (1 - alpha) * self.prob0 + alpha * p
        vl = max(v_ego + mu[0, 0, 2] - model_v_ego, 0.0)
        return bool(self.prob0 > LEAD_PRESENT and mu[0, 0, 0] - self.front + vl * vl / (2 * COMFORT_BRAKE) < safe_distance(v_ego))

    def update(self, t_us: int, v_ego: float, a_ego: float, lead, lead_prob, model_v_ego: float, a_e2e: float):
        """lead (144,) raw lead output (mean 3 x 6 x 4 then log std), lead_prob (3,) logits, model_v_ego the plan's own speed at t = 0
        -> (t, v) of the lead MPC's speed solution after this call's ticks, info."""
        mu, p = np.asarray(lead, np.float64)[:72].reshape(3, 6, 4), 1 / (1 + np.exp(-np.asarray(lead_prob, np.float64)))
        reset = self.t_us is None
        n = 1 if reset else int(np.clip(round((t_us - self.t_us) * 1e-6 / DT_MDL), 1, 20))
        self.t_us = t_us
        if reset:                                           # LongitudinalPlanner reset_state
            self.v_des, self.a_out = v_ego, float(np.clip(a_ego, ACCEL_MIN, ACCEL_MAX))
        alpha_p, alpha_v = DT_MDL / (LEAD_PROB_RC + DT_MDL), DT_MDL / (V_DESIRED_RC + DT_MDL)
        a_mpc, leads = 0.0, [None, None]
        for _ in range(n):
            for i in range(2):                              # radard: asymmetric filter on the lead probability
                self.prob[i] = p[i] if p[i] > self.prob[i] else (1 - alpha_p) * self.prob[i] + alpha_p * p[i]
            leads = [(mu[i, 0, 0] - self.front, v_ego + mu[i, 0, 2] - model_v_ego, mu[i, 0, 3], VISION_A_LEAD_TAU)
                     if self.prob[i] > LEAD_PRESENT else None for i in range(2)]
            self.v_des = max(0.0, (1 - alpha_v) * self.v_des + alpha_v * v_ego)
            vs, as_ = self.mpc.update(self.v_des, self.a_out, leads, not (reset or v_ego < V_STANDSTILL))
            reset = False
            a_mpc = float(get_accel_from_plan(vs, as_, T_MPC, ACTION_T))
            a_new = float(np.clip(min(a_mpc, a_e2e), ACCEL_MIN, ACCEL_MAX))
            self.v_des += DT_MDL * (a_new + self.a_out) / 2.0
            self.a_out = a_new
        l0 = leads[0]
        return self.mpc.v_sol, dict(n=n, p=[round(float(x), 3) for x in self.prob], d=None if l0 is None else round(float(l0[0]), 2),
                                    vl=None if l0 is None else round(float(l0[1]), 2), a_mpc=round(a_mpc, 3), a_e2e=round(float(a_e2e), 3),
                                    a_out=round(self.a_out, 3), src="mpc" if a_mpc < a_e2e else "e2e")


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
