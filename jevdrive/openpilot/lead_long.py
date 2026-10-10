"""openpilot's own lead path, from the model's lead outputs to a speed limit: radard's vision lead, the lead MPC of long_mpc.py and the
planner tick of longitudinal_planner.py (openpilot master ec95db3f, the commit lib/op_ctrl.py emulates). Written by lane FIX1 for the
AlpaSim drivers (experiments/alpasim/lib/serve_fix.py, switch JEV_LEAD, decision 226; that module's docstring lists what is taken from
which openpilot file and what is not emulated) and moved here unchanged when a third topic needed it (HUGSIM agent option `op_lead`,
lane HLEAD). Importers: experiments/alpasim/lib/serve_fix.py (re-exports every name), experiments/op_parity/scripts/diag1_lead.py and
experiments/body1 through serve_fix, experiments/hugsim/lib/zs_agent.py through PlanLead below.

Needs numpy; scipy is imported when a LongMpc is built.
"""
from __future__ import annotations

import numpy as np

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


# ---------------------------------------------------------------- a position plan on any time grid (boards whose controller tracks plan points)
T_E2E = 1.0                                               # the planner's "e2e" candidate: the plan's own acceleration over its first second (serve_fix.T_TRACK)
DT_PROFILE = 0.05


class PlanLead:
    """The lead limit on a position plan given at arbitrary times: the rule of serve_fix.Serve.__call__ with JEV_LEAD=1 (cap = max(v0 +
    v_mpc(t) - v_mpc(0), 0), served speed = pointwise min(plan speed, cap), a_e2e = the plan's acceleration over the first T_E2E seconds)
    without its fixed 8 x 0.5 s grid. The plan's path is kept; its points are pulled back along it. Only removes speed, no latch. One
    instance per episode. All arguments are in the model's units and clock (a board that dilates the clock converts before calling)."""

    def __init__(self, cam_to_front: float):
        self.planner, self.v_prev, self.t_prev = LeadPlanner(cam_to_front), None, None

    def __call__(self, xy, t, v0: float, t_now: float, lead_mu, lead_p, model_v: float):
        """xy (n, 2) plan points from the device position (the t = 0 point included), t (n,) their times (s, t[0] = 0), v0 the ego speed,
        t_now the decision time (s), lead_mu (3, 6, 4) and lead_p (3,) the decoded lead means and probabilities of the same forward pass,
        model_v the plan's own speed at t = 0 -> (xy', info). xy' is xy itself when the limit removes nothing."""
        xy, t, v0 = np.asarray(xy, np.float64), np.asarray(t, np.float64), max(float(v0), 0.0)
        a0 = 0.0 if self.v_prev is None or t_now <= self.t_prev else (v0 - self.v_prev) / (t_now - self.t_prev)
        self.v_prev, self.t_prev = v0, float(t_now)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        tg = np.arange(DT_PROFILE / 2, t[-1], DT_PROFILE)                      # profile grid (interval midpoints), as serve_fix.TG
        v = (np.diff(s) / np.diff(t))[np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 2)]
        a_e2e = (float(np.interp(T_E2E, tg, v)) - v0) / T_E2E
        p = np.clip(np.asarray(lead_p, np.float64).ravel()[:3], 1e-6, 1 - 1e-6)
        raw = np.concatenate([np.asarray(lead_mu, np.float64).reshape(72), np.zeros(72)])   # the raw layout LeadPlanner reads (means, then log std)
        vs, info = self.planner.update(int(round(t_now * 1e6)), v0, a0, raw, np.log(p / (1 - p)), float(model_v), a_e2e)
        cap = np.maximum(v0 + np.interp(tg, T_MPC, vs) - vs[0], 0.0)
        vn = np.minimum(v, cap)
        sn = np.interp(t, np.r_[0.0, tg + DT_PROFILE / 2], np.r_[0.0, np.cumsum(vn * DT_PROFILE)])
        q = np.minimum(s, sn)
        info.update(cut=round(float(s[-1] - q[-1]), 3), changed=bool(np.max(s - q) > 1e-6))
        if not info["changed"]:
            return xy, info
        i = np.clip(np.searchsorted(s, q, side="right") - 1, 0, len(s) - 2)
        ds = s[i + 1] - s[i]
        f = np.clip(np.where(ds > 1e-9, (q - s[i]) / np.where(ds > 1e-9, ds, 1.0), 0.0), 0.0, 1.0)
        return xy[i] + f[:, None] * (xy[i + 1] - xy[i]), info
