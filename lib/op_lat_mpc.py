"""openpilot's legacy lateral MPC (the controller that tracked the model path before openpilot 0.9.5 moved it into the model), in numpy.

Source (read, not recalled): commaai/openpilot tag v0.9.4 (68aba7cef46d96201944da2f054712b708563c29), files
  selfdrive/controls/lib/lateral_planner.py        LateralPlanner.update: weights, reference, p, x0 carry-over, publish (psis, curvatures)
  selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py gen_lat_ocp / LateralMpc: model, cost, horizon (identical in v0.9.5)
  selfdrive/controls/lib/drive_helpers.py          get_lag_adjusted_curvature (controlsd's actuator-delay compensation and curvature-rate limit)
Mirrored: state [x, y, psi, psi_rate], control psi_accel, N = 32 nodes at T_IDXS, parameters p = (v_plan_i, rotation radius), NONLINEAR_LS cost on
  y, v' psi, v' psi_rate, v' u, u / (v + 0.1)  with v' = v + SPEED_OFFSET 10 and W = diag(PATH 1.0, LATERAL_MOTION 0.11, LATERAL_ACCEL 0.0,
  LATERAL_JERK 0.04, STEERING_RATE 700); references y_pts, heading_pts (and yaw_rate_pts, weight 0) of the model plan in the CURRENT vehicle frame;
  x0 = [0, 0, 0, psi_rate_carry], carry = x_sol[:, 3] interpolated one model step ahead (the controller's own desired yaw rate, not a measurement);
  curvatures = x_sol[:, 3] / v_plan[0], psis = x_sol[:CONTROL_N, 2]; get_lag_adjusted_curvature with delay = `delay` (0.275 model s = the spec chain's
  action time) and MAX_LATERAL_JERK 5.
NOT mirrored (declared): acados SQP_RTI with one HPIPM iteration is replaced by the exact solution of the QP of the small-angle linearisation
  (sin psi ~ psi, cos psi ~ 1: y' = v psi + R psi_rate, psi' = psi_rate, psi_rate' = u, exact discretisation per interval, control constant per
  interval as ERK nodes); the state bounds |psi| <= 90 deg, |psi_rate| <= 50 deg/s are not imposed (they do not bind at HUGSIM speeds);
  the rotation radius uses Toyota-class constants (wheelbase 2.69 m, centerToFront 0.44 wb, mass 1.5 t, rear stiffness 40 kN/rad);
  get_speed_error is not applied (v_plan = |plan velocity| clipped at MIN_SPEED 1).
Hook: experiments/hugsim/lib/zs_agent.py opt op_ctrl_src "plan_mpc" (preset spec_plan_mpc); the result is the curvature that rides to lib/op_ctrl.py.
"""
import numpy as np

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
N, CONTROL_N, DT_MDL, MIN_SPEED, MAX_LATERAL_JERK = 32, 17, 0.05, 1.0, 5.0
SPEED_OFFSET = 10.0
PATH_COST, LATERAL_MOTION_COST, LATERAL_ACCEL_COST, LATERAL_JERK_COST, STEERING_RATE_COST = 1.0, 0.11, 0.0, 0.04, 700.0
WB = 2.69
FACTOR1 = WB - 0.44 * WB
FACTOR2 = 0.44 * WB * 1500.0 / (WB * 40000.0)


class LatMpc:
    def __init__(self, delay=0.275):
        self.delay = delay
        self.w = 0.0                                   # x0[3]: the carried desired yaw rate

    def reset(self):
        self.w = 0.0

    def solve(self, y_ref, psi_ref, v):
        """Exact QP of the linearised problem. Returns x_sol columns (y, psi, psi_rate) at the 33 nodes and u (32)."""
        dt = np.diff(T_IDXS)
        R = np.clip(FACTOR1 - FACTOR2 * v ** 2, 0.0, None)
        s0 = np.array([0.0, 0.0, self.w])
        Phi = np.zeros((N + 1, 3, 3)); Gam = np.zeros((N + 1, 3, N))
        Phi[0] = np.eye(3)
        for k in range(N):
            h, vk, Rk = dt[k], v[k], R[k]
            F = np.array([[1, vk * h, vk * h * h / 2 + Rk * h], [0, 1, h], [0, 0, 1]])
            G = np.array([vk * h ** 3 / 6 + Rk * h * h / 2, h * h / 2, h])
            Phi[k + 1] = F @ Phi[k]
            Gam[k + 1] = F @ Gam[k]
            Gam[k + 1][:, k] += G
        vo = v + SPEED_OFFSET
        rows, rhs = [], []
        for k in range(N + 1):
            base = Phi[k] @ s0
            rows.append(np.sqrt(PATH_COST) * Gam[k, 0]); rhs.append(np.sqrt(PATH_COST) * (y_ref[k] - base[0]))
            rows.append(np.sqrt(LATERAL_MOTION_COST) * vo[k] * Gam[k, 1]); rhs.append(np.sqrt(LATERAL_MOTION_COST) * vo[k] * (psi_ref[k] - base[1]))
        for k in range(N):
            e = np.zeros(N)
            e[k] = np.sqrt(LATERAL_JERK_COST) * vo[k]; rows.append(e.copy()); rhs.append(0.0)
            e[k] = np.sqrt(STEERING_RATE_COST) / (v[k] + 0.1); rows.append(e.copy()); rhs.append(0.0)
        u = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        x = np.einsum("kij,j->ki", Phi, s0) + np.einsum("kin,n->ki", Gam, u)
        return x, u

    def step(self, pos, yaw, vel, n_model_steps=1):
        """One controlsd + lateral planner update. pos (33, 3+) x forward / y lateral, yaw (33,), vel (33,) forward speed of the model plan, all in the
        current vehicle frame and model units. Returns the lag-adjusted desired curvature (1/m)."""
        v = np.clip(np.asarray(vel, float), MIN_SPEED, np.inf)
        x, u = self.solve(np.asarray(pos, float)[:N + 1, 1], np.asarray(yaw, float)[:N + 1], v)
        v_ego = v[0]
        if not np.isfinite(x).all():
            self.reset()
            return 0.0
        self.w = float(np.interp(DT_MDL * n_model_steps, T_IDXS, x[:, 2]))
        curv = x[:CONTROL_N, 2] / v_ego
        psis = x[:CONTROL_N, 1]
        psi = np.interp(self.delay, T_IDXS[:CONTROL_N], psis)
        des = 2 * psi / (v_ego * self.delay) - curv[0]
        mr = MAX_LATERAL_JERK / v_ego ** 2
        return float(np.clip(des, curv[0] - mr * DT_MDL, curv[0] + mr * DT_MDL))
