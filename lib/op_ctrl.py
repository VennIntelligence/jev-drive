"""openpilot's production lateral path from the model's desired curvature to the car, for simulators that otherwise track plan positions.

Plan: experiments/hugsim/plans/2026-10-05-op-control-stack-prereg.md. Source (read, not recalled): openpilot master ec95db3f (2026-10-02) with
opendbc 35f7e081. The functions marked VERBATIM are copied with their logic unchanged (clip_curvature returns only the curvature); the rest wires them the
way selfdrive/modeld/modeld.py and selfdrive/controls/controlsd.py do at 20 Hz / 100 Hz:

  modeld (20 Hz)      desired_curvature = action[0] / max(1, v)^2 (Cinque has an `action` head; decode() in jevdrive/openpilot/model.py);
                      LAT_SMOOTH_SECONDS = 0.0 (no smoothing); v_ego <= MIN_LAT_CONTROL_SPEED 0.3: hold the previous action's curvature.
  controlsd (100 Hz)  latActive needs v_ego > max(minSteerSpeed, 0.3) (Toyota: minSteerSpeed 0, steerAtStandstill False); inactive ->
                      desired curvature := measured curvature (the wheel is left where it is); clip_curvature (jerk 5 m/s^3 / max(v, 1)^2,
                      lateral accel 3 m/s^2, |curvature| <= 0.2).
  car (plant)         openpilot's own model of the closed steering loop: the realised curvature is the desired curvature delayed by
                      lateralDelay (selfdrive/locationd/lagd.py identifies exactly this pure delay, clipped to [0.15, 0.65] s, initial
                      steerActuatorDelay + 0.2; latcontrol_torque uses the same delay to pick its setpoint). Optional: the steering-wheel angle
                      rate limit of an angle-controlled car (Toyota LTA, opendbc toyota/values.py ANGLE_LIMITS), as a sensitivity only.

Rule parameters (dict, JSON): {"delay": 0.25, "v_hold": 0.3, "v_active": 0.3, "lta": false, "sr": 14.3, "wb": 2.68986}.
delay is in simulator seconds; 0.25 = one HUGSIM step, consistent with the model's action_t 0.275 model-s (lateralDelay 0.2 + 0.075 frame and
action delay) under the exam's 1.25 clock dilation. Hook: hugsim_steer (HUGSIM tree `opctrl`, env OP_CTRL).
"""
import math

import numpy as np

# ---- openpilot constants (selfdrive/controls/lib/drive_helpers.py, selfdrive/modeld/modeld.py, common/realtime.py)
MIN_SPEED = 1.0
MAX_CURVATURE = 0.2
MAX_LATERAL_JERK = 5.0               # m/s^3
MAX_LATERAL_ACCEL_NO_ROLL = 3.0      # m/s^2
LAT_SMOOTH_SECONDS = 0.0
MIN_LAT_CONTROL_SPEED = 0.3
DT_CTRL = 0.01
G = 9.81
# ---- opendbc toyota/values.py CarControllerParams.ANGLE_LIMITS (LTA), deg of steering wheel per 10 ms frame, by v (m/s)
LTA_UP = ([5, 25], [0.3, 0.15])
LTA_DOWN = ([5, 25], [0.36, 0.26])

DEFAULT = dict(delay=0.25, v_hold=MIN_LAT_CONTROL_SPEED, v_active=0.3, lta=False, sr=14.3, wb=2.68986)


def clip_curvature(v_ego, prev_curvature, new_curvature, roll=0.0):
    """VERBATIM drive_helpers.clip_curvature."""
    v_ego = max(v_ego, MIN_SPEED)
    max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)
    new_curvature = np.clip(new_curvature, prev_curvature - max_curvature_rate * DT_CTRL, prev_curvature + max_curvature_rate * DT_CTRL)
    roll_compensation = roll * G
    max_lat_accel = MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
    min_lat_accel = -MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
    new_curvature = float(np.clip(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2))
    return float(np.clip(new_curvature, -MAX_CURVATURE, MAX_CURVATURE))


def smooth_value(val, prev_val, tau, dt=0.05):
    """VERBATIM drive_helpers.smooth_value."""
    alpha = 1 - np.exp(-dt / tau) if tau > 0 else 1
    return alpha * val + (1 - alpha) * prev_val


def apply_std_steer_angle_limits(apply_angle, apply_angle_last, v_ego):
    """VERBATIM opendbc lateral.apply_std_steer_angle_limits (lat_active, STEER_ANGLE_MAX 94.9 deg never binds here) with Toyota LTA limits."""
    steer_up = apply_angle_last * apply_angle >= 0. and abs(apply_angle) > abs(apply_angle_last)
    rl = LTA_UP if steer_up else LTA_DOWN
    lim = float(np.interp(v_ego, rl[0], rl[1]))
    return float(np.clip(apply_angle, apply_angle_last - lim, apply_angle_last + lim))


class OpLateral:
    """State of modeld's action filter, controlsd's desired curvature and the car's delayed realised curvature (all openpilot sign:
    + = right, the HUGSIM sign too). step() takes the model's curvature for one simulator step and returns the mean realised curvature
    over the step (what a constant-speed bicycle needs to reproduce the step's heading change) plus a trace for logging."""

    def __init__(self, rule=None):
        self.r = dict(DEFAULT, **(rule or {}))
        self.n_delay = max(0, int(round(self.r["delay"] / DT_CTRL)))
        self.act = 0.0                       # modeld prev_action.desiredCurvature
        self.des = 0.0                       # controlsd desired_curvature
        self.real = 0.0                      # realised curvature now
        self.wheel = 0.0                     # LTA: steering-wheel angle (deg), left +
        self.buf = [0.0] * (self.n_delay + 1)

    def model_step(self, kappa_model, v_ego):
        """modeld.get_action_from_model, lateral half."""
        if v_ego > self.r["v_hold"]:
            self.act = float(smooth_value(kappa_model, self.act, LAT_SMOOTH_SECONDS))
        return self.act

    def step(self, kappa_model, v_ego, dt):
        act = self.model_step(kappa_model, v_ego)
        active = v_ego > self.r["v_active"]
        ks = []
        for _ in range(int(round(dt / DT_CTRL))):
            new = act if active else self.real                       # controlsd: inactive -> current curvature
            self.des = clip_curvature(v_ego, self.des, new)
            cmd = self.des if active else self.real
            if self.r["lta"]:                                        # angle car: wheel angle rate-limited, curvature = -angle / (sr * wb)
                want = -math.degrees(cmd * self.r["sr"] * self.r["wb"])
                self.wheel = apply_std_steer_angle_limits(want, self.wheel, v_ego) if active else self.wheel
                cmd = -math.radians(self.wheel) / (self.r["sr"] * self.r["wb"])
            self.buf.append(cmd)
            self.real = self.buf.pop(0)
            ks.append(self.real)
        return float(np.mean(ks)), dict(act=act, des=self.des, real=self.real, active=active)


def hugsim_steer(state, kappa_model, v_now, dt, wheelbase):
    """Front-wheel angle for HUGSIM's step (kinematic bicycle, theta += v_next tan(steer) / L dt): the angle whose curvature is the mean realised
    curvature of the emulated car over the step, so the simulator's heading change equals openpilot's (speed is the simulator's own)."""
    k, log = state.step(kappa_model, v_now, dt)
    return math.atan(k * wheelbase), dict(log, kmean=k)
