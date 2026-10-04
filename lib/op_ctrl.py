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
        self.buf = [0.0] * self.n_delay

    def sync(self, kappa):
        """The path is not the lateral owner (controlsd inactive: desired curvature := measured curvature): follow the measured curvature, so the
        hand-over back to this path starts from where the wheel is, not from a stale command or a stale delay line."""
        self.act = self.des = self.real = float(kappa)
        self.buf = [float(kappa)] * self.n_delay

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


# ---------------------------------------------------------------------------------------------------------------------------------------
# Longitudinal path (plans/2026-10-04-op-control-stack-long-prereg.md). Source (read, not recalled): openpilot master ec95db3f, opendbc 35f7e081.
#   modeld.get_action_from_model   desired_accel = action[1]; stop = should_stop(v_ego, desired_accel) = v_ego < 0.3 and a < 0.1 on the RAW action;
#                                  desiredAcceleration = smooth_value(raw, prev, LONG_SMOOTH_SECONDS = 0.3) at 20 Hz (DT_MDL 0.05)
#   longitudinal_planner (experimental mode, the mode the e2e model drives in): a_target = min(mpc, cruise, e2e); with no lead and a set speed above
#                                  the car's, mpc and cruise are positive, so the e2e accel wins: output_a_target = clip(e2e, ACCEL_MIN, ACCEL_MAX),
#                                  output_should_stop = any(candidate stop) = the e2e stop (mpc and cruise never stop here). No radar in the simulator.
#   controlsd -> LongControl       longcontrol.long_control_state_trans / LongControl.update (Toyota TSS2: kpV 0, kiV 0 from interfaces.py, so the pid
#                                  state outputs a_target as pure feed-forward); stopping state: output = last output, ramped to stopAccel -2.0 at
#                                  1 m/s^2 per second; pid -> stopping when should_stop, stopping -> pid when not should_stop (cruise standstill and
#                                  brake not modelled). Accel limits = Toyota CarControllerParams (ACCEL_MIN -3.5, RAISED_ACCEL_LIMIT -> ACCEL_MAX 2.0).
#   Toyota carcontroller           the request is rate limited to +-4.0 m/s^2 per 3 control frames (jerk 4 m/s^3) before the PCM loop
#                                  (ACCEL_WINDUP_LIMIT / ACCEL_WINDDOWN_LIMIT); the PCM + long_pid loop makes a_ego follow it, modelled as the same
#                                  pure delay openpilot gives the model, CP.longitudinalActuatorDelay 0.15 s (non-hybrid; hybrids 0.05).
# Units. The model runs on a clock `dil` times faster than the simulator (speed fed as dil * v, one 0.25 s step = 4 model steps = 0.2 model s), so the
# controller works in the model's frame (v_m = dil * v, a_m = dil^2 * a_sim, 100 Hz ticks of 0.01 model s) and the result is converted back:
# a_sim = mean realised a_m over the step / dil^2 (= the velocity change HUGSIM's `velo += acc * dt` needs).
# Rule parameters (dict, JSON): {"dil": 1.25, "delay": 0.15, "tau": 0.3, "jerk": 4.0, "stop_accel": -2.0, "a_min": -3.5, "a_max": 2.0}.
# Hook: hugsim_acc (HUGSIM tree `opctrl_long`, env OP_CTRL_LONG).
V_STOP, A_STOP = 0.3, 0.1            # drive_helpers.should_stop
DT_MDL = 0.05
LONG_SMOOTH_SECONDS = 0.3
LONG_DEFAULT = dict(dil=1.25, delay=0.15, tau=LONG_SMOOTH_SECONDS, jerk=4.0, stop_accel=-2.0, a_min=-3.5, a_max=2.0)
OFF, STOPPING, PID = "off", "stopping", "pid"


def should_stop(v_ego, a_target):
    """VERBATIM drive_helpers.should_stop."""
    return bool(v_ego < V_STOP and a_target < A_STOP)


def long_control_state_trans(active, state, stop, brake_pressed=False, cruise_standstill=False):
    """VERBATIM longcontrol.long_control_state_trans (states as strings)."""
    starting_condition = not stop and not cruise_standstill and not brake_pressed
    if not active:
        state = OFF
    else:
        if state == OFF:
            state = PID if starting_condition else STOPPING
        elif state == STOPPING:
            if starting_condition:
                state = PID
        elif state == PID:
            if stop:
                state = STOPPING
    return state


class OpLongitudinal:
    """modeld's accel filter + longitudinal planner (e2e candidate) + LongControl + the Toyota request rate limit + the actuator delay.
    step() takes the model's raw action acceleration (model units), the car speed (simulator units) and the simulator step, and returns the mean
    realised acceleration over the step in simulator units plus a trace."""

    def __init__(self, rule=None):
        self.r = dict(LONG_DEFAULT, **(rule or {}))
        self.n_delay = max(0, int(round(self.r["delay"] / DT_CTRL)))
        self.act = 0.0                       # modeld prev_action.desiredAcceleration
        self.state = OFF
        self.last_out = 0.0                  # LongControl.last_output_accel
        self.prev_cmd = 0.0                  # carcontroller prev_accel
        self.buf = [0.0] * self.n_delay
        self.real = 0.0

    def step(self, accel_raw, v_sim, dt_sim):
        r, d = self.r, self.r["dil"]
        v_m, dt_m = d * v_sim, dt_sim / d
        stop = False
        for _ in range(max(1, int(round(dt_m / DT_MDL)))):          # modeld at 20 Hz; the model's last output stands for the step
            stop = should_stop(v_m, accel_raw)
            self.act = float(smooth_value(accel_raw, self.act, r["tau"], DT_MDL))
        a_target = float(np.clip(self.act, r["a_min"], r["a_max"]))  # planner: output_a_target (e2e candidate), shouldStop = e2e stop
        out, jl = [], r["jerk"] * DT_CTRL
        for _ in range(int(round(dt_m / DT_CTRL))):
            self.state = long_control_state_trans(True, self.state, stop)
            if self.state == STOPPING:
                o = self.last_out
                if o > r["stop_accel"]:
                    o = min(o, 0.0) - 1.0 * DT_CTRL
            else:
                o = a_target                                         # kp = ki = 0: feed-forward only
            o = float(np.clip(o, r["a_min"], r["a_max"]))
            self.last_out = o
            cmd = float(np.clip(o, self.prev_cmd - jl, self.prev_cmd + jl))   # toyota carcontroller rate_limit
            self.prev_cmd = cmd
            self.buf.append(cmd)
            self.real = self.buf.pop(0)
            out.append(self.real)
        a = float(np.mean(out)) / d ** 2
        return a, dict(a_raw=accel_raw, a_target=a_target, state=self.state, stop=stop, a_sim=a)


def hugsim_acc(state, accel_raw, v_now, dt):
    """HUGSIM longitudinal command: `velo += acc * dt` with velo >= 0 (the emulated car does not reverse)."""
    a, log = state.step(accel_raw, v_now, dt)
    return max(a, -v_now / dt), log
