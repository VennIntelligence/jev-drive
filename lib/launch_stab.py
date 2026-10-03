"""Launch stabilisation (experiments/hugsim/plans/2026-10-04-launch-stab-prereg.md): an inference-time rule against the
closed-loop spin at launch (decisions 96, 100, 109). Shared by the HUGSIM agent (experiments/hugsim/archive/zs_agent.py) and
the B2D openpilot server (experiments/op_closed_loop/archive/op_arb_server.py).

After a standstill the model's response to the first ~1 deg of its own yaw is near a step (5 deg / deg), the loop grows x1.6-1.8
per step and the car spins. The rule removes the self-induced yaw from what the model sees during the launch only:

  gate     armed by a standstill or creep (v < v_stop = 1 m/s for >= t_stop s; the episode start counts); the launch phase starts
           at the first step with v >= v_stop and ends after t_launch s, when the heading has moved >= d_max deg from the launch heading
           (a real turn, or the field of view would run out), on a non-zero desire (a route turn / lane change), or on a new
           standstill of >= t_stop s. Inputs: own speed, own heading (odometry), the route desire, the clock. Nothing else.
  model    a second session forked from the native one at the last standstill step (exact state copy) is stepped on the
           frames re-rendered (rotation only, exact for any depth) at the launch heading: a camera that translates with the
           car but does not turn with it, so the model never sees its own yaw. The native session keeps stepping on the
           real frames and drives again when the phase ends.
  output   during the phase the lateral part of the plan comes from the stabilised session, rotated from the launch-heading
           frame into the car frame (so a drift is steered back, not ignored); longitudinal (x, speed, accel) stays native.

Frames: openpilot device / calib frame, x forward, y right; headings right-positive (HUGSIM ego_pose2d theta, CARLA yaw).
"""
import math

import numpy as np


class LaunchGate:
    """Per-episode state machine. step() once per planning step, before the model call; returns one of
    'off' (native only), 'fork' (copy native -> stab, then step both), 'on' (step both, use stab lateral).
    'Slow' is v < v_stop; slow for >= t_stop s arms the gate (and ends a phase in progress); the first step at v >= v_stop
    while armed and without a desire forks. A shorter slow-down inside a phase does not end it."""

    def __init__(self, v_stop=1.0, t_stop=1.0, t_launch=8.0, d_max=10.0, start_armed=True):
        self.v_stop, self.t_stop, self.t_launch, self.d_max = v_stop, t_stop, t_launch, math.radians(d_max)
        self.still_since = None
        self.armed = start_armed
        self.t0 = self.ref = None                          # launch start time, launch heading (rad)
        self.last_th, self.why = None, "init"

    def step(self, t, v, th, desire=0):
        prev_th, self.last_th = self.last_th, th
        if v < self.v_stop:
            self.still_since = t if self.still_since is None else self.still_since
            if t - self.still_since >= self.t_stop - 1e-6:
                self.armed = True
                if self.t0 is not None:
                    self.t0 = self.ref = None
                    self.why = "still"
        else:
            self.still_since = None
        if self.t0 is None:
            if v < self.v_stop or not self.armed:
                self.why = "slow" if v < self.v_stop else "moving"
                return "off"
            self.armed = False
            if desire:                                     # moving off into a route turn / lane change: no phase
                self.why = "desire"
                return "off"
            self.t0 = t
            self.ref = prev_th if prev_th is not None else th   # the heading of the step the state is forked from
            self.why = "launch"
            return "fork"
        d = wrap(th - self.ref)
        end = "time" if t - self.t0 >= self.t_launch - 1e-6 else "turn" if abs(d) >= self.d_max else "desire" if desire else None
        if end:
            self.t0 = self.ref = None
            self.why = end
            return "off"
        self.why = "on"
        return "on"

    def delta(self, th):
        """Heading now minus launch heading (rad, right-positive)."""
        return wrap(th - self.ref)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_car(pos, yaw, delta):
    """A plan in the launch-heading frame (pos (N, >=2) x fwd / y right, yaw (N,) rad right-positive) -> the car frame of a car
    whose heading is delta (rad, right-positive) past the launch heading."""
    c, s = math.cos(delta), math.sin(delta)
    p = np.array(pos, np.float32, copy=True)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0], p[:, 1] = x * c + y * s, y * c - x * s
    return p, None if yaw is None else np.asarray(yaw, np.float32) - np.float32(delta)


def merge_lateral(native_pos, stab_pos_car):
    """Lateral (y) from the stabilised plan in the car frame, longitudinal (x, z) native."""
    p = np.array(native_pos, np.float32, copy=True)
    p[:, 1] = stab_pos_car[:, 1]
    return p


def fork_state(dst, src):
    """Exact copy of an openpilot session's recurrent state (jevdrive.openpilot.model.OPModel): dst continues as if it had
    seen everything src has. Device sessions alternate two state buffer sets by step parity; host sessions keep dicts."""
    dst.prev_desire = np.array(src.prev_desire, copy=True)
    dst.extra = dict(src.extra)
    if not src.queued:
        for k in ("img_q", "desire_q", "feat_q", "prev_feat"):
            setattr(dst, k, np.array(getattr(src, k), copy=True))
    if src.device:
        ks, kd = src.n % len(src.bindings), dst.n % len(dst.bindings)
        for n in src.state_names:
            dst.sets[kd][n].update_inplace(np.ascontiguousarray(src.sets[ks][n].numpy()))
    else:
        dst.state = {n: np.array(v, copy=True) for n, v in src.state.items()}
