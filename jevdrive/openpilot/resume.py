"""Emulated driver resume from a held standstill: one rule shared by every closed-loop board (interface key `lon.resume` = "rule").

Why: on many cars openpilot's standstill exit needs the driver to press resume or tap the gas, so the model never had to launch
from a held standstill alone. In closed loop (HUGSIM `spec`, decisions 118 / 119) Cinque then stays on a stop plan for the rest of
the episode. The rule plays that driver with what a driver behind a comma device can be emulated with without privileged state:
the ego speed and the model's own lead head. No light, route, map or actor box is read.
Pre-registration: experiments/op_resume/plans/2026-10-05-op-resume-prereg.md (thresholds and their reasons).

States: idle -> launch -> cooldown -> idle.
  idle      standstill (v < v_stand) accumulates; any v >= v_stand resets it. Trigger when the standstill has lasted t_stand s
            AND the lead head has been clear for the last t_clear s. Clear = not (lead_prob > p_lead and lead_x < d_lead).
  launch    the board drives a launch profile: from the speed at the step, accelerate at `a` up to v_end, never slower than the
            model's own plan (`profile` gives the distance at times t). Ends (hand-back) when v >= v_end ("speed"), the
            model's own plan asks at least as much as the profile at 1 s ("plan"), the lead head stops being clear ("lead"),
            or after t_max s ("timeout").
  cooldown  t_cool s after a hand-back; the standstill clock only counts again after it.
Stdlib only and Python 3.8 (the CARLA route process imports it).
"""
import math

# defaults = the pre-registered values (prereg section 2)
DEFAULT = dict(v_stand=0.1, t_stand=10.0, p_lead=0.5, d_lead=15.0, t_clear=1.0, a=1.2, v_end=2.5, t_max=6.0, t_cool=10.0)
IDLE, LAUNCH, COOL = "idle", "launch", "cooldown"


def launch_dist(v0, a, v_end, t):
    """Distance after t s accelerating at a from v0 up to v_end, then holding v_end (v0 >= v_end: constant v0)."""
    if v0 >= v_end:
        return v0 * t
    tc = (v_end - v0) / a
    if t <= tc:
        return v0 * t + 0.5 * a * t * t
    return v0 * tc + 0.5 * a * tc * tc + v_end * (t - tc)


class ResumeRule:
    """step(t, v, lead_prob, lead_x, plan_s1) once per control step -> (active, why). active True = drive `profile(v, times)`.

    t        board clock (s, monotonic)            v         ego speed (m/s)
    lead_prob, lead_x   the model's lead head now (probability, distance ahead of the device in m); None = no lead output
    plan_s1  distance the model's own plan covers in its first 1 s (m), in the board's frame (after the board's own plan
             post-processing); None = unknown (then the plan never takes over)
    why      the step's state reason: "stand", "lead", "fire", "launch", "speed" / "plan" / "lead_abort" / "timeout" (hand-back),
             "cool", "moving"
    """

    def __init__(self, **kw):
        unknown = set(kw) - set(DEFAULT)
        if unknown:
            raise ValueError("unknown resume rule keys %s" % sorted(unknown))
        self.p = dict(DEFAULT, **kw)
        self.state = IDLE
        self.stand_since = None          # start of the counted standstill
        self.blocked_t = -1e9            # last time the lead head was not clear
        self.t0 = self.v0 = 0.0          # launch start
        self.cool_until = -1e9
        self.n = 0                       # launches so far
        self.events = []                 # (t, event, detail) for the log

    def clear(self, lead_prob, lead_x):
        p = self.p
        if lead_prob is None or lead_x is None:
            return True
        return not (float(lead_prob) > p["p_lead"] and float(lead_x) < p["d_lead"])

    def profile(self, v, times):
        """Launch distances at `times` (s) from speed v (the board takes max with the model's own plan)."""
        return [launch_dist(max(v, 0.0), self.p["a"], self.p["v_end"], float(t)) for t in times]

    def step(self, t, v, lead_prob=None, lead_x=None, plan_s1=None):
        p = self.p
        ok = self.clear(lead_prob, lead_x)
        if not ok:
            self.blocked_t = t
        if self.state == LAUNCH:
            why = None
            if v >= p["v_end"]:
                why = "speed"
            elif not ok:
                why = "lead_abort"
            elif plan_s1 is not None and plan_s1 >= launch_dist(max(v, 0.0), p["a"], p["v_end"], 1.0) - 1e-6 and t > self.t0:
                why = "plan"
            elif t - self.t0 >= p["t_max"]:
                why = "timeout"
            if why is None:
                return True, "launch"
            self.state, self.cool_until, self.stand_since = COOL, t + p["t_cool"], None
            self.events.append((t, "handback", why))
            return False, why
        if self.state == COOL:
            if t < self.cool_until:
                return False, "cool"
            self.state = IDLE
        if v >= p["v_stand"]:
            self.stand_since = None
            return False, "moving"
        if self.stand_since is None:
            self.stand_since = max(t, self.cool_until)
        if t - self.stand_since < p["t_stand"] - 1e-6:
            return False, "stand"
        if t - self.blocked_t < p["t_clear"] - 1e-6:
            return False, "lead"
        self.state, self.t0, self.v0 = LAUNCH, t, v
        self.n += 1
        self.events.append((t, "fire", round(t - self.stand_since, 2)))
        return True, "fire"

    def describe(self):
        return dict(self.p, rule="jevdrive.openpilot.resume.ResumeRule")


def plan_s1_from_points(points, dt):
    """Distance of a plan polyline (from the origin, points every dt s) at 1 s: arc length, linearly interpolated."""
    s, prev, out = 0.0, (0.0, 0.0), [0.0]
    for x, y in points:
        s += math.hypot(x - prev[0], y - prev[1])
        out.append(s)
        prev = (x, y)
    k = 1.0 / dt
    i = int(math.floor(k))
    if i + 1 >= len(out):
        return out[-1]
    return out[i] + (k - i) * (out[i + 1] - out[i])
