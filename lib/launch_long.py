"""Non-privileged longitudinal launch assist (agent side; lateral untouched, the model's inputs untouched).

Observation: HUGSIM launches spend a median 6.4 s of the first 10 s below 3 m/s because the model's own plan is slow, while real
cars leave the 0-3 m/s window in ~1.8 s (decision 115). The assist re-times the model's plan along its own path so that it covers
at least the distance of a launch profile (acceleration a, speed cap vcap) from the current speed; the path geometry is kept.
Inputs are the ego speed and the model's own outputs (plan, lead head); nothing from the scene, the route or the map.

Phase (latch): armed at start and whenever v < v_rearm (a stop); disarmed the first time v >= v_end, so a car that the model then
slows down is never pushed back up (no speed floor outside a launch).
Gate, per step, inside the phase: the plan is not a stop (arc length at 3 s >= dmin), it does not ask to slow down (1 s arc length
>= keep * v * 1 s), and the lead head reports no close lead (not (lead_prob > p_lead and 0 < lead_x < x_safe)).
Parameters (JSON, defaults = the pre-registered values, plans/2026-10-04-launch-long-prereg.md): a 1.2 m/s^2 (iLQR realises ~1.6-1.7
m/s^2 in the first second and 3.4 m/s at +2 s, matching comma1M's 1.5 / 3.4 m/s at +1 / +2 s, offline check in the report), vcap 4.0
(so that v crosses v_end = 3.5 instead of approaching it), x_safe 20 m (5.7 s at 3.5 m/s), p_lead 0.5, dmin 3 m, keep 0.7.
"""
import numpy as np

DEFAULT = dict(a=1.2, vcap=4.0, v_end=3.5, v_rearm=0.5, x_safe=20.0, p_lead=0.5, dmin=3.0, keep=0.7)
PLAN_DT = 0.5


def ref_dist(v, a, vcap, t):
    """Distance covered after t s by accelerating at a from v up to vcap, then holding vcap."""
    tc = max((vcap - v) / a, 0.0)
    return np.where(t <= tc, v * t + 0.5 * a * t * t, v * tc + 0.5 * a * tc * tc + vcap * (t - tc))


def arclen(plan):
    pts = np.vstack([[0.0, 0.0], plan])
    return np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))


def retime(plan, s_new):
    """Points of the plan's polyline (from the origin, extended straight beyond the last point) at arc lengths s_new."""
    pts = np.vstack([[0.0, 0.0], plan])
    seg = np.diff(pts, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    s = np.r_[0.0, np.cumsum(ln)]
    d = seg[-1] / ln[-1] if ln[-1] > 1e-6 else np.array([0.0, 1.0])
    out = np.empty((len(s_new), 2))
    for i, q in enumerate(s_new):
        out[i] = pts[-1] + (q - s[-1]) * d if q >= s[-1] else np.array([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])])
    return out


class LaunchAssist:
    def __init__(self, **kw):
        self.p = dict(DEFAULT, **kw)
        self.armed = True

    def step(self, v, plan, lead_prob=None, lead_x=None, stop=False):
        """-> (plan, why). why is "on" when the plan was re-timed, else the reason it was not (disarmed / stop / slowing / lead / done / fast)."""
        p = self.p
        if v < p["v_rearm"]:
            self.armed = True
        if v >= p["v_end"]:
            self.armed = False
        if not self.armed:
            return plan, "disarmed"
        s = arclen(plan)
        if stop or s[-1] < p["dmin"]:
            return plan, "stop"
        if s[1] < p["keep"] * v * 1.0:
            return plan, "slowing"
        if lead_prob is not None and lead_x is not None and lead_prob > p["p_lead"] and 0 < lead_x < p["x_safe"]:
            return plan, "lead"
        t = PLAN_DT * np.arange(1, len(plan) + 1)
        s_new = np.maximum(s, ref_dist(v, p["a"], p["vcap"], t))
        if np.all(s_new <= s + 1e-6):
            return plan, "model_faster"
        return retime(np.asarray(plan, float), s_new), "on"
