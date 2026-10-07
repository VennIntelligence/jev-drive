"""Lead standstill margin: one execution-layer rule on the plan's speed profile, shared by the HUGSIM client and the NAVSIM export
(interface trick `lead_margin`; experiments/op_parity/plans/2026-10-07-lead-margin-prereg.md, values fixed there, not tuned).

Why: on HUGSIM P2H's lead head reads a close lead ~2 m too far (lead_x - true gap = +1.95 m when the gap is < 3 m,
experiments/op_parity/results/four_dirs/hugsim.md), so the plan stops against a lead it thinks is further away and runs into stopped /
slower leads. The rule moves the stop point back with the measured near-range bias and a fixed standstill margin.
Inputs: the plan (positions from the device / camera origin and their times, in the board's real seconds) and the model's own
lead head (probability, distance ahead of the device, speed). Nothing from the scene, the route or the actor boxes.

  trigger  lead_prob > p_lead
  bias     b(lead_x) = b_near for lead_x < x_near, linear to 0 at x_far, 0 beyond (hugsim_lead_calib.csv)
  gap      g = lead_x - b(lead_x)
  cap      s_max(t) = max(0, g + max(lead_v, 0) t - d_min)
  plan     s'(t) = min(s(t), s_max(t)): from the first plan time whose cumulative arc length exceeds the cap, the poses are pulled back
           along the plan's own path to the cap (path shape kept, only the speed profile changes); s' never exceeds s, so the rule only
           adds caution and never starts a car. Untriggered plans are returned unchanged.
"""
import numpy as np

DEFAULT = dict(p_lead=0.5, d_min=2.5, b_near=2.0, x_near=6.0, x_far=10.0)


def params(p=None):
    p = {} if p is None or p is True else dict(p)
    unknown = set(p) - set(DEFAULT)
    if unknown:
        raise ValueError("unknown lead_margin keys %s" % sorted(unknown))
    return dict(DEFAULT, **p)


def bias(lead_x, p=DEFAULT):
    """Near-range distance bias of the lead head (m): b_near below x_near, linear to 0 at x_far."""
    return float(np.interp(lead_x, [p["x_near"], p["x_far"]], [p["b_near"], 0.0]))


def cap(t, lead_prob, lead_x, lead_v, p=DEFAULT):
    """Allowed arc length s_max at times t (s), or None when the rule does not trigger."""
    if lead_prob is None or lead_x is None or not float(lead_prob) > p["p_lead"]:
        return None
    g = float(lead_x) - bias(float(lead_x), p)
    v = max(float(lead_v or 0.0), 0.0)
    return np.maximum(0.0, g + v * np.asarray(t, np.float64) - p["d_min"])


def arclen(xy):
    """Cumulative arc length of a polyline (n, 2) from its first point."""
    xy = np.asarray(xy, np.float64)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def at_arc(s, q, *arrays):
    """Values of per-point arrays (n, ...) at arc lengths q along the polyline with cumulative arc s (linear between points;
    zero-length segments take the earlier point)."""
    q = np.asarray(q, np.float64)
    i = np.clip(np.searchsorted(s, q, side="right") - 1, 0, len(s) - 2)
    ds = s[i + 1] - s[i]
    f = np.where(ds > 1e-9, (q - s[i]) / np.where(ds > 1e-9, ds, 1.0), 0.0)
    f = np.clip(f, 0.0, 1.0)
    out = []
    for a in arrays:
        a = np.asarray(a, np.float64)
        ff = f.reshape((-1,) + (1,) * (a.ndim - 1))
        out.append(a[i] + ff * (a[i + 1] - a[i]))
    return out


def apply(xy, t, lead_prob, lead_x, lead_v, extra=(), p=None):
    """xy (n, 2) plan positions starting at the device position at t[0] (include the t = 0 point), t (n,) board seconds.
    extra: per-point arrays (yaw, z, ...) re-sampled at the same path positions. Returns (xy', [extra'], info);
    info = dict(trig, changed, ds_max (m pulled back at most), g (corrected gap), s3 (plan arc at the last point))."""
    p = params(p)
    xy = np.asarray(xy, np.float64)
    s = arclen(xy)
    sm = cap(t, lead_prob, lead_x, lead_v, p)
    info = dict(trig=sm is not None, changed=False, ds_max=0.0, s_end=round(float(s[-1]), 3))
    if sm is None:
        return xy, [np.asarray(a) for a in extra], info
    info["g"] = round(float(lead_x) - bias(float(lead_x), p), 3)
    q = np.minimum(s, sm)
    ds = float(np.max(s - q))
    if ds <= 1e-6:
        return xy, [np.asarray(a) for a in extra], info
    out = at_arc(s, q, xy, *extra)
    info.update(changed=True, ds_max=round(ds, 3), s_end_new=round(float(q[-1]), 3))
    return out[0], out[1:], info


class LeadMargin:
    """Configured rule: rule(xy, t, lead_prob, lead_x, lead_v, extra=()) -> (xy', extra', info)."""

    def __init__(self, cfg=None):
        self.p = params(cfg)

    def __call__(self, xy, t, lead_prob, lead_x, lead_v, extra=()):
        return apply(xy, t, lead_prob, lead_x, lead_v, extra, self.p)

    def describe(self):
        return dict(self.p, rule="jevdrive.openpilot.lead_margin")
