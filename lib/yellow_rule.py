"""Yellow-light go / stop decision of arm `vred3` (plan experiments/vlm_arb/plans/2026-10-02-pbyp2-vred2.md, section 11).

Pure functions, no CARLA, Python 3.8 compatible: used by lib/vlm_arb_agent.py and unit-tested on a grid in tests/test_yellow_rule.py.

Geometry of the scorer (scenario_runner RunningRedLightTest, read on the box): the light's line is the lane's last waypoint before the
intersection, i.e. the junction entrance; the test fires while the light is Red and the segment from 0.8 x half-length to
(half-length + 1 m) behind the vehicle centre crosses that line. The segment is 1.96 m to 3.45 m behind the centre, so a car whose front
crossed the line on green or yellow is penalised when the light turns red before the tail has passed: the car has cleared the line when the
front bumper is `TAIL_M` = 2.45 + 3.45 = 5.9 m beyond the entrance line.
"""
import math

HALF_LEN = 2.4508                    # MKZ 2020 bounding box half length
TAIL_M = HALF_LEN + (HALF_LEN + 1.0)  # front bumper beyond the line at which tail_far (centre - half length - 1 m) has crossed it

# Registered before the vred3 batch (plan section 11); `yellow_s` measured from the logged light states of these routes
PARAMS = dict(yellow_s=3.0,         # CARLA yellow duration on these routes (every complete yellow in the logs: 3.0 s)
              lag_margin_s=0.5,     # subtracted from the remaining yellow: the model may keep answering green into the yellow (offline, section 11)
              a_comfort=3.0,        # m/s^2, comfortable deceleration of the stop option
              tau_s=0.5,            # s, reaction delay of the stop option (answer already aged inside `age`; plan to actuation)
              tail_margin_m=0.5,    # m, extra clearance beyond TAIL_M for the go option
              a_go=1.0,             # m/s^2, mild acceleration of the go option, up to the cruise speed
              t_go_s=0.3,           # s, planner latency before the go option accelerates
              max_age_s=2.0)        # s, a last green answer older than this says nothing about when the yellow started


def remaining_yellow(age_s, p=PARAMS):
    """Conservative remaining yellow time: the yellow may have started right after the last green frame (age = now - that frame's time),
    the model may lag by `lag_margin_s`."""
    if age_s is None or age_s > p["max_age_s"]:
        return -1.0
    return p["yellow_s"] - p["lag_margin_s"] - age_s


def stop_distance(v, p=PARAMS):
    """Distance covered from the decision to standstill: reaction delay at the current speed, then the comfortable deceleration."""
    return v * p["tau_s"] + v * v / (2.0 * p["a_comfort"])


def clear_time(dist, v, v_cruise, p=PARAMS):
    """Time to cover `dist` from speed v: planner latency at v, then a_go up to v_cruise, then v_cruise."""
    if dist <= 0:
        return 0.0
    t0 = p["t_go_s"]
    d0 = v * t0
    if dist <= d0:
        return dist / max(v, 1e-6)
    rest = dist - d0
    a = p["a_go"]
    t_acc = max(v_cruise - v, 0.0) / a
    d_acc = v * t_acc + 0.5 * a * t_acc * t_acc
    if rest <= d_acc:
        # v t + a t^2 / 2 = rest
        return t0 + (-v + math.sqrt(v * v + 2.0 * a * rest)) / a
    return t0 + t_acc + (rest - d_acc) / max(v_cruise, 1e-6)


def decide(d_stop, d_line, v, remaining, v_cruise, p=PARAMS):
    """Decision at the first non-green answer after green.

    d_stop     front bumper to the light's stop line, minus the 0.5 m margin of R2 (the stop target), m
    d_line     front bumper to the scorer's line (junction entrance), m
    v          speed, m/s
    remaining  conservative remaining yellow time (`remaining_yellow`), s; <= 0: treat as red
    Returns "stop" (comfortable stop possible), "go" (the whole car clears the line before red), or "stop_hard" (neither: stop anyway)."""
    if d_stop >= stop_distance(v, p):
        return "stop"
    if remaining > 0 and clear_time(d_line + TAIL_M + p["tail_margin_m"], v, v_cruise, p) <= remaining:
        return "go"
    return "stop_hard"
