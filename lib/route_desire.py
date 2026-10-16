"""Route-derived lane-change desire (experiments/desire_cap, decision 332; plans/2026-10-16-desire-route-design.md with the 2026-10-17 fixes):
openpilot's DesireHelper with the blinker replaced by the route's lane-change intent. One module for the served drivers (pai_driver /
sh30_driver, JEV_DESIRE) and the offline training rows (D1 / D2), so a desire appears at the same moments in training and serving.

Input: the route-memory features of a decision (experiments/route_mem/lib/rmem.py, either layout: DIM = 200 projected or DIM_O = 203 ordered;
only the 40 x 5 point block is read): points at route arc offsets -6, -4, .., +72 m, [valid, x / 20, y / 20, cos, sin] in the rear-axle
frame (x forward, y left). No map, no lane ids: the route is the only lane information the driver has.

lc_intent / intent (stateless, numpy, vectorised over rows)
  road dir  hn = the median route heading over -6 .. +20 m (a lane-change slant is a minority of those slots); every angle below is taken
            against the route, never the ego's yaw (at 40 m the ego's heading error reads 0.7 m per degree)
  in_turn   the route's heading spans >= TURN_IN (30) deg within -6 .. +10 m: the ego is inside a junction turn -> blocked
  d_turn    the first arc in 0 .. +72 m where the route departs >= TURN_DEG (30) deg from hn and stays (next slot too): the junction entry
            (rmem.turn_dist's rule against hn; inf when none within the valid memory)
  judge arc a = the target lane is judged just before the junction entry: a = the last slot <= min(d_turn - PRE_M, LOOK_FAR) still within
            STRAIGHT_DEG of hn, when a turn is ahead (turn-lane entry happens 30-100 m before the junction, so the design's "no lane change
            within 60 m of a turn" rule is gone); a = LOOK (40 m) without a turn ahead, with the route settled there (the +a + 20 m point within
            SETTLE_M of a across the a .. a + 20 m chord, when that slot is valid)
  gap       memory that starts ahead (the first route message lies ~42 m out; held reads of 18 decisions never see +0 m): the first valid
            slot (<= GAP_MAX) replaces +0 m, hn is taken over its next 20 m, and the ego's own heading must be within GAP_YAW of hn (the
            road is taken as straight through the gap); in_turn is not judged there
  blocked   d_turn <= BLOCK_M (20 m: the stop line / junction mouth; no new lane change starts there) or in_turn or a < A_MIN
  offset    y_t = the route point at a relative to the ego, across the chord direction href = the median route heading over 0 .. a (a
            constant-curvature road reads ~0, a lane change its lane offset); every slot 0 .. a valid, none > LC_MAX_DEG off href
  side      sign(y_t) when |y_t| >= ON_M (half a lane), else 0; +1 = left (laneChangeLeft, log.Desire 3), -1 = right (laneChangeRight, 4)

RouteDesire (stateful, one per session): openpilot desire_helper.py's lane-change states, the blinker = lc_intent
  off -> starting   side != 0, not blocked, v >= v_min, no cooldown: rising edge, t_on = t (modeld feeds the policy only this edge)
  starting -> finishing   the remaining offset toward the side |y_t| < ON_M (the ego crossed into the route's lane) or, when given, the
                          model's own desire_state for the lane change < 0.01
  finishing -> off        remaining < OFF_M (done)
  abort -> off            in_turn, timeout T_MAX (10 s, LANE_CHANGE_TIME_MAX), v < v_min - 1, the route flips side; then COOL s before a
                          new start (openpilot needs a new blinker press; here the route re-arms after the cooldown)
  v_min     the speed gate (openpilot LANE_CHANGE_SPEED_MIN = 8.9 m/s, 20 mph; below it openpilot sends turn desires, which decision 332
            shows are not a junction-turn pathway): a parameter, chosen by the low-speed read (plans/2026-10-17-desire-d0-prereg.md)

desire_queue(d, t_on, t): the policy's desire input `_to_copy` (1, 33, 8) as modeld builds it: 20 Hz queue of 132 steps, max over 4 ->
  33 blocks of 0.2 s, block 32 = the newest; a pulse of age A s sits in block 32 - floor(A / 0.2) while A < 6.6 s. Zeros when d = 0 or the
  state is off (a cleared state clears the queue: unlike modeld, the route can take a desire back). dc.py's open-loop ages 0 / 0.4 / 1 / 2 s
  are the same blocks.
"""
from __future__ import annotations

import warnings

import numpy as np

N_PT, NF = 40, 5
OFFS = -6.0 + 2.0 * np.arange(N_PT)
LANE_CHANGE_SPEED_MIN = 8.9           # openpilot DesireHelper (20 mph)
V_MIN = LANE_CHANGE_SPEED_MIN         # default speed gate; the low-speed read decides the served value
T_MAX = 10.0                          # LANE_CHANGE_TIME_MAX
COOL = 2.0                            # s after an abort / timeout before the route may start a new lane change
ON_M, OFF_M = 1.75, 0.5               # start at half a lane of route offset; done within 0.5 m
LOOK, LOOK_FAR, PRE_M, A_MIN = 40.0, 60.0, 4.0, 12.0
SETTLE_M, STRAIGHT_DEG = 1.0, 10.0
TURN_DEG, TURN_IN, BLOCK_M = 30.0, 30.0, 20.0
GAP_MAX, GAP_YAW = 44.0, 5.0              # memory starting ahead: first valid point within 44 m, ego within 5 deg of the road
LC_MAX_DEG = 25.0                     # a lane-change route segment may slant this much (rf_dc P_lca: 0-60 m < 25 deg)
DESIRE = {1: 3, -1: 4}                # side -> log.Desire (laneChangeLeft 3, laneChangeRight 4)
QUEUE_S = 6.6
WHY = ("ok", "small", "no_route", "in_turn", "near_junction", "curved", "unsettled", "short")


def _slot(a):
    return np.clip(np.floor((np.asarray(a, float) + 6.0) / 2.0 + 1e-9).astype(int), 0, N_PT - 1)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _cmed(h, m):
    """Median direction of headings h (n, k) over mask m (n, k), relative to each row's first masked heading (0 where none)."""
    k0 = np.argmax(m, 1)
    base = h[np.arange(len(h)), k0]
    d = np.where(m, _wrap(h - base[:, None]), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        md = np.nanmedian(np.where(m.any(1)[:, None], d, 0.0), 1)
    return _wrap(base + np.nan_to_num(md))


def intent(F) -> dict:
    """Route features (n, D) or (D,) of either rmem layout -> dict of (n,) arrays: side (-1 / 0 / +1), y_t (signed target offset, m, left +),
    y_0 (the route's offset from the ego where the memory starts: ~0 = a lane change ahead on the route, ~y_t = the ego is off the route's
    lane already), a (judge arc, m), d_turn (m, inf = none ahead), blocked, in_turn, gap (bool), why (index into WHY; 0 = a lane change)."""
    F = np.asarray(F, np.float32)
    one = F.ndim == 1
    F = F.reshape(-1, F.shape[-1])
    P = F[:, :N_PT * NF].reshape(-1, N_PT, NF).astype(np.float64)
    n, R, rng = len(P), np.arange(len(P)), np.arange(N_PT)[None]
    v = P[..., 0] > 0.5
    x, y, h = 20.0 * P[..., 1], 20.0 * P[..., 2], np.arctan2(P[..., 4], P[..., 3])
    i0 = int(_slot(0.0))
    first = np.argmax(v & (rng >= i0), 1)                                  # the first valid slot at or after +0 m
    near_ok = v[:, i0]
    gap = ~near_ok & v[R, first] & (OFFS[first] <= GAP_MAX)                # memory starts ahead (first decisions; held 18 at all)
    s0 = np.where(near_ok, i0, first)
    win = v & (OFFS[None] <= OFFS[s0][:, None] + 20.0) & ((rng >= s0[:, None]) | near_ok[:, None])
    hn = _cmed(h, win)                                                     # the road's direction (a lane-change slant is a minority)
    gap &= np.degrees(np.abs(hn)) <= GAP_YAW                               # without the near route: the ego must run along the road
    dn = np.degrees(np.abs(_wrap(h - hn[:, None])))
    hh = np.where(v & (OFFS[None] <= 10.0) & near_ok[:, None], _wrap(h - hn[:, None]), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        in_turn = np.nan_to_num(np.nanmax(hh, 1) - np.nanmin(hh, 1)) >= np.radians(TURN_IN)
    tc = v & (dn >= TURN_DEG) & (rng >= s0[:, None])
    tc[:, :-1] &= tc[:, 1:] | ~v[:, 1:]                                    # persistent over the next slot (a merge kink is one segment)
    d_turn = np.where(tc.any(1), OFFS[np.argmax(tc, 1)], np.inf)
    turn = np.isfinite(d_turn)
    top = np.where(turn, np.minimum(_slot(np.where(turn, d_turn, 0.0) - PRE_M), _slot(LOOK_FAR)), np.maximum(_slot(LOOK), s0))
    flat = v & (dn <= STRAIGHT_DEG) & (rng >= s0[:, None]) & (rng <= top[:, None])
    ia = np.where(turn, np.where(flat.any(1), N_PT - 1 - np.argmax(flat[:, ::-1], 1), s0), top)   # just before the junction / LOOK
    a = OFFS[ia]
    span = (rng >= s0[:, None]) & (rng <= ia[:, None])
    ok_route = (near_ok | gap) & ~(span & ~v).any(1)
    href = _cmed(h, span & v)                                              # chord direction of 0 .. a: a constant curve reads 0 offset
    c, s = np.cos(href), np.sin(href)
    y_t = np.where(ok_route, -s * x[R, ia] + c * y[R, ia], 0.0)           # the route at a, relative to the ego, across the road
    dr = np.degrees(np.abs(_wrap(h - href[:, None])))
    straight = (np.where(span & v, dr, 0.0) <= LC_MAX_DEG).all(1) & (dn[R, ia] <= STRAIGHT_DEG)
    ib = _slot(a + 20.0)
    later = v[R, ib] & ~turn & (OFFS[ib] > a)
    hab = _cmed(h, v & (rng >= ia[:, None]) & (rng <= ib[:, None]))      # a .. a + 20 m: still moving across (a curve's chord reads 0)
    settled = ~later | (np.abs(-np.sin(hab) * (x[R, ib] - x[R, ia]) + np.cos(hab) * (y[R, ib] - y[R, ia])) <= SETTLE_M)
    blocked = in_turn | (d_turn <= BLOCK_M) | (a < A_MIN)
    why = np.select([~ok_route, in_turn, d_turn <= BLOCK_M, a < A_MIN, ~straight, ~settled, np.abs(y_t) < ON_M],
                    [2, 3, 4, 7, 5, 6, 1], 0)
    side = np.where(why == 0, np.sign(y_t), 0).astype(int)
    y_0 = np.where(ok_route, -s * x[R, s0] + c * y[R, s0], 0.0)         # the ego's own offset from the route where it starts (return-to-route when large)
    out = dict(side=side, y_t=y_t, y_0=y_0, a=a, d_turn=d_turn, blocked=blocked, in_turn=in_turn, gap=gap & ~near_ok, why=why)
    return {k: w[0] for k, w in out.items()} if one else out


def lc_intent(f):
    """One decision's route features -> (side, y_t): +1 left / -1 right / 0 none, and the signed target offset (m)."""
    r = intent(f)
    return int(r["side"]), float(r["y_t"])


class RouteDesire:
    """openpilot desire_helper.py's lane-change state machine driven by the route (module docstring). step() once per decision."""

    def __init__(self, v_min: float = V_MIN, t_max: float = T_MAX, cool: float = COOL):
        self.v_min, self.t_max, self.cool = float(v_min), float(t_max), float(cool)
        self.state, self.side, self.t_on, self.t_off, self.why = "off", 0, None, -np.inf, ""

    def _off(self, t, why):
        self.state, self.side, self.t_on, self.t_off, self.why = "off", 0, None, t, why

    def step(self, t: float, v: float, f, ds=None) -> int:
        """t (s, session clock), v (m/s), f the decision's route features (None = no route), ds the model's desire_state (8,) of the last
        decision or None -> the desire index fed now (0, 3 or 4); self.t_on is the pulse time for desire_queue."""
        r = intent(f) if f is not None else dict(side=0, y_t=0.0, blocked=True, in_turn=False, why=2)
        side, y_t = int(r["side"]), float(r["y_t"])
        self.last = r
        if self.state != "off":
            rem = self.side * y_t                                           # remaining offset toward the side
            if bool(r["in_turn"]):
                self._off(t, "abort_in_turn")
            elif t - self.t_on > self.t_max:
                self._off(t, "abort_timeout")
            elif v < self.v_min - 1.0:
                self._off(t, "abort_slow")
            elif side == -self.side:
                self._off(t, "abort_flip")
            elif self.state == "starting" and (abs(rem) < ON_M or (ds is not None and float(ds[DESIRE[self.side]]) < 0.01)):
                self.state, self.why = "finishing", "crossed"
            elif self.state == "finishing" and abs(rem) < OFF_M:
                self._off(t, "done")
        if self.state == "off" and side and v >= self.v_min and t - self.t_off >= (self.cool if self.why.startswith("abort") else 0.0):
            self.state, self.side, self.t_on, self.why = "starting", side, t, "start"
        return DESIRE.get(self.side, 0) if self.state != "off" else 0

    def log(self) -> dict:
        r = getattr(self, "last", {})
        return dict(state=self.state, side=self.side, t_on=self.t_on, why=self.why, y_t=round(float(r.get("y_t", 0.0)), 3), y_0=round(float(r.get("y_0", 0.0)), 3),
                    d_turn=float(r["d_turn"]) if np.isfinite(r.get("d_turn", np.inf)) else None, intent=int(r.get("side", 0)),
                    iwhy=WHY[int(r.get("why", 2))])


def block_of(t_on, t):
    """The queue block of a pulse sent at t_on, read at t (s): 32 - floor(age / 0.2), or None when the pulse is not in the queue."""
    if t_on is None:
        return None
    age = t - t_on
    if age < -1e-6 or age >= QUEUE_S - 1e-9:
        return None
    return 32 - int(np.floor(max(age, 0.0) / 0.2 + 1e-9))


def desire_queue(d: int, t_on, t) -> np.ndarray:
    """The policy input `_to_copy` (1, 33, 8) float32 for desire index d sent at t_on, read at t (module docstring)."""
    q = np.zeros((1, 33, 8), np.float32)
    b = block_of(t_on, t) if d else None
    if b is not None:
        q[0, b, d] = 1.0
    return q
