"""JEV_TURN_GAIN=<g>: a serve-side turn-magnitude switch for ROUTE-MEM checkpoints (diagnostic / exam technique, NOT adopted; prereg
experiments/route_follow/plans/2026-10-15-turn-gain-switch-prereg.md). Off by default (unset, empty or 1: the served plan is untouched).

Question: with the policy on its own states (no log handover), is turn magnitude the bottleneck? Decisions 296 / 298: a lateral gain x1.25 on
the policy's own plan completes many turns when started at a handover; the approach start (D40) rarely does on the exam.

Rule, at every inference, from the route memory the driver fed (rmem features, either layout: the first 40 x 5 numbers are the points
[valid, x / 20, y / 20, cos h, sin h] at nominal arc offsets OFFS = -6 .. +72 m in the current rear-axle frame) and the model's plan:
  h_ref   the route heading at the +0 m slot when it is valid, else the ego's own heading (0): rmem's d_turn reference
  dr40    the signed route heading change of the valid slot in 0 .. +40 m with the largest |h - h_ref|
  L2      arc length of the plan from the ego through its poses at 0.5 .. 2.0 s; dplan = the plan's yaw at 2.0 s
  dr_arc  the route heading change at arc L2 (slot headings unwrapped, linear in the offset, (0, h_ref) prepended)
  on      |dr40| >= 45 deg, dr_arc has dr40's sign, and sign(dr_arc) x dplan < 0.6 |dr_arc| (the plan turns less than 60 % of the route
          over the arc it covers in 2 s); since the smoke (prereg Amendment 1) also L2 >= 1 m and |dr_arc| >= 2 deg (no firing on a standing
          or near-straight stretch) and sign(dr_arc) x dplan >= -1 deg (a plan that turns against the route is not scaled: the gain would
          amplify the wrong-way turn, outside the magnitude question)
While on, the poses handed to the controller are branch_poses(plan, g): every pose yaw and segment course about the ego multiplied by g at
the same segment lengths (experiments/onpol_pai/lib/onp_log_driver.branch_poses with s = 1, the curvature push of decisions 290 / 296;
copied here because that driver is not mounted in the PAI stack; selftest() checks the copy). Inputs are the driver's own (the route
messages it received and the model's plan): no log, no map.
"""
from __future__ import annotations

import os

import numpy as np

G = float(os.environ.get("JEV_TURN_GAIN", "") or 1.0)
ON = G != 1.0
SUFFIX = f"-tg{G:g}" if ON else ""
N_PT, NF = 40, 5
OFFS = -6.0 + 2.0 * np.arange(N_PT)                       # rmem.OFFS (both layouts: the same 40 nominal slots)
I0 = int(np.flatnonzero(OFFS == 0.0)[0])
TURN_DEG, AHEAD_M, SHARE = 45.0, 40.0, 0.6
MIN_L2, MIN_ARC_DEG, AGAINST_DEG = 1.0, 2.0, -1.0           # prereg Amendment 1


def branch_poses(poses, g: float):
    """(8, 3) x, y, yaw in the ego frame -> every pose's yaw and every segment's course multiplied by g, segment lengths kept."""
    q = np.asarray(poses, np.float64)
    if g == 1.0:
        return q
    xy = np.r_[[[0.0, 0.0]], q[:, :2]]
    d = np.diff(xy, axis=0)
    ds = np.hypot(*d.T)
    c = g * np.unwrap(np.r_[0.0, np.arctan2(d[:, 1], d[:, 0])])[1:]
    yaw = g * np.unwrap(np.r_[0.0, q[:, 2]])
    xy = np.r_[[[0.0, 0.0]], np.cumsum(ds[:, None] * np.c_[np.cos(c), np.sin(c)], 0)]
    return np.c_[xy[1:], yaw[1:]]


def decide(rf, poses) -> dict:
    """rf: the fed route features (>= 200 numbers) or None; poses (8, 3). -> {on, dr40, dr_arc, dplan, L2} (degrees, m)."""
    q = np.asarray(poses, np.float64)
    xy = np.r_[[[0.0, 0.0]], q[:4, :2]]
    L2 = float(np.hypot(*np.diff(xy, axis=0).T).sum())
    dplan = float(np.degrees(np.unwrap(np.r_[0.0, q[:4, 2]])[-1]))
    out = dict(on=False, dr40=None, dr_arc=None, dplan=round(dplan, 2), L2=round(L2, 2))
    if rf is None:
        return out
    P = np.asarray(rf, np.float64)[:N_PT * NF].reshape(N_PT, NF)
    v = P[:, 0] > 0.5
    h = np.arctan2(P[:, 4], P[:, 3])
    href = h[I0] if v[I0] else 0.0
    ahead = v & (OFFS >= 0.0) & (OFFS <= AHEAD_M)
    if not ahead.any():
        return out
    dh = (h - href + np.pi) % (2 * np.pi) - np.pi
    i = np.flatnonzero(ahead)[np.argmax(np.abs(dh[ahead]))]
    dr40 = float(np.degrees(dh[i]))
    sel = v & (OFFS > 0.0)
    s = np.r_[0.0, OFFS[sel]]
    hu = np.unwrap(np.r_[0.0, dh[sel]])
    dr_arc = float(np.degrees(np.interp(L2, s, hu)))
    toward = np.sign(dr_arc) * dplan
    on = (abs(dr40) >= TURN_DEG and dr_arc * dr40 > 0 and L2 >= MIN_L2 and abs(dr_arc) >= MIN_ARC_DEG and AGAINST_DEG <= toward < SHARE * abs(dr_arc))
    out.update(on=bool(on), dr40=round(dr40, 2), dr_arc=round(dr_arc, 2))
    return out


def apply(rf, poses):
    """-> (served poses, record). Off (G == 1): the input itself and None."""
    if not ON:
        return poses, None
    r = decide(rf, poses)
    r["g"] = G
    return (branch_poses(poses, G) if r["on"] else poses), r


def selftest():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "onpol_pai" / "lib"))
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = np.cumsum(rng.uniform(-0.2, 0.2, 8))
        P = np.c_[np.cumsum(np.cos(a) * 2), np.cumsum(np.sin(a) * 2), a]
        ref = _onp_branch(P, 1.25)
        assert np.allclose(branch_poses(P, 1.25), ref, atol=1e-12)
    # a route turning left 90 deg on a 20 m radius from +10 m; a plan that goes straight -> on; a plan that follows -> off
    s = OFFS
    th = np.clip((s - 10) / 20, 0, np.pi / 2)
    x = np.where(s < 10, s, 10 + 20 * np.sin(th))
    y = np.where(s < 10, 0, 20 * (1 - np.cos(th)))
    rf = np.c_[np.ones(N_PT), x / 20, y / 20, np.cos(th), np.sin(th)].reshape(-1)
    straight = np.c_[np.arange(1, 9) * 3.0, np.zeros(8), np.zeros(8)]
    assert decide(rf, straight)["on"]
    arc = 3.0 * np.arange(1, 9)
    tf = np.clip((arc - 10) / 20, 0, np.pi / 2)
    follow = np.c_[np.where(arc < 10, arc, 10 + 20 * np.sin(tf)), np.where(arc < 10, 0, 20 * (1 - np.cos(tf))), tf]
    assert not decide(rf, follow)["on"], decide(rf, follow)
    assert not decide(None, straight)["on"] and not decide(rf * 0, straight)["on"]
    assert not decide(rf, straight * 0.01)["on"]                                    # standing: L2 < 1 m
    against = follow * np.array([1, -1, -1])                                        # the mirrored plan turns right
    assert not decide(rf, against)["on"]
    print("turn_gain selftest ok")


def _onp_branch(P, g):
    """onp_log_driver.branch_poses (s = 1), read from its source without importing the driver (its import has side effects)."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "onpol_pai" / "lib" / "onp_log_driver.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "branch_poses")
    ns = {"np": np}
    exec(compile(ast.Module([fn], []), "onp_branch", "exec"), ns)
    return ns["branch_poses"](P, g, 1.0)


if __name__ == "__main__":
    selftest()
