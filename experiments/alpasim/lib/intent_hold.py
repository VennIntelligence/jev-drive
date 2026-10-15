"""JEV_INTENT_HOLD=1: a serve-side intent-hold switch (diagnostic / exam technique, NOT adopted; prereg
experiments/route_follow/plans/2026-10-15-intent-hold-prereg.md). Off by default (unset, empty or 0: this module is not even imported by
pai_driver.py and the served plan is the plan).

Question (decision 308, plan creep): in failed turns every new plan re-anchors at the car's actual pose and accepts the car's shortfall
(0.063 m outward per metre travelled vs 0.015 in completed turns). Does holding the policy to its own earlier path complete the turns?

Rule, at every inference, from the anchor p0 = (x, y, yaw) of t0 in the session's local frame, the plan P (8 rear-axle poses at 0.5 s,
after serve_fix), the fed speed v, the fed route memory rf and the command:
  W      P in the world frame with the rear axle as its origin (9 points)
  turn   the route memory shows a heading change >= TURN_DEG within 0 .. AHEAD_M (turn_gain's dr40 on the same 40 slots) or the command is
         LEFT / RIGHT
  reset  (serve P unchanged, I = W) when v < V_MIN, P's arc < MIN_ARC, no turn, I is empty, or P's 2 s point lies more than FAR_M from I
  hold   otherwise: s = arc length from the current rear axle; I_r = I from the rear axle's projection on; the new intent
         B(s) = (1 - a) I_r(s) + a W(s) on the common arc, beyond I_r's end W(s) shifted by (1 - a) times the junction gap (continuous);
         served path Q(s) = B(s) + w(s) (rear axle - B(0)), w = cosine ramp 1 -> 0 over L1 = max(P's arc at 1.0 s, RAMP_MIN) (starts at
         the rear axle, on B after ~1 s); correction Q - W clipped to CLIP_M on every point (the guard); I = B.
         Served poses: the clipped path at P's own arc lengths per step (P's speed profile), yaw = P's yaw + (the served path's chord
         heading - P's chord heading) at that arc (chord +- CHORD_M); a pose still more than CLIP_M from P's pose (arc reparametrisation)
         is pulled back to CLIP_M.
Logged per inference as drive.jsonl `ih` {on, why, d2, e0, cmax, clip, dr40, v, a}; `poses_ih` (served) while on; `poses` stays the plan.
Inputs are the driver's own (its plan, its ego poses, the route messages it received): no log, no map.
"""
from __future__ import annotations

import os

import numpy as np

ON = os.environ.get("JEV_INTENT_HOLD", "") not in ("", "0")
A = float(os.environ.get("JEV_IH_A", "") or 0.3)
SUFFIX = f"-ih{A:g}" if ON else ""
N_PT, NF = 40, 5
OFFS = -6.0 + 2.0 * np.arange(N_PT)                       # rmem.OFFS (both layouts: the same 40 nominal slots, as turn_gain.py)
I0 = int(np.flatnonzero(OFFS == 0.0)[0])
TURN_DEG, AHEAD_M, FAR_M, CLIP_M, V_MIN, MIN_ARC = 15.0, 40.0, 1.5, 1.0, 1.0, 1.0
RAMP_MIN, DS, CHORD_M = 1.0, 0.25, 0.5
LEFT, RIGHT = 0, 2                                         # navsim_transfuser_challenge.navigation.DriveCommand


def new() -> dict:
    """Per-session state: the intent path I (world polyline, (n, 2)) or None."""
    return {"I": None}


def dr40(rf):
    """Signed route heading change (deg) of the valid slot in 0 .. AHEAD_M with the largest |change| vs the +0 slot (or the ego's heading);
    None without a valid slot ahead."""
    if rf is None:
        return None
    P = np.asarray(rf, np.float64)[:N_PT * NF].reshape(N_PT, NF)
    v = P[:, 0] > 0.5
    h = np.arctan2(P[:, 4], P[:, 3])
    href = h[I0] if v[I0] else 0.0
    ahead = v & (OFFS >= 0.0) & (OFFS <= AHEAD_M)
    if not ahead.any():
        return None
    dh = (h - href + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(dh[np.flatnonzero(ahead)[np.argmax(np.abs(dh[ahead]))]]))


def arc(Q):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]


def at(Q, s, S=None):
    """Points of polyline Q at arc lengths s (linear; beyond the end along the last segment)."""
    S = arc(Q) if S is None else S
    s = np.atleast_1d(np.asarray(s, np.float64))
    out = np.c_[np.interp(s, S, Q[:, 0]), np.interp(s, S, Q[:, 1])]
    hi = s > S[-1]
    if hi.any():
        j = len(S) - 1
        while j > 0 and S[j] - S[j - 1] < 1e-9:
            j -= 1
        if j > 0:
            d = (Q[j] - Q[j - 1]) / (S[j] - S[j - 1])
            out[hi] = Q[-1] + (s[hi] - S[-1])[:, None] * d
    return out


def dense(Q, ds=DS):
    """Polyline Q resampled every ds metres of arc (its end kept); repeated points dropped."""
    S = arc(Q)
    keep = np.r_[True, np.diff(S) > 1e-9]
    Q, S = Q[keep], S[keep]
    if S[-1] < 1e-9:
        return Q[:1]
    s = np.r_[np.arange(0.0, S[-1], ds), S[-1]]
    return at(Q, s, S)


def project(Q, p):
    """(arc of the nearest point of polyline Q to p, signed offset of p, left +); the last segment is extended past the end."""
    a, d = Q[:-1], np.diff(Q, axis=0)
    L2 = (d ** 2).sum(1)
    ok = L2 > 1e-12
    if not ok.any():
        return 0.0, float(np.hypot(*(p - Q[0])))
    S = arc(Q)[:-1]
    a, d, L2, S = a[ok], d[ok], L2[ok], S[ok]
    t = ((p - a) * d).sum(1) / L2
    hi = np.ones(len(t))
    hi[-1] = np.inf
    t = np.clip(t, 0.0, hi)
    f = a + t[:, None] * d
    i = int(np.argmin(((p - f) ** 2).sum(1)))
    v = p - f[i]
    return float(S[i] + t[i] * np.sqrt(L2[i])), float(np.sign(d[i, 0] * v[1] - d[i, 1] * v[0]) * np.hypot(*v))


def world(p0, poses):
    """Plan (8, 3) in the rear-axle frame of p0 = (x, y, yaw) -> world points (9, 2), the rear axle first."""
    c, s = np.cos(p0[2]), np.sin(p0[2])
    q = np.asarray(poses, np.float64)[:, :2]
    return np.r_[[p0[:2]], np.c_[p0[0] + c * q[:, 0] - s * q[:, 1], p0[1] + s * q[:, 0] + c * q[:, 1]]]


def chord(Q, s, S, h=CHORD_M):
    a, b = at(Q, np.maximum(s - h, 0.0), S), at(Q, s + h, S)
    d = b - a
    return np.arctan2(d[:, 1], d[:, 0])


def served_poses(p0, poses, B, Wd, W):
    """The served plan (8, 3) and its record fields. B and Wd are the new intent and the plan sampled on the same arc grid (DS from the
    rear axle): Q = B + ramp (rear axle - B(0)), correction Q - Wd clipped, read at P's arc lengths, yaw = P's yaw + chord heading change."""
    q = np.asarray(poses, np.float64)
    Sw = arc(Wd)
    sP = arc(W)[1:]                                        # P's own arc at 0.5 .. 4 s
    L1 = max(float(arc(W)[2]), RAMP_MIN)                   # P's arc at 1.0 s
    w = np.where(Sw < L1, 0.5 * (1 + np.cos(np.pi * np.clip(Sw / L1, 0, 1))), 0.0)
    Q = B + w[:, None] * (np.asarray(p0[:2], np.float64) - B[0])
    c = Q - Wd                                             # the correction on the common arc grid
    n = np.hypot(*c.T)
    clip = n > CLIP_M
    c[clip] *= (CLIP_M / n[clip])[:, None]
    Q = Wd + c
    Sq = arc(Q)
    xy = at(Q, sP, Sq)
    d = xy - W[1:]
    n2 = np.hypot(*d.T)
    far = n2 > CLIP_M
    xy[far] = W[1:][far] + d[far] * (CLIP_M / n2[far])[:, None]
    yaw_w = q[:, 2] + p0[2] + (chord(Q, sP, Sq) - chord(Wd, sP, Sw) + np.pi) % (2 * np.pi) - np.pi
    cs, sn = np.cos(p0[2]), np.sin(p0[2])
    r = xy - np.asarray(p0[:2], np.float64)
    out = np.c_[cs * r[:, 0] + sn * r[:, 1], -sn * r[:, 0] + cs * r[:, 1], yaw_w - p0[2]]
    return out, dict(cmax=round(float(np.hypot(*(xy - W[1:]).T).max()), 4), clip=int(clip.sum() + far.sum()))


def step(st: dict, p0, poses, v: float, rf, cmd: int, a: float = None):
    """One inference -> (served poses (8, 3), record). st is the session's state (new())."""
    a = A if a is None else a
    p0 = np.asarray(p0, np.float64)
    q = np.asarray(poses, np.float64)
    W = world(p0, q)
    d40 = dr40(rf)
    turn = (d40 is not None and abs(d40) >= TURN_DEG) or int(cmd) in (LEFT, RIGHT)
    rec = dict(on=False, why="", d2=None, e0=None, cmax=0.0, clip=0, dr40=None if d40 is None else round(d40, 2), v=round(float(v), 3), a=a)
    I = st["I"]
    if v < V_MIN:
        rec["why"] = "slow"
    elif arc(W)[-1] < MIN_ARC:
        rec["why"] = "short"
    elif not turn:
        rec["why"] = "noturn"
    elif I is None or len(I) < 2:
        rec["why"] = "empty"
    else:
        rec["d2"] = round(abs(project(I, W[4])[1]), 4)       # P's 2 s point (pose 4 of 8) vs I
        if rec["d2"] > FAR_M:
            rec["why"] = "far"
    if rec["why"]:
        st["I"] = dense(W)
        return q, rec
    s_c, e0 = project(I, p0[:2])
    rec["e0"] = round(e0, 4)
    Si = arc(I)
    Ir = np.r_[at(I, [s_c], Si), I[Si > s_c]]                # I from the rear axle's projection on
    Lr = arc(Ir)[-1]
    Wd = dense(W)
    s = arc(Wd)
    Is = at(Ir, np.minimum(s, Lr))
    gap = at(Ir, [min(Lr, s[-1])])[0] - at(Wd, [min(Lr, s[-1])], s)[0]
    B = np.where((s <= Lr)[:, None], (1 - a) * Is + a * Wd, Wd + (1 - a) * gap)
    out, r = served_poses(p0, q, B, Wd, W)
    st["I"] = B
    rec.update(on=True, **r)
    return out, rec


def selftest():
    rng = np.random.default_rng(0)
    # 1. a straight route, a straight plan: never on (no turn); state reset to the plan
    st = new()
    straight = np.c_[np.arange(1, 9) * 2.0, np.zeros(8), np.zeros(8)]
    o, r = step(st, [0, 0, 0], straight, 4.0, None, 1)
    assert not r["on"] and r["why"] == "noturn" and o is not None and np.array_equal(o, straight)
    # 2. a left command: first decision empty -> reset, then the same plan from the same place: on, served == plan (no correction)
    st = new()
    th = np.linspace(0.05, 0.6, 8)
    P = np.c_[np.cumsum(2 * np.cos(th)), np.cumsum(2 * np.sin(th)), th]
    o, r = step(st, [0, 0, 0], P, 4.0, None, LEFT)
    assert not r["on"] and r["why"] == "empty"
    o, r = step(st, [0, 0, 0], P, 4.0, None, LEFT)
    assert r["on"] and np.abs(o[:, :2] - P[:, :2]).max() < 0.02 and np.abs(o[:, 2] - P[:, 2]).max() < 0.02, (o - P, r)
    # 3. the car 0.3 m outside the intent with a plan re-anchored there (shifted by the car's offset): served goes back toward I
    st = new()
    step(st, [0, 0, 0], P, 4.0, None, LEFT)
    p1 = np.array([0.0, -0.3, 0.0])                          # right of the intent = outside of a left turn
    o, r = step(st, p1, P, 4.0, None, LEFT)
    assert r["on"] and abs(r["e0"] + 0.3) < 1e-6
    Wsv = world(p1, o)
    assert np.allclose(Wsv[0], p1[:2])
    lat = np.array([project(world([0, 0, 0], P), x)[1] for x in Wsv[3:]])
    assert np.all(np.abs(lat) < 0.3 * 0.75), lat              # after ~1 s within (1 - a) of the offset back toward the intent
    assert np.all(lat > -0.3 + 1e-3)
    # 4. guard: an intent 5 m away from a plan that is still within FAR_M at 2 s is impossible; force a big e0 and check the clip
    st = {"I": dense(world([0, 1.4, 0], P))}
    o, r = step(st, [0, 0, 0], P, 4.0, None, LEFT)
    assert r["on"] and r["cmax"] <= CLIP_M + 1e-9
    # 5. reset on a far plan, a slow car, a short plan
    st = {"I": dense(world([0, 3.0, 0], P))}
    _, r = step(st, [0, 0, 0], P, 4.0, None, LEFT)
    assert r["why"] == "far" and not r["on"]
    _, r = step(new(), [0, 0, 0], P, 0.5, None, LEFT)
    assert r["why"] == "slow"
    _, r = step({"I": dense(world([0, 0, 0], P))}, [0, 0, 0], P * 0.01, 4.0, None, LEFT)
    assert r["why"] == "short"
    # 6. route memory: a 90 deg left turn 20 m ahead switches the hold on with a straight command
    s = OFFS
    tt = np.clip((s - 10) / 20, 0, np.pi / 2)
    rf = np.c_[np.ones(N_PT), np.zeros(N_PT), np.zeros(N_PT), np.cos(tt), np.sin(tt)].reshape(-1)
    assert dr40(rf) > 80 and dr40(rf * 0) is None
    st = new()
    step(st, [0, 0, 0], P, 4.0, rf, 1)
    _, r = step(st, [0, 0, 0], P, 4.0, rf, 1)
    assert r["on"]
    # 7. random plans: served starts at the rear axle, every pose within CLIP_M of the plan, arc per step kept away from the guard
    for _ in range(200):
        a_ = np.cumsum(rng.uniform(-0.15, 0.15, 8))
        Pr = np.c_[np.cumsum(np.cos(a_) * 2), np.cumsum(np.sin(a_) * 2), a_]
        st = {"I": dense(world([0, rng.uniform(-1.2, 1.2), 0], Pr))}
        o, r = step(st, [0, 0, rng.uniform(-0.05, 0.05)], Pr, 4.0, None, LEFT)
        if r["on"]:
            assert np.hypot(*(world([0, 0, 0], o)[1:] - world([0, 0, 0], Pr)[1:]).T).max() <= CLIP_M + 1e-6
    print("intent_hold selftest ok")


if __name__ == "__main__":
    selftest()
