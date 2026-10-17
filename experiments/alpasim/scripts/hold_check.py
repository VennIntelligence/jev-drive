#!/usr/bin/env python3
"""JEV_HOLD (lib/static_hold.py, lib/sh30_hold.py) checks, CPU, no simulator (experiments/alpasim_proxy/plans/2026-10-17-arena-prereg.md,
supplement 1). Box, repo root, envs/op-train (sh30_driver imports torch / grpc / the AlpaSim protos):

  1. off: with JEV_HOLD unset, static_hold is off and importing sh30_hold patches nothing (Driver.drive, Core.plan and make_cached_plan
     are sh30_driver's own objects), so `python sh30_hold.py` runs sh30_driver.main() unchanged
  2. limit_arc: on random plans, a limit at or beyond the plan's arc returns the input object; otherwise the poses within the limit are
     bit-identical, the rest lie on the polyline at arc s_lim, and the served polyline's arc is s_lim
  3. step, per mode (module reloaded with JEV_HOLD set): pre-handoff calls unchanged; the latch at the 0.5 s call (held below 0.3 m/s,
     free above, free sessions bit-identical); full = every position at the origin; leash3 = arc cut to 3 - |ego - xy0|; lead = held until a
     present lead moves (or a seen lead is gone) while the plan's 2 s arc >= 2 m, never released without a lead
  hold_check.py [--n 2000]
"""
import argparse
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "lib")
sys.path.insert(0, LIB)


def mod(mode):
    os.environ.pop("JEV_HOLD", None)
    if mode:
        os.environ["JEV_HOLD"] = mode
    import static_hold
    return importlib.reload(static_hold)


def plan(rng, v):
    yaw = rng.normal(0, 0.05, 8).cumsum()
    sp = np.clip(v + rng.normal(0, 1, 8).cumsum() * 0.5, 0, None)
    xy = np.cumsum(np.c_[np.cos(yaw), np.sin(yaw)] * sp[:, None] * 0.5, 0)
    return np.c_[xy, yaw].astype(np.float32)


def arc(P):
    p = np.r_[np.zeros((1, 2)), np.asarray(P, float)[:, :2]]
    return np.hypot(*np.diff(p, axis=0).T).cumsum()


def check_off():
    H = mod(None)
    assert not H.ON and H.SUFFIX == "" and H.MODE == "off"
    import sh30_driver as D
    ref = (D.Driver.drive, D.C.Core.plan, D.make_cached_plan)
    import sh30_hold  # noqa: F401
    assert (D.Driver.drive, D.C.Core.plan, D.make_cached_plan) == ref and D.make_cached_plan.__module__.startswith("navsim_transfuser_challenge")
    print("1 off: static_hold off, importing sh30_hold patches nothing")


def check_limit(n):
    H, rng = mod(None), np.random.default_rng(0)
    for _ in range(n):
        P = plan(rng, rng.uniform(0, 15))
        c = arc(P)
        if c[-1] < 1e-3:
            continue
        assert H.limit_arc(P, c[-1] + rng.uniform(0, 5)) is P
        s = rng.uniform(0, c[-1])
        Q = H.limit_arc(P, s)
        keep = c <= s
        assert np.array_equal(Q[keep], P[keep]) and Q.dtype == P.dtype
        assert abs(arc(Q)[-1] - s) < 1e-3 * max(1.0, s), (arc(Q)[-1], s)
        assert np.allclose(Q[~keep], Q[~keep][0]), "every cut pose sits on the one point at s_lim"
    print(f"2 limit_arc: {n} random plans ok")


def check_step(n):
    rng = np.random.default_rng(1)
    T = lambda k: 10**6 + 500_000 * k  # noqa: E731   decisions every 0.5 s; k = 1 is the handoff call
    for mode in ("full", "leash3", "lead"):
        H = mod(mode)
        assert H.ON and H.SUFFIX == f"-hold{mode}"
        for _ in range(n // 10):
            v0 = rng.choice([0.0, 0.1, 0.29, 0.31, 2.0, 8.0])
            st = H.new()
            P = plan(rng, 3.0)
            Q, i = H.step(st, P, v0, (0.0, 0.0), T(0))
            assert Q is P and i["state"] == "pre"
            xy = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50)])
            free = v0 >= 0.3
            for k in range(1, 11):
                P = plan(rng, rng.uniform(0, 8))
                d = 0.0 if k == 1 else rng.uniform(0, 5)
                pos = xy + d * np.array([np.cos(k), np.sin(k)])
                o = None
                if mode == "lead":
                    lead = np.zeros(144)
                    lead[0], lead[2] = 10.0, (2.0 if k >= 6 else 0.0)    # a lead 10 m ahead, moving from k = 6
                    o = dict(lead=lead, lead_prob=np.array([4.0, 0, 0]), mu=np.zeros((1, 4)))
                Q, i = H.step(st, P, 0.0 if not free else v0, pos, T(k), o)
                if free:
                    assert Q is P and i["state"] == "free"
                    continue
                if mode == "full" or (mode == "lead" and i["state"] == "held"):
                    assert i["state"] == "held" and np.all(Q[:, :2] == 0) and arc(Q)[-1] == 0 and Q.dtype == P.dtype
                if mode == "leash3":
                    lim = max(0.0, 3.0 - d)
                    assert arc(Q)[-1] <= lim + 1e-4 and (Q is P) == (arc(P)[-1] <= lim)
                if mode == "lead" and i["state"] == "free":
                    assert k >= 6 and Q is P
                if mode == "lead" and k >= 6 and H.arc2(P) >= 2.0:
                    assert i["state"] == "free"
        st = H.new()                                     # lead mode without any lead: never released
        if mode == "lead":
            for k in range(0, 11):
                Q, i = H.step(st, plan(rng, 6.0), 0.0, (0, 0), T(k), dict(lead=np.zeros(144), lead_prob=np.array([-6.0, 0, 0]), mu=np.zeros((1, 4))))
            assert i["state"] == "held"
        print(f"3 step {mode}: ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    check_limit(a.n)
    check_step(a.n)
    check_off()
    print("hold_check: all checks passed")
