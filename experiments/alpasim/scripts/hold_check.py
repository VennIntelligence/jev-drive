#!/usr/bin/env python3
"""JEV_HOLD (lib/static_hold.py, lib/sh30_hold.py) checks, CPU, no simulator (experiments/alpasim_proxy/plans/2026-10-17-arena-prereg.md,
supplement 1). Box, repo root, envs/op-train (sh30_driver imports torch / grpc / the AlpaSim protos):

  1. off: with JEV_HOLD unset, static_hold is off and importing sh30_hold patches nothing (Driver.drive, Core.plan and make_cached_plan
     are sh30_driver's own objects), so `python sh30_hold.py` runs sh30_driver.main() unchanged
  2. limit_arc: on random plans, a limit at or beyond the plan's arc returns the input object; otherwise the poses within the limit are
     bit-identical, the rest lie on the polyline at arc s_lim, and the served polyline's arc is s_lim
  3. step, per mode (module reloaded with JEV_HOLD set): pre-handoff calls unchanged; the latch at the 0.5 s call (held below 0.3 m/s,
     free above, free sessions bit-identical); full = every position at the origin; leash3 = arc cut to 3 - |ego - xy0|; lead = held until a
     present lead moves (or a seen lead is gone) while the plan's 2 s arc >= 2 m, never released without a lead; stop3 = leash3 on the
     same random sessions (constant speed, so no stop latch)
  4. stop3 (supplement 2): served poses bit-identical to leash3's on sessions with random speeds below 0.3 m/s at the handoff; at or above
     it, a stop latch exactly when (v1 - v0) / 0.5 < -0.5 and v1 / -a <= 3, with s_lim = v1^2 / (-2a) + 3 - |ego - xy0|; free otherwise
  5. wrapper (sh30_hold.install, JEV_HOLD=stop3, the driver's make_cached_plan replaced by a recorder): on random sessions the plan the
     wrapper hands on is the one static_hold.step serves (same object when uncut, bit-identical when cut) and hold.jsonl carries step's
     records; with step raising, every call hands on the model's own plan object and the exception is logged once; get_version appends
     -holdstop3 to sh30_driver's version
  hold_check.py [--n 2000]
"""
import argparse
import importlib
import json
import logging
import os
import sys
import tempfile
from types import SimpleNamespace as NS

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
    for mode in ("full", "leash3", "lead", "stop3"):
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
                if mode in ("leash3", "stop3"):
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


def check_stop(n):
    T = lambda k: 10**6 + 500_000 * k  # noqa: E731
    run = {}
    for mode in ("leash3", "stop3"):
        H, rng, out = mod(mode), np.random.default_rng(2), []
        for _ in range(n // 10):
            v0, v1 = rng.uniform(0, 6), rng.choice([rng.uniform(0, 0.3), rng.uniform(0.3, 6)])
            st, xy, seq = H.new(), rng.uniform(-50, 50, 2), []
            for k in range(0, 11):
                P = plan(rng, rng.uniform(0, 8))
                d = 0.0 if k <= 1 else rng.uniform(0, 8)
                Q, i = H.step(st, P, (v0, v1)[min(k, 1)] if k <= 1 else rng.uniform(0, 3), xy + d * np.array([np.cos(k), np.sin(k)]), T(k))
                seq.append((Q is P, np.asarray(Q).copy(), i["state"], i.get("s_lim"), d, P))
            out.append((v0, v1, seq))
        run[mode] = out
    ns = 0
    for (v0, v1, a), (_, _, b) in zip(run["leash3"], run["stop3"]):
        if v1 < 0.3:
            assert all(x[0] == y[0] and np.array_equal(x[1], y[1]) and x[2] == y[2] for x, y in zip(a, b)), "stop3 differs from leash3 below 0.3"
            continue
        acc = (v1 - v0) / 0.5
        hit = acc < -0.5 and v1 / -acc <= 3.0
        ns += hit
        for k, (same, Q, state, s_lim, d, P) in enumerate(b[1:], 1):
            assert state == ("held" if hit else "free")
            if not hit:
                assert same
                continue
            lim = max(0.0, v1 * v1 / (-2 * acc) + 3.0 - d)
            assert abs(s_lim - round(lim, 3)) < 2e-3 and arc(Q)[-1] <= lim + 1e-4 and same == (arc(P)[-1] <= lim)
    assert ns > 0
    print(f"4 stop3: identical to leash3 below 0.3 m/s; {ns} stop latches exactly where the rule says")


def check_wrap(n):
    H = mod("stop3")
    import sh30_driver as D
    import sh30_hold as W
    ref = (D.Driver.drive, D.C.Core.plan, D.make_cached_plan, D.Driver.get_version)
    D.make_cached_plan = lambda t0, anchor, poses: poses       # records what the wrapper hands on
    logs = []
    h = logging.Handler(level=logging.ERROR)
    h.emit = logs.append
    D.LOG.addHandler(h)
    tmp = tempfile.mkdtemp()
    try:
        W.install(tmp)
        rng, T, recs, nc, nh = np.random.default_rng(3), (lambda k: 10**6 + 500_000 * k), [], 0, 0  # noqa: E731
        for j in range(n // 10):
            v0, v1 = rng.uniform(0, 6), rng.choice([rng.uniform(0, 0.3), rng.uniform(0.3, 6)])
            st, xy = H.new(), rng.uniform(-50, 50, 2)
            for k in range(11):
                P = plan(rng, rng.uniform(0, 8))
                v = (v0, v1)[k] if k <= 1 else rng.uniform(0, 3)
                pos = xy + (0.0 if k <= 1 else rng.uniform(0, 8)) * np.array([np.cos(k), np.sin(k)])
                W.TL.s, W.TL.uuid, W.TL.o = NS(scene=f"sc{j}", velocity=lambda t, v=v: np.array([[v * 0.6, v * 0.8]])), f"u{j}", None
                got = D.make_cached_plan(T(k), NS(pose=NS(vec=NS(x=pos[0], y=pos[1]))), P)
                Q, i = H.step(st, P, float(np.hypot(v * 0.6, v * 0.8)), pos, T(k))
                assert (got is P) == (Q is P) and np.array_equal(got, Q) and got.dtype == Q.dtype
                recs.append(i)
                nc += Q is not P
                nh += i["state"] == "held"
        logged = [json.loads(x) for x in open(os.path.join(tmp, "hold.jsonl"))]
        assert len(logged) == len(recs) and all({k: r[k] for k in i} == i for r, i in zip(logged, recs)) and nc > 0 and nh > 0
        step0 = W.HD.step
        W.HD.step = lambda *a, **k: 1 / 0
        try:
            for k in range(20):
                P = plan(rng, 5.0)
                W.TL.s, W.TL.uuid, W.TL.o = NS(scene="scx", velocity=lambda t: np.zeros((1, 2))), "ux", None
                assert D.make_cached_plan(T(k), NS(pose=NS(vec=NS(x=0.0, y=0.0))), P) is P
        finally:
            W.HD.step = step0
        assert len(logs) == 1 and logs[0].exc_info, logs
        core = NS(tag="RSG-L18-s0", cold="backwarp")
        v_w = D.Driver.get_version(NS(core=core), None, None).version_id
        v_0 = ref[3](NS(core=core), None, None).version_id
        assert v_w == v_0 + "-holdstop3", (v_w, v_0)
    finally:
        D.Driver.drive, D.C.Core.plan, D.make_cached_plan, D.Driver.get_version = ref
        D.LOG.removeHandler(h)
    print(f"5 wrapper stop3: {len(recs)} calls ({nh} held, {nc} cut) hand on step's plan bit for bit, hold.jsonl = step's records; "
          f"step raising: 20 / 20 calls serve the model's plan object, 1 log line; version {v_w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    check_limit(a.n)
    check_step(a.n)
    check_stop(a.n)
    check_off()
    check_wrap(a.n)
    print("hold_check: all checks passed")
