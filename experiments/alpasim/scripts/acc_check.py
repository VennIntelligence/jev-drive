#!/usr/bin/env python3
"""JEV_ACC (serve_fix.py (d)) checks, CPU, no simulator (experiments/route_follow/plans/2026-10-16-pai-acc-prereg.md).

  1. identity: with JEV_ACC unset, serve_fix.Serve on random plans, lead outputs, ego states and route arguments returns the same poses and
     log as serve_fix at <ref> (a git revision, default the parent of the commit that added JEV_ACC), bit for bit, for the served
     configuration (VCONT 1.0 + LEAD) and three more switch sets
  2. behaviour (JEV_ACC=1): set speed held at the handoff, curve limit from a route with a 90 deg turn, the stop guard on a stopping plan,
     the straight extension past the plan's end
  3. PAI ACC identity (nuPlan wiring, plans/2026-10-16-nuplan-acc-prereg.md): with JEV_ACC=1 and the default handoff (PAI's 1.6 s) the
     poses and log equal serve_fix at <acc_ref> (the revision the PAI ACC read ran on), bit for bit, on random sessions with routes and clocks
  4. nuPlan handoff: Serve(acc_t_hand=ACC_T_HAND_NUPLAN) on a 0.5 s session clock locks v_set at the 0.5 s call (the nuPlan handoff step)
  The nuPlan driver's own off-identity (sh30_driver.py with JEV_ACC unset against the previous revision) is a closed-loop check on the box
  (rf_nuacc.sh ident), not here.
  acc_check.py [--ref <rev>] [--acc-ref <rev>] [--n 3000]
"""
import argparse, importlib.util, os, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB, REPO = os.path.join(HERE, "..", "lib"), os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)


def load(path, name, env):
    for k in ("JEV_VCONT", "JEV_LEAD", "JEV_BASE", "JEV_ACC"):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def rand_case(rng):
    v = rng.uniform(0, 30)
    sp = np.clip(v * rng.uniform(0.0, 1.3) + rng.normal(0, 1, 8).cumsum() * 0.3, 0, None)
    yaw = rng.normal(0, 0.05, 8).cumsum()
    xy = np.cumsum(np.c_[np.cos(yaw), np.sin(yaw)] * sp[:, None] * 0.5, 0)
    o = dict(poses=np.c_[xy, yaw], poses_base=np.c_[xy * rng.uniform(0.7, 1.2), yaw], lead=rng.normal(0, 3, 144) + np.r_[rng.uniform(5, 60), np.zeros(143)],
             lead_prob=rng.normal(0, 2, 3), mu=np.array([[0, 0, 0, v * rng.uniform(0.8, 1.1)]]))
    M = np.c_[np.arange(10) * 4.2 + rng.uniform(-5, 45), rng.normal(0, 2) + rng.normal(0, 0.3, 10).cumsum()]
    return o, v, rng.normal(0, 1)


def identity(ref, n):
    with tempfile.TemporaryDirectory() as d:
        old = os.path.join(d, "serve_fix_ref.py")
        open(old, "w").write(subprocess.check_output(["git", "-C", REPO, "show", f"{ref}:experiments/alpasim/lib/serve_fix.py"], text=True))
        sys.path.insert(0, LIB)
        for env in ({"JEV_VCONT": "1.0", "JEV_LEAD": "1"}, {"JEV_VCONT": "1.0"}, {"JEV_LEAD": "1"}, {"JEV_VCONT": "0.5", "JEV_BASE": "2", "JEV_LEAD": "1"}):
            A, B = load(old, "fx_ref", env), load(os.path.join(LIB, "serve_fix.py"), "fx_new", env)
            assert not B.ACC and B.SUFFIX == A.SUFFIX
            rng, bad = np.random.default_rng(0), 0
            for sess in range(n // 50):
                sa, sb = A.Serve(A.VCONT, A.LEAD, 2.3, A.BASE), B.Serve(B.VCONT, B.LEAD, 2.3, B.BASE, B.ACC)
                for k in range(50):
                    o, v, a = rand_case(rng)
                    t = 10**6 * sess + 100_000 * k
                    pa, ia = sa({**o}, v, a, t)
                    oa = {**o}
                    ib = B.apply(sb, oa, v, a, t)
                    pb = oa["poses"]
                    ia.pop("ms", None), ib.pop("ms", None)
                    bad += int(not (np.array_equal(pa, pb) and ia == ib))
            print(f"identity {env}: {n // 50 * 50} decisions, {bad} differ")
            assert bad == 0


def acc_identity(ref, n):
    """3: JEV_ACC=1, PAI's default handoff: new serve_fix == serve_fix at ref, with routes and session clocks."""
    with tempfile.TemporaryDirectory() as d:
        old = os.path.join(d, "serve_fix_accref.py")
        open(old, "w").write(subprocess.check_output(["git", "-C", REPO, "show", f"{ref}:experiments/alpasim/lib/serve_fix.py"], text=True))
        sys.path.insert(0, LIB)
        for env in ({"JEV_VCONT": "1.0", "JEV_LEAD": "1", "JEV_ACC": "1"}, {"JEV_VCONT": "1.0", "JEV_ACC": "1"}):
            A, B = load(old, "fxa_ref", env), load(os.path.join(LIB, "serve_fix.py"), "fxa_new", env)
            assert A.ACC and B.ACC and A.SUFFIX == B.SUFFIX
            rng, bad = np.random.default_rng(1), 0
            for sess in range(n // 50):
                sa, sb = A.Serve(A.VCONT, A.LEAD, 2.3, 0, True), B.Serve(B.VCONT, B.LEAD, 2.3, 0, True)
                for k in range(50):
                    o, v, a = rand_case(rng)
                    M = rand_case(rng)[0]["poses"][:, :2] * 3 + rng.uniform(-10, 40, 2) if rng.random() < 0.8 else None
                    t, t_s = 10**6 * sess + 100_000 * k, 0.1 * k
                    oa, ob = {**o}, {**o}
                    ia, ib = A.apply(sa, oa, v, a, t, route=M, t_s=t_s), B.apply(sb, ob, v, a, t, route=M, t_s=t_s)
                    ia.pop("ms", None), ib.pop("ms", None)
                    bad += int(not (np.array_equal(oa["poses"], ob["poses"]) and ia == ib))
            print(f"ACC identity vs {ref} {env}: {n // 50 * 50} decisions, {bad} differ")
            assert bad == 0


def nuplan_handoff():
    """4: the nuPlan handoff (0.5 s steps): v_set locks at the 0.5 s call, not at PAI's 1.6 s."""
    sys.path.insert(0, LIB)
    F = load(os.path.join(LIB, "serve_fix.py"), "fx_nu", {"JEV_VCONT": "1.0", "JEV_ACC": "1"})
    s = F.Serve(F.VCONT, False, 2.3, 0, True, F.ACC_T_HAND_NUPLAN)
    straight = lambda v: dict(poses=np.c_[v * 0.5 * np.arange(1, 9), np.zeros(8), np.zeros(8)])  # noqa: E731
    i0 = F.apply(s, straight(5.0), 12.0, 0.0, 0, route=None, t_s=0.0)["acc"]       # forced step: no v_set yet (the ego speed)
    i1 = F.apply(s, straight(5.0), 11.0, 0.0, 1, route=None, t_s=0.5)["acc"]       # handoff: v_hand = 11 (the human's speed)
    i2 = F.apply(s, straight(5.0), 8.0, 0.0, 2, route=None, t_s=1.0)["acc"]
    print("nuPlan handoff", i0["v_hand"], i1["v_hand"], i2["v_set"])
    assert i0["v_hand"] is None and i1["v_hand"] == 11.0 and i2["v_set"] == 11.0
    print("nuPlan handoff ok")


def behaviour():
    sys.path.insert(0, LIB)
    F = load(os.path.join(LIB, "serve_fix.py"), "fx_acc", {"JEV_VCONT": "1.0", "JEV_ACC": "1"})
    s = F.Serve(F.VCONT, False, 2.3, 0, True)
    straight = lambda v: dict(poses=np.c_[v * 0.5 * np.arange(1, 9), np.zeros(8), np.zeros(8)])  # noqa: E731
    # handoff at 15 m/s, model plans 10 m/s: set speed held, served ~15 m/s, path extended straight beyond the plan's end (40 m)
    for k, t_s in enumerate((0.0, 0.8, 1.6, 1.7)):
        o = straight(10.0)
        F.apply(s, o, 15.0, 0.0, k, route=None, t_s=t_s)
    i = s.acc.profile(straight(10.0)["poses"], 15.0, np.full(len(F.TG), 10.0), None, 1.8)[1]
    o = straight(10.0)
    info = F.apply(s, o, 15.0, 0.0, 9, route=None, t_s=1.9)
    print("handoff", i, "served 4 s x", o["poses"][-1, 0].round(2))
    assert i["v_set"] == 15.0 and info["acc"]["won"] == "set" and abs(o["poses"][-1, 0] - 60.0) < 0.5 and np.allclose(o["poses"][:, 1], 0)
    # 90 deg left turn of radius 10 m, 40 m ahead: v_curve at the turn = sqrt(2.4 * 0.1) * 10 ~ 4.9 m/s, decelerating at 1.5 m/s^2 before it
    th = np.linspace(0, np.pi / 2, 40)
    M = np.r_[np.c_[np.arange(-20, 40, 2.0), np.zeros(30)], np.c_[40 + 10 * np.sin(th), 10 - 10 * np.cos(th)][1:], np.c_[np.full(10, 50.0), 10 + 4.2 * np.arange(1, 11)]]
    lim, cov = F.route_limit(M, 15.0)
    vmin = lim[1].min()
    s_turn = lim[0][np.argmin(lim[1])]
    print("curve: v_min %.2f at %.1f m, v at 0 m %.2f, cover %.0f m" % (vmin, s_turn, lim[1][0], cov))
    assert 4.3 < vmin < 5.8 and 38 < s_turn < 58 and abs(lim[1][0] - np.sqrt(vmin ** 2 + 2 * 1.5 * s_turn)) < 1.0
    o = straight(15.0)
    info = F.apply(s, o, 15.0, 0.0, 10, route=M, t_s=2.0)
    print("curve served", info["acc"])
    assert info["acc"]["v_curve"] is not None and info["acc"]["v_curve"] < 15.0
    # stop guard: the model's previous plan stops -> min(ACC, model)
    s2 = F.Serve(F.VCONT, False, 2.3, 0, True)
    F.apply(s2, straight(12.0), 12.0, 0.0, 0, route=None, t_s=1.7)
    F.apply(s2, straight(0.3), 12.0, 0.0, 1, route=None, t_s=1.8)           # this plan stops: the next decision is guarded
    o = straight(0.3)
    info = F.apply(s2, o, 11.0, 0.0, 2, route=None, t_s=1.9)
    print("guard", info["acc"])
    assert info["acc"]["guard"] == "stop" and info["acc"]["won"] in ("model", "ramp") and o["poses"][-1, 0] < 30
    o = straight(8.0)
    info = F.apply(s2, o, 9.0, 0.0, 3, route=None, t_s=2.0)                # previous plan still the stop: guarded once more
    o = straight(8.0)
    info = F.apply(s2, o, 9.0, 0.0, 4, route=None, t_s=2.1)                # the model moves again: ACC resumes towards v_set 12
    print("resume", info["acc"])
    assert info["acc"]["guard"] == "" and info["acc"]["v15"] > 9.0
    print("behaviour ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="fc245afc~1")       # the parent of the commit that added JEV_ACC
    ap.add_argument("--acc-ref", default="dd4c8d24")   # serve_fix as the PAI ACC read ran it (unchanged since bed9cb8e)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    identity(a.ref, a.n)
    acc_identity(a.acc_ref, a.n)
    nuplan_handoff()
    behaviour()
