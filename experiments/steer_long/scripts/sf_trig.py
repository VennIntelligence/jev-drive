#!/usr/bin/env python3
"""SPIN-FIX v8 trigger screen on plain (or any) reads (numpy only, either box). v8 serves the plan as is unless a call triggers, then the v7 gate
reference for HOLD s. Triggers per call (logged plan, v0, measured a0): dec = some plan point at 0.5 .. 2 s lies more than M m behind the
car braking at A m/s^2 from v0 (the plan asks for more deceleration than the car has: the spin precursor of results/spin_fix.md section 1);
cur = track_ref gate ratio (curv rebuild, segments + PP) > GATE. Per read: share of calls triggered, rollouts touched (would differ from the
plain read), and for every spin rollout (sf_detect) the share of the second before the onset covered (triggered or held).
  sf_trig.py --runs <dir> ... [--list f] [--a -5] [--m 1.0] [--hold 1.0] [--cfg pp=1]"""
import argparse, csv, glob, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "../../alpasim/lib"))
from sf_detect import detect
from sf_screen import sims
import track_ref as TR


def dec_margin(poses, v0, A):
    """min over the 0.5 .. 2 s points of (plan arc - arc of the car braking at A from v0)."""
    p = np.vstack([[0, 0], np.asarray(poses)[:4, :2]])
    s = np.cumsum(np.hypot(*np.diff(p, axis=0).T))
    t = np.arange(1, 5) * 0.5
    sm = np.where(v0 + A * t > 0, v0 * t + 0.5 * A * t ** 2, v0 ** 2 / max(-2 * A, 1e-9))
    return float((s - sm).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--list")
    ap.add_argument("--a", type=float, nargs="+", default=[-5.0])
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--cfg", default="pp=1")
    ap.add_argument("--nocur", action="store_true")
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (x.split("=") for x in filter(None, a.cfg.split(",")))}
    keep = set(open(a.list).read().split()) if a.list else None
    res = {A: dict(calls=0, trig=0, roll=0, touched=0, cur=0, spins=[]) for A in a.a}
    for d in a.runs:
        for sim, dj in sims(d):
            R = {}
            for line in open(dj):
                r = json.loads(line)
                if r.get("kind") == "drive" and "poses" in r and r.get("infer", True):
                    R.setdefault(r["session"], []).append(r)
            for ro in json.load(open(os.path.join(sim, "aggregate", "results-summary.json")))["rollouts"]:
                if (keep is not None and ro["clipgt_id"] not in keep) or ro["rollout_id"] not in R:
                    continue
                c = os.path.join(sim, "controller", f"alpasim_controller_{ro['rollout_id']}.csv")
                if not os.path.exists(c):
                    continue
                rows = list(csv.DictReader(open(c)))
                C = {k: np.array([float(x[k]) for x in rows]) for k in ("timestamp_us", "vx", "vy", "wz", "front_steering_angle", "acceleration")}
                ts = C["timestamp_us"]; t = (ts - ts[0]) / 1e6 + 0.1
                dt = detect(t, C["vx"], C["vy"], C["wz"], C["front_steering_angle"])
                calls = []
                for r in R[ro["rollout_id"]]:
                    v0 = r.get("track", {}).get("v0", r.get("fix", {}).get("v0", r["ego"][4] * 10))
                    i = int(np.abs(ts - r["t0"]).argmin())
                    cur = False
                    if not a.nocur:
                        _, inf = TR.build(np.asarray(r["poses"]), v0, 0.0, mode="curv", a0=float(C["acceleration"][i]), **kw)
                        cur = inf["gate"] > kw.get("gate", TR.GATE)
                    calls.append(((r["t0"] - ts[0]) / 1e6 + 0.1, cur, {A: dec_margin(r["poses"], v0, A) < -a.m for A in a.a}))
                for A, z in res.items():
                    tr = np.array([c[0] for c in calls if c[1] or c[2][A]])
                    z["calls"] += len(calls); z["trig"] += len(tr); z["roll"] += 1; z["touched"] += int(len(tr) > 0)
                    z["cur"] += sum(c[1] for c in calls)
                    if dt["spin"]:
                        w = np.arange(dt["t_first"] - 1.0, dt["t_first"] + 1e-6, 0.1)
                        cov = np.mean([((tr <= x + 1e-6) & (tr >= x - a.hold)).any() for x in w]) if len(tr) else 0.0
                        lead = (dt["t_first"] - tr[tr <= dt["t_first"]].min()) if (tr <= dt["t_first"]).any() else None
                        z["spins"].append((ro["clipgt_id"][7:15], dt["kind"], round(dt["t_first"], 1), round(float(cov), 2), lead and round(lead, 1)))
    for A, z in res.items():
        sp = z["spins"]
        print(f"A {A} M {a.m} hold {a.hold}: calls triggered {z['trig']} / {z['calls']} ({100 * z['trig'] / max(z['calls'], 1):.2f} %, curvature {z['cur']}), "
              f"rollouts touched {z['touched']} / {z['roll']}; spins {len(sp)}, window fully covered {sum(s[3] == 1 for s in sp)}, "
              f"uncovered {[s for s in sp if s[3] < 1]}")


if __name__ == "__main__":
    main()
