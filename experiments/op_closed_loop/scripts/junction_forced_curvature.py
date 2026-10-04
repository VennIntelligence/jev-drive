"""Desired vs clipped vs executed curvature of arm D (action head only) through forced turns (results/junction_forced_and_arrow.md).

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/junction_forced_curvature.py [--turns 28180:0,24944:0,28008:0,24758:0]

All curves right-positive (CARLA yaw sign), against route arc length s relative to the turn start (dense centreline projection of the car):
  route    curvature the dense route needs (the turn itself)
  desired  head output act_k of plans.jsonl (what the head asks for, per plan tick)
  clipped  desired passed through openpilot's clip_curvature (lib/op_ctrl.py: rate 5 / v^2 per s, lat accel 3 m/s^2, |k| <= 0.2), 100 Hz inner steps, desired held
           between plans, speed from the ticks. Counterfactual: the shipped D arm does NOT apply it (no OP_CTRL), the car's steer comes straight from `desired`
  steer    the curvature the logged CARLA steer command stands for (b2d_zeroshot_agent._steer_curvature: bicycle + steering curve, after the steer-rate limit)
  actual   the car's own path curvature, d yaw / d s of the truth poses (3-sample smoothing)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[2] / "lib")]
import junction_cl_report as R  # noqa: E402
import op_ctrl  # noqa: E402

WB, MAXST, CURVE = 2.8604714913890885, math.radians(69.99999237060547), np.array([[0.0, 1.0], [20.0, 0.9], [60.0, 0.8], [120.0, 0.7]])


def steer_k(steer, v):
    return math.tan(steer * MAXST * float(np.interp(v * 3.6, CURVE[:, 0], CURVE[:, 1]))) / WB


def one(rid, mi, seed=2):
    RES = R.RUN / "arms"
    at = next((u / "attempts" / rid / str(json.loads((u / "done" / (rid + ".json")).read_text()).get("attempt", 1))
                         for u in sorted(RES.glob("v2-jfaD-s%d-*" % seed)) + sorted(RES.glob("v2-jclD-s%d-*" % seed)) if (u / "done" / (rid + ".json")).exists()))
    D, gd, psiD, turns = R.turn_geometry(np.array(json.load(open(at / "route.json"))["xy"]))
    T = next(t for t in turns if t["mi"] == mi)
    tk = [t for t in map(json.loads, open(at / "ticks.jsonl")) if "truth" in t]
    pl = [p for p in map(json.loads, open(at / "plans.jsonl")) if not p.get("warm")]
    t = np.array([x["t"] for x in tk])
    v = np.array([x["v"] for x in tk])
    tr = np.array([x["truth"][:2] for x in tk])
    yaw = np.unwrap(np.array([x["truth"][2] for x in tk]))
    near, dev = R.project(tr, D)
    s = gd[near] - T["g0"]
    ds = np.maximum(np.gradient(gd[near]), 1e-6)
    kact = np.convolve(np.gradient(yaw) / np.maximum(v * 0.05, 0.3), np.ones(3) / 3, mode="same")
    ksteer = np.array([steer_k(x["steer"], x["v"]) for x in tk])
    pt = np.array([p["t"] for p in pl])
    kd = np.array([pl[max(np.searchsorted(pt, tt, side="right") - 1, 0)]["act_k"] for tt in t])
    kc, prev = np.zeros(len(t)), 0.0
    for i in range(len(t)):
        for _ in range(5):
            prev = op_ctrl.clip_curvature(max(v[i], 0.0), prev, kd[i])
        kc[i] = prev
    from route_poly import curvature
    P, g = R.L.resample(np.array(json.load(open(at / "route.json"))["xy"]))
    kroute, gd = curvature(P, 0.5, 3.0), g
    return dict(s=s, dev=dev, kd=kd, kc=kc, ks=ksteer, ka=kact, v=v, T=T, sr=gd - T["g0"], kr=kroute, near=near, a0=T["it0"], i1=T["i1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="28180:0,24944:0,28008:0,24758:0")
    ap.add_argument("--out", default=str(HERE.parent / "figs/junction_forced_curvature.png"))
    a = ap.parse_args()
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    lines = []
    for ax, x in zip(axs.ravel(), a.turns.split(",")):
        rid, mi = x.split(":")
        d = one(rid, int(mi))
        T = d["T"]
        m = (d["near"] >= d["a0"] - 80) & (d["near"] <= d["i1"] + 60) & (d["dev"] < 30) & (d["v"] >= 1.0)   # below 1 m/s the steer-implied / path curvature is numerically meaningless
        mr = (d["sr"] > -20) & (d["sr"] < (T["g1"] - T["g0"]) + 15)
        ax.plot(d["sr"][mr], d["kr"][mr], color="#999999", lw=3, label="route (dense centreline)")
        ax.plot(d["s"][m], d["kd"][m], color="#0072B2", lw=1.8, label="desired: head act_k")
        ax.plot(d["s"][m], d["kc"][m], color="#D55E00", lw=1.5, ls="--", label="after clip_curvature (counterfactual)")
        ax.plot(d["s"][m], d["ks"][m], color="#009E73", lw=1.5, label="steer command as curvature (executed)")
        ax.plot(d["s"][m], d["ka"][m], color="#000000", lw=1.0, alpha=.6, label="car path curvature (truth)")
        lv = np.where(m & (d["dev"] > 1.75))[0]
        if len(lv):
            ax.axvline(d["s"][lv[0]], color="#CC79A7", lw=1, ls=":", label="car > 1.75 m off the route")
        ax.axvspan(0, T["g1"] - T["g0"], color="#eeeeee", zorder=0)
        ax.set_ylim(-0.25, 0.25)
        ax.set_title("route %s turn %d: %+.0f deg, R_min %.1f m (1/R_min = %.3f 1/m)" % (rid, int(mi), T["angle"], T["rmin"], 1 / T["rmin"]), fontsize=9)
        ax.set_xlabel("route arc length from the turn start (m)")
        ax.set_ylabel("curvature (1/m, right +)")
        ax.grid(alpha=.3)
        kk = d["kd"][m & (d["s"] > -5) & (d["s"] < T["g1"] - T["g0"] + 5)]
        kv = d["kr"][mr & (d["sr"] > 0) & (d["sr"] < T["g1"] - T["g0"])]
        sgn = np.sign(T["angle"])
        lines.append("- route %s (R_min %.1f m): required peak %.3f; head desired peak (in the turn window, same sign) %.3f, clipped %.3f, steer-implied %.3f, car %.3f; min speed in the window %.1f m/s" % (
            rid, T["rmin"], np.max(sgn * kv), np.max(sgn * kk) if len(kk) else np.nan, np.max(sgn * d["kc"][m & (d["s"] > -5) & (d["s"] < T["g1"] - T["g0"] + 5)]),
            np.max(sgn * d["ks"][m & (d["s"] > -5) & (d["s"] < T["g1"] - T["g0"] + 5)]), np.max(sgn * d["ka"][m & (d["s"] > -5) & (d["s"] < T["g1"] - T["g0"] + 5)]), d["v"][(d["near"] >= d["a0"] - 80) & (d["near"] <= d["i1"] + 60) & (d["dev"] < 30)].min()))
    axs.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
