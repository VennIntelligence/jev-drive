#!/usr/bin/env python3
"""STEER-LONG offline MPC replay (research/next-round/steer-long.md amendment 4): re-solve AlpaSim's own nonlinear MPC (run as shipped, the
alpasim-base image the PAI stacks used) on the logged states and the rebuilt served reference, step by step, around the sudden steering
jumps at speed; record whether the replay reproduces the logged command, the IPOPT return status and iterations at every step, and the
command under alternative gains / iteration limits on the same states (open loop: the logged states are not changed by the arm).

On the Tokyo box, inside the image (do_mpc, casadi, alpasim_controller, alpasim_utils):
  docker run --rm -v /data:/data -v <repo>:/jev alpasim-base:0.89.0 bash -c \
    'cd /repo && uv run python /jev/experiments/steer_long/scripts/sl_mpc.py --pull /data/runs/steer_long/pai --events <events.json> --out <out.json>'
events.json: [{"label", "session", "t_on", "kind"}], t_on = the step to centre on (us). Arms: name -> gains / max_iter.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict

import numpy as np

ARMS = {"default": ({}, 30), "w_steer20": ({"rel_front_steering_angle_weight": 20.0}, 30),
        "w_steer50": ({"rel_front_steering_angle_weight": 50.0}, 30), "iter200": ({}, 200)}


def yaw_q(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def load(pull, label, sessions):
    D = defaultdict(list)
    for line in gzip.open(f"{pull}/{label}/drive.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["session"] in sessions:
            D[r["session"]].append(r)
    C = defaultdict(list)
    with gzip.open(f"{pull}/{label}/ctrl.csv.gz", "rt") as f:
        rd = csv.reader(f)
        head = next(rd)
        for row in rd:
            if row[0] in sessions:
                C[row[0]].append([float(x) for x in row[1:]])
    cols = head[1:]
    return D, {s: {c: np.array([r[i] for r in v]) for i, c in enumerate(cols)} for s, v in C.items()}


def served_local(rec):
    """build_trajectory_from_plan on PAI (now == t0): 0.1 s samples 0 .. 4 s of anchor + 8 points, local frame."""
    ax, ay, a0 = rec["anchor"]
    P = np.asarray(rec["poses"], float)
    c, s = math.cos(a0), math.sin(a0)
    X = np.r_[ax, ax + c * P[:, 0] - s * P[:, 1]]
    Y = np.r_[ay, ay + s * P[:, 0] + c * P[:, 1]]
    W = np.unwrap(np.r_[a0, a0 + P[:, 2]])
    tp = np.arange(9) * 0.5
    g = np.arange(41) * 0.1
    return rec["t0"] + np.round(g * 1e6).astype(np.int64), np.interp(g, tp, X), np.interp(g, tp, Y), np.interp(g, tp, W)


def main():
    from alpasim_controller.mpc_controller import ControllerInput, MPCGains
    from alpasim_controller.mpc_impl.nonlinear_mpc import NonlinearMPC
    from alpasim_utils.geometry import Pose, Trajectory
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--before", type=float, default=1.5)
    ap.add_argument("--after", type=float, default=0.4)
    a = ap.parse_args()
    evs = json.load(open(a.events))
    by = defaultdict(list)
    for e in evs:
        by[e["label"]].append(e)
    out = []
    for label, es in by.items():
        D, C = load(a.pull, label, {e["session"] for e in es})
        for e in es:
            recs = sorted(D[e["session"]], key=lambda r: r["now"])
            now = np.array([r["now"] for r in recs], np.int64)
            c = C[e["session"]]
            ts = c["timestamp_us"].astype(np.int64)
            steps = [int(t) for t in ts if e["t_on"] - a.before * 1e6 <= t <= e["t_on"] + a.after * 1e6]
            res = {"label": label, "session": e["session"], "t_on": e["t_on"], "kind": e.get("kind"), "steps": steps, "u_log": [], "arms": {}}
            for t in steps:
                i = int(np.searchsorted(ts, t))
                res["u_log"].append(float(c["u_steering_angle"][i + 1]) if i + 1 < len(ts) else None)   # command solved at t, logged at t + 0.1 s
            for arm, (g, it) in ARMS.items():
                mpc = NonlinearMPC(gains=MPCGains(**g))
                us, st, itc = [], [], []
                for t in steps:
                    i = int(np.searchsorted(ts, t))
                    j = int(np.searchsorted(now, t, side="right")) - 1
                    tt, X, Y, W = served_local(recs[j])
                    x0, y0, w0 = c["x"][i], c["y"][i], yaw_q(c["qx"][i], c["qy"][i], c["qz"][i], c["qw"][i])
                    cc, ss = math.cos(w0), math.sin(w0)
                    keep = tt >= recs[j]["now"]
                    poses = [Pose(np.array([cc * (x - x0) + ss * (y - y0), -ss * (x - x0) + cc * (y - y0), 0.0], np.float32),
                                  np.array([0, 0, math.sin((w - w0) / 2), math.cos((w - w0) / 2)], np.float32))
                             for x, y, w in zip(X[keep], Y[keep], W[keep])]
                    ref = Trajectory.from_poses(timestamps=tt[keep].astype(np.uint64), poses=poses)
                    state = np.array([0, 0, 0, c["vx"][i], c["vy"][i], c["wz"][i], c["front_steering_angle"][i], c["acceleration"][i]], float)
                    if mpc._mpc is None:
                        mpc._current_reference = ControllerInput(state=state, reference_trajectory=ref, timestamp_us=t)
                        mpc._setup_mpc()
                        if it != 30:
                            mpc._mpc.settings.nlpsol_opts = {"ipopt.max_iter": it}
                            mpc._mpc.setup()
                        mpc._mpc.x0 = state.reshape(-1, 1)
                        mpc._mpc.set_initial_guess()
                    o = mpc.compute_control(ControllerInput(state=state, reference_trajectory=ref, timestamp_us=t))
                    ss_ = getattr(mpc._mpc, "solver_stats", {}) or {}
                    us.append(float(o.control[0]))
                    st.append(str(ss_.get("return_status", o.status)))
                    itc.append(int(ss_.get("iter_count", -1)))
                res["arms"][arm] = {"u": us, "status": st, "iter": itc}
            out.append(res)
            print(label, e["session"][:8], e.get("kind"), "steps", len(steps), {k: (round(max(map(abs, v["u"])), 3), v["status"].count("Solve_Succeeded"))
                                                                               for k, v in res["arms"].items()}, flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
