#!/usr/bin/env python
"""Turn-onset read of B2D route attempts (experiments/b2d_monitor/lib/op_arb_viz_agent.py logs): where the car starts turning relative to the route's junction entry,
and what the plan was aiming at then. usage: rsg_turns.py <run dir> [<run dir> ...]

Per attempt: junction entry s_j = start of the first LEFT / RIGHT command run of the dense route (route.json cmd, RoadOption 1 / 2);
route onset = where the dense route's heading has turned 10 deg from its heading 30 m before s_j; car onset = where the car's true heading
(ticks.jsonl truth, rear axle) has turned 10 deg towards the turn from its heading when it was 30 m before s_j (car position = its projection
on the route). At the decision nearest the car onset: the plan's 2 s / 4 s poses (plans.jsonl poses) and the route point nearest to each
(arc offset ahead of the ego's projection) with the lateral miss. Event / DS from stop_event.json and results.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def read(att):
    R = json.load(open(att / "route.json"))
    xy, cmd = np.array(R["xy"]), np.array(R["cmd"])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    hd = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    tk = [json.loads(l) for l in open(att / "ticks.jsonl")]
    pl = [json.loads(l) for l in open(att / "plans.jsonl")] if (att / "plans.jsonl").exists() else []
    out = {"route": att.parent.name}
    turn = np.flatnonzero(np.isin(cmd, (1, 2)))
    if not len(turn):
        out["note"] = "no LEFT / RIGHT command"
        return out
    j0 = int(turn[0])
    sign = 1 if cmd[j0] == 2 else -1                     # CARLA yaw: a right turn increases it, a left turn decreases it
    out["turn"] = "right" if sign > 0 else "left"
    sj = s[j0]
    i30 = int(np.searchsorted(s, sj - 30.0))
    h0 = hd[i30]
    k = np.flatnonzero((sign * (hd - h0) > math.radians(10)) & (s > s[i30]))
    out["route_onset_m"] = round(float(s[k[0]] - sj), 1) if len(k) else None
    T = np.array([x["truth"][:3] for x in tk], float)
    t = np.array([x["t"] for x in tk])
    ci = np.array([int(np.argmin(np.hypot(*(xy - p[:2]).T))) for p in T])
    cs = s[ci]
    yaw = np.unwrap(T[:, 2])
    before = np.flatnonzero(cs <= sj - 30.0)
    ref = yaw[before[-1]] if len(before) else yaw[0]
    k = np.flatnonzero((sign * (yaw - ref) > math.radians(10)) & (np.arange(len(t)) > (before[-1] if len(before) else 0)))
    if not len(k):
        out["car_onset_m"] = None
    else:
        i = int(k[0])
        out["car_onset_m"], out["car_onset_t"] = round(float(cs[i] - sj), 1), round(float(t[i]), 2)
        lat = np.hypot(*(xy[ci[i]] - T[i, :2]))
        out["car_lat_at_onset_m"] = round(float(lat), 2)
        if pl:
            q = min(pl, key=lambda q: abs(q["t"] - t[i]))
            pose = np.array(q["poses"])
            yw, c, s_ = T[i, 2], math.cos(T[i, 2]), math.sin(T[i, 2])
            aim = {}
            for name, row in (("2s", 3), ("4s", 7)):
                lx, ly = pose[row, 0], pose[row, 1]                  # rear-axle frame, y left; CARLA world: inverse of world_to_local
                w = np.array([T[i, 0] + c * lx + s_ * ly, T[i, 1] + s_ * lx - c * ly])
                d = np.hypot(*(xy - w).T)
                m = int(np.argmin(d))
                aim[name] = {"plan_xy": [round(lx, 1), round(ly, 1)], "route_pt_ahead_m": round(float(s[m] - cs[i]), 1),
                             "route_pt_vs_junction_m": round(float(s[m] - sj), 1), "miss_m": round(float(d[m]), 2)}
            out["aim"] = aim
            out["cmd4_at_onset"] = q.get("cmd4")
    se = att / "stop_event.json"
    if se.exists():
        out["event"] = json.load(open(se))
    try:
        g = json.load(open(att / "results.json"))["_checkpoint"]["global_record"]
        out["DS"], out["RC"] = round(g["scores_mean"]["score_composed"], 1), round(g["scores_mean"]["score_route"], 1)
    except (KeyError, FileNotFoundError, TypeError):
        pass
    # where the route command (desire) switched to the turn for the model: first plan with cmd4 != straight
    cm = [q for q in pl if q.get("cmd4") and q["cmd4"][1] < 0.5]
    if cm:
        p0 = np.array(tk[int(np.argmin(np.abs(t - cm[0]["t"])))]["truth"][:2])
        out["cmd_on_m"] = round(float(s[int(np.argmin(np.hypot(*(xy - p0).T)))] - sj), 1)
    return out


for d in sys.argv[1:]:
    for att in sorted(Path(d).glob("attempts/*/1")):
        if (att / "route.json").exists() and (att / "ticks.jsonl").exists():
            print(Path(d).name, json.dumps(read(att)))
