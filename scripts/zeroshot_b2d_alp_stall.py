#!/usr/bin/env python3
"""Stall diagnosis of the Alpamayo zero-shot Bench2Drive runs (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md).

Reads the per-route logs of one or more b2d_run output dirs (plans.jsonl: every 2 Hz plan with the Zoo PID metadata and the
CoC text; ticks.jsonl: every 20 Hz control; results.json: official scores; route.json: route commands) and writes

  routes.csv   one row per route: status, DS, RC, ticks, time at v < 0.5 m/s, stall episodes by where they start
  plans.csv    one row per plan: speed, plan distance at 1 / 3 / 5 s, reverse flag, Zoo PID desired speed / brake / throttle,
               stall class, where, CoC keywords
  cot.json     CoC texts of stalled plans, counted

Stall classes of a plan made while v < 0.5 m/s:
  stay        the model plans to stay: forward distance at 3 s < 1.2 m (0.4 m/s, the Zoo PID brake threshold, over 3 s)
  go_braked   the model plans to go (>= 1.2 m at 3 s) but the controller brakes
  go_throttle the model plans to go and the controller gives throttle
  reverse     the plan moves backwards by > 0.3 m before 3 s (the controller reads |displacement| as speed)
Python 3.8, standard library + NumPy only.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

LEFT, RIGHT, STRAIGHT = 1, 2, 3
V_STALL = 0.5
KEYS = {"red_light": r"red light|traffic light|signal", "lead": r"lead vehicle|vehicle ahead|car ahead|lead car|queue|traffic ahead|stopped vehicle",
        "ped": r"pedestrian|cyclist|bicycl", "yield": r"yield|cross traffic|crossing|oncoming|intersection|junction",
        "stop_sign": r"stop sign", "go": r"accelerat|proceed|resume|go straight|keep driving|continue"}


def load_jsonl(p):
    out = []
    with open(p) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def where(route_i, cmd, s, near_m=15.0):
    """Route command context at route index i: 'junction' when inside or within near_m of a LEFT/RIGHT/STRAIGHT command."""
    i = min(route_i, len(cmd) - 1)
    ahead = np.nonzero(np.isin(cmd[i:], [LEFT, RIGHT, STRAIGHT]))[0]
    if len(ahead) and s[i + ahead[0]] - s[i] <= near_m:
        return "junction"
    return "lane"


def keywords(cot):
    c = (cot or "").lower()
    return [k for k, rx in KEYS.items() if re.search(rx, c)]


def analyse(attempt, run):
    ticks, plans = load_jsonl(attempt / "ticks.jsonl"), load_jsonl(attempt / "plans.jsonl")
    if not ticks or not plans:
        return None, []
    rec = json.loads((attempt / "results.json").read_text())["_checkpoint"]["records"][0]
    route = json.loads((attempt / "route.json").read_text()) if (attempt / "route.json").exists() else None
    cmd = np.array(route["cmd"]) if route else np.zeros(1, int)
    xy = np.array(route["xy"]) if route else np.zeros((1, 2))
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    v = np.array([t["v"] for t in ticks])
    t = np.array([t["t"] for t in ticks])
    moved = np.nonzero(v >= V_STALL)[0]
    t_first_move = float(t[moved[0]]) if len(moved) else float("inf")
    rows = []
    for p in plans:
        path = np.array(p["path"])
        x1, x3, x5 = path[3, 0], path[11, 0], path[19, 0]
        zoo = p.get("zoo_pid") or {}
        rev = float(path[:12, 0].min()) < -0.3
        stalled = p["speed"] < V_STALL
        if not stalled:
            klass = "moving"
        elif rev:
            klass = "reverse"
        elif x3 < 1.2:
            klass = "stay"
        elif zoo.get("brake", 0) > 0:
            klass = "go_braked"
        else:
            klass = "go_throttle"
        phase = "start" if p["t"] < t_first_move else where(p.get("route_index", 0), cmd, s)
        rows.append({"run": run, "route": rec["route_id"].split("_")[1], "t": round(p["t"], 2), "speed": round(p["speed"], 3),
                     "x1": x1, "x3": x3, "x5": x5, "y3": path[11, 1], "xmin": float(path[:, 0].min()),
                     "desired": zoo.get("desired_speed"), "brake": zoo.get("brake"), "throttle": zoo.get("throttle"),
                     "steer": zoo.get("steer"), "class": klass, "where": phase, "nav": p.get("nav_text") or "",
                     "kw": "|".join(keywords(p.get("cot"))), "cot": p.get("cot", "")})
    # stall episodes on ticks
    st = v < V_STALL
    edges = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
    eps = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if (b - a) * 0.05 >= 2.0]   # >= 2 s
    sc = rec["scores"]
    inf = rec["infractions"]
    row = {"run": run, "route": rec["route_id"].split("_")[1], "scenario": rec["scenario_name"].rsplit("_", 1)[0],
           "town": rec.get("town_name", ""), "status": rec["status"], "DS": sc["score_composed"], "RC": sc["score_route"],
           "ticks": len(ticks), "sim_s": round(float(t[-1]), 1), "stall_frac": round(float(st.mean()), 3),
           "stall_s": round(float(st.sum()) * 0.05, 1), "t_first_move": round(t_first_move, 1),
           "n_eps": len(eps), "longest_ep_s": round(max([(b - a) * 0.05 for a, b in eps], default=0.0), 1),
           "final_stall_s": round(float((len(st) - eps[-1][0]) * 0.05) if eps and eps[-1][1] == len(st) else 0.0, 1),
           "v_mean_moving": round(float(v[~st].mean()) if (~st).any() else 0.0, 2),
           "collisions": sum(len(inf.get(k, [])) for k in ("collisions_layout", "collisions_vehicle", "collisions_pedestrian")),
           "red_light": len(inf.get("red_light", [])), "blocked": len(inf.get("vehicle_blocked", [])),
           "timeout": len(inf.get("route_timeout", [])) + int("TickRuntime" in rec["status"] or "tick" in rec["status"].lower())}
    pr = [r for r in rows if r["class"] != "moving"]
    for k in ("stay", "go_braked", "go_throttle", "reverse"):
        row["n_" + k] = sum(r["class"] == k for r in pr)
    for w in ("start", "junction", "lane"):
        row["stallplans_" + w] = sum(r["where"] == w for r in pr)
    return row, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="b2d_run output dirs (label=path or path)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    routes, plans = [], []
    for spec in a.runs:
        label, _, path = spec.rpartition("=")
        path = Path(path)
        label = label or path.name
        for rdir in sorted((path / "attempts").iterdir()):
            atts = sorted((d for d in rdir.iterdir() if (d / "results.json").exists()), key=lambda d: int(d.name))
            if not atts:
                continue
            try:
                row, rows = analyse(atts[-1], label)
            except (KeyError, IndexError, ValueError) as exc:
                print("skip", atts[-1], exc)
                continue
            if row:
                routes.append(row)
                plans += rows
    for name, rows in (("routes.csv", routes), ("plans.csv", plans)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    cots = {}
    for label in sorted({r["run"] for r in plans}):
        c = Counter(r["cot"] for r in plans if r["run"] == label and r["class"] != "moving")
        cots[label] = c.most_common(60)
    (out / "cot.json").write_text(json.dumps(cots, indent=1))
    print("routes", len(routes), "plans", len(plans), "->", out)


if __name__ == "__main__":
    main()
