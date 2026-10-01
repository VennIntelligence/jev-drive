#!/usr/bin/env python3
"""op-drive: classify every collision and every stop-then-go event of the op-drive arms from the CARLA event records.

Collisions: results.json lists each counted collision with the other actor's type and the EGO position at impact (leaderboard
CollisionTest; a collision at ego speed < 0.1 m/s is not counted at all, so there is no "ego stationary" case in the records; anything counted
has the ego moving, and the score does not ask whose fault it was). Each one is matched to the ego tick whose truth position is nearest
(among ticks with v >= 0.1) and classified from the ego's own state and ground-truth context around it (plans.jsonl ctx): the other
actor's pose is not logged in these runs, so a side / rear hit is inferred, not measured.

  ped      walker
  static   layout / static object
  lead     vehicle, a route-aligned lead within 4 m at impact (the ego ran into its lead)
  creep    vehicle, ego speed < 1.0 m/s at impact and no lead (ego moving off / creeping when hit or touching)
  junction vehicle, ego in a junction or in a route-command zone (LEFT / RIGHT / STRAIGHT), no lead: cross or turning traffic
             sub-labels: launch (v rose > 1 m/s over the previous 3 s), through (steady) or braking
  other    vehicle on a lane-follow segment with no lead in the route corridor: a vehicle from the side / behind, or a cut-in

Stop-then-go events: a standstill (v < 0.3 m/s for >= 2 s after warm-up) followed by v > 1 m/s; labelled by ground truth at the start:
  red      a red / yellow light within 40 m ahead of the bumper, and how the start was released (plan / lead / timer resume / other)
  other    everything else.
    python3 experiments/op_closed_loop/archive/op_drive_collisions.py [--root DIR] [--tags dv d1 r0 r1 r2 f] [--arms ...] [--csv OUT_PREFIX]"""
import argparse
import glob
import json
import math
import os
import re
from collections import Counter

MSG = re.compile(r"type=(\S+) and id=(\d+) at \(x=(-?[\d.]+), y=(-?[\d.]+)")
KEYS = {"collisions_vehicle": "veh", "collisions_pedestrian": "ped", "collisions_layout": "layout"}


def attempts(arm_dir):
    for f in sorted(glob.glob(os.path.join(arm_dir, "done", "*.json"))):
        rid = os.path.basename(f)[:-5]
        yield rid, os.path.join(arm_dir, "attempts", rid, str(json.load(open(f))["attempt"]))


def jl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def collisions(adir):
    res = json.load(open(os.path.join(adir, "results.json")))
    rec = res["_checkpoint"]["records"][0]
    out = []
    for k, kind in KEYS.items():
        for m in rec["infractions"].get(k, []):
            g = MSG.search(m)
            if g:
                out.append((kind, g.group(1), int(g.group(2)), float(g.group(3)), float(g.group(4))))
    return out


def classify(kind, tid, tick, ticks, plan):
    if kind == "ped":
        return "ped"
    if kind == "layout":
        return "static"
    ctx = (plan or {}).get("ctx", tick.get("ctx", {}))
    v = tick["v"]
    if ctx.get("lead_gap", 99) <= 4.0:
        return "lead"
    if v < 1.0:
        return "creep"
    zone = (plan or {}).get("zone") or ctx.get("junc")
    dv = v - min((x["v"] for x in ticks if tick["t"] - 3 <= x["t"] <= tick["t"]), default=v)
    if zone:
        return "junction/" + ("launch" if dv > 1.0 else "through")
    return "other"


def events(rs, warm_skip=True):
    out, i, n = [], 0, len(rs)
    while i < n:
        if rs[i]["warm"] or rs[i]["v"] >= 0.3:
            i += 1
            continue
        j = i
        while j < n and rs[j]["v"] < 0.3 and not rs[j]["warm"]:
            j += 1
        if j < n and rs[j]["t"] - rs[i]["t"] >= 2.0:       # a standstill of >= 2 s ends with a start
            k = j
            while k < n and rs[k]["v"] <= 1.0:
                k += 1
            if k < n:
                r = rs[i:j]
                c = [x.get("ctx", {}) for x in r]
                red = sum(1 for x in c if x.get("tl") in (1, 2) and x.get("tl_dist", 99) < 40) > 0.5 * len(c)
                still_red = rs[k].get("ctx", {}).get("tl") in (1, 2) and rs[k]["ctx"].get("tl_dist", 99) < 40
                rel = next((x["rel"] for x in rs[max(i - 40, 0):k + 1] if isinstance(x.get("rel"), str)), None)
                out.append({"t": rs[i]["t"] - rs[0]["t"], "dur": rs[j]["t"] - rs[i]["t"], "red": red, "on_red": bool(red and still_red),
                            "rel": rel})
        i = max(j, i + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ["DATA_DIR"], "runs", "op_drive", "arms"))
    ap.add_argument("--tags", nargs="+", default=["dv"])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    crow, srow = [], []
    for tag in a.tags:
        for d in sorted(glob.glob(os.path.join(a.root, f"{tag}-*-s*"))):
            arm, seed = os.path.basename(d)[len(tag) + 1:].rsplit("-s", 1)
            if a.arms and arm not in a.arms:
                continue
            for rid, adir in attempts(d):
                ticks, plans = jl(os.path.join(adir, "ticks.jsonl")), jl(os.path.join(adir, "plans.jsonl"))
                by_frame = {p["frame"]: p for p in plans}
                for kind, tid, oid, x, y in collisions(adir):
                    cand = [t for t in ticks if t["v"] >= 0.1 and t.get("truth")]
                    if not cand:
                        continue
                    tk = min(cand, key=lambda t: math.hypot(t["truth"][0] - x, t["truth"][1] - y))
                    pl = by_frame.get(tk["frame"])
                    ctx = (pl or {}).get("ctx", tk.get("ctx", {}))
                    crow.append({"tag": tag, "arm": arm, "seed": int(seed), "route": rid, "kind": kind, "other": tid, "t": round(tk["t"] - ticks[0]["t"], 1),
                                 "v": round(tk["v"], 2), "cls": classify(kind, tid, tk, ticks, pl), "junc": ctx.get("junc"), "lead_gap": ctx.get("lead_gap"),
                                 "tl": ctx.get("tl"), "src": (pl or {}).get("src"), "cmd": ((pl or {}).get("cmd") or [None])[0]})
                for e in events(plans):
                    srow.append({"tag": tag, "arm": arm, "seed": int(seed), "route": rid, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in e.items()}})
    print("collisions per arm and class")
    cnt = Counter((r["tag"], r["arm"], r["cls"]) for r in crow)
    for k in sorted(cnt):
        print(" ", *k, cnt[k])
    print("\nstop-then-go events per arm: total / at a red light / started while still red / released by timer resume")
    ev = {}
    for r in srow:
        e = ev.setdefault((r["tag"], r["arm"]), [0, 0, 0, 0])
        e[0] += 1; e[1] += r["red"]; e[2] += r["on_red"]; e[3] += r["rel"] == "timeout"
    for k in sorted(ev):
        print(" ", *k, *ev[k])
    if a.csv:
        import csv
        for name, rows in (("collisions", crow), ("stops", srow)):
            if rows:
                with open(f"{a.csv}_{name}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)


if __name__ == "__main__":
    main()
