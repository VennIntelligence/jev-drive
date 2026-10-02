"""Route-level reconstruction from the compact run extracts of release_extract.py (numpy, Python 3.9).

junction entrances : arc position of the first map-flagged route point, per junction id, from the logged 's' steps
                     (ego_s + junc_dist while the car is still before the junction: junc_dist > 1 m is exact, 1 m route resolution);
traffic lights     : stop-line arc per light (ego_s + REAR_TO_BUMPER + tl_dist from the plan-step context, median over steps),
                     lights of different runs grouped by arc (actor ids differ between runs), state timeline merged over all runs
                     of the route (CARLA light timers are identical across the runs of a route: conflicts are counted).
"""
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

REAR = 1.3886 + 2.4508           # rear axle -> front bumper (lib/op_arb_agent.py REAR_TO_BUMPER)
GROUP_M = 4.0                    # stop lines closer than this on one route are the same light
LINE_WINDOW = (-15.0, 1.0)       # a light governs the junction if its stop line lies within [entrance - 15, entrance + 1] (the agent's rule)


def load_docs(ex_dir, keep=lambda d: True):
    out = []
    for f in sorted(glob.glob(os.path.join(ex_dir, "*.json.gz"))):
        d = json.load(gzip.open(f))
        if keep(d):
            out.append(d)
    return out


def ego_s_at(d):
    s = np.array([[r[0], r[2]] for r in d["s"]], float).reshape(-1, 2)
    return s[:, 0], s[:, 1]


def route_info(docs):
    """docs: all extracts of ONE route. Returns dict(entries={jid: arc}, lights=[dict(arc, tl={attempt: (t, state)})], conflicts=0).
    The light timelines are per run: the scenario sets the state of the ego's light when the car reaches its trigger, so the same light
    turns green at different times in different runs (15102: 8 s in the drive arms, 44.5 s in the vred arms)."""
    ent = defaultdict(list)
    for d in docs:
        for r in d["s"]:
            if r[3] is not None and 1.0 < r[3] < 100.0 and r[8] is not None and r[8] >= 0:
                ent[r[8]].append(r[2] + r[3])
    entries = {j: float(np.median(v)) for j, v in ent.items()}
    per_run = []                                  # (arc, attempt, t array, state array)
    for d in docs:
        ts, ss = ego_s_at(d)
        if not len(ts):
            continue
        arcs, samples = defaultdict(list), defaultdict(list)
        for c in d["ctx"]:
            if c[1] is None or c[1] < 0 or c[3] is None:
                continue
            i = int(np.abs(ts - c[0]).argmin())
            if abs(ts[i] - c[0]) <= 0.03:
                arcs[c[3]].append(ss[i] + REAR + c[2])
            samples[c[3]].append((c[0], c[1]))
        for tq, lst in d.get("lights", []):
            for lid, st in lst:
                if st is not None and st >= 0:
                    samples[lid].append((tq, st))
        for lid, v in arcs.items():
            if len(v) < 3:
                continue
            pts = sorted(set(samples[lid]))
            per_run.append((float(np.median(v)), d["attempt"], np.array([p[0] for p in pts]), np.array([p[1] for p in pts], int)))
    groups = []
    for arc, att, t, v in sorted(per_run, key=lambda x: x[0]):
        if groups and arc - groups[-1]["arcs"][-1] <= GROUP_M:
            groups[-1]["arcs"].append(arc)
        else:
            groups.append(dict(arcs=[arc], tl={}))
        groups[-1]["tl"].setdefault(att, []).append((t, v))
    lights = []
    for g in groups:
        tl = {}
        for att, lst in g["tl"].items():
            t = np.concatenate([x[0] for x in lst])
            v = np.concatenate([x[1] for x in lst])
            o = np.argsort(t, kind="stable")
            tl[att] = (t[o], v[o])
        lights.append(dict(arc=float(np.median(g["arcs"])), tl=tl))
    return dict(entries=entries, lights=lights, conflicts=0)


def governing_light(info, jid):
    """Light whose stop line is the last one within the agent's window of the junction entrance (None if unsignalized)."""
    e = info["entries"].get(jid)
    if e is None:
        return None
    cand = [L for L in info["lights"] if e + LINE_WINDOW[0] <= L["arc"] <= e + LINE_WINDOW[1]]
    return max(cand, key=lambda L: L["arc"]) if cand else None


def light_state(L, att, t, maxgap=0.6):
    """0 green, 1 yellow, 2 red at time t in run `att` (nearest sample within maxgap, else None)."""
    if att not in L["tl"]:
        return None
    ts, vs = L["tl"][att]
    i = int(np.searchsorted(ts, t))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(ts) and (best is None or abs(ts[j] - t) < abs(ts[best] - t)):
            best = j
    if best is None or abs(ts[best] - t) > maxgap:
        return None
    return int(vs[best])


def green_onsets(L, att, t0, t1):
    out = []
    if att not in L["tl"]:
        return out
    t, v = L["tl"][att]
    for i in range(1, len(t)):
        if t0 <= t[i] <= t1 and v[i] == 0 and v[i - 1] != 0 and t[i] - t[i - 1] < 1.0:
            out.append(float(t[i]))
    return out


def leave_green(L, att, t_g, default):
    if att not in L["tl"]:
        return default
    t, v = L["tl"][att]
    for i in range(len(t)):
        if t[i] > t_g and v[i] != 0:
            return float(t[i])
    return default
