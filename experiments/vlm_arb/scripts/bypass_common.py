"""Shared helpers for the offline bypass analyses (loads the compact extracts of bypass_extract.py; numpy, Python 3.9)."""
import glob
import gzip
import json
import os

import numpy as np

TARGET = ("24497", "2520", "19324", "19832")      # static-obstacle routes
OBST_TYPE = {"24497": "ConstructionObstacle", "2520": "ConstructionObstacle", "19324": "Accident", "19832": "ParkedObstacle"}
STATIC_V = 0.2                                      # m/s, as PARAMS["static_speed"]
PLAN_COLS = ["t", "v", "warm", "tl", "tl_dist", "tl_id", "junc", "lead_gap", "lead_v", "lead0", "lp", "bypass", "src", "lat", "lat_why"]


def load_all(ex_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(ex_dir, "*.json.gz"))):
        d = json.load(gzip.open(f))
        out[(d["arm"], d["seed"], d["route"])] = d
    return out


def route_s(xy):
    xy = np.asarray(xy, float)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def project(points, xy):
    """Arc position and signed lateral offset of points on the polyline xy (nearest segment)."""
    xy = np.asarray(xy, float)
    pts = np.asarray(points, float).reshape(-1, 2)
    delta = np.diff(xy, axis=0)
    ln = np.linalg.norm(delta, axis=1)
    keep = ln > 1e-6
    a, dl, ll = xy[:-1][keep], delta[keep], ln[keep]
    arc = route_s(xy)[:-1][keep]
    w = pts[:, None] - a[None]
    fr = np.clip(np.einsum("nki,ki->nk", w, dl) / ll ** 2, 0, 1)
    cl = a[None] + fr[..., None] * dl[None]
    d2 = ((pts[:, None] - cl) ** 2).sum(-1)
    k = d2.argmin(1)
    j = np.arange(len(pts))
    unit = dl[k] / ll[k, None]
    nrm = np.stack([-unit[:, 1], unit[:, 0]], -1)
    lat = ((pts - cl[j, k]) * nrm).sum(-1)
    return arc[k] + fr[j, k] * ll[k], lat


def junction_table(runs, route):
    """Junction entries and inside-intervals on a route's arc, reconstructed from drive runs' 's' lines
    (ego_s, junc_dist: distance to the next junction entrance, within [-1, 0] while inside one, 999 if none within 100 m).
    Entry resolution is exact (ego_s + junc_dist), exit resolution is the ego's sampling step."""
    entries, inside = [], []
    for (arm, seed, r), d in runs.items():
        if r != route or arm != "drive":
            continue
        for e in d["vlm"]:
            if e["k"] != "s" or e["junc_dist"] is None:
                continue
            if 0 < e["junc_dist"] < 100:
                entries.append(e["ego_s"] + e["junc_dist"])
            elif -1.01 <= e["junc_dist"] <= 0.01:
                inside.append(e["ego_s"])
    entries = sorted(entries)
    ent = []
    for x in entries:
        if not ent or x - ent[-1][-1] > 3:
            ent.append([x])
        else:
            ent[-1].append(x)
    ent = [float(np.median(g)) for g in ent]
    ins = sorted(set(round(x, 1) for x in inside))
    iv = []
    for x in ins:
        if iv and x - iv[-1][1] <= 6:
            iv[-1][1] = x
        else:
            iv.append([x, x])
    return ent, iv


def next_entry(entries, s):
    """Distance from arc position s to the next junction entrance ahead (inf if none)."""
    ahead = [e - s for e in entries if e - s >= -1.0]
    return min(ahead) if ahead else float("inf")


def in_junction(intervals, s, pad=1.0):
    return any(a - pad <= s <= b + pad for a, b in intervals)


def plan_arr(d):
    """Plan-step signals as a dict of numpy arrays (None -> nan)."""
    rows = d["plans"]
    out = {}
    for i, c in enumerate(d["plan_cols"]):
        if c in ("t", "v", "warm", "tl", "tl_dist", "junc", "lead_gap", "lead_v", "bypass"):
            out[c] = np.array([np.nan if r[i] is None else r[i] for r in rows], float)
    out["lead0"] = [r[d["plan_cols"].index("lead0")] for r in rows]
    out["lp"] = [r[d["plan_cols"].index("lp")] for r in rows]
    return out


def actor_tracks(d):
    """id -> dict(type, t, x, y, v, stat) arrays over the 0.2 s privileged snapshots where the actor was within 70 m."""
    tr = {}
    for p in d["priv"]:
        for a in p["acts"]:
            k = tr.setdefault(a[0], dict(type=a[1], t=[], x=[], y=[], v=[], stat=[], gap=[], yaw=[], ext=(a[7], a[8])))
            k["t"].append(p["t"])
            k["x"].append(a[2])
            k["y"].append(a[3])
            k["v"].append(float(np.hypot(a[4], a[5])))
            k["stat"].append(a[9])
            k["gap"].append(a[10])
            k["yaw"].append(a[6])
    for k in tr.values():
        for n in ("t", "x", "y", "v", "stat", "gap", "yaw"):
            k[n] = np.array(k[n], float)
    return tr


def route_lights(runs, route):
    """Traffic lights the ego met on a route, reconstructed from every run's ego ctx: id-independent stop positions on the
    route arc (ego_s + tl_dist) and a time -> state function (CARLA light timers are identical across runs of a route:
    transition times agree to 0.1 s in the logs). Returns list of dict(s_stop, t, state) sorted by s_stop."""
    pts = []
    for (arm, seed, r), d in runs.items():
        if r != route:
            continue
        ci = {c: i for i, c in enumerate(d["plan_cols"])}
        pt, ps = ego_s_series(d)
        for row in d["plans"]:
            tl, td, tid = row[ci["tl"]], row[ci["tl_dist"]], row[ci["tl_id"]]
            if tl is None or td is None or not len(pt):
                continue
            es = float(np.interp(row[0], pt, ps))
            pts.append((es + td, row[0], tl, (arm, seed, tid)))
    if not pts:
        return []
    pts.sort()
    groups = []
    for s, t, tl, key in pts:
        for g in groups:
            if abs(np.median(g["s"]) - s) < 12:
                g["s"].append(s); g["t"].append(t); g["st"].append(tl); break
        else:
            groups.append(dict(s=[s], t=[t], st=[tl]))
    out = []
    for g in groups:
        if len(g["s"]) < 3:
            continue
        o = np.argsort(g["t"])
        out.append(dict(s_stop=float(np.median(g["s"])), t=np.array(g["t"])[o], state=np.array(g["st"])[o]))
    return sorted(out, key=lambda x: x["s_stop"])


def light_state_at(light, t):
    """State (0 green, 1 yellow, 2 red) of a reconstructed light at time t, nearest logged sample (None if > 3 s away)."""
    i = int(np.abs(light["t"] - t).argmin())
    return int(light["state"][i]) if abs(light["t"][i] - t) <= 3.0 else None


def ego_s_series(d):
    """(t, ego arc position) from the privileged snapshots (0.2 s) or, when a run has none, from the logged 's' lines (0.5 s)."""
    if d["priv"]:
        return np.array([p["t"] for p in d["priv"]]), np.array([p["ego_s"] for p in d["priv"]], float)
    ss = [e for e in d["vlm"] if e["k"] == "s"]
    ss = sorted({e["t"]: e["ego_s"] for e in ss}.items())
    return np.array([a for a, _ in ss], float), np.array([b for _, b in ss], float)
