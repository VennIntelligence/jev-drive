"""How far is the official route's command change from the true junction entrance and the light's stop line? (numpy, Python 3.9)

Official input (Bench2Drive leaderboard/autoagents/autonomous_agent.py set_global_plan): the dense plan (one point per ~1 m, each with a road
command; kept by the leaderboard as global_plan_world_coord / _plan_gps_HACK) and the plan downsampled by route_manipulation.downsample_route(plan, 50).
Used here: the dense plan as logged by the agent (route.json of every attempt: xy and cmd), the downsample rule re-implemented
verbatim (2-D distances: the logs have no z), the junction entrances and stop lines reconstructed from the run logs (route_common.py).

  python3 route_junction_error.py   -> prints the tables; writes results/route_junction_error.csv
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_common as rc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EX = os.path.join(ROOT, "tmp/vlm_arb_offline/tmp_release_ex")
RES = os.path.join(ROOT, "experiments/vlm_arb/results")
LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHL, CHR = 1, 2, 3, 4, 5, 6
TURN = (LEFT, RIGHT, STRAIGHT)
NAME = {1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "LANEFOLLOW", 5: "CHANGELANELEFT", 6: "CHANGELANERIGHT"}


def downsample_ids(xy, cmd, sample_factor=50.0):
    """route_manipulation.downsample_route, same branch order."""
    ids, prev, dist = [], None, 0.0
    for i in range(len(cmd)):
        cur = cmd[i]
        if prev is None:
            ids.append(i)
            dist = 0.0
        elif cur in (CHL, CHR):
            ids.append(i)
            dist = 0.0
        elif prev != cur and prev not in (CHL, CHR):
            ids.append(i)
            dist = 0.0
        elif dist > sample_factor:
            ids.append(i)
            dist = 0.0
        elif i == len(cmd) - 1:
            ids.append(i)
            dist = 0.0
        else:
            dist += float(np.hypot(*(xy[i] - xy[i - 1])))
        prev = cur
    return ids


def runs_of(cmd, s):
    out, i = [], 0
    while i < len(cmd):
        j = i
        while j + 1 < len(cmd) and cmd[j + 1] == cmd[i]:
            j += 1
        out.append((int(cmd[i]), i, j, float(s[i]), float(s[j])))
        i = j + 1
    return out


def proj_arc(poly, arcs, q):
    """Arc position of the projection of point q on the polyline."""
    d = np.diff(poly, axis=0)
    ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    fr = np.clip(((q - poly[:-1]) * d).sum(1) / ln ** 2, 0, 1)
    cl = poly[:-1] + fr[:, None] * d
    k = int(np.argmin(np.hypot(*(cl - q).T)))
    return float(arcs[k] + fr[k] * ln[k])


def stats(x):
    x = np.array(x, float)
    if not len(x):
        return ["-"] * 9
    return [len(x), "%+.2f" % x.mean(), "%+.2f" % np.median(x), "%+.2f" % x.min(), "%+.2f" % x.max(), "%.2f" % np.abs(x).mean(), "%.2f" % np.percentile(np.abs(x), 90),
            "%d / %d" % (int((np.abs(x) <= 1).sum()), len(x)), "%d / %d" % (int((np.abs(x) <= 3).sum()), len(x))]


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(":--" if i == 0 else "--:" for i in range(len(head))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def main():
    docs = rc.load_docs(EX)
    by = defaultdict(list)
    for d in docs:
        by[d["route"]].append(d)
    rows, per_j, notes = [], [], []
    for route in sorted(by, key=int):
        v = by[route]
        rt = next(d["route_xy"] for d in v if d["route_xy"])
        for d in v:                                   # the plan is identical in every attempt of a route
            if d["route_xy"] and d["route_xy"]["cmd"] != rt["cmd"]:
                notes.append("route %s: plans differ between attempts" % route)
        xy, cmd = np.array(rt["xy"], float), np.array(rt["cmd"], int)
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        ids = downsample_ids(xy, cmd)
        s_ds = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy[ids], axis=0).T))]       # arc along the downsampled polyline
        info = rc.route_info(v)
        runs = runs_of(cmd, s)
        turns = [r for r in runs if r[0] in TURN]
        # junction exits from the 's' steps still inside (junc_dist in [-1, 0.01])
        exits = defaultdict(float)
        for d in v:
            for r in d["s"]:
                if r[3] is not None and -1.01 <= r[3] <= 0.01 and r[8] is not None and r[8] >= 0:
                    exits[r[8]] = max(exits[r[8]], r[2])
        covered = max((d["s"][-1][2] if d["s"] else 0.0) for d in v) + 100.0 >= s[-1]
        used = set()
        for jid, E in sorted(info["entries"].items(), key=lambda x: x[1]):
            L = rc.governing_light(info, jid)
            S = None if L is None else L["arc"]
            X = exits.get(jid)
            # the command run of this junction: a turn/straight run overlapping [E - 5, exit + 5] (exit unknown: E + 40), nearest start
            cand = [r for r in turns if r[4] >= E - 5.0 and r[3] <= (X if X else E + 40.0) + 5.0]
            cand.sort(key=lambda r: abs(r[3] - E))
            if not cand:
                per_j.append(dict(route=route, jid=jid, E=E, S=S, X=X, cmd=None))
                continue
            r = cand[0]
            used.add(r[1])
            # downsampled plan: first sample with this run's command at or after its start, and the sample before it
            k = next((n for n, i in enumerate(ids) if i >= r[1] and cmd[i] == r[0]), None)
            nxt = ids[k] if k is not None else None
            prv = ids[k - 1] if (k is not None and k > 0) else None
            rem = {}
            if k is not None:
                for o in (5, 10, 20, 30, 50):
                    if E - o >= 0:
                        i0 = int(np.argmin(np.abs(s - (E - o))))
                        a0 = proj_arc(xy[ids], s_ds, xy[i0])
                        rem[o] = float(s_ds[k] - a0) - (E - s[i0])            # remaining distance to the command sample on the downsampled plan, minus truth
            gaps = np.diff(s)
            per_j.append(dict(route=route, jid=jid, E=E, S=S, X=X, cmd=r[0], C=r[3], Cend=r[4], rem=rem, maxgap=float(gaps[max(r[1] - 2, 0):r[1] + 2].max()),
                              C_ds=None if nxt is None else float(s[nxt]), arc_ds=None if nxt is None else float(s_ds[k]),
                              prev_ds=None if prv is None else float(s[prv]), n_ds=len(ids)))
        free = [r for r in turns if r[1] not in used]
        rows.append(dict(route=route, length=s[-1], covered=covered, n_dense=len(xy), n_ds=len(ids), runs=runs, ds_ids=ids, junctions=len(info["entries"]),
                         free_turns=[(NAME[r[0]], round(r[3], 1), round(r[4], 1)) for r in free], lights=[round(L["arc"], 1) for L in info["lights"]],
                         lane_changes=[(NAME[r[0]], round(r[3], 1), round(r[4], 1)) for r in runs if r[0] in (CHL, CHR)]))
    # ---- tables
    out = []
    head = ["route", "plan length m", "dense pts", "downsampled pts", "junctions (map flag)", "turn / straight runs (cmd, start..end m)", "lane-change runs", "turn runs with no flagged junction"]
    out.append(md([[r["route"], "%.0f" % r["length"], r["n_dense"], r["n_ds"], r["junctions"],
                    "; ".join("%s %.1f..%.1f" % (NAME[x[0]], x[3], x[4]) for x in r["runs"] if x[0] in TURN) or "-",
                    "; ".join("%s %.1f..%.1f" % x for x in r["lane_changes"]) or "-",
                    "; ".join("%s %.1f..%.1f" % x for x in r["free_turns"]) or "-"] for r in rows], head))
    out.append("\n")
    jr = []
    for j in per_j:
        if j["cmd"] is None:
            jr.append([j["route"], j["jid"], "%.1f" % j["E"], "-" if j["S"] is None else "%.1f" % j["S"], "none", "-", "-", "-", "-", "-", "-"])
        else:
            jr.append([j["route"], j["jid"], "%.1f" % j["E"], "-" if j["S"] is None else "%.1f" % j["S"], NAME[j["cmd"]], "%.1f" % j["C"],
                       "%+.1f" % (j["C"] - j["E"]), "-" if j["S"] is None else "%+.1f" % (j["C"] - j["S"]), "%.1f" % j["arc_ds"], "%+.1f" % (j["arc_ds"] - j["E"]),
                       "-" if j["prev_ds"] is None else "%.1f" % (j["C_ds"] - j["prev_ds"])])
    out.append(md(jr, ["route", "junction id", "true entrance s m", "stop line s m", "command", "dense: command start s m", "dense error vs entrance m", "dense error vs stop line m",
                       "downsampled: arc of first new-command sample m", "downsampled error vs entrance m", "gap to previous sample m"]))
    out.append("\n")
    have = [j for j in per_j if j["cmd"] is not None]
    sig = [j for j in have if j["S"] is not None]
    srows = [["dense plan vs entrance"] + stats([j["C"] - j["E"] for j in have]),
             ["downsampled plan vs entrance"] + stats([j["arc_ds"] - j["E"] for j in have]),
             ["dense plan vs stop line"] + stats([j["C"] - j["S"] for j in sig]),
             ["downsampled plan vs stop line"] + stats([j["arc_ds"] - j["S"] for j in sig]),
             ["stop line vs entrance (S - E)"] + stats([j["S"] - j["E"] for j in sig])]
    out.append(md(srows, ["quantity (signed, command position minus truth)", "n", "mean", "median", "min", "max", "mean |e|", "p90 |e|", "n within 1 m", "n within 3 m"]))
    out.append("\nRemaining distance to the junction as read from the downsampled plan (chord length from the car's projection to the first new-command sample) minus the true distance to the entrance, signed, by distance of the car before the entrance:\n")
    rr = []
    for o in (5, 10, 20, 30, 50):
        vals = [j["rem"][o] for j in have if o in j["rem"]]
        rr.append(["%d m" % o] + stats(vals))
    out.append(md(rr, ["car before the entrance", "n", "mean", "median", "min", "max", "mean |e|", "p90 |e|", "n within 1 m", "n within 3 m"]))
    out.append("\nmaximum dense point spacing next to the command change: %.1f m (median %.1f m)" % (max(j["maxgap"] for j in have), float(np.median([j["maxgap"] for j in have]))))
    out.append("\njunctions without a command change: %d of %d; junctions with a signal: %d" % (len(per_j) - len(have), len(per_j), len(sig)))
    out.append("junctions per route covered by the logs: " + ", ".join("%s%s" % (r["route"], "" if r["covered"] else "*") for r in rows) + "  (* = the logged ego never got within 100 m of the plan's end)")
    for n in notes:
        out.append(n)
    print("\n".join(out))
    with open(os.path.join(RES, "route_junction_error.csv"), "w") as fh:
        keys = ["route", "jid", "E", "S", "X", "cmd", "C", "Cend", "C_ds", "arc_ds", "prev_ds", "n_ds"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for j in per_j:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in j.items()})


if __name__ == "__main__":
    main()
