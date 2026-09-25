"""Per-phase summary of zero-shot Bench2Drive runs with junction outcomes (b2d_zeroshot_agent.py logs route.json and the
truth rear-axle pose per tick).

A junction event is a maximal run of LEFT / RIGHT (or STRAIGHT) route commands. It is reached when the ego's truth
track comes within 5 m of the run's first point, and passed when, after that, the track comes within 3 m of the route
point 10 m past the run's end (or of the run's end when the route ends there). Pass rate = passed / reached.

    python3 scripts/zeroshot_b2d_junctions.py RUN_DIR [RUN_DIR ...] [--csv out.csv]
"""
import argparse, csv, json, re, sys
from pathlib import Path

import numpy as np

LEFT, RIGHT, STRAIGHT = 1, 2, 3


def events(route):
    xy, cmd = np.asarray(route["xy"]), np.asarray(route["cmd"])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] in (LEFT, RIGHT, STRAIGHT):
            j = i
            while j + 1 < len(cmd) and cmd[j + 1] == cmd[i]:
                j += 1
            k = int(np.searchsorted(s, s[j] + 10.0))
            out.append((int(cmd[i]), xy[i], xy[min(k, len(xy) - 1)]))
            i = j + 1
        else:
            i += 1
    return out


def attempt_summary(att):
    res = json.loads((att / "results.json").read_text())["_checkpoint"]["records"][0]
    row = dict(route=att.parent.name, ds=res["scores"]["score_composed"], rc=res["scores"]["score_route"],
               status=res["status"], route_dev=len(res["infractions"].get("route_dev", [])),
               outside_lanes=len(res["infractions"].get("outside_route_lanes", [])),
               collisions=sum(len(res["infractions"].get(k, [])) for k in
                              ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")))
    ticks = [json.loads(l) for l in open(att / "ticks.jsonl")]
    tr = np.array([t["truth"][:2] for t in ticks if "truth" in t])
    row["moved"] = bool(max(t["v"] for t in ticks) > 0.5)
    moving = [t["t"] - ticks[0]["t"] for t in ticks if t["v"] > 0.5 and not str(t.get("reason", "")).startswith("engage")]
    row["first_own_move_s"] = round(moving[0], 1) if moving else None
    row["junction_steer_ticks"] = sum("junction_steer" in str(t.get("reason", "")) for t in ticks)
    for k in ("turn", "straight"):
        row[k + "_reached"] = row[k + "_passed"] = 0
    # shared control: time and distance per driver, and which driver held the car at each infraction
    drv = [t.get("driver", "model") for t in ticks]
    dist = {d: 0.0 for d in ("model", "partner", "blend")}
    for t, d in zip(ticks, drv):
        dist[d] = dist.get(d, 0.0) + max(t["v"], 0.0) * 0.05
    total = sum(dist.values()) or 1.0
    for d in ("model", "partner", "blend"):
        row["dist_m_" + d] = round(dist[d], 1)
        row["dist_share_" + d] = round(dist[d] / total, 3)
        row["time_share_" + d] = round(drv.count(d) / max(len(drv), 1), 3)
    by = {}
    txy = np.array([t["truth"][:2] if "truth" in t else [np.nan, np.nan] for t in ticks])
    for kind, items in res["infractions"].items():
        if kind in ("min_speed_infractions",):
            continue
        for text in items:
            m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", str(text))
            who = "?"
            if m and np.isfinite(txy).any():
                k = int(np.nanargmin(np.linalg.norm(txy - [float(m.group(1)), float(m.group(2))], axis=1)))
                who = drv[k]
            by.setdefault(who, []).append(kind)
    row["infractions_by_driver"] = json.dumps(by)
    if (att / "route.json").exists() and len(tr):
        for c, a, b in events(json.loads((att / "route.json").read_text())):
            key = "straight" if c == STRAIGHT else "turn"
            da = np.linalg.norm(tr - a, axis=1)
            if da.min() > 5.0:
                continue
            row[key + "_reached"] += 1
            after = tr[int(np.argmin(da)):]
            row[key + "_passed"] += int(np.linalg.norm(after - b, axis=1).min() <= 3.0)
    return row


def summarize(rows, label):
    n = len(rows)
    tot = lambda k: sum(r[k] for r in rows)  # noqa: E731
    by = {}
    for r in rows:
        for who, kinds in json.loads(r["infractions_by_driver"]).items():
            by[who] = by.get(who, 0) + len(kinds)
    return dict(phase=label, n=n, ds=round(float(np.mean([r["ds"] for r in rows])), 1),
                dist_share_model=round(tot("dist_m_model") / max(sum(tot("dist_m_" + d) for d in ("model", "partner", "blend")), 1e-9), 3),
                dist_share_partner=round(tot("dist_m_partner") / max(sum(tot("dist_m_" + d) for d in ("model", "partner", "blend")), 1e-9), 3),
                time_share_model=round(float(np.mean([r["time_share_model"] for r in rows])), 3),
                infractions_by_driver=json.dumps(by),
                rc=round(float(np.mean([r["rc"] for r in rows])), 1), moved=sum(r["moved"] for r in rows),
                turns=f"{tot('turn_passed')}/{tot('turn_reached')}", straights=f"{tot('straight_passed')}/{tot('straight_reached')}",
                route_dev=tot("route_dev"), outside_lanes=tot("outside_lanes"), collisions=tot("collisions"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    per, summ = [], []
    for run in a.runs:
        rows = [attempt_summary(att) for att in sorted(run.glob("attempts/*/*")) if (att / "results.json").exists()]
        # the last attempt of each route counts
        rows = list({r["route"]: r for r in rows}.values())
        per += [dict(phase=run.name, **r) for r in rows]
        summ.append(summarize(rows, run.name))
    w = csv.DictWriter(sys.stdout, list(summ[0]))
    w.writeheader()
    w.writerows(summ)
    if a.csv:
        with open(a.csv, "w") as fh:
            w = csv.DictWriter(fh, list(per[0]))
            w.writeheader()
            w.writerows(per)
