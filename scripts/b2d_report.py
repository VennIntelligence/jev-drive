#!/usr/bin/env python
"""Per-town report for a scripts/b2d_run.py output directory.

b2d_run.py's own summary.json aggregates over the whole run, which is the right thing when every
route is in the same town and the wrong thing for bench2drive220.xml: 151 of its 220 routes are
Large Maps (Town12 104, Town13 47, Town11 and Town15 7 each) and the rest are small towns, so a
single mean hides the only split that matters. This joins the per-attempt records back to the town
from the route XML and reports each town separately.

Reads only what a run already wrote (`attempts/<id>/<n>/attempt.json`, which carries the route's
own `profile` plus the runner's view of which server it ran on and how old that server was). It
never starts anything, so it is safe to run against a live directory.

    scripts/b2d_report.py --out $DATA_DIR/runs/b2d/full220 [--routes .../bench2drive220.xml]
    ... --csv results.csv        # one row per finished route

Three tables, in the order the questions get asked:
  per town     ms/tick (the route's own tick profile, warmup dropped), ticks and wall per route
  reliability  attempts, restarts and never-finished routes per town - R6, which says a score over
               the routes that happened to finish is a score on a selected subset
  server age   attempts and failures bucketed by how many routes that server process had already
               served - R7's curve, the one that decides whether --recycle-routes buys anything

Python 3.8+ and no third-party imports, so it runs in envs/carla next to the run itself.
"""
import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
DEFAULT_ROUTES = DATA_DIR / "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
# Town11/12/13/15 stream tiles and hold far more actors than the base package's towns; every
# per-tick number we had before 2026-09-22 was measured on the small ones.
LARGE_MAPS = {"Town11", "Town12", "Town13", "Town15"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="a b2d_run.py --out directory")
    p.add_argument("--routes", default=str(DEFAULT_ROUTES))
    p.add_argument("--csv", default="", help="also write one row per finished route here")
    p.add_argument("--json", default="", help="also write the per-town table here")
    return p.parse_args()


def towns_of(routes_xml):
    root = ET.parse(routes_xml).getroot()
    return dict((r.get("id"), r.get("town")) for r in root.findall("route"))


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def collect(out):
    """One record per attempt, newest attempt last, in route order."""
    rows = []
    for rdir in sorted((Path(out) / "attempts").glob("*")):
        for adir in sorted(rdir.glob("*"), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            route = read_json(adir / "route_result.json") or {}
            prof = route.get("profile", {})
            # attempt.json is the runner's view (which server, how old); route_result.json is the
            # route's own. Runs made before attempt.json existed only have the latter, and a run
            # killed between the two has only the latter too, so fall back rather than drop it.
            d = read_json(adir / "attempt.json") or (route or None)
            if d is None:
                rows.append({"route_id": rdir.name, "attempt": int(adir.name or 1),
                             "status": "no result file"})
                continue
            rows.append({
                "route_id": rdir.name, "attempt": int(adir.name or 1),
                "status": d.get("status"), "killed": d.get("killed"),
                "wall_s": d.get("wall_s"), "ticks": prof.get("ticks"),
                "ticks_used": prof.get("ticks_used"),
                "total_ms": prof.get("total_ms_mean"),
                "world_tick_ms": prof.get("world_tick_ms_mean"),
                "tree_ms": prof.get("tree_ms_mean"),
                "agent_ms": prof.get("agent_ms_mean"),
                "mib_in": prof.get("mib_in"),
                "server_age_routes": d.get("server_age_routes"),
                "server_index": d.get("server_index"),
            })
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def fmt(x, nd=1):
    return "-" if x is None else ("%%.%df" % nd) % x


def per_town(rows, town_of):
    """Cost, from the finished attempts only: a killed attempt's ms/tick is the cost of dying."""
    towns = {}
    for r in rows:
        t = town_of.get(r["route_id"], "?")
        towns.setdefault(t, []).append(r)
    lines = ["| town | large map | routes | ms/tick mean | ms/tick median | world_tick | tree | "
             "ticks/route | wall_s/route |",
             "|---|:--:|--:|--:|--:|--:|--:|--:|--:|"]
    table = {}
    for t in sorted(towns, key=lambda t: (t not in LARGE_MAPS, t)):
        done = [r for r in towns[t] if r["status"] == "finished" and r.get("total_ms")]
        if not done:
            lines.append("| %s | %s | 0 | - | - | - | - | - | - |"
                         % (t, "yes" if t in LARGE_MAPS else "no"))
            continue
        row = {"town": t, "large_map": t in LARGE_MAPS, "routes": len(done),
               "ms_per_tick_mean": mean(r["total_ms"] for r in done),
               "ms_per_tick_median": median([r["total_ms"] for r in done]),
               "world_tick_ms": mean(r["world_tick_ms"] for r in done),
               "tree_ms": mean(r["tree_ms"] for r in done),
               "ticks_per_route": mean(r["ticks"] for r in done),
               "wall_s_per_route": mean(r["wall_s"] for r in done)}
        table[t] = row
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            t, "yes" if row["large_map"] else "no", row["routes"],
            fmt(row["ms_per_tick_mean"]), fmt(row["ms_per_tick_median"]),
            fmt(row["world_tick_ms"]), fmt(row["tree_ms"]),
            fmt(row["ticks_per_route"], 0), fmt(row["wall_s_per_route"])))
    return "\n".join(lines), table


def reliability(rows, town_of):
    """R6: restarts and never-finished routes, per town. A town with a systematically higher
    failure rate is a result about that town, not noise to be averaged away."""
    by_town = {}
    for r in rows:
        t = town_of.get(r["route_id"], "?")
        b = by_town.setdefault(t, {"attempts": 0, "routes": set(), "finished": set(),
                                   "failed_attempts": 0})
        b["attempts"] += 1
        b["routes"].add(r["route_id"])
        if r["status"] == "finished":
            b["finished"].add(r["route_id"])
        else:
            b["failed_attempts"] += 1
    lines = ["| town | routes | finished | attempts | failed attempts | restarts | never finished |",
             "|---|--:|--:|--:|--:|--:|---|"]
    for t in sorted(by_town, key=lambda t: (t not in LARGE_MAPS, t)):
        b = by_town[t]
        never = sorted(b["routes"] - b["finished"])
        lines.append("| %s | %d | %d | %d | %d | %d | %s |" % (
            t, len(b["routes"]), len(b["finished"]), b["attempts"], b["failed_attempts"],
            b["attempts"] - len(b["routes"]), ", ".join(never) or "-"))
    return "\n".join(lines)


def server_age(rows):
    """R7's curve: failures against how many routes that server process had already served."""
    buckets = {}
    for r in rows:
        age = r.get("server_age_routes")
        if age is None:
            continue
        b = buckets.setdefault(age, {"attempts": 0, "failed": 0})
        b["attempts"] += 1
        if r["status"] != "finished":
            b["failed"] += 1
    lines = ["| server age (routes served) | attempts | failed | failure rate |",
             "|--:|--:|--:|--:|"]
    for age in sorted(buckets):
        b = buckets[age]
        lines.append("| %d | %d | %d | %s |" % (age, b["attempts"], b["failed"],
                                                fmt(100.0 * b["failed"] / b["attempts"]) + "%"))
    return "\n".join(lines)


def main():
    a = parse_args()
    town_of = towns_of(a.routes)
    rows = collect(a.out)
    if not rows:
        print("no attempts under %s" % a.out)
        return 1
    for r in rows:
        r["town"] = town_of.get(r["route_id"], "?")

    cost, table = per_town(rows, town_of)
    done = [r for r in rows if r["status"] == "finished"]
    large = [r for r in done if r["town"] in LARGE_MAPS and r.get("total_ms")]
    small = [r for r in done if r["town"] not in LARGE_MAPS and r.get("total_ms")]
    print("## Cost per town\n\n%s\n" % cost)
    if large and small:
        lm, sm = mean(r["total_ms"] for r in large), mean(r["total_ms"] for r in small)
        print("Large maps %s ms/tick over %d routes, small towns %s over %d: **%.2fx**.\n"
              % (fmt(lm), len(large), fmt(sm), len(small), lm / sm))
    print("## Reliability per town\n\n%s\n" % reliability(rows, town_of))
    print("## Failures by server age\n\n%s\n" % server_age(rows))

    wall = [r["wall_s"] for r in done if r.get("wall_s")]
    print("%d attempts, %d finished routes, %.1f route-hours of wall clock in the routes "
          "themselves" % (len(rows), len(done), sum(wall) / 3600.0))

    if a.csv:
        cols = ["route_id", "town", "attempt", "status", "killed", "wall_s", "ticks", "total_ms",
                "world_tick_ms", "tree_ms", "agent_ms", "mib_in", "server_age_routes",
                "server_index"]
        with open(a.csv, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % a.csv)
    if a.json:
        Path(a.json).write_text(json.dumps(table, indent=2))
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
