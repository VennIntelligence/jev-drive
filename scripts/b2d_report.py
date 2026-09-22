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
import math
import re
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
    p.add_argument("--csv", default="", help="write every attempt, including failures and missing routes")
    p.add_argument("--json", default="", help="write legacy per-town cost table")
    p.add_argument("--controller-json", default="", help="write all attempts and controller metrics")
    return p.parse_args()


def towns_of(routes_xml):
    root = ET.parse(routes_xml).getroot()
    return dict((r.get("id"), r.get("town")) for r in root.findall("route"))


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def quantile(xs, q):
    xs = sorted(x for x in xs if finite(x))
    if not xs:
        return None
    position = (len(xs) - 1) * q
    lo, hi = int(position), min(int(position) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (position - lo)


def metric(xs):
    xs = [x for x in xs if finite(x)]
    return {"n": len(xs), "min": min(xs) if xs else None, "max": max(xs) if xs else None, "sum_squared": sum(x * x for x in xs),
            "rms": math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None,
            "median": quantile(xs, .5), "p90": quantile(xs, .9), "p95": quantile(xs, .95),
            "p99": quantile(xs, .99), "abs_p95": quantile([abs(x) for x in xs], .95),
            "abs_max": max([abs(x) for x in xs]) if xs else None}


def official_result(adir):
    """Keep evaluator status and driving score distinct from subprocess success."""
    path = adir / "results.json"
    raw = read_json(path)
    state = "missing" if not path.exists() else "invalid_json"
    records = []
    if isinstance(raw, dict):
        records = (raw.get("_checkpoint") or {}).get("records") or []
        state = "empty_records" if not records else "ok"
    result = {"results_state": state, "official_status": None, "completion": None,
              "driving_completed": None, "official_finalized": False, "official_tick_runtime": False, "official_records": records}
    if not records:
        return result
    # One route per process is the harness contract. Never silently select a best record.
    if len(records) != 1 or not isinstance(records[0], dict):
        result["results_state"] = "unexpected_records"
        return result
    r = records[0]
    scores = r.get("scores") or {}
    infractions = r.get("infractions") or {}
    minimum = infractions.get("min_speed_infractions", [])
    percentages = []
    for event in minimum:
        if isinstance(event, dict) and finite(event.get("percentage")):
            percentages.append(event["percentage"])
        elif isinstance(event, str):
            match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", event)
            if match:
                percentages.append(float(match.group(1)))
    result.update({"official_tick_runtime": "tickruntime" in str(r.get("status", "")).lower().replace(" ", ""), "official_finalized": r.get("status") not in (None, "Started"), "official_status": r.get("status"), "completion": scores.get("score_route"),
                   "score_composed": scores.get("score_composed"),
                   "score_penalty": scores.get("score_penalty"),
                   "scenario": r.get("scenario_name"), "infractions": infractions,
                   "minimum_speed_events": len(minimum) if "min_speed_infractions" in infractions else None, "minimum_speed_percentages": percentages,
                   "minimum_speed_penalty_mode": "unused (Bench2Drive 0.0.4)",
                   "minimum_speed_penalty": None,
                   "minimum_speed_penalty_note": "Not independently serialized; no penalty inferred from events.",
                   "sim_time_s": (r.get("meta") or {}).get("duration_game")})
    completion = result["completion"]
    result["driving_completed"] = finite(completion) and completion >= 100
    for key in ("vehicle_blocked", "route_dev", "collisions_layout", "collisions_pedestrian",
                "collisions_vehicle", "route_timeout", "scenario_timeouts"):
        result[key] = len(infractions[key]) if isinstance(infractions.get(key), list) else None
    return result


TELEMETRY_METRICS = ("truth_cross_track_m", "route_cross_track_m", "cross_track_m",
                     "heading_error_rad", "raw_pose_error_m", "pose_error_m",
                     "pose_heading_error_rad", "trajectory_age_s", "controller_step_ms")


def telemetry(adir, official):
    """Stream JSONL, retain partial-file evidence and never replace absent truth with estimates."""
    path = adir / "control.jsonl"
    metrics = dict((key, []) for key in TELEMETRY_METRICS)
    metrics.update(command_speed_error_mps=[], reference_speed_error_mps=[])
    before = dict((key, []) for key in metrics)
    collisions = []
    for key in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"):
        collisions.extend((official.get("infractions") or {}).get(key, []))
    frames = []
    for event in collisions:
        if isinstance(event, dict) and finite(event.get("frame")):
            frames.append(event["frame"])
        elif isinstance(event, str):
            match = re.search(r"(?:frame|at frame)\s*[:=]?\s*(\d+)", event, re.I)
            if match:
                frames.append(int(match.group(1)))
    captured = read_json(adir / "criterion_events.json") or {}
    captured_frames = [e["frame"] for e in captured.get("events", [])
                       if e.get("type", "").startswith("COLLISION_") and finite(e.get("frame"))]
    if captured.get("state") == "ok" and len(captured_frames) == len(collisions):
        frames = captured_frames
    collision_frame = min(frames) if frames else None
    can_split = ("infractions" in official) and (not collisions or len(frames) == len(collisions))
    quality = {"state": "missing", "lines": 0, "valid_rows": 0, "invalid_lines": 0,
               "nonmonotonic_frames": 0, "nonmonotonic_time": 0, "missing_frames": 0,
               "sensor_frame_mismatch": 0, "invalid_controls": 0, "truth_frame_mismatch": 0,
               "missing_fields": {}}
    reasons, stale, saturation, low_progress = {}, 0, 0, []
    previous_frame = previous_time = first_time = last_time = low_start = None
    if not path.exists():
        return {"telemetry_quality": quality, "tracking": {}, "before_collision": {},
                "before_collision_state": "missing_telemetry"}
    with path.open() as fh:
        for line in fh:
            quality["lines"] += 1
            try:
                d = json.loads(line)
                if not isinstance(d, dict):
                    raise ValueError("non-object")
            except ValueError:
                quality["invalid_lines"] += 1
                continue
            quality["valid_rows"] += 1
            frame, stamp = d.get("frame"), d.get("sim_time")
            for key in ("frame", "sim_time", "speed_mps", "trajectory_time", "trajectory_age_s",
                        "sensor_frames", "truth_cross_track_m", "pose_error_m"):
                if d.get(key) is None:
                    quality["missing_fields"][key] = quality["missing_fields"].get(key, 0) + 1
            if finite(frame):
                if previous_frame is not None:
                    quality["nonmonotonic_frames"] += int(frame <= previous_frame)
                    quality["missing_frames"] += max(0, int(frame - previous_frame - 1))
                previous_frame = frame
            if finite(stamp):
                if previous_time is not None:
                    quality["nonmonotonic_time"] += int(stamp <= previous_time)
                if first_time is None:
                    first_time = stamp
                previous_time = last_time = stamp
            sensors = d.get("sensor_frames") or {}
            if any(v != frame for k, v in sensors.items() if k.upper() in ("GPS", "GNSS", "IMU", "SPEED")):
                quality["sensor_frame_mismatch"] += 1
            truth_aligned = d.get("truth_frame", frame) == frame
            if not truth_aligned:
                quality["truth_frame_mismatch"] += 1
            controls = [d.get(k) for k in ("throttle", "steer", "brake")]
            if (not all(finite(x) for x in controls) or not 0 <= controls[0] <= 1
                    or not -1 <= controls[1] <= 1 or not 0 <= controls[2] <= 1
                    or (controls[0] > 0 and controls[2] > 0)):
                quality["invalid_controls"] += 1
            else:
                saturation += int(controls[0] >= .75 or abs(controls[1]) >= .8 or controls[2] >= 1)
            reason = str(d.get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
            stale += int("stale" in reason)
            values = dict((k, d.get(k)) for k in TELEMETRY_METRICS)
            if not truth_aligned:
                for k in ("truth_cross_track_m", "raw_pose_error_m", "pose_error_m", "pose_heading_error_rad"):
                    values[k] = None
            speed = d.get("speed_mps")
            for name, key in (("command", "target_speed_mps"), ("reference", "reference_speed_mps")):
                values[name + "_speed_error_mps"] = speed - d[key] if finite(speed) and finite(d.get(key)) else None
            for key, value in values.items():
                if finite(value):
                    metrics[key].append(value)
                    if can_split and (collision_frame is None or (finite(frame) and frame < collision_frame)):
                        before[key].append(value)
            low = finite(speed) and speed < .5 and finite(d.get("target_speed_mps")) and d["target_speed_mps"] >= 2
            if low and finite(stamp):
                if low_start is None:
                    low_start = stamp
            elif low_start is not None:
                if finite(stamp) and stamp - low_start >= 5:
                    low_progress.append({"start_s": low_start, "end_s": stamp, "duration_s": stamp - low_start})
                low_start = None
    if low_start is not None and last_time - low_start >= 5:
        low_progress.append({"start_s": low_start, "end_s": last_time, "duration_s": last_time - low_start})
    quality["state"] = "partial" if quality["invalid_lines"] else "ok" if quality["valid_rows"] else "empty"
    return {"telemetry_quality": quality, "tracking": dict((k, metric(v)) for k, v in metrics.items()),
            "before_collision": dict((k, metric(v)) for k, v in before.items()) if can_split else {},
            "before_collision_state": "available" if can_split else "collision_frame_unavailable",
            "first_collision_frame": collision_frame, "reasons": reasons, "stale_ticks": stale,
            "saturation_fraction": saturation / quality["valid_rows"] if quality["valid_rows"] else None,
            "telemetry_duration_s": last_time - first_time if first_time is not None else None,
            "low_progress_segments": low_progress}


def collect(out):
    """All attempts, including failed and partial ones, in route and attempt order."""
    rows = []
    for rdir in sorted((Path(out) / "attempts").glob("*")):
        for adir in sorted((p for p in rdir.glob("*") if p.is_dir() and p.name.isdigit()),
                           key=lambda p: int(p.name)):
            route = read_json(adir / "route_result.json") or {}
            attempt = read_json(adir / "attempt.json") or {}
            d = dict(route, **attempt)
            prof = route.get("profile") or attempt.get("profile") or {}
            config = d.get("config") or read_json(adir / "agent_config.json") or {}
            row = {"route_id": rdir.name, "attempt": int(adir.name),
                   "status": d.get("status", "running"), "killed": d.get("killed"),
                   "wall_s": d.get("wall_s"), "ticks": prof.get("ticks", d.get("ticks")),
                   "ticks_used": prof.get("ticks_used"), "total_ms": prof.get("total_ms_mean"),
                   "world_tick_ms": prof.get("world_tick_ms_mean"), "tree_ms": prof.get("tree_ms_mean"),
                   "agent_ms": prof.get("agent_ms_mean"), "mib_in": prof.get("mib_in"),
                   "server_age_routes": d.get("server_age_routes"), "server_index": d.get("server_index"),
                   "preset": config.get("controller_preset") if config.get("drive") == "controller" else config.get("drive", "legacy"),
                   "seed": config.get("tm_seed"), "tick_cap_configured": config.get("max_ticks", 0),
                   "capped": d.get("capped", bool(config.get("max_ticks") and prof.get("ticks", 0) >= config["max_ticks"])),
                   "process_returncode": d.get("returncode"), "harness_finished": d.get("status") == "finished"}
            if not d and any((adir / name).exists() for name in ("attempt.json", "route_result.json")):
                row["status"] = "unreadable_result"
            official = official_result(adir)
            row.update(official)
            row["harness_capped"] = row["capped"]
            row["capped"] = bool(row["harness_capped"] or row["official_tick_runtime"])
            row.update(telemetry(adir, official))
            row["partial"] = (row["status"] != "finished" or row["capped"] or official["results_state"] != "ok"
                              or not official.get("official_finalized")
                              or row["telemetry_quality"]["state"] == "partial"
                              or (config.get("drive") == "controller" and row["telemetry_quality"]["state"] != "ok"))
            rows.append(row)
    # A requested route with no directory must remain visible in the denominator.
    summary = read_json(Path(out) / "summary.json") or {}
    known = set(r["route_id"] for r in rows)
    requested = set(summary.get("routes_never_finished", [])) | set((summary.get("attempts") or {}).keys())
    requested_config = {}
    for manifest_path in sorted((Path(out) / "invocations").glob("*.json")):
        manifest = read_json(manifest_path) or {}
        for route in manifest.get("routes", []):
            rid = route["route_id"]
            requested.add(rid)
            requested_config[rid] = manifest.get("config") or {}
    for rid in sorted(requested - known):
        config = requested_config.get(rid, {})
        rows.append({"route_id": rid, "attempt": 0, "status": "missing_attempt", "partial": True,
                     "results_state": "missing", "harness_finished": False, "driving_completed": None,
                     "preset": config.get("controller_preset", "unknown"), "seed": config.get("tm_seed")})
    return rows


def controller_summary(rows):
    """First evaluator-returned attempt is official; later attempts never improve selection."""
    grouped = {}
    for row in rows:
        key = (row.get("preset", "unknown"), row.get("seed"))
        grouped.setdefault(key, {}).setdefault(row["route_id"], []).append(row)
    result = {}
    for (preset, seed), routes in grouped.items():
        selected, all_attempts = [], []
        for tries in routes.values():
            tries.sort(key=lambda r: r["attempt"])
            chosen = next((r for r in tries if r.get("harness_finished")), tries[-1])
            selected.append(chosen)
            all_attempts.extend(r for r in tries if r["attempt"] > 0)
            for row in tries:
                row["selected_attempt"] = row is chosen
                row["retry_count"] = max(0, sum(r["attempt"] > 0 for r in tries) - 1)
                row["all_attempt_wall_s"] = sum(r.get("wall_s") or 0 for r in tries)
        metrics = {}
        for key in TELEMETRY_METRICS + ("command_speed_error_mps", "reference_speed_error_mps"):
            scopes = {}
            for scope in ("tracking", "before_collision"):
                ms = [(r.get(scope) or {}).get(key, {}) for r in selected]
                n = sum(m.get("n", 0) for m in ms)
                scopes[scope] = {"samples": n, "routes_with_samples": sum(bool(m.get("n")) for m in ms),
                                 "tick_weighted_rms": math.sqrt(sum(m.get("sum_squared", 0) for m in ms) / n) if n else None,
                                 "route_mean_rms": mean(m.get("rms") for m in ms)}
            metrics[key] = scopes
        result["%s/seed=%s" % (preset, seed)] = {
            "routes": len(routes), "attempts": len(all_attempts), "retries": sum(r["retry_count"] for r in selected),
            "harness_finished": sum(bool(r.get("harness_finished")) for r in selected),
            "driving_completed": sum(r.get("driving_completed") is True for r in selected),
            "missing_official_results": sum(r.get("results_state") != "ok" for r in selected),
            "completion_mean_observed": mean(r.get("completion") for r in selected),
            "completion_observed_routes": sum(finite(r.get("completion")) for r in selected),
            "all_attempt_wall_s": sum(r.get("wall_s") or 0 for r in all_attempts),
            "retry_wall_s": sum(r.get("wall_s") or 0 for r in all_attempts if not r["selected_attempt"]),
            "capped": sum(bool(r.get("capped")) for r in selected), "metrics": metrics,
            "tick_distribution_completed": metric(r.get("ticks") for r in selected if r.get("driving_completed") is True),
            "tick_distribution_incomplete": metric(r.get("ticks") for r in selected if r.get("driving_completed") is not True),
            "selection_policy": "first harness-finished attempt, else latest; all attempts retained"}
    return result


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
                                   "running": set(), "failed_attempts": 0})
        b["attempts"] += int(r.get("attempt", 1) > 0)
        b["routes"].add(r["route_id"])
        if r["status"] == "finished":
            b["finished"].add(r["route_id"])
        elif r["status"] == "running":
            b["running"].add(r["route_id"])
        elif r.get("attempt", 1) > 0:
            b["failed_attempts"] += 1
    lines = ["| town | routes | finished | running | attempts | failed attempts | restarts | "
             "never finished |",
             "|---|--:|--:|--:|--:|--:|--:|---|"]
    for t in sorted(by_town, key=lambda t: (t not in LARGE_MAPS, t)):
        b = by_town[t]
        # A route with an attempt still in flight has not failed and has not finished; it is
        # neither a restart nor a never-finished. Only a completed run has an empty `running`.
        never = sorted(b["routes"] - b["finished"] - b["running"])
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %s |" % (
            t, len(b["routes"]), len(b["finished"]), len(b["running"]), b["attempts"],
            b["failed_attempts"], sum(max(0, sum(r["route_id"] == rid and r.get("attempt", 1) > 0 for r in rows) - 1) for rid in b["routes"]), ", ".join(never) or "-"))
    return "\n".join(lines)


def server_age(rows):
    """R7's curve: failures against how many routes that server process had already served."""
    buckets = {}
    for r in rows:
        age = r.get("server_age_routes")
        if age is None:
            continue
        if r["status"] == "running":
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

    controller = controller_summary(rows)
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
          "themselves" % (sum(r.get("attempt", 1) > 0 for r in rows), len(done), sum(wall) / 3600.0))

    print("Harness finished means evaluator returned; official completion is reported separately.")
    print("| preset / seed | routes | completed | missing results | capped | retries | completion observed |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for key, group in sorted(controller.items()):
        print("| %s | %d | %d | %d | %d | %d | %s |" % (
            key, group["routes"], group["driving_completed"], group["missing_official_results"],
            group["capped"], group["retries"], fmt(group["completion_mean_observed"])))
    if a.controller_json:
        Path(a.controller_json).write_text(json.dumps({"schema_version": 1, "attempts": rows,
                                                     "by_preset_seed": controller}, indent=2, allow_nan=False))
        print("wrote %s" % a.controller_json)
    if a.csv:
        cols = ["route_id", "town", "attempt", "status", "killed", "wall_s", "ticks", "total_ms",
                "world_tick_ms", "tree_ms", "agent_ms", "mib_in", "server_age_routes",
                "server_index", "preset", "seed", "scenario", "official_status", "completion",
                "score_composed", "score_penalty", "driving_completed", "results_state", "capped", "harness_capped", "official_tick_runtime",
                "partial", "selected_attempt", "retry_count", "all_attempt_wall_s", "vehicle_blocked",
                "route_dev", "collisions_layout", "collisions_vehicle", "collisions_pedestrian",
                "minimum_speed_events", "minimum_speed_percentages", "minimum_speed_penalty_mode",
                "sim_time_s", "stale_ticks", "saturation_fraction", "telemetry_duration_s", "telemetry_quality"]
        csv_rows = [dict(row) for row in rows]
        for key in TELEMETRY_METRICS + ("command_speed_error_mps", "reference_speed_error_mps"):
            for stat in ("n", "rms", "median", "p90", "abs_p95", "abs_max", "p99"):
                col = key + "_" + stat
                cols.append(col)
                for row in csv_rows:
                    row[col] = ((row.get("tracking") or {}).get(key) or {}).get(stat)
        for row in csv_rows:
            for key in cols:
                if isinstance(row.get(key), (dict, list)):
                    row[key] = json.dumps(row[key], sort_keys=True, allow_nan=False)
        with open(a.csv, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)
        print("wrote %s" % a.csv)
    if a.json:
        Path(a.json).write_text(json.dumps(table, indent=2))
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
