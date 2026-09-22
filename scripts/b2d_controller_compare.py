#!/usr/bin/env python
"""Compare controller campaign groups without dropping incomplete routes or selecting best retries.

Read-only inputs; writes comparison.json, routes.csv, attempts.csv and comparison.md under --out.
Existing controller-report.json files are preferred; pending groups are rebuilt from partial files.
Python 3.8, no CARLA or third-party dependencies.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import b2d_report as report

INFRACTIONS = ("vehicle_blocked", "route_dev", "collisions_layout", "collisions_vehicle",
               "collisions_pedestrian", "route_timeout", "scenario_timeouts", "minimum_speed_events")
METRICS = ("truth_cross_track_m", "heading_error_rad", "command_speed_error_mps", "reference_speed_error_mps",
           "pose_error_m", "raw_pose_error_m", "pose_heading_error_rad", "trajectory_age_s", "controller_step_ms")


def read_object(path):
    data = report.read_json(path)
    return data if isinstance(data, dict) else {}


def sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def aggregate_metrics(rows, scope, key):
    metrics = [(r.get(scope) or {}).get(key) or {} for r in rows]
    count = sum(m.get("n", 0) for m in metrics)
    squared = sum(m.get("sum_squared", 0) for m in metrics)
    return {"samples": count, "routes_with_samples": sum(m.get("n", 0) > 0 for m in metrics),
            "route_mean_rms": report.mean(m.get("rms") for m in metrics),
            "tick_weighted_rms": (squared / count) ** .5 if count else None,
            "route_mean_p90": report.mean(m.get("p90") for m in metrics),
            "route_mean_abs_p95": report.mean(m.get("abs_p95") for m in metrics)}


def summarize_group(rows):
    final = [r for r in rows if r.get("selected_attempt")]
    actual = [r for r in rows if r.get("attempt", 0) > 0]
    official = [r for r in final if r.get("results_state") == "ok" and r.get("official_finalized")]
    complete_data = len(official) == len(final) and bool(final) and all(report.finite(r.get("completion")) for r in official) and not any(r.get("harness_capped", r.get("capped")) for r in final)
    summary = {"requested_routes": len(final), "attempts": len(actual),
               "retries": sum(r.get("retry_count", 0) for r in final),
               "official_finalized_routes": len(official), "data_complete": complete_data,
               "harness_finished": sum(bool(r.get("harness_finished")) for r in final),
               "true_completed": sum(r.get("driving_completed") is True for r in official),
               "completion_mean_observed": report.mean(r.get("completion") for r in official),
               "completion_mean_all_routes": report.mean(r.get("completion") for r in official) if complete_data else None,
               "missing_official_route_ids": [r["route_id"] for r in final if r not in official],
               "capped_route_ids": [r["route_id"] for r in final if r.get("capped")],
               "harness_capped_route_ids": [r["route_id"] for r in final if r.get("harness_capped")],
               "official_tick_runtime_route_ids": [r["route_id"] for r in final if r.get("official_tick_runtime")],
               "failed_route_ids": [r["route_id"] for r in official if not r.get("driving_completed")],
               "failed_attempts": [{"route_id": r["route_id"], "attempt": r["attempt"],
                                    "harness_status": r.get("status"), "official_status": r.get("official_status"),
                                    "completion": r.get("completion")} for r in actual
                                   if not r.get("harness_finished") or r.get("driving_completed") is not True],
               "all_attempt_wall_s": sum(r.get("wall_s") or 0 for r in actual),
               "unknown_wall_attempts": sum(not report.finite(r.get("wall_s")) for r in actual),
               "all_attempt_ticks": sum(r.get("ticks") or 0 for r in actual),
               "selected_ticks": report.metric(r.get("ticks") for r in final),
               "completed_ticks": report.metric(r.get("ticks") for r in final if r.get("driving_completed") is True),
               "incomplete_ticks": report.metric(r.get("ticks") for r in final if r.get("driving_completed") is not True),
               "precollision_unavailable_routes": [r["route_id"] for r in final if r.get("before_collision_state") != "available"],
               "infractions": {}, "metrics": {}}
    for key in INFRACTIONS:
        observed = [r[key] for r in final if report.finite(r.get(key))]
        summary["infractions"][key] = {"count_observed": sum(observed), "routes_observed": len(observed),
                                       "missing_routes": len(final) - len(observed)}
    for key in METRICS:
        summary["metrics"][key] = {scope: aggregate_metrics(final, scope, key)
                                   for scope in ("tracking", "before_collision")}
    return summary


def load_campaign(path):
    path = Path(path).resolve()
    manifest = read_object(path / "manifest.json")
    config = manifest.get("config") or {}
    route_path = Path(config.get("routes", "missing.xml"))
    routes = {}
    if route_path.is_file():
        for route in ET.parse(route_path).getroot().findall("route"):
            routes[route.get("id")] = {"town": route.get("town"),
                                       "scenario_xml": [s.get("type") for s in route.findall("scenarios/scenario")]}
    expected = [(preset, int(seed)) for preset in config.get("presets", "").split(",") if preset
                for seed in str(config.get("seeds", "0")).split(",")]
    if not expected:
        for directory in sorted(path.glob("*-seed*")):
            preset, _, seed = directory.name.rpartition("-seed")
            if seed.isdigit():
                expected.append((preset, int(seed)))
    groups = []
    for preset, seed in expected:
        directory = path / ("%s-seed%d" % (preset, seed))
        source = directory / "controller-report.json"
        saved = read_object(source)
        rows = saved.get("attempts")
        if not isinstance(rows, list):
            rows = report.collect(directory)
        # Strip old selection; recompute the same first-finished policy over all attempts.
        for row in rows:
            row.update(preset=preset, seed=seed)
            # Older reports defaulted absent infraction lists to zero; restore missingness.
            for key in INFRACTIONS:
                raw_key = "min_speed_infractions" if key == "minimum_speed_events" else key
                if raw_key not in (row.get("infractions") or {}):
                    row[key] = None
            # v1 group reports only exposed the harness cap. Upgrade from explicit status,
            # never from a guessed tick count, and preserve official terminal failures.
            if "official_tick_runtime" not in row:
                row["harness_capped"] = bool(row.get("capped"))
                row["official_tick_runtime"] = "tickruntime" in str(row.get("official_status", "")).lower().replace(" ", "")
                row["capped"] = row["harness_capped"] or row["official_tick_runtime"]
        present = {r["route_id"] for r in rows}
        for rid in sorted(set(routes) - present):
            rows.append({"route_id": rid, "attempt": 0, "status": "missing_attempt",
                         "preset": preset, "seed": seed, "results_state": "missing"})
        report.controller_summary(rows)
        key = "%s/%s-seed%d" % (path.name, preset, seed)
        for row in rows:
            row.update(group=key, source_directory=str(directory), source_commit=manifest.get("git_commit"),
                       dataset_sha256=sha256(route_path), **routes.get(row["route_id"], {}))
        groups.append({"group": key, "campaign": str(path), "preset": preset, "seed": seed,
                       "dataset_sha256": sha256(route_path), "source_report": str(source) if saved else None,
                       "source_mode": "saved_report" if saved else "partial_attempt_files",
                       "source_report_sha256": sha256(source),
                       "controller_config_sha256_at_report_time": sha256(config.get("controller_config", "missing.json")),
                       "manifest": manifest, "rows": rows, "summary": summarize_group(rows)})
    return groups


def pair_groups(left, right):
    a = {r["route_id"]: r for r in left["rows"] if r.get("selected_attempt")}
    b = {r["route_id"]: r for r in right["rows"] if r.get("selected_attempt")}
    ids = sorted(set(a) | set(b))
    paired = [rid for rid in ids if rid in a and rid in b and a[rid].get("official_finalized")
              and b[rid].get("official_finalized") and report.finite(a[rid].get("completion"))
              and report.finite(b[rid].get("completion")) and not a[rid].get("harness_capped", a[rid].get("capped"))
              and not b[rid].get("harness_capped", b[rid].get("capped"))]
    result = {"left": left["group"], "right": right["group"], "seed": left["seed"],
              "expected_routes": len(ids), "paired_final_routes": len(paired),
              "missing_pair_route_ids": sorted(set(ids) - set(paired)),
              "complete_pair": bool(ids) and len(paired) == len(ids),
              "completion_delta_right_minus_left": report.mean(b[rid].get("completion", 0) - a[rid].get("completion", 0) for rid in paired),
              "metrics": {}}
    for key in METRICS:
        result["metrics"][key] = {}
        for scope in ("tracking", "before_collision"):
            deltas = []
            for rid in paired:
                av = ((a[rid].get(scope) or {}).get(key) or {}).get("rms")
                bv = ((b[rid].get(scope) or {}).get(key) or {}).get("rms")
                if report.finite(av) and report.finite(bv):
                    deltas.append(bv - av)
            result["metrics"][key][scope] = {"paired_routes": len(deltas), "mean_rms_delta": report.mean(deltas)}
    return result


def strongest_references(groups):
    datasets = {}
    for group in groups:
        if group["preset"] in ("carla", "tcp"):
            datasets.setdefault((group["dataset_sha256"], group["seed"]), []).append(group)
    result = []
    for (dataset, seed), candidates in datasets.items():
        names = [g["preset"] for g in candidates]
        ready = (set(names) == {"carla", "tcp"} and len(names) == 2 and dataset is not None
                 and all(g["summary"]["data_complete"] for g in candidates))
        winner = None
        if ready:
            def rank(group):
                summary = group["summary"]
                return (-summary["completion_mean_all_routes"], -summary["true_completed"],
                        sum(summary["infractions"][k]["count_observed"] for k in ("vehicle_blocked", "route_dev")),
                        0 if group["preset"] == "carla" else 1)
            winner = sorted(candidates, key=rank)[0]["group"]
        result.append({"dataset_sha256": dataset, "seed": seed, "ready": ready, "strongest_observed_reference": winner,
                       "rule": "complete paired groups only; mean completion, then completed count, then raw blocked+deviation, then carla",
                       "not_default_selection": "G1/G2, two seeds, holdout and failure attribution remain required"})
    return result


def read_g2(paths):
    result = []
    for path in paths:
        data = report.read_json(path)
        rows = data if isinstance(data, list) else []
        grouped = {}
        for row in rows:
            grouped.setdefault(row.get("preset", "unknown"), []).append(row)
        result.append({"source": str(Path(path).resolve()), "sha256": sha256(path), "state": "ok" if rows else "missing_or_empty",
                       "by_preset": {preset: {"cases": len(cases), "passed_cases": sum(r.get("gate_pass") is True for r in cases),
                                                "all_recorded_gates_pass": all(r.get("gate_pass") is True for r in cases),
                                                "failed_cases": [{"route_id": r.get("route_id"), "failed_gates": [k for k, v in (r.get("gates") or {}).items() if not v], "exception": r.get("exception")} for r in cases if not r.get("gate_pass")],
                                                "scope": "recorded development cases; slope/held-out gate coverage is not inferred"}
                                     for preset, cases in grouped.items()}, "cases": rows})
    return result


def flatten(row):
    out = {key: row.get(key) for key in ("group", "route_id", "town", "scenario", "preset", "seed", "attempt",
           "selected_attempt", "retry_count", "status", "official_status", "completion", "driving_completed",
           "results_state", "partial", "capped", "harness_capped", "official_tick_runtime", "ticks", "wall_s", "all_attempt_wall_s",
           "score_composed", "score_penalty", "minimum_speed_penalty", "minimum_speed_penalty_mode",
           "sim_time_s", "total_ms", "world_tick_ms", "tree_ms", "agent_ms", "server_age_routes",
           "stale_ticks", "saturation_fraction", "first_collision_frame", "before_collision_state",
           "source_commit", "source_directory") + INFRACTIONS}
    out["telemetry_quality"] = json.dumps(row.get("telemetry_quality"), sort_keys=True)
    for key in ("minimum_speed_percentages", "low_progress_segments", "reasons"):
        out[key] = json.dumps(row.get(key), sort_keys=True)
    for key in METRICS:
        for scope in ("tracking", "before_collision"):
            metric = (row.get(scope) or {}).get(key) or {}
            for field in ("n", "rms", "median", "p90", "p95", "p99", "abs_p95", "abs_max"):
                out["%s_%s_%s" % (scope, key, field)] = metric.get(field)
    return out


def write_outputs(out, groups, g2, pairs, references):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Comparison output is not empty; use a new snapshot directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for group in groups for r in group["rows"]]
    for name, chosen in (("attempts.csv", rows), ("routes.csv", [r for r in rows if r.get("selected_attempt")])):
        flattened = [flatten(r) for r in chosen]
        with (out / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flatten({})))
            writer.writeheader()
            writer.writerows(flattened)
    summary = {"schema_version": 1, "comparison_code_sha256": sha256(__file__), "report_code_sha256": sha256(report.__file__), "groups": [{k: v for k, v in g.items() if k != "rows"} for g in groups],
               "paired": pairs, "strongest_references": references, "g2": g2,
               "selection_policy": "first harness-finished attempt else latest; all attempts in attempts.csv",
               "failure_attribution": "raw blocked/deviation are retained; controller-caused attribution remains unknown",
               "minimum_speed": "B2D 0.0.4 unused; raw counts retained, no inferred per-event penalty",
               "reference_speed_semantics": "Derivative of timed controller input including origin-to-first-point bridge; not independent route cruise truth. Startup/lateral route offset can inflate reference and command speed. Recorded values retained.",
               "cap_semantics": "capped = harness_capped or official_tick_runtime; official TickRuntime failures remain in finalized-route denominator"}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ["# Controller comparison", "", "Partial snapshots retain all requested routes; missing results are not success.", "",
             "| group | final / requested | mean completion observed | true completed | blocked observed | collisions observed | attempt wall s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        s = group["summary"]
        def infra_cell(keys):
            count = sum(s["infractions"][k]["count_observed"] for k in keys)
            observed = min(s["infractions"][k]["routes_observed"] for k in keys)
            return "%s (%d/%d known)" % (count if observed else "-", observed, s["requested_routes"])
        lines.append("| %s | %d / %d | %s | %d | %s | %s | %.1f |" % (
            group["group"], s["official_finalized_routes"], s["requested_routes"],
            report.fmt(s["completion_mean_observed"]), s["true_completed"],
            infra_cell(("vehicle_blocked",)),
            infra_cell(("collisions_layout", "collisions_vehicle", "collisions_pedestrian")),
            s["all_attempt_wall_s"]))
    lines += ["", "Speed errors use the controller trajectory derivative, including its origin-to-first-point bridge. A fixed cruise setting of 8 m/s is not independent truth for this metric; route offset can inflate initial reference/command speed. G2 independent cruise metrics remain separate.",
              "", "Vendor TickRuntime failures stay in the completion denominator; effective caps and artificial harness caps are distinct.",
              "", "Pairs are right minus left; partial pairs cannot select a default.", "", "| left | right | paired / expected | completion delta |", "|---|---|---:|---:|"]
    for pair in pairs:
        lines.append("| %s | %s | %d / %d | %s |" % (pair["left"], pair["right"], pair["paired_final_routes"], pair["expected_routes"], report.fmt(pair["completion_delta_right_minus_left"])))
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign", action="append", required=True)
    parser.add_argument("--g2", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    groups = [g for path in args.campaign for g in load_campaign(path)]
    pairs = [pair_groups(a, b) for a, b in itertools.combinations(groups, 2)
             if a["seed"] == b["seed"] and a["dataset_sha256"] is not None
             and a["dataset_sha256"] == b["dataset_sha256"] and a["preset"] != b["preset"]]
    summary = write_outputs(args.out, groups, read_g2(args.g2), pairs, strongest_references(groups))
    print(json.dumps({"groups": len(groups), "paired_tables": len(pairs), "strongest_references": summary["strongest_references"], "out": str(Path(args.out).resolve())}, indent=2))


if __name__ == "__main__":
    main()
