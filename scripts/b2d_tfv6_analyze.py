"""Frozen W2 summaries, route-cluster bootstrap and tracking/comfort figures."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import plot_style


PAIRS = (("C", "B"), ("A", "B"), ("D", "C"))
HORIZONS = (0.5, 1.0)
BOOTSTRAPS = 10000
REAR_M = 1.389


def csv_write(path, rows):
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0])
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(rows)


def rear_pose(row):
    truth = row["truth"]
    yaw = math.radians(truth["rotation"][2])
    origin = np.asarray(truth["location"][:2], dtype=float)
    return origin - REAR_M * np.array([math.cos(yaw), math.sin(yaw)]), yaw


def vector_error(source, future, horizon):
    r = np.asarray(source["rear_waypoint"], dtype=float)[int(horizon / .25) - 1]
    current, yaw = rear_pose(source)
    actual, _ = rear_pose(future)
    c, s = math.cos(yaw), math.sin(yaw)
    predicted = current + np.array([c * r[0] + s * r[1], s * r[0] - c * r[1]])
    delta = predicted - actual
    return c * delta[0] + s * delta[1], s * delta[0] - c * delta[1]


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def frame_metrics(frame_path, infraction_path, arm):
    with open(frame_path) as stream:
        frames = [json.loads(line) for line in stream]
    collision_step = math.inf
    if infraction_path.exists():
        for event in json.loads(infraction_path.read_text()).get("infractions", []):
            if "COLLISION" in event.get("event_type", "").upper():
                collision_step = min(collision_step, int(event["step"]))
    by_time = {round(row["sim_time"], 2): row for row in frames if row["truth"]}
    track = {(h, segment): {"long": [], "lateral": []}
             for h in HORIZONS for segment in ("pre", "post")}
    comfort = {segment: {key: [] for key in ("long_accel", "lateral_accel", "long_jerk",
                                              "lateral_jerk", "steer_rate")}
               for segment in ("pre", "post")}
    prior = None
    prior_accel = None
    for row in frames:
        if not row["truth"]:
            continue
        t, step = row["sim_time"], row["step"]
        segment = "pre" if step < collision_step else "post"
        points = row["rear_waypoint"]
        if arm in "BCD" and points is not None:
            planned_speed = np.linalg.norm(np.asarray(points[3]) - points[1]) / .5
            if planned_speed > 1:
                for horizon in HORIZONS:
                    future = by_time.get(round(t + horizon, 2))
                    if future is None:
                        continue
                    target_segment = "pre" if future["step"] < collision_step else "post"
                    long_err, lateral_err = vector_error(row, future, horizon)
                    track[(horizon, target_segment)]["long"].append(abs(long_err))
                    track[(horizon, target_segment)]["lateral"].append(abs(lateral_err))
        velocity = np.asarray(row["truth"]["velocity"][:2])
        if prior is not None:
            dt = t - prior["sim_time"]
            if 0 < dt < .2:
                yaw = math.radians(row["truth"]["rotation"][2])
                c, s = math.cos(yaw), math.sin(yaw)
                a = (velocity - np.asarray(prior["truth"]["velocity"][:2])) / dt
                local_a = np.array([c * a[0] + s * a[1], s * a[0] - c * a[1]])
                comfort[segment]["long_accel"].append(abs(local_a[0]))
                comfort[segment]["lateral_accel"].append(abs(local_a[1]))
                control = row["executed_control"]
                previous_control = prior["executed_control"]
                comfort[segment]["steer_rate"].append(abs(control["steer"] - previous_control["steer"]) / dt)
                if prior_accel is not None:
                    jerk = (local_a - prior_accel) / dt
                    comfort[segment]["long_jerk"].append(abs(jerk[0]))
                    comfort[segment]["lateral_jerk"].append(abs(jerk[1]))
                prior_accel = local_a
        prior = row
    output = []
    for (horizon, segment), values in track.items():
        output.append({"arm": arm, "segment": segment, "horizon_s": horizon,
                       "n": len(values["long"]),
                       "long_median_m": percentile(values["long"], 50),
                       "long_p95_m": percentile(values["long"], 95),
                       "lateral_median_m": percentile(values["lateral"], 50),
                       "lateral_p95_m": percentile(values["lateral"], 95),
                       **{key + "_p95": percentile(comfort[segment][key], 95)
                          for key in comfort[segment]}})
    return output


def bootstrap(paired, comparison, field="ds_diff"):
    rows = [r for r in paired if r["comparison"] == comparison and r[field] is not None]
    routes = sorted(set(r["route"] for r in rows), key=int)
    if not routes:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_pairs": 0, "n_routes": 0}
    route_values = {route: np.asarray([r[field] for r in rows if r["route"] == route])
                    for route in routes}
    values = np.asarray([r[field] for r in rows])
    rng = np.random.default_rng(20260923)
    samples = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        draw = rng.choice(routes, len(routes), replace=True)
        samples[i] = np.mean(np.concatenate([route_values[route] for route in draw]))
    return {"mean": float(np.mean(values)), "ci_low": float(np.percentile(samples, 2.5)),
            "ci_high": float(np.percentile(samples, 97.5)),
            "n_pairs": len(values), "n_routes": len(routes)}


def collect_done_items(run_roots, extra=None):
    # The level directory also contains preserved, voided attempts under
    # aborted/. Only the canonical case tree is eligible for analysis.
    paths = [path for root in run_roots
             for path in root.glob("cases/*/route-*/seed-*/*/done.json")]
    paths.extend(extra or ())
    items = []
    seen = set()
    for path in sorted(paths):
        item = json.loads(path.read_text())
        level, route, seed, arm = (item[key] for key in ("level", "route", "seed", "arm"))
        case_dir = path.parent
        expected_case = (case_dir.parents[4] / "cases" / str(level) /
                         f"route-{route}" / f"seed-{seed}" / str(arm))
        if case_dir.resolve() != expected_case.resolve():
            raise ValueError(f"Done path does not match case identity: {path}")
        expected_run = case_dir / f"attempt-{int(item['attempt'])}" / "run"
        if Path(item["run_dir"]).resolve() != expected_run.resolve():
            raise ValueError(f"Done run_dir does not match selected attempt: {path}")
        key = (str(level), str(route), int(seed), str(arm))
        if key in seen:
            raise ValueError(f"Duplicate case key {key}: {path}")
        seen.add(key)
        items.append(item)
    return items


def analyze(run_roots, output, extra=None):
    output.mkdir(parents=True, exist_ok=True)
    items = collect_done_items(run_roots, extra)
    cases, tracking = [], []
    for item in items:
        route, seed, arm, level = (item[key] for key in ("route", "seed", "arm", "level"))
        run_dir = Path(item["run_dir"])
        attempt_dir = run_dir.parent
        report_dir = run_dir / "attempts" / route / "1"
        result_path = report_dir / "results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Official result missing for {route}/{seed}/{arm}: {result_path}")
        results = json.loads(result_path.read_text())
        records = results["_checkpoint"]["records"]
        if len(records) != 1:
            raise ValueError(f"Expected one official record: {result_path}")
        record = records[0]
        score = record["scores"]
        infractions = record["infractions"]
        length_km = record["meta"]["route_length"] / 1000
        row = {"level": level, "route": route, "seed": seed, "arm": arm,
               "ds": score["score_composed"], "rc": score["score_route"],
               "completed": score["score_route"] >= 100,
               "sr": record["status"] in ("Completed", "Perfect") and
               all(not events for name, events in infractions.items() if name != "min_speed_infractions"),
               "wall_s": item["wall_s"], "infra_retries": item.get("infra_retries", 0),
               "length_km": length_km, "official_status": record["status"]}
        row.update({name + "_per_km": len(events) / length_km if length_km > 0 else None
                    for name, events in infractions.items()})
        cases.append(row)
        frame_path = attempt_dir / "frames.jsonl"
        if frame_path.exists():
            for metric in frame_metrics(frame_path, attempt_dir / "infractions.json", arm):
                tracking.append({"level": level, "route": route, "seed": seed, **metric})
    cases.sort(key=lambda r: (r["level"], int(r["route"]), r["seed"], r["arm"]))
    csv_write(output / "cases.csv", cases)
    csv_write(output / "tracking.csv", tracking)
    index = {(row["level"], row["route"], row["seed"], row["arm"]): row for row in cases}
    paired = []
    missing = []
    expected = {key[:3] for key in index}
    for root in run_roots:
        if root.name not in ("level1", "level1r", "level2") or not root.exists():
            continue
        level = root.name.removeprefix("level")
        xml_path = (Path(__file__).resolve().parents[1] /
                    "todos/2026-09-22-b2d-controller/results/holdout.xml"
                    if level == "2" else
                    Path("/data/runs/b2d/tfv6-repro/runtime/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml"))
        for node in ET.parse(xml_path).getroot():
            for seed in ((0,) if level == "1r" else (0, 1, 2)):
                expected.add((level, node.attrib["id"], seed))
    for level, route, seed in sorted(expected, key=lambda x: (x[0], int(x[1]), x[2])):
        arms_present = {arm for l, r, s, arm in index if (l, r, s) == (level, route, seed)}
        if arms_present != ({"A"} if level == "1r" else set("ABCD")):
            missing.append({"level": level, "route": route, "seed": seed,
                            "available_arms": sorted(arms_present)})
            continue
        for treatment, control in PAIRS:
            if level == "1r":
                continue
            left, right = index[(level, route, seed, treatment)], index[(level, route, seed, control)]
            differences = {key + "_diff": left[key] - right[key]
                           for key in left if key.endswith("_per_km")}
            paired.append({"level": level, "route": route, "seed": seed,
                           "comparison": f"{treatment}-{control}",
                           "ds_diff": left["ds"] - right["ds"],
                           "rc_diff": left["rc"] - right["rc"],
                           "completed_diff": int(left["completed"]) - int(right["completed"]),
                           "sr_diff": int(left["sr"]) - int(right["sr"]), **differences})
    csv_write(output / "paired.csv", paired)
    summary = {"missing_groups": missing, "bootstrap_draws": BOOTSTRAPS,
               "by_level": {level: {f"{a}-{b}": bootstrap([r for r in paired if r["level"] == level], f"{a}-{b}")
                                   for a, b in PAIRS} for level in ("smoke", "1", "2")},
               "combined": {f"{a}-{b}": bootstrap([r for r in paired if r["level"] in ("1", "2")], f"{a}-{b}")
                            for a, b in PAIRS}}
    secondary_fields = [key for key in paired[0] if key.endswith("_diff") and key != "ds_diff"] if paired else []
    summary["secondary"] = {
        level: {comparison: {field: bootstrap([r for r in paired if level == "combined" or r["level"] == level],
                                                 comparison, field)
                             for field in secondary_fields}
                for comparison in ("C-B", "A-B", "D-C")}
        for level in ("1", "2", "combined")}
    summary["arm_counts"] = {
        level: {arm: {"cases": sum(r["level"] == level and r["arm"] == arm for r in cases),
                      "completed": sum(r["level"] == level and r["arm"] == arm and r["completed"] for r in cases),
                      "sr": sum(r["level"] == level and r["arm"] == arm and r["sr"] for r in cases)}
                for arm in "ABCD"}
        for level in ("1", "1r", "2")}
    repeat = []
    for key, rerun in index.items():
        if key[0] == "1r":
            original = index.get(("1", key[1], key[2], "A"))
            if original is not None:
                repeat.append({"route": key[1], "seed": key[2],
                               "ds_diff": rerun["ds"] - original["ds"]})
    summary["same_seed_a_repeat"] = repeat
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_style.apply()
    routes = sorted({r["route"] for r in paired}, key=int)
    fig, ax = plt.subplots(figsize=(plot_style.DOUBLE_COLUMN_IN,
                                    max(3.0, .28 * len(routes) + 1.1)))
    colors = {"C-B": plot_style.PALETTE["blue"], "A-B": plot_style.PALETTE["vermillion"],
              "D-C": plot_style.PALETTE["green"]}
    for offset, (comparison, color) in enumerate(colors.items()):
        means, lower, upper = [], [], []
        for route in routes:
            values = np.asarray([r["ds_diff"] for r in paired if r["route"] == route and
                                 r["comparison"] == comparison])
            mean = float(np.mean(values))
            rng = np.random.default_rng(20260923 + int(route) + offset)
            sampled = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
            low, high = np.percentile(sampled, (2.5, 97.5))
            means.append(mean)
            lower.append(max(0., mean - low))
            upper.append(max(0., high - mean))
        ax.errorbar(means, np.arange(len(routes)) + (offset - 1) * .18,
                    xerr=np.asarray([lower, upper]), fmt="o", markersize=2.7,
                    elinewidth=.6, capsize=1.5, label=comparison, color=color)
    ax.axvline(0, color=plot_style.BASELINE, lw=.6)
    ax.set_yticks(np.arange(len(routes)), routes)
    ax.set_xlabel("Paired Driving Score difference (points)")
    ax.set_ylabel("Route")
    ax.legend()
    fig.tight_layout()
    plot_style.save(fig, output / "paired-ds")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(plot_style.SINGLE_COLUMN_IN, 2.55))
    for arm, color in (("B", plot_style.PALETTE["orange"]),
                       ("C", plot_style.PALETTE["blue"]), ("D", plot_style.PALETTE["green"])):
        values = [r for r in tracking if r["arm"] == arm and r["segment"] == "pre" and r["n"]]
        ax.plot(HORIZONS, [np.median([r["lateral_median_m"] for r in values if r["horizon_s"] == h])
                           if any(r["horizon_s"] == h for r in values) else np.nan for h in HORIZONS],
                marker="o", label=arm, color=color)
    ax.set_xlabel("Horizon (s)")
    ax.set_ylabel("Pre-collision median lateral error (m)")
    ax.legend()
    fig.tight_layout()
    plot_style.save(fig, output / "tracking")
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--extra", action="append", type=Path)
    args = parser.parse_args()
    analyze(args.runs, args.out, args.extra)
