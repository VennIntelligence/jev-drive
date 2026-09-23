"""Offline actor-to-rear transform check against future CARLA rear-axle truth."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from b2d_tfv6_coordinates import REAR_OFFSET_M, rear_waypoints

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import plot_style


def rear_world(row):
    truth = row["truth"]
    yaw = math.radians(truth["rotation"][2])
    origin = np.asarray(truth["location"][:2], dtype=float)
    return origin - REAR_OFFSET_M * np.array([math.cos(yaw), math.sin(yaw)]), yaw


def variants(actor):
    correct = rear_waypoints(actor)
    no_flip = rear_waypoints(actor * np.array([1.0, -1.0]))
    flipped = actor * np.array([1.0, -1.0])
    reversed_offset = 2.0 * flipped - correct
    return {"correct": correct, "no_y_flip": no_flip,
            "no_origin_shift": flipped, "reverse_shift": reversed_offset}


def rows_from_files(paths):
    rows = []
    for path in paths:
        with open(path) as stream:
            rows.extend(json.loads(line) for line in stream)
    return [row for row in rows if row["arm"] == "B" and row["truth"] and row["waypoint"]]


def classify_segment(row, future):
    yaw0 = math.radians(row["truth"]["rotation"][2])
    yaw1 = math.radians(future["truth"]["rotation"][2])
    # CARLA world y points right; positive yaw is a right turn.
    delta = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    if delta < -8:
        return "left"
    if delta > 8:
        return "right"
    if abs(delta) < 2:
        return "straight"
    return None


def analyze(paths, output):
    source = rows_from_files(paths)
    by_route = {}
    for row in source:
        by_route.setdefault(row["route"], []).append(row)
    errors = {segment: {variant: {h: [] for h in range(1, 9)} for variant in
               ("correct", "no_y_flip", "no_origin_shift", "reverse_shift")}
              for segment in ("straight", "left", "right")}
    segment_sources = {segment: [] for segment in errors}
    selected_route = {}
    for route, route_rows in by_route.items():
        by_time = {round(row["sim_time"], 2): row for row in route_rows}
        for row in route_rows:
            t = row["sim_time"]
            future_1s = by_time.get(round(t + 1.0, 2))
            if future_1s is None:
                continue
            segment = classify_segment(row, future_1s)
            if segment is None:
                continue
            selected_route.setdefault(segment, route)
            if route != selected_route[segment]:
                continue
            segment_sources[segment].append((route, t))
            current, yaw = rear_world(row)
            c, s = math.cos(yaw), math.sin(yaw)
            for name, local in variants(np.asarray(row["waypoint"], dtype=float)).items():
                world = current + np.column_stack((c * local[:, 0] + s * local[:, 1],
                                                   s * local[:, 0] - c * local[:, 1]))
                for h in range(1, 9):
                    future = by_time.get(round(t + 0.25 * h, 2))
                    if future is None:
                        continue
                    truth, _ = rear_world(future)
                    delta = world[h - 1] - truth
                    errors[segment][name][h].append((c * delta[0] + s * delta[1],
                                                     s * delta[0] - c * delta[1]))
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for segment, variants_ in errors.items():
        for name, horizons in variants_.items():
            for h, values in horizons.items():
                array = np.asarray(values, dtype=float).reshape(-1, 2)
                table.append({"segment": segment, "variant": name, "horizon_s": h * 0.25,
                              "n": len(array),
                              "long_abs_median_m": float(np.median(np.abs(array[:, 0]))) if len(array) else None,
                              "long_abs_p95_m": float(np.percentile(np.abs(array[:, 0]), 95)) if len(array) else None,
                              "lateral_abs_median_m": float(np.median(np.abs(array[:, 1]))) if len(array) else None,
                              "lateral_abs_p95_m": float(np.percentile(np.abs(array[:, 1]), 95)) if len(array) else None,
                              "lateral_signed_median_m": float(np.median(array[:, 1])) if len(array) else None})
    with open(output / "coordinate-check.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table[0]); writer.writeheader(); writer.writerows(table)
    (output / "coordinate-source.json").write_text(json.dumps(segment_sources, indent=2) + "\n")
    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN_IN, 2.65), sharex=True)
    colors = {"correct": plot_style.PALETTE["blue"], "no_y_flip": plot_style.PALETTE["vermillion"],
              "no_origin_shift": plot_style.PALETTE["orange"], "reverse_shift": plot_style.PALETTE["purple"]}
    for ax, segment in zip(axes, ("left", "right")):
        for name in colors:
            group = [row for row in table if row["segment"] == segment and row["variant"] == name]
            ax.plot([row["horizon_s"] for row in group], [row["lateral_abs_median_m"] for row in group],
                    label=name.replace("_", " "), color=colors[name])
        ax.set_title(f"{segment.title()} turn")
        ax.set_xlabel("Horizon (s)")
    axes[0].set_ylabel("Median absolute lateral error (m)")
    axes[1].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    plot_style.save(fig, output / "coordinate-check")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.frames, args.out)
