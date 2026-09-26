#!/usr/bin/env python3
"""Count independent pedestrian-in-corridor events in OpenScene log pickles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SPEED_BINS = ((0.0, 2.0, "0–2"), (2.0, 5.0, "2–5"), (5.0, 10.0, "5–10"), (10.0, math.inf, ">10"))
MAX_PATH_M = 30.0
MAX_LATERAL_M = 4.0


def _yaw_from_quaternion(q: Iterable[float]) -> float:
    """Return yaw for a w, x, y, z quaternion."""
    w, x, y, z = map(float, q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _speed_bin(speed: float) -> str:
    for lo, hi, label in SPEED_BINS:
        if lo <= speed < hi:
            return label
    raise AssertionError(f"Invalid speed: {speed}")


def _path_to_30m(xy: np.ndarray, cumulative: np.ndarray, frame_idx: int, segment_end: int | None = None) -> np.ndarray:
    """Build the logged future polyline, clipped at 30 m of traveled arclength."""
    target = cumulative[frame_idx] + MAX_PATH_M
    stop = int(np.searchsorted(cumulative, target, side="left"))
    stop = min(stop, len(xy) - 1 if segment_end is None else segment_end)
    path = xy[frame_idx : stop + 1].copy()
    if cumulative[stop] > target and stop > frame_idx:
        seg_len = cumulative[stop] - cumulative[stop - 1]
        if seg_len > 0:
            alpha = (target - cumulative[stop - 1]) / seg_len
            path[-1] = xy[stop - 1] + alpha * (xy[stop] - xy[stop - 1])
    return path


def _project_to_path(points: np.ndarray, path: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Return arclength and unsigned lateral distance for points in the current ego frame."""
    if len(points) == 0 or len(path) < 2:
        return np.full(len(points), np.nan), np.full(len(points), np.inf)

    delta = path - path[0]
    c, s = math.cos(yaw), math.sin(yaw)
    local_path = np.column_stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]))
    starts, vectors = local_path[:-1], np.diff(local_path, axis=0)
    length_sq = np.einsum("ij,ij->i", vectors, vectors)
    valid = length_sq > 1e-10
    if not np.any(valid):
        return np.full(len(points), np.nan), np.full(len(points), np.inf)

    starts, vectors, length_sq = starts[valid], vectors[valid], length_sq[valid]
    seg_len = np.sqrt(length_sq)
    raw_seg_len = np.linalg.norm(np.diff(local_path, axis=0), axis=1)
    seg_start_s = np.concatenate(([0.0], np.cumsum(raw_seg_len)))[:-1][valid]
    rel = points[:, None, :] - starts[None, :, :]
    raw_fraction = np.einsum("pij,ij->pi", rel, vectors) / length_sq[None, :]
    fraction = np.clip(raw_fraction, 0.0, 1.0)
    closest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - closest, axis=2)
    nearest = np.argmin(distance, axis=1)
    row = np.arange(len(points))
    arclength = seg_start_s[nearest] + fraction[row, nearest] * seg_len[nearest]
    lateral = distance[row, nearest]
    # Endpoint caps must not turn points behind the ego or past the future path
    # into points inside its longitudinal interval.
    outside = ((nearest == 0) & (raw_fraction[row, nearest] < -1e-9)) | (
        (nearest == len(seg_len) - 1) & (raw_fraction[row, nearest] > 1.0 + 1e-9)
    )
    arclength[outside] = np.nan
    lateral[outside] = np.inf
    return arclength, lateral


def _is_consecutive(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if previous is None:
        return False
    previous_token = previous.get("token")
    sample_prev = current.get("sample_prev")
    if sample_prev not in (None, ""):
        return sample_prev == previous_token
    return int(current["timestamp"]) > int(previous["timestamp"])


def _count_log(task: tuple[str, str]) -> dict[str, Any]:
    split, filename = task
    path = Path(filename)
    with path.open("rb") as f:
        frames = pickle.load(f)

    if not frames:
        return {"split": split, "file": path.name, "frames": 0, "pedestrian_boxes": 0, "events": [], "errors": []}

    xy = np.asarray([row["ego2global_translation"][:2] for row in frames], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    segment_ends = np.empty(len(frames), dtype=np.int64)
    end = len(frames) - 1
    for idx in range(len(frames) - 1, -1, -1):
        if idx < len(frames) - 1 and not _is_consecutive(frames[idx], frames[idx + 1]):
            end = idx
        segment_ends[idx] = end
    active: set[str] = set()
    previous: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    pedestrian_boxes = 0
    errors: list[str] = []
    boundary_counts: Counter[str] = Counter()

    for frame_idx, row in enumerate(frames):
        if not _is_consecutive(previous, row):
            active.clear()
        anns = row["anns"]
        names = list(anns["gt_names"])
        pedestrian_idx = [idx for idx, name in enumerate(names) if str(name) == "pedestrian"]
        pedestrian_boxes += len(pedestrian_idx)
        current: set[str] = set()
        path_xy = _path_to_30m(xy, cumulative, frame_idx, int(segment_ends[frame_idx]))
        path_steps = np.diff(path_xy, axis=0)
        path_length = float(np.linalg.norm(path_steps, axis=1).sum())
        if not np.any(np.einsum("ij,ij->i", path_steps, path_steps) > 1e-10):
            boundary_counts["frames_without_valid_future_segment"] += 1
            boundary_counts["pedestrian_frames_without_valid_future_segment"] += bool(pedestrian_idx)
        elif path_length < MAX_PATH_M - 1e-9:
            boundary_counts["frames_with_short_future_path"] += 1
            boundary_counts["pedestrian_frames_with_short_future_path"] += bool(pedestrian_idx)
        if pedestrian_idx:
            boxes = np.asarray(anns["gt_boxes"])[pedestrian_idx]
            track_tokens = [str(anns["track_tokens"][idx]) for idx in pedestrian_idx]
            instance_tokens = [str(anns["instance_tokens"][idx]) for idx in pedestrian_idx]
            yaw = _yaw_from_quaternion(row["ego2global_rotation"])
            arclength, lateral = _project_to_path(boxes[:, :2], path_xy, yaw)
            speed = float(np.linalg.norm(np.asarray(row["ego_dynamic_state"][:2], dtype=np.float64)))
            for local_idx, token in enumerate(track_tokens):
                qualifies = (
                    np.isfinite(arclength[local_idx])
                    and -1e-9 <= arclength[local_idx] <= MAX_PATH_M + 1e-9
                    and lateral[local_idx] <= MAX_LATERAL_M + 1e-9
                )
                if not qualifies:
                    continue
                current.add(token)
                if token in active:
                    continue
                event_id = f"{split}:{row['log_name']}:{token}:{frame_idx}"
                events.append(
                    {
                        "event_id": event_id,
                        "split": split,
                        "log_name": str(row["log_name"]),
                        "city": str(row["map_location"]),
                        "track_token": token,
                        "instance_token": instance_tokens[local_idx],
                        "onset_frame_idx": frame_idx,
                        "onset_token": str(row["token"]),
                        "onset_timestamp_us": int(row["timestamp"]),
                        "ego_speed_mps": speed,
                        "speed_bin_mps": _speed_bin(speed),
                        "path_arclength_m": float(arclength[local_idx]),
                        "lateral_distance_m": float(lateral[local_idx]),
                        "box_x_m": float(boxes[local_idx, 0]),
                        "box_y_m": float(boxes[local_idx, 1]),
                    }
                )
        active = current
        previous = row

    return {
        "split": split,
        "file": path.name,
        "frames": len(frames),
        "pedestrian_boxes": pedestrian_boxes,
        "events": events,
        "errors": errors,
        "boundary_counts": dict(boundary_counts),
    }


class RunLog:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "log.txt"
        self.events_path = output_dir / "events.jsonl"

    def line(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
        text = f"[{stamp}] {message}"
        print(text, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def event(self, kind: str, **fields: Any) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, **fields}, ensure_ascii=False) + "\n")


def _write_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fields = [
        "event_id", "split", "log_name", "city", "track_token", "instance_token", "onset_frame_idx",
        "onset_token", "onset_timestamp_us", "ego_speed_mps", "speed_bin_mps", "path_arclength_m",
        "lateral_distance_m", "box_x_m", "box_y_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def _self_test() -> None:
    xy = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 10.0], [40.0, 10.0]])
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    path = _path_to_30m(xy, cumulative, 0)
    assert math.isclose(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 30.0)
    along, lateral = _project_to_path(np.asarray([[5.0, 3.0], [-1.0, 0.0]]), path, 0.0)
    assert math.isclose(along[0], 5.0) and math.isclose(lateral[0], 3.0)
    assert np.isnan(along[1]) and np.isinf(lateral[1])
    along, lateral = _project_to_path(
        np.asarray([[-1.0, 0.0], [0.0, 4.0], [30.0, 4.0], [31.0, 0.0]]),
        np.asarray([[0.0, 0.0], [30.0, 0.0]]), 0.0,
    )
    assert np.allclose(along[1:3], [0.0, 30.0])
    assert np.allclose(lateral[1:3], [4.0, 4.0])
    assert np.all(np.isnan(along[[0, 3]])) and np.all(np.isinf(lateral[[0, 3]]))
    short_path = _path_to_30m(xy, cumulative, 0, segment_end=1)
    assert np.array_equal(short_path, xy[:2])
    along, lateral = _project_to_path(np.asarray([[1.0, 0.0]]), np.zeros((3, 2)), 0.0)
    assert np.isnan(along[0]) and np.isinf(lateral[0])
    assert not _is_consecutive({"token": "a", "timestamp": 1}, {"sample_prev": "b", "timestamp": 2})
    assert [_speed_bin(x) for x in (0.0, 1.999, 2.0, 4.999, 5.0, 9.999, 10.0)] == [
        "0–2", "0–2", "2–5", "2–5", "5–10", "5–10", ">10"
    ]
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.logs_root is None or args.output_dir is None:
        parser.error("--logs-root and --output-dir are required")
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be 1 or 2")

    runlog = RunLog(args.output_dir)
    split_paths = sorted(p for p in args.logs_root.iterdir() if p.is_dir())
    tasks = [(split.name, str(path)) for split in split_paths for path in sorted(split.glob("*.pkl"))]
    inventory = {split.name: len(list(split.glob("*.pkl"))) for split in split_paths}
    runlog.line("Inventory before event counting: " + ", ".join(f"{key}={value} logs" for key, value in inventory.items()))
    runlog.line(
        "Operational definition: exact GT class pedestrian; box center projected onto the current ego's logged "
        "future rear-axle polyline clipped at 30 m arclength; qualify at arclength [0,30] m and lateral distance "
        "at most 4 m. One event is a maximal consecutive qualifying-frame run for one track token."
    )
    runlog.line("Speed bins: [0,2), [2,5), [5,10), [10,infinity) m/s; city is map_location metadata.")
    runlog.event("start", logs_root=str(args.logs_root), inventory=inventory, workers=args.workers)

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for idx, result in enumerate(pool.map(_count_log, tasks, chunksize=1), start=1):
            results.append(result)
            if idx % 25 == 0 or idx == len(tasks):
                elapsed = time.monotonic() - started
                rate = idx / elapsed if elapsed else 0.0
                eta = (len(tasks) - idx) / rate if rate else 0.0
                print(f"logs {idx}/{len(tasks)} rate={rate:.2f}/s eta={eta:.0f}s", flush=True)
                runlog.event("progress", completed_logs=idx, total_logs=len(tasks), rate_logs_s=rate, eta_s=eta)

    all_events = [event for result in results for event in result["events"]]
    all_events.sort(key=lambda row: (row["split"], row["log_name"], row["onset_timestamp_us"], row["track_token"]))
    duplicate_ids = len(all_events) - len({row["event_id"] for row in all_events})
    if duplicate_ids:
        raise RuntimeError(f"Duplicate event IDs: {duplicate_ids}")
    _write_csv(args.output_dir / "events.csv", all_events)

    by_city_speed = Counter((row["city"], row["speed_bin_mps"]) for row in all_events)
    metrics = {
        "inventory": inventory,
        "logs_processed": len(results),
        "frames_processed": sum(row["frames"] for row in results),
        "pedestrian_boxes": sum(row["pedestrian_boxes"] for row in results),
        "boundary_counts": dict(sum((Counter(row.get("boundary_counts", {})) for row in results), Counter())),
        "independent_events": len(all_events),
        "by_city_speed": [
            {"city": city, "speed_bin_mps": speed_bin, "events": count}
            for (city, speed_bin), count in sorted(by_city_speed.items())
        ],
        "elapsed_s": time.monotonic() - started,
        "definition": {
            "category": "pedestrian",
            "corridor_lateral_m": MAX_LATERAL_M,
            "corridor_forward_arclength_m": MAX_PATH_M,
            "event": "maximal consecutive qualifying-frame run per track token",
            "speed_bins_mps": ["[0,2)", "[2,5)", "[5,10)", "[10,infinity)"],
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")
    runlog.line(
        f"Completed: logs={metrics['logs_processed']} frames={metrics['frames_processed']} "
        f"pedestrian_boxes={metrics['pedestrian_boxes']} independent_events={metrics['independent_events']} "
        f"elapsed_s={metrics['elapsed_s']:.1f}"
    )
    runlog.event("end", status="ok", **{key: metrics[key] for key in ("logs_processed", "frames_processed", "independent_events", "elapsed_s")})


if __name__ == "__main__":
    main()
