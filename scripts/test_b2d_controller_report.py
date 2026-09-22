"""Offline CLI, evaluator-result and partial-telemetry regression fixtures (no CARLA)."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("DATA_DIR", "/data")
import b2d_report as report
import b2d_route
import b2d_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, rid="1", attempt=1, completion=31, status="finished", **extra):
    adir = root / "attempts" / rid / str(attempt)
    record = {"status": status, "wall_s": 10, "profile": {"ticks": 4, "total_ms_mean": 50},
              "config": {"drive": "controller", "controller_preset": "pursuit", "tm_seed": 0}}
    record.update(extra)
    write(adir / "route_result.json", record)
    write(adir / "results.json", {"_checkpoint": {"records": [{
        "status": "Completed" if completion == 100 else "Failed - Agent blocked",
        "scores": {"score_route": completion, "score_composed": completion, "score_penalty": 1},
        "infractions": {"min_speed_infractions": ["Average speed is 40.2% of the surrounding traffic's one"],
                        "vehicle_blocked": ["blocked"] if completion < 100 else []}}]}})
    return adir


def control(frame, error=0.0):
    return {"frame": frame, "sim_time": frame * .05, "truth_frame": frame,
            "sensor_frames": {"GPS": frame, "IMU": frame, "SPEED": frame},
            "speed_mps": 2, "target_speed_mps": 3, "reference_speed_mps": 4,
            "trajectory_time": 0, "trajectory_age_s": .05,
            "throttle": .5, "steer": 0, "brake": 0,
            "truth_cross_track_m": error, "route_cross_track_m": 50,
            "raw_pose_error_m": .3, "pose_error_m": .1, "pose_heading_error_rad": .01,
            "controller_step_ms": .2, "reason": "tracking"}


class ReportTests(unittest.TestCase):
    def test_finished_is_not_completion_or_minimum_speed_penalty(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp))
            row = report.collect(tmp)[0]
            self.assertTrue(row["harness_finished"])
            self.assertFalse(row["driving_completed"])
            self.assertEqual(row["completion"], 31)
            self.assertEqual(row["minimum_speed_percentages"], [40.2])
            self.assertEqual(row["minimum_speed_events"], 1)
            self.assertEqual(row["score_penalty"], 1)
            self.assertIsNone(row["minimum_speed_penalty"])
            self.assertEqual(row["telemetry_quality"]["state"], "missing")

    def test_partial_corrupt_missing_and_capped_results_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adir = fixture(root, capped=True)
            (adir / "results.json").write_text('{"_checkpoint":')
            write(root / "invocations/1.json", {"routes": [{"route_id": "2"}], "config": {
                "drive": "controller", "controller_preset": "pursuit", "tm_seed": 0}})
            rows = report.collect(tmp)
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["capped"])
            self.assertTrue(rows[0]["partial"])
            self.assertEqual(rows[0]["results_state"], "invalid_json")
            self.assertEqual(rows[1]["status"], "missing_attempt")
            self.assertEqual(report.controller_summary(rows)["pursuit/seed=0"]["missing_official_results"], 2)

    def test_first_finished_not_best_and_all_retry_wall_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, attempt=1, status="crashed", completion=0)
            fixture(root, attempt=2, completion=31)
            fixture(root, attempt=3, completion=100)
            rows = report.collect(tmp)
            summary = report.controller_summary(rows)["pursuit/seed=0"]
            self.assertEqual(summary["completion_mean_observed"], 31)
            self.assertEqual(summary["retries"], 2)
            self.assertEqual(summary["all_attempt_wall_s"], 30)
            self.assertEqual([r["selected_attempt"] for r in rows], [False, True, False])

    def test_telemetry_metrics_use_truth_and_validate_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            adir = fixture(Path(tmp))
            ticks = [control(1, .1), control(2, .3), control(2, .5), control(4, 99)]
            ticks[-1]["truth_frame"] = 3
            ticks[-1]["sensor_frames"]["GPS"] = 3
            ticks[-1]["brake"] = .2
            (adir / "control.jsonl").write_text("\n".join(json.dumps(t) for t in ticks) + '\n{"frame":')
            row = report.collect(tmp)[0]
            quality = row["telemetry_quality"]
            self.assertEqual(quality["state"], "partial")
            self.assertEqual(quality["invalid_lines"], 1)
            self.assertEqual(quality["nonmonotonic_frames"], 1)
            self.assertEqual(quality["missing_frames"], 1)
            self.assertEqual(quality["truth_frame_mismatch"], 1)
            self.assertEqual(quality["sensor_frame_mismatch"], 1)
            self.assertEqual(quality["invalid_controls"], 1)
            self.assertEqual(row["tracking"]["truth_cross_track_m"]["n"], 3)
            self.assertAlmostEqual(row["tracking"]["truth_cross_track_m"]["rms"], (.35 / 3) ** .5)
            self.assertEqual(row["tracking"]["route_cross_track_m"]["rms"], 50)
            self.assertEqual(row["tracking"]["command_speed_error_mps"]["rms"], 1)
            self.assertEqual(row["tracking"]["reference_speed_error_mps"]["rms"], 2)

    def test_pre_collision_requires_frame_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            adir = fixture(Path(tmp))
            official = {"infractions": {"collisions_vehicle": ["Agent collided at x=3"]}}
            (adir / "control.jsonl").write_text("\n".join(json.dumps(control(i, i)) for i in (1, 2, 3)))
            self.assertEqual(report.telemetry(adir, official)["before_collision_state"], "collision_frame_unavailable")
            write(adir / "criterion_events.json", {"state": "ok", "events": [{"type": "COLLISION_VEHICLE", "frame": 3}]})
            summary = report.telemetry(adir, official)
            self.assertEqual(summary["before_collision"]["truth_cross_track_m"]["n"], 2)
            self.assertEqual(summary["tracking"]["truth_cross_track_m"]["n"], 3)

    def test_live_server_age_does_not_divide_by_zero(self):
        self.assertIn("server age", report.server_age([{"server_age_routes": 0, "status": "running"}]))

    def test_weighted_and_route_equal_metrics_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rid, errors in (("1", [1]), ("2", [3, 3, 3])):
                adir = fixture(Path(tmp), rid=rid)
                (adir / "control.jsonl").write_text("\n".join(json.dumps(control(i, e)) for i, e in enumerate(errors)))
            summary = report.controller_summary(report.collect(tmp))["pursuit/seed=0"]
            metric = summary["metrics"]["truth_cross_track_m"]["tracking"]
            self.assertEqual(metric["route_mean_rms"], 2)
            self.assertAlmostEqual(metric["tick_weighted_rms"], 7 ** .5)


class CLIAndEventTests(unittest.TestCase):
    def test_runner_passes_options_to_route_and_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "controller.json"
            write(config, {"wheelbase": 2.7})
            args = b2d_run.parse_args(["--out", tmp, "--drive", "controller", "--controller-preset", "pursuit",
                                       "--cruise-mps", "6", "--tm-seed", "123", "--decimate", "4",
                                       "--controller-config", str(config)])
            runner = b2d_run.Runner(args, [("1", "Town01")])
            fixture(Path(tmp), completion=100)
            server = SimpleNamespace(tm_port=8000, port=2000, index=0, log="mock.log", routes_served=0, started_at=time.time())
            proc = SimpleNamespace(pid=99999999, returncode=0)
            with patch.object(runner, "supervise", return_value=None), \
                    patch.object(b2d_run.subprocess, "Popen", return_value=proc) as launch, \
                    patch.object(b2d_run, "port_free", return_value=True):
                runner.run_once(0, server, "1", 1)
            runner.events.close()
            route_args = b2d_route.parse_args(launch.call_args[0][0][2:])
            self.assertEqual(route_args.controller_preset, "pursuit")
            self.assertEqual(route_args.cruise_mps, 6)
            self.assertEqual(route_args.controller_config, str(config))
            self.assertEqual(route_args.decimate, 4)
            self.assertEqual(b2d_route._leaderboard_args(route_args, Path("cfg"), Path(tmp)).traffic_manager_seed, 123)

    def test_invalid_cli_fails_before_server_start(self):
        for flags in (["--cruise-mps", "nan"], ["--cruise-mps", "0"], ["--decimate", "0"],
                      ["--drive", "controller", "--policy", "gpu"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                b2d_run.parse_args(["--out", "/unused"] + flags)

    def test_event_snapshot_preserves_statistics_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = SimpleNamespace(get_dict=lambda: {"percentage": 50}, get_type=lambda: SimpleNamespace(name="MIN_SPEED_INFRACTION"),
                                    get_frame=lambda: 4, get_message=lambda: "speed 50%")
            stats = SimpleNamespace(_scenario=SimpleNamespace(get_criteria=lambda: [SimpleNamespace(events=[event])]),
                                    compute_route_statistics=lambda value: value * 2)
            b2d_route._capture_criterion_events(stats, Path(tmp))
            self.assertEqual(stats.compute_route_statistics(3), 6)
            self.assertEqual(json.loads((Path(tmp) / "criterion_events.json").read_text())["events"][0]["frame"], 4)


if __name__ == "__main__":
    unittest.main()
