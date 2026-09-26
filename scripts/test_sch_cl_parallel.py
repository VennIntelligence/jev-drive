#!/usr/bin/env python3
"""CPU-only isolation and admission tests; no simulator or process launch."""
import json
import fcntl
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sch_cl_parallel as S


class ParallelPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, value in {"DATA": self.root, "MAIN": self.root / "runs/nq3/b",
                            "PILOT": self.root / "runs/sched/pilot/b",
                            "CONTROL": self.root / "runs/sched/pilot/b/parallel"}.items():
            patcher = patch.object(S, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.slot = dict(name="a", gpu=1, workers=2, idx=390, span=10, cpus="110-117", model_gb=30,
                         table_lane="sch-pilot-b")
        self.row = dict(lane="sch-pilot-b", gpus="1", workers="5", idx0="390", idx_span="30",
                        cpus="110-117", status="pilot", go="-")

    def test_explicit_outputs_and_control_namespaces(self):
        with patch.object(S, "route_ids", return_value=[str(x) for x in range(10)]):
            first, env1, out1 = S.invocation("cl5", self.slot, True)
            ten, env2, out2 = S.invocation("cl5", self.slot)
            _, env3, out3 = S.invocation("cl6", self.slot)
        self.assertNotEqual(out1, out2)
        self.assertNotEqual(out2, out3)
        self.assertEqual(first[-1], str(out1))
        self.assertEqual(ten[-1], str(out2))
        self.assertEqual(first[4], "0")
        self.assertEqual(ten[4], ",".join(str(x) for x in range(10)))
        self.assertEqual(env1["WORKERS"], "1")
        self.assertEqual(env2["WORKERS"], "2")
        self.assertNotEqual(env2["B_DIR"], env3["B_DIR"])
        self.assertEqual(env2["B_REPORT"], "0")
        self.assertTrue(env2["RUN_FLAGS"].endswith("--index-span 10"))

    def test_slot_conflicts_and_reservation_limit(self):
        second = dict(self.slot, name="b", idx=400)
        S.validate_slots([self.slot, second], [self.row])
        for bad in (dict(second, idx=395), dict(second, idx=420), dict(second, gpu=0),
                    dict(second, workers=4)):
            with self.assertRaises(ValueError):
                S.validate_slots([self.slot, bad], [self.row])

    def test_pending_workers_count_before_servers_exist(self):
        rows = [self.row, dict(self.row, lane="batch", gpus="0,2", workers="18")]
        probe = dict(pids=9000, cores_used=60, gpus=[dict(gpu=1, carla=20, used_gb=10)])
        ok, why = S.capacity([self.slot], {"a"}, rows, probe)
        self.assertFalse(ok)
        self.assertIn("pending-inclusive", why)
        probe["gpus"][0]["carla"] = 41
        self.assertTrue(S.capacity([self.slot], {"a"}, rows, probe)[0])
        probe["gpus"][0]["used_gb"] = 50
        self.assertFalse(S.capacity([self.slot], {"a"}, rows, probe)[0])

    def test_fail_skip_preserves_t10_and_does_not_skip_running_arm(self):
        S.MAIN.mkdir(parents=True)
        (S.DATA / "runs/sched").mkdir(parents=True)
        target = S.MAIN / "SKIP"
        target.write_text("cl2 0\ncl5d 0\n")
        value = dict(verdict="FAIL", fail=["crash rate > 0.25"], dir="example")
        S.escalate("cl6", value, True)
        self.assertEqual(target.read_text(), "cl2 0\ncl5d 0\n")
        S.escalate("cl6", value, False)
        S.escalate("cl6", value, False)
        self.assertEqual(target.read_text().splitlines(), ["cl2 0", "cl5d 0", "cl6 0", "cl6 1", "cl6 2"])
        requested = S.MAIN / "arms/cl8/s0/requested.json"
        requested.parent.mkdir(parents=True)
        requested.write_text("[]")
        S.escalate("cl8", value, False)
        self.assertNotIn("cl8", target.read_text())

    def test_existing_verdict_or_skip_prevents_launch(self):
        S.MAIN.mkdir(parents=True)
        (S.MAIN / "SKIP").write_text("cl5 0\n")
        with patch.object(S, "run_command", side_effect=AssertionError("must not launch")):
            self.assertEqual(S.pilot("cl5", self.slot)["state"], "already settled")
        final = S.PILOT / "arms/cl6/s0/verdict.json"
        final.parent.mkdir(parents=True)
        final.write_text(json.dumps({"verdict": "PASS"}))
        with patch.object(S, "run_command", side_effect=AssertionError("must not launch")):
            self.assertEqual(S.pilot("cl6", self.slot)["state"], "already settled")

    def test_another_owner_claim_prevents_launch(self):
        claim = S.CONTROL / "claims/cl5.lock"
        claim.parent.mkdir(parents=True)
        with claim.open("a+") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(S, "run_command", side_effect=AssertionError("must not launch")):
                self.assertEqual(S.pilot("cl5", self.slot)["state"], "claimed elsewhere")

    def test_bad_runner_cannot_publish_pass(self):
        out = S.PILOT / "arms/cl5/s0"
        out.mkdir(parents=True)
        pending = out / "verdict.pending.json"
        pending.write_text(json.dumps({"verdict": "PASS"}))
        with patch.object(S.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")):
            with self.assertRaises(RuntimeError):
                S.checklist(out, runner_rc=1)
        self.assertFalse((out / "verdict.json").exists())

    def test_one_route_flag_stops_before_ten(self):
        calls = []

        def runner(cmd, env, logfile):
            calls.append((cmd, env))
            out = Path(cmd[-1])
            out.mkdir(parents=True)
            (out / "requested.json").write_text('["0"]')
            return 0

        with patch.object(S, "route_ids", return_value=[str(x) for x in range(10)]), \
                patch.object(S, "port_block_free", return_value=True), \
                patch.object(S, "run_command", side_effect=runner), \
                patch.object(S, "checklist", return_value={"verdict": "FLAG", "dir": "example"}), \
                patch.object(S, "escalate") as escalation:
            result = S.pilot("cl5", self.slot)
        self.assertEqual(result["state"], "FLAG")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["WORKERS"], "1")
        self.assertTrue(escalation.call_args.args[-1])
        self.assertFalse((S.PILOT / "arms/cl5/s0/verdict.json").exists())


if __name__ == "__main__":
    unittest.main()
