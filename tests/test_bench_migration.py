"""Regression checks for lane migration: identities, parameter transport, legacy readers and gate decisions. No GPU needed."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from jevdrive import bench
from jevdrive.bench import compat as C, hugsim as H, models as M, navsim as N, runner as R, tables as T

REPO = M.REPO


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments/op_parity/scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Migration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        patch = mock.patch.dict(os.environ, DATA_DIR=str(self.d))
        patch.start()
        self.addCleanup(patch.stop)

    def raw(self, d, scenario="x", tag="bench"):
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(tag=tag, scenario=scenario, end="complete", run_dir=str(d / "trace"))]).to_csv(d / "results.csv", index=False)

    def units(self, name, scores=(0.8, 0.9), dac=(1.0, 1.0)):
        d = bench.run_dir(name, "navtest")
        d.mkdir(parents=True, exist_ok=True)
        u = pd.DataFrame(dict(token=["a", "b"], log=["log-a", "log-b"], score=scores))
        for key in N.SUBS:
            u[key] = dac if key == "DAC" else 1.0
        u.to_csv(d / "units.csv", index=False)
        (d / "DONE").write_text("done")
        return d

    def test_identity_and_aliases(self):
        m = M.resolve("cinque")
        self.assertEqual(H.run_key(m, "spec"), "cinque_spec")
        self.assertEqual(H.run_key(m, "opctrl_d118"), "cinque_spec")
        a = H.run_key(m, "spec", opts={"resume": {}, "desire": True})
        self.assertEqual(a, H.run_key(m, "spec", opts='{"desire": true, "resume": {}}'))
        variants = [H.run_key(m, "spec"), a, H.run_key(m, "spec", repeat="0"), H.run_key(m, "spec", repeat="1"),
                    H.run_key(m, "spec", controller_env={"OP_CTRL": {"delay": 0.3}}),
                    H.run_key(m, "exam", controller="ideal"), H.run_key(m, "exam", onnx=str(self.d / "custom.onnx"))]
        self.assertEqual(len(variants), len(set(variants)))
        for kw in (dict(opts=[]), dict(opts={"parity": {"socket": "foreign"}}), dict(repeat="../bad"),
                   dict(controller_env={"CUDA_VISIBLE_DEVICES": "1"}), dict(controller="ideal")):
            with self.assertRaises(ValueError):
                H.configuration("spec", **kw)

    def test_presets_configuration_and_resume(self):
        d = self.d / "run"
        with mock.patch.object(R, "box_cards", return_value=[0]):
            stages = H.stages(M.resolve("cinque"), "spec_cold", d, ["a.yaml"], opts={"resume": {}}, repeat="1")
        cfg = json.loads((d / "config.json").read_text())
        self.assertEqual(cfg["resolved_opts"]["warmup_s"], 0)
        self.assertEqual(cfg["resolved_opts"]["resume"], {})
        self.assertEqual(cfg["resolved_controller_env"]["OP_CTRL"]["delay"], 0.25)
        self.raw(d, "a")
        (d / "DONE").write_text("done")
        with mock.patch.object(R, "box_cards", return_value=[0]):
            again = H.stages(M.resolve("cinque"), "spec_cold", d, ["a.yaml"], opts={"resume": {}}, repeat="1")
            self.assertEqual([s.name for s in again], ["collect"])
            more = H.stages(M.resolve("cinque"), "spec_cold", d, ["b.yaml"], opts={"resume": {}}, repeat="1")
        self.assertEqual([s.name for s in more], ["w0", "collect"])
        self.assertFalse((d / "DONE").exists())
        with self.assertRaises(ValueError):
            H.stages(M.resolve("cinque"), "spec_cold", d, ["a.yaml"], opts={}, repeat="1")
        for preset in ("spec_hold", "spec_plan_smooth", "spec_plan_mpc"):
            self.assertEqual(H.preset_args(preset)[1], preset)
        self.assertEqual(stages[0].vram, 16.5)

    def test_rule_transport_merges_parity_and_clears_ambient_controller(self):
        d = self.d / "run"
        self.raw(d)
        (d / "workers").mkdir()
        cfg = dict(agent="cinque", preset="exam", controller="opctrl_long", opts={"op_ctrl": True, "op_long": True,
                    "resume": {}, "parity": {"ego": False}}, controller_env={"OP_CTRL_LONG": {"max_accel": 2}}, timeout_s=10, stall_s=10)
        srv = mock.Mock(bias_sock="/tmp/bias", op_sock="/tmp/policy")
        p = mock.Mock()
        p.poll.return_value = 0
        with mock.patch.object(H, "scenario_dir", return_value=d), mock.patch("jevdrive.cl.procs.capture"), \
                mock.patch.object(H.subprocess, "Popen", return_value=p) as popen, mock.patch.dict(os.environ, OP_CTRL='{"delay": 99}'):
            self.assertEqual(H.run_one(cfg, d, "x.yaml", srv, "0", d / "log"), "done")
        argv = popen.call_args.args[0]
        opts = json.loads(argv[argv.index("--opts") + 1])
        self.assertEqual(opts["resume"], {})
        self.assertEqual(opts["parity"], {"ego": False, "socket": "/tmp/bias"})
        self.assertEqual(argv[argv.index("--controller") + 1], "opctrl_long")
        self.assertEqual(json.loads(argv[argv.index("--controller-env") + 1]), cfg["controller_env"])
        self.assertNotIn("OP_CTRL", popen.call_args.kwargs["env"])

    def test_zs_controller_env_overrides_preset(self):
        # Check the real zs_run job path without launching its simulator.
        path = REPO / "experiments/hugsim/archive/zs_run.py"
        spec = importlib.util.spec_from_file_location("zs_run_migration", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        scen = self.d / "nuscenes/x.yaml"
        scen.parent.mkdir()
        scen.write_text("scene_name: scene-x\nmode: easy_00\n")
        tag = self.d / "tag"
        tag.mkdir()
        a = SimpleNamespace(preset="spec_hold", controller=None, agent="cinque", gpu="0", socket="", timeout=1,
                            opts='{"resume": {}}', controller_env='{"OP_CTRL": {"delay": 0.7}}')
        mod.apply_preset(a)
        process = mock.Mock()
        process.wait.return_value = 0
        with mock.patch.object(mod, "unpack"), mock.patch.object(mod, "check_tree"), \
                mock.patch.object(mod.subprocess, "Popen", return_value=process) as popen:
            mod.run_job(a, scen, tag, False)
        env = popen.call_args.kwargs["env"]
        self.assertEqual(json.loads(env["OP_CTRL"]), {"delay": 0.7})
        opts = json.loads(env["HUGSIM_ZS_OPTS"])
        self.assertEqual((opts["op_clock"], opts["warmup_s"], opts["resume"]), ("hold", 0.0, {}))

    def test_reader_prefers_bench_and_retains_legacy_schema(self):
        legacy = self.d / "old.csv"
        pd.DataFrame([dict(token="a", score=0.1)]).to_csv(legacy, index=False)
        self.assertEqual(C.navtest_csv("P0@warp", legacy), legacy)
        self.units("P0@warp")
        f = C.navtest_csv("P0@warp", legacy)
        t, avg = N.read_devkit_csv(f)
        self.assertEqual(t.index.tolist(), ["a", "b"])
        np.testing.assert_array_equal(t[N.SUBS["DAC"]], [1, 1])
        self.assertAlmostEqual(avg.score, 0.85)
        E = load_script("pp_eval")
        E.FRAMES = "warp"
        self.assertEqual(E.eval_csv("P0"), f)
        pred = N.pred_file(M.resolve("P0@warp"), "navtest")
        pred.parent.mkdir(parents=True)
        pred.touch()
        self.assertEqual(E.pred_file("P0"), pred)

    def test_gate_same_verdict_from_legacy_and_bench(self):
        E, G = load_script("pp_eval"), load_script("pp_hinge_report")
        idx = [dict(token=t, log_name=f"log-{t}") for t in ("a", "b")]
        args = mock.Mock(new="P2H10-F-s0", ref="P2-F-s0")
        with mock.patch.dict(sys.modules, pp_eval=E), mock.patch("jevdrive.navsim_zs.load_index", return_value=idx), mock.patch("builtins.print"):
            for score, dac, verdict in ((0.9, (1, 1), 0), (0.89, (1, 1), 1), (0.9, (0, 1), 2)):
                for model, sc, dc in ((args.ref, 0.9, (0, 1)), (args.new, score, dac)):
                    d = self.units(model, (sc, sc), dc)
                    legacy = self.d / "runs/navsim/eval" / f"v2_navtest_opi_lb_navtest_warp-cinque_PP{model}__base" / "time/scores.csv"
                    legacy.parent.mkdir(parents=True, exist_ok=True)
                    pd.read_csv(d / "units.csv").rename(columns=N.SUBS).to_csv(legacy, index=False)
                    (d / "DONE").unlink()
                with self.assertRaises(SystemExit) as old:
                    G.cmd_gate(args)
                for model in (args.new, args.ref):
                    (bench.run_dir(model, "navtest") / "DONE").write_text("done")
                with self.assertRaises(SystemExit) as new:
                    G.cmd_gate(args)
                self.assertEqual((old.exception.code, new.exception.code), (verdict, verdict))

    def test_raw_reader_and_publisher_do_not_mix_rule_identities(self):
        d = bench.run_dir("P0", "hugsim", "spec")
        self.raw(d)
        self.assertEqual(C.parity_hugsim_rows({"pp-spec-P0": ("spec", "P0")})[0]["tag"], "pp-spec-P0")
        out = self.d / "legacy"
        with self.assertRaises(RuntimeError):
            C.publish_hugsim(d, out, "rule")
        (d / "DONE").write_text("done")
        (d / "summary.json").write_text('{"missing": []}')
        C.publish_hugsim(d, out, "rule")
        C.publish_hugsim(d, out, "rule")
        self.assertEqual(len(C.read_rows(out / "results.csv")), 1)
        other = bench.run_dir("P0", "hugsim", "spec", opts={"resume": {}})
        self.raw(other, "y")
        (other / "DONE").write_text("done")
        (other / "summary.json").write_text('{"missing": []}')
        C.publish_hugsim(other, out, "rule")
        self.assertEqual([r["scenario"] for r in C.read_rows(out / "results.csv")], ["y"])

    def test_custom_reports_never_fall_back_to_baseline(self):
        d = bench.run_dir("cinque", "hugsim", "spec")
        d.mkdir(parents=True)
        pd.DataFrame([dict(scenario="x", hdscore=0.4)]).to_csv(d / "units.csv", index=False)
        self.assertEqual(T.load("hugsim", "run:cinque_spec")[0].index.tolist(), ["x"])
        self.assertIsNone(T.load("hugsim", "cinque", "spec", opts={"resume": {}})[0])
        with self.assertRaises(ValueError):
            T.load("hugsim", "run:../cinque_spec")

    def test_incomplete_collect_fails(self):
        d = self.d / "run"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(dict(model="cinque", preset="exam", scenarios=["x.yaml"])))
        with mock.patch.object(H, "routes", return_value={}), self.assertRaises(RuntimeError):
            H.collect(str(d))
        self.assertFalse((d / "DONE").exists())
        self.assertTrue((d / "ERROR").exists())

    def test_deadline_cancels_only_requested_jobs(self):
        d = self.d / "run"
        d.mkdir()
        (d / "jobs.json").write_text('{"w0": "ours", "collect": "after-ours"}')
        with mock.patch.object(R, "state", return_value=dict(w0="running", collect="queued")), \
                mock.patch.object(R.time, "monotonic", side_effect=[0, 2]), mock.patch("jevdrive.cl.pool.cancel") as cancel:
            self.assertFalse(R.wait([d], timeout_s=1, quiet=True))
        self.assertEqual([c.args[0] for c in cancel.call_args_list], ["ours", "after-ours"])
        self.assertTrue((d / "WAIT_TIMEOUT").exists())

    def test_shell_adapter_parameter_transport(self):
        fake = self.d / "fake.py"
        fake.write_text('import json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        script = f'''source scripts/bench_lane.sh
B=("{sys.executable}" "{fake}")
OUT="{self.d}/out"
bench_hugsim cinque exam rule spin10 2 lowspeed '{{"resume": {{}}}}'
'''
        env = dict(os.environ, LOWSPEED_CTRL='{"jerk": 5}', CL_POOL_JOB="existing-job", BENCH_REPEAT="force-123")
        result = subprocess.run(["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, check=True)
        args = json.loads(result.stdout)
        self.assertEqual(json.loads(args[args.index("--opts") + 1]), {"resume": {}})
        self.assertEqual(json.loads(args[args.index("--controller-env") + 1]), {"LOWSPEED_CTRL": {"jerk": 5}})
        self.assertIn("--in-pool", args)
        self.assertEqual(args[args.index("--publish-tag") + 1], "rule")
        self.assertEqual(args[args.index("--repeat") + 1], "force-123")

    def test_leased_worker_uses_shared_stages_without_submitting(self):
        d = self.d / "leased"
        done = self.d / "already-done"
        done.touch()
        stages = [R.Stage("onnx", ["existing"], done=str(done)), R.Stage("w0", ["shared-worker"], done=str(d / "worker")),
                  R.Stage("collect", ["shared-collector"], done=str(d / "DONE"))]
        with mock.patch.dict(os.environ, CL_POOL_JOB="lease"), mock.patch.object(bench, "plan", return_value=(d, stages)) as plan, \
                mock.patch.object(C.subprocess, "run") as run, mock.patch("jevdrive.cl.pool.submit") as submit:
            self.assertEqual(C.in_pool("P0", "hugsim", workers=2), d)
        self.assertEqual(plan.call_args.kwargs["jobs"], 1)
        self.assertEqual([c.args[0] for c in run.call_args_list], [["shared-worker"], ["shared-collector"]])
        submit.assert_not_called()

    def test_cli_independent_repeats(self):
        from jevdrive.bench.__main__ import main
        with mock.patch.object(bench, "run", return_value=self.d) as run, mock.patch("builtins.print"):
            self.assertEqual(main(["run", "--model", "P0", "--bench", "hugsim", "--preset", "spec_plan_smooth",
                                   "--repeat", "r0", "r1", "r0", "--opts", '{"resume": {}}']), 0)
        self.assertEqual([c.kwargs["repeat"] for c in run.call_args_list], ["r0", "r1"])
        self.assertTrue(all(c.kwargs["preset"] == "spec_plan_smooth" for c in run.call_args_list))

    def test_long_pool_names_keep_configuration_and_stage(self):
        stages = [R.Stage("w0", ["worker"], done=str(self.d / "worker")),
                  R.Stage("collect", ["collect"], done=str(self.d / "DONE"), after=["w0"])]
        with mock.patch("jevdrive.cl.pool.submit", side_effect=["a", "b", "c", "d"]) as submit:
            for i in (0, 1):
                R.submit(self.d / str(i), "bn-hugsim-" + "long-model-" * 6 + f"-r{i}", stages)
        names = [c.kwargs["name"] for c in submit.call_args_list]
        self.assertEqual(len(set(names)), 4)
        self.assertTrue(all(len(n) <= 60 for n in names))
        self.assertTrue(names[0].endswith("-w0") and names[1].endswith("-collect"))

    def test_h_export_is_published_only_after_equivalence_gate(self):
        m = M.Model("H-test", "adapt_h", ckpt="checkpoint", benches=("hugsim",))
        out = self.d / "H-test.onnx"
        distance = "0.6000"
        def check_run(argv, **kw):
            if "build" in argv:
                Path(argv[argv.index("--out") + 1]).write_text("onnx")
            if "stdout" in kw:
                kw["stdout"].write(f"stream 0 frames 8 | all cols max 1.0000 p99 0.01000 | plan xy max {distance} m, mean dist 0.01000 m\n")
        with mock.patch.object(H.subprocess, "run", side_effect=check_run):
            for distance in ("0.6000", "nan", "inf"):
                with self.assertRaises(RuntimeError):
                    H.build_h_onnx(m, out)
                self.assertFalse(out.exists())
            distance = "0.1000"
            H.build_h_onnx(m, out)
        self.assertEqual(out.read_text(), "onnx")

    def test_trace_readers_follow_canonical_exports(self):
        d = self.d / "canonical/bench/zs/scene"
        d.mkdir(parents=True)
        (d / "zs_steps.jsonl").write_text('{"derot":{"dpos":0.1}}\n')
        out = self.d / "legacy"
        out.mkdir()
        row = dict(tag="legacy-arm", scenario="scene", end="complete", run_dir=str(d))
        pd.DataFrame([row]).to_csv(out / "results.csv", index=False)
        path = REPO / "experiments/hugsim/scripts/derot_report.py"
        spec = importlib.util.spec_from_file_location("derot_report_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.runs(out / "results.csv", out, {"legacy-arm"})[("legacy-arm", "scene")][1], d)
        proc = subprocess.run([sys.executable, str(path.with_name("sel3_window_report.py")), str(out)],
                              capture_output=True, text=True, check=True)
        self.assertIn("replays 1 in 1 scenes", proc.stdout)
        relocated = dict(row, run_dir="/missing/old/zs/scene")
        self.assertEqual(C.trace_dir(relocated, out), out / "legacy-arm/zs/scene")


if __name__ == "__main__":
    unittest.main()
