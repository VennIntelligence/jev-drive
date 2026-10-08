"""jevdrive.bench without GPU / network / benchmarks: registry, sets, EPDMS algebra (vs the lane code it replaces), stage graph ->
pool inbox, the HUGSIM scenario queue and watchdog (Linux: needs /proc), the CARLA command, pose scoring (no devkit).

    python -m unittest tests.test_bench -v
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jevdrive.bench import models as M, runner as RN, sets as ST, tables as RP  # noqa: E402


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TmpData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.D = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"DATA_DIR": str(self.D)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class TestModels(TmpData):
    def test_resolve(self):
        m = M.resolve("cinque")
        self.assertEqual((m.family, m.frames, m.base, m.onnx), ("onnx", "gimm", "cinque", ""))
        self.assertEqual(M.resolve("wajepa").name, "WA-JEPA")
        p0 = M.resolve("P0@gimm")
        self.assertEqual((p0.family, p0.frames, p0.spec, p0.shipped), ("parity", "gimm", "P0@gimm", True))
        self.assertEqual(M.resolve("P0").frames, "warp")
        self.assertEqual(M.resolve("P2-F-s0").frames, "warp")
        self.assertEqual(M.resolve("P2-G-s1").frames, "gimm")
        self.assertEqual(M.resolve("P1-N-s0").frames, "keys")
        self.assertEqual(M.resolve("P2-s0").frames, "gimm")             # pilot arms: GIMM protocol
        self.assertEqual(M.resolve("UF-V-s0").frames, "vh140")
        n = M.resolve("P3-F-s0:noside")
        self.assertEqual((n.opt, n.key("navtest"), n.key("hugsim")), ("noside", "P3-F-s0@warp_noside", "P3-F-s0"))
        self.assertTrue(M.resolve("UF-U2-s0").unfreeze)
        self.assertTrue(M.resolve("P2-F-s0").ckpt.endswith("runs/op_parity/runs/P2-F-s0/ckpt-final.pt"))
        with self.assertRaises(ValueError):
            M.resolve("cinque:noside")
        with self.assertRaises(ValueError):
            M.resolve("P2-F-s0@bogus")
        with self.assertRaises(ValueError):
            M.resolve("nonsense")
        with self.assertRaises(FileNotFoundError):
            M.resolve("P2-F-s0", check=True)

    def test_guard_candidates(self):
        c = M.guard_candidates()
        plain = [k for k, v in c.items() if k != "shipped" and not (v.get("command_adapter") or v.get("route_adapter"))]
        if plain:
            m = M.resolve(plain[0])
            self.assertEqual(m.family, "onnx")
            self.assertTrue(m.onnx.startswith(str(self.D)))
        adapted = [k for k, v in c.items() if v.get("command_adapter")]
        if adapted:
            with self.assertRaises(ValueError):
                M.resolve(adapted[0])


class TestSets(unittest.TestCase):
    def test_hugsim_sets(self):
        self.assertEqual(len(ST.hugsim_scenarios("all64")), 64)
        self.assertEqual(len(ST.hugsim_scenarios("spin10")), 10)
        self.assertEqual(len(ST.hugsim_scenarios("spec29")), 29)
        self.assertEqual(len(ST.hugsim_scenarios("small11")), 11)
        t = ST.hugsim_scenarios("turn23")
        self.assertEqual(len(t), 23)
        self.assertTrue(set(t) <= set(ST.hugsim_scenarios("all64")))
        self.assertEqual(ST.hugsim_scenarios("scene-0013-medium-00"), ["nuscenes/scene-0013-medium-00.yaml"])
        with self.assertRaises(ValueError):
            ST.hugsim_scenarios("nope-nope")

    def test_log_shards(self):
        rng = np.random.default_rng(0)
        logs = np.array([f"log{i}" for i in rng.integers(0, 40, 3000)])
        sh = ST.log_shards(logs, 3)
        self.assertEqual(sh, ST.log_shards(logs, 3))                    # deterministic
        flat = [x for s in sh for x in s]
        self.assertEqual(sorted(flat), sorted(set(logs)))               # every log once
        sizes = [np.isin(logs, s).sum() for s in sh]
        self.assertLess(max(sizes) - min(sizes), 200)                   # balanced by tokens


class TestAlgebra(TmpData):
    def X(self, n=200, seed=0):
        r = np.random.default_rng(seed)
        X = r.uniform(0, 1, (n, 9))
        X[:, :4] = r.choice([0.0, 0.5, 1.0], (n, 4), p=[0.05, 0.05, 0.9])
        X[r.uniform(size=n) < 0.3, 8] = np.nan
        return X

    def test_shapley_matches_lane_and_sums(self):
        G = load_script("experiments/op_parity/scripts/pp_gap_tables.py", "pp_gap_tables_t")
        A, B = self.X(seed=1), self.X(seed=2)
        B[:, 8] = np.where(np.isnan(A[:, 8]), np.nan, np.nan_to_num(B[:, 8], nan=0.7))   # EC presence shared, as in the data
        np.testing.assert_array_equal(RP.score_of(A), G.score_of(A))
        phi = RP.shapley(A, B)
        np.testing.assert_array_equal(phi, G.shapley(A, B))
        np.testing.assert_allclose(phi.sum(1), RP.score_of(B) - RP.score_of(A), atol=1e-12)

    def test_motion_matches_opj_build(self):
        O = load_script("experiments/op_probe/scripts/opj_build.py", "opj_build_t")
        r = np.random.default_rng(3)
        for _ in range(300):
            v0 = float(r.uniform(0, 15))
            k = r.uniform(-0.15, 0.15)
            t = 0.5 * np.arange(1, 9)
            s = v0 * t + 0.5 * r.uniform(-2, 2) * t ** 2
            fut = np.stack([s * np.cos(k * s / 4), s * np.sin(k * s / 4) + r.uniform(-3, 3) * t / 4, k * s / 4], 1)
            d, man = RP.motion(fut, v0)
            ref = O.motion(fut, v0)
            self.assertEqual(man, ref["maneuver"])
            self.assertAlmostEqual(d, ref["dyaw"], places=12)
        self.assertEqual(RP.motion(np.full((8, 3), np.nan), 1.0)[1], "unknown")


class TestRunner(TmpData):
    def test_submit_graph(self):
        pool = self.D / "runs" / "pool"
        (pool / "inbox").mkdir(parents=True)
        rd = self.D / "runs" / "bench" / "navtest" / "X"
        done_out = rd / "already.npz"
        done_out.parent.mkdir(parents=True)
        done_out.write_text("x")
        st = [RN.Stage("plans", ["python", "-c", "1"], done=str(done_out), vram=24, cpu=8),
              RN.Stage("export", ["python", "-c", "1"], done=str(rd / "p.npz"), after=["plans"]),
              RN.Stage("s0", "echo {gpu}", done=str(rd / "s0.csv"), after=["export"], carla=2),
              RN.Stage("collect", ["python", "-c", "1"], done=str(rd / "DONE"), after=["s0", "export"])]
        ids = RN.submit(rd, "bn-test", st)
        self.assertEqual(ids["plans"], "done")
        inbox = {json.loads(f.read_text())["id"]: json.loads(f.read_text())["spec"] for f in (pool / "inbox").glob("*.json")}
        self.assertEqual(len(inbox), 3)
        self.assertEqual(inbox[ids["export"]]["after"], [])             # the finished stage is not a dependency
        self.assertEqual(inbox[ids["s0"]]["after"], [ids["export"]])
        self.assertEqual(inbox[ids["s0"]]["carla"], 2)
        self.assertEqual(sorted(inbox[ids["collect"]]["after"]), sorted([ids["s0"], ids["export"]]))
        self.assertTrue(inbox[ids["export"]]["log_dir"].endswith("pool/export"))
        again = RN.submit(rd, "bn-test", st)                            # idempotent: live jobs are reused
        self.assertEqual(again, ids)
        self.assertEqual(len(list((pool / "inbox").glob("*.json"))), 3)
        with self.assertRaises(ValueError):
            RN.submit(rd, "bn-test", [RN.Stage("a", "x", done=str(rd / "a"), after=["missing"])])


class TestHugsim(TmpData):
    def setUp(self):
        super().setUp()
        from jevdrive.bench import hugsim as H
        self.H = H
        self.rd = self.D / "run"
        self.rd.mkdir()

    def test_stages_lpt_and_sizing(self):
        H = self.H
        scen = ST.hugsim_scenarios("spin10")
        walls = {Path(s).stem: float(i) for i, s in enumerate(scen)}
        with mock.patch.object(H, "expected_walls", return_value=walls), mock.patch.object(RN, "box_cards", return_value=[0, 1, 2]):
            st = H.stages(M.resolve("P2-F-s0"), "spec", self.rd, scen, workers=4)
        cfg = json.loads((self.rd / "config.json").read_text())
        self.assertEqual(cfg["scenarios"], scen[::-1])                  # longest expected first
        self.assertEqual([s.name for s in st], ["onnx", "w0", "w1", "w2", "collect"])
        self.assertEqual(st[1].after, ["onnx"])
        self.assertEqual(st[-1].after, ["w0", "w1", "w2"])
        with mock.patch.object(H, "expected_walls", return_value={}), mock.patch.object(RN, "box_cards", return_value=[0, 1, 2]):
            st = H.stages(M.resolve("cinque"), "exam", self.D / "r2", scen[:3], workers=6)
        self.assertEqual([s.name for s in st], ["w0", "collect"])      # 3 scenarios: one job of 3 slots
        self.assertEqual(json.loads((self.D / "r2" / "config.json").read_text())["workers"], 3)
        with self.assertRaises(SystemExit):
            H.stages(M.resolve("WA-JEPA"), "spec", self.D / "r3", scen)

    def test_queue(self):
        H = self.H
        scen = ["nuscenes/a.yaml", "nuscenes/b.yaml", "nuscenes/c.yaml"]
        q0, q1 = H.Queue(self.rd, scen, 0), H.Queue(self.rd, scen, 1)
        self.assertEqual(q0.next(), "nuscenes/a.yaml")
        self.assertEqual(q1.next(), "nuscenes/b.yaml")
        with open(self.rd / "results.csv", "w") as f:
            f.write("scenario,tag,end\nc,bench,complete\n")
        self.assertIsNone(q1.next())                                    # c is done, a / b are claimed
        q1.release("nuscenes/b.yaml", failed="stall after 3 attempts")
        self.assertIsNone(q0.next())                                    # b failed in this submission
        old = time.time() - 1000
        os.utime(self.rd / "claims" / "a.claim", (old, old))
        self.assertEqual(q1.next(), "nuscenes/a.yaml")                  # stale claim taken over
        H.Queue(self.rd, scen, 1)                                       # a restarted worker 1 drops its own claims
        self.assertFalse((self.rd / "claims" / "a.claim").exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "jevdrive.cl.procs needs /proc")
    def test_watchdog(self):
        H = self.H
        sc = self.D / "datasets/hugsim/scenarios/nuscenes"
        sc.mkdir(parents=True)
        (sc / "x.yaml").write_text("scene_name: scene-x\nmode: easy_00\n")
        fake = self.D / "fake_zs_run.py"
        fake.write_text("import sys, time, pathlib\nout = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                        "d = out / 'bench' / 'zs' / 'scene-x_easy_00'; d.mkdir(parents=True, exist_ok=True)\n"
                        "(d / 'sim.log').write_text('ego pose\\n'); time.sleep(600)\n")
        srv = mock.Mock(bias_sock="", op_sock="")
        cfg = dict(agent="cinque", preset="spec", timeout_s=600, stall_s=6)
        with mock.patch.object(H, "ZS_RUN", fake), mock.patch.object(RN, "py", return_value=sys.executable):
            (self.rd / "workers").mkdir()
            t0 = time.time()
            res = H.run_one(cfg, self.rd, "nuscenes/x.yaml", srv, "0", self.rd / "w.log")
        self.assertEqual(res, "stall")
        self.assertLess(time.time() - t0, 60)


class TestPoses(TmpData):
    """jevdrive.bench.poses without the devkit: input checks, sizing, the chunk queue, the merge, the stage graph."""

    def setUp(self):
        super().setUp()
        from jevdrive.bench import poses as PS
        self.PS = PS
        self.toks = [f"t{i:03d}" for i in range(50)]
        mc = PS.mcache_dir()
        for t in self.toks:
            (mc / "log" / "unknown" / t).mkdir(parents=True)
            (mc / "log" / "unknown" / t / "metric_cache.pkl").write_bytes(b"")
        self.npz = self.D / "p.npz"
        np.savez(self.npz, tokens=np.array(self.toks), a=np.zeros((50, 8, 3)), b=np.ones((50, 8, 3)), other=np.zeros((50, 3)))

    def test_check_inputs(self):
        PS = self.PS
        self.assertEqual(PS.check_inputs(self.npz), (["a", "b"], self.toks))
        self.assertEqual(PS.check_inputs(self.npz, ["b"], self.toks[:3]), (["b"], self.toks[:3]))
        for keys, toks, msg in ((["other"], None, "pose arrays"), (["zz"], None, "pose arrays"), ([], ["t001", "t001"], "duplicate"),
                                ([], ["nope"], "not in"), ([], [], "no tokens")):
            with self.assertRaisesRegex(ValueError, msg):
                PS.check_inputs(self.npz, keys, toks)
        (PS.mcache_dir() / "log/unknown/t007/metric_cache.pkl").unlink()
        with self.assertRaisesRegex(ValueError, "metric cache"):
            PS.check_inputs(self.npz)
        dup = self.D / "d.npz"
        np.savez(dup, tokens=np.array(["x", "x"]), a=np.zeros((2, 8, 3)))
        with self.assertRaisesRegex(ValueError, "duplicate tokens in the poses"):
            PS.check_inputs(dup, mcache=False)

    def test_memo_policy(self):
        calls = []
        pol = mock.Mock(simulate_environment=lambda st, mc: calls.append(st.sum()) or [st.sum()], name_attr="x")
        m = self.PS.MemoPolicy(pol)
        a, b = np.zeros((41, 11)), np.ones((41, 11))
        self.assertEqual([m.simulate_environment(x, None) for x in (a, b, a, b)], [[0.0], [41 * 11.0], [0.0], [41 * 11.0]])
        self.assertEqual((len(calls), m.hits, m.name_attr), (2, 2, "x"))

    def test_plan_jobs(self):
        PS = self.PS
        self.assertEqual(PS.plan_jobs(12146, budget=75), (6, 12))       # 75-core quota: 6 jobs x 12 cores
        self.assertEqual(PS.plan_jobs(12146, cpu=24, budget=75), (3, 24))
        self.assertEqual(PS.plan_jobs(300, budget=75), (2, 12))         # few tokens: >= 16 per worker
        self.assertEqual(PS.plan_jobs(5, budget=75), (1, 12))
        self.assertEqual(PS.plan_jobs(12146, jobs=2, budget=75), (2, 12))

    def test_chunks(self):
        PS = self.PS
        rd = self.D / "run"
        q0, q1 = PS.Chunks(rd, 3, 0), PS.Chunks(rd, 3, 1)
        self.assertEqual((q0.next(), q1.next()), (0, 1))
        PS._atomic_pickle(q0.out(2), {})                               # chunk 2 finished elsewhere
        self.assertIsNone(q1.next())
        q1.release(1)
        self.assertEqual(q0.next(), 1)
        old = time.time() - 1000
        os.utime(q0.claim_path(0), (old, old))
        self.assertEqual(PS.Chunks(rd, 3, 2).next(), 0)                 # stale claim taken over
        PS.Chunks(rd, 3, 0)                                             # a restarted worker 0 drops its own claims
        self.assertFalse(q0.claim_path(1).exists())
        self.assertTrue(q0.claim_path(0).exists())                      # worker 2's claim stays

    def test_frame(self):
        PS = self.PS
        row = lambda k, t: {"key": k, "token": t, **{m: 1.0 for m in PS.SUBS}, "score": 1.0, "raw_out": False, "raw_depth": 0.0,
                            "lqr_out": False, "out_depth": 0.0}
        toks, keys = self.toks[:3], ["b", "a"]
        rows = [row(k, t) for t in reversed(toks) for k in keys]
        df = PS.frame(rows, toks, keys)
        self.assertEqual(list(df.columns), PS.COLUMNS)
        self.assertEqual(list(zip(df.token, df.key)), [(t, k) for t in toks for k in keys])
        with self.assertRaisesRegex(RuntimeError, "missing"):
            PS.frame(rows[1:], toks, keys)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            PS.frame(rows + rows[:1], toks, keys)

    def test_stages_and_resume(self):
        PS = self.PS
        out = self.D / "o.csv"
        with mock.patch.object(PS, "pool_budget", return_value=75.0):
            d, st = PS.stages(self.npz, out, keys=["a"], chunk=10)
        cfg = json.loads((d / "config.json").read_text())
        self.assertEqual((cfg["n"], cfg["n_chunks"], cfg["keys"], cfg["jobs"]), (50, 5, ["a"], 1))
        self.assertEqual([s.name for s in st], ["w0", "collect"])
        self.assertEqual(st[0].cpu, 12)
        self.assertEqual(st[0].env["OMP_NUM_THREADS"], "1")
        self.assertEqual(st[0].env["OPENBLAS_CORETYPE"], "Haswell")
        self.assertIn("navsim2", st[0].cmd[0])
        self.assertEqual(st[1].after, ["w0"])
        self.assertEqual(PS.read_tokens(d / "tokens.txt"), self.toks)
        d2, _ = PS.stages(self.npz, out, keys=["b"], chunk=10)
        self.assertNotEqual(d, d2)                                      # keys are part of the run identity
        d3, _ = PS.stages(self.npz, self.D / "nr.csv", keys=["a"], chunk=10, traffic="non_reactive")
        self.assertNotEqual(d, d3)                                      # so is the traffic policy; the reactive default keeps its key
        self.assertEqual(json.loads((d3 / "config.json").read_text())["traffic"], "non_reactive")
        self.assertEqual(d.name, PS.run_key(str(self.npz.resolve()), ["a"], self.toks))
        with self.assertRaisesRegex(ValueError, "traffic"):
            PS.stages(self.npz, out, keys=["a"], traffic="idm")
        for j in range(5):                                              # every chunk finished: collect only
            PS._atomic_pickle(d / "chunks" / f"c{j:05d}.pkl", {})
        _, st = PS.stages(self.npz, self.D / "o2.csv", keys=["a"], chunk=10)
        self.assertEqual([s.name for s in st], ["collect"])
        self.assertEqual(len(json.loads((d / "config.json").read_text())["outs"]), 2)

    def test_collect(self):
        PS = self.PS
        out = self.D / "o.csv"
        with mock.patch.object(PS, "pool_budget", return_value=75.0):
            d, _ = PS.stages(self.npz, out, chunk=20)
        row = lambda k, t: {"key": k, "token": t, **{m: 0.5 for m in PS.SUBS}, "score": 0.1 + 0.2, "raw_out": True, "raw_depth": 0.25,
                            "lqr_out": False, "out_depth": 0.0}
        cost = dict(load=0.1, pdm=1.0, diag=0.1, union=0.0, rss_gb=0.4)
        for j in range(3):
            tk = self.toks[j * 20:(j + 1) * 20]
            PS._atomic_pickle(d / "chunks" / f"c{j:05d}.pkl", dict(rows=[row(k, t) for t in tk for k in ("a", "b")],
                                                                   cost=[dict(cost, token=t) for t in tk]))
        PS.collect(d)
        import pandas as pd
        df = pd.read_csv(out, float_precision="round_trip")
        self.assertEqual(len(df), 100)
        self.assertEqual(df.score.iloc[0], 0.1 + 0.2)                   # floats round-trip exactly
        self.assertTrue((d / "DONE").exists())
        (d / "chunks" / "c00001.pkl").unlink()
        with self.assertRaisesRegex(RuntimeError, "chunks missing"):
            PS.collect(d)


class TestB2D(TmpData):
    def test_cmd(self):
        from jevdrive.bench import b2d as B
        a = B.B2DAgent("pdm", agent="lib/x_agent.py", agent_config="cfg.json")
        with mock.patch.object(RN, "box_cards", return_value=[0, 1]):
            st = B.stages(a, self.D / "b2d", routes=str(self.D / "r.xml"), ids=["1", "2", "3"], workers=2)
        self.assertEqual([s.name for s in st], ["w0", "w1", "collect"])
        c = st[0].cmd
        self.assertIsInstance(c, str)
        self.assertIn('--server-args "{server_args}"', c)
        self.assertIn("--route-ids 1,2,3", c)
        self.assertEqual(st[0].carla, 2)

    def test_submit_dry(self):
        from jevdrive.bench import b2d as B
        with mock.patch.object(RN, "box_cards", return_value=[0]), mock.patch.object(RN, "submit") as sub:
            d = B.submit(B.B2DAgent("pdm"), routes=str(self.D / "r.xml"), route_ids=["1"], workers=1, run_dir=self.D / "b2d")
        self.assertEqual(d, self.D / "b2d")
        self.assertEqual([s.name for s in sub.call_args.args[2]], ["w0", "collect"])


if __name__ == "__main__":
    unittest.main()
