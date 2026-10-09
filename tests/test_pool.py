"""jevdrive.cl.pool with fakes: no GPU, no CARLA. The placement tests run anywhere; the dispatcher tests start real short
shell jobs and read /proc (Linux: run them on the box).

    python -m unittest tests.test_pool -v
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevdrive.cl import capacity, pool as P  # noqa: E402
from jevdrive.cl.box import Box, Card  # noqa: E402

LINUX = Path("/proc/self/stat").exists()
CFG = dict(P.DEFAULTS)


def acct(g, used=0.0, **kw):
    return P.CardAcct(g, 0, 83.6, used, foreign_gb=used, **kw)


class Placement(unittest.TestCase):
    def test_fits_vram_carla_train_exclusive(self):
        s = P.Spec(["x"], vram_gb=30).check()
        self.assertEqual(P.fits(s, acct(0), CFG), "")
        self.assertIn("VRAM", P.fits(s, acct(0, used=60), CFG))
        a = acct(0, pool_gb=50)                     # declared but not yet allocated VRAM stays booked
        self.assertIn("VRAM", P.fits(s, a, CFG))
        c = P.Spec(["x"], carla=2).check()
        self.assertEqual(c.vram_gb, 2 * capacity.VRAM_PER_WORKER_GB)
        self.assertIn("CARLA", P.fits(c, acct(0, carla_foreign=3, carla_pool=2), CFG))
        self.assertIn("this round", P.fits(c, acct(0, started_carla=True), CFG))
        t = P.Spec(["x"], vram_gb=10, train=True).check()
        self.assertIn("training", P.fits(t, acct(0, train=2), CFG))
        e = P.Spec(["x"], exclusive=True).check()
        self.assertIn("not empty", P.fits(e, acct(0, jobs=1), CFG))
        self.assertIn("exclusive job", P.fits(s, acct(0, exclusive=True), CFG))
        self.assertIn("held", P.fits(s, acct(0, whole_hold="dagger"), CFG))
        self.assertIn("not allowed", P.fits(P.Spec(["x"], vram_gb=1, gpus=[1]).check(), acct(0), CFG))
        with self.assertRaises(ValueError):
            P.Spec(["x"]).check()                   # no VRAM declared

    def test_choose_spreads_then_best_fit(self):
        s = P.Spec(["x"], vram_gb=10).check()
        cards = [acct(0, used=10, jobs=2), acct(1, used=40), acct(2, used=5)]
        best, _ = P.choose(s, cards, CFG, "j")
        self.assertEqual(best.index, 1)              # 1 and 2 have no jobs; 1 leaves the tighter fit
        cards[1].reserved_for = "other"
        self.assertEqual(P.choose(s, cards, CFG, "j")[0].index, 2)
        big = P.Spec(["x"], vram_gb=80).check()
        best, why = P.choose(big, cards, CFG, "j")
        self.assertIsNone(best)
        self.assertEqual(sorted(why), [0, 1, 2])

    def test_index_blocks_avoid_ports_and_tm_shadow(self):
        lo, hi = capacity.index_bounds(32768)
        self.assertEqual((lo, hi), (160, 494))
        used = P.port_blocks({2000 + 50 * 170, 2000 + 50 * 170 + 1, 8000 + 50 * 171 + 7})    # RPC of 170, TM of 171
        self.assertEqual(used, {170, 291})
        i0 = P.free_index_block(used, 12, 160, 494)
        self.assertEqual(i0, 172)                     # 160-171 hits 170; a block's TM (i + 120) must miss 291 too
        self.assertFalse(P.blocks_of(range(i0, i0 + 12)) & used)
        self.assertNotIn(291, P.blocks_of(range(172, 184)))
        used2 = used | P.blocks_of(range(172, 184))
        self.assertEqual(P.free_index_block(used2, 4, 160, 494), 160)
        self.assertEqual(P.free_index_block(used2 | P.blocks_of(range(160, 170)), 4, 160, 494), 184)
        self.assertIsNone(P.free_index_block(set(range(0, 700)), 2, 160, 494))

    def test_submit_and_cancel_files(self):
        tmp = Path(tempfile.mkdtemp())
        jid = P.submit(["echo", "hi"], name="t", pool=tmp, vram_gb=2, after=["a", ""])
        j = json.loads((tmp / "inbox" / ("%s.json" % jid)).read_text())
        self.assertEqual((j["spec"]["vram_gb"], j["spec"]["after"]), (2, ["a"]))
        P.cancel(jid, pool=tmp)
        self.assertEqual((tmp / "cancel" / jid).read_text(), "stop")

    def test_preflight_shared_and_ahead(self):
        tmp = Path(tempfile.mkdtemp())
        a = P.preflight("python x.py --limit 2", "b", pool=tmp, vram_gb=4, cwd="/r", log_dir="/l", after=["z"])
        b = P.preflight("python x.py --limit 2", "b", pool=tmp, vram_gb=4, cwd="/r")
        c = P.preflight("python x.py --limit 3", "b", pool=tmp, vram_gb=4, cwd="/r")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        j = json.loads((tmp / "inbox" / ("%s.json" % a)).read_text())["spec"]
        self.assertEqual((j["name"], j["priority"], j["after"], j["log_dir"], j["env"]["CL_PREFLIGHT"]),
                         ("b-pf", 1000.0, [], "/l/preflight", "1"))

    def test_retarget_file(self):
        tmp = Path(tempfile.mkdtemp())
        P.retarget("j1", [0, 2], pool=tmp)
        P.retarget("j2", [], pool=tmp)
        self.assertEqual(((tmp / "retarget" / "j1").read_text(), (tmp / "retarget" / "j2").read_text()), ("0,2", ""))

    def test_static_check(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "ok.py").write_text("x = 1\n")
        (tmp / "bad.py").write_text("def f(:\n")
        self.assertEqual(P.static_check(["python", "ok.py", "--n", "3"], str(tmp)), [])
        self.assertEqual(len(P.static_check("python bad.py && bash missing.sh", str(tmp))), 2)


class MeasuredAndHistory(unittest.TestCase):
    def test_cpu_charge_declared_young_then_measured(self):
        now = time.time()
        j = dict(spec=dict(cpu=20), t0=now - 60, cores_peak=2.0)
        self.assertEqual(P.cpu_charge(j, now), 20)                  # young: declared
        j["t0"] = now - 1000
        self.assertAlmostEqual(P.cpu_charge(j, now), 2.4)           # 1.2 x peak
        j["cores_peak"] = 0.1
        self.assertEqual(P.cpu_charge(j, now), 1.0)                 # at least one core
        del j["cores_peak"]
        self.assertEqual(P.cpu_charge(j, now), 20)                  # no measurement yet: declared

    def test_measure_cpu_rolling_peak(self):
        tmp = Path(tempfile.mkdtemp())
        d = P.Dispatcher(tmp)
        d.st["jobs"]["a"] = j = dict(id="a", spec=dict(cpu=8))
        ticks = {1: 0, 2: 0}
        d.ticks_fn = lambda p: ticks.get(p)
        t = 1000.0
        d.measure_cpu({"a": [1, 2]}, t)                              # first sight: no delta yet
        self.assertNotIn("cores_peak", j)
        for step, cores in enumerate([4, 1, 1]):
            t += 20
            ticks[1] += int(cores * P.CLK_TCK * 20)
            d.measure_cpu({"a": [1, 2]}, t)
        self.assertEqual((j["cores_now"], j["cores_peak"]), (1.0, 4.0))
        for _ in range(16):                                          # the 4-core sample leaves the 5-min window
            t += 20
            ticks[1] += int(1 * P.CLK_TCK * 20)
            d.measure_cpu({"a": [1, 2]}, t)
        self.assertEqual(j["cores_peak"], 1.0)
        self.assertGreaterEqual(j["cores_max"], 4.0)
        ticks.pop(2)                                                 # a member exits: no negative delta
        t += 20
        d.measure_cpu({"a": [1]}, t)
        self.assertGreaterEqual(j["cores_now"], 0)

    def test_history_prefix_and_defaults(self):
        for n, want in [("op-eval-s3", "op-eval"), ("op-eval-k12-pf", "op-eval"), ("Dg_Train-007", "dg_train"),
                        ("op_parity-u2-t0", "op_parity"), ("train2", "train"), ("b2d-route", "b2d-route"), ("s3", "s")]:
            self.assertEqual(P.hist_prefix(n), want, n)
        tmp = Path(tempfile.mkdtemp())
        self.assertEqual(P.history_defaults("x-s1", tmp), {})
        for v, c in [(10, 4), (20, 8), (30, 5)]:
            P.hist_record(tmp, "x-s%d" % v, v, c)
        self.assertEqual(P.history_defaults("x-k9", tmp), dict(vram_gb=36.0, cpu=10))     # p95 = max here
        from jevdrive.cl import __main__ as cli
        old = os.environ.get("CL_POOL_DIR")
        os.environ["CL_POOL_DIR"] = str(tmp)
        try:
            self.assertEqual(cli.main(["submit", "--name", "x-s7", "--no-check", "--", "true"]), 0)
            self.assertEqual(cli.main(["submit", "--name", "x-s8", "--no-check", "--vram", "5", "--cpu", "2", "--", "true"]), 0)
        finally:
            os.environ.pop("CL_POOL_DIR") if old is None else os.environ.update(CL_POOL_DIR=old)
        specs = {j["spec"]["name"]: j["spec"] for j in (json.loads(f.read_text()) for f in (tmp / "inbox").glob("*.json"))}
        self.assertEqual((specs["x-s7"]["vram_gb"], specs["x-s7"]["cpu"]), (36.0, 10))
        self.assertEqual((specs["x-s8"]["vram_gb"], specs["x-s8"]["cpu"]), (5, 2))        # explicit wins

class BookedByMeasurement(unittest.TestCase):
    """A declaration is an upper bound; what the pool books follows what the job (or its name prefix) measured."""

    def test_vram_need_and_charge(self):
        hist = {"x": dict(vram=[10, 12, 11], cores=[2, 3, 2.5], ram=[8, 9, 7])}
        known = P.known_caps(hist, "x-s4")
        self.assertEqual(known, dict(vram_gb=15.4, cpu=3.6, ram_gb=11.8))
        self.assertEqual(P.known_caps({"x": dict(vram=[10, 12], cores=[])}, "x-s4"), {})      # < 3 runs: not trusted
        keyed = dict(name="bn-navtest-NEWMODEL-s0@warp-plans", env={"CL_HIST_KEY": "X"})             # a submitter's own history key
        self.assertEqual((P.hist_key(keyed), P.known_caps(hist, keyed)), ("x", known))
        self.assertEqual(P.known_caps(hist, dict(name="bn-navtest-NEWMODEL-s0@warp-plans")), {})
        from jevdrive.bench import runner
        self.assertEqual(runner.hist_key(runner.stage_cmd("jev", "hugsim-worker", "/r", 3), 59.0), "bn-hugsim-worker@59gb")
        self.assertEqual(P.hist_prefix("bn-hugsim-worker@59gb"), "bn-hugsim-worker@59gb")
        self.assertEqual(runner.hist_key("python x.py && touch y", 59.0), "")
        spec = dict(vram_gb=24.0)
        self.assertEqual(P.vram_need(spec, known), 15.4)
        self.assertEqual(P.vram_need(spec, known, trust=False), 24.0)
        self.assertEqual(P.vram_need(dict(vram_gb=10.0), known), 10.0)                        # never above the declaration
        self.assertEqual(P.vram_need(dict(vram_gb=24.0, carla=2), known), 24.0)               # CARLA: declared
        now = time.time()
        j = dict(spec=spec, t0=now - 30)
        self.assertEqual(P.vram_charge(j, now, known), 15.4)                                  # history: from the start
        j["vram_peak"] = 18.0
        self.assertAlmostEqual(P.vram_charge(j, now, known), 22.6)                            # own peak above history
        j["vram_peak"] = 40.0
        self.assertEqual(P.vram_charge(j, now, known), 49.0)                                  # broke the declaration: 1.2 x peak + 1
        j = dict(spec=spec, t0=now - 30, vram_peak=5.7)
        self.assertEqual(P.vram_charge(j, now, {}), 24.0)                                     # no history, young: declared
        j["t0"] = now - P.VRAM_SETTLE_S - 1
        self.assertAlmostEqual(P.vram_charge(j, now, {}), 1.2 * 5.7 + 1.0)                    # settled: own peak
        self.assertEqual(P.vram_charge(dict(j, vram_peak=0.0), now, {}), 24.0)                # never measured: declared
        self.assertEqual(P.vram_charge(j, now, {}, trust=False), 24.0)
        self.assertEqual(P.vram_charge(dict(j, spec=dict(vram_gb=24.0, exclusive=True)), now, {}), 24.0)

    def test_under_declared_vram_books_by_measurement(self):
        spec, now = dict(vram_gb=12.0, name="prep-s3"), time.time()
        self.assertFalse(P.over_declared(spec, 14.0))                                         # within 1.1 x + 1 GB
        self.assertTrue(P.over_declared(spec, 29.1))
        self.assertFalse(P.over_declared(dict(spec, exclusive=True), 60.0))
        hist = {"prep": dict(vram=[29.1], cores=[])}                                          # one run is enough
        self.assertEqual(P.hist_max(hist, spec), 29.1)
        self.assertEqual(P.hist_max(hist, dict(name="other")), 0.0)
        self.assertAlmostEqual(P.vram_need(spec, {}, True, P.hist_max(hist, spec)), 30.1)     # queued: recorded peak + 1
        self.assertAlmostEqual(P.vram_need(spec, {}, False, 29.1), 30.1)                      # whatever trust says
        self.assertEqual(P.vram_need(dict(spec, vram_gb=30.0), {}, True, 29.1), 30.0)         # declaration fixed: back to it
        j = dict(spec=spec, t0=now - 30, vram_peak=2.0)
        self.assertEqual(P.vram_charge(j, now, {}), 12.0)
        self.assertAlmostEqual(P.vram_charge(j, now, {}, True, 29.1), 30.1)                   # running, history says more
        j["vram_peak"] = 41.8
        self.assertAlmostEqual(P.vram_charge(j, now, {}, False), 51.2)                        # own peak, kept after it drops
        self.assertAlmostEqual(P.vram_charge(dict(j, spec=dict(spec, carla=1)), now, {}), 51.2)

    def test_vram_report(self):
        tmp, now = Path(tempfile.mkdtemp()), time.time()
        jobs = {"a": dict(id="a", spec=dict(name="prep-s0", vram_gb=12.0), t0=now - 60, vram_peak=29.1, state="done"),
                "b": dict(id="b", spec=dict(name="prep-s1", vram_gb=12.0), t0=now - 50, vram_peak=0.0, state="failed",
                          why="rc 1 (CUDA OOM)"),
                "c": dict(id="c", spec=dict(name="ok-s0", vram_gb=20.0), t0=now - 50, vram_peak=15.0, state="done"),
                "d": dict(id="d", spec=dict(name="old-s0", vram_gb=1.0), t0=now - 9e5, vram_peak=15.0, state="done")}
        (tmp / "state.json").write_text(json.dumps(dict(jobs=jobs)))
        (tmp / "history.json").write_text(json.dumps({"prep": dict(vram=[29.1])}))
        rows = P.vram_report(48.0, tmp)
        self.assertEqual([r["key"] for r in rows], ["prep", "ok"])
        self.assertEqual({k: rows[0][k] for k in ("runs", "declared", "peak", "over", "oom", "books")},
                         dict(runs=2, declared=12.0, peak=29.1, over=1, oom=1, books=30.1))
        self.assertEqual((rows[1]["over"], rows[1]["books"]), (0, 20.0))

    def test_ram_reserve_and_cpu_known(self):
        now = time.time()
        j = dict(spec=dict(ram_gb=44, cpu=14), t0=now - 60, rss_now=30.0)
        self.assertAlmostEqual(P.ram_reserve(j, now), 14.0 * 0.8)    # what it has not allocated yet, tapering over 5 min
        self.assertAlmostEqual(P.ram_reserve(j, now, known=35.0), 5.0 * 0.8)
        self.assertEqual(P.ram_reserve(dict(j, t0=now), now), 14.0)
        self.assertEqual(P.ram_reserve(dict(j, rss_now=80.0), now), 0.0)
        self.assertEqual(P.ram_reserve(dict(j, t0=now - 1000), now), 0.0)
        self.assertEqual(P.cpu_charge(j, now), 14)
        self.assertEqual(P.cpu_charge(j, now, known=2.4), 2.4)       # young, history knows the prefix
        self.assertEqual(P.cpu_charge(dict(j, cores_peak=5.0), now, known=2.4), 6.0)
        self.assertEqual(P.cpu_charge(j, now, known=40.0), 14)

    def test_wait_cause_and_gpu_flag(self):
        for why, c in [("after 1008-1 (running)", "after"), ("waiting for /x", "gate"), ("start limit this round", "start_limit"),
                       ("CPU 70 + 14 cores > budget 75", "cpu"), ("memory 200 + 44 GB > 85% of 276", "ram"),
                       ("card 0: card not allowed; card 1: VRAM 3.0 GB free of the pool's view < 59.0; card 2: card not allowed", "vram"),
                       ("card 0: card not allowed", "pinned"),
                       ("card 0: VRAM 3.0 GB free of the pool's view < 59.0", "vram"), ("card all: no 48 free cores to pin", "cpu_pin"),
                       ("card 0: reserved for blocked job 1", "reserved"), ("new", "new"), ("?", "other")]:
            self.assertEqual(P.wait_cause(why), c, why)
        self.assertTrue(P.is_gpu(dict(vram_gb=6)) and P.is_gpu(dict(vram_gb=0, exclusive=True)) and P.is_gpu(dict(carla=1)))
        self.assertFalse(P.is_gpu(dict(vram_gb=0.5)))

    def test_serial_hint(self):
        self.assertEqual(P.serial_hint(["python", "run.py", "--jobs", "a", "b"]), "")
        self.assertEqual(P.serial_hint("cd /x && python run.py --arm a"), "")
        self.assertIn("2 runs", P.serial_hint("python run.py --arm a && python run.py --arm b"))
        self.assertIn("loop", P.serial_hint("for s in 0 1 2; do python train.py --seed $s; done"))
        self.assertIn("2 runs", P.serial_hint(["bash", "-c", "python a.py; python b.py"]))

    def test_fanout_jobs_and_collect(self):
        tmp = Path(tempfile.mkdtemp())
        out = P.fanout(["python", "run.py", "--arm", "{arm}"], ["V1", "V2", "V3"], "wax", collect="python collect.py {arms}",
                       pool=tmp, vram_gb=7, cpu=2, env={"TAG": "t-{arm}"}, log_dir="/logs/wax")
        specs = {j["id"]: j["spec"] for j in (json.loads(f.read_text()) for f in (tmp / "inbox").glob("*.json"))}
        self.assertEqual(len(specs), 4)
        a = specs[out["arms"]["V2"]]
        self.assertEqual((a["name"], a["cmd"], a["env"], a["log_dir"], a["vram_gb"]),
                         ("wax-V2", ["python", "run.py", "--arm", "V2"], {"TAG": "t-V2"}, "/logs/wax/V2", 7))
        c = specs[out["collect"]]
        self.assertEqual((c["name"], c["cmd"], sorted(c["after"]), c["vram_gb"], c["log_dir"]),
                         ("wax-collect", "python collect.py V1 V2 V3", sorted(out["arms"].values()), 0.5, "/logs/wax/collect"))
        with self.assertRaises(ValueError):
            P.fanout("x", ["a", "a"], "n", pool=tmp, vram_gb=1)
        from jevdrive.cl import __main__ as cli
        old = os.environ.get("CL_POOL_DIR")
        os.environ["CL_POOL_DIR"] = str(tmp)
        try:
            self.assertEqual(cli.main(["fanout", "--name", "f", "--arms", "a,b", "--vram", "6", "--no-check", "--collect",
                                       "echo {arms}", "--", "echo", "{arm}"]), 0)
            self.assertEqual(cli.main(["submit", "--name", "ser", "--vram", "6", "--no-check", "--",
                                       "python a.py && python b.py"]), 0)          # a hint on stderr, still submitted
        finally:
            os.environ.pop("CL_POOL_DIR") if old is None else os.environ.update(CL_POOL_DIR=old)
        names = sorted(json.loads(f.read_text())["spec"]["name"] for f in (tmp / "inbox").glob("*.json"))
        self.assertEqual([n for n in names if n[0] in "fs"], ["f-a", "f-b", "f-collect", "ser"])

    def test_usage_report_splits_idle_by_cause(self):
        tmp = Path(tempfile.mkdtemp())
        t = 1_000_000

        def line(i, cards, q_gpu=None, oldest=None):
            return json.dumps(dict(t=t + 60 * i, cards=cards, cpu=[30.0, 40.0, 105.0], quota=75.0, q_gpu=q_gpu or {}, q_cpu={},
                                   not_ready=0, oldest=oldest or {}))
        busy, idle, lazy = [95, 30, 40, 1, 0], [0, 0, 0, 0, 0], [0, 6, 14, 1, 0]
        rows = [line(0, {"0": busy, "1": busy}), line(1, {"0": busy, "1": idle}, oldest={"0": 60}),
                line(2, {"0": busy, "1": idle}, oldest={"0": 4000}), line(3, {"0": busy, "1": idle}, q_gpu={"ram": 2, "vram": 1}),
                line(4, {"0": idle, "1": idle}), line(5, {"0": lazy, "1": [0, 0, 0, 0, 1]})]
        (tmp / "usage.jsonl").write_text("\n".join(rows) + "\n")
        u = P.usage_report(1.0, tmp, now=t + 400)
        c = {k: round(v * 60) for k, v in u["cards"].items()}
        self.assertEqual(c, {"computing": 5, "no work queued": 3, "serial": 1, "queued: ram": 1, "job, GPU idle": 1, "held": 1})
        self.assertAlmostEqual(u["card_h"], 12 / 60)
        self.assertAlmostEqual(u["core_h"], 75 * 6 / 60)
        self.assertAlmostEqual(u["core_used_h"], 30 * 6 / 60)
        self.assertEqual(P.usage_report(1.0, tmp, now=t + 10 * 3600)["samples"], 0)


def fake_box(cards=(0, 1, 2), used_mib=0, pids=700):
    cs = [Card(g, "GPU-%d" % g, "0000:%02x:00.0" % g, used_mib, 85651, 0, 0 if g == 0 else 1) for g in cards]
    return Box(208, 75, list(range(208)), {0: list(range(52)), 1: list(range(52, 104))},
               {c: [c] for c in range(208)}, 20480, pids, 0, 0, (32768, 60999), cs)


@unittest.skipUnless(LINUX, "dispatcher tests start real processes and read /proc")
class Dispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.box = fake_box()

    def disp(self, **cfg):
        c = dict(poll_s=0.05, hold_s=1e9)
        c.update(cfg)
        return P.Dispatcher(self.tmp, probe_fn=lambda rows: self.box, ports_fn=set, smi_fn=lambda q, a: [], cfg=c)

    def until(self, d, cond, limit=30.0):
        end = time.time() + limit
        while time.time() < end:
            d.round()
            if cond():
                return
            time.sleep(0.1)
        self.fail("timeout; states %s" % {k: (j["state"], j.get("why")) for k, j in d.st["jobs"].items()})

    def cancel_all(self, d):
        for j in d.st["jobs"].values():
            if j["state"] not in P.FINAL:
                P.cancel(j["id"], pool=self.tmp)
        d.halt = True                                  # nothing starts while the others stop
        self.until(d, lambda: not d.jobs("running"))

    def states(self, d):
        return {j["spec"]["name"]: j["state"] for j in d.st["jobs"].values()}

    def test_placeholders_env_done_and_spread(self):
        rec = self.tmp / "seen"
        ids = [P.submit(["sh", "-c", 'echo "$CL_POOL_JOB $1 $2 $3 $CUDA_VISIBLE_DEVICES $OMP_NUM_THREADS" >> %s; sleep 1' % rec,
                         "x", "{gpu}", "{idx}", "{span}"], name="j%d" % i, pool=self.tmp, carla=1) for i in range(3)]
        d = self.disp()
        d.round()
        self.assertEqual(sorted(j["gpu"] for j in d.st["jobs"].values()), [0, 1, 2])     # one per card: spread
        self.until(d, lambda: all(s == "done" for s in self.states(d).values()))
        lines = [l.split() for l in rec.read_text().splitlines()]
        idx = sorted(int(x[2]) for x in lines)
        self.assertEqual(len(set(idx)), 3)
        self.assertTrue(all(x[3] == "2" and x[4] == x[1] and x[5] == "2" for x in lines))   # span 2, CUDA = card, profile env
        for jid in ids:
            self.assertTrue((self.tmp / "jobs" / jid / "DONE").exists())
            self.assertIn("done", (self.tmp / "jobs" / jid / "STATUS").read_text())

    def test_vram_booking_queues_and_frees_after_exit(self):
        for i in range(4):
            P.submit("sleep 0.6", name="v%d" % i, pool=self.tmp, vram_gb=50, gpus=[1])
        d = self.disp()
        d.round()
        self.assertEqual(sorted(self.states(d).values()), ["queued"] * 3 + ["running"])
        q = [j for j in d.st["jobs"].values() if j["state"] == "queued"]
        self.assertTrue(all("VRAM" in j["why"] for j in q))
        conc, end = 0, time.time() + 20
        while time.time() < end and not all(s == "done" for s in self.states(d).values()):
            d.round()
            conc = max(conc, len(d.jobs("running")))
            time.sleep(0.05)
        self.assertEqual(conc, 1)
        self.assertTrue(all(s == "done" for s in self.states(d).values()))

    def test_priority_after_retry_and_failure(self):
        a = P.submit("exit 3", name="bad", pool=self.tmp, vram_gb=1, tries=2)
        P.submit("true", name="child", pool=self.tmp, vram_gb=1, after=[a])
        ok = P.submit("true", name="ok", pool=self.tmp, vram_gb=1)
        P.submit("true", name="next", pool=self.tmp, vram_gb=1, after=[ok])
        d = self.disp()
        self.until(d, lambda: all(j["state"] in P.FINAL for j in d.st["jobs"].values()))
        st = self.states(d)
        self.assertEqual(st, dict(bad="failed", child="failed", ok="done", next="done"))
        bad = d.st["jobs"][a]
        self.assertEqual((bad["tries"], bad["rc"]), (2, 3))
        self.assertIn("rc 3", (self.tmp / "jobs" / a / "ERROR").read_text())

    def events(self, kind):
        return [e for e in (json.loads(l) for l in (self.tmp / "events.jsonl").read_text().splitlines()) if e["kind"] == kind]

    def test_sigkill_is_retried_without_counting_and_counted_per_day(self):
        mark = self.tmp / "second"
        a = P.submit("if [ -e %s ]; then exit 0; fi; touch %s; kill -9 $$" % (mark, mark), name="wod-s0", pool=self.tmp, vram_gb=1)
        b = P.submit("kill -9 $$", name="always", pool=self.tmp, vram_gb=1, tries=2)
        c = P.submit("exit 9", name="plain", pool=self.tmp, vram_gb=1)
        d = self.disp(kill_retries=2)
        self.until(d, lambda: all(j["state"] in P.FINAL for j in d.st["jobs"].values()))
        j = d.st["jobs"]
        self.assertEqual((j[a]["state"], j[a]["tries"], j[a]["kill_tries"]), ("done", 2, 1))    # tries = 1, retried anyway
        self.assertEqual((j[b]["state"], j[b]["tries"], j[b]["rc"]), ("failed", 4, 137))        # 2 free + 2 own tries
        self.assertIn("SIGKILL", j[b]["why"])
        self.assertEqual((j[c]["state"], j[c]["tries"]), ("failed", 1))
        ev = self.events("kill")
        self.assertEqual(sorted(e["id"] for e in ev), sorted([a] + [b] * 4))
        self.assertEqual(len(ev[0]["mem"]), 6)                # anon, current, high, high events, working set, kill line
        rep = P.kill_report(1, self.tmp)
        self.assertEqual([(v["n"], v["jobs"]) for v in rep.values()], [(5, 2)])                 # each kill once
        self.assertIn("always", list(rep.values())[0]["last"] + " always")

    def test_kill_report_counts_events_from_before_the_kill_event(self):
        day = time.strftime("%F")
        lines = [dict(t=day + " 00:19:29", kind="end", id="a", state="failed", rc=137, why="rc 137"),
                 dict(t=day + " 00:35:54", kind="retry", id="b", rc=137, tries=1),
                 dict(t=day + " 00:36:00", kind="end", id="c", state="failed", rc=1, why="rc 1"),
                 dict(t=day + " 00:37:00", kind="kill", id="d", name="x", rc=137),
                 dict(t=day + " 00:37:01", kind="end", id="d", state="failed", rc=137, why="rc 137 (SIGKILL)"),
                 dict(t=day + " 00:38:00", kind="retry", id="e", rc=137, tries=1, sigkill=True)]
        (self.tmp / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        self.assertEqual(P.kill_report(1, self.tmp), {day: dict(n=3, jobs=3, last="00:37:00 x")})

    def test_working_set_rule_holds_a_start_below_the_kill_line(self):
        import dataclasses
        self.box = dataclasses.replace(fake_box(), mem_max_gb=552.0, mem_used_gb=100.0, mem_ws_gb=500.0)   # line 541
        d = self.disp()
        a = P.submit("true", name="wod", pool=self.tmp, vram_gb=1, ram_gb=40)
        b = P.submit("true", name="small", pool=self.tmp, vram_gb=1)                  # no declaration: 8 GB + 16 reserve
        d.round()
        ja, jb = d.st["jobs"][a], d.st["jobs"][b]
        self.assertEqual((ja["state"], P.wait_cause(ja["why"])), ("queued", "ram"))
        self.assertIn("working set 500 + 0 young + 40 GB + 16 reserve > kill line 541", ja["why"])
        self.assertEqual(d.ws_want, 56.0)                     # what the trimmer is asked to free
        self.assertIn(jb["state"], ("running", "done"))       # 500 + 8 + 16 <= 541
        self.box = dataclasses.replace(self.box, mem_ws_gb=440.0)
        self.until(d, lambda: ja["state"] == "done")
        self.box = dataclasses.replace(self.box, mem_ws_gb=530.0)
        c = P.submit("true", name="wod", pool=self.tmp, vram_gb=1, ram_gb=40)
        d.cfg_over["ws_wait_s"] = 0.0                         # the cache could not be dropped: start anyway, and say so
        self.until(d, lambda: d.st["jobs"][c]["state"] == "done")
        self.assertEqual([e["id"] for e in self.events("ws_override")], [c])

    def test_cuda_oom_is_retried_once_beyond_tries(self):
        mark = self.tmp / "second"
        oom = "echo 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB'"
        a = P.submit("if [ -e %s ]; then exit 0; fi; touch %s; %s; exit 1" % (mark, mark, oom), name="victim", pool=self.tmp, vram_gb=1)
        b = P.submit("%s; exit 1" % oom, name="always", pool=self.tmp, vram_gb=1)
        c = P.submit("echo plain failure; exit 1", name="plain", pool=self.tmp, vram_gb=1)
        d = self.disp()
        self.until(d, lambda: all(j["state"] in P.FINAL for j in d.st["jobs"].values()))
        j = d.st["jobs"]
        self.assertEqual((j[a]["state"], j[a]["tries"], j[a]["oom_tries"]), ("done", 2, 1))    # tries = 1, retried anyway
        self.assertEqual((j[b]["state"], j[b]["tries"]), ("failed", 2))
        self.assertIn("CUDA OOM", j[b]["why"])
        self.assertEqual((j[c]["state"], j[c]["tries"], j[c]["why"]), ("failed", 1, "rc 1"))    # not an OOM: no extra try
        self.assertEqual(sorted(e["id"] for e in self.events("oom")), sorted([a, b, b]))

    def test_over_declared_job_is_flagged_booked_and_stopped_when_the_card_is_full(self):
        d = self.disp()
        d.smi_fn = lambda q, age: ["GPU-1, %d, 30000" % j["pid"] for j in d.jobs("running") if j["spec"]["name"] == "prep-s0"]
        a = P.submit("sleep 60", name="prep-s0", pool=self.tmp, vram_gb=12, gpus=[1])
        d.round()                                             # launched
        b = P.submit("sleep 60", name="big", pool=self.tmp, vram_gb=50, gpus=[1])
        d.round()                                             # measured: 29.3 GB against 12 declared
        ja, jb = d.st["jobs"][a], d.st["jobs"][b]
        self.assertEqual((ja["vram_over"], ja["vram_booked"]), (29.3, 36.2))
        self.assertEqual(jb["state"], "queued")               # 83.6 - 4 - 36.2 < 50; the old booking (29.3) let it in
        self.assertIn("VRAM", jb["why"])
        ev = self.events("vram_over")
        self.assertEqual((len(ev), ev[0]["id"], ev[0]["declared"], ev[0]["peak"]), (1, a, 12, 29.3))
        self.assertIn("29.3 GB measured > 12.0 GB declared", (self.tmp / "jobs" / a / "STATUS").read_text())
        d.round()
        self.assertEqual(len(self.events("vram_over")), 1)    # said once
        self.box = fake_box(used_mib=85000)                   # the card is about to run out
        self.until(d, lambda: ja["state"] == "queued")        # the job above its declaration is stopped and requeued
        self.assertEqual((len(self.events("vram_stop")), ja["oom_tries"]), (1, 1))
        self.assertIn("< 30.3", ja["why"])                    # and waits for room for what it measured
        self.assertEqual(json.loads((self.tmp / "history.json").read_text())["prep"]["vram"], [29.3])
        self.box = fake_box()
        d.smi_fn = lambda q, age: []
        P.cancel(b, pool=self.tmp)
        self.until(d, lambda: ja["state"] == "running" and jb["state"] == "cancelled")
        self.assertEqual(ja["vram_booked"], 30.3)             # relaunched, booked by what it measured before
        self.cancel_all(d)

    def test_cancel_stops_whole_tree_and_orphans_are_reaped(self):
        marker = self.tmp / "child.pid"
        jid = P.submit("setsid sleep 60 & echo $! > %s; sleep 60" % marker, name="long", pool=self.tmp, vram_gb=1)
        orphan = P.submit("setsid sleep 60 & echo $! > %s.2; exit 0" % marker, name="orph", pool=self.tmp, vram_gb=1)
        d = self.disp()
        self.until(d, lambda: marker.exists() and Path(str(marker) + ".2").exists())
        pids = [int(marker.read_text()), int(Path(str(marker) + ".2").read_text())]
        self.until(d, lambda: d.st["jobs"][orphan]["state"] == "done")
        P.cancel(jid, pool=self.tmp)
        self.until(d, lambda: d.st["jobs"][jid]["state"] == "cancelled")
        time.sleep(0.2)
        for p in pids:
            self.assertFalse(Path("/proc/%d" % p).exists() and "Z" not in Path("/proc/%d/stat" % p).read_text().split(")")[1][:3])

    def test_whole_hold_and_hold_expiry(self):
        P.add_hold(0, "legacy chain", whole=True, pool=self.tmp)
        P.add_hold(1, "renderer", vram_gb=80, pid=os.getpid(), pool=self.tmp)
        jid = P.submit("true", name="h", pool=self.tmp, vram_gb=10, gpus=[0, 1])
        d = self.disp()
        d.round()
        why = d.st["jobs"][jid]["why"]
        self.assertIn("held: legacy chain", why)
        self.assertIn("card 1: VRAM", why)
        hs = P.load_holds(self.tmp)
        hs[1]["proc"]["start_ticks"] = "1"           # its process "exited"
        (self.tmp / "holds.json").write_text(json.dumps(hs))
        self.until(d, lambda: d.st["jobs"][jid]["state"] == "done")
        self.assertEqual(d.st["jobs"][jid]["gpu"], 1)

    def test_restart_adopts_running_job(self):
        jid = P.submit("sleep 1.5", name="adopt", pool=self.tmp, vram_gb=1)
        d = self.disp()
        d.round()
        self.assertEqual(d.st["jobs"][jid]["state"], "running")
        d2 = self.disp()                               # a new dispatcher reads state.json
        self.until(d2, lambda: d2.st["jobs"][jid]["state"] == "done")

    def test_head_job_reserves_a_card(self):
        P.submit("sleep 30", name="fill", pool=self.tmp, vram_gb=60, gpus=[0])
        d = self.disp(hold_s=0)
        d.round()
        P.submit("true", name="big", pool=self.tmp, vram_gb=70, gpus=[0], priority=5)
        P.submit("true", name="small", pool=self.tmp, vram_gb=5, gpus=[0])
        d.round()
        st = {j["spec"]["name"]: j for j in d.st["jobs"].values()}
        self.assertIn("reserves card 0", st["big"]["why"])
        self.assertIn("reserved for", st["small"]["why"])
        for j in d.jobs("running"):
            P.cancel(j["id"], pool=self.tmp)
        self.until(d, lambda: st["big"]["state"] == "done" or d.st["jobs"][st["big"]["id"]]["state"] == "done")

    def test_measured_cpu_charge_frees_budget(self):
        P.submit("sleep 30", name="a", pool=self.tmp, vram_gb=5, cpu=3)
        d = self.disp(cpu_budget=4, idle_s=1e9)
        d.round()
        b = P.submit("sleep 30", name="b", pool=self.tmp, vram_gb=5, cpu=3)
        d.round()
        self.assertIn("CPU 3 + 3 cores > budget 4", d.st["jobs"][b]["why"])      # declared while young
        a = next(j for j in d.jobs("running"))
        a["t0"] -= 1000
        a["cores_peak"] = 0.5                                                    # measured: charged 1 core
        d.round()
        self.assertEqual(d.st["jobs"][b]["state"], "running")
        self.cancel_all(d)

    def test_idle_card_admits_cpu_blocked_job(self):
        P.submit("sleep 30", name="a", pool=self.tmp, vram_gb=5, cpu=3)
        small = P.submit("sleep 30", name="cpuonly", pool=self.tmp, vram_gb=1, cpu=3)
        b = P.submit("sleep 30", name="b", pool=self.tmp, vram_gb=5, cpu=3)
        d = self.disp(cpu_budget=4, idle_s=1e9)
        d.round()
        self.assertEqual(d.st["jobs"][b]["state"], "queued")                     # no card idle long enough
        d2 = self.disp(cpu_budget=4, idle_s=0)
        d2.round()
        self.assertEqual(d2.st["jobs"][b]["state"], "running")                   # idle card: CPU bookkeeping relaxed
        self.assertEqual(d2.st["jobs"][small]["state"], "queued")                # vram_gb <= 1 never uses the rule
        self.assertIn("CPU", d2.st["jobs"][small]["why"])
        ev = [json.loads(l) for l in (self.tmp / "events.jsonl").read_text().splitlines()]
        self.assertIn(b, [e["id"] for e in ev if e["kind"] == "admit_idle"])
        self.cancel_all(d2)

    def test_idle_rule_never_relaxes_ram(self):
        self.box.mem_max_gb, self.box.mem_used_gb = 100.0, 90.0
        jid = P.submit("true", name="r", pool=self.tmp, vram_gb=5, cpu=3, ram_gb=1)
        d = self.disp(cpu_budget=1, idle_s=0)
        d.round()
        self.assertIn("memory", d.st["jobs"][jid]["why"])

    def test_auto_retarget_and_pin_strict(self):
        P.submit("sleep 30", name="fill", pool=self.tmp, vram_gb=70, gpus=[0])
        wide = P.submit("sleep 30", name="w", pool=self.tmp, vram_gb=70, gpus=[0])
        strict = P.submit("sleep 30", name="s", pool=self.tmp, vram_gb=70, gpus=[0], pin_strict=True)
        d = self.disp(idle_s=1e9)
        d.round()
        self.assertEqual((d.st["jobs"][wide]["state"], d.st["jobs"][strict]["state"]), ("queued", "queued"))
        d = self.disp(idle_s=0)
        d.round()
        j = d.st["jobs"][wide]
        self.assertEqual(j["state"], "running")
        self.assertIn(j["gpu"], (1, 2))
        self.assertEqual(sorted(j["spec"]["gpus"]), sorted({0, j["gpu"]}))
        self.assertEqual(d.st["jobs"][strict]["state"], "queued")
        self.assertEqual(d.st["jobs"][strict]["spec"]["gpus"], [0])              # pinned: never widened
        self.assertIn("auto_retarget", (self.tmp / "events.jsonl").read_text())
        self.cancel_all(d)

    def test_history_books_measured_vram_and_packs(self):
        self.box = fake_box(cards=(0,))
        for v in (10, 11, 12):
            P.hist_record(self.tmp, "pk-s9", v, 1.0, 2.0)
        for i in range(3):
            P.submit("sleep 30", name="pk-s%d" % i, pool=self.tmp, vram_gb=50)       # declared 50 GB, known to need 15.4
        P.submit("sleep 30", name="fresh", pool=self.tmp, vram_gb=50)               # no history: books its declaration
        d = self.disp(trust_measured=False)
        d.round()
        self.assertEqual(len(d.jobs("running")), 1)                                 # declarations only: one per card
        self.cancel_all(d)
        ids = [P.submit("sleep 30", name="pk-s%d" % i, pool=self.tmp, vram_gb=50) for i in range(3)]
        d = self.disp()
        d.round()
        self.assertEqual([d.st["jobs"][i]["state"] for i in ids], ["running"] * 3)
        self.assertEqual(d.st["jobs"][ids[0]]["vram_booked"], 15.4)
        d.round()
        self.assertAlmostEqual(d.last[0].pool_gb, 3 * 15.4, places=1)
        self.assertIn('"booked_gb": 15.4', (self.tmp / "events.jsonl").read_text())
        self.cancel_all(d)

    def test_cpu_only_job_leaves_its_card_idle(self):
        P.submit("sleep 30", name="score", pool=self.tmp, vram_gb=0.5, cpu=2)
        d = self.disp()
        d.round()
        d.round()
        g = d.jobs("running")[0]["gpu"]
        self.assertEqual((d.last[g].jobs, d.last[g].cpu_jobs), (0, 1))
        self.assertIn(g, d.idle_since)                                              # still an idle card for work conservation
        ids = [P.submit("sleep 30", name="gpu%d" % i, pool=self.tmp, vram_gb=10) for i in range(3)]
        d.round()
        self.assertEqual(sorted(d.st["jobs"][i]["gpu"] for i in ids), [0, 1, 2])   # it does not count as load either
        self.cancel_all(d)

    def test_wait_causes_and_usage_are_recorded(self):
        self.box = fake_box(cards=(0,))
        P.submit("sleep 1.5", name="first", pool=self.tmp, vram_gb=60)
        b = P.submit("true", name="second", pool=self.tmp, vram_gb=60)
        d = self.disp()
        d.round()
        self.assertEqual(d.st["jobs"][b]["cause"], "vram")
        time.sleep(0.3)
        d.round()
        self.assertGreater(d.st["jobs"][b]["waited"]["vram"], 0.2)
        self.until(d, lambda: d.st["jobs"][b]["state"] == "done")
        ev = [json.loads(l) for l in (self.tmp / "events.jsonl").read_text().splitlines()]
        launch = next(e for e in ev if e["kind"] == "launch" and e["id"] == b)
        self.assertGreater(launch["waited"]["vram"], 1.0)
        self.assertNotIn("t_cause", d.st["jobs"][b])
        row = json.loads((self.tmp / "usage.jsonl").read_text().splitlines()[0])
        self.assertEqual((row["cards"]["0"][3], row["q_gpu"], row["quota"]), (1, {"vram": 1}, 75))
        self.assertEqual(P.usage_report(1.0, self.tmp)["samples"], 1)
        st = json.loads((self.tmp / "status.json").read_text())
        self.assertIn("idle", st)

    def test_non_card_block_does_not_reserve(self):
        P.submit("sleep 30", name="pinner", pool=self.tmp, vram_gb=5, cpu=200)      # pins every core of the fake box
        d = self.disp(hold_s=0, cpu_budget=1000)
        d.round()
        big = P.submit("true", name="needs-cores", pool=self.tmp, vram_gb=5, cpu=50, priority=5)
        small = P.submit("true", name="small", pool=self.tmp, vram_gb=5)
        d.round()
        self.assertIn("free cores to pin", d.st["jobs"][big]["why"])
        self.assertNotIn("reserves", d.st["jobs"][big]["why"])
        self.assertIn(d.st["jobs"][small]["state"], ("running", "done"))
        self.cancel_all(d)


class CacheTrim(unittest.TestCase):
    def test_trim_drops_idle_files_oldest_first_until_the_margin_is_back(self):
        from jevdrive.cl import cache
        tmp = Path(tempfile.mkdtemp())
        for i, name in enumerate(("new.npy", "old.npy", "open.npy", "small.npy")):
            with (tmp / name).open("wb") as f:
                f.truncate(cache.MIN_BYTES if name != "small.npy" else 1024)          # sparse
            os.utime(tmp / name, (1000.0 * (10 - i if name != "old.npy" else 1), 0))
        st = os.stat(tmp / "open.npy")
        busy = {(os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)}
        files = cache.file_list([tmp], tmp / "list.json")
        self.assertEqual(sorted(Path(f[0]).name for f in files), ["new.npy", "old.npy", "open.npy"])
        self.assertEqual(cache.file_list([tmp], tmp / "list.json"), files)            # second call: the cached list
        dropped, state = [], dict(m=10.0)

        def drop(path):
            dropped.append(Path(path).name)
            state["m"] += 20.0
            return True
        r = cache.trim(45.0, [tmp], tmp / "list.json", margin_fn=lambda: state["m"], drop_fn=drop, busy=busy)
        self.assertEqual(dropped, ["old.npy", "new.npy"])     # idle files, oldest access first; the open one is left
        self.assertEqual((r["before"], r["after"], r["files"], r["busy"]), (10.0, 50.0, 2, 0))
        dropped.clear()
        r = cache.trim(45.0, [tmp], tmp / "list.json", margin_fn=lambda: state["m"], drop_fn=drop, busy=busy)
        self.assertEqual((dropped, r["files"]), ([], 0))      # enough margin: nothing is dropped
        state["m"] = 0.0
        r = cache.trim(55.0, [tmp], tmp / "list.json", margin_fn=lambda: state["m"], drop_fn=drop, busy=busy)
        self.assertEqual((dropped, r["busy"]), (["old.npy", "new.npy", "open.npy"], 1))      # open files last

    @unittest.skipUnless(LINUX, "reads /proc and the page cache")
    def test_in_use_sees_open_and_mapped_files_and_drop_empties_the_cache(self):
        import mmap
        from jevdrive.cl import cache
        tmp = Path(tempfile.mkdtemp(dir=os.environ.get("DATA_DIR", None)))
        p = tmp / "x.bin"
        p.write_bytes(os.urandom(4 * 2 ** 20))
        st = os.stat(p)
        key = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
        self.assertNotIn(key, cache.in_use())
        with p.open("rb") as f:
            self.assertIn(key, cache.in_use())
            m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        self.assertIn(key, cache.in_use())                    # mapped, no descriptor
        m.close()
        os.sync()
        self.assertTrue(cache.drop(str(p)))
        self.assertFalse(cache.drop(str(tmp / "missing")))
        if cache.mem():
            self.assertGreater(cache.mem()["max"], cache.mem()["ws"])
            self.assertEqual(cache.margin(), cache.margin(frac=cache.KILL_FRAC))


if __name__ == "__main__":
    unittest.main()
