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


if __name__ == "__main__":
    unittest.main()
