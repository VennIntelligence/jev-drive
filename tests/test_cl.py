"""jevdrive.cl with fakes: no GPU, no CARLA. The probe and lease tests run anywhere (fake /sys and /proc trees); the
lane tests start real short shell jobs and need Linux /proc (run them on the box).

    python -m unittest tests.test_cl -v
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevdrive.cl import capacity, lease as L, profiles  # noqa: E402
from jevdrive.cl.box import Box, Card, format_cpus, parse_cpus, probe  # noqa: E402
from jevdrive.cl.lane import Job, Lane, b2d, b2d_finished  # noqa: E402

LINUX = Path("/proc/self/stat").exists()


def write(root, rel, text):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def fake_root(tmp):
    """The 2026-10-01 box: 208 host CPUs (siblings i, i + 104), 2 NUMA nodes, 75-core quota, three cards."""
    write(tmp, "sys/fs/cgroup/cpu.max", "7500000 100000\n")
    write(tmp, "sys/fs/cgroup/pids.max", "20480\n")
    write(tmp, "sys/fs/cgroup/pids.current", "716\n")
    write(tmp, "sys/fs/cgroup/memory.max", "296352743424\n")
    write(tmp, "sys/fs/cgroup/memory.current", "129582235648\n")
    write(tmp, "sys/devices/system/cpu/online", "0-207\n")
    write(tmp, "proc/self/status", "Name:\tx\nCpus_allowed_list:\t0-207\n")
    write(tmp, "proc/sys/net/ipv4/ip_local_port_range", "32768\t60999\n")
    write(tmp, "proc/loadavg", "0.5 0.5 1.0 2/6504 311179\n")
    write(tmp, "sys/devices/system/node/node0/cpulist", "0-51,104-155\n")
    write(tmp, "sys/devices/system/node/node1/cpulist", "52-103,156-207\n")
    for c in range(208):
        p = c % 104
        write(tmp, "sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c, "%d,%d\n" % (p, p + 104))
    for bus, node in (("0000:1b:00.0", 0), ("0000:9b:00.0", 1), ("0000:9c:00.0", 1)):
        write(tmp, "sys/bus/pci/devices/%s/numa_node" % bus, "%d\n" % node)
    # one CARLA server of another lane on card 0
    write(tmp, "proc/4242/stat", "4242 (CarlaUE4-Linux-) S 1 4242 4242 0 -1 0 0 0 0 0 0 0 0 0 20 0 149 0 12345 0 0\n")
    (Path(tmp) / "proc/4242/cmdline").write_bytes(
        b"/x/CarlaUE4-Linux-Shipping\0CarlaUE4\0-carla-rpc-port=12000\0-graphicsadapter=0\0")


def fake_smi(query):
    if query.startswith("gpu="):
        return ["0, GPU-a, 00000000:1B:00.0, 9000, 85651, 40",
                "1, GPU-b, 00000000:9B:00.0, 0, 85651, 0",
                "2, GPU-c, 00000000:9C:00.0, 0, 85651, 0"]
    return ["GPU-a, 4242"]


class Probe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        fake_root(self.tmp)
        self.box = probe(Path(self.tmp), query=fake_smi)

    def test_cgroup_and_topology(self):
        b = self.box
        self.assertEqual((b.host_cpus, b.cores, b.pids_max, b.pids_current), (208, 75.0, 20480, 716))
        self.assertAlmostEqual(b.mem_max_gb, 276.0, places=0)
        self.assertEqual(b.numa[1][:2], [52, 53])
        self.assertEqual(len(b.primary_cpus(1)), 52)
        self.assertNotIn(156, b.primary_cpus(1))
        self.assertEqual(b.ephemeral, (32768, 60999))

    def test_cards(self):
        c0, c1, _ = self.box.cards
        self.assertEqual((c0.numa, c1.numa, c0.carla, c1.carla, c0.compute_pids), (0, 1, 1, 0, [4242]))
        self.assertEqual(c1.bus, "0000:9b:00.0")
        self.assertAlmostEqual(c1.free_gb, 83.6, places=1)

    def test_cpu_lists(self):
        self.assertEqual(parse_cpus("0-3,8, 10-11"), [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(format_cpus([11, 0, 1, 2, 3, 8, 10, 3]), "0-3,8,10-11")


class Capacity(unittest.TestCase):
    def test_thread_model_reproduces_measurements(self):
        red, stock = profiles.get("reduced"), profiles.get("stock")
        self.assertEqual(capacity.server_threads(16, red, 208), 109)       # docs/carla.md, 16-CPU affinity
        self.assertEqual(capacity.server_threads(16, stock, 208), 301)
        self.assertEqual(capacity.server_threads(24, red, 208), 149)       # G lane, 24-core slice
        self.assertEqual(capacity.client_threads(red, 208, agent=0), 29)    # --client-threads 8
        self.assertGreater(capacity.worker_threads(stock, 25, 208), 2 * capacity.worker_threads(red, 25, 208))

    def test_admission_from_pids_max(self):
        box = Box(208, 75, list(range(208)), {}, {}, 20480, 716, 0, 0, (32768, 60999), [])
        adm = capacity.Admission.from_box(box)
        self.assertEqual((adm.plan_cap, adm.wait_cap), (16384, 17408))
        self.assertEqual(adm.room(16000, 200), 1)
        self.assertEqual(adm.room(716, 200, pending=10), (16384 - 716) // 200 - 10)

    def test_index_bounds_and_sizing(self):
        self.assertEqual(capacity.index_bounds(32768), (160, 494))
        self.assertEqual(capacity.workers_per_card(25, 83.6), 6)                       # GPU knee
        self.assertEqual(capacity.workers_per_card(12, 83.6), 4)                       # cores
        self.assertEqual(capacity.workers_per_card(25, 40, vram_per_worker_gb=9), 3)   # VRAM


class Profiles(unittest.TestCase):
    def test_flags_and_env(self):
        r, s = profiles.get("reduced"), profiles.get("stock")
        self.assertEqual(r.server_args(), ["-RPCThreads=4", "-StreamingThreads=4", "-SecondaryThreads=4"])
        self.assertEqual(s.server_args(), [])
        self.assertEqual(s.environ(), {"B2D_CARLA_POOLS": "stock"})
        self.assertEqual(r.environ()["OMP_NUM_THREADS"], "2")
        self.assertEqual(profiles.get("reduced", num_threads=None, client_threads=None).num_threads, 2)
        self.assertEqual(profiles.get("reduced", client_threads=0).client_threads, 0)


ROWS = [
    dict(lane="a", gpus="0", workers="6", idx0="0:300", idx_span="12", cpus="0:0-24", status="running", go="-"),
    dict(lane="b", gpus="4,5", workers="6", idx0="60", idx_span="12", cpus="134-145", status="batch", go="-"),
    dict(lane="old", gpus="1", workers="6", idx0="1:160", idx_span="12", cpus="1:52-76", status="done 2026", go="-"),
]


class Lease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        fake_root(self.tmp)
        self.box = probe(Path(self.tmp), query=fake_smi)

    def test_row_parsing(self):
        self.assertEqual(L.index_blocks(ROWS[1]), {4: range(60, 72), 5: range(72, 84)})
        self.assertEqual(L.card_cpus(dict(ROWS[0], cpus="1:0-3,8,2:10-12", gpus="1,2")), {1: "0-3,8", 2: "10-12"})
        self.assertEqual(len(L.live(ROWS)), 2)

    def test_conflicts(self):
        clash = dict(ROWS[0], lane="c", gpus="2", idx0="2:180", cpus="2:24-30")        # 180 = 300 - 120; core 24
        bad = L.conflicts(ROWS + [clash])
        self.assertTrue(any("index conflict a x c" in b for b in bad), bad)
        self.assertTrue(any("core overlap a x c: 24" in b for b in bad), bad)
        high = dict(ROWS[0], lane="h", gpus="3", idx0="3:490", idx_span="10", cpus="3:90-95")
        self.assertTrue(any("ephemeral" in b for b in L.conflicts([high])))

    def test_find_free(self):
        ls = L.find_free(self.box, ROWS, "new", n_gpus=2, cores_per_card=25, span=24, workers=8)
        self.assertEqual(sorted(ls.cards), [1, 2])                    # card 0 is held and has a CARLA server
        self.assertEqual(ls.cards[1]["cpus"], "52-76")                 # NUMA 1, physical cores, done row ignored
        self.assertEqual(ls.cards[2]["cpus"], "77-101")
        ix = {i for c in ls.cards.values() for i in range(c["idx0"], c["idx0"] + 24)}
        taken = L.shadow(L.indices(ROWS[0]) | L.indices(ROWS[1]))
        self.assertFalse(ix & taken)
        self.assertTrue(all(i >= 160 for i in ix))
        self.assertEqual(L.conflicts([r for r in ROWS] + [ls.row()]), [])
        with self.assertRaises(RuntimeError):
            L.find_free(self.box, ROWS, "new", want=[0])

    def test_grant_and_finish(self):
        path = Path(self.tmp) / "sched/table.tsv"
        path.parent.mkdir(parents=True)
        L.save(ROWS, path)
        ls = L.find_free(self.box, ROWS, "new", n_gpus=1, cores_per_card=10, span=12, workers=4)
        ls.status = "running test"
        L.grant(ls, path)
        got = L.get("new", L.load(path))
        self.assertEqual((got.cards, got.span, got.workers), (ls.cards, 12, 4))
        bad = L.Lease("evil", {3: {"cpus": "0-2", "idx0": 300}}, 12, 4, "running")   # same block and cores as lane a
        with self.assertRaises(RuntimeError):
            L.grant(bad, path)
        L.finish("new", "done test", path)
        self.assertIsNone(L.get("new", L.load(path)))
        self.assertIn("done test", (path.parent / "archive.tsv").read_text())


def fake_box(cards=(1, 2), free_mib=85651, pids=700):
    cs = [Card(g, "GPU-%d" % g, "0000:%02x:00.0" % g, 85651 - free_mib, 85651, 0, 1) for g in cards]
    return Box(208, 75, list(range(208)), {}, {}, 20480, pids, 0, 0, (32768, 60999), cs)


@unittest.skipUnless(LINUX, "lane tests start real processes and read /proc")
class LaneRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.lease = L.Lease("t", {1: {"cpus": "", "idx0": 160}, 2: {"cpus": "", "idx0": 200}}, 24, 6, "running")

    def lane(self, jobs, **kw):
        kw.setdefault("probe_fn", lambda rows: fake_box())
        return Lane("t", jobs, self.tmp / "root", lease=self.lease, poll_s=0.1, stagger_s=0, sample_s=0, **kw)

    def sh(self, name, script, **kw):
        return Job(name, ["sh", "-c", script, "x", "{gpu}", "{idx}", "{span}", "{workers}", "{out}"], **kw)

    def test_placement_placeholders_and_done(self):
        rec = self.tmp / "seen"
        jobs = [self.sh("j%d" % i, 'echo "$CL_JOB $1 $2 $3 $4 $OMP_NUM_THREADS $CUDA_VISIBLE_DEVICES" >> %s; sleep 0.5' % rec,
                        workers=3) for i in range(5)]
        rc = self.lane(jobs).run()
        self.assertEqual(rc, 0)
        lines = [l.split() for l in rec.read_text().splitlines()]
        self.assertEqual(len(lines), 5)
        for job, gpu, idx, span, w, omp, cuda in lines:
            self.assertEqual((w, span, omp, cuda), ("3", "6", "2", gpu))
            self.assertIn(int(idx), range(160, 184) if gpu == "1" else range(200, 224))
        self.assertTrue((self.tmp / "root/DONE").exists())
        st = json.loads((self.tmp / "root/state.json").read_text())
        self.assertTrue(all(s["state"] == "done" for s in st["jobs"].values()))

    def test_capacity_never_exceeded(self):
        log = self.tmp / "conc"
        script = 'echo "+ $1 $(date +%%s.%%N)" >> {0}; sleep 0.6; echo "- $1 $(date +%%s.%%N)" >> {0}'.format(log)
        jobs = [self.sh("j%d" % i, script, workers=4) for i in range(6)]       # 4 + 4 > 6: one per card at a time
        self.assertEqual(self.lane(jobs).run(), 0)
        events = sorted((float(t), s, g) for s, g, t in (l.split() for l in log.read_text().splitlines()))
        live = {}
        for _, s, g in events:
            live[g] = live.get(g, 0) + (1 if s == "+" else -1)
            self.assertLessEqual(live[g], 1)

    def test_retry_then_error(self):
        n = self.tmp / "n"
        jobs = [self.sh("bad", "echo x >> %s; exit 3" % n, tries=2), self.sh("after", "true", deps=("bad",)),
                self.sh("ok", "true")]
        rc = self.lane(jobs).run()
        self.assertEqual(rc, 1)
        self.assertEqual(len(n.read_text().split()), 2)
        self.assertTrue((self.tmp / "root/ERROR.bad").exists())
        st = json.loads((self.tmp / "root/state.json").read_text())["jobs"]
        self.assertEqual((st["bad"]["state"], st["after"]["state"], st["ok"]["state"]), ("failed", "queued", "done"))

    def test_ok_check_and_deps_order(self):
        order = self.tmp / "order"
        a = self.sh("a", "echo a >> %s" % order)
        b = self.sh("b", "echo b >> %s" % order, deps=("a",))
        flaky = self.sh("flaky", "true", tries=1)
        flaky.ok = lambda j: False                       # rc 0 but the output check fails
        self.assertEqual(self.lane([b, a, flaky]).run(), 1)
        self.assertEqual(order.read_text().split(), ["a", "b"])

    def test_orphans_are_reaped_by_identity(self):
        pidf = self.tmp / "orphan.pid"
        job = self.sh("o", "setsid sleep 30 & echo $! > %s; exit 0" % pidf)
        self.assertEqual(self.lane([job]).run(), 0)
        pid = int(pidf.read_text())
        time.sleep(0.3)
        self.assertFalse(Path("/proc/%d" % pid).exists() and "sleep" in Path("/proc/%d/cmdline" % pid).read_text())

    def test_drain_keeps_jobs_queued(self):
        root = self.tmp / "root"
        root.mkdir(parents=True)
        (root / "DRAIN").touch()
        rc = self.lane([self.sh("x", "true")]).run()
        self.assertEqual(rc, 3)
        self.assertEqual(json.loads((root / "state.json").read_text())["jobs"]["x"]["state"], "queued")

    def test_exclusive_and_vram(self):
        log = self.tmp / "ex"
        ex = self.sh("ex", "echo ex $1 >> %s; sleep 0.5" % log, workers=2, exclusive=True, gpus=(1,))
        small = [self.sh("s%d" % i, "echo s $1 >> %s; sleep 0.3" % log, workers=1, gpus=(1,), priority=1) for i in range(2)]
        self.assertEqual(self.lane([ex] + small).run(), 0)
        self.assertEqual(log.read_text().split()[:2], ["ex", "1"])
        big = self.sh("big", "true", workers=1, vram_gb=200)
        lane = self.lane([big])
        lane.root.mkdir(parents=True, exist_ok=True)
        lane.schedule(self.lease, {})
        self.assertIn("VRAM", " ".join(lane.blocked.values()))

    def test_b2d_job_and_finished(self):
        out = self.tmp / "b2d"
        j = b2d("x", out, ["1", "2"], workers=4, min_done=0.5)
        self.assertIn("{server_args}", j.cmd)
        (out / "done").mkdir(parents=True)
        self.assertFalse(j.ok(j))
        (out / "done/1.json").write_text("{}")
        self.assertTrue(j.ok(j))
        self.assertFalse(b2d_finished(out, ["1", "2"], 1.0))


if __name__ == "__main__":
    unittest.main()
