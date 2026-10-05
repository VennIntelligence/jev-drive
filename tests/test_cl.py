"""jevdrive.cl box probe, capacity model and profiles with fakes: no GPU, no CARLA, runs anywhere (fake /sys and /proc
trees). The pool's tests are tests/test_pool.py.

    python -m unittest tests.test_cl -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevdrive.cl import capacity, profiles  # noqa: E402
from jevdrive.cl.box import Box, Card, format_cpus, parse_cpus, probe  # noqa: E402


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
    write(tmp, "sys/fs/cgroup/memory.stat", "anon 10737418240\nfile 107374182400\nkernel 1073741824\nshmem 0\n")
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
        self.assertAlmostEqual(b.mem_used_gb, 11.0, places=3)   # page cache ("file") not counted
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


if __name__ == "__main__":
    unittest.main()
