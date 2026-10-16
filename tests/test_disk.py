"""jevdrive.cl.disk: the safe-prune rule and the per-job env, on fake trees (runs anywhere).

    python -m unittest tests.test_disk -v
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevdrive.cl import disk  # noqa: E402

OLD = time.time() - 3 * 86400


def mk(p: Path, n: int = 10, t: float = OLD) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)
    os.utime(p, (t, t))
    return p


class Prune(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.data, self.tmp = self.base / "data", self.base / "tmp"
        self.tmp.mkdir()
        self.c = dict(disk.DEFAULTS, caps_gb={"{data}/cache/torchinductor": 150 / disk.G})   # cap 150 bytes

    def run_prune(self, held=(set(), set()), alive=lambda pid: pid == 1, dry=False):
        # ctime cannot be set back, so "now" is moved forward past min_age_s instead
        return disk.prune(self.data, self.tmp, dry=dry, held_fn=lambda: held, alive_fn=alive,
                          now=time.time() + 2 * self.c["min_age_s"], c=self.c)

    def test_rule(self):
        root = disk.job_root(self.data)
        done, live = root / "1016-1", root / "1016-2"
        for d in (done, live):
            mk(d / "a.so", 100)
        pch_dead, pch_live = self.tmp / "__nvrtc_auto_pch_999999_0", self.tmp / "__nvrtc_auto_pch_1_0"
        mk(pch_dead / "h.pch")
        mk(pch_live / "h.pch")
        user_file = mk(self.tmp / "notes.txt", 50)
        lookalike = mk(self.tmp / "__nvrtc_auto_pch_x" / "f")           # not the pattern: a stray, kept
        os.utime(lookalike.parent, (OLD, OLD))
        ind = disk.cache_dir("torchinductor", self.data)
        old_leaf, new_leaf, open_leaf = ind / "fx" / "aa", ind / "fx" / "bb", ind / "fx" / "cc"
        mk(old_leaf / "k", 60, OLD - 100)
        mk(open_leaf / "k", 60, OLD - 50)                                # older than new_leaf but held open
        mk(new_leaf / "k", 60, OLD)
        held = ({str(open_leaf / "k")}, {str(live)})                      # a live process has TMPDIR=live
        r = self.run_prune(held)
        gone = {(Path(p).name, rule) for p, n, rule in r["removed"]}
        self.assertEqual(gone, {("1016-1", "job_tmp"), (pch_dead.name, "nvrtc_pch"), ("aa", "cache_cap")})
        self.assertFalse(done.exists() or pch_dead.exists() or old_leaf.exists())
        for p in (live, pch_live, user_file, lookalike, open_leaf, new_leaf):
            self.assertTrue(p.exists(), p)
        strays = {Path(p).name for p, n in r["strays"]}
        self.assertTrue({"notes.txt", "__nvrtc_auto_pch_x"} <= strays)
        self.assertNotIn(pch_live.name, strays)

    def test_dry_run_and_cap_not_reached(self):
        root = disk.job_root(self.data)
        mk(root / "j" / "f")
        r = self.run_prune(dry=True)
        self.assertEqual([Path(p).name for p, n, rule in r["removed"]], ["j"])
        self.assertTrue((root / "j").exists())
        mk(disk.cache_dir("torchinductor", self.data) / "a" / "k", 50)   # under the 150-byte cap: untouched
        self.assertEqual(self.run_prune()["removed"][-1][2], "job_tmp")

    def test_job_env_and_release(self):
        env = disk.job_env("1016-x", data=self.data)
        t = Path(env["TMPDIR"])
        self.assertEqual(t, disk.job_root(self.data) / "1016-x")
        self.assertTrue(t.is_dir() and Path(env["CUDA_CACHE_PATH"]).is_dir())
        mk(t / "f", 1000)
        self.assertEqual(disk.release("1016-x", data=self.data), 1000)
        self.assertFalse(t.exists())
        self.assertEqual(disk.release("../cache", data=self.data), 0)      # never outside the job root
        self.assertTrue(disk.cache_dir("torchinductor", self.data).exists())


if __name__ == "__main__":
    unittest.main()
