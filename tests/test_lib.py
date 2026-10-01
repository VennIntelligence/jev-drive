"""Shared libraries (jevdrive.run, data.splits, cache, par, stats) with fakes: no GPU, no network, no DATA_DIR.

    python -m unittest tests.test_lib -v
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevdrive import cache, par, stats  # noqa: E402
from jevdrive.data import splits as S  # noqa: E402
from jevdrive.run import Run, seed_everything  # noqa: E402


def square(x):
    if x == 3:
        raise ValueError("three is bad")
    return x * x


def events(d: Path) -> list:
    return [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_contract_on_success(self):
        with Run("exp", "tag", root=self.tmp, seed=3, config=dict(lr=0.1), probe=False) as run:
            run.info("hello %d", 1)
            run.scalar("loss", np.float32(0.5), step=1)
            for _ in run.tqdm(range(5), desc="units", status_s=0):
                pass
            run.summary["acc"] = 0.9
            d = run.dir
        self.assertEqual(d.parent, self.tmp / "exp" / "tag")
        for f in ("log.txt", "events.jsonl", "meta.json", "STATUS", "DONE"):
            self.assertTrue((d / f).exists(), f)
        self.assertFalse((d / "ERROR").exists())
        self.assertEqual(json.loads((d / "DONE").read_text())["acc"], 0.9)
        meta = json.loads((d / "meta.json").read_text())
        self.assertTrue(meta["ok"] and meta["seed"] == 3 and meta["config"] == {"lr": 0.1})
        self.assertIn("sha", meta["git"])
        self.assertIsNotNone(meta["end"])
        kinds = [e["kind"] for e in events(d)]
        self.assertEqual((kinds[0], kinds[-1]), ("start", "end"))
        self.assertIn("scalar", kinds)
        prog = [e for e in events(d) if e["kind"] == "progress"]
        self.assertEqual((prog[-1]["n"], prog[-1]["total"]), (5, 5))
        self.assertIn("hello 1", (d / "log.txt").read_text())
        self.assertIn("exp/tag: done", (d / "STATUS").read_text())

    def test_error_and_resume(self):
        with self.assertRaises(KeyError):
            with Run("exp", root=self.tmp, probe=False) as run:
                raise KeyError("boom")
        d = run.dir
        self.assertIn("KeyError: 'boom'", (d / "ERROR").read_text())
        self.assertFalse((d / "DONE").exists())
        self.assertFalse(json.loads((d / "meta.json").read_text())["ok"])
        self.assertIn("failed: KeyError", (d / "STATUS").read_text())
        with Run("exp", resume=d, probe=False) as run2:          # the same dir, the stale ERROR is rotated
            self.assertFalse((d / "ERROR").exists())
        self.assertEqual(run2.dir, d)
        self.assertTrue((d / "DONE").exists() and list(d.glob("ERROR.*")))

    def test_system_exit_zero_is_done(self):
        with self.assertRaises(SystemExit):
            with Run("exp", root=self.tmp, probe=False) as run:
                sys.exit(0)
        self.assertTrue((run.dir / "DONE").exists())

    def test_use_split_and_same_second(self):
        sp = S.Split("ds", "val", 1, "scene", ("a", "b"), S.membership_hash(["a", "b"]))
        with Run("exp", root=self.tmp, probe=False) as a, Run("exp", root=self.tmp, probe=False) as b:
            a.use_split(sp)
            self.assertNotEqual(a.dir, b.dir)
        self.assertEqual(json.loads((a.dir / "meta.json").read_text())["splits"], {"ds/val": sp.id})

    def test_seed_everything(self):
        r1 = seed_everything(7).random(3)
        x1 = np.random.rand()
        r2 = seed_everything(7).random(3)
        self.assertTrue(np.array_equal(r1, r2))
        self.assertEqual(x1, np.random.rand())


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_cached_reuse_invalidate_force(self):
        calls = []

        def make():
            calls.append(1)
            return np.arange(4)

        p, k1 = self.tmp / "a.npy", cache.key(dict(layer=1), code=make)
        for _ in range(2):
            np.testing.assert_array_equal(cache.cached(p, k1, make), np.arange(4))
        self.assertEqual(len(calls), 1)
        cache.cached(p, cache.key(dict(layer=2), code=make), make)       # new params: recomputed
        cache.cached(p, cache.key(dict(layer=2), code=make), make, force=True)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len((self.tmp / cache.MANIFEST).read_text().splitlines()), 3)
        self.assertEqual(sorted(x.name for x in self.tmp.iterdir()), ["MANIFEST.jsonl", "a.npy", "a.npy.key"])

    def test_key_sensitivity(self):
        f = self.tmp / "in.txt"
        f.write_text("x")
        k = cache.key(dict(a=1, b=[1, 2]), inputs=[f], version="1")
        self.assertEqual(k, cache.key(dict(b=(1, 2), a=1), inputs=[f], version="1"))   # order / tuple-insensitive
        self.assertNotEqual(k, cache.key(dict(a=1, b=[1, 2]), inputs=[f], version="2"))
        self.assertNotEqual(cache.key(code=square), cache.key(code=events))
        time.sleep(0.01)
        f.write_text("y")
        self.assertNotEqual(k, cache.key(dict(a=1, b=[1, 2]), inputs=[f], version="1"))

    def test_crash_mid_compute_leaves_no_valid_entry(self):
        p = self.tmp / "b.json"
        with self.assertRaises(RuntimeError):
            cache.cached(p, "k", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        self.assertFalse(p.exists() or cache.valid(p, "k"))
        self.assertEqual(cache.cached(p, "k", lambda: {"v": 1}), {"v": 1})

    def test_formats_and_feature_path(self):
        df = pd.DataFrame(dict(a=[1, 2]))
        for name, obj in (("x.csv", df), ("x.parquet", df), ("x.npz", dict(a=np.ones(2))), ("x.pkl", {1: 2})):
            cache.save(self.tmp / name, obj)
            got = cache.load(self.tmp / name)
            self.assertEqual(type(got), type(obj))
        p = cache.feature_path("/f", "qwen2.5-vl", "L18_mean", "nuscenes/val@v1:3f2a", "scene-0001")
        self.assertEqual(str(p), "/f/qwen2.5-vl/L18_mean/nuscenes-val-v1-3f2a/scene-0001.npy")


class TestPar(unittest.TestCase):
    def test_shards_partition(self):
        items = list(range(23))
        got = [par.shards(items, 4, i) for i in range(4)]
        self.assertEqual(sorted(sum(got, [])), items)
        self.assertEqual(got[1][:3], [1, 5, 9])
        self.assertEqual(par.shard_name(3, 16), "shard-0003-of-0016")
        with self.assertRaises(ValueError):
            par.shards(items, 4, 4)

    def test_pmap_failures_skip_and_run(self):
        for workers in (0, 2):
            res = par.pmap(square, range(6), workers=workers, skip=lambda x: x == 5)
            self.assertEqual(res.values, [0, 1, 4, None, 16, None])
            self.assertEqual((sorted(res.errors), res.skipped, res.ok), ([3], [5], 4))
            self.assertIn("three is bad", res.errors[3])
            with self.assertRaises(RuntimeError):
                res.raise_if_failed()
        with Run("exp", root=Path(tempfile.mkdtemp()), probe=False) as run:
            par.pmap(square, range(4), workers=2, threads=True, run=run, desc="sq")
        ev = [e for e in events(run.dir) if e["kind"] == "pmap"][0]
        self.assertEqual((ev["ok"], ev["failed"]), (3, 1))


# --- verbatim reference implementations (provenance in the comments) --------------------------------------------
def ref_boot_mean(x, n=10_000, seed=0):            # experiments/night_queue_3/lib/nq3_cl_report.py boot_mean (convention A)
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ref_boot_paired(a, b, n=10000, seed=0):        # experiments/tfv6_rules/lib/tfv6_rules.py boot_paired
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return ref_boot_mean(a[ok] - b[ok], n, seed)


def ref_boot_ci(v, scenes, b=1000, seed=0, alpha=0.05):   # jevdrive/traj.py boot_ci (convention B)
    codes, uniq = pd.factorize(scenes)
    s, n = np.bincount(codes, v, len(uniq)), np.bincount(codes, minlength=len(uniq)).astype(float)
    idx = np.random.default_rng(seed).integers(len(uniq), size=(b, len(uniq)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


class TestStats(unittest.TestCase):
    rng = np.random.default_rng(123)
    x = rng.normal(0.3, 1.0, 157)
    y = rng.normal(0.1, 1.0, 157)
    g = rng.integers(0, 23, 157).astype(str)

    def test_bit_compatible_with_dominant_conventions(self):
        for seed in (0, 1):
            r = stats.bootstrap(self.x, seed=seed)
            self.assertEqual((r["mean"], r["lo"], r["hi"]), ref_boot_mean(self.x, seed=seed))
        a, b = self.x.copy(), self.y.copy()
        a[[3, 9]] = np.nan
        r = stats.paired(a, b)
        self.assertEqual((r["mean"], r["lo"], r["hi"]), ref_boot_paired(a, b))
        self.assertEqual(r["n"], 155)
        for nb in (1000, 2000):
            r = stats.bootstrap(self.x, groups=self.g, n_boot=nb)
            self.assertEqual((r["lo"], r["hi"]), ref_boot_ci(self.x, self.g, b=nb))
            self.assertEqual(r["units"], len(set(self.g)))

    def test_empty_and_table(self):
        self.assertTrue(np.isnan(stats.bootstrap([np.nan])["mean"]))
        rows = [dict(arm=k, **stats.paired(self.x, self.y + d)) for k, d in (("a", 0), ("b", 0.2))]
        stem = Path(tempfile.mkdtemp()) / "results"
        stats.write_table(rows, stem, note="unit: route")
        self.assertEqual(list(pd.read_csv(stem.with_suffix(".csv")).columns[:6]),
                         ["arm", "n", "units", "mean", "lo", "hi"])
        md = stem.with_suffix(".md").read_text()
        self.assertIn("n_boot = 10000", md)
        self.assertNotIn("| n_boot", md)
        self.assertRegex(stats.fmt(rows[0]), r"^-?\d\.\d{3} \[-?\d\.\d{3}, -?\d\.\d{3}\]$")


class TestSplits(unittest.TestCase):
    def test_registry_verifies(self):
        refs = S.available()
        self.assertGreater(len(refs), 20)
        for ref in refs:                               # every def's members still match its recorded hash
            sp = S.load(ref)
            self.assertRegex(sp.id, r"^[\w.-]+/[\w.-]+@v\d+:[0-9a-f]{12}$")
        val, train = S.load("nuscenes/val"), S.load("nuscenes/train@v1")
        self.assertEqual((len(train), len(val)), (700, 150))
        S.check_disjoint(train, val)
        S.check_disjoint(S.load("wod/train"), S.load("wod/val"), S.load("wod/test"))
        self.assertTrue(val.mask(["nope", val.members[0]]).tolist() == [False, True])

    def test_define_is_versioned_and_immutable(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(S, "DEFS", Path(t)):
            S._load.cache_clear()
            a = S.define("ds", "dev", ["b", "a"], "scene", "test")
            self.assertEqual(S.define("ds", "dev", ["a", "b", "a"], "scene", "test").id, a.id)   # idempotent
            big = S.define("ds", "big", map(str, range(S.INLINE_MAX + 1)), "token", "test")
            self.assertTrue((Path(t) / "ds" / "big.v1.txt.gz").exists() and len(S.load("ds/big")) == S.INLINE_MAX + 1)
            b = S.define("ds", "dev", ["a", "c"], "scene", "test")                 # new members: v2, v1 kept
            self.assertEqual((b.version, S.load("ds/dev@v1").id), (2, a.id))
            with self.assertRaises(ValueError):
                S.define("ds", "dev", ["z"], "scene", "test", version=1)
            p = Path(t) / "ds" / "dev.v1.json"
            d = json.loads(p.read_text())
            p.write_text(json.dumps({**d, "members": ["a", "x"]}))                 # an in-place edit is caught
            S._load.cache_clear()
            with self.assertRaises(ValueError):
                S.load("ds/dev@v1")
            with self.assertRaises(ValueError):
                S.check_disjoint(S.load("ds/dev@v2"), S.Split("ds", "o", 1, "scene", ("c",), S.membership_hash("c")))
        S._load.cache_clear()


if __name__ == "__main__":
    unittest.main()
