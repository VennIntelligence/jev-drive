"""Deterministic sharding and a failure-collecting parallel map, one job per file / shard (docs/lib.md).

    res = pmap(work, units, run=run, skip=lambda u: cache.done(out / f"{u}.npy", k))
    res.raise_if_failed()                      # after the whole batch: per-item tracebacks are in res.errors

Workers default to jevdrive.common.n_cpus() (the cgroup quota, not the host count). `fn` must be a module-level
function (it is pickled into worker processes); workers=0 runs inline, for debugging.
"""
from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


def cpus() -> int:
    """jevdrive.common.n_cpus(), falling back to os.cpu_count() where sched_getaffinity is missing (macOS)."""
    try:
        from .common import n_cpus
        return n_cpus()
    except AttributeError:
        return os.cpu_count() or 1


def shards(items: Sequence, n: int, i: int) -> list:
    """Shard i of n: items[i::n] (strided, so slow and fast units mix). Deterministic for a fixed item order;
    pass a sorted list or a split's members. The n shards partition the items."""
    if not 0 <= i < n:
        raise ValueError(f"shard {i} of {n}")
    return list(items)[i::n]


def shard_name(i: int, n: int) -> str:
    """File/dir name of a shard: shard-0003-of-0016."""
    return f"shard-{i:04d}-of-{n:04d}"


@dataclass
class Result:
    values: list                                       # aligned with items; None for failed / skipped
    errors: dict = field(default_factory=dict)         # index -> traceback text
    skipped: list = field(default_factory=list)        # indices skipped as already done
    items: list = field(default_factory=list)

    @property
    def ok(self) -> int:
        return len(self.values) - len(self.errors) - len(self.skipped)

    def summary(self) -> str:
        s = f"{self.ok} ok, {len(self.skipped)} skipped, {len(self.errors)} failed of {len(self.values)}"
        if self.errors:
            i = min(self.errors)
            s += f"; first failure item {self.items[i]!r}: {self.errors[i].strip().splitlines()[-1]}"
        return s

    def raise_if_failed(self) -> None:
        if self.errors:
            raise RuntimeError("pmap: " + self.summary())


def _call(fn, x):
    try:
        return True, fn(x)
    except Exception:                                  # noqa: BLE001 - collected per item, reported at the end
        return False, traceback.format_exc()


def pmap(fn: Callable[[Any], Any], items: Iterable, workers: int | None = None, run=None, desc: str = "pmap",
         skip: Callable[[Any], bool] | None = None, threads: bool = False, mp_context=None) -> Result:
    """Apply fn to every item in parallel; never stops on a failing item (see Result). `skip(item)` true means the
    unit is already done (e.g. cache.done / a DONE file) and it is not run: re-running a batch resumes it.
    threads=True uses threads (I/O-bound work). With `run` (jevdrive.run.Run): its tqdm/STATUS and a `pmap` event."""
    items = list(items)
    res = Result([None] * len(items), items=items)
    todo = [i for i, x in enumerate(items) if not (skip and skip(x))]
    res.skipped = sorted(set(range(len(items))) - set(todo))
    workers = min(cpus() if workers is None else workers, len(todo))
    bar_kw = dict(total=len(items), initial=len(res.skipped), desc=desc, dynamic_ncols=True)
    if run is not None:
        bar = run.tqdm(**bar_kw)
    else:
        from tqdm import tqdm
        bar = tqdm(**bar_kw)

    def take(i, ok, v):
        if ok:
            res.values[i] = v
        else:
            res.errors[i] = v
        bar.update(1)

    try:
        if workers <= 1:
            for i in todo:
                take(i, *_call(fn, items[i]))
        else:
            ex = ThreadPoolExecutor(workers) if threads else ProcessPoolExecutor(workers, mp_context=mp_context)
            with ex:
                futs = {ex.submit(_call, fn, items[i]): i for i in todo}
                try:
                    for f in as_completed(futs):
                        try:
                            r = f.result()
                        except Exception:              # noqa: BLE001 - a dead worker (OOM, segfault) fails its item
                            r = False, traceback.format_exc()
                        take(futs[f], *r)
                except BaseException:
                    for f in futs:
                        f.cancel()
                    raise
    finally:
        bar.close()
    if run is not None:
        run.event("pmap", desc=desc, n=len(items), ok=res.ok, skipped=len(res.skipped), failed=len(res.errors),
                  workers=workers, failures={str(items[i]): res.errors[i][-1500:] for i in sorted(res.errors)[:20]})
        (run.log.warning if res.errors else run.log.info)("%s: %s", desc, res.summary())
    return res
