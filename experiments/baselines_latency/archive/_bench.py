"""Shared latency harness for the baseline benchmarks (see docs/baselines.md).

Stdlib + torch only, because every baseline runs in its own venv.
Protocol: batch 1, serial requests, `warmup` untimed calls, then `iters` timed calls, each bracketed by
torch.cuda.synchronize(). Reports mean / p50 / p95 / p99 in ms and peak allocated + reserved VRAM.
Raw per-request times go to $DATA_DIR/runs/bench_baselines/<model>/<tag>/<YYYYmmdd-HHMMSS>/.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch


def run_dir(model: str, tag: str) -> Path:
    root = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "runs" / "bench_baselines"
    d = root / model / tag / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bench(
    model: str,
    tag: str,
    fn: Callable[[], Any],
    *,
    warmup: int = 20,
    iters: int = 200,
    meta: dict | None = None,
    extra: Callable[[Any], dict] | None = None,
) -> dict:
    """Time `fn()` and write summary.json + times_ms.json. `extra(out)` adds per-request fields (e.g. tokens)."""
    out_dir = run_dir(model, tag)
    for i in range(warmup):
        fn()
        print(f"[{model}/{tag}] warmup {i + 1}/{warmup}", end="\r", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, extras = [], []
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        if extra is not None:
            extras.append(extra(out))
        if (i + 1) % 20 == 0:
            print(f"[{model}/{tag}] {i + 1}/{iters}  last {times[-1]:.1f} ms  mean {statistics.fmean(times):.1f} ms", flush=True)
    summary = {
        "model": model,
        "tag": tag,
        "warmup": warmup,
        "iters": iters,
        "mean_ms": statistics.fmean(times),
        "p50_ms": pct(times, 0.50),
        "p95_ms": pct(times, 0.95),
        "p99_ms": pct(times, 0.99),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        **(meta or {}),
    }
    if extras:
        keys = {k for e in extras for k in e if isinstance(e[k], (int, float))}
        summary["per_request_mean"] = {k: statistics.fmean(e[k] for e in extras if k in e) for k in sorted(keys)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "times_ms.json").write_text(json.dumps({"times_ms": times, "extras": extras}))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")
    return summary
