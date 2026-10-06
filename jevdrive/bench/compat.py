"""Legacy lane inputs/outputs at the boundary of bench; execution stays in the shared stages.

Readers prefer bench artifacts, falling back to the supplied historical location. HUGSIM exports retag raw rows while
keeping their real run_dir (traces never get copied); locked upserts let old reports keep their existing filenames.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
from pathlib import Path

from . import runner as R
from .models import data_dir, resolve


def read_rows(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def trace_dir(row: dict, legacy_root=None) -> Path:
    """Use the recorded trace path; relocated historical exports can still reconstruct their local directory."""
    d = Path(row["run_dir"])
    if d.exists() or legacy_root is None:
        return d
    return Path(legacy_root) / row["tag"] / d.parent.name / d.name


def navtest_csv(spec: str, legacy=None):
    from .navsim import SUBS
    d = R.bench_root("navtest", resolve(spec).key("navtest"))
    if (d / "units.csv").exists() and (d / "DONE").exists():
        import pandas as pd
        u = pd.read_csv(d / "units.csv").rename(columns=SUBS)
        mean = u.select_dtypes("number").mean().to_dict()
        mean["token"] = "average"
        u = pd.concat([u, pd.DataFrame([mean])], ignore_index=True)
        out = d / "scores.csv"
        R.atomic_write(out, u.to_csv(index=False))
        return out
    return Path(legacy) if legacy and Path(legacy).exists() else None


def navhard_dir(spec: str, legacy=None):
    d = R.bench_root("navhard", resolve(spec).key("navhard")) / "harness"
    return d if (d / "harness_groups.csv").exists() else Path(legacy) if legacy else d


def pred_file(spec: str, bench: str = "navtest", legacy=None):
    from .navsim import pred_file as new_pred
    p = new_pred(resolve(spec), bench)
    return p if p.exists() else Path(legacy) if legacy else p


def hugsim_rows(spec: str, preset="exam", legacy=None, tag="", **kw) -> list:
    from . import run_dir
    d = run_dir(spec, "hugsim", preset, **kw)
    rows = {}
    for f, source_tag in ((Path(legacy) if legacy else None, tag), (d / "results.csv", "bench")):
        if f and f.exists():
            for r in read_rows(f):
                if r["tag"] == source_tag and r["end"] != "crash":
                    rows[r["scenario"]] = dict(r, tag=tag or r["tag"])
    return list(rows.values())


def parity_hugsim_rows(want: dict, legacy=None) -> list:
    aliases = {"specplan": "spec_plan", "smooth": "spec_plan_smooth", "mpc": "spec_plan_mpc"}
    legacy = legacy or data_dir() / "runs/op_parity/hugsim/results.csv"
    return [r for tag, (preset, model) in want.items()
            for r in hugsim_rows(model, aliases.get(preset, preset), legacy, tag)]


def publish_hugsim(run_dir, out, tag, allow_partial=False) -> None:
    """Publish only a complete run; repeat/rule identities stay in the canonical config and manifest."""
    import fcntl
    d, out = Path(run_dir), Path(out)
    sm = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    if not allow_partial and (sm.get("missing") or not (d / "DONE").exists()):
        raise RuntimeError(f"cannot publish incomplete HUGSIM run: {d}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".bench-publish.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        csv_path = out / "results.csv"
        rows = read_rows(csv_path)
        manifest = out / "bench-runs.json"
        runs = json.loads(manifest.read_text()) if manifest.exists() else {}
        if tag in runs and runs[tag] != str(d):
            rows = [r for r in rows if r["tag"] != tag]    # changing a rule under a legacy tag must not mix configurations
        new = [dict(r, tag=tag) for r in read_rows(d / "results.csv") if r["tag"] == "bench" and r["end"] != "crash"]
        keys = {(r["tag"], r["scenario"]) for r in new}
        rows = [r for r in rows if (r["tag"], r["scenario"]) not in keys] + new
        columns = list(dict.fromkeys(k for r in rows for k in r))
        buf = io.StringIO()
        w = csv.DictWriter(buf, columns)
        w.writeheader()
        w.writerows(rows)
        R.atomic_write(csv_path, buf.getvalue())
        runs[tag] = str(d)
        R.atomic_write(manifest, json.dumps(runs, indent=1))


def in_pool(model: str, bench: str, **kw):
    """Compatibility for an already leased shell worker: run the same stages without nesting pool submissions."""
    from . import plan
    if not os.environ.get("CL_POOL_JOB"):
        raise RuntimeError("--in-pool requires an existing GPU-pool job")
    if bench != "hugsim":
        raise ValueError("--in-pool compatibility is for HUGSIM workers only")
    kw["jobs"] = 1
    d, stages = plan(model, bench, **kw)
    for s in stages:
        if Path(s.done).exists():
            continue
        subprocess.run(s.cmd, check=True, cwd=R.REPO, env=dict(os.environ, **s.env))
    return d
