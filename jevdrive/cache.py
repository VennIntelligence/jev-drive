"""Idempotent artifact cache: recompute a file only when its inputs, params or code changed (docs/lib.md).

    k = key(params=dict(layer=12), inputs=[ckpt], code=extract)      # code: functions whose source is hashed
    feats = cached(out / "feats.npy", k, lambda: extract(...), force=args.force)

An artifact is valid iff the file exists and its sidecar `<file>.key` holds the same key. Writes are atomic (tmp +
rename), the sidecar is written last, so a crash leaves a stale entry that is recomputed, never a half file that is
reused. Every write appends one line to `<dir>/MANIFEST.jsonl`. Feature files are named by `feature_path`.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable

KEY_SUFFIX = ".key"
MANIFEST = "MANIFEST.jsonl"


def _canon(x) -> Any:
    """JSON-stable form: dict keys sorted, paths/tuples/sets/numpy normalised."""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_canon(v) for v in x)
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def file_sig(path, content: bool = False) -> dict:
    """Identity of an input file: (path, size, mtime_ns), or a sha256 of its bytes with content=True."""
    p = Path(path)
    st = p.stat()
    if not content:
        return dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return dict(path=str(p), size=st.st_size, sha256=h.hexdigest())


def code_sig(*objs) -> str:
    """sha256 of the source of functions / classes / modules: edits to them invalidate the cache."""
    return hashlib.sha256("\n".join(inspect.getsource(o) for o in objs).encode()).hexdigest()[:16]


def key(params: dict | None = None, inputs=(), code=(), version: str = "", content: bool = False) -> str:
    """Cache key (sha256 hex, 24 chars) over params, input files (see file_sig), source of `code` and a free-form
    version string (bump it when something the key cannot see changed, e.g. a library upgrade)."""
    code = code if isinstance(code, (list, tuple)) else [code]
    doc = dict(params=_canon(params or {}), inputs=[file_sig(p, content) for p in inputs],
               code=code_sig(*code) if code else "", version=version)
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:24]


# ---------------------------------------------------------------------- (de)serialisation by suffix
def save(path: Path, obj) -> None:
    """Atomic write by suffix: .npy .npz .pt .json .parquet .csv, else pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}{path.suffix}")
    s = path.suffix
    try:
        if s == ".npy":
            import numpy as np
            np.save(tmp, obj)
        elif s == ".npz":
            import numpy as np
            np.savez(tmp, **obj)
        elif s == ".pt":
            import torch
            torch.save(obj, tmp)
        elif s == ".json":
            tmp.write_text(json.dumps(_canon(obj), indent=1))
        elif s == ".parquet":
            obj.to_parquet(tmp)
        elif s == ".csv":
            obj.to_csv(tmp, index=False)
        else:
            with tmp.open("wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load(path: Path):
    path, s = Path(path), Path(path).suffix
    if s == ".npy":
        import numpy as np
        return np.load(path)
    if s == ".npz":
        import numpy as np
        with np.load(path) as z:
            return dict(z)
    if s == ".pt":
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    if s == ".json":
        return json.loads(path.read_text())
    if s in (".parquet", ".csv"):
        import pandas as pd
        return pd.read_parquet(path) if s == ".parquet" else pd.read_csv(path)
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------- cache
def _keyfile(path: Path) -> Path:
    return path.with_name(path.name + KEY_SUFFIX)


def valid(path, k: str) -> bool:
    """Does `path` hold the artifact for key `k`?"""
    path = Path(path)
    try:
        return path.exists() and _keyfile(path).read_text().split("\n", 1)[0] == k
    except OSError:
        return False


def mark(path, k: str, **info) -> None:
    """Record that `path` (already written) is the artifact for key `k`: sidecar + manifest line."""
    path = Path(path)
    kf = _keyfile(path)
    tmp = kf.with_name(f".{kf.name}.tmp{os.getpid()}")
    tmp.write_text(k + "\n")
    os.replace(tmp, kf)
    line = dict(t=time.strftime("%F %T"), file=path.name, key=k, bytes=path.stat().st_size if path.is_file() else None,
                **info)
    with (path.parent / MANIFEST).open("a") as f:     # one short O_APPEND write per line: safe across processes
        f.write(json.dumps(line, default=str) + "\n")


def cached(path, k: str, fn: Callable[[], Any], force: bool = False, writer: Callable | None = None,
           reader: Callable | None = None):
    """Return the artifact at `path` if it is valid for key `k`, else compute `fn()`, save it atomically and return it.
    force=True always recomputes. writer(path, obj) / reader(path) override the by-suffix save / load (a writer must
    itself write atomically, or write to a path and let cached rename; the default does)."""
    path = Path(path)
    if not force and valid(path, k):
        return (reader or load)(path)
    t0 = time.time()
    obj = fn()
    (writer or save)(path, obj)
    mark(path, k, wall_s=round(time.time() - t0, 2))
    return obj


def done(path, k: str) -> bool:
    """Alias of `valid`, for skip-if-done checks in pmap / shard loops."""
    return valid(path, k)


# ---------------------------------------------------------------------- feature naming
def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.=" else "-" for c in str(s)).strip("-")


def feature_path(root, model: str, version: str, split_id: str, unit, ext: str = ".npy") -> Path:
    """Convention for feature files: <root>/<model>/<version>/<split_id slug>/<unit><ext>.

    `version` names the extraction recipe (layer, resolution, dtype, code rev); `split_id` comes from
    jevdrive.data.splits (its hash suffix makes the dir content-addressed: a changed split gets a new dir);
    `unit` is the shard / scene / route the file holds. Old feature dirs need not be migrated."""
    return Path(root) / _slug(model) / _slug(version) / _slug(split_id) / f"{_slug(unit)}{ext}"
