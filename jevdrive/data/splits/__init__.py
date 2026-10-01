"""The one source of truth for dataset splits: versioned, hashed membership files under defs/ (docs/lib.md).

    from jevdrive.data import splits
    val = splits.load("nuscenes/val")             # latest version; "nuscenes/val@v1" pins one
    run.use_split(val)                             # logs val.id ("nuscenes/val@v1:3f2a...") in meta.json
    rows = df[val.mask(df.scene)]
    splits.check_disjoint(splits.load("nuscenes/train"), val)

A definition is defs/<dataset>/<name>.v<k>.json (+ <name>.v<k>.txt.gz when the list is long). Its sha256 covers the
sorted membership only; load() refuses a file whose members no longer match it. A version is never edited: a changed
membership is a new version (or a new name), so every split_id in an old meta.json still means what it meant.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFS = Path(__file__).resolve().parent / "defs"
INLINE_MAX = 5000                                     # longer lists go to a sibling .txt.gz


def membership_hash(members) -> str:
    """sha256 of the sorted, de-duplicated members as strings, one per line."""
    return hashlib.sha256(("\n".join(sorted({str(m) for m in members})) + "\n").encode()).hexdigest()


@dataclass(frozen=True)
class Split:
    dataset: str
    name: str
    version: int
    unit: str                                          # what a member is: scene / sequence / token / log / route
    members: tuple = field(repr=False)                 # sorted strings
    sha256: str = field(repr=False)
    info: dict = field(default_factory=dict, repr=False, compare=False)   # origin, used_by, notes, status

    @property
    def id(self) -> str:
        """Stable id logged by experiments: <dataset>/<name>@v<k>:<sha256[:12]>."""
        return f"{self.dataset}/{self.name}@v{self.version}:{self.sha256[:12]}"

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def __contains__(self, x) -> bool:
        return str(x) in self._set

    @property
    def _set(self) -> frozenset:
        return _frozen(self.members)

    def mask(self, values):
        """Boolean numpy array: which of `values` (any iterable, e.g. a DataFrame column) are members."""
        import numpy as np
        s = self._set
        return np.fromiter((str(v) in s for v in values), bool)


@lru_cache(maxsize=64)
def _frozen(members: tuple) -> frozenset:
    return frozenset(members)


def _path(dataset: str, name: str, version: int) -> Path:
    return DEFS / dataset / f"{name}.v{version}.json"


def versions(dataset: str, name: str) -> list:
    return sorted(int(m.group(1)) for p in (DEFS / dataset).glob(f"{name}.v*.json")
                  if (m := re.fullmatch(re.escape(name) + r"\.v(\d+)\.json", p.name)))


def available(dataset: str | None = None) -> list:
    """Every registered split as '<dataset>/<name>@v<k>'."""
    out = []
    for p in sorted(DEFS.glob(f"{dataset or '*'}/*.v*.json")):
        name, v = p.name[:-5].rsplit(".v", 1)
        out.append(f"{p.parent.name}/{name}@v{v}")
    return out


@lru_cache(maxsize=None)
def _load(dataset: str, name: str, version: int) -> Split:
    p = _path(dataset, name, version)
    d = json.loads(p.read_text())
    members = d.pop("members", None)
    if members is None:
        members = gzip.decompress(p.with_name(d.pop("members_file")).read_bytes()).decode().split("\n")[:-1]
    members = tuple(sorted({str(m) for m in members}))
    h = membership_hash(members)
    if h != d["sha256"] or len(members) != d["n"]:
        raise ValueError(f"{p}: members do not match the recorded sha256 / n (edited in place? make a new version)")
    return Split(d.pop("dataset"), d.pop("name"), d.pop("version"), d.pop("unit"), members, d.pop("sha256"), d)


def load(ref: str, version: int | None = None) -> Split:
    """'<dataset>/<name>' (latest version) or '<dataset>/<name>@v<k>'."""
    ref, _, v = ref.partition("@v")
    dataset, name = ref.split("/", 1)
    vs = versions(dataset, name)
    if not vs:
        raise KeyError(f"no split {dataset}/{name}; registered: {', '.join(available(dataset)) or 'none'}")
    return _load(dataset, name, int(v) if v else (version or vs[-1]))


def define(dataset: str, name: str, members, unit: str, origin: str, version: int | None = None, **info) -> Split:
    """Register a split (new name, or the next version of a name). Idempotent when the same membership is already
    the given / latest version; raises rather than change an existing version. info: used_by, notes, status, ..."""
    members = sorted({str(m) for m in members})
    h, vs = membership_hash(members), versions(dataset, name)
    if version is None:
        if vs and _load(dataset, name, vs[-1]).sha256 == h:
            return _load(dataset, name, vs[-1])
        version = (vs[-1] + 1) if vs else 1
    if version in vs:
        old = _load(dataset, name, version)
        if old.sha256 != h:
            raise ValueError(f"{dataset}/{name}@v{version} exists with other members; register v{vs[-1] + 1}")
        return old
    p = _path(dataset, name, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = dict(dataset=dataset, name=name, version=version, unit=unit, n=len(members), sha256=h, origin=origin, **info)
    if len(members) > INLINE_MAX:
        f = p.with_name(f"{name}.v{version}.txt.gz")
        f.write_bytes(gzip.compress(("\n".join(members) + "\n").encode(), mtime=0))
        d["members_file"] = f.name
    else:
        d["members"] = members
    p.write_text(json.dumps(d, indent=1) + "\n")
    _load.cache_clear()
    return _load(dataset, name, version)


def overlap(a: Split, b: Split) -> set:
    return a._set & b._set


def check_disjoint(*splits: Split) -> None:
    """Raise if any two of the splits share a member (train / test leakage guard)."""
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            if a.unit != b.unit:
                raise ValueError(f"{a.id} ({a.unit}) and {b.id} ({b.unit}) hold different units; map one first")
            if o := overlap(a, b):
                raise ValueError(f"{a.id} and {b.id} share {len(o)} {a.unit}s, e.g. {sorted(o)[:3]}")
