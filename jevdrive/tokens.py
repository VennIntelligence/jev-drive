"""Zero-copy token stores: a merged pool's `front.npy` as an index over its sources' files instead of a copy.

    fr = open_front(CR / "on1_JTPH@warp/front.npy")   # np.load(f, mmap_mode="r") when the file exists, else the CatMM of front.parts.json
    fr[rows], fr[a:b], fr[rows, 7], len(fr), fr.shape  # same rows and bytes as the copied file would hold

A zero-copy merge (rf_pait.py merge, the default) writes `<pool>@warp/front.parts.json` and no front.npy: the row ranges of the source
front.npy files, in row order, paths relative to the cache root (the manifest's grandparent). A source that is itself zero-copy is
expanded into its own parts, so a manifest only ever names real files. Every part's file size and mtime are recorded and checked at open:
a part that was deleted, rebuilt or truncated fails loudly instead of feeding other rows. The sources must therefore outlive the merged pool;
`python -m jevdrive.tokens refs <cache root>` lists which manifests use which files (run it before deleting a pool).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PARTS = "front.parts.json"


def parts_file(f) -> Path:
    """The manifest that stands for the token file f (<dir>/front.npy -> <dir>/front.parts.json)."""
    return Path(f).with_name(PARTS)


def is_tokens(f) -> bool:
    """f (a front.npy path) exists as a real file or as a zero-copy manifest."""
    return Path(f).exists() or parts_file(f).exists()


class CatMM:
    """Read-only row concatenation of memory-mapped (N_i, ...) .npy files: indexing by an int, a slice, an index / bool array, or a tuple
    whose first element is one of those returns exactly what the concatenated array would (a plain ndarray, gathered part by part)."""

    def __init__(self, manifest):
        manifest = Path(manifest)
        J = json.loads(manifest.read_text())
        root = manifest.parent.parent
        self.files, self.mms, ab = [], [], []
        for p in J["parts"]:
            f = root / p["file"]
            st = f.stat() if f.exists() else None
            if st is None or (st.st_size, st.st_mtime_ns) != (p["bytes"], p["mtime_ns"]):
                raise FileNotFoundError(f"{manifest}: part {f} is missing or was rewritten (size / mtime differ from the merge's); "
                                        "a zero-copy pool needs its sources unchanged")
            m = np.load(f, mmap_mode="r")
            assert m.dtype == np.dtype(J["dtype"]) and list(m.shape[1:]) == J["shape"][1:] and p["stop"] <= len(m), f"{f}: shape / dtype"
            self.files.append(f), self.mms.append(m), ab.append((p["start"], p["stop"]))
        self.ab = np.array(ab, np.int64).reshape(-1, 2)
        self.off = np.concatenate([[0], np.cumsum(self.ab[:, 1] - self.ab[:, 0])])
        self.dtype, self.shape = np.dtype(J["dtype"]), tuple(J["shape"])
        assert self.off[-1] == self.shape[0], f"{manifest}: rows"
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def _rows(self, rows):
        if isinstance(rows, slice):
            return np.arange(self.shape[0])[rows]
        r = np.asarray(rows)
        if r.dtype == bool:
            assert r.shape == (self.shape[0],), "bool index: one entry per row"
            return np.flatnonzero(r)
        r = r.astype(np.int64)
        if ((r < -self.shape[0]) | (r >= self.shape[0])).any():
            raise IndexError(f"row index out of range for {self.shape[0]} rows")
        return np.where(r < 0, r + self.shape[0], r)

    def __getitem__(self, idx):
        rows, rest = (idx[0], idx[1:]) if isinstance(idx, tuple) else (idx, ())
        if rows is Ellipsis:
            rows, rest = slice(None), (Ellipsis,) + rest
        if isinstance(rows, (int, np.integer)):
            r = int(rows) + (self.shape[0] if rows < 0 else 0)
            if not 0 <= r < self.shape[0]:
                raise IndexError(f"row {rows} out of range for {self.shape[0]} rows")
            s = int(np.searchsorted(self.off, r, side="right") - 1)
            return np.asarray(self.mms[s][(int(self.ab[s, 0] + r - self.off[s]),) + rest])
        if np.ndim(rows) > 1 or any(not isinstance(x, (int, np.integer, slice, type(Ellipsis))) for x in rest):
            u, inv = np.unique(self._rows(np.ravel(rows)), return_inverse=True)   # broadcast / mixed fancy indexing: numpy's own semantics on
            return self[u][(inv.reshape(np.shape(rows)),) + rest]                   # the gathered unique rows
        r = self._rows(rows)
        k = np.searchsorted(self.off, r, side="right") - 1
        out = None
        for s in np.unique(k):
            m = k == s
            loc = r[m] - self.off[s] + self.ab[s, 0]
            run = loc[-1] - loc[0] == len(loc) - 1 and (np.diff(loc) == 1).all()     # a contiguous run: a slice read, no fancy gather
            g = self.mms[s][(slice(int(loc[0]), int(loc[-1]) + 1) if run else loc,) + rest]
            if out is None:
                out = np.empty((len(r),) + g.shape[1:], self.dtype)
            out[m] = g
        return out if out is not None else np.empty((0,) + self.mms[0][(slice(0, 0),) + rest].shape[1:], self.dtype)

    def take(self, rows, out):
        """out[:] = self[rows] for ascending rows (pp_train Tokens' pinned gather): np.take part by part straight into out."""
        cut = np.searchsorted(rows, self.off)
        for s in range(len(self.mms)):
            i, j = cut[s], cut[s + 1]
            if i < j:
                np.take(self.mms[s], rows[i:j] - self.off[s] + self.ab[s, 0], axis=0, out=out[i:j], mode="clip")
        return out

    def __array__(self, dtype=None, copy=None):
        a = self[:]
        return a if dtype is None else a.astype(dtype)


def open_front(f):
    """A token file f (…/front.npy) memory-mapped: np.load(f, mmap_mode="r") if it exists, else the CatMM of its front.parts.json."""
    f = Path(f)
    if f.exists():
        return np.load(f, mmap_mode="r")
    if parts_file(f).exists():
        return CatMM(parts_file(f))
    raise FileNotFoundError(f"{f} (and no {PARTS} beside it)")


def _stamp(f) -> dict:
    st = Path(f).stat()
    return dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def _entries(f, root: Path) -> tuple:
    """Manifest entries of one source token file (a real file: its whole row range; a zero-copy one: its own entries, re-rooted)."""
    f = Path(f)
    if f.exists():
        m = np.load(f, mmap_mode="r")
        return [dict(file=os.path.relpath(f, root), start=0, stop=len(m), **_stamp(f))], m.dtype, m.shape[1:]
    c = CatMM(parts_file(f))
    E = [dict(file=os.path.relpath(g, root), start=int(a), stop=int(b), **_stamp(g)) for g, (a, b) in zip(c.files, c.ab)]
    return E, c.dtype, c.shape[1:]


def link_front(dst, sources) -> dict:
    """Write dst/front.parts.json: the row concatenation of the token files `sources` (each a front.npy path, real or zero-copy), no
    tokens copied. dst must not hold a front.npy. -> the manifest."""
    dst = Path(dst)
    assert not (dst / "front.npy").exists(), f"{dst}/front.npy exists: refusing to shadow a real token file"
    root = dst.parent
    parts, dt, sh = [], None, None
    for f in sources:
        E, d, s = _entries(f, root)
        assert dt is None or (d == dt and s == sh), f"{f}: dtype / row shape {d} {s} != {dt} {sh}"
        dt, sh = d, s
        parts += E
    J = dict(shape=[sum(p["stop"] - p["start"] for p in parts)] + list(sh), dtype=str(dt), parts=parts)
    tmp = dst / (PARTS + ".tmp")
    tmp.write_text(json.dumps(J, indent=1))
    os.replace(tmp, dst / PARTS)
    return J


def refs(root) -> dict:
    """{source file (relative to root): [zero-copy pools that index it]} over every <root>/*/front.parts.json."""
    R = {}
    for m in sorted(Path(root).glob(f"*/{PARTS}")):
        for p in json.loads(m.read_text())["parts"]:
            R.setdefault(p["file"], []).append(m.parent.name)
    return R


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "refs":
        raise SystemExit("usage: python -m jevdrive.tokens refs <cache root>")
    for f, ps in refs(sys.argv[2]).items():
        print(f, " ".join(sorted(set(ps))))
