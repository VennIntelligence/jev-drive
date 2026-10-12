"""junction_eyes step 1 data: pixel pair stores, the keep (comma) store, row sources, a thread prefetcher.

Pixel pairs (executor 2, results/pixels.md): npy (N, [V,] 2, 2, 6, 128, 256) uint8 = ([view,] [prev, cur], [road, wide], 6, 128, 256),
the t0 image pair of a row (J_nav / navtest: protocol W's t0 slot; J_on / held / zero-98: the driver's (slot 6, slot 7) pair).

    PS = PairStore(path, view=None)           # view: the index on the view axis of a multi-view file (e.g. px_side)
    prev, cur = PS.get(rows)                  # (B, 2, 6, 128, 256) uint8 each, page-locked host tensors

Keep store (comma1M, model frames at the 5 Hz context phase + the factory teacher, see KeepStore):

    KS = KeepStore(dir)
    prev, cur, valid = KS.get(rows)           # (B, 9, 2, 6, 128, 256) uint8 pairs (slot j = (frame j, frame j + 1)), valid (B, 9)

Reads are positional (os.preadv releases the GIL) into page-locked memory on a thread pool, as lib/pixel_store.py.
"""
from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FRAME = (2, 6, 128, 256)                       # one model frame [road, wide]
FB = int(np.prod(FRAME))                       # 393 216 bytes
CTX = 9                                        # policy context: 9 hidden states, 0.2 s apart


def _cores() -> int:
    from jevdrive.common import n_cpus
    try:
        return max(1, min(n_cpus(), len(os.sched_getaffinity(0))))
    except AttributeError:
        return n_cpus()


class _Reader:
    """Positional reads of fixed-size records of one npy file into page-locked buffers."""

    def __init__(self, path, threads=0):
        self.path = Path(path)
        m = np.load(self.path, mmap_mode="r")
        assert m.dtype == np.uint8, f"{path}: {m.dtype}"
        self.shape, self.base = m.shape, int(m.offset)
        self.rec = int(np.prod(m.shape[1:]))
        del m
        self.fd = os.open(self.path, os.O_RDONLY)
        self.threads = threads or max(2, min(16, _cores()))
        self.pool = ThreadPoolExecutor(self.threads)

    def __len__(self):
        return self.shape[0]

    def read(self, rows, off: int, size: int, out_shape):
        """rows (n,) -> page-locked uint8 tensor (n, *out_shape): `size` bytes at byte `off` of each record."""
        import torch
        r = np.asarray(rows, np.int64).reshape(-1)
        buf = torch.empty((len(r),) + tuple(out_shape), dtype=torch.uint8, pin_memory=torch.cuda.is_available())
        mv = memoryview(buf.numpy()).cast("B")
        at = self.base + r * self.rec + off

        def rd(js):
            for j in js:
                got = 0
                while got < size:
                    k = os.preadv(self.fd, [mv[j * size + got:(j + 1) * size]], int(at[j]) + got)
                    if k <= 0:
                        raise IOError(f"short read of record {int(r[j])} in {self.path}")
                    got += k
        order = np.argsort(at, kind="stable")
        list(self.pool.map(rd, [order[i::self.threads] for i in range(min(self.threads, len(r)))]))
        return buf


class PairStore:
    def __init__(self, path, view: int | None = None, threads=0):
        self.R = _Reader(path, threads)
        sh = self.R.shape[1:]
        if view is None:
            assert sh == (2,) + FRAME, f"{path}: rows {sh}, want (2, 2, 6, 128, 256) or pass view="
            self.off = 0
        else:
            assert sh[1:] == (2,) + FRAME and 0 <= view < sh[0], f"{path}: rows {sh}, view {view}"
            self.off = view * 2 * FB

    def __len__(self):
        return len(self.R)

    def get(self, rows):
        g = self.R.read(rows, self.off, 2 * FB, (2,) + FRAME)
        return g[:, 0], g[:, 1]


class KeepStore:
    """dir/frames.npy (F, 2, 6, 128, 256) uint8: model frames at the 5 Hz context phase (every 4th 20 Hz frame).
    dir/rows.npz: fidx (N, 10) int64 frame indices [t0 - 1.8 s, .., t0] (-1 = before the segment start: zero image / no hidden
    state, as the 20 Hz stream from a zero state), teacher (N, n_out) float32 = the factory model's raw `outputs` at t0 stepped
    modeld-style at 20 Hz from the segment start (jevdrive.op_adapt.Stepper), seg (N,) str, held (N,) bool (the 60 held-out
    segments: never trained on), action_t (2,)."""

    def __init__(self, d, threads=0):
        d = Path(d)
        self.R = _Reader(d / "frames.npy", threads)
        assert self.R.shape[1:] == FRAME, f"{d}/frames.npy: {self.R.shape}"
        z = np.load(d / "rows.npz", allow_pickle=False)
        self.fidx, self.teacher = z["fidx"].astype(np.int64), z["teacher"].astype(np.float32)
        self.seg, self.held = z["seg"], z["held"].astype(bool)
        self.action_t = tuple(float(x) for x in z["action_t"]) if "action_t" in z.files else (0.275, 0.525)
        assert self.fidx.shape[1] == CTX + 1 and self.fidx.max() < len(self.R)
        self.n = len(self.fidx)

    def __len__(self):
        return self.n

    def get(self, rows):
        """-> prev, cur (B, 9, 2, 6, 128, 256) uint8 page-locked, valid (B, 9) bool (hidden state j exists: its cur frame does)."""
        import torch
        f = self.fidx[np.asarray(rows)]
        u, inv = np.unique(np.maximum(f, 0), return_inverse=True)
        fr = self.R.read(u, 0, FB, FRAME)[torch.from_numpy(inv.reshape(f.shape))]          # (B, 10, ...)
        fr[torch.from_numpy(f < 0)] = 0
        return fr[:, :-1], fr[:, 1:], torch.from_numpy(f[:, 1:] >= 0)


class Prefetch:
    """Batches fetched ahead on worker threads; draws happen in order on the caller's thread (deterministic streams)."""

    def __init__(self, draw, fetch, n, depth=4, workers=3):
        self.q, self.ex = queue.Queue(), ThreadPoolExecutor(workers)
        self.draw, self.fetch, self.left = draw, fetch, n
        self.lock = threading.Lock()
        for _ in range(min(depth, n)):
            self._push()

    def _push(self):
        if self.left > 0:
            self.left -= 1
            d = self.draw()
            self.q.put(self.ex.submit(self.fetch, d))

    def get(self):
        f = self.q.get()
        self._push()
        return f.result()


# ---------------------------------------------------------------- row sources
class Source:
    """One training row source in the trainer's form. px: PairStore; ego (N, 20) f32; ra (N, 15, 2) f32 (nan = no label);
    rb (N, 48, 48) int8 (-1 = no label); w (N,) sampling weight; fit / val (N,) bool; names (N,) str; unit (N,) str (log / scene)."""

    def __init__(self, name, px, ego, ra, rb, fit, val, names, unit, w=None):
        n = len(px)
        assert all(len(x) == n for x in (ego, ra, rb, fit, val, names, unit)), name
        assert not (fit & val).any(), f"{name}: a row is in both fit and val"
        assert not (set(unit[fit].tolist()) & set(unit[val].tolist())), f"{name}: a log / scene unit is in both fit and val"
        self.name, self.px, self.ego, self.ra, self.rb = name, px, ego.astype(np.float32), ra.astype(np.float32), rb.astype(np.int8)
        self.fit, self.val, self.names, self.unit = fit, val, names, unit
        self.w = np.ones(n) if w is None else np.asarray(w, float)

    def fit_rows(self):
        return np.flatnonzero(self.fit)

    def fit_p(self):
        r = self.fit_rows()
        p = self.w[r]
        return r, p / p.sum()


def per_unit_weight(unit: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """1 / (rows of the unit inside mask): every scene draws the same share (J_on is sampled uniformly per scene)."""
    w = np.zeros(len(unit))
    u, inv, cnt = np.unique(unit[mask], return_inverse=True, return_counts=True)
    w[np.flatnonzero(mask)] = 1.0 / cnt[inv]
    return w


def synthetic_source(name, px_path, view=None, n=0, seed=0, val_frac=0.1):
    """Plumbing only (smoke / unit tests): real image pairs, random labels with the real shapes and scales. Never read."""
    px = PairStore(px_path, view)
    n = n or len(px)
    rng = np.random.default_rng([seed, 99, len(name)])
    rows = np.arange(n)
    unit = np.array([f"{name}-u{i // 40}" for i in rows])
    val = np.isin(unit, np.unique(unit)[::10]) if val_frac else np.zeros(n, bool)
    s = np.linspace(5, 30, 5)
    ra = np.stack([np.repeat(s, 3), np.tile([0.0, 1.8, -1.8], 5)], -1)[None] + rng.normal(0, 1.0, (n, 15, 2))
    ra[rng.random((n, 15)) < 0.05] = np.nan
    rb = rng.integers(0, 3, (n, 48, 48)).astype(np.int8)
    sub = _Sub(px, n)
    return Source(name, sub, rng.normal(0, 1, (n, 20)), ra, rb, ~val, val, np.array([f"{name}-{i}" for i in rows]), unit)


class MultiPairStore:
    """Rows spread over several pair files: row i lives in file fi[i] at row ri[i]."""

    def __init__(self, paths, fi, ri, view=None, threads=0):
        self.st = [PairStore(p, view, threads) for p in paths]
        self.fi, self.ri = np.asarray(fi), np.asarray(ri, np.int64)
        for k, s in enumerate(self.st):
            assert self.ri[self.fi == k].max(initial=-1) < len(s), paths[k]

    def __len__(self):
        return len(self.fi)

    def get(self, rows):
        import torch
        r = np.asarray(rows)
        fi, ri = self.fi[r], self.ri[r]
        parts = {k: np.flatnonzero(fi == k) for k in np.unique(fi)}
        got = {k: self.st[k].get(ri[j]) for k, j in parts.items()}
        out = []
        for i in range(2):
            o = torch.empty((len(r),) + FRAME, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
            for k, j in parts.items():
                o[torch.from_numpy(j)] = got[k][i]
            out.append(o)
        return out[0], out[1]


def real_source(name: str, rows_parquet, labels_dir, px_root, cache_root, px_file="frames-t0.npy", per_unit=False) -> Source:
    """A J source from executor 1's rows (results/rows.md) and executor 2's pixel caches (results/pixels.md): labels by `rid`, ego from
    the frozen-token cache's tab.npz (`tab_file`, `tok_row`), pixels from <px_root>/<tab_file>/<px_file> at row `tok_row`."""
    import pandas as pd
    df = pd.read_parquet(rows_parquet)
    df = df[df.split.isin(["fit", "val"])].reset_index(drop=True)
    labels_dir = Path(labels_dir)
    ra = np.load(labels_dir / "ra.npy", mmap_mode="r")[df.rid.values].reshape(-1, 15, 2)
    rb = np.load(labels_dir / "rb.npy", mmap_mode="r")[df.rid.values].astype(np.int8)
    files = sorted(df.tab_file.unique())
    fi = df.tab_file.map({f: i for i, f in enumerate(files)}).values
    ego = np.zeros((len(df), 20), np.float32)
    for i, f in enumerate(files):
        m = fi == i
        z = np.load(Path(cache_root) / f / "tab.npz")
        ego[m] = z["ego"][df.tok_row.values[m]]
        if name == "nav":                                   # navtrain token caches carry the token names
            assert (z["names"][df.tok_row.values[m]] == df.token.values[m]).all(), f"{f}: tab rows do not match the row table"
    px = MultiPairStore([Path(px_root) / f / px_file for f in files], fi, df.tok_row.values)
    fit, val = (df.split == "fit").values, (df.split == "val").values
    w = df.w_scene.values if per_unit else None
    return Source(name, px, ego, ra, rb, fit, val, df.token.values.astype(str), df.unit.values.astype(str), w)


class _Sub:
    def __init__(self, px, n):
        self.px, self.n = px, n

    def __len__(self):
        return self.n

    def get(self, rows):
        return self.px.get(rows)


def comma_frames(seg: str, max_frames: int = 1200) -> np.ndarray:
    """20 Hz model frames (n, 2, 6, 128, 256) of a comma1M segment (jevdrive.openpilot.frames.segment_model_frames). The hevc decoder
    (PyAV) lives in the openpilot env, not in op-train, so the decode runs there in a child process and comes back through a temp npy."""
    import subprocess
    import tempfile
    from jevdrive.common import data_dir
    repo = Path(__file__).resolve().parents[3]
    py = data_dir() / "envs" / "openpilot" / "bin" / "python"
    with tempfile.TemporaryDirectory(dir=data_dir() / "tmp") as d:
        out = Path(d) / "f.npy"
        code = ("import sys, numpy as np; sys.path.insert(0, sys.argv[1]); from jevdrive.openpilot.frames import segment_model_frames as f; "
                "np.save(sys.argv[4], f(sys.argv[2], max_frames=int(sys.argv[3]))[0])")
        subprocess.run([str(py), "-c", code, str(repo), str(data_dir() / "datasets" / "comma1M" / seg), str(max_frames), str(out)], check=True)
        return np.load(out)
