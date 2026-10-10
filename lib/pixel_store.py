"""Pre-rendered protocol-W pixels for training Cinque's vision encoder, and the fast encoder step.

The cache (built by experiments/vis_train/scripts/vt_px.py, one dir per pp_prep data dir, rows in its tab.npz order):

  $DATA_DIR/runs/vis_train/px/<data>/frames.npy   (N, 8, 2, 6, 128, 256) uint8: the model frames [road, wide] at the policy steps
                                                   2, 6, .., 30 (t0 - 1.4 s .. t0, 0.2 s apart). Slot j of the policy reads the image
                                                   pair (frame j - 1, frame j); frame -1 is a zero image. The t0 pair (slot 7) is
                                                   frames 6:8, the last 786 432 bytes of a row: one contiguous read.
  $DATA_DIR/runs/vis_train/px/<data>/meta.npz     names, log, ts (microseconds) per row: the future-token index

    PS = PixelStore(datas, dev, mode="t0")       # rows index the concatenated tab.npz order of `datas`, as pp_train.Store
    prev, cur = PS.t0(rows)                      # uint8 (B, 2, 6, 128, 256) each, on dev: the slot-7 pair
    prev, cur = PS.all(rows)                     # uint8 (B, 8, 2, 6, 128, 256) each: all 8 slots (prev[:, 0] is zeros)
    row2, ok = PS.future(rows, k)                # the row of the same log k half-seconds later (ok False: not in `datas`)

Reads are positional (os.preadv releases the GIL) straight into page-locked memory on a thread pool and uploaded without blocking, so
t0 / all may be called from pp_train.Prefetch worker threads. The page cache is shared by every run that reads the same files.

    h = fast_encode(net, prev, cur, grad=True)   # (n, 2, 6, 128, 256) uint8 pairs -> (n, 32, 512) tokens of the port `net`
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevdrive.common import data_dir, n_cpus

FRAME = (2, 6, 128, 256)
NF = 8                                         # frames per row: steps 2, 6, .., 30
FB = int(np.prod(FRAME))                       # bytes per frame (393 216)
T0 = NF - 2                                    # first frame of the t0 pair
TOL_US = 100_000                               # future-row match tolerance on the log timestamp


def px_root() -> Path:
    return data_dir() / "runs" / "vis_train" / "px"


def cores() -> int:
    """Cores of this process: the pool's pin (affinity) inside the cgroup quota."""
    return max(1, min(n_cpus(), len(os.sched_getaffinity(0))))


class PixelStore:
    def __init__(self, datas: list[str], dev, mode: str = "t0", root=None, threads: int = 0):
        """mode 't0': only the t0 pair will be read (PS.all still works); 'all': full rows. threads: reader threads (0: from the box)."""
        assert mode in ("t0", "all")
        root = Path(root) if root else px_root()
        self.dev, self.mode, self.datas = dev, mode, list(datas)
        self.files = [root / d / "frames.npy" for d in datas]
        self.fd, self.base, n = [], [], []
        for f in self.files:
            m = np.load(f, mmap_mode="r")
            assert m.shape[1:] == (NF,) + FRAME and m.dtype == np.uint8, f"{f}: {m.shape} {m.dtype}"
            self.base.append(int(m.offset)), n.append(len(m))
            del m
            self.fd.append(os.open(f, os.O_RDONLY))
        self.off = np.cumsum([0] + n)
        self.n = int(self.off[-1])
        self.threads = threads or max(2, min(16, cores()))
        self.pool = ThreadPoolExecutor(self.threads)
        self._meta, self._fut = None, {}

    def __len__(self):
        return self.n

    def close(self):
        self.pool.shutdown(wait=False)
        for fd in self.fd:
            os.close(fd)
        self.fd = []

    # ---- pixels
    def host(self, rows, f0: int = 0, nf: int = NF):
        """rows -> page-locked uint8 tensor (B, nf, 2, 6, 128, 256): frames f0 .. f0 + nf of every row, one read per row."""
        import torch
        r = rows.cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        r = r.astype(np.int64).reshape(-1)
        buf = torch.empty((len(r), nf) + FRAME, dtype=torch.uint8, pin_memory=True)
        mv, sz = memoryview(buf.numpy()).cast("B"), nf * FB
        s = np.searchsorted(self.off, r, side="right") - 1
        at = [self.base[k] + int(i - self.off[k]) * NF * FB + f0 * FB for i, k in zip(r, s)]

        def rd(js):
            for j in js:
                got, fd = 0, self.fd[s[j]]
                while got < sz:
                    k = os.preadv(fd, [mv[j * sz + got:(j + 1) * sz]], at[j] + got)
                    if k <= 0:
                        raise IOError(f"short read of row {int(r[j])} in {self.files[s[j]]}")
                    got += k
        order = np.argsort(np.asarray(at) + s * (1 << 50), kind="stable")          # file order inside each thread's share
        list(self.pool.map(rd, [order[i::self.threads] for i in range(min(self.threads, len(r)))]))
        return buf

    def frames(self, rows, f0: int = 0, nf: int = NF):
        """rows -> uint8 (B, nf, 2, 6, 128, 256) on the device (upload issued without blocking)."""
        return self.host(rows, f0, nf).to(self.dev, non_blocking=True)

    def t0(self, rows):
        g = self.frames(rows, T0, 2)
        return g[:, 0], g[:, 1]

    def all(self, rows):
        import torch
        g = self.frames(rows)
        return torch.cat([torch.zeros_like(g[:, :1]), g[:, :-1]], 1), g

    # ---- the future-token index (arm B)
    def meta(self) -> dict:
        if self._meta is None:
            zs = [np.load(f.with_name("meta.npz")) for f in self.files]
            self._meta = {k: np.concatenate([z[k] for z in zs]) for k in ("names", "log", "ts")}
        return self._meta

    def future_index(self, k: int):
        """(row_index (N,), ok (N,)) of every row: the row whose log is the same and whose timestamp is 0.5 k s later (within 0.1 s)."""
        if k not in self._fut:
            m = self.meta()
            _, lg = np.unique(m["log"], return_inverse=True)
            ts = m["ts"].astype(np.int64)
            key = lg.astype(np.int64) * (1 << 44) + (ts - ts.min())                 # logs span < 2^44 us (203 days)
            assert (ts - ts.min()).max() < (1 << 44) - 10 ** 8
            o = np.argsort(key, kind="stable")
            ks, q = key[o], key + k * 500_000
            j = np.clip(np.searchsorted(ks, q), 1, len(ks) - 1)
            j = np.where(np.abs(ks[j - 1] - q) <= np.abs(ks[j] - q), j - 1, j)
            ok = np.abs(ks[j] - q) <= TOL_US
            self._fut[k] = (np.where(ok, o[j], np.arange(len(key))), ok)
        return self._fut[k]

    def future(self, rows, k: int):
        """rows -> (row_index, ok) of the token k half-seconds ahead in the same log; not ok: row_index is the row itself.
        numpy rows give numpy, tensor rows give tensors on their device."""
        import torch
        idx, ok = self.future_index(k)
        if torch.is_tensor(rows):
            r = rows.cpu().numpy()
            return torch.from_numpy(idx[r]).to(rows.device), torch.from_numpy(ok[r]).to(rows.device)
        r = np.asarray(rows)
        return idx[r], ok[r]


# ---------------------------------------------------------------- the encoder step
def fast_encode(net, prev, cur, grad: bool = True, chunk: int = 0, compiled: bool = False):
    """Image pairs (n, 2, 6, 128, 256) uint8 -> (n, 32, 512) vision tokens `view_39` of the port `net` (jevdrive.op_adapt.load).
    grad True: one pass with autograd over the whole batch (no activation checkpointing); grad False: no autograd, in chunks of
    `chunk` pairs (default 128). compiled: the pass runs through torch.compile (one graph per net and batch shape, built at first use)."""
    import torch
    from jevdrive import op_adapt as A
    f = getattr(net, "_fast_enc", None) if compiled else None
    if f is None:
        f = lambda p, c: net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), *A.H_SHAPE)  # noqa: E731
        if compiled:
            f = net._fast_enc = torch.compile(f, dynamic=False)
    n, k = len(cur), chunk or (len(cur) if grad else 128)
    with torch.set_grad_enabled(grad):
        return f(prev, cur) if k >= n else torch.cat([f(prev[i:i + k], cur[i:i + k]) for i in range(0, n, k)])
