"""Twin token stores (round7 HARM-MIX, experiments/round7/plans/2026-10-22-harmonizer-mix.md): the same pool row rendered a second way
(Harmonizer ON) whose tokens replace the row's own with probability p when the row is drawn. Everything else of the row (labels, ego,
route features, teacher, name, time scale) stays the row's own.

  store    $DATA_DIR/runs/op_parity/cache/on1_<NAME>/tab.npz (names: the source pool row names, e.g. on:paih:<scene>:<k>) and
           on1_<NAME>@warp/front.npy or front.parts.json (zero-copy), row-aligned with the names
  match    a training store row matches a twin row by name; merged pools (tab `pool`) name their rows on:<part>:<source name without on:>,
           so both sides are compared without their on:[<part>:] prefix (rm_train open_rows' rule)
  coin     per drawn row with a twin, swap with probability p; the generator is seeded [seed, 0, 223, <the batch's pool-draw hash>]: one
           stream per batch, independent of the fetch threads' order and of every other stream of the trainer
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

STREAM = 223


def strip(names, parts=None) -> list:
    """Row names without on:[<part>:] (parts: the merged pool's per-row part prefix, or None)."""
    if parts is None:
        return [x[3:] if x.startswith("on:") else x for x in names]
    return [x[len(f"on:{p}:"):] if x.startswith(f"on:{p}:") else (x[3:] if x.startswith("on:") else x) for x, p in zip(names, parts)]


class Twin:
    def __init__(self, spec: str, cache_root, pool_tab, on0: int, n: int, seed: int):
        """spec = <store name>:<p> (store dir on1_<name> or the full dir name); pool_tab = the --on pool's tab (rows on0 .. n - 1 of the
        training store)."""
        from jevdrive.tokens import open_front
        name, p = spec.rsplit(":", 1)
        name = name if name.startswith("on1_") else f"on1_{name}"
        self.p, self.seed, self.name = float(p), int(seed), name
        assert 0.0 <= self.p <= 1.0, f"--twin {spec}: p outside [0, 1]"
        cr = Path(cache_root)
        tn = np.load(cr / name / "tab.npz")["names"].astype(str)
        self.front = open_front(cr / f"{name}@warp" / "front.npy")
        assert len(self.front) == len(tn), f"{name}: {len(self.front)} token rows for {len(tn)} names"
        pn = pool_tab["names"].astype(str)
        assert len(pn) == n - on0, f"--twin: pool tab has {len(pn)} rows, the store {n - on0} pool rows"
        key = strip(pn.tolist(), pool_tab["pool"].astype(str).tolist() if "pool" in pool_tab.files else None)
        pos = {x: i for i, x in enumerate(strip(tn.tolist()))}
        self.ix = np.full(n, -1, np.int64)                           # store row -> twin row (-1: none)
        self.ix[on0:] = [pos.get(x, -1) for x in key]
        self.npool = n - on0
        self.lock, self.drawn, self.swapped = threading.Lock(), 0, 0
        self.rec = dict(store=name, p=self.p, stream=[self.seed, 0, STREAM, "pool-draw hash"], twin_rows=int(len(tn)),
                        pool_rows_with_twin=int((self.ix >= 0).sum()), twin_rows_unmatched=int(len(tn) - len(set(self.ix[self.ix >= 0].tolist()))))

    def mass(self, w=None, rows=None) -> float:
        """Draw mass of the pool rows with a twin: w = per-pool-row draw weights, or rows = the drawable pool rows (uniform); neither = all
        pool rows uniform."""
        h = self.ix[len(self.ix) - self.npool:] >= 0
        if w is not None:
            w = np.asarray(w, np.float64)
            return float(w[h].sum() / w.sum())
        if rows is not None:
            return float(h[np.asarray(rows)].mean())
        return float(h.mean())

    def swap(self, rows, hsh: str, front):
        """front (the fetched tokens of `rows`, a tensor on the card) with the drawn twin rows replaced in place -> front."""
        import torch
        j = self.ix[np.asarray(rows)]
        has = np.flatnonzero(j >= 0)
        if not len(has):
            return front
        g = np.random.default_rng([self.seed, 0, STREAM, int(hsh, 16)])
        hit = has[g.random(len(has)) < self.p]
        with self.lock:
            self.drawn += len(has)
            self.swapped += len(hit)
        if len(hit):
            o = np.argsort(j[hit], kind="stable")
            tok = np.empty((len(hit),) + tuple(self.front.shape[1:]), np.float16)
            tok[o] = self.front[j[hit][o]]
            front[torch.as_tensor(hit, device=front.device)] = torch.from_numpy(tok).to(front.device, front.dtype)
        return front

    def summary(self) -> dict:
        return self.rec | dict(drawn_with_twin=self.drawn, swapped=self.swapped, swap_share=self.swapped / max(1, self.drawn))
