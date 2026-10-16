"""Desire training helpers (WAIT-TRAIN prereg Amendment C, experiments/wait_cap/plans/2026-10-16-wait-train-prereg.md; design
experiments/desire_cap/plans/2026-10-16-desire-route-design.md section 2): the policy's desire queue in dc.py's protocol, the D1 row table,
the D2 teacher (shipped Cinque's desire increment) and the D2 loss. Used by experiments/route_mem/scripts/rm_train.py (--desire-rows,
--d1-k, --d2-lam) and experiments/desire_cap/scripts/dt.py. The lane-change rule itself (lc_intent, speed gate) is lib/route_desire.py's
(the D0 line's); nothing here decides when a desire fires.

Queue (dc.py, read from the ONNX): `_to_copy` (B, 1, 33, 8), block 32 = the newest 0.2 s; modeld feeds only the rising edge, so a pulse of
age D s sits in block 32 - round(D / 0.2). Desire index: log.Desire laneChangeLeft 3, laneChangeRight 4. Side +1 = left.
Arms of the D2 teacher, in this order: LCL at AGES, then LCR at AGES (8 arms).
"""
from __future__ import annotations

import numpy as np

AGES = (0.0, 0.4, 1.0, 2.0)                      # s; dc.py's pulse ages
BLK = tuple(32 - int(round(a / 0.2)) for a in AGES)
LC = {1: 3, -1: 4}                               # side -> log.Desire laneChangeLeft / laneChangeRight
N_ARM = 2 * len(AGES)


def arm_of(side, age_i):
    """side (+1 / -1) and age index -> teacher arm index (array-friendly)."""
    return np.where(np.asarray(side) > 0, 0, len(AGES)) + np.asarray(age_i)


def queue(side, age_i, dev, dtype=None):
    """side (B,) int tensor (+1 left, -1 right, 0 none), age_i (B,) int tensor -> desire queue (B, 1, 33, 8); rows with side 0 stay all zero."""
    import torch
    B = len(side)
    q = torch.zeros(B, 1, 33, 8, device=dev, dtype=dtype or torch.float32)
    m = side != 0
    if m.any():
        b = torch.nonzero(m)[:, 0]
        blk = torch.as_tensor(BLK, device=dev)[age_i[b]]
        d = torch.where(side[b] > 0, LC[1], LC[-1])
        q[b, 0, blk, d] = 1.0
    return q


def rows_side(names, n_base, on0, tab):
    """D1 row table (npz: names, side, src 'nav' / 'pool') -> per Store row side (n,) int8 (+1 / -1 / 0) and the table's navtrain rows (store
    index). names: the Store's names; navtrain names are matched in rows [0, n_base) (the normal rows; the yaw-rate copies repeat the
    tokens), pool names (on:...) in rows [on0, n)."""
    side = np.zeros(len(names), np.int8)
    src, nm, sd = tab["src"].astype(str), tab["names"].astype(str), tab["side"].astype(np.int8)
    names = np.asarray(names).astype(str)
    for lo, hi, k in ((0, n_base, "nav"), (on0, len(names), "pool")):
        m = src == k
        if not m.any():
            continue
        pos = {x: i for i, x in enumerate(names[lo:hi].tolist())}
        r = np.array([pos.get(x, -1) for x in nm[m].tolist()], np.int64)
        assert (r >= 0).all(), f"D1 table: {int((r < 0).sum())} {k} names not in the Store"
        side[lo + r] = sd[m]
    return side, np.flatnonzero(side[:n_base] != 0)


def load_teacher(paths, names_base):
    """Per-shard teacher files (names, d (n, N_ARM, 8, 2) f16/f32 = P0's rear-axle x, y with the arm's desire minus without) -> (n_base,
    N_ARM, 8, 2) float32 in Store order (asserted name for name)."""
    out = []
    o = 0
    for p in paths:
        z = np.load(p)
        n = len(z["names"])
        assert np.array_equal(z["names"].astype(str), np.asarray(names_base[o:o + n]).astype(str)), f"{p}: teacher rows differ from the Store's"
        out.append(z["d"].astype(np.float32))
        o += n
    assert o == len(names_base), (o, len(names_base))
    return np.concatenate(out)


def d2_loss(xy_d, xy_0, t):
    """Per-row D2 loss (n,): mean over the 8 poses of || (plan(x, d) - sg[plan(x, 0)]) - (plan_P0(x, d) - plan_P0(x, 0)) || (m).
    xy_d, xy_0 (n, 8, 2) rear-axle x, y (left +); t (n, 8, 2) the teacher's increment."""
    e = (xy_d - xy_0.detach()) - t
    return ((e * e).sum(-1) + 1e-6).sqrt().mean(1)
