#!/usr/bin/env python3
"""SLOT-RULE identity (plans/2026-10-22-slot-rule-prereg.md): with SH30_SLOT unset, sh30_core.lattice / lattice_gpu of this tree give the
same bytes as those of sh30_core at <rev> (before the switch), on random keyframes and tracks, warm and cold (1..4 keys); and fwd / bwd
change only the synthesised slots. Usage: $TR slot_ident.py <rev>"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / "experiments/alpasim/lib")]
import sh30_core as NEW  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402

src = subprocess.run(["git", "-C", str(R), "show", f"{sys.argv[1]}:experiments/alpasim/lib/sh30_core.py"], capture_output=True, text=True, check=True).stdout
old = R / "experiments/alpasim/lib/_sh30_core_ident_old.py"           # beside the original: its repo-root path rule needs the same depth
old.write_text(src)
try:
    spec = importlib.util.spec_from_file_location("sh30_core_old", old)
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
finally:
    old.unlink()
assert NEW.SLOT == "" and not hasattr(OLD, "SLOT")
rng, dev, n = np.random.default_rng(0), torch.device("cuda"), 0
cam = np.array([1.5, 0.0, 1.5])
for it in range(24):
    m = 1 + it % 4
    e = 4 - m
    keys = NEW.stack_keys([rng.integers(0, 255, NEW.FRAME, dtype=np.uint8) for _ in range(m)])
    v = rng.uniform(0, 15)
    P = np.c_[v * I.T_KEY, rng.normal(0, 0.3, 4), rng.normal(0, 0.05, 4)]
    P[-1] = 0
    V = np.c_[np.full(4, v), np.zeros(4)]
    tr = I.track_navsim(P, V)
    a, va = NEW.lattice_gpu(keys, e, tr, cam, "backwarp", dev)
    b, vb = OLD.lattice_gpu(keys, e, tr, cam, "backwarp", dev)
    assert torch.equal(a, b) and np.array_equal(va, vb), it
    c, vc = NEW.lattice(keys, e, tr, cam, "backwarp")
    d, vd = OLD.lattice(keys, e, tr, cam, "backwarp")
    assert np.array_equal(c, d) and np.array_equal(vc, vd), it
    for q in ("fwd", "bwd"):
        f, _ = NEW.lattice_gpu(keys, e, tr, cam, "backwarp", dev, rule=q)
        g, _ = NEW.lattice(keys, e, tr, cam, "backwarp", rule=q)
        same = [bool(torch.equal(f[j], a[j])) for j in range(8)]
        keyslot = [bool(np.isclose(I.T_KEY, t).any()) or t < I.T_KEY[e] - 1e-9 for t in NEW.SLOT_T]
        assert all(s for s, k in zip(same, keyslot) if k), (it, q, same)       # keys and cold backwarp slots untouched
        n += sum(not s for s, k in zip(same, keyslot) if not k)
        cg = float((g.astype(int) - f.cpu().numpy().astype(int)).__abs__().mean())
        assert cg < 2.0, (it, q, cg)                                           # cpu and gpu fwd / bwd agree (lat1: a few levels)
print(f"identity ok: 24 cases, lattice and lattice_gpu bit-identical to {sys.argv[1]} with SH30_SLOT off; fwd / bwd changed {n} synthesised slots")
