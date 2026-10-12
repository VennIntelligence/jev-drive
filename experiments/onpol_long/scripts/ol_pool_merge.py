#!/usr/bin/env python3
"""ONPOL-LONG Amendment 9: one on-policy pool from several (DAgger-style aggregation across rounds), in the pool contract of
experiments/onpol_nuplan/plans/on_pool_format.md. Row names get a round prefix (on:<prefix>:<old name minus 'on:'>) because two rounds can
share <scene>:<seed>:<k>; the label files' tokens are renamed the same way. Box, envs/op-train (numpy).

  ol_pool_merge.py --out on1_L18YA --pools on1_L18Y:r1 on1_L18YS:r2
"""
import argparse
import os
from pathlib import Path

import numpy as np

CR = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/op_parity/cache"


def main(a):
    src = [(CR / p, pre) for p, pre in (x.split(":") for x in a.pools)]
    out, warp = CR / a.out, CR / f"{a.out}@warp"
    out.mkdir(exist_ok=True), warp.mkdir(exist_ok=True)
    ren = lambda v, pre: np.array([f"on:{pre}:{t[3:]}" for t in v.tolist()])  # noqa: E731
    for f, key in (("tab.npz", "names"), ("agent.npz", "tokens"), ("road.npz", "tokens"), ("drivable.npz", "tokens")):
        Z = [dict(np.load(d / f)) for d, _ in src]
        keys = set.intersection(*[set(z) for z in Z])
        M = {}
        for k in keys:
            v0 = Z[0][k]
            if k == key:
                M[k] = np.concatenate([ren(z[k], pre) for z, (_, pre) in zip(Z, src)])
            elif v0.ndim and len(v0) == len(Z[0][key]):
                M[k] = np.concatenate([z[k] for z in Z])
            else:
                assert all(np.array_equal(z[k], v0) for z in Z), f"{f}:{k} differs between pools"
                M[k] = v0
        np.savez(out / f, **M)
        print(f, len(M[key]), "rows")
    names = np.load(out / "tab.npz")["names"]
    assert len(set(names.tolist())) == len(names)
    F = [np.load(CR / f"{d.name}@warp/front.npy", mmap_mode="r") for d, _ in src]
    o = np.lib.format.open_memmap(warp / "front.npy", "w+", F[0].dtype, (sum(len(x) for x in F),) + F[0].shape[1:])
    i = 0
    for x in F:
        for j in range(0, len(x), 4096):
            o[i + j:i + min(j + 4096, len(x))] = x[j:j + 4096]
        i += len(x)
    o.flush()
    T = [dict(np.load(CR / f"{d.name}@warp/teacher.npz")) for d, _ in src]
    assert all(np.array_equal(t["di"], T[0]["di"]) and np.array_equal(t["pi"], T[0]["pi"]) for t in T)
    np.savez(warp / "teacher.npz", out=np.concatenate([t["out"] for t in T]), plan=np.concatenate([t["plan"] for t in T]), di=T[0]["di"], pi=T[0]["pi"])
    assert len(o) == len(names)
    print("front", o.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True), ap.add_argument("--pools", nargs="+", required=True)
    main(ap.parse_args())
