#!/usr/bin/env python3
"""ROUTE v2 arm c (plans/2026-10-15-route-v2-prereg.md Amendment 7): one pool = RSG-L18's on-policy S rows (on1_L18RB0, prefix r1) + the
junction counterfactual rows of on1_CF (classes j0 / j1, prefix cf; lane-change rows excluded), so rm_train's single --on pool carries both
(RB-F's recipe for the CF rows: route hinge, dead zone 0.5 m, lateral only, + road-grid hinge, no imitation, no agent hinge).

  tab.npz       the keys both pools have, names renamed on:<prefix>:<name minus 'on:'>, + `pool` ('s' / 'cf') for --on-weight
  agent / road  concatenated (CF rows: agent ok False, the token's own road raster); drivable.npz: the S rows' only (CF rows absent = ok False)
  route.npz     S rows ok False (RSG trained them without a route hinge), CF rows = on1_CF's labels (dz 0.5)
  @warp         front.npy and teacher.npz concatenated
  feat          runs/route_mem/feat/on1_<out>.npz and .o.npz (old and ordered memory) concatenated, names renamed alike
Box, envs/op-train (numpy only).   rf_v2cpool.py [--out L18RB0J]
"""
import argparse
import os
from pathlib import Path

import numpy as np

D = Path(os.environ["DATA_DIR"])
CR, FE = D / "runs/op_parity/cache", D / "runs/route_mem/feat"


def main(a):
    S, C = CR / "on1_L18RB0", CR / "on1_CF"
    out, warp = CR / f"on1_{a.out}", CR / f"on1_{a.out}@warp"
    out.mkdir(exist_ok=True), warp.mkdir(exist_ok=True)
    ts, tc = dict(np.load(S / "tab.npz")), dict(np.load(C / "tab.npz"))
    jc = np.flatnonzero(np.isin(np.array([n.split(":")[3] for n in tc["names"].tolist()]), ["j0", "j1"]))
    sel = [(S, "r1", np.arange(len(ts["names"]))), (C, "cf", jc)]
    ren = lambda v, pre: np.array([f"on:{pre}:{t[3:] if t.startswith('on:') else t}" for t in v.tolist()])  # noqa: E731

    def cat(fname, key, extra=None):
        Z = [dict(np.load(d / fname)) for d, _, _ in sel]
        keys = set.intersection(*[set(z) for z in Z])
        M = {}
        for k in keys:
            if k == key:
                M[k] = np.concatenate([ren(z[k][r], p) for z, (_, p, r) in zip(Z, sel)])
            elif Z[0][k].ndim and len(Z[0][k]) == len(Z[0][key]):
                M[k] = np.concatenate([z[k][r] for z, (_, _, r) in zip(Z, sel)])
            else:
                assert all(np.array_equal(z[k], Z[0][k]) for z in Z), f"{fname}:{k} differs between pools"
                M[k] = Z[0][k]
        if extra:
            M.update(extra)
        np.savez(out / fname, **M)
        print(fname, len(M[key]), "rows", sorted(M))
        return M

    n_s, n_c = len(ts["names"]), len(jc)
    T = cat("tab.npz", "names", dict(pool=np.r_[np.full(n_s, "s"), np.full(n_c, "cf")]))
    cat("agent.npz", "tokens")
    cat("road.npz", "tokens")
    dz = dict(np.load(S / "drivable.npz"))
    dz["tokens"] = ren(dz["tokens"], "r1")
    np.savez(out / "drivable.npz", **dz)
    rc = dict(np.load(C / "route.npz"))
    assert np.array_equal(rc["tokens"], tc["names"])
    rs_poly = np.full((n_s,) + rc["poly"].shape[1:], np.nan, np.float32)
    np.savez(out / "route.npz", tokens=T["names"], ok=np.r_[np.zeros(n_s, bool), rc["ok"][jc]], poly=np.r_[rs_poly, rc["poly"][jc]],
             seglen=np.r_[np.full((n_s,) + rc["seglen"].shape[1:], np.nan, np.float32), rc["seglen"][jc]],
             dz=np.r_[np.ones(n_s, np.float32), rc["dz"][jc]], cov20=np.r_[np.zeros(n_s, bool), rc["cov20"][jc]])
    F = [(np.load(CR / f"{d.name}@warp/front.npy", mmap_mode="r"), r) for d, _, r in sel]
    o = np.lib.format.open_memmap(warp / "front.npy", "w+", F[0][0].dtype, (n_s + n_c,) + F[0][0].shape[1:])
    i = 0
    for x, r in F:
        for j in range(0, len(r), 4096):
            rr = r[j:j + 4096]
            o[i + j:i + j + len(rr)] = x[rr]
        i += len(r)
    o.flush()
    TT = [dict(np.load(CR / f"{d.name}@warp/teacher.npz")) for d, _, _ in sel]
    assert all(np.array_equal(t["di"], TT[0]["di"]) and np.array_equal(t["pi"], TT[0]["pi"]) for t in TT)
    np.savez(warp / "teacher.npz", out=np.concatenate([t["out"][r] for t, (_, _, r) in zip(TT, sel)]),
             plan=np.concatenate([t["plan"][r] for t, (_, _, r) in zip(TT, sel)]), di=TT[0]["di"], pi=TT[0]["pi"])
    for sfx in ("", ".o"):
        Z = [np.load(FE / f"{d.name}{sfx}.npz") for d, _, _ in sel]
        for z, (d, _, _) in zip(Z, sel):
            assert np.array_equal(z["names"], np.load(d / "tab.npz")["names"]), (d, sfx)
        names = np.concatenate([ren(z["names"][r], p) for z, (_, p, r) in zip(Z, sel)])
        assert np.array_equal(names, T["names"])
        keys = ("f", "ok", "trav") if sfx else ("f", "ok")
        np.savez(FE / f"on1_{a.out}{sfx}.npz", names=names, **{k: np.concatenate([z[k][r] for z, (_, _, r) in zip(Z, sel)]) for k in keys})
    w = 0.48 * n_s / (0.52 * n_c)
    print(f"on1_{a.out}: {n_s} S rows + {n_c} junction CF rows; front {o.shape}; --on-weight pool=cf:{w:.4f} gives 12 / 25 CF draws")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="L18RB0J")
    main(ap.parse_args())
