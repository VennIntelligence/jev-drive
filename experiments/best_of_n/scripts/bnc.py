#!/usr/bin/env python3
"""BEST-OF-N control-continuation replication (plans/2026-10-19-bn-prereg.md, supplement 1). Box, repo root, op-train python.

  report   arena held native + held 460 @18 for BNC2K-L18-s0, BNC4K-L18-s1 (and BNC2K-L18-s1 as the d379 reference), paired against their
           parents -> results/continue.{json,md}; the continuation rule is evaluated on BNC2K-L18-s0 vs JTN-L18-s0.
"""
from __future__ import annotations

import json
import os
import sys

import bn

PAIRS = (("BNC2K-L18-s0", "JTN-L18-s0"), ("BNC4K-L18-s1", "JTN-L18-s1"), ("BNC2K-L18-s1", "JTN-L18-s1"), ("BNC4K-L18-s1", "BNC2K-L18-s1"))
RULE = ("BNC2K-L18-s0", "JTN-L18-s0")


def main():
    import pandas as pd
    from jevdrive import stats
    sys.path[:0] = [str(bn.REPO / "experiments/defect_fix/scripts")]
    import df
    J = bn.DATA / "runs/defect_fix/jt/arena"
    tmp = bn.W / "cont_reads"
    tmp.mkdir(parents=True, exist_ok=True)
    tags = sorted({t for p in PAIRS for t in p})
    for t in tags:
        (tmp / t).unlink(missing_ok=True)
        (tmp / t).symlink_to(J / f"{t}@off")
    os.system(f"{sys.executable} {bn.REPO}/experiments/alpasim_proxy/scripts/arena.py collect --reads {tmp} --out {bn.W}/cont_rows.csv.gz > /dev/null")
    Sx = pd.read_csv(bn.DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").set_index("id")
    Hs = Sx[Sx.split == "held"]
    R = pd.read_csv(bn.W / "cont_rows.csv.gz")
    X = {t: R[(R.tag == t) & R.id.isin(Hs.index)].drop_duplicates("id").set_index("id") for t in tags}
    assert all(len(X[t]) >= 0.99 * len(Hs) for t in tags), {t: len(X[t]) for t in tags}

    def pair(t, ref, strat=None):
        A_, B = X[t], X[ref]
        bad = set(A_.index[A_.type == "route-fail"]) | set(B.index[B.type == "route-fail"])
        c = A_.index.intersection(B.index).difference(bad)
        if strat:
            c = c[Hs.loc[c, "stratum"].values == strat]
        x, y, g = A_.loc[c, "score"].values, B.loc[c, "score"].values, Hs.loc[c, "log"].values
        d = stats.paired(x, y, groups=g) if len(c) >= 20 else None
        return dict(n=len(c), mean=float(x.mean()), ref=float(y.mean()), d=d, zeros=int((x == 0).sum()), zeros_ref=int((y == 0).sum()),
                    new_zeros=int(((y > 0) & (x == 0)).sum()), fixed_zeros=int(((y == 0) & (x > 0)).sum()))
    ar = {f"{t} vs {r}": {s or "all": pair(t, r, s) for s in (None,) + bn.STRATA} for t, r in PAIRS}
    refs = sorted({r for _, r in PAIRS})
    h = {t: df.cl_read(t, [r for r in refs if r != t]) for t in tags}
    H = {t: {k: v for k, v in (h[t]["held"] or {}).items() if k != "pair"} for t in h}
    assert all(H[t].get("scenes") == 460 for t in H), {t: H[t].get("scenes") for t in H}
    p0, (arm, par) = ar[f"{RULE[0]} vs {RULE[1]}"]["all"], RULE
    g1 = bool(p0["d"]["mean"] >= 0.005 and p0["d"]["lo"] > 0)
    g2 = bool(H[arm]["zeros"] <= H[par]["zeros"] + 2)
    v = dict(rule="BNC2K-L18-s0 vs JTN-L18-s0 paired >= +0.005 and CI lo > 0, and held 460 zeros <= parent + 2", arena_paired=p0["d"], arena_pass=g1,
             held_zeros=H[arm]["zeros"], held_zeros_parent=H[par]["zeros"], held_pass=g2, helps=g1 and g2)
    bn.RES.mkdir(parents=True, exist_ok=True)
    (bn.RES / "continue.json").write_text(json.dumps(dict(verdict=v, arena=ar, held=H, held_pairs={t: (h[t]["held"] or {}).get("pair") for t in h}),
                                                    indent=1, default=str) + "\n")
    f = lambda d: "-" if d is None else stats.fmt(d, "+.4f")  # noqa: E731
    L = [f"# BEST-OF-N control continuation, replication: {'HELPS' if v['helps'] else 'DOES NOT HELP'} (BNC2K-L18-s0)", "",
         "Arena held split native (no hold, `jt.sh aread <tag> off`), paired on common windows without a route-fail, 95 % CI clustered by log (jevdrive.stats). "
         "Held 460 @18: `rm_chain.sh read`.", "",
         f"- rule: {v['rule']}", f"- arena {arm} - {par}: **{f(p0['d'])}** ({'pass' if g1 else 'fail'}); held zeros {H[arm]['zeros']} vs {H[par]['zeros']} ({'pass' if g2 else 'fail'})", "",
         "| pair | stratum | windows | tag mean | ref mean | paired [CI] | zeros tag / ref | new / fixed zeros |", "|:--|:--|--:|--:|--:|:--|--:|--:|"]
    for k, by in ar.items():
        for s, p in by.items():
            L.append(f"| {k} | {s} | {p['n']} | {p['mean']:.4f} | {p['ref']:.4f} | {f(p['d'])} | {p['zeros']} / {p['zeros_ref']} | {p['new_zeros']} / {p['fixed_zeros']} |")
    L += ["", "Held 460 @18:", "", "| tag | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired vs parent |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    par_of = {a: b for a, b in PAIRS}
    for t in tags:
        x, pr = H[t], (h[t]["held"] or {}).get("pair", {})
        r = par_of.get(t)
        pp = f"{pr[r]['mean']:+.4f} [{pr[r]['lo']:+.4f}, {pr[r]['hi']:+.4f}]" if r and pr.get(r) else ""
        L.append(f"| {t} | {x['zeros']} | {x['mean']:.4f} | {x['at_fault']} | {x['offroad']} | {x['lat_zeros']} | {x['coll']['front']} / {x['coll']['lateral']} / "
                 f"{x['coll']['rear']} | {x['progress_rel']:.3f} | {x['creep']} / {x['no_launch']} | {pp} |")
    (bn.RES / "continue.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
