"""NUACC readouts (nuacc.sh): pair = JTN+stop3+ACC+GAP vs RSG+stop3+ACC+GAP on the arena held split (strata, zero types, log-clustered CIs);
held = held 460 @18 of both accgap reads against the stored native RSG-L18-s0 read (and JTN vs RSG accgap)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def cmd_pair(a):
    import pandas as pd
    sys.path[:0] = [str(REPO), str(REPO / "experiments/alpasim_proxy/scripts")]
    import arena as A
    from jevdrive import stats
    import os
    ARENA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/alpasim_proxy/arena"
    Hs = pd.read_csv(ARENA / "set/set.csv.gz").set_index("id")
    Hs = Hs[Hs.split == "held"]
    X = pd.read_csv(a.jtn).drop_duplicates("id").set_index("id")
    B = pd.read_csv(a.rsg).drop_duplicates("id").set_index("id")
    X, B = X[X.index.isin(Hs.index)], B[B.index.isin(Hs.index)]
    bad = set(B.index[B.type == "route-fail"]) | set(X.index[X.type == "route-fail"])
    c = X.index.intersection(B.index).difference(bad)
    x, y, g = X.loc[c, "score"].values, B.loc[c, "score"].values, Hs.loc[c, "log"].values
    L = ["# JTN+stop3+ACC+GAP minus RSG+stop3+ACC+GAP, arena held split", "",
         f"{len(c)} common windows; JTN {x.mean():.4f}, RSG {y.mean():.4f}; paired {stats.fmt(stats.paired(x, y, groups=g), '+.4f')}; zeros {int((x == 0).sum())} vs {int((y == 0).sum())}, "
         f"zero-rate diff {stats.fmt(stats.paired((x == 0).astype(float), (y == 0).astype(float), groups=g), '+.4f')}", "",
         "| stratum | windows | RSG-accgap | JTN-accgap | paired diff [CI] | contribution | new / fixed zeros |", "|---|--:|--:|--:|--:|--:|--:|"]
    for s in A.STRATA:
        cs = c[Hs.loc[c, "stratum"].values == s]
        if not len(cs):
            continue
        xs, ys = X.loc[cs, "score"].values, B.loc[cs, "score"].values
        ds = stats.paired(xs, ys, groups=Hs.loc[cs, "log"].values) if len(cs) >= 20 else None
        L.append(f"| {s} | {len(cs)} | {ys.mean():.3f} / {(ys == 0).mean():.3f} | {xs.mean():.3f} / {(xs == 0).mean():.3f} | "
                 f"{'-' if ds is None else stats.fmt(ds, '+.3f')} | {(xs - ys).sum() / len(c):+.4f} | {int(((ys > 0) & (xs == 0)).sum())} / {int(((ys == 0) & (xs > 0)).sum())} |")
    L += ["", "| read | " + " | ".join(A.ZC) + " | " + " | ".join(A.PC) + " |", "|---|" + "--:|" * (len(A.ZC) + len(A.PC))]
    for nm, Q in (("RSG+stop3+ACC+GAP", B.loc[c]), ("JTN+stop3+ACC+GAP", X.loc[c])):
        zc = Q[Q.score == 0].type.map(A.zclass).value_counts()
        pc = Q[Q.score > 0].type.value_counts()
        L.append(f"| {nm} | " + " | ".join(str(int(zc.get(k, 0))) for k in A.ZC) + " | " + " | ".join(str(int(pc.get(k, 0))) for k in A.PC) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_held(a):
    sys.path[:0] = [str(HERE)]
    import df
    refs = ["RSG-L18-s0", "RSG-L18-s0-accgap"]
    for t in ("RSG-L18-s0-accgap", "JTN-L18-s0-accgap"):
        r = df.cl_read(t, [q for q in refs if q != t])
        print(json.dumps({k: v for k, v in r.items() if k not in ("held", "pub")} | {"held": r["held"]}, default=str))
        print("\n".join(df.fmt_cl(t, r, [q for q in refs if q != t])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pair")
    p.add_argument("--jtn", required=True), p.add_argument("--rsg", required=True), p.add_argument("--out", required=True)
    sp.add_parser("held")
    a = ap.parse_args()
    dict(pair=cmd_pair, held=cmd_held)[a.cmd](a)
