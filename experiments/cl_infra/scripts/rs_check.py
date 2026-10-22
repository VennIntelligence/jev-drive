#!/usr/bin/env python3
"""READ-SUMMARIES identity: the facts-cache summary (rs_bench.sh new) against the old path (rs_bench.sh old) and the lane's stored outputs.

  .venv/bin/python rs_check.py --new DIR --old DIR --lane DIR --lane-md FILE --out JSON
Extract tables (dec / win / steps / set_census parquet, hum.pkl) must be equal frame by frame (same rows, order, dtypes, values); arena
pairs / held / latsplit / p1.md byte-identical; the score-only `facts table` numbers equal nd.py quick's (same pairs, means, CIs, zeros).
Exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def frames(a, b):
    out = {}
    for n in ("dec", "win", "steps", "set_census"):
        x, y = pd.read_parquet(a / f"{n}.parquet"), pd.read_parquet(b / f"{n}.parquet")
        if n == "dec":
            x["poses"], y["poses"] = x.poses.map(lambda v: list(np.asarray(v, float))), y.poses.map(lambda v: list(np.asarray(v, float)))
        eq = x.shape == y.shape and list(x.columns) == list(y.columns) and x.reset_index(drop=True).equals(y.reset_index(drop=True))
        r = dict(rows=[len(x), len(y)], equal=bool(eq))
        if not eq and x.shape == y.shape:
            r["columns_differ"] = [c for c in x.columns if c in y and not x[c].reset_index(drop=True).equals(y[c].reset_index(drop=True))][:10]
        out[n] = r
    ha, hb = pickle.load(open(a / "hum.pkl", "rb")), pickle.load(open(b / "hum.pkl", "rb"))
    same = list(ha) == list(hb) and all(all(np.array_equal(ha[w][k], hb[w][k]) for k in ha[w]) for w in ha)
    out["hum"] = dict(windows=[len(ha), len(hb)], equal=bool(same))
    return out


def files(a, b, names):
    import gzip
    rd = lambda f: gzip.decompress(f.read_bytes()) if f.suffix == ".gz" else f.read_bytes()  # noqa: E731
    return {n: dict(equal=rd(a / n) == rd(b / n)) if (a / n).exists() and (b / n).exists() else dict(equal=False, missing=True) for n in names}


def table_vs_quick(new):
    q = json.load(open(new / "p1_quick.json"))
    out = {}
    for bench, rd, zk, zq in (("arena", "arena", "zeros", "zeros|arena held"), ("held", "held 460", "zeros", "zeros|held 460")):
        t = json.load(open(new / f"table_{bench}.json"))
        pairs = {k: v for k, v in t.items() if k != zk}
        qp = {k.rsplit("|", 1)[0]: v for k, v in q.items() if k.endswith(f"|{rd}") and not k.startswith("zeros|")}
        out[bench] = dict(pairs=len(pairs), equal=pairs == qp and t[zk] == q[zq])
    return out


def main():
    p = argparse.ArgumentParser()
    for k in ("new", "old", "lane", "lane-md", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    N, O, L = Path(a.new), Path(a.old), Path(a.lane)
    J = sorted(x.name for x in N.glob("arena_*.json"))
    r = dict(ext_vs_old=frames(N / "ext", O / "ext"), ext_vs_lane=frames(N / "ext", L / "ext"),
             outputs_vs_old=files(N, O, [*J, "held.json", "latsplit.csv", "p1_arena_windows.csv.gz"]),
             p1md_vs_old_lines=dict(equal=sorted((N / "p1.md").read_text().splitlines()) == sorted((O / "p1.md").read_text().splitlines())),
             outputs_vs_lane=files(N, L, [*J, "held.json", "latsplit.csv"]),
             p1md_vs_lane=dict(equal=(N / "p1.md").read_text() == Path(a.lane_md).read_text()),
             table_vs_quick=table_vs_quick(N))
    flat = [v["equal"] for g in r.values() for v in (g.values() if "equal" not in g else [g])]
    r["identical"] = all(flat)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["identical"] else 1)


if __name__ == "__main__":
    main()
