#!/usr/bin/env python3
"""VAL160 extension report: D1-s0, D1-s1, D1 2-seed mean, R8K-s2, P1L, INTP75-NR on the 152 val160 scenes (ACC + GAP, Harmonizer off).
  v160x.py --dir <val160b> --old-dir <val160> --out <dir>/results
New scorer = scorer_v2 rollouts.csv (old-dir sv2 for D1-s0 / R8K / P1L, dir sv2 for D1-s1 / INTP75); old scorer = stored facts."""
import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive.stats import paired

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
F_ = DATA / "runs/pai_fix/facts"
ARMS = {"D1s0": "NT3D1-F-s0", "D1s1": "NT3D1-F-s1", "R8Ks2": "R8K-L18-s2", "P1L": "PCL1L-F-s0", "INTP75": "INTP75-NR"}
SCENES = Path(__file__).resolve().parents[1] / "results/val160/chunks"
ZC = ["overshoot", "route-turn", "offroad", "AF-front", "side", "zero-other"]


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fa = {k: pd.read_csv(F_ / f"{t}-v160ag.csv").set_index("scene") for k, t in ARMS.items()}
    nr = pd.concat([pd.read_csv(Path(d) / "sv2/new/rollouts.csv") for d in (a.old_dir, a.dir)])
    nw = {k: nr[nr.read == f"{t}-v160"].drop_duplicates("scene", keep="last").set_index("scene").score for k, t in ARMS.items()}
    sub = {s: c.stem[0] for c in sorted(SCENES.glob("*.txt")) for s in c.read_text().split()}
    common = set.intersection(*[set(f.index) & set(nw[k].dropna().index) for k, f in fa.items()])
    for f in fa.values():
        common -= set(f.index[f.rpc.fillna(False).astype(bool)])
    sc = sorted(common)
    rpc = {k: int(f.rpc.sum()) for k, f in fa.items()}
    S = {"old": {k: fa[k].score.reindex(sc).to_numpy(float) for k in ARMS}, "new": {k: nw[k].reindex(sc).to_numpy(float) for k in ARMS}}
    for sv in S:
        S[sv]["D1m"] = (S[sv]["D1s0"] + S[sv]["D1s1"]) / 2
    names = ["D1s0", "D1s1", "D1m", "R8Ks2", "P1L", "INTP75"]
    grp = np.array([sub.get(s, "?") for s in sc])
    L = ["# VAL160 extension: D1 seeds, R8K-s2, P1L, INTP75-NR on PAI val160 (ACC + GAP, Harmonizer off)\n",
         f"Common scenes {len(sc)} (v = {int((grp == 'v').sum())}, w = {int((grp == 'w').sum())}); RPC-error rollouts excluded: {rpc}. D1m = per-scene mean of D1-s0 and D1-s1.\n"]
    rows = []
    for name, m in (("all", np.ones(len(sc), bool)), ("v (selected)", grp == "v"), ("w (unselected)", grp == "w")):
        for sv in ("new", "old"):
            r = dict(subset=name, scorer=sv, n=int(m.sum()))
            for k in names:
                r[k] = float(S[sv][k][m].mean())
                r[f"{k}_zeros"] = int((S[sv][k][m] == 0).sum())
            for x, y in (("D1m", "R8Ks2"), ("INTP75", "R8Ks2"), ("D1m", "INTP75"), ("D1m", "P1L"), ("D1s0", "D1s1")):
                p = paired(S[sv][x][m], S[sv][y][m])
                r[f"{x}-{y}"] = f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
            for k in ("D1s0", "D1s1", "D1m", "P1L", "INTP75"):
                r[f"flip_{k}"] = int(((S[sv]["R8Ks2"][m] >= 0.98) & (S[sv][k][m] < 0.98)).sum())
            rows.append(r)
    T = pd.DataFrame(rows)
    T.round(4).to_csv(out / "val160x.csv", index=False)
    L.append("## Means, zeros, paired differences (scene bootstrap 95 % CI), flips = R8K-s2 >= 0.98 and model < 0.98\n\n" + T.round(3).T.to_markdown() + "\n")
    zc = {k: Counter(fa[k].fclass.reindex(sc)[fa[k].score.reindex(sc) == 0]) for k in ARMS}
    Z = pd.DataFrame({k: [zc[k][c] for c in ZC] + [sum(zc[k].values())] for k in ARMS}, index=ZC + ["zeros"])
    L.append("## Zero classes (old scorer, common scenes)\n\n" + Z.to_markdown() + "\n")
    pd.DataFrame({"scene": sc, "subset": grp, **{f"{k}_{sv}": S[sv][k] for sv in S for k in names}}).round(4).to_csv(out / "per_scene_x.csv", index=False)
    (out / "val160x.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--old-dir", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
