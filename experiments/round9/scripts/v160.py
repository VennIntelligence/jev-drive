#!/usr/bin/env python3
"""VAL160 report: D1 (NT3D1-F-s0), R8K-s2, P1L (PCL1L-F-s0) on the 152 PAI val160 scenes, ACC + GAP, Harmonizer off.
  v160.py report --dir $DATA_DIR/runs/round9/val160 --out <dir>/results
Old scorer = the stored scores (facts csv, identical to the stored results-summary); new scorer = scorer_v2 `score --variant new` rollouts.csv.
Common scenes = scored by all three reads (both scorers) with no RPC error in any. Subsets: v = the 72 PAI-FIX scenes (selected on NTV0-s0 failures plus a
guard), w = the other 80 (never selected). Paired CIs: jevdrive.stats.paired (scene bootstrap)."""
import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive.stats import paired

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
F_ = DATA / "runs/pai_fix/facts"
ARMS = {"D1": "NT3D1-F-s0", "R8Ks2": "R8K-L18-s2", "P1L": "PCL1L-F-s0"}
SCENES = Path(__file__).resolve().parents[1] / "results/val160/chunks"
ZC = ["overshoot", "route-turn", "offroad", "AF-front", "side", "zero-other"]
# test281 reference: D1 - R8Ks2 (old scorer), from the stored 281 facts
F281 = {"D1": "NT3D1-F-s0-281ag", "R8Ks2": "R8K-L18-s2-281ag", "P1L": "PCL1L-F-s0-281ag"}


def main(a):
    d, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fa = {k: pd.read_csv(F_ / f"{t}-v160ag.csv").set_index("scene") for k, t in ARMS.items()}
    nr = pd.read_csv(d / "sv2/new/rollouts.csv")
    nw = {k: nr[nr.read == f"{t}-v160"].drop_duplicates("scene", keep="last").set_index("scene").score for k, t in ARMS.items()}
    sub = {c.stem[0]: s for c in sorted(SCENES.glob("*.txt")) for s in c.read_text().split()}
    common = set.intersection(*[set(f.index) & set(nw[k].dropna().index) for k, f in fa.items()])
    for f in fa.values():
        common -= set(f.index[f.rpc.fillna(False).astype(bool)])
    sc = sorted(common)
    rpc = {k: int(f.rpc.sum()) for k, f in fa.items()}
    S = {"old": {k: fa[k].score.reindex(sc).to_numpy(float) for k in ARMS}, "new": {k: nw[k].reindex(sc).to_numpy(float) for k in ARMS}}
    grp = np.array([sub.get(s, "?") for s in sc])
    L = [f"# VAL160: D1 / R8K-s2 / P1L on PAI val160 (ACC + GAP, Harmonizer off)\n",
         f"Common scenes {len(sc)} (v = {int((grp == 'v').sum())} PAI-FIX scenes, w = {int((grp == 'w').sum())} never selected); RPC-error rollouts excluded: {rpc}.\n"]
    rows = []
    for name, m in (("all", np.ones(len(sc), bool)), ("v (selected)", grp == "v"), ("w (unselected)", grp == "w")):
        for sv in ("new", "old"):
            r = dict(subset=name, scorer=sv, n=int(m.sum()))
            for k in ARMS:
                r[k] = float(S[sv][k][m].mean())
                r[f"{k}_zeros"] = int((S[sv][k][m] == 0).sum())
            for x, y in (("D1", "R8Ks2"), ("D1", "P1L"), ("R8Ks2", "P1L")):
                p = paired(S[sv][x][m], S[sv][y][m])
                r[f"{x}-{y}"] = f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
            rows.append(r)
    T = pd.DataFrame(rows)
    T.round(4).to_csv(out / "val160.csv", index=False)
    L.append("## Means, zeros, paired differences (scene bootstrap 95 % CI)\n\n" + T.round(3).to_markdown(index=False) + "\n")
    # zero classes (old-scorer fclass of the facts; zeros of the old scorer), all common scenes
    zc = {k: Counter(fa[k].fclass.reindex(sc)[fa[k].score.reindex(sc) == 0]) for k in ARMS}
    Z = pd.DataFrame({k: [zc[k][c] for c in ZC] + [sum(zc[k].values())] for k in ARMS}, index=ZC + ["zeros"])
    L.append("## Zero classes (old scorer, common scenes)\n\n" + Z.to_markdown() + "\n")
    # ordering vs test281 (old scorer, common 281 scenes of the three)
    f2 = {k: pd.read_csv(F_ / f"{t}.csv").set_index("scene") for k, t in F281.items()}
    c2 = set.intersection(*[set(f.index[~f.rpc.fillna(False).astype(bool)]) for f in f2.values()])
    c2 = sorted(c2)
    q = {k: f2[k].score.reindex(c2).to_numpy(float) for k in ARMS}
    L.append(f"## test281 reference (old scorer, {len(c2)} common scenes)\n")
    for x, y in (("D1", "R8Ks2"), ("D1", "P1L"), ("R8Ks2", "P1L")):
        p = paired(q[x], q[y])
        L.append(f"- {x} - {y}: {p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]  (means {p['mean_a']:.3f} / {p['mean_b']:.3f})")
    L.append("")
    risk = nr[(nr.score > 0) & (~nr.evaluated.fillna(False).astype(bool)) & nr.ev_plain.notna()]
    L.append(f"New scorer: nonzero rollouts without an ASL that hit the truncation event (unverified, kept at stored value): {risk.groupby('read').size().to_dict()}\n")
    ch = {k: int((S['new'][k] != S['old'][k]).sum()) for k in ARMS}
    L.append(f"Scenes whose score differs between scorers: {ch}\n")
    pd.DataFrame({"scene": sc, "subset": grp, **{f"{k}_old": S["old"][k] for k in ARMS}, **{f"{k}_new": S["new"][k] for k in ARMS}}).round(4).to_csv(out / "per_scene.csv", index=False)
    (out / "val160.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["report"])
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
