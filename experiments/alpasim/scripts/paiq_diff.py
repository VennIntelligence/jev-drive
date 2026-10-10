"""Lane alpasim-pai-quick: paired difference of two pai_run.sh runs on their common scenes (bootstrap over scenes, 10000 draws, seed 0),
zero kinds including 'other' (a zero with none of the three failure flags), and the reference subjects' means on the same scenes.
  paiq_diff.py --a <label>=<run dir> --b <label>=<run dir> --ref <ref dir> [--out md]      (plain python3 + numpy)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pai_report as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True), ap.add_argument("--b", required=True), ap.add_argument("--ref", required=True), ap.add_argument("--out")
a = ap.parse_args()
(la, da), (lb, db) = (x.split("=", 1) for x in (a.a, a.b))
R = {la: P.rollouts(Path(da)), lb: P.rollouts(Path(db))}
sc = sorted(set(R[la]) & set(R[lb]))
ref = P.refs(Path(a.ref))
S = {k: np.array([v[x]["score"] for x in sc]) for k, v in R.items()}
d = S[lb] - S[la]
rng = np.random.default_rng(0)
bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
L = [f"Scenes scored by both arms: {len(sc)}.", "", "| Row | Mean | Zeros | at-fault collision | offroad | left corridor | other zero |", "|---|--:|--:|--:|--:|--:|--:|"]
for k, v in R.items():
    z = [x for x in sc if v[x]["score"] == 0]
    cnt = [sum(bool(v[x]["score_metrics"].get(f)) for x in z) for f in P.FAIL]
    other = sum(not any(v[x]["score_metrics"].get(f) for f in P.FAIL) for x in z)
    L.append(f"| {k} | {S[k].mean():.4f} | {len(z)} | " + " | ".join(map(str, cnt)) + f" | {other} |")
for k, v in ref.items():
    if all(x in v for x in sc):
        L.append(f"| {k} (reference) | {np.mean([np.mean([r['score'] for r in v[x]]) for x in sc]):.4f} | | | | | |")
L += ["", f"Paired difference {lb} - {la}: {d.mean():+.4f}, bootstrap 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] over scenes; "
      f"{lb} better on {(d > 0.01).sum()} scenes, worse on {(d < -0.01).sum()}, within 0.01 on {(abs(d) <= 0.01).sum()}; "
      f"zero in {la} only: {((S[la] == 0) & (S[lb] > 0)).sum()}, in {lb} only: {((S[lb] == 0) & (S[la] > 0)).sum()}, both: {((S[la] == 0) & (S[lb] == 0)).sum()}."]
print("\n".join(L))
if a.out:
    Path(a.out).write_text("\n".join(L) + "\n")
