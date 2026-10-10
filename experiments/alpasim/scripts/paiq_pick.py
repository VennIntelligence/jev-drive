"""Lane alpasim-pai-quick: the 40 PAI scenes no earlier run of ours has touched. Rule (fixed before any score): curated-validation scenes
(they have the organisers' reference rows) whose usdz is already in the scene cache and that are in none of the used lists, sorted by
scene id, the first N. Output has the columns of pai_scenes.py (stage 1 for all). Plain python3, on the Tokyo box.
  paiq_pick.py --src <alpasim checkout> --ref <ref dir> --cache <all-usdzs> --used <file of scene ids> --n 40 --out <tsv>
"""
import argparse
from pathlib import Path

import pai_scenes as S

ap = argparse.ArgumentParser()
for k in ("src", "ref", "cache", "used", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--n", type=int, default=40)
a = ap.parse_args()
R, cat = S.ref_scores(Path(a.ref)), S.catalog(Path(a.src))
have = {p.stem for p in Path(a.cache).glob("*.usdz")}
used = set(Path(a.used).read_text().split())
mean = {s: sum(R[k][s] for k in R) / len(R) for s in cat if all(s in R[k] for k in R)}
pool = sorted(s for s in mean if cat[s]["uuid"] in have and s not in used)
print(f"curated val {len(cat)}, with reference {len(mean)}, in cache {sum(cat[s]['uuid'] in have for s in mean)}, unused {len(pool)}")
with open(a.out, "w") as f:
    f.write("scene_id\tuuid\tpath\trevision\tref_mean\talpamayo1\tvavam_nonlinear\tstage\n")
    for s in pool[:a.n]:
        c = cat[s]
        f.write(f"{s}\t{c['uuid']}\t{c['path']}\t{c.get('hf_revision') or ''}\t{mean[s]:.4f}\t{R['alpamayo1'][s]:.4f}\t{R['vavam-nonlinear'][s]:.4f}\t1\n")
