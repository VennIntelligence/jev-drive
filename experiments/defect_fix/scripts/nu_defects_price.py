#!/usr/bin/env python3
"""NU-DEFECTS pricing (Mac, tmp/pcs_local/.venv): frozen board-mix / g15 PCS of setting B2-s1's stop-approach misses to 1
(results/nu_defects/stop_windows.csv.gz; needs tmp/native_gap/set.csv.gz as easy_items.py subject). Run from the repo root."""
import sys, numpy as np, pandas as pd
from pathlib import Path
REPO = Path.cwd(); sys.path[:0] = [str(REPO / "experiments/alpasim_proxy/scripts"), str(REPO / "experiments/defect_fix/scripts"), str(REPO)]
import slow_rescore as SR, pcs_local as PL, easy_items as EI
res = REPO / "experiments/alpasim_proxy/results"
sc, ty, w = PL.load(res / "pcs_local")
items = list(pd.read_csv(res / "pcs_local/items_boardmix.csv.gz", index_col=0).index)
S = pd.read_csv(REPO / "tmp/native_gap/set.csv.gz").drop_duplicates("id").set_index("id")
R = pd.read_csv("experiments/defect_fix/results/stand_b/arena_windows_full_H_s1.csv.gz")
y = R[R.tag == "STANDFMXH-L18-s1"].drop_duplicates("id").set_index("id").score.reindex(items)
PR = {k: EI.Pricer(SR.Frozen(p, sc), S.log.reindex(items).to_numpy()) for k, p in SR.FITS.items()}
p = pd.read_csv("experiments/defect_fix/results/nu_defects/stop_windows.csv.gz"); p = p[(p.tag == "STANDFMXH-L18-s1")]
sets = {"stop-approach zeros (human brakes to rest, no lead)": set(p.id[p.score == 0]) & set(items),
        "stop-approach non-1": set(p.id[p.score < 1]) & set(items)}
y0 = y.to_numpy(float)
for nm, ids in sets.items():
    m = np.isin(items, list(ids)); y1 = y0.copy(); y1[m] = 1
    print(nm, int(m.sum()), "points %.1f" % (1 - y0[m]).sum(), {k: tuple(round(v, 1) for v in pr.diff(y0, y1)) for k, pr in PR.items()})
