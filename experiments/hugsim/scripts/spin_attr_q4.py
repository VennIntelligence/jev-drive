"""Q4: navhard tokens that legitimately need an early heading change, and how the history-rule arms score on them (Mac).
    python spin_attr_q4.py <navhard_ref.csv> <navhard_arms.csv> <out_dir>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ref = pd.read_csv(sys.argv[1])
arms = pd.read_csv(sys.argv[2])
out = Path(sys.argv[3])
d = arms.merge(ref, on="token", how="inner")
d = d[d.stage.isin([1, 2])]
print("tokens", len(d), d.stage.value_counts().to_dict())
A = ["base", "rot0", "straight", "sel06", "selrot0", "selstraight"]
rows = []
rng = np.random.default_rng(0)
for src in ("pdm",):
    for T in (1, 2):
        h = d[f"{src}_h{T}"].abs()
        for th in (5, 10):
            for stg in (1, 2):
                m = (d.stage == stg) & (h >= th)
                n = int(m.sum())
                row = dict(ref=src, t=T, thr=th, stage=stg, n=n, frac=n / int((d.stage == stg).sum()))
                for a in A:
                    row[a] = float(d.loc[m, a].mean()) if n else np.nan
                    if a != "base" and n:
                        diff = (d.loc[m, a] - d.loc[m, "base"]).values
                        b = [diff[rng.integers(0, n, n)].mean() for _ in range(1000)]
                        row[a + "_d"] = float(diff.mean())
                        row[a + "_lo"], row[a + "_hi"] = np.percentile(b, [2.5, 97.5])
                rows.append(row)
R = pd.DataFrame(rows)
out.mkdir(parents=True, exist_ok=True)
R.to_csv(out / "q4_early_turn.csv", index=False)
pd.set_option("display.width", 250)
print(R[["ref", "t", "thr", "stage", "n", "frac", "base"] + [a + "_d" for a in A[1:]]].round(3).to_string())
# complement for the main definition (pdm, 2 s, 5 deg)
for stg in (1, 2):
    m = (d.stage == stg) & (d.pdm_h2.abs() >= 5)
    print("stage", stg, "complement mean base", d.loc[~(m) & (d.stage == stg), "base"].mean(), {a: round(float((d.loc[~m & (d.stage == stg), a] - d.loc[~m & (d.stage == stg), "base"]).mean()), 3) for a in A[1:]})
print(d[["pdm_h1", "pdm_h2"]].abs().describe().round(2))
