#!/usr/bin/env python3
"""G3 (d) reader: navtest EPDMS per turn bucket and the > 45 deg turn-oracle rates, new checkpoint vs P2H10-F of the same seed.

  python experiments/body1/scripts/bd4_g3d.py --replay body1a5 --new P2H10B-F --base P2H10-F --out experiments/body1/results/loss/g3_a5_full_d
Needs the replay parquet of `turn_oracle.py replay --name <replay> --models <the four specs>`. Env: envs/navsim2 or op-train python.
Cluster bootstrap by log via jevdrive.stats.paired (seed-mean per token, difference new - base).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_R / "experiments/op_parity/scripts"), str(_R / "experiments/op_probe/scripts")]
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import turn_oracle as T  # noqa: E402
from jevdrive import stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--replay", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--out", required=True)
a = ap.parse_args()
D = T.Data([a.replay])
spec = lambda arm, s: f"{arm}-s{s}"  # noqa: E731
new = sum(D.one(spec(a.new, s))[0] for s in a.seeds) / len(a.seeds)
base = sum(D.one(spec(a.base, s))[0] for s in a.seeds) / len(a.seeds)
per_seed = {s: (D.one(spec(a.new, s))[0], D.one(spec(a.base, s))[0]) for s in a.seeds}
logs = np.asarray(D.log)
rows = []
for name, m in D.sets.items():
    for col in ("EPDMS", "DAC fail %", "inside-cut %", "cannot-make-turn %", "other DAC fail %", "EP"):
        x, y = new[col].to_numpy()[m], base[col].to_numpy()[m]
        r = stats.paired(x, y, groups=logs[m])
        rows.append(dict(stratum=name, metric=col, n=int(m.sum()), new=float(x.mean()), base=float(y.mean()), diff=float(r["mean"]), lo=float(r["lo"]), hi=float(r["hi"])))
        for s, (n_, b_) in per_seed.items():
            rows[-1][f"diff_s{s}"] = float(n_[col].to_numpy()[m].mean() - b_[col].to_numpy()[m].mean())
df = pd.DataFrame(rows)
_pl.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out + ".csv", index=False)
json.dump(D.check, open(a.out + "_check.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(df[df.stratum.isin(["all", "T45 (> 45 deg)"])].round(3).to_string())
print(json.dumps(D.check))
