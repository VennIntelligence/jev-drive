"""Register the fixed token subset of the DriveVLA-W0 projection controls (prereg addendum, plans/2026-10-23-p1-proj-artifact-prereg.md):
3 000 navtest tokens stratified by the bench's logged-manoeuvre class (jevdrive.bench.tables.navtest_strata): every class gets
max(proportional share, min(class size, 200)), the proportional classes are scaled so the total is 3 000; tokens drawn per class
with numpy default_rng(0). Chosen before any control arm ran.

  .venv/bin/python experiments/wm_surplus/scripts/p1_w0ctl_split.py <navtest strata csv>   -> jevdrive/data/splits/defs/navsim/p1-w0-ctl.v1.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive.data import splits  # noqa: E402

N, FLOOR = 3000, 200
d = pd.read_csv(sys.argv[1]).sort_values("token")
size = d.maneuver.value_counts().sort_index()
floor = np.minimum(size, FLOOR)
prop = size * N / size.sum()
fixed = floor[floor >= prop]                       # classes held up by the floor (or taken whole)
rest = size.drop(fixed.index)
alloc = pd.concat([fixed, (rest * (N - fixed.sum()) / rest.sum()).round().astype(int)]).sort_index()
alloc.iloc[alloc.index.get_loc(alloc.idxmax())] += N - alloc.sum()
rng = np.random.default_rng(0)
pick = [t for c, n in alloc.items() for t in rng.choice(d.token[d.maneuver == c].to_numpy(), int(n), replace=False)]
s = splits.define("navsim", "p1-w0-ctl", pick, unit="token", origin="experiments/wm_surplus/scripts/p1_w0ctl_split.py",
                  used_by="wm_surplus P1 DriveVLA-W0 projection controls", status="eval subset",
                  notes="3000 navtest tokens, stratified by logged manoeuvre (floor 200 / class), rng 0; per class: "
                        + ", ".join(f"{c} {n}/{size[c]}" for c, n in alloc.items()))
print(s.id, dict(alloc))
