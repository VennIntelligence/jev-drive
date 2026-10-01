"""Preliminary readout 3 (perturbation collapse, shift/swap vs orig) plus whatever readout 1 (ghost) the DONE cells
give, on the G-lane cells finished as of 2026-09-28 (todos/2026-09-26-night-queue-4.md, G section). Uses
jevdrive.nq4_g.collect / g_tables unchanged -- only the input rows are filtered to the DONE (cand, variant, seed)
cells below; still-running cells (blue.swap.0, all simlingo shift/swap) are excluded.
Output: ~/data/runs/nq4/g/prelim-shift-swap-20260928/ (the lane's own results/g/ is untouched)."""
from pathlib import Path
from jevdrive import nq4_g as NG

ALLOWED = {
    ("tfv6", "shift", 0), ("bridgedrive", "shift", 0), ("blue", "shift", 0),
    ("tfv6", "swap", 0), ("bridgedrive", "swap", 0),
    ("tfv6", "ghost", 0), ("tfv6", "ghost", 1),
    ("bridgedrive", "ghost", 0), ("bridgedrive", "ghost", 1),
    ("blue", "ghost", 0), ("blue", "ghost", 1),
    ("simlingo", "ghost", 0),
    ("tfv6", "orig", 0), ("tfv6", "orig", 1),
    ("bridgedrive", "orig", 0), ("bridgedrive", "orig", 1),
    ("blue", "orig", 0), ("blue", "orig", 1),
    ("simlingo", "orig", 0),
}

out = Path.home() / "data" / "runs" / "nq4" / "g" / "prelim-shift-swap-20260928"
out.mkdir(parents=True, exist_ok=True)

df = NG.collect()
df.to_csv(out / "per_run_all_cells.csv", index=False)          # everything collect() sees, for audit
mask = df.apply(lambda r: (r.cand, r.variant, int(r.seed)) in ALLOWED, axis=1)
d = df[mask].copy()
d.to_csv(out / "per_run.csv", index=False)

NG.g_tables(d, out)                                             # registered readouts 1/2/3/5, unchanged code

# coverage of the DONE cells actually found in collect() (sanity check against the task's cell list)
rows = []
for (cand, variant, seed) in sorted(ALLOWED):
    g = d[(d.cand == cand) & (d.variant == variant) & (d.seed == seed) & d.get("window").isna()]
    ok = g[g.status == "ok"]
    rows.append({"cand": cand, "variant": variant, "seed": seed, "routes_ok": len(ok), "routes_total": len(g)})
import pandas as pd
cov = pd.DataFrame(rows)
cov.to_csv(out / "cells_included.csv", index=False)
print(cov.to_string(index=False))
print(open(out / "g.md").read())
