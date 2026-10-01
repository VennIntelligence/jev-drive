"""Final readout (perturbation collapse readout 3, plus ghost readout 1/2) on the G-lane cells now that the lane
finished (DONE 2026-09-29 08:31 CST, todos/2026-09-26-night-queue-4.md, G section). Uses jevdrive.nq4_g.collect /
g_tables unchanged -- only the input rows are filtered to the registered (cand, variant, seed) cells below.
PDM-Lite keeps only ghost seeds 0-2 (registered scope narrowing, 2026-09-28) plus its orig seed 0, reused from
night-queue-3 (jevdrive.nq4_g.reuse_dirs, cl1_expert); its shift/swap/orig seeds 1-2 are DROPPED and excluded here
same as they were dropped in the lane. Supersedes the two preliminaries (research/results/nq4/g/prelim_s0/,
research/results/nq4/g/prelim_shift_swap/), which covered partial seed counts.
Output: ~/data/runs/nq4/g/final-20260929/ (the lane's own results/g/ is untouched)."""
from pathlib import Path
from jevdrive import nq4_g as NG

CANDS = ("tfv6", "bridgedrive", "blue", "simlingo")
ALLOWED = {(c, "shift", 0) for c in CANDS} | {(c, "swap", 0) for c in CANDS} \
    | {(c, v, s) for c in CANDS for v in ("ghost", "orig") for s in (0, 1, 2)} \
    | {("pdm", "ghost", s) for s in (0, 1, 2)} | {("pdm", "orig", 0)}

out = Path.home() / "data" / "runs" / "nq4" / "g" / "final-20260929"
out.mkdir(parents=True, exist_ok=True)

df = NG.collect()
df.to_csv(out / "per_run_all_cells.csv", index=False)          # everything collect() sees, for audit
mask = df.apply(lambda r: (r.cand, r.variant, int(r.seed)) in ALLOWED, axis=1)
d = df[mask].copy()
d.to_csv(out / "per_run.csv", index=False)

NG.g_tables(d, out)                                             # registered readouts 1/2/3/5, unchanged code

# coverage of the registered cells actually found in collect() (sanity check against the lane's own DONE cell list)
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
