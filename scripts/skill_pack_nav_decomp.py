"""NAVSIM navtest per-token PDMS loss decomposition (gate-first: NC/DAC, then weighted EP/TTC/C) and best-of oracle across per-token CSVs.
Usage: python scripts/skill_pack_nav_decomp.py DIR  (DIR/csv/{gimm,hydra,clsref}.csv = devkit per-token outputs)."""
import pandas as pd, numpy as np, sys
S = sys.argv[1]
cols = dict(no_at_fault_collisions="NC", drivable_area_compliance="DAC", ego_progress="EP", time_to_collision_within_bound="TTC", comfort="C", score="PDMS")
R = {}
for n in ("gimm", "hydra", "clsref"):
    d = pd.read_csv(f"{S}/csv/{n}.csv"); d = d[d.token.str.len() == 16].rename(columns=cols).set_index("token")
    R[n] = d[list(cols.values())].astype(float)
tok = R["gimm"].index
for n in R: R[n] = R[n].loc[tok]
def decomp(d):
    M = d.NC * d.DAC; W = (5 * d.EP + 5 * d.TTC + 2 * d.C) / 12
    chk = np.abs(M * W - d.PDMS).max()
    lm = (1 - M); share_nc = np.where((1 - d.NC) + (1 - d.DAC) > 0, (1 - d.NC) / ((1 - d.NC) + (1 - d.DAC)).replace(0, 1), 0)
    out = {"PDMS": 100 * d.PDMS.mean(), "lost": 100 * (1 - d.PDMS.mean()),
           "NC": 100 * (lm * share_nc).mean(), "DAC": 100 * (lm * (1 - share_nc)).mean(),
           "EP": 100 * (M * 5 * (1 - d.EP) / 12).mean(), "TTC": 100 * (M * 5 * (1 - d.TTC) / 12).mean(), "C": 100 * (M * 2 * (1 - d.C) / 12).mean(),
           "NC fail %": 100 * (d.NC < 1).mean(), "DAC fail %": 100 * (d.DAC < 1).mean(), "TTC fail %": 100 * (d.TTC < 1).mean(),
           "zero %": 100 * (d.PDMS == 0).mean(), "maxerr": chk}
    return out
T = pd.DataFrame({n: decomp(d) for n, d in R.items()}).T
print(T.round(2).to_string())
g, h, c = (R[n].PDMS for n in ("gimm", "hydra", "clsref"))
print("\nper-token best-of oracle PDMS:")
print(" gimm|hydra", round(100 * np.maximum(g, h).mean(), 2), " gimm|hydra|clsref", round(100 * np.max([g, h, c], 0).mean(), 2))
print(" gimm zero & hydra >0.8: %.2f%%   hydra zero & gimm >0.8: %.2f%%" % (100 * ((g == 0) & (h > .8)).mean(), 100 * ((h == 0) & (g > .8)).mean()))
print(" corr(gimm,hydra) PDMS:", round(np.corrcoef(g, h)[0, 1], 3))
# which sub-score kills tokens (PDMS==0) in gimm
z = R["gimm"][R["gimm"].PDMS == 0]
print(" gimm zero tokens:", len(z), "NC=0:", int((z.NC == 0).sum()), "DAC=0:", int((z.DAC == 0).sum()), "both:", int(((z.NC == 0) & (z.DAC == 0)).sum()))
# mean-level approximation formula check
d = R["gimm"].mean(); print(" mean-formula approx PDMS gimm:", round(100 * d.NC * d.DAC * (5 * d.EP + 5 * d.TTC + 2 * d.C) / 12, 2))
