"""Descriptive seed-0 peek at G (not a verdict): the registered readouts of jevdrive.nq4_g, unchanged, on seed 0 only,
plus a few descriptive side tables. Output: $DATA_DIR/runs/nq4/gk/results/g_prelim_s0/ (the lane's results/g untouched)."""
import json
import numpy as np
import pandas as pd
from jevdrive import nq4_g as NG

CANDS = ("pdm", "tfv6", "bridgedrive", "blue", "simlingo")
out = NG.root("results", "g_prelim_s0")
df = NG.collect()
df.to_csv(out / "per_run_all_seeds.csv", index=False)
d0 = df[(df.seed == 0) & df.cand.isin(CANDS)].copy()
d0.to_csv(out / "per_run.csv", index=False)
NG.g_tables(d0, out)                                   # registered readouts, seed 0

W, R = d0[d0.window.notna()], d0[d0.window.isna()]
ok = W[W.w_status == "ok"].copy()
ok["reaction"] = ok.w_reaction.astype(float)
md = ["# Seed-0 descriptive side tables (not registered verdicts)"]

# coverage: which G routes are missing per candidate x variant (seed 0)
routes = NG.route_table().base.tolist()
cov = []
for (c, v), g in R.groupby(["cand", "variant"]):
    have = set(g[g.status == "ok"].base)
    miss = sorted(set(routes) - have)
    traced = int(g.get("traced", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    cov.append({"cand": c, "variant": v, "runs_ok": len(have), "traced_with_actors": traced,
                "source": ",".join(sorted(g.source.dropna().unique())), "missing": " ".join(miss)})
cov = pd.DataFrame(cov)
cov.to_csv(out / "route_coverage.csv", index=False)
md += ["## Route coverage (seed 0)", cov.to_markdown(index=False)]

# trigger window vs control window, per candidate x world, route bootstrap; paired trigger - control on shared routes
rows = []
for (c, v), g in ok.groupby(["cand", "variant"]):
    pr = g.groupby(["wkind", "base"]).reaction.mean()
    t = pr.get("trigger", pd.Series(dtype=float))
    k = pr.get("control", pd.Series(dtype=float))
    mt, lt, ht, nt = NG.boot(t.to_numpy())
    mk, lk, hk, nk = NG.boot(k.to_numpy())
    j = pd.concat([t.rename("t"), k.rename("k")], axis=1, join="inner").dropna()
    md_, ld, hd, nd = NG.boot((j.t - j.k).to_numpy())
    tr = g[g.wkind == "trigger"]
    rows.append({"cand": c, "variant": v, "trigger_rate": mt, "trig_lo": lt, "trig_hi": ht, "trig_routes": nt,
                 "control_rate": mk, "ctrl_lo": lk, "ctrl_hi": hk, "ctrl_routes": nk,
                 "paired_trig_minus_ctrl": md_, "pair_lo": ld, "pair_hi": hd, "pair_routes": nd,
                 "trig_decel_mean": tr.w_decel.mean(), "trig_lat_mean": tr.w_lat.mean(),
                 "trig_decel_hit": (tr.w_decel >= NG.DECEL_MPS).mean(), "trig_lat_hit": (tr.w_lat >= NG.LAT_M).mean(),
                 "trig_v_entry_mean": tr.w_v_entry.mean()})
tc = pd.DataFrame(rows)
tc.to_csv(out / "trigger_vs_control.csv", index=False)
md += ["## Reaction rate, trigger vs control window, per world (seeds: 0; route bootstrap 10k)",
       tc.to_markdown(index=False, floatfmt=".3f")]

# orig vs ghost at the hazard point: paired per route
og = []
for c, g in ok[ok.wkind == "trigger"].groupby("cand"):
    p = g.pivot_table(index="base", columns="variant", values=["reaction", "w_decel", "w_lat"], aggfunc="mean")
    if ("reaction", "orig") not in p or ("reaction", "ghost") not in p:
        continue
    p = p.dropna(subset=[("reaction", "orig"), ("reaction", "ghost")])
    m, lo, hi, n = NG.boot((p["reaction", "orig"] - p["reaction", "ghost"]).to_numpy())
    both = ((p["reaction", "orig"] > 0) & (p["reaction", "ghost"] > 0)).mean()
    og.append({"cand": c, "routes": n, "orig_rate": p["reaction", "orig"].mean(), "ghost_rate": p["reaction", "ghost"].mean(),
               "orig_minus_ghost": m, "lo": lo, "hi": hi, "react_in_both_share": both,
               "orig_decel_mean": p["w_decel", "orig"].mean(), "ghost_decel_mean": p["w_decel", "ghost"].mean(),
               "orig_lat_mean": p["w_lat", "orig"].mean(), "ghost_lat_mean": p["w_lat", "ghost"].mean()})
og = pd.DataFrame(og)
og.to_csv(out / "orig_vs_ghost_trigger.csv", index=False)
md += ["## Trigger window, orig vs ghost, paired by route (seed 0; route bootstrap)", og.to_markdown(index=False, floatfmt=".3f")]

# ghost trigger rate by route set (sudden-hazard P5 vs obstacle P6)
bs = []
for (c, st), g in ok[(ok.variant == "ghost") & (ok.wkind == "trigger")].groupby(["cand", "set"]):
    m, lo, hi, n = NG.boot(g.groupby("base").reaction.mean().to_numpy())
    bs.append({"cand": c, "set": st, "ghost_rate": m, "lo": lo, "hi": hi, "routes": n})
bs = pd.DataFrame(bs)
bs.to_csv(out / "ghost_rate_by_set.csv", index=False)
md += ["## Ghost trigger-window rate by route set (seed 0)", bs.to_markdown(index=False, floatfmt=".3f")]

# PDM-Lite ghost baseline with every seed available (the registered baseline pools its seeds)
pg = df[(df.cand == "pdm") & (df.variant == "ghost") & (df.wkind == "trigger") & (df.w_status == "ok")].copy()
pg["reaction"] = pg.w_reaction.astype(float)
pb = []
for sel, g in (("seed0", pg[pg.seed == 0]), ("seeds_all_done", pg)):
    m, lo, hi, n = NG.boot(g.groupby("base").reaction.mean().to_numpy())
    pb.append({"pdm_ghost": sel, "seeds": ",".join(map(str, sorted(g.seed.unique()))), "runs": len(g), "rate": m, "lo": lo, "hi": hi, "routes": n})
pb = pd.DataFrame(pb)
pb.to_csv(out / "pdm_ghost_baseline.csv", index=False)
md += ["## PDM-Lite ghost baseline, seed 0 and all finished seeds", pb.to_markdown(index=False, floatfmt=".3f")]

# which routes react in the ghost trigger window (per candidate), to watch for consistency as seeds 1-2 arrive
gr = ok[(ok.variant == "ghost") & (ok.wkind == "trigger") & (ok.reaction > 0)]
gl = gr.groupby("cand").apply(lambda g: " ".join(f"{b}({f})" for b, f in zip(g.base, g.family))).rename("ghost_reacting_routes")
gl.to_csv(out / "ghost_reacting_routes.csv")
md += ["## Routes with a ghost reaction in the trigger window (seed 0)", gl.to_frame().to_markdown()]
(out / "side.md").write_text("\n\n".join(md) + "\n")
print(open(out / "g.md").read())
print(open(out / "side.md").read())
