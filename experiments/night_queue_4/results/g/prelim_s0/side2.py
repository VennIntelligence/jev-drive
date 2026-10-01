"""Descriptive sensitivity (not registered): the following exclusion dropped ~2/3 of windows; same rates with every
reached window kept; ghost-world cruise speed in control windows; TFv6 orig vs ghost without the exclusion."""
import numpy as np, pandas as pd
from jevdrive import nq4_g as NG
out = NG.root("results", "g_prelim_s0")
d = pd.read_csv(out / "per_run.csv", dtype={"base": str}, low_memory=False)
W = d[d.window.notna() & d.w_status.isin(["ok", "following"])].copy()
W["reaction"] = W.w_reaction.astype(float)
md = ["# Seed-0 sensitivity: following windows kept (descriptive, not the registered readout)"]
rows = []
for (c, v), g in W[W.variant.isin(["ghost", "orig"])].groupby(["cand", "variant"]):
    pr = g.groupby(["wkind", "base"]).reaction.mean()
    t, k = pr.get("trigger", pd.Series(dtype=float)), pr.get("control", pd.Series(dtype=float))
    mt, lt, ht, nt = NG.boot(t.to_numpy()); mk, lk, hk, nk = NG.boot(k.to_numpy())
    j = pd.concat([t.rename("t"), k.rename("k")], axis=1, join="inner").dropna()
    mp, lp, hp, npair = NG.boot((j.t - j.k).to_numpy())
    rows.append({"cand": c, "variant": v, "trigger_rate_all": mt, "lo": lt, "hi": ht, "routes": nt,
                 "control_rate_all": mk, "c_lo": lk, "c_hi": hk, "c_routes": nk,
                 "trig_minus_ctrl": mp, "p_lo": lp, "p_hi": hp, "p_routes": npair})
s = pd.DataFrame(rows)
s.to_csv(out / "sensitivity_following_kept.csv", index=False)
md += ["## Reaction rate with following windows kept", s.to_markdown(index=False, floatfmt=".3f")]
p = s[(s.variant == "ghost")].set_index("cand").trigger_rate_all
md.append(f"PDM-Lite ghost trigger rate with following kept: {p.get('pdm', np.nan):.3f}")
cr = W[(W.variant == "ghost") & (W.wkind == "control")].groupby(["cand", "base"]).w_v_mean.mean().groupby("cand").agg(["mean", "size"])
cr.columns = ["ghost_control_mean_mps", "routes"]
cr.to_csv(out / "ghost_control_speed.csv")
md += ["## Ghost world: mean speed in the control windows (all reached windows, m/s)", cr.to_markdown(floatfmt=".2f")]
t = W[(W.cand == "tfv6") & (W.wkind == "trigger") & W.variant.isin(["orig", "ghost"])].pivot_table(index="base", columns="variant", values=["reaction", "w_decel", "w_lat", "w_v_entry"])
t = t.dropna()
m, lo, hi, n = NG.boot((t["reaction", "orig"] - t["reaction", "ghost"]).to_numpy())
tt = pd.DataFrame([{"routes": n, "orig_rate": t["reaction", "orig"].mean(), "ghost_rate": t["reaction", "ghost"].mean(),
                    "orig_minus_ghost": m, "lo": lo, "hi": hi,
                    "orig_decel_med": t["w_decel", "orig"].median(), "ghost_decel_med": t["w_decel", "ghost"].median(),
                    "orig_lat_med": t["w_lat", "orig"].median(), "ghost_lat_med": t["w_lat", "ghost"].median(),
                    "orig_ventry_med": t["w_v_entry", "orig"].median(), "ghost_ventry_med": t["w_v_entry", "ghost"].median()}])
tt.to_csv(out / "tfv6_orig_vs_ghost_all.csv", index=False)
md += ["## TFv6 trigger window, orig vs ghost, following kept (paired by route)", tt.to_markdown(index=False, floatfmt=".3f")]
# per-route ghost reactions under the looser count, to see which routes carry the rate
g = W[(W.variant == "ghost") & (W.wkind == "trigger") & (W.reaction > 0)]
lst = g.groupby("cand").apply(lambda x: " ".join(f"{b}({f})" for b, f in zip(x.base, x.family))).rename("routes")
lst.to_csv(out / "ghost_reacting_routes_following_kept.csv")
md += ["## Ghost trigger reactions, following kept", lst.to_frame().to_markdown()]
(out / "sensitivity.md").write_text("\n\n".join(md) + "\n")
print(open(out / "sensitivity.md").read())
