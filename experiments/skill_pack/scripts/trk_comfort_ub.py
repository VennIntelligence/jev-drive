"""Tracker-lag pre-compensation (plans/2026-10-04-tracker-precomp-plan.md): comfort-neutral upper bound on navhard.
Every compensated / ideal arm rescored with the uncompensated arm's history comfort and extended comfort per token (EPDMS
recomposed from the in-process token table of trk_report.py navhard; exact for the base arm), group bootstrap as there.
navsim2 env:  trk_comfort_ub.py [_path]   -> results/tracker-precomp/comfort_ub<mode>.json"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import offroad_lib as L
from offroad_gain import group_scores
mode = sys.argv[1] if len(sys.argv) > 1 else ""
df = pd.read_parquet(L.D / f"runs/skill_pack/trk/navhard_tokens{mode}.parquet")
_, _, _, mapping, _ = L.setup_scoring()
def recompute(d):
    m = d.no_at_fault_collisions * d.drivable_area_compliance * d.driving_direction_compliance * d.traffic_light_compliance
    w = (5 * d.ego_progress + 5 * d.time_to_collision_within_bound + 2 * d.lane_keeping + 2 * d.history_comfort + 2 * d.two_frame_extended_comfort) / 16
    return m * w
arms = df.index.get_level_values(0).unique()
out = {}
for m in ("native", "n4"):
    b = df.loc[f"{m}/base"].copy()
    err = float(np.nanmax(np.abs(recompute(b) - b.score)))
    gb = group_scores(b.reset_index(), mapping)
    for v in [a for a in arms if a.startswith(m) and not a.endswith("base")]:
        d = df.loc[v].copy()
        d["history_comfort"], d["two_frame_extended_comfort"] = b.history_comfort, b.two_frame_extended_comfort
        d["score"] = recompute(d)
        g = group_scores(d.reset_index(), mapping)
        dg = 100 * (g - gb)
        B = np.random.default_rng(0).integers(0, len(dg), (5000, len(dg)))
        out[v] = dict(combined_base_comfort=100 * float(g.mean()), delta=float(dg.mean()), ci=[float(np.percentile(dg[B].mean(1), q)) for q in (2.5, 97.5)], recompute_err_base=err)
        print(v, {k: (round(x, 3) if isinstance(x, float) else [round(y, 2) for y in x]) for k, x in out[v].items()})
json.dump(out, open(Path(__file__).resolve().parents[1] / f"results/tracker-precomp/comfort_ub{mode}.json", "w"), indent=1)
