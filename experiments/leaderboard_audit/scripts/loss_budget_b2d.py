"""Loss budget, B2D (19 diagnostic routes x seeds 0-3): DS if each official infraction class were removed, for `drive` and `vmerge2`.

Input: experiments/vlm_arb/results/vmerge2_runs.csv (official per-run DS, RC, infraction counts, status). Official penalty multipliers
(Bench2Drive / CARLA leaderboard 2.0): pedestrian 0.50, vehicle 0.60, static layout 0.65, red light 0.70, stop sign 0.80; the lane-departure
factor (outside_route_lanes) is recovered per run as DS / (RC x the five multipliers). The recomposition reproduces the logged DS exactly
in 91% of runs and the rest are runs with an outside-route-lanes factor.
Removing a class: its multiplier -> 1; "blocked / timeout" (status Failed - Agent got blocked / TickRuntime, the 4000-tick cap) -> the run
completes: (A) mechanical, RC -> 100 with the other factors unchanged; (B) exposure-aware, DS = mean DS of the same route's completed runs
(both arms, same seeds pool), which carries the infractions that the longer route exposes. ORACLE CEILINGS, not achievable gains.

  .venv/bin/python experiments/leaderboard_audit/scripts/loss_budget_b2d.py   -> results/loss_budget/b2d.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path(__file__).resolve().parents[2]
OUT = E / "leaderboard_audit/results/loss_budget"
MULT = dict(collisions_pedestrian=.5, collisions_vehicle=.6, collisions_layout=.65, red_light=.7, stop_infraction=.8)
CLASSES = {"red_light": ["red_light"], "stop_sign": ["stop_infraction"], "collision_vehicle": ["collisions_vehicle"], "collision_layout": ["collisions_layout"],
           "collision_pedestrian": ["collisions_pedestrian"], "route_deviation": ["outside_route_lanes"], "blocked_timeout": ["blocked"]}


def load():
    d = pd.read_csv(E / "vlm_arb/results/vmerge2_runs.csv")
    d = d[d.arm.isin(["drive", "vmerge2"])].copy()
    assert d.groupby("arm").size().to_dict() == {"drive": 76, "vmerge2": 76}
    d["blocked"] = (d.status != "Completed").astype(int)
    m = np.prod([MULT[k] ** d[k] for k in MULT], axis=0)
    d["f_out"] = (d.DS / (d.RC * m)).clip(upper=1.0)
    d["ds_hat"] = d.RC * m * d.f_out
    assert np.abs(d.ds_hat - d.DS).max() < 1e-6
    d["f_other"] = m
    return d


def ds_without(d, cls, completed):
    """DS of every run with class `cls` removed (index aligned with d)."""
    rc, m, f = d.RC / 100, d.f_other.copy(), d.f_out.copy()
    if cls == "blocked_timeout":
        rc = pd.Series(1.0, index=d.index).where(d.blocked == 1, rc)
        out = 100 * rc * m * f
        # exposure-aware: the same route's completed runs of both arms
        route_ds = d[d.blocked == 0].groupby("route").DS.mean()
        out_b = out.copy()
        sel = d.blocked == 1
        out_b[sel] = [max(float(route_ds.get(r, np.nan)), x) if not np.isnan(route_ds.get(r, np.nan)) else x for r, x in zip(d.route[sel], out[sel])]
        return out, out_b
    if cls == "route_deviation":
        f = pd.Series(1.0, index=d.index)
    else:
        for k in CLASSES[cls]:
            m = m / MULT[k] ** d[k]
    out = 100 * rc * m * f
    return out, out


def main():
    d = load()
    res = {"n_runs": {a: int((d.arm == a).sum()) for a in ("drive", "vmerge2")}, "routes": int(d.route.nunique())}
    for a in ("drive", "vmerge2"):
        g = d[d.arm == a]
        base = float(g.DS.mean())
        fail = g.DS < 99.999
        rows = {}
        for c in CLASSES:
            has = (g[CLASSES[c][0]] > 0) if c != "blocked_timeout" else (g.blocked == 1)
            A, B = ds_without(g, c, True)
            rows[c] = dict(n_runs=int(has.sum()), share_of_failing_runs=float((has & fail).sum() / fail.sum()), events=int(g[CLASSES[c][0]].sum()),
                           ceiling=float(A.mean() - base), ceiling_exposure_aware=float(B.mean() - base))
        # all collisions together, and all classes together (sequentially applied; mechanical)
        mcoll = g.f_other.copy()
        for k in ("collisions_pedestrian", "collisions_vehicle", "collisions_layout"):
            mcoll = mcoll / MULT[k] ** g[k]
        rows["collision_all"] = dict(n_runs=int(((g.collisions_vehicle + g.collisions_layout + g.collisions_pedestrian) > 0).sum()),
                                     share_of_failing_runs=float((((g.collisions_vehicle + g.collisions_layout + g.collisions_pedestrian) > 0) & fail).sum() / fail.sum()),
                                     events=int((g.collisions_vehicle + g.collisions_layout + g.collisions_pedestrian).sum()),
                                     ceiling=float((100 * g.RC / 100 * mcoll * g.f_out).mean() - base))
        res[a] = dict(ds=base, n_failing_runs=int(fail.sum()), rows=rows, all_removed=100 - base)
        res[a]["by_status"] = {k: int(v) for k, v in g.status.value_counts().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b2d.json").write_text(json.dumps(res, indent=1))
    for a in ("drive", "vmerge2"):
        print(a, round(res[a]["ds"], 2), res[a]["n_failing_runs"])
        for c, r in res[a]["rows"].items():
            print(f"  {c:22s} runs={r['n_runs']:3d} share={100 * r['share_of_failing_runs']:.0f}% ceil={r['ceiling']:.2f} expo={r.get('ceiling_exposure_aware', float('nan')):.2f}")


if __name__ == "__main__":
    main()
