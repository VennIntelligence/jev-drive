"""Loss budget, HUGSIM 64: failure classes of two reference arms from the existing runs, and the HD ceiling per class.

No new driving. Arms are the logged PR #57-controller runs on the 64-scenario exam (experiments/hugsim/results/derot/derot_runs.csv,
experiments/op_adapt_h/results/hugsim64/spins.csv, experiments/op_adapt_h/results/one_driver/hugsim64/*/spins.csv, experiments/hugsim/results/
hugsim-exam/scored_{op,base}.csv). Ceiling per class = HD points (mean HD x 100 over the 64 scenarios) gained if every scenario of the class
were set to (a) its best HD over the arms run so far (never lowered), or (b) 1.0. ORACLE CEILINGS, not achievable gains.

  .venv/bin/python experiments/leaderboard_audit/scripts/loss_budget_hugsim.py   -> results/loss_budget/hugsim.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path(__file__).resolve().parents[2]            # experiments/
OUT = E / "leaderboard_audit/results/loss_budget"
N = 64


def arms():
    """Cinque-family arms of ours (all PR #57 controller) and the other models' arms, each indexed by scenario. Inputs are copied under
    results/loss_budget/hugsim_inputs/ (results.csv of each run, derot_runs.csv, the exam's scored_*.csv)."""
    I = OUT / "hugsim_inputs"
    A = {}
    d = pd.read_csv(I / "derot_runs.csv")                                          # arm, spin, rc, end, hdscore
    for a, g in d.groupby("arm"):
        A[f"cinque:{a}"] = g.set_index("scenario")
    A["cinque:fixed"] = A.pop("cinque:base")                                       # shipped Cinque (tag cinque-fixed)
    spins = {"it_dw3-s0_sel3": E / "op_adapt_h/results/one_driver/hugsim64/it_dw3-s0_sel3/spins.csv",
             "it_dw3-s0_all64": E / "op_adapt_h/results/one_driver/hugsim64/it_dw3-s0_all64/spins.csv",
             "pilot-s0_all64": E / "op_adapt_h/results/hugsim64/spins.csv"}
    for f in sorted(I.glob("*-s0*.csv")):
        g = pd.read_csv(f).set_index("scenario")
        if f.stem in spins:
            g["spin"] = pd.read_csv(spins[f.stem]).set_index("scenario").spin.reindex(g.index)
        else:
            g["spin"] = False
        A[f.stem.replace("it_dw3-s0_sel3", "it_dw3+sel3").replace("it_dw3-s0_all64", "it_dw3")] = g
    others = {}
    for f, tags in (("exam_scored_base.csv", ("ltf-fixed", "cv-fixed")), ("exam_scored_op.csv", ("lebowski-fixed",))):
        d = pd.read_csv(I / f)
        for t in tags:
            others[t] = d[d.tag == t].set_index("scenario")
    return A, others


def classify(g):
    """Exclusive failure class of one run (rows = scenarios)."""
    c = pd.Series("complete (HD < 1)", index=g.index)
    c[g["end"] == "off_route"] = "off-road / off-route"
    c[g["end"] == "max_steps"] = "stopped / max_steps"
    c[g["end"] == "bg_collision"] = "collision: background (bg)"
    c[g["end"] == "fg_collision"] = "collision: foreground (fg)"
    late = g["rc"].fillna(0) >= 0.9
    c[g["end"].isin(["bg_collision", "fg_collision"]) & late] = "route-end crash (collision at RC >= 0.9)"
    c[g["spin"].astype(bool)] = "spin (heading > 45 deg off route)"
    c[g.hdscore >= 0.999] = "solved (HD ~ 1)"
    return c


def main():
    A, others = arms()
    scen = A["cinque:fixed"].index
    assert len(scen) == N
    own = {k: v for k, v in A.items()}
    best_ours = pd.concat([v.hdscore for v in own.values()], axis=1).max(axis=1).reindex(scen)
    best_any = pd.concat([v.hdscore for v in list(own.values()) + list(others.values())], axis=1).max(axis=1).reindex(scen)
    res = {"arms_ours": sorted(own), "arms_other_models": sorted(others), "n": N, "best_obs_ours_mean": 100 * float(best_ours.mean()),
           "best_obs_any_mean": 100 * float(best_any.mean())}
    for ref in ("cinque:fixed", "it_dw3+sel3"):
        g = A[ref].reindex(scen)
        g["cls"] = classify(g)
        hd = g.hdscore
        rows = {}
        for c, idx in g.groupby("cls").groups.items():
            idx = list(idx)
            rows[c] = dict(n=len(idx), share_of_failures=len(idx) / max(int((g.cls != "solved (HD ~ 1)").sum()), 1) if c != "solved (HD ~ 1)" else None,
                           hd_mean_points=100 * float(hd[idx].sum() / N),
                           ceiling_best_ours=100 * float(np.maximum(best_ours[idx] - hd[idx], 0).sum() / N),
                           ceiling_best_any=100 * float(np.maximum(best_any[idx] - hd[idx], 0).sum() / N),
                           ceiling_one=100 * float((1 - hd[idx]).sum() / N))
        res[ref] = dict(hd_mean=100 * float(hd.mean()), rows=rows, n_failing=int((g.cls != "solved (HD ~ 1)").sum()),
                        spins=int(g.spin.astype(bool).sum()), all_best_ours=100 * float(np.maximum(best_ours - hd, 0).mean()),
                        all_best_any=100 * float(np.maximum(best_any - hd, 0).mean()), all_one=100 * float((1 - hd).mean()))
        g[["hdscore", "end", "spin", "rc", "cls"]].to_csv(OUT / f"hugsim_classes_{ref.replace(':', '_').replace('+', '_')}.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hugsim.json").write_text(json.dumps(res, indent=1))
    for ref in ("cinque:fixed", "it_dw3+sel3"):
        print(ref, round(res[ref]["hd_mean"], 1), "failing", res[ref]["n_failing"], "all:", round(res[ref]["all_best_ours"], 1), round(res[ref]["all_best_any"], 1), round(res[ref]["all_one"], 1))
        for c, r in res[ref]["rows"].items():
            print(f"  {c:45s} n={r['n']:3d} best_ours={r['ceiling_best_ours']:.1f} best_any={r['ceiling_best_any']:.1f} one={r['ceiling_one']:.1f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
