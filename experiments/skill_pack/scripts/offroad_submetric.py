"""Q4: where do the N4 (and native) navhard points go, by sub-metric and stage (plan: experiments/skill_pack/plans/2026-10-03-...).

navsim2 env, CPU. Reads tokens_<model>_base.csv written by offroad_replay_cf.py (per-token sub-scores, official stage-two weights,
the aggregator's two-frame comfort), recomputes the official combined EPDMS with the devkit's `calculate_individual_mapping_scores`
and the loss of every term as "combined if that term is set to 1": all tokens / stage 1 only / stage 2 only. The progress loss is
split by starting state (v0 < 1 m/s: plan not started / started; moving) and traffic-light compliance is reported with its own row.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
MULT, WT = SUBS[:4], dict(ego_progress=5, time_to_collision_within_bound=5, lane_keeping=2, history_comfort=2, two_frame_extended_comfort=2)
SH = dict(zip(SUBS, ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]))


def aggregate(d, mapping):
    from navsim.planning.script.run_pdm_score import calculate_individual_mapping_scores
    cols = SUBS + ["score"]
    comb, s1, s2 = calculate_individual_mapping_scores(d[cols + ["token", "weight"]], mapping)
    return comb, s1, s2


def rescore(d):
    d = d.copy()
    d["score"] = np.prod([d[c] for c in MULT], axis=0) * sum(WT[k] * d[k] for k in WT) / 16
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/skill_pack/results/navhard-offroad/tables"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _, _, _, mapping, _ = L.setup_scoring()
    idx = {e["token"]: e for e in L.index()}
    pn, p4 = L.poses_by_token(L.NATIVE_POSES), L.poses_by_token(L.N4_POSES)
    res, loss_rows, ep_rows = {}, [], []
    for m, P in (("n4", p4), ("native", pn)):
        d = pd.read_csv(L.OUT / f"tokens_{m}_base.csv")
        d = d[d.valid.astype(bool)] if "valid" in d else d
        # merge the stage columns of the final df into single columns (the CSV keeps the per-stage names only for the official file)
        for s in SUBS + ["score"]:
            assert s in d.columns, s
        d["stage"] = [idx[t]["stage"] for t in d.token]
        d["v0"] = [float(np.linalg.norm(idx[t]["vel"][-1])) for t in d.token]
        d["plan_x4"] = [float(P[t][-1, 0]) for t in d.token]
        comb, s1, s2 = aggregate(d, mapping)
        base = float(comb["score"])
        res[m] = dict(combined=100 * base, stage1=100 * float(s1["score"]), stage2=100 * float(s2["score"]))
        for s in SUBS:
            res[m][SH[s] + "_s1"], res[m][SH[s] + "_s2"] = 100 * float(s1[s]), 100 * float(s2[s])
        for s in SUBS:
            row = dict(model=m, term=SH[s], kind="multiplicative" if s in MULT else f"weighted x{WT[s]}/16", s1=100 * float(s1[s]), s2=100 * float(s2[s]))
            for scope, mask in (("all", np.ones(len(d), bool)), ("stage1", (d.stage == "one").to_numpy()), ("stage2", (d.stage == "two").to_numpy())):
                x = d.copy()
                x.loc[mask, s] = 1.0
                row[f"gain_{scope}"] = 100 * (float(aggregate(rescore(x), mapping)[0]["score"]) - float(aggregate(rescore(d), mapping)[0]["score"]))
            loss_rows.append(row)
        # progress split by starting state
        groups = {"v0<1 not started (plan x4 < 2 m)": (d.v0 < 1) & (d.plan_x4 < 2), "v0<1 started (plan x4 >= 2 m)": (d.v0 < 1) & (d.plan_x4 >= 2),
                  "v0 1-3": (d.v0 >= 1) & (d.v0 < 3), "v0 >= 3": d.v0 >= 3}
        for g, mk in groups.items():
            for st in ("one", "two"):
                k = mk & (d.stage == st)
                x = d.copy()
                x.loc[k, "ego_progress"] = 1.0
                ep = d.loc[k, "ego_progress"]
                ep_rows.append(dict(model=m, group=g, stage=st, n=int(k.sum()), ep_mean=float(ep.mean()) if k.any() else np.nan,
                                    ep_lt_half=float((ep < .5).mean()) if k.any() else np.nan,
                                    gain_if_ep1=100 * (float(aggregate(rescore(x), mapping)[0]["score"]) - float(aggregate(rescore(d), mapping)[0]["score"]))))
        # traffic light: tokens where TLC < 1, stage split, and their other losses
        for st in ("one", "two"):
            k = (d.stage == st) & (d.traffic_light_compliance < 1)
            ep_rows.append(dict(model=m, group="TLC<1 tokens", stage=st, n=int(k.sum()), ep_mean=float(d.loc[k, "ego_progress"].mean()) if k.any() else np.nan,
                                ep_lt_half=np.nan, gain_if_ep1=np.nan))
        d.to_pickle(L.OUT / f"submetric_{m}.pkl")
    pd.DataFrame(loss_rows).round(3).to_csv(out / "q4_submetric_loss.csv", index=False)
    pd.DataFrame(ep_rows).round(3).to_csv(out / "q4_progress_split.csv", index=False)
    (out / "q4_official_subscores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(pd.DataFrame(loss_rows).round(2).to_string(index=False))
    print(pd.DataFrame(ep_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
