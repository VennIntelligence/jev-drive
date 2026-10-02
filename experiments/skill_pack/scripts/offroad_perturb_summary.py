"""Summarise the input-perturbation battery over the opposite-side class and a random control (navsim2 env, CPU).

perturb_opp.pkl / perturb_ctrl.pkl (offroad_perturb.py) -> per variant: share of tokens still on the opposite side of the PDM
reference, median shift of the 4 s lateral end toward the reference side, and the DAC of the perturbed plan under the official
per-token scorer (devkit pdm_score, reactive agents), against the baseline plan.
"""
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

_W = {}


def _init():
    sim, scorer, policy, mapping, samp = L.setup_scoring()
    _W.update(sim=sim, scorer=scorer, policy=policy, samp=samp, cp=L.cache_paths())


def score(job):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    t, name, p8 = job
    mc = L.load_cache(_W["cp"][t])
    row, _ = pdm_score(metric_cache=mc, model_trajectory=Trajectory(np.asarray(p8, np.float64)), future_sampling=_W["samp"], simulator=_W["sim"],
                       scorer=_W["scorer"], traffic_agents_policy=_W["policy"])
    return t, name, float(row["drivable_area_compliance"].iloc[0]), float(row["ego_progress"].iloc[0])


def main():
    tab = pd.read_pickle(L.OUT / "token_table.pkl").set_index("token")
    out = []
    for tag in ("opp", "ctrl", "camopp"):
        f = L.OUT / f"perturb_{tag}.pkl"
        if not f.exists():
            continue
        P = pickle.load(open(f, "rb"))
        jobs = [(t, k, np.asarray(v["pose8"])) for t, r in P.items() for k, v in r.items() if isinstance(v, dict)]
        with mp.get_context("fork").Pool(24, initializer=_init) as pool:
            sc = {(t, k): (d, e) for t, k, d, e in pool.map(score, jobs, chunksize=4)}
        names = sorted({k for r in P.values() for k in r if isinstance(r[k], dict)}, key=lambda k: (k != "base", k))
        for k in names:
            ys, dy, dac, opp = [], [], [], []
            for t, r in P.items():
                if k not in r:
                    continue
                y = np.asarray(r[k]["pose8"])[-1, 1]
                yb = np.asarray(r["base"]["pose8"])[-1, 1]
                ref = tab.loc[t, "ref_end_y"]
                dy.append(np.sign(ref) * (y - yb))
                opp.append((y * ref < 0) and abs(y) > 1 and abs(ref) > 1 and abs(y - ref) > 2)
                dac.append(sc[(t, k)][0])
            out.append(dict(set=tag, variant=k, n=len(dy), still_opposite=float(np.mean(opp)), shift_toward_ref_median_m=float(np.median(dy)),
                            dac_pass=float(np.mean(dac))))
    df = pd.DataFrame(out)
    df.to_csv(REPO / "experiments/skill_pack/results/navhard-offroad/tables/q1_perturbation_class.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
