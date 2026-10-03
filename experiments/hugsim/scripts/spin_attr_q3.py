"""Q3: do the models spin on the same scenes?  python spin_attr_q3.py <out_dir>   (Mac, reads committed result CSVs only)
Arms (64 exam scenarios, PR#57 controller, one run each): native Cinque, Lebowski (spin_episodes.csv), pilot / it_dw3 (decision 98), ln1 / ln3 (decision 106),
derot3 and sel3 rule arms (history_derotate)."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
H = Path(__file__).resolve().parents[2] / "op_adapt_h" / "results"


def load():
    ep = pd.read_csv(R / "spin/spin_episodes.csv")
    ep = ep[ep.controller == "fixed"]
    arms = {}
    for a in ("cinque", "lebowski"):
        d = ep[ep.agent == a]
        arms[a] = dict(zip(d.scenario, d.spin.astype(bool)))
    for name, f in (("pilot", H / "hugsim64/spins.csv"), ("it_dw3", H / "one_driver/hugsim64/it_dw3-s0_all64/spins.csv"),
                    ("ln1", H / "round2/hugsim64_ln1-s0.csv"), ("ln3", H / "round2/hugsim64_ln3-s0.csv")):
        d = pd.read_csv(f)
        arms[name] = dict(zip(d.scenario, d.spin.astype(bool)))
    d = pd.read_csv(R / "derot/derot_runs.csv")
    for a in ("derot3", "sel3"):
        x = d[d.arm == a]
        if len(x) >= 60:
            arms[a] = dict(zip(x.scenario, x.spin.astype(bool)))
    return arms


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    arms = load()
    scen = sorted(arms["cinque"])
    M = pd.DataFrame({a: [int(arms[a].get(s, False)) for s in scen] for a in arms}, index=scen)
    M.to_csv(out / "q3_matrix.csv")
    rows = []
    rng = np.random.default_rng(0)
    for a, b in combinations(arms, 2):
        x, y = M[a].values.astype(bool), M[b].values.astype(bool)
        inter, uni = int((x & y).sum()), int((x | y).sum())
        # hypergeometric p of an intersection at least this large, margins fixed
        from scipy.stats import hypergeom
        p = hypergeom.sf(inter - 1, len(x), int(x.sum()), int(y.sum()))
        rows.append(dict(a=a, b=b, na=int(x.sum()), nb=int(y.sum()), inter=inter, union=uni, jaccard=inter / uni if uni else np.nan,
                         expected_inter=x.sum() * y.sum() / len(x), p_hyper=p))
    J = pd.DataFrame(rows)
    J.to_csv(out / "q3_pairs.csv", index=False)
    print(J.round(3).to_string())
    # scene-level: how many of the six model arms spin on a scene (native Cinque, Lebowski, pilot, it_dw3, ln1, ln3)
    six = ["cinque", "lebowski", "pilot", "it_dw3", "ln1", "ln3"]
    cnt = M[six].sum(1)
    print("scenes by number of the six arms that spin:", cnt.value_counts().sort_index().to_dict())
    # permutation: sum over pairs of intersections under independent scene draws with the arms' own margins
    obs = sum(int((M[a].values & M[b].values).sum()) for a, b in combinations(six, 2))
    sims = []
    for _ in range(5000):
        cols = [rng.permutation(M[a].values) for a in six]
        sims.append(sum(int((x & y).sum()) for x, y in combinations(cols, 2)))
    print("pairwise intersections (six arms): observed", obs, "null mean", np.mean(sims), "p", (np.sum(np.array(sims) >= obs) + 1) / 5001)
    T = M.loc[(M[six].sum(1) > 0) | (M[[c for c in M if c in ("derot3", "sel3")]].sum(1) > 0)]
    T = T.assign(n_six=T[six].sum(1)).sort_values("n_six", ascending=False)
    T.to_csv(out / "q3_table.csv")
    print(T.to_string())
    json.dump(dict(pairs_obs=obs, null_mean=float(np.mean(sims)), p=float((np.sum(np.array(sims) >= obs) + 1) / 5001), by_count=cnt.value_counts().sort_index().to_dict()),
              open(out / "q3.json", "w"))


if __name__ == "__main__":
    main()
