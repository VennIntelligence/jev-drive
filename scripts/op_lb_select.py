"""Pre-registered arm selection for the op-lb lane (todos/2026-09-29-op-leaderboard.md, "Arm selection rule").

Per model: A* = best non-`none` gimm arm by navtrain-subset PDMS; paired per-token delta A* - none with a token
bootstrap 95% CI (B = 2000, seed 0). CI lower bound > 0 -> A* goes to the test splits, else the headline is `none`.
Also reports every arm's paired delta (descriptive) and the delta per driving command for A*.

  python scripts/op_lb_select.py [out_dir]     (envs/jevdrive; reads $DATA_DIR/runs/{op_lb,navsim/eval})
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive.common import data_dir

B, SEED = 2000, 0
CMD = {0: "left", 1: "straight", 2: "right", 3: "unknown"}
ARMS = [f"turn@{t}" for t in ("m1.5", "m1.0", "m0.5", "0", "onset")] + [f"lc@{t}" for t in ("m1.5", "m1.0", "m0.5", "0")]


def stem(model, arm):  # `turn@m1.5` -> `gimm-cinque.turn_m1.5__base`; `none` has no schedule tag
    return f"gimm-{model}__base" if arm == "none" else f"gimm-{model}.{arm.replace('@', '_')}__base"


def boot(d, rng, idx):
    return np.percentile(d[idx].mean(1), [2.5, 97.5])


def main(out):
    root = data_dir() / "runs/op_lb/lb_navtrain"
    meta = json.load(open(root / "meta.json"))
    toks, cmd = meta["names"], np.array([c[-1] for c in meta["cmds"]])

    def load(name):
        fs = sorted(glob.glob(str(data_dir() / f"runs/navsim/eval/v1_navtrain_opi_lb_navtrain_{name}/*/*.csv")))
        r = pd.read_csv(fs[-1])
        r = r[r["token"].isin(set(toks)) & r["valid"].astype(bool)].set_index("token")["score"]
        return r.reindex(toks).to_numpy()

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(toks), (B, len(toks)))  # one shared resample for every comparison
    rows, sel = [], {}
    for model in ("cinque", "lebowski"):
        none = load(stem(model, "none"))
        res = {}
        for arm in ARMS:
            s = load(stem(model, arm))
            d = (s - none) * 100
            lo, hi = boot(d, rng, idx)
            res[arm] = (s.mean() * 100, d.mean(), lo, hi)
            rows.append(dict(model=model, arm=arm, pdms=s.mean() * 100, none=none.mean() * 100, delta=d.mean(), lo=lo, hi=hi))
        best = max(res, key=lambda a: res[a][0])
        sel[model] = dict(arm=best, pdms=res[best][0], none=none.mean() * 100, delta=res[best][1], lo=res[best][2],
                          hi=res[best][3], advance=bool(res[best][2] > 0))
        s = load(stem(model, best))
        for c in (0, 1, 2):
            m = cmd == c
            d = ((s - none) * 100)[m]
            lo, hi = np.percentile(d[rng.integers(0, m.sum(), (B, m.sum()))].mean(1), [2.5, 97.5])
            rows.append(dict(model=model, arm=f"{best} | {CMD[c]}", pdms=s[m].mean() * 100, none=none[m].mean() * 100,
                             delta=d.mean(), lo=lo, hi=hi))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "navtrain_paired.csv", index=False, float_format="%.3f")
    json.dump(sel, open(out / "selection.json", "w"), indent=1)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(json.dumps(sel, indent=1))




def test_paired(out, model="cinque", arm="lc@m1.0"):
    """navtest paired delta of the selected arm vs none (single pass, per the registration): all, left+right, straight,
    left, right. Token bootstrap 95% CI, B = 2000, seed 0."""
    root = data_dir() / "runs/op_lb/lb_navtest"
    meta = json.load(open(root / "meta.json"))
    toks, cmd = meta["names"], np.array([c[-1] for c in meta["cmds"]])

    def load(name):
        fs = sorted(glob.glob(str(data_dir() / f"runs/navsim/eval/v1_navtest_opi_lb_navtest_{name}/*/*.csv")))
        r = pd.read_csv(fs[-1])
        return r[r["token"].isin(set(toks)) & r["valid"].astype(bool)].set_index("token")["score"].reindex(toks).to_numpy()

    a, n = load(stem(model, arm)), load(stem(model, "none"))
    rng, rows = np.random.default_rng(SEED), []
    for lab, m in (("all", cmd >= 0), ("left+right", (cmd == 0) | (cmd == 2)), ("straight", cmd == 1), ("left", cmd == 0),
                   ("right", cmd == 2)):
        d = ((a - n) * 100)[m]
        lo, hi = np.percentile(d[rng.integers(0, m.sum(), (B, m.sum()))].mean(1), [2.5, 97.5])
        rows.append(dict(model=model, arm=arm, group=lab, n=int(m.sum()), arm_pdms=a[m].mean() * 100, none_pdms=n[m].mean() * 100,
                         delta=d.mean(), lo=lo, hi=hi))
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "navtest_paired.csv", index=False, float_format="%.3f")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:2] == ["test"]:
        test_paired(Path(sys.argv[2] if len(sys.argv) > 2 else data_dir() / "runs/op_lb/lb_navtest/select"))
    else:
        main(Path(sys.argv[1] if len(sys.argv) > 1 else data_dir() / "runs/op_lb/lb_navtrain/select"))
