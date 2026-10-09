#!/usr/bin/env python3
"""BODY1 gate G2 harness, part 2: sanity reproduction through g2.score() of the two published AUCs of decision 220.

  python experiments/body1/scripts/g2_verify.py [--dump DIR]    # DIR = $DATA_DIR/runs/body1/g2 (box) or a local copy; needs the dump + fresh replay

1. plan_std2_FT recomputed (a) from the freshly replayed fp16 `plan` slice with the swv1_bc.geo_signals formula, (b) from the dump's
   fp32 plan_raw; 2. road-edge margin read from the decision table (edge_margin_P0, as published; the table is the swv1 signal, no
   sim state is recomputed here); both through g2.score(). Published: 0.786 [0.738, 0.865] and 0.718 [0.643, 0.822].
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "lib")]
import g2  # noqa: E402
from swv1_bc import T_IDXS  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))


def std_at(plan, t=2.0):
    sd = np.exp(np.minimum(np.asarray(plan)[495:].reshape(33, 15)[:, 1], 11))
    return float(np.interp(t, T_IDXS, sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(DATA / "runs/body1/g2"))
    a = ap.parse_args()
    d = Path(a.dump)
    table = g2.load_table()
    rows_a, rows_b = [], []
    for grp in ("P2H10-F-s0", "P2H10-F-s1", "ctrl"):
        fresh = pickle.load(open(d / "replay/replay" / f"{grp}.pkl", "rb"))
        z = np.load(d / "dump" / f"{grp}.npz")
        n = 0
        for sc, recs in fresh.items():
            for k, r in enumerate(recs):
                rows_a.append((grp, sc, k, std_at(r["FT"]["plan"].astype(np.float32))))
                rows_b.append((grp, sc, k, std_at(z["plan_raw"][n])))
                assert z["scene"][n] == sc and z["k"][n] == k
                n += 1
    out = {}
    for name, rows in (("plan_std2_FT from fresh replay fp16 plan", rows_a), ("plan_std2_FT from dump plan_raw fp32", rows_b)):
        out[name] = g2.score(pd.DataFrame(rows, columns=["set", "scene", "k", "score"]), table)
    t = pd.DataFrame(table)
    out["edge_margin_P0 (decision table)"] = g2.score(t.rename(columns={"edge_margin_P0": "score"})[["set", "scene", "k", "score"]], table)
    for k, v in out.items():
        print(f"{k}: AUC {v['auc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}], {v['n_pos']} / {v['n_neg']}, missing {v['n_missing']}")
    # the decision table's own plan_std2_FT versus the recomputed one
    tb = {(r["set"], r["scene"], int(r["k"])): r["plan_std2_FT"] for r in table}
    for name, rows in (("fresh", rows_a), ("dump", rows_b)):
        diff = [abs(tb[(g, s, k)] - v) for g, s, k, v in rows if (g, s, k) in tb]
        print(f"max |table plan_std2_FT - {name}| over {len(diff)} decisions: {max(diff):.2e}")


if __name__ == "__main__":
    main()
