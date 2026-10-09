"""BODY1 gate G2 scorer: score any contact predictor on decision 220's nuPlan P2H10-F decision set with its unchanged AUC reader.

  python experiments/body1/lib/g2.py --scores f.csv        # columns: set (or group), scene, k, score; higher = more likely contact
  from g2 import score; score(df)                          # df with the same columns -> dict(auc, lo, hi, thr10, n_pos, n_neg, ...)

Positives: decisions of collision rollouts (grp C) of P2H10-F-s0 / -s1 with t_to_ev <= 8 s whose served plan sweep intersects the struck
object (hit_st). Negatives: decisions of the clean control group (set ctrl, grp N, ok) whose sweep intersects nothing (not hit_any).
Speed-matched negative weights, cluster bootstrap by log (1000 resamples, seed 0): all inside `swv1_bc.read_auc`, imported unchanged.
The decision table (`results/g2/decision_table.csv.gz`) is `tmp/swv1/sig_nuplan.pkl` of decision 220 (swv1_a.py -> swv1_bc.py) as csv.
`k` is the decision index inside the scene (0-based, in the order of the driver requests), the key of the dump (`g2_dump.py`).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "alpasim" / "scripts")]
import swv1_bc as B  # noqa: E402  (the unchanged AUC reader and speed weights)

TABLE = HERE.parent / "results/g2/decision_table.csv.gz"
POS_SETS = ("P2H10-F-s0", "P2H10-F-s1")
KEY = ["set", "scene", "k"]


def load_table(path=TABLE) -> list[dict]:
    df = pd.read_csv(path)
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()} for r in df.to_dict("records")]


def split(rows):
    """-> (positives, negatives) exactly as swv1_bc.main builds them for the nuPlan primary read."""
    pos = [d for d in rows if d["grp"] == "C" and d["set"] in POS_SETS and d["t_to_ev"] <= 8.0 and d["hit_st"]]
    neg = [d for d in rows if d["grp"] == "N" and d["ok"] and not d["hit_any"]]
    return pos, neg


def score(df: pd.DataFrame, table=None, n_boot: int = 1000, strict: bool = True) -> dict:
    """AUC of df['score'] on decision 220's set. df: columns set (alias group), scene, k, score. Decisions of the set without a score are
    reported in `missing` and raise unless strict=False (read_auc then drops them, as it does for any signal that is absent)."""
    df = df.rename(columns={"group": "set"}) if "set" not in df else df
    if df.duplicated(KEY).any():
        raise ValueError("duplicate (set, scene, k) rows in the scores")
    s = {(r.set, r.scene, int(r.k)): float(r.score) for r in df.itertuples()}
    pos, neg = split(table or load_table())
    miss = [(d["set"], d["scene"], d["k"]) for d in pos + neg if (d["set"], d["scene"], int(d["k"])) not in s]
    if miss and strict:
        raise ValueError(f"{len(miss)} of {len(pos) + len(neg)} decisions have no score, e.g. {miss[:3]}; pass strict=False to drop them")
    for d in pos + neg:
        d["score"] = s.get((d["set"], d["scene"], int(d["k"])))
    r = B.read_auc(pos, neg, "score", n_boot)
    if r is None:
        raise ValueError("too few scored positives / negatives")
    return dict(r, n_missing=len(miss), n_pos_all=len(pos), n_neg_all=len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="csv with set (or group), scene, k, score")
    ap.add_argument("--table", default=str(TABLE)), ap.add_argument("--boot", type=int, default=1000), ap.add_argument("--lenient", action="store_true")
    a = ap.parse_args()
    r = score(pd.read_csv(a.scores), load_table(a.table), a.boot, not a.lenient)
    print(json.dumps(r, indent=1))
    print(f"G2 AUC {r['auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  ({r['n_pos']} positive / {r['n_neg']} negative decisions); "
          f"line: >= 0.80 and lower > 0.70 -> {'PASS' if r['auc'] >= 0.8 and r['lo'] > 0.7 else 'no'}")


if __name__ == "__main__":
    main()
