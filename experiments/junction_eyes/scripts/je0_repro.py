#!/usr/bin/env python3
"""junction_eyes step 0, G-repro (c) (plans/2026-10-12-prereg-A.md amendment 5): road_probe/rp_real.py's own `train` (arm A: decision 275's
on-policy rows; arm B: all navtrain), unchanged, with its output directory moved to $DATA_DIR/runs/junction_eyes/step0/repro/ so the stored
rp_real predictions stay untouched; `check` applies the registered pass rule.

  train  (envs/op-train, GPU)  --arm A|B --seed S -> step0/repro/pred/<arm>-s<S>.npz
  check  (AlpaSim venv, CPU)                      -> step0/repro/g_repro_c.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/road_probe/scripts")]
import numpy as np  # noqa: E402
import rp_real as RR  # noqa: E402

OUT = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/junction_eyes/step0/repro"


def train(a):
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    for f in ("navtest.parquet", "navtrain.parquet"):
        if not (OUT / "labels" / f).exists():
            (OUT / "labels" / f).symlink_to(RR.R / "labels" / f)
    RR.R = OUT                                   # rp_real.train reads labels from R/labels and writes R/pred
    RR.train(argparse.Namespace(arm=a.arm, seed=a.seed, steps=8000, batch=256, eval=500, lr=3e-4, wd=0.05, warm=300, smoke=False))


def check(a):
    import pandas as pd
    P = {k: np.load(OUT / "pred" / f"{k}.npz") for k in ("A-s0", "A-s1", "B-s0", "B-s1")}
    Dh = pd.read_parquet(RR.W / "labels/held.parquet")
    Dn = pd.read_parquet(RR.R / "labels/navtest.parquet")
    th = (Dh.turn40 >= 20).values
    mn = ((Dn.turn40 >= 20) & (Dn.jdist <= 20)).values

    def held(p):
        return float(np.nanmedian(np.abs(p[:, 4:6] - Dh[["L20", "R20"]].values)[th]))

    def near(p):
        return float(np.nanmedian(np.abs(p[:, 4:6] - Dn[["L20", "R20"]].values)[mn]))
    A = 0.5 * (P["A-s0"]["held"] + P["A-s1"]["held"])
    B = 0.5 * (P["B-s0"]["navtest"] + P["B-s1"]["navtest"])
    g = dict(held_turn20=held(A), held_seeds=[held(P[f"A-s{s}"]["held"]) for s in (0, 1)], navtest_near=near(B),
             navtest_seeds=[near(P[f"B-s{s}"]["navtest"]) for s in (0, 1)], held_range=[0.850, 1.046], navtest_range=[1.126, 1.398])
    g["pass"] = bool(0.850 <= g["held_turn20"] <= 1.046 and 1.126 <= g["navtest_near"] <= 1.398)
    (OUT / "g_repro_c.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", required=True, choices=("A", "B")), p.add_argument("--seed", type=int, default=0)
    sub.add_parser("check")
    a = ap.parse_args()
    dict(train=train, check=check)[a.cmd](a)
