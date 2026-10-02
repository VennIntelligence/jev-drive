"""Confidence selector between the shipped rollout and a history-aligned rollout (plans/2026-10-04-history-align-plan.md,
addendum 2, post hoc): per token keep the rule's plan iff its summed lateral position std over the knots t <= 4 s is lower.
Writes $DATA_DIR/runs/op_lb/<data>/preds/gimm-cinque_al-sel-<rule>__base.npz (scored by op_interp_score.sh). Any env with numpy.

  python experiments/skill_pack/scripts/hist_align_select.py --rule rot0 --data lb_navhard lb_navtest
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
ap = argparse.ArgumentParser()
ap.add_argument("--rule", required=True)
ap.add_argument("--data", nargs="+", default=["lb_navhard", "lb_navtest"])
a = ap.parse_args()
k = T_IDXS <= 4.0 + 1e-6
for data in a.data:
    R = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/op_lb" / data
    zb, zr = np.load(R / "plans/gimm@cinque.npz"), np.load(R / f"plans/gimm@cinque_al-{a.rule}.npz")
    pb, pr = np.load(R / "preds/gimm-cinque__base.npz"), np.load(R / f"preds/gimm-cinque_al-{a.rule}__base.npz")
    assert (zb["names"] == zr["names"]).all() and (pb["tokens"] == zb["names"]).all() and (pr["tokens"] == zb["names"]).all()
    pick = zr["plan_std"][:, k, 1].sum(1) < zb["plan_std"][:, k, 1].sum(1)
    poses = np.where(pick[:, None, None], pr["poses"], pb["poses"])
    out = R / f"preds/gimm-cinque_al-sel-{a.rule}__base.npz"
    np.savez(out, tokens=pb["tokens"], poses=poses, pick=pick)
    print(json.dumps({"data": data, "rule": a.rule, "n": int(len(pick)), "pick_rate": float(pick.mean()), "out": str(out)}))
