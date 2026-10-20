#!/usr/bin/env python3
"""PAI-FIX: the targeted PAI exam (stop overshoot + route turns, small launch / lead guard). Plan: ../plans/2026-10-20-pai-fix-prereg.md.

  facts   (box, repo .venv)  pai_fix.py facts --pkl <rf_tax pkl> --runs <read root>... --out <csv>
          per scene: score, failure class (decision 381's rules: rf_tax.classify, pai281.ptype), the human's facts from the log
          (pai_gap.human / rest_cue: handoff speed, stop, ends still, lead at the rest point, turn), driver RPC flag
  select  (Mac)  pai_fix.py select --facts <val160 facts csv>... --out experiments/pai_fix/scenes
          the exam lists from the registered rules (selection rules below), written once and committed before any exam read
  report  pai_fix.py report --read NAME=<facts csv> ... [--ref NAME] --out <md>      per-subset metrics of exam reads (paired against --ref)
  calib   (Mac)  pai_fix.py calib --inv tmp/pai0/inv.json --gap <pai_gap scenes.csv> --out <dir>
          stored full PAI 281 reads: does the 281-derived exam (the same rules on test281) rank checkpoints like the full read; noise floor

Failure classes of a scene (one each): overshoot = zero, corridor past the end of the log while the human ends still (cor_route_end);
route-turn = zero, corridor at a route turn (cor_straight_at_turn / cor_turn_under / cor_turn_over / cor_wrong_exit); offroad = zero offroad
(junction / curve / straight); AF-front; side = AF-side / AF-rear; zero-other (drift, lane change, other); partials (0 < s < 0.98): rear-hit,
slow (slow / creep / no-launch), hit. Launch scenes (the human is below 1 m/s at the handoff and the log moves > 5 m): launched = the ego
covers >= 2 m and >= 30 % of the log's distance.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/defect_fix/scripts"),
                str(REPO / "experiments/wait_cap/scripts")]

TURN_SUB = ("cor_straight_at_turn", "cor_turn_under", "cor_turn_over", "cor_wrong_exit")
ZCLS = ["overshoot", "route-turn", "offroad", "AF-front", "side", "zero-other"]
PCLS = ["rear-hit", "slow", "hit"]
OK = 0.98


def fclass(score, ptype, primary):
    if score >= OK:
        return "ok"
    if score > 0:
        return {"creep": "slow", "no-launch": "slow"}.get(ptype, ptype if ptype in PCLS else "hit")
    if primary == "cor_route_end":
        return "overshoot"
    if primary in TURN_SUB:
        return "route-turn"
    if ptype.startswith("AF-front"):
        return "AF-front"
    if ptype.startswith("AF-"):
        return "side"
    if "offroad" in ptype:
        return "offroad"
    return "zero-other"


# ---------------------------------------------------------------- facts (box)
def _facts_job(job):
    import pai_gap as PG
    s, o = job
    out, _ = PG.scene_job((s, o, [], None))
    m = o["metrics"] or {}
    out.update(dist=float(m.get("dist_traveled_m") or 0.0), gt_dist=float(m.get("gt_dist_traveled_m") or 0.0))
    return out


def cmd_facts(a):
    import pai_gap as PG
    bad = set()
    for r in a.runs:
        bad |= PG.summaries(r)[1]
    A = {o["scene"]: o for o in pickle.load(open(a.pkl, "rb"))["rolls"] if "ctrl" in o}
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(_facts_job, sorted(A.items()), chunksize=4))
    T = pd.DataFrame(rows).set_index("scene")
    T["rpc"] = T.index.isin(bad)
    T["fclass"] = [fclass(s, p, q) for s, p, q in zip(T.score, T.ptype, T.primary.fillna(""))]
    T.to_csv(a.out)
    print(len(T), "scenes,", int(T.rpc.sum()), "RPC ->", a.out, dict(Counter(T.fclass)))


# ---------------------------------------------------------------- shared metrics
def launch_mask(F):
    return (F.v_hand < 1.0) & (F.gt_dist > 5.0)


def launched(F):
    return (F.dist >= 2.0) & (F.dist >= 0.3 * F.gt_dist)


def metrics(F):
    """F: facts rows of one read on one subset -> dict of the registered metrics."""
    c = Counter(F.fclass)
    L = launch_mask(F)
    return dict(n=len(F), mean=float(F.score.mean()) if len(F) else np.nan, zeros=int((F.score == 0).sum()),
                **{f"z_{k}": c[k] for k in ZCLS}, **{f"p_{k}": c[k] for k in PCLS},
                launch=f"{int((L & launched(F)).sum())}/{int(L.sum())}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("facts")
    f.add_argument("--pkl", required=True), f.add_argument("--runs", nargs="+", required=True), f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"facts": cmd_facts}[a.cmd](a)


if __name__ == "__main__":
    main()
