#!/usr/bin/env python3
"""STEER-LONG speed-history counterfactual read (amendment 2, item 1), Mac. Inputs: sl_cf.py outputs (tmp/steer_long/cf/<label>.pkl),
sl_pull.py copies (stored plans, for the gate), zeros_mech per_decision / per_rollout (log stations, event decision).

  sl_cf_read.py [--root tmp/steer_long] [--out experiments/steer_long/results]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
T0, DT = 17_000, 500_000


def v12x(P):
    P = np.asarray(P, float)
    return float(np.hypot(*(P[2, :2] - P[1, :2])) + np.hypot(*(P[3, :2] - P[2, :2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "tmp/steer_long"))
    ap.add_argument("--out", default=str(REPO / "experiments/steer_long/results"))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    PD = pd.read_parquet(root / "per_decision.parquet")
    PR = pd.read_parquet(root / "per_rollout.parquet").set_index(["label", "scene"])
    rows, gate = [], [0, 0]
    for f in sorted((root / "cf").glob("*.pkl")):
        lab = f.stem
        X = pickle.load(open(f, "rb"))
        stored = {}
        for line in gzip.open(root / "nuplan" / lab / "drive.jsonl.gz", "rt"):
            r = json.loads(line)
            stored[(r["scene"], r["now"])] = np.asarray(r["poses"], float)
        for scene, recs in X.items():
            g = PD[(PD.label == lab) & (PD.scene == scene)].sort_values("k")
            if len(g) < 3:
                continue
            t = g.t_us.to_numpy(float)
            s = np.maximum.accumulate(g.s_log_t.to_numpy(float))
            vlog_t = np.gradient(s, t / 1e6)
            su, iu = np.unique(s, return_index=True)
            kev = int(PR.loc[(lab, scene)].k_ev) if (lab, scene) in PR.index else 99
            for r in recs:
                if "F" not in r:
                    continue
                k = int(round((r["now"] - T0) / DT))
                st = stored.get((scene, r["now"]))
                if st is not None:
                    gate[0] += 1
                    gate[1] += float(np.hypot(*(st[7, :2] - r["F"][7, :2]))) <= 0.10
                if k < 3 or k >= kev or k not in set(g.k):
                    continue
                se = float(g.s_e[g.k == k].iloc[0])
                tq = float(np.interp(se, su, t[iu]))                      # when the log passed the ego's station
                v_ref = float(np.interp(tq + 1.5e6, t, vlog_t))            # the log's speed 1.5 s after that
                if v_ref < 2 or se > s[-1] - 1:
                    continue
                rows.append(dict(label=lab, scene=scene, k=k, v_ref=v_ref, v_ego=float(r["v_ego_keys"][-1]), vF=v12x(r["F"]), vA=v12x(r["A"]),
                                 vB=v12x(r["B"]), vlog_now=float(r["v_log_keys"][-1])))
    D = pd.DataFrame(rows)
    D.to_csv(out / "cf_rows.csv", index=False)
    L = ["# STEER-LONG speed-history counterfactual (amendment 2 item 1)", "",
         f"G-rep: replayed F plan within 0.10 m of the stored plan at 4 s: {gate[1]} / {gate[0]} ({100 * gate[1] / max(gate[0], 1):.2f} %)", "",
         f"Decisions read (zero rollouts, k >= 3, before the event, log moving >= 2 m/s 1.5 s ahead): {len(D)}, scenes {D.scene.nunique()}", "",
         "| label | n | ego / log now | F plan 1-2 s / log | A | B | recovery A | recovery B |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    allr = []
    for lab, g in [*D.groupby("label"), ("all", D)]:
        rF, rA, rB = (g.vF / g.v_ref).median(), (g.vA / g.v_ref).median(), (g.vB / g.v_ref).median()
        re = (g.v_ego / g.vlog_now.where(g.vlog_now > 0.5)).median()
        recA, recB = (rA - rF) / (1 - rF) if rF < 1 else np.nan, (rB - rF) / (1 - rF) if rF < 1 else np.nan
        L.append(f"| {lab} | {len(g)} | {re:.3f} | {rF:.3f} | {rA:.3f} | {rB:.3f} | {recA:+.2f} | {recB:+.2f} |")
        allr.append(dict(label=lab, rF=rF, rA=rA, rB=rB, recA=recA, recB=recB))
    # where the ego was already slower than the log
    slow = D[D.v_ego < 0.8 * D.vlog_now]
    if len(slow):
        rF, rA, rB = (slow.vF / slow.v_ref).median(), (slow.vA / slow.v_ref).median(), (slow.vB / slow.v_ref).median()
        L += ["", f"Subset ego < 0.8 x log now ({len(slow)} decisions): F {rF:.3f}, A {rA:.3f}, B {rB:.3f}; recovery A {(rA - rF) / (1 - rF):+.2f}, B {(rB - rF) / (1 - rF):+.2f}"]
    rec = [x for x in allr if x["label"] == "all"][0]["recB"]
    L += ["", f"Registered line on B (all): recovery {rec:+.2f} -> " + ("history feedback explains the slowness" if rec >= 0.5 else "not the history" if rec <= 0.2 else "partial")]
    (out / "sl_cf.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
