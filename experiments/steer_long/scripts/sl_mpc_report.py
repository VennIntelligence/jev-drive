#!/usr/bin/env python3
"""STEER-LONG amendment 4 report, Mac: the per-step MPC replay (sl_mpc.py, mpc_full.json), the force-GT handoff replay (sl_handoff.py,
ho_out.json) and the rate of violent lateral motion right after the handoff in every PAI rollout (rows.parquet from sl_read.py).

  sl_mpc_report.py [--root tmp/steer_long] [--out experiments/steer_long/results]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "tmp/steer_long"))
    ap.add_argument("--out", default=str(REPO / "experiments/steer_long/results"))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    L = ["# STEER-LONG amendment 4: high-speed steering jumps and the force-GT handoff", ""]
    R = pd.read_parquet(root / "rows.parquet", columns=["track", "label", "session", "t", "v", "u", "A_exec"])
    R["rel"] = (R.t - R.groupby("session").t.transform("min")) / 1e6 + 0.1
    E = pd.read_csv(out / "rollouts.csv").set_index("session")
    for tr, ho in (("nuplan", 0.5), ("pai", 1.7)):
        g = R[R.track == tr].groupby("session")
        d = pd.DataFrame(dict(v=g.apply(lambda x: x.v[(x.rel - ho).abs().idxmin()]),
                              a=g.apply(lambda x: x.A_exec[(x.rel > ho) & (x.rel <= ho + 1.5)].max())))
        d = d.join(E[["label", "score"]])
        d["vb"] = pd.cut(d.v, [0, 5, 10, 15, 20, 25, 40])
        t = d.groupby("vb", observed=False).agg(rollouts=("a", "size"), violent=("a", lambda x: int((x > 6).sum())),
                                                 violent_zero=("score", lambda s: int(((s == 0) & (d.loc[s.index, "a"] > 6)).sum())))
        L += [f"## {tr}: |vx wz| > 6 m/s^2 within 1.5 s after the force-GT handoff (t = {ho} s), by speed at the handoff", "",
              "| speed m/s | rollouts | violent | of which zero |", "|:--|--:|--:|--:|"]
        L += [f"| {i} | {r.rollouts} | {r.violent} | {r.violent_zero} |" for i, r in t.iterrows()]
        if tr == "pai":
            by = d.assign(h=d.a > 6).groupby("label").agg(zeros_handoff=("h", "sum"))
            L += ["", "Per label: " + ", ".join(f"{k} {int(v)}" for k, v in by.zeros_handoff.items())]
        L.append("")
    H = json.load(open(root / "ho_out.json"))
    L += ["## Handoff replay (force-GT phase replayed with the logged reference; |u| at the handoff step, rad)", "",
          "| rollout | logged | default | w_steer 20 | w_steer 50 | iter 200 | GT reference | plan at ego speed |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for h in H:
        k = h["steps"].index(1.7)
        A = h["arms"]
        L.append(f"| {h['label']} {h['session'][:8]} | {h['u_log'][k]:+.3f} | {A['default'][k]:+.3f} | {A['w_steer20'][k]:+.3f} | {A['w_steer50'][k]:+.3f} | "
                 f"{A['iter200'][k]:+.3f} | {A['gt_ref'][k]:+.3f} | {A['plan_vego'][k]:+.3f} |")
    X = json.load(open(root / "mpc_full.json"))
    ctl = [x for x in X if x["kind"] == "control"]
    dif = {arm: np.median([np.max(np.abs(np.array(x["arms"][arm]["u"]) - np.array(x["arms"]["default"]["u"]))) for x in ctl]) for arm in ("w_steer20", "w_steer50")}
    rep = [np.nanmax(np.abs(np.array(x["arms"]["default"]["u"][-6:]) - np.array([np.nan if u is None else u for u in x["u_log"][-6:]]))) for x in ctl]
    L += ["", f"Controls (15 clean PAI rollouts, v >= 20 m/s, 2 s of steps, step-by-step replay on logged states): default reproduces the logged command "
          f"to median {np.median(rep):.4f} rad; max |u_arm - u_default| median: w_steer20 {dif['w_steer20']:.4f}, w_steer50 {dif['w_steer50']:.4f} rad."]
    (out / "sl_mpc.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
