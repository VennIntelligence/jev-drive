#!/usr/bin/env python3
"""OTHERS-TURN mechanism table: on the 24 held turns, where each driver's zero falls (route station of its last decision before the zero:
before entry - 20 m / approach / in the turn = entry .. exit + 10 m / after), its entry speed, and how much of the route's turn angle its own
heading had covered by then (ego yaw change from the decision 20 m before the entry to the zero or to exit + 10 m, over the route's TH).
Reads $DATA_DIR/runs/others_turn/follow (rf_follow.py scenes.csv / decisions.parquet) and the drive records of the runs (anchor yaw). AlpaSim's venv.

  ot_turn_mech.py --follow <follow dir> --runs LABEL=<dir>[,<dir>] .. --out results/turn_mech.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R_ / "experiments/anchor_offset/scripts"))
from ao_e1 import drive_records  # noqa: E402

T0, DT = 17_000, 500_000


def main(a):
    F = Path(a.follow)
    S, D = pd.read_csv(F / "scenes.csv"), pd.read_parquet(F / "decisions.parquet")
    S = S[S.turn == True]  # noqa: E712
    yaw = {}
    for spec in a.runs:
        lab, dirs = spec.split("=", 1)
        for d in dirs.split(","):
            for sc, rr in drive_records(d).items():
                if sc in set(S.scene):
                    yaw.setdefault(lab, {})[sc] = {int((r["t0"] - T0) // DT): float(r["anchor"][2]) for r in rr}
    rows = []
    for _, r in S.iterrows():
        d = D[(D.label == r.label) & (D.scene == r.scene)].sort_values("k")
        Y = yaw.get(r.label, {}).get(r.scene)
        end = d[d.k <= r.kz] if r.kz >= 0 else d
        st_end = end.st.iloc[-1] if len(end) else np.nan
        where = ("-" if r.kz < 0 else "before" if st_end < r.entry - 20 else "approach" if st_end < r.entry else "turn" if st_end <= r.exit + 10
                 else "after")
        pre = d[d.st <= r.entry - 20]
        k0 = int(pre.k.iloc[-1]) if len(pre) else int(d.k.iloc[0])
        stop = end[end.st <= r.exit + 10]
        k1 = int(stop.k.iloc[-1]) if len(stop) else k0
        frac = np.nan
        if Y and k0 in Y and k1 in Y:
            dy = np.unwrap([Y[k] for k in sorted(Y) if k0 <= k <= k1])
            frac = float((dy[-1] - dy[0]) / np.radians(r.turn_deg))
        de = d[d.st <= r.entry]
        rows.append(dict(label=r.label, scene=r.scene, outcome=r.outcome, where=where, v_entry=float(de.v.iloc[-1]) if len(de) else np.nan, frac=frac))
    O = pd.DataFrame(rows)
    md = ["# OTHERS-TURN: where the 24 held turns fail, per driver", "",
          "where = route station of the last decision before the zero (before = < entry - 20 m, approach = entry - 20 .. entry, turn = entry .. exit + 10, "
          "after); heading covered = ego yaw change from the decision 20 m before the entry to the zero (or exit + 10 m) over the route's turn angle "
          "(1.0 = turned as much as the route); entry speed = the last decision at or before the entry (m/s).", "",
          "| label | completed | zero before / approach / turn / after | no zero | heading covered, completed (median) | heading covered, failed (median) | entry speed median |",
          "|:--|--:|:--|--:|--:|--:|--:|"]
    for lab, x in O.groupby("label", sort=False):
        c, f = x[x.outcome == "completed"], x[x.outcome != "completed"]
        w = x["where"].value_counts()
        md.append(f"| {lab} | {len(c)} / {len(x)} | {w.get('before', 0)} / {w.get('approach', 0)} / {w.get('turn', 0)} / {w.get('after', 0)} | {w.get('-', 0)} | "
                  f"{c.frac.median():.2f} | {f.frac.median():.2f} | {x.v_entry.median():.2f} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    O.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print("\n".join(md))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--follow", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
