#!/usr/bin/env python3
"""PNX secondary readout (PN prereg supplement 3, not a gate): frozen board-mix PCS (decision 382) of each arm's native arena held read
against its parent, with slow-partial and near-1 counts. Reuses defect_fix/scripts/slow_rescore.py unchanged (Frozen, classes, boot_w:
same definitions, same frozen item fits boardmix / g15, log-clustered bootstrap with items fixed).

Mac, pcs_local env:
  tmp/pcs_local/.venv/bin/python experiments/practice_nuplan/scripts/pnx_pcs.py --rows tmp/pnx/rows.csv.gz [--native-gap tmp/native_gap]
      -> experiments/practice_nuplan/results/pcs.{json,md}
rows = the summary job's `jt.py arena` collection ($DATA_DIR/runs/practice_nuplan/pnx/arena/rows.csv.gz).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/defect_fix/scripts")]
import slow_rescore as SR  # noqa: E402

PAIRS = [("NTVXS-L18-s0@off", "NTV0-L18-s0@off"), ("JTXS-L18-s1@off", "JTN-L18-s1@off"),
         ("JTXS-L18-s1@off", "NTV0-L18-s0@off"), ("NTVXS-L18-s0@off", "JTN-L18-s1@off")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True), ap.add_argument("--native-gap", default=str(REPO / "tmp/native_gap"))
    ap.add_argument("--out", default=str(REPO / "experiments/practice_nuplan/results"))
    a = ap.parse_args()
    S = pd.read_csv(Path(a.native_gap) / "set.csv.gz").drop_duplicates("id").set_index("id")
    held = S.index[S.split == "held"]
    R = pd.read_csv(a.rows).drop_duplicates(["tag", "id"]).set_index(["tag", "id"])
    rd = {t: g.droplevel(0) for t, g in R.groupby(level=0)}
    sc_pool, _, _ = SR.PL.load(REPO / "experiments/alpasim_proxy/results/pcs_local")
    FZ = {k: SR.Frozen(p, sc_pool) for k, p in SR.FITS.items()}
    items = FZ["boardmix"].items
    vec = lambda t: rd[t].score.reindex(items).to_numpy(float)  # noqa: E731
    rng, res = np.random.default_rng(0), []
    L = ["# PNX: frozen board-mix PCS on the native arena held read (secondary, not a gate)", "",
         "slow_rescore.py's definitions (decision 382 fits, items frozen; one read = one more submission; CI = log-clustered bootstrap of the "
         "paired PCS difference, 2 000 draws). Slow = 0 < score < 0.98, type slow / creep / no-launch, non-static, non-launch stratum; "
         "launch-short = the same in the launch stratum; near-1 = 0.98 <= score < 1.", "",
         "| A - B | windows | mean diff [CI] | PCS board-mix A / B, diff [CI] | PCS g15 A / B, diff [CI] | zeros A / B | slow A / B | launch-short A / B | near-1 A / B | exact 1 A / B |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for ta, tb in PAIRS:
        A, B = rd[ta], rd[tb]
        ids = A.index.intersection(B.index).intersection(held)
        ids = ids[(A.type.reindex(ids) != "route-fail").values & (B.type.reindex(ids) != "route-fail").values]
        m = SR.stats.paired(A.score.reindex(ids).values, B.score.reindex(ids).values, S.log.reindex(ids).values)
        row = dict(pair=f"{ta} - {tb}", windows=len(ids), d_mean=m["mean"], d_lo=m["lo"], d_hi=m["hi"], a=SR.classes(A.loc[ids], S), b=SR.classes(B.loc[ids], S))
        ya, yb = vec(ta), vec(tb)
        sub = np.isfinite(ya) & np.isfinite(yb)
        codes, cnt = SR.boot_w(S.log.reindex(items).to_numpy()[sub], rng)
        for k, fz in FZ.items():
            y1, y2 = np.where(sub, ya, np.nan), np.where(sub, yb, np.nan)
            L1, L2 = fz.L(y1), fz.L(y2)
            w0 = sub.astype(float)
            W = np.repeat(w0[:, None], SR.NB, 1); W[sub] = cnt[codes]
            d = fz.pcs(L1, W) - fz.pcs(L2, W)
            lo, hi = np.quantile(d, [0.025, 0.975])
            p1, p2 = fz.pcs(L1, w0), fz.pcs(L2, w0)
            row[f"pcs_{k}"] = dict(a=float(p1), b=float(p2), d=float(p1 - p2), lo=float(lo), hi=float(hi), items=int(sub.sum()), of=len(items))
        res.append(row)
        ca, cb, pb, pg = row["a"], row["b"], row["pcs_boardmix"], row["pcs_g15"]
        L.append(f"| {row['pair']} | {len(ids)} | {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] | {pb['a']:.0f} / {pb['b']:.0f}, {pb['d']:+.0f} [{pb['lo']:+.0f}, {pb['hi']:+.0f}] | "
                 f"{pg['a']:.0f} / {pg['b']:.0f}, {pg['d']:+.0f} [{pg['lo']:+.0f}, {pg['hi']:+.0f}] | "
                 + " | ".join(f"{ca[c]} / {cb[c]}" for c in ("zeros", "slow", "launch_short", "near1", "exact1")) + " |")
        print(L[-1], flush=True)
    L += ["", f"Items scored: {res[0]['pcs_boardmix']['items']} of {res[0]['pcs_boardmix']['of']} frozen items (MAP on the read's items)."]
    out = Path(a.out)
    (out / "pcs.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "pcs.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
