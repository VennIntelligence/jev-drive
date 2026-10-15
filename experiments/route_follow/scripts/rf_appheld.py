#!/usr/bin/env python3
"""APPROACH COURSE held read (plans/2026-10-15-approach-course-prereg.md section 4, reported): the entry state of the held turns per label
from an rf_follow output dir (scenes.csv, decisions.parquet). CPU, AlpaSim venv (pandas).

  rf_appheld.py --follow DIR --out DIR [--pairs A@18:B@18 ..]  -> DIR/entry.{json,md}

Per label x turn scene (rf_follow's turn: >= 60 deg, entry at 10 % of the change, on that label's own route), decisions before the first
zero event only: v_entry = the speed where the station track first crosses the entry (rf_entryspeed.cross_speed: linear in station, else
the nearest decision within 10 m); lat_entry / lat_m20 = the signed lateral offset to the route at the entry / 20 m before it, + toward
the turn's inside (linear in station between decisions; NaN when the track does not reach the station before the zero). Pairs: scenes
that are turns in both labels; mean A - B with a log-clustered bootstrap (B 10 000, rng 0) and the median of the paired differences.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_entryspeed as ES  # noqa: E402


def at(st, x, s):
    st, x = np.asarray(st, float), np.asarray(x, float)
    i = np.flatnonzero(st >= s)
    if not len(i) or i[0] == 0:
        return np.nan
    j = i[0]
    w = (s - st[j - 1]) / max(st[j] - st[j - 1], 1e-9)
    return float((1 - w) * x[j - 1] + w * x[j])


def main(a):
    import pandas as pd
    S = pd.read_csv(Path(a.follow) / "scenes.csv")
    Dx = pd.read_parquet(Path(a.follow) / "decisions.parquet")
    S = S[S.turn == True]                                                          # noqa: E712
    rows = []
    for _, s in S.iterrows():
        d = Dx[(Dx.label == s.label) & (Dx.scene == s.scene)].sort_values("k")
        if int(s.kz) >= 0:
            d = d[d.k <= int(s.kz)]
        sg = 1.0 if s.side == "L" else -1.0
        ent = float(s.entry)
        rows.append(dict(label=s.label, scene=s.scene, log=s.log, side=s.side, completed=bool(s.completed), outcome=s.outcome,
                         v_entry=ES.cross_speed(d.st, d.v, ent) if len(d) else np.nan, lat_entry=sg * at(d.st, d.lat, ent),
                         lat_m20=sg * at(d.st, d.lat, ent - 20.0)))
    R = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q = lambda x: [float(np.nanpercentile(x, p)) if np.isfinite(x).any() else np.nan for p in (25, 50, 75)]  # noqa: E731
    res = dict(labels={}, pairs={})
    for lb, g in R.groupby("label"):
        res["labels"][lb] = dict(turns=int(len(g)), completed=int(g.completed.sum()), v_entry=q(g.v_entry), lat_entry=q(g.lat_entry), lat_m20=q(g.lat_m20),
                                 v_entry_n=int(g.v_entry.notna().sum()), lat_entry_n=int(g.lat_entry.notna().sum()))
    for p in a.pairs:
        A, B = p.split(":")
        m = R[R.label == A].merge(R[R.label == B], on=["scene", "log"], suffixes=("_a", "_b"))
        res["pairs"][p] = {}
        for k in ("v_entry", "lat_entry", "lat_m20"):
            x = m[[f"{k}_a", f"{k}_b", "log"]].dropna()
            dd = (x[f"{k}_a"] - x[f"{k}_b"]).to_numpy()
            if not len(dd):
                res["pairs"][p][k] = dict(n=0)
                continue
            lg = x.log.to_numpy()
            ul = np.unique(lg)
            li = np.searchsorted(ul, lg)
            sm, n = np.bincount(li, dd, len(ul)), np.bincount(li, minlength=len(ul))
            C = np.random.default_rng(0).integers(0, len(ul), (10000, len(ul)))
            b = np.array([sm[c].sum() / max(n[c].sum(), 1) for c in C])
            res["pairs"][p][k] = dict(n=int(len(dd)), mean=float(dd.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                                      median=float(np.median(dd)))
    R.to_csv(out / "entry_rows.csv", index=False)
    (out / "entry.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda v: " / ".join(f"{x:.2f}" for x in v)  # noqa: E731
    md = ["# Held turns: entry state per label (rf_appheld.py)", "", "Decisions before the first zero; lateral + toward the turn's inside; q25 / median / q75.", "",
          "| label | turns | completed | v_entry m/s | lat_entry m | lat at entry - 20 m |", "|:--|--:|--:|:--|:--|:--|"]
    md += [f"| {k} | {v['turns']} | {v['completed']} | {f(v['v_entry'])} (n {v['v_entry_n']}) | {f(v['lat_entry'])} (n {v['lat_entry_n']}) | {f(v['lat_m20'])} |"
           for k, v in res["labels"].items()]
    if res["pairs"]:
        md += ["", "| pair | read | n | mean A - B [95 % CI] | median diff |", "|:--|:--|--:|:--|--:|"]
        for p, v in res["pairs"].items():
            for k, x in v.items():
                md.append(f"| {p} | {k} | {x['n']} | " + (f"{x['mean']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] | {x['median']:+.2f} |" if x["n"] else "- | - |"))
    (out / "entry.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--follow", required=True), ap.add_argument("--out", required=True), ap.add_argument("--pairs", nargs="*", default=[])
    main(ap.parse_args())
