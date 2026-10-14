#!/usr/bin/env python3
"""P4 training round 1 (prereg experiments/route_follow/plans/2026-10-14-p4-self-branch-prereg.md, Amendments 3-4): line L1 reads. CPU, any
python with numpy / pandas and this repo on the path.

  l1a  --reads LABEL=<read.json> ... --rows <branches.json> --out FILE.md [--json FILE]
       PAI identity rollouts (rf_p4.py read; tag `id`, P2's completion rule `done`) of each checkpoint on the 27 P4 train turns: completed turns
       per label, split into the turns that supplied rows (scenes of the pool's G-shape-passing branches) and the others; score and zero kind
       per scene; the per-scene matrix.
  l1b  --follow <rf_follow scenes.csv> --mean NAME=LABEL,LABEL ... --pairs A:B ... --out FILE.md [--json FILE]
       held 460 turn completion (rf_follow: `turn` scenes, `completed`), seed-mean labels = per-scene mean of the seeds, pairs A - B on the
       common turn scenes with a log-clustered bootstrap 95 % CI (jevdrive.stats.paired, log = rf_follow's `log` column).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import stats  # noqa: E402


def l1a(a):
    rows = json.load(open(a.rows))["branches"]
    rs = sorted({b["scene"] for b in rows if b["g_shape"]})
    L = {}
    for kv in a.reads:
        lab, f = kv.split("=", 1)
        L[lab] = {r["scene"]: r for r in json.load(open(f))["rows"] if r["tag"] == "id"}
    sc = sorted(set.union(*(set(v) for v in L.values())))
    out = dict(row_scenes=rs, labels={})
    md = ["# L1a: PAI identity completion on the 27 P4 train turns (P2's rule)", "",
          f"Row turns (scenes of the pool's branches): {len(rs)}; other turns: {len(sc) - len(rs)}.", "",
          "| label | scenes read | completed | on row turns | on other turns | zeros | mean score |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for lab, m in L.items():
        d = {s: bool(m[s].get("done")) for s in m}
        n_r, n_o = sum(d.get(s, False) for s in rs), sum(v for s, v in d.items() if s not in rs)
        z = sum((m[s].get("score") or 0) == 0 for s in m)
        out["labels"][lab] = dict(n=len(m), done=sum(d.values()), done_rows=n_r, done_other=n_o, zeros=z,
                                  mean=float(np.mean([m[s].get("score") or 0 for s in m])), per_scene=d)
        md.append(f"| {lab} | {len(m)} | {sum(d.values())} | {n_r} / {len(rs)} | {n_o} / {len(m) - sum(s in m for s in rs)} | {z} | {out['labels'][lab]['mean']:.4f} |")
    md += ["", "Per scene (C = completed, z = zero score with its kind, . = neither; * = row turn):", "",
           "| scene | " + " | ".join(L) + " |", "|:--|" + "--:|" * len(L)]
    for s in sc:
        cells = []
        for m in L.values():
            r = m.get(s)
            cells.append("-" if r is None else "C" if r.get("done") else f"z {r.get('zero')}" if (r.get("score") or 0) == 0 else ".")
        md.append(f"| {s[7:15]}{'*' if s in rs else ''} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(md) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print("\n".join(md))


def l1b(a):
    import pandas as pd
    S = pd.read_csv(a.follow)
    S = S[S.turn.astype(str) == "True"]
    S["c"] = S.completed.astype(str).eq("True").astype(float)
    P = S.pivot_table(index="scene", columns="label", values="c")
    logs = S.drop_duplicates("scene").set_index("scene")["log"]
    for m in a.mean:
        name, labs = m.split("=", 1)
        P[name] = P[labs.split(",")].mean(1)
    out, md = {}, ["# L1b: held 460 turn completion (rf_follow), paired on common turn scenes, log-clustered 95 % CI", "",
                   "| label | turns | completion |", "|:--|--:|--:|"]
    for c in P.columns:
        v = P[c].dropna()
        md.append(f"| {c} | {len(v)} | {v.mean():.3f} ({v.sum():g}) |")
    md += ["", "| pair A - B | turns | difference (pp) [95 % CI] |", "|:--|--:|:--|"]
    for pr in a.pairs:
        A, B = pr.split(":")
        X = P[[A, B]].dropna()
        r = stats.paired(X[A].values, X[B].values, logs.loc[X.index].values)
        out[pr] = r
        md.append(f"| {A} - {B} | {r['n']} | {stats.fmt(r, '+.1f', 100)} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("l1a")
    p.add_argument("--reads", nargs="+", required=True), p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True), p.add_argument("--json", default="")
    p = sub.add_parser("l1b")
    p.add_argument("--follow", required=True), p.add_argument("--mean", nargs="*", default=[]), p.add_argument("--pairs", nargs="*", default=[])
    p.add_argument("--out", required=True), p.add_argument("--json", default="")
    a = ap.parse_args()
    dict(l1a=l1a, l1b=l1b)[a.cmd](a)
