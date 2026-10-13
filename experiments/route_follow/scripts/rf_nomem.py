#!/usr/bin/env python3
"""ROUTE-FOLLOW closed-loop reliance (route-night prereg Amendment 6, experiment 2): the same checkpoint read with the route memory fed empty
(SH30_RMEM=0) against its stored read, per scene. Box, AlpaSim's venv (pandas).

  rf_nomem.py --runs LABEL=<run dir>[,..] ... --pairs A:B ... [--did A:B,C:D ...] [--offroad DIR ..] --out DIR   -> DIR/nomem.md, nomem.json
Per label (its own scenes): n, mean score, zeros, zeros with offroad / left corridor / at-fault collision, at-fault zeros (offroad or
collision at fault), mean GT and lateral GT distance, inside-of-right-turn offroad zeros (rf_offroad held.parquet in --offroad: inside &
turn20 < 0). Per pair A - B on the scenes both labels scored: paired differences (counts summed, distances mean), 95 % CI from a scene
bootstrap and a log-clustered bootstrap (log = scene id minus its last '-' field), B = 10 000. --did (A - B) - (C - D) on the scenes all four
scored, same two bootstraps.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ("score", "zero", "z_off", "z_cor", "z_col", "z_atf", "gt", "latgt", "z_inright")
SUM = {"zero", "z_off", "z_cor", "z_col", "z_atf", "z_inright"}


def load(spec, inr):
    lab, dirs = spec.split("=", 1)
    rows = {}
    for d in dirs.split(","):
        for r in json.loads((Path(d) / "aggregate/results-summary.json").read_text())["rollouts"]:
            if r.get("score") is None:
                continue
            m, s = r.get("metrics") or {}, r["score"]
            z = s == 0
            rows[r["clipgt_id"]] = dict(scene=r["clipgt_id"], score=s, zero=int(z), z_off=int(z and (m.get("offroad") or 0) > 0),
                                        z_cor=int(z and (m.get("left_corridor_laterally") or 0) > 0), z_col=int(z and (m.get("collision_at_fault") or 0) > 0),
                                        z_atf=int(z and (m.get("offroad_or_collision_at_fault") or 0) > 0), gt=m.get("dist_to_gt_trajectory"),
                                        latgt=m.get("lateral_dist_to_gt_trajectory"), z_inright=int((lab, r["clipgt_id"]) in inr))
    return lab, pd.DataFrame(rows.values()).set_index("scene")


def boot(D, logs, B=10000, seed=0):
    """D: scenes x columns of per-scene paired differences -> estimate, scene CI, log CI (sums for counts, NaN-skipping means otherwise)."""
    rng = np.random.default_rng(seed)
    X = D.to_numpy(float)
    V, X0 = np.isfinite(X), np.nan_to_num(X)
    issum = np.array([c in SUM for c in D.columns])
    stat = lambda W: np.where(issum, W @ X0, (W @ X0) / np.maximum(W @ V, 1e-12))  # noqa: E731   W: draws x scenes
    n = len(D)
    est = stat(np.ones((1, n)))[0]
    sc = stat(np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float))
    L, li = np.unique(logs, return_inverse=True)
    lg = stat(np.stack([np.bincount(rng.integers(0, len(L), len(L)), minlength=len(L))[li] for _ in range(B)]).astype(float))
    ci = lambda S, j: [float(np.percentile(S[:, j], 2.5)), float(np.percentile(S[:, j], 97.5))]  # noqa: E731
    return {c: dict(d=float(est[j]), ci_scene=ci(sc, j), ci_log=ci(lg, j)) for j, c in enumerate(D.columns)}


def main(a):
    inr = set()
    for d in a.offroad:
        h = pd.read_parquet(Path(d) / "held.parquet")
        h = h[h.get("inside", False).fillna(False).astype(bool) & (h.turn20 < 0)]
        inr |= set(zip(h.label, h.scene))
    T = dict(load(s, inr) for s in a.runs)
    res = dict(labels={}, pairs={}, did={})
    md = ["# Closed-loop reliance: route memory fed empty (SH30_RMEM=0) vs the stored read", "",
          "| label | n | mean | zeros | offroad / corridor / collision | at-fault | GT m | lat GT m | inside-right |", "|:--|--:|--:|--:|:--|--:|--:|--:|--:|"]
    for lab, X in T.items():
        r = dict(n=len(X), mean=float(X.score.mean()), **{c: int(X[c].sum()) for c in SUM}, gt=float(X["gt"].mean()), latgt=float(X["latgt"].mean()))
        res["labels"][lab] = r
        md.append(f"| {lab} | {r['n']} | {r['mean']:.4f} | {r['zero']} | {r['z_off']} / {r['z_cor']} / {r['z_col']} | {r['z_atf']} | {r['gt']:.3f} | "
                  f"{r['latgt']:.3f} | {r['z_inright']} |")
    f = lambda v, nd: f"{v['d']:+.{nd}f} [{v['ci_scene'][0]:+.{nd}f}, {v['ci_scene'][1]:+.{nd}f}] ({v['ci_log'][0]:+.{nd}f}, {v['ci_log'][1]:+.{nd}f})"  # noqa: E731
    hdr = ["| pair | n | mean | zeros | offroad z | corridor z | collision z | at-fault | GT m | lat GT m | inside-right |", "|:--|--:|" + "--|" * 9]
    md += ["", "Paired differences: estimate [scene-bootstrap 95 % CI] (log-clustered 95 % CI), B = 10 000.", ""] + hdr

    def row(name, D, key):
        logs = np.array([s.rsplit("-", 1)[0] for s in D.index])
        b = boot(D, logs)
        res[key][name] = dict(n=len(D), **b)
        md.append(f"| {name} | {len(D)} | {f(b['score'], 4)} | {f(b['zero'], 1)} | {f(b['z_off'], 1)} | {f(b['z_cor'], 1)} | {f(b['z_col'], 1)} | "
                  f"{f(b['z_atf'], 1)} | {f(b['gt'], 3)} | {f(b['latgt'], 3)} | {f(b['z_inright'], 1)} |")
    for p in a.pairs:
        A, B_ = p.split(":")
        sc = T[A].index.intersection(T[B_].index)
        row(f"{A} - {B_}", (T[A].loc[sc, list(COLS)].astype(float) - T[B_].loc[sc, list(COLS)].astype(float)), "pairs")
    if a.did:
        md += ["", "Difference of differences (A - B) - (C - D) on the scenes all four scored.", ""] + hdr
    for q in a.did:
        (A, B_), (C, D_) = (x.split(":") for x in q.split(","))
        sc = T[A].index.intersection(T[B_].index).intersection(T[C].index).intersection(T[D_].index)
        g = lambda k: T[k].loc[sc, list(COLS)].astype(float)  # noqa: E731
        row(f"({A} - {B_}) - ({C} - {D_})", (g(A) - g(B_)) - (g(C) - g(D_)), "did")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nomem.json").write_text(json.dumps(res, indent=1))
    (out / "nomem.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--did", nargs="*", default=[])
    ap.add_argument("--offroad", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
