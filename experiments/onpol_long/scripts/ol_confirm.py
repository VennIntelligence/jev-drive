#!/usr/bin/env python3
"""ONPOL-LONG Amendment 5 confirmation reads: checkpoints against a base on scene sets other than held 460 (box, AlpaSim's venv).

  ol_confirm.py --runs <tag>@<set>=<run dir>[,<run dir>] ... --base TAG --out DIR
      per set (train18 = the 1031 train scenes at 18 decisions, which supplied the rows: a guardrail, not clean evidence; all10 = all 1491 public
      scenes at 10 decisions): per label mean score, zeros by kind, at-fault zeros, GT distance; per checkpoint against the base on the common scenes
      with a rollout (route-fold scenes dropped): paired difference, log-clustered 95 % CI (10 000 draws). -> DIR/confirm.md, confirm.json
  --mean NAME=TAG,TAG   (Amendment 10) a seed-mean label: per-scene mean of the tags (zeros then count halves); --pairs A:B .. extra pairs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/alpasim_long/scripts"))
import long_read as LR  # noqa: E402


def main(a):
    import pandas as pd
    out = Path(a.out)
    LR.collect(argparse.Namespace(runs=a.runs, out=str(out), jobs=a.jobs, scenes=None))
    S = pd.read_csv(out / "per_scene.csv")
    S["tag"], S["set"] = S.label.str.split("@").str[0], S.label.str.split("@").str[1]
    z = S.score == 0
    S["zero"], S["z_atf"] = z.astype(int), (z & (S.offroad_or_collision_at_fault > 0)).astype(int)
    S["z_col"], S["z_off"], S["z_cor"] = ((z & (S[c] > 0)).astype(int) for c in ("collision_at_fault", "offroad", "left_corridor_laterally"))
    S["gt"] = S.dist_to_gt_trajectory
    md, res = ["# ONPOL-LONG Amendment 5: confirmation reads against " + a.base, ""], {}
    for st, X in S.groupby("set"):
        tags = sorted(X.tag.unique(), key=lambda t: t != a.base)
        sc = set.intersection(*[set(X[(X.tag == t) & X.offroad_or_collision_at_fault.notna()].scene) for t in tags])
        X = X[X.scene.isin(sc)]
        logs = pd.Series({x: x.rsplit("-", 1)[0] for x in sc})
        L = np.array(sorted(logs.unique()))
        P = {c: X.pivot(index="scene", columns="tag", values=c) for c in ("score", "zero", "z_atf", "z_col", "z_off", "z_cor", "gt")}
        ms = dict(m.split("=", 1) for m in a.mean)
        ms = {n: v.split(",") for n, v in ms.items() if set(v.split(",")) <= set(tags)}
        for n, v in ms.items():
            for c in P:
                P[c][n] = P[c][v].mean(1, skipna=c != "gt")
        tags += list(ms)
        li = np.searchsorted(L, logs.reindex(P["score"].index).to_numpy())
        W = np.stack([np.bincount(d, minlength=len(L)) for d in np.random.default_rng(0).integers(0, len(L), (10000, len(L)))])[:, li]
        md += [f"## {st}: {len(sc)} scenes, {len(L)} logs", "", "| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) |",
               "|:--|--:|--:|:--|--:|--:|"]
        md += [f"| {t} | {P['score'][t].mean():.4f} | {P['zero'][t].sum():g} | {P['z_col'][t].sum():g} / {P['z_off'][t].sum():g} / {P['z_cor'][t].sum():g} | "
               f"{P['z_atf'][t].sum():g} | {P['gt'][t].mean():.3f} |" for t in tags]
        md += ["", f"| a - {a.base} | mean score | zeros | at-fault zeros | GT distance (m) |", "|:--|:--|:--|:--|:--|"]
        res[st] = {}
        for t, b in [(t, a.base) for t in tags[1:]] + [tuple(q.split(":")) for q in a.pairs if set(q.split(":")) <= set(tags)]:
            r = {}
            for c, tot in (("score", False), ("zero", True), ("z_atf", True), ("gt", False)):
                d = (P[c][t] - P[c][b]).to_numpy() * (len(sc) if tot else 1)
                ok = np.isfinite(d)
                bs = (W[:, ok] @ d[ok]) / W[:, ok].sum(1)
                r[c] = [float(d[ok].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            res[st][t if b == a.base else f"{t} - {b}"] = r
            f = lambda v, p=4: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"  # noqa: E731
            md.append(f"| {t}{'' if b == a.base else f' - {b}'} | {f(r['score'])} | {f(r['zero'], 1)} | {f(r['z_atf'], 1)} | {f(r['gt'], 3)} |")
        md.append("")
    if "all10" in res:
        g = {t: v["score"][0] >= -0.02 for t, v in res["all10"].items() if " - " not in t}
        md += [f"Guardrail (Amendment 5): all 1491 at 10 decisions not more than 0.02 below {a.base} on the public mean: {g}.", ""]
        res["guardrail_all10"] = g
    md += ["train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts "
           "(ot3 / cf1 manifests); 12 route-fold scenes have no rollout.", ""]
    (out / "confirm.md").write_text("\n".join(md))
    (out / "confirm.json").write_text(json.dumps(res, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True), ap.add_argument("--base", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--mean", nargs="*", default=[]), ap.add_argument("--pairs", nargs="*", default=[])
    main(ap.parse_args())
