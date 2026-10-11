#!/usr/bin/env python3
"""ALPASIM-LONG read-out (plans/2026-10-12-long-prereg.md): per-scene and per-decision metrics of AlpaSim runs, and the 10- against
18-decision comparison. Box, AlpaSim's venv (numpy, pandas, polars).

  long_read.py collect --runs <label>=<run dir>[,<run dir>] ... --out DIR [--jobs 32]
      DIR/per_scene.csv  (label, scene, score, the scored flags, GT distance as aggregated, driven / logged metres, decisions,
                          GT distance rebuilt from the per-step series over the whole run and cut at k <= 10)
      DIR/per_scene_k.parquet (label, scene, k, dist_gt): the evaluator's untruncated dist_to_gt_trajectory at t = 0.017 s + 0.5 s k
  long_read.py report --in DIR [DIR ...] --out <md> [--json <json>] [--pairs A:B ...]
      label = <tag>@<length>, e.g. YR10m10-F-s0@18; recipe = tag without -s<seed>.

GT distance (the leaderboard's) = mean over scenes of the per-scene max of dist_to_gt_trajectory over the steps the evaluator keeps (up to
and including the first step at or beyond 4 m or with a collision or offroad). The rebuilt value is checked against the aggregate's own per
rollout; the same rule over k <= 10 gives the long run's first-10-decision GT distance, corridor zeros and at-fault events.
Drift slope = OLS slope over k = 4..K of the per-k mean distance to the logged path (aud1_drift_by_decision.md, all scenes).
CIs: log-clustered bootstrap over the held logs (10 000 draws), paired across labels on the same scenes.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

T0, DT = 17000, 500000
FLAGS = ("collision_at_fault", "offroad", "left_corridor_laterally", "offroad_or_collision_at_fault", "dist_to_gt_trajectory", "dist_traveled_m",
         "gt_dist_traveled_m")


def cut(x, kmax=None):
    """The evaluator's truncation on one rollout's per-k rows (k sorted): keep steps up to and including the first step with
    dist_to_gt >= 4 m or (collision_any or offroad) (aggregation/main.py and processing.py DEFAULT_MODIFIERS). Returns
    (GT distance = max dist over the kept steps, corridor zero, at-fault event (offroad or front / lateral collision)), optionally
    only over k <= kmax."""
    if kmax is not None:
        x = x[x.k <= kmax]
    d, ev, atf = x.dist_gt.to_numpy(float), x.ev.to_numpy(bool), x.atf.to_numpy(bool)
    hit = np.flatnonzero((d >= 4.0) | ev)
    n = hit[0] + 1 if len(hit) else len(d)
    return float(np.nanmax(d[:n])) if n else np.nan, bool((d[:n] >= 4.0).any() and not ev[:n].any()), bool(atf[:n].any())


def one(a):
    label, D, scene = a
    import polars as pl
    f = sorted(glob.glob(f"{D}/rollouts/{scene}/*/metrics.parquet"))
    if not f:
        return []
    names = ("dist_to_gt_trajectory", "collision_any", "offroad", "collision_front", "collision_lateral")
    m = pl.read_parquet(f[0]).filter(pl.col("name").is_in(names)).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    ser = {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}
    g = lambda n, t: (ser.get(n, {}).get(t) or 0.0) > 0
    ks = sorted((t - T0) // DT for t in ser["dist_to_gt_trajectory"] if t >= T0 and (t - T0) % DT == 0)
    return [dict(label=label, scene=scene, k=int(k), dist_gt=ser["dist_to_gt_trajectory"][t], ev=g("collision_any", t) or g("offroad", t),
                 atf=g("offroad", t) or g("collision_front", t) or g("collision_lateral", t)) for k in ks for t in [T0 + DT * k]]


def collect(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for spec in a.runs:
        label, dirs = spec.split("=", 1)
        seen = set()
        for D in dirs.split(","):
            for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]:
                s = r["clipgt_id"]
                if s in seen:
                    continue
                seen.add(s)
                mm = r.get("metrics") or {}
                rows.append(dict(label=label, scene=s, run=D, score=r.get("score"), fail=r.get("failure_reason"), **{n: mm.get(n) for n in FLAGS}))
                jobs.append((label, D, s))
    with ProcessPoolExecutor(a.jobs) as ex:
        K = pd.DataFrame([r for rr in ex.map(one, jobs, chunksize=8) for r in rr])
    S = pd.DataFrame(rows)
    g = K.sort_values("k").groupby(["label", "scene"])
    full, c10 = g.apply(lambda x: cut(x), include_groups=False), g.apply(lambda x: cut(x, 10), include_groups=False)
    S = S.merge(pd.DataFrame(dict(decisions=g.k.max(), gt_rebuilt=full.str[0], gt_cut10=c10.str[0], corridor_cut10=c10.str[1],
                                  atfault_cut10=c10.str[2])).reset_index(), on=["label", "scene"], how="left")
    S.to_csv(out / "per_scene.csv", index=False)
    K.to_parquet(out / "per_scene_k.parquet")
    print(S.groupby("label").agg(n=("scene", "size"), score=("score", "mean"), dec_min=("decisions", "min"), dec_max=("decisions", "max"),
                                 gt=("dist_to_gt_trajectory", "mean"), gt_rebuilt=("gt_rebuilt", "mean")).to_string())


def report(a):
    import pandas as pd
    S = pd.concat([pd.read_csv(Path(d) / "per_scene.csv") for d in a.inp], ignore_index=True)
    K = pd.concat([pd.read_parquet(Path(d) / "per_scene_k.parquet") for d in a.inp], ignore_index=True)
    S["log"] = S.scene.str.rsplit("-", n=1).str[0]
    S["tag"], S["length"] = S.label.str.split("@").str[0], S.label.str.split("@").str[1].astype(int)
    S["recipe"] = S.tag.str.replace(r"-s\d+$", "", regex=True)
    K = K.merge(S[["label", "scene", "log", "score"]], on=["label", "scene"])
    scenes = sorted(set.intersection(*[set(x.scene) for _, x in S.groupby("label")]))
    S, K = S[S.scene.isin(scenes)], K[K.scene.isin(scenes)]
    logs = np.array(sorted(S.log.unique()))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(d, minlength=len(logs)) for d in rng.integers(0, len(logs), (10000, len(logs)))])

    def per_log(x, col):   # (sum, count) per log, in `logs` order
        g = x.groupby("log")[col].agg(["sum", "count"]).reindex(logs).fillna(0)
        return g["sum"].to_numpy(), g["count"].to_numpy()

    def boot(sums, cnts):
        return (W @ sums) / (W @ cnts)

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return f"[{lo:+.4f}, {hi:+.4f}]"

    def slope(y, k):
        x = k - k.mean()
        return float(((y - y.mean()) * x).sum() / (x * x).sum())

    tab, res = [], {}
    for lab, x in S.groupby("label", sort=True):
        kk = K[K.label == lab]
        Kmax = int(x.length.iloc[0])
        curve = kk[kk.k.between(4, Kmax)].groupby("k").dist_gt.mean()
        c10 = kk[kk.k.between(4, 10)].groupby("k").dist_gt.mean()
        z = x.score == 0
        r = dict(label=lab, n=len(x), decisions_median=float(x.decisions.median()), decisions_min=int(x.decisions.min()),
                 score=x.score.mean(), zeros=int(z.sum()), zero_atfault=int((z & (x.offroad_or_collision_at_fault > 0)).sum()),
                 zero_offroad=int((z & (x.offroad > 0)).sum()), zero_corridor=int((z & (x.left_corridor_laterally > 0)).sum()),
                 atfault_events=int((x.offroad_or_collision_at_fault > 0).sum()),
                 atfault_km=x.dist_traveled_m.sum() / max(1, (x.offroad_or_collision_at_fault > 0).sum()) / 1000,
                 gt_dist=x.dist_to_gt_trajectory.mean(), gt_rebuilt_match=float((abs(x.gt_rebuilt - x.dist_to_gt_trajectory) < 1e-6).mean()),
                 gt_cut10=x.gt_cut10.mean(), corridor_by_k10=int(x.corridor_cut10.sum()), atfault_by_k10=int(x.atfault_cut10.sum()),
                 slope_4_K=slope(curve.to_numpy(), curve.index.to_numpy()), slope_4_10=slope(c10.to_numpy(), c10.index.to_numpy()),
                 driven_m=x.dist_traveled_m.mean(), logged_m=x.gt_dist_traveled_m.mean())
        tab.append(r)
    T = pd.DataFrame(tab)
    # recipe x length seed means and paired recipe differences
    R = S.groupby(["recipe", "length", "scene", "log"]).agg(score=("score", "mean"), gtd=("dist_to_gt_trajectory", "mean")).reset_index()
    rec = []
    for (rc, L), x in R.groupby(["recipe", "length"]):
        rec.append(dict(recipe=rc, length=L, seeds=S[(S.recipe == rc) & (S.length == L)].tag.nunique(), score=x.score.mean(), gtd=x.gtd.mean()))
    RT = pd.DataFrame(rec)
    for col in ("score", "gtd"):
        RT[f"rank_{col}"] = RT.groupby("length")[col].rank(ascending=(col == "gtd")).astype(int)
    pairs = []
    recipes = sorted(R.recipe.unique())
    for L in sorted(R.length.unique()):
        for i, p in enumerate(recipes):
            for q in recipes[i + 1:]:
                A, B = R[(R.recipe == p) & (R.length == L)].set_index("scene"), R[(R.recipe == q) & (R.length == L)].set_index("scene")
                if A.empty or B.empty:
                    continue
                d = (A[["score", "gtd"]] - B[["score", "gtd"]]).join(A["log"]).reset_index()
                for col in ("score", "gtd"):
                    s_, c_ = per_log(d, col)
                    pairs.append(dict(length=L, a=p, b=q, metric=col, diff=d[col].mean(), ci=ci(boot(s_, c_))))
    P = pd.DataFrame(pairs)
    md = ["## Per checkpoint and length", "", T.to_markdown(index=False, floatfmt=".4f"), "",
          "## Recipes (seed mean per scene), ranks per length", "", RT.to_markdown(index=False, floatfmt=".4f"), "",
          "## Paired recipe differences a - b (log-clustered bootstrap, 10 000 draws)", "", P.to_markdown(index=False, floatfmt=".4f"), "",
          f"Scenes in every label: {len(scenes)}; logs: {len(logs)}."]
    Path(a.out).write_text("\n".join(md) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(labels=tab, recipes=rec, pairs=pairs, scenes=len(scenes)), indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--out", required=True), p.add_argument("--jobs", type=int, default=32)
    p.set_defaults(fn=collect)
    p = sub.add_parser("report")
    p.add_argument("--in", dest="inp", nargs="+", required=True), p.add_argument("--out", required=True), p.add_argument("--json")
    p.set_defaults(fn=report)
    a = ap.parse_args()
    a.fn(a)
