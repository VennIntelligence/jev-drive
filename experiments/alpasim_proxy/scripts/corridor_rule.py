#!/usr/bin/env python3
"""CORRIDOR RULE: when does the AlpaSim scorer's corridor (left_corridor_laterally) zero a rollout whose human stands still?
Measured from the scorer's own per-step metrics of stored reads (results/corridor_rule.md). Box, repo root, AlpaSim's venv ($P).

  $P corridor_rule.py extract --runs D [D ...] --tag T --out F.csv.gz   one row per rollout of each run dir (<D>/aggregate/
        metrics_unprocessed.parquet): the scorer's GT length, the first metric step whose dist_to_gt_trajectory >= 4 m (the time cutoff
        RemoveTimestepsAfterEvent keeps that step and drops the later ones), the ego's distance travelled and lateral distance there,
        the largest distance travelled before it, and whether the corridor fired
  $P corridor_rule.py gtend --set F --out F.csv.gz [--workers N]   per window of an arena / wait set: the human's last 10 Hz GT segment
        (length, cosine to the final heading; frames start + 108 -> start + 110 of the raw log), the scorer GT's shape at its end
  corridor_rule.py report --rows F [F ...] --arena-set F [--wait-set F] [--gtend F ...] --out F.md   tables (Mac or box); with --gtend
        the end-state table: "still" = the human's last 10 Hz GT segment < 1 mm (the corridor is a 4 m disc around the rest point),
        "moving" = >= 1 mm along the final heading (cos >= 0.9; overshooting the end is free), "off-heading" = the rest

Scorer (third_party alpasim src/eval, read 2026-10-18): GT = the human's recorded track of the scored scene (the 5.5 s trajdata window,
nothing after it), in the ego AABB-centre frame. Per metric step (0.5 s here): d = distance from the ego polygon centroid to the GT
polyline; lateral = d, except when the centroid projects onto the GT's far end (and the GT has length > 0 and a non-degenerate last
segment), then lateral = |offset from the GT end x unit heading of the last GT segment| (overshoot along the path is free). Aggregation
drops every step after the first with d >= max_dist_to_gt_trajectory (4.0 m), keeps that step, and left_corridor_laterally = max over
the kept steps of [lateral >= 4.0]; a corridor exit zeroes the scene.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

CORRIDOR = 4.0
COLS = ["dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory", "left_corridor_laterally", "dist_traveled_m", "gt_dist_traveled_m",
        "eval_relevant", "dist_to_gt_location"]


def cmd_extract(a):
    import pandas as pd
    import polars as pl
    out = []
    for D in a.runs:
        f = Path(D) / "aggregate/metrics_unprocessed.parquet"
        if not f.exists():
            continue
        m = pl.read_parquet(f, columns=["name", "timestamps_us", "values", "clipgt_id", "rollout_id"]).filter(pl.col("name").is_in(COLS))
        m = m.with_columns(pl.col("timestamps_us").cast(pl.Int64)).to_pandas()
        summ = {r["clipgt_id"]: r for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]}
        for (cid, rid), g in m.groupby(["clipgt_id", "rollout_id"]):
            p = g.pivot_table(index="timestamps_us", columns="name", values="values").sort_index()
            d, lat = p["dist_to_gt_trajectory"].values, p["lateral_dist_to_gt_trajectory"].values
            trav, rel = p["dist_traveled_m"].values, p["eval_relevant"].values > 0
            trip = np.flatnonzero(d >= CORRIDOR)
            k = int(trip[0]) if len(trip) else len(d) - 1
            r = summ.get(cid, {})
            out.append(dict(tag=a.tag, id=cid, run=str(D), score=float(r.get("score") or 0.0), fail=r.get("failure_reason") or "",
                            gt_len=float(p["gt_dist_traveled_m"].values[-1]), steps=len(d), first_rel_s=float(p.index[rel][0] / 1e6) if rel.any() else np.nan,
                            tripped=bool(len(trip)), trip_s=float(p.index[k] / 1e6) if len(trip) else np.nan,
                            d_trip=float(d[k]), lat_trip=float(lat[k]), trav_trip=float(trav[k]),
                            lat_ratio=float(lat[k] / d[k]) if d[k] > 0 else np.nan, d_max_kept=float(d[:k + 1].max()), lat_max_kept=float(lat[:k + 1].max()),
                            d_before=float(d[:k].max()) if k > 0 else 0.0, trav_end=float(trav[-1]), d_end=float(d[-1]),
                            corridor=int(p["left_corridor_laterally"].values[:k + 1].max() > 0)))
    pd.DataFrame(out).to_csv(a.out, index=False)
    print(json.dumps(dict(tag=a.tag, rollouts=len(out), corridor=int(sum(r["corridor"] for r in out)))))


def _gtend(job):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import wait_set as WS
    log, ws = job
    L, out = WS.Log(log), []
    for i, w0 in ws:
        xy = L.xy[w0:w0 + WS.WIN:2]
        d = np.diff(xy, axis=0)
        n = np.hypot(*d.T)
        h = L.yaw[w0 + WS.WIN - 1]
        out.append(dict(id=i, seg_last_mm=float(n[-1] * 1e3), seg_cos=float(d[-1] @ [np.cos(h), np.sin(h)] / n[-1]) if n[-1] > 0 else np.nan,
                        gt_len_raw=float(n.sum()), end_still_s=float(np.argmax(np.cumsum(n[::-1]) > 0.05) / 10.0) if n.sum() > 0.05 else 5.5))
    return out


def cmd_gtend(a):
    import pandas as pd
    from multiprocessing import Pool
    f = a.set
    if f.endswith(".json"):
        S = pd.DataFrame(json.load(open(f))["windows"])
    else:
        S = pd.read_csv(f)
    jobs = [(lg, list(zip(g.id, g.start_idx.astype(int)))) for lg, g in S.groupby("log")]
    with Pool(a.workers) as p:
        rows = [r for rr in p.map(_gtend, jobs) for r in rr]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(json.dumps(dict(windows=len(rows))))


def cmd_report(a):
    import pandas as pd
    R = pd.concat([pd.read_csv(f) for f in a.rows], ignore_index=True)
    S = pd.read_csv(a.arena_set).set_index("id")
    L = ["# corridor rule: per-step scorer metrics of stored reads", ""]
    A = R[R.id.isin(S.index)].copy()
    A["split"], A["stratum"], A["h_path"], A["h_maxdisp"] = (A.id.map(S[c]) for c in ("split", "stratum", "path", "maxdisp"))
    gb = [0, 0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 1e9]
    A["gt_bin"] = pd.cut(A.gt_len, gb, right=False)
    L += ["## Arena, static stratum (human max displacement < 1 m), by the scorer's GT length", "",
          "| tag | split | GT length (m) | windows | corridor zeros | tripped (d >= 4 m) | corridor / tripped | median lateral / d at the trip |", "|---|---|---|--:|--:|--:|--:|--:|"]
    for (t, sp), g in A[A.stratum == "static"].groupby(["tag", "split"]):
        for b, h in g.groupby("gt_bin", observed=True):
            tr = h[h.tripped]
            L.append(f"| {t} | {sp} | {b} | {len(h)} | {int(h.corridor.sum())} | {len(tr)} | {h.corridor.sum() / max(len(tr), 1):.2f} | "
                     f"{tr.lat_ratio.median() if len(tr) else float('nan'):.2f} |")
    L.append("")
    L += ["## Arena, every stratum: corridor zeros given a trip, by GT length (all tags pooled)", "",
          "| GT length (m) | windows | tripped | corridor | corridor / tripped | median lat/d at trip |", "|---|--:|--:|--:|--:|--:|"]
    for b, h in A.groupby("gt_bin", observed=True):
        tr = h[h.tripped]
        L.append(f"| {b} | {len(h)} | {len(tr)} | {int(h.corridor.sum())} | {h.corridor.sum() / max(len(tr), 1):.2f} | {tr.lat_ratio.median():.2f} |")
    L.append("")
    if a.gtend:
        G = pd.concat([pd.read_csv(f) for f in a.gtend]).drop_duplicates("id").set_index("id")
        E = R.join(G, on="id")
        E["end"] = np.where(E.seg_last_mm < 1, "still", np.where(E.seg_cos >= 0.9, "moving", "off-heading"))
        E["split"], E["stratum"] = E.id.map(S["split"]), E.id.map(S["stratum"])
        if a.wait_set:
            var = {w["id"]: w.get("variant") for w in json.load(open(a.wait_set))["windows"]}
            wa = (E.id.map(var) == "a") & E.run.str.contains("/wait_set/")
            E.loc[wa, "split"], E.loc[wa, "stratum"] = "wait-set a", "static"
        E = E[E.split.notna()]
        L += ["## End state of the human's GT track: corridor zeros given a trip (the first metric step >= 4 m from the GT)", "",
              "| tag | split | stratum | end | windows | tripped | corridor | corridor / tripped | median d_max kept (m), no corridor | min d at a corridor zero (m) |",
              "|---|---|---|---|--:|--:|--:|--:|--:|--:|"]
        for (t, sp, st, e), h in E.assign(stratum=np.where(E.stratum == "static", "static", "all other")).groupby(["tag", "split", "stratum", "end"]):
            tr, z = h[h.tripped], h[h.corridor == 1]
            L.append(f"| {t} | {sp} | {st} | {e} | {len(h)} | {len(tr)} | {len(z)} | {len(z) / max(len(tr), 1):.2f} | "
                     f"{h[h.corridor == 0].d_max_kept.median():.2f} | {z.d_max_kept.min() if len(z) else float('nan'):.2f} |")
        L.append("")
    if a.wait_set:
        W = json.load(open(a.wait_set))
        var = {w["id"]: w.get("variant") for w in W["windows"]}
        B = R[R.id.isin(var) & R.run.str.contains("/wait_set/")].copy()
        B["variant"] = B.id.map(var)
        B["gt_bin"] = pd.cut(B.gt_len, gb, right=False)
        L += ["## Wait set (decision 344), variant a (human never moves), by GT length", "",
              "| tag | GT length (m) | windows | corridor zeros | tripped | median d at the trip |", "|---|---|--:|--:|--:|--:|"]
        for t, g in B[B.variant == "a"].groupby("tag"):
            for b, h in g.groupby("gt_bin", observed=True):
                L.append(f"| {t} | {b} | {len(h)} | {int(h.corridor.sum())} | {int(h.tripped.sum())} | {h[h.tripped].d_trip.median():.2f} |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--tag", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--rows", nargs="+", required=True), p.add_argument("--arena-set", required=True), p.add_argument("--wait-set")
    p.add_argument("--gtend", nargs="*")
    p.add_argument("--out", required=True)
    p = sp.add_parser("gtend")
    p.add_argument("--set", required=True), p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    dict(extract=cmd_extract, gtend=cmd_gtend, report=cmd_report)[a.cmd](a)
