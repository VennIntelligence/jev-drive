#!/usr/bin/env python3
"""ROUTE-ONLY summary (plans/2026-10-13-route-only-prereg.md): per label on the common scene set, score, zeros by type, at-fault, GT,
progress, and the JEV_ROUTE coverage logged per decision. CPU only, read-only on the run dirs. Box, any python with numpy.

  ro_report.py --runs LABEL=<run dir>[,<run dir>..] ... --out DIR   -> DIR/summary.json, summary.md
A run dir holds aggregate/results-summary.json and driver-logs/drive.jsonl. Zero types from the score metrics of a zero-score rollout:
collision (collision_at_fault > 0), offroad, corridor (left_corridor_laterally), else its failure_reason (sanity / other).
Coverage (labels whose drive.jsonl logs `route`): decisions with the route path served (cov > 0), mean poses on it (0..8), the gap
(m the memory starts ahead of the ego) and the reason a decision kept the model plan (gap / no_route / behind_end / no_gt).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def load(dirs):
    R, D = {}, []
    for d in dirs.split(","):
        for r in json.load(open(f"{d}/aggregate/results-summary.json"))["rollouts"]:
            R[r["clipgt_id"]] = r
        f = Path(d) / "driver-logs/drive.jsonl"
        if f.exists():
            for line in f.open():
                if '"route"' in line:
                    o = json.loads(line)
                    if o.get("kind") == "drive" and "route" in o:
                        D.append((o["scene"], o["k"], o["route"]))
    return R, D


def ztype(r):
    m = r.get("score_metrics") or r.get("metrics") or {}
    if r.get("failure_reason"):
        return "sanity" if "sanity" in str(r["failure_reason"]) else "other"
    if (m.get("collision_at_fault") or 0) > 0:
        return "collision"
    if (m.get("offroad") or 0) > 0:
        return "offroad"
    if (m.get("left_corridor_laterally") or 0) > 0:
        return "corridor"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = {s.split("=", 1)[0]: load(s.split("=", 1)[1]) for s in a.runs}
    common = sorted(set.intersection(*(set(k for k, r in R.items() if r.get("score") is not None or r.get("failure_reason")) for R, _ in L.values())))
    out = {"n_common": len(common), "labels": {}}
    for lab, (R, D) in L.items():
        rs = [R[c] for c in common]
        sc = np.array([r.get("score") or 0.0 for r in rs])
        met = lambda k: np.array([(r.get("metrics") or {}).get(k, np.nan) for r in rs], float)  # noqa: E731
        z = collections.Counter(ztype(r) for r in rs if (r.get("score") or 0) == 0)
        af = sum(1 for r in rs if (r.get("score") or 0) == 0 and (((r.get("metrics") or {}).get("offroad_or_collision_at_fault") or 0) > 0
                                                                   or ((r.get("metrics") or {}).get("left_corridor_laterally") or 0) > 0))
        e = dict(n=len(rs), mean=float(sc.mean()), zeros=int((sc == 0).sum()), ztypes=dict(z), at_fault_or_corridor=af,
                 gt=float(np.nanmean(met("dist_to_gt_trajectory"))), lat_gt=float(np.nanmean(met("lateral_dist_to_gt_trajectory"))),
                 progress_rel=float(np.nanmean(met("progress_rel"))), dist_m=float(np.nanmean(met("dist_traveled_m"))),
                 gt_dist_m=float(np.nanmean(met("gt_dist_traveled_m"))))
        if D:
            cs = set(common)
            D = [d for d in D if d[0] in cs]
            cov = np.array([d[2].get("cov", 0) for d in D])
            k = np.array([d[1] for d in D])
            why = collections.Counter(d[2].get("why", "?") for d in D)
            gap = np.array([d[2].get("gap") for d in D if d[2].get("gap") is not None], float)
            dl = np.array([d[2]["dlat"] for d in D if d[2].get("dlat") is not None], float).reshape(-1, 3)
            e["coverage"] = dict(decisions=len(D), served_share=float((cov > 0).mean()), full_share=float((cov == 8).mean()),
                                 mean_cov=float(cov.mean()), why=dict(why),
                                 served_share_by_k={int(i): round(float((cov[k == i] > 0).mean()), 3) for i in np.unique(k)},
                                 gap_median=float(np.median(gap)) if len(gap) else None,
                                 dlat_abs_median_1_2_4s=np.round(np.median(np.abs(dl), 0), 3).tolist() if len(dl) else None,
                                 dlat_abs_p90_1_2_4s=np.round(np.percentile(np.abs(dl), 90, 0), 3).tolist() if len(dl) else None)
        out["labels"][lab] = e
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(out, indent=1))
    md = [f"# ROUTE-ONLY summary ({len(common)} common scenes)", "",
          "| label | mean | zeros | collision / offroad / corridor / sanity / other | GT (m) | lateral GT (m) | progress_rel | travelled / log (m) | route served (decisions) | mean poses on route | gap median (m) | lateral change 1 / 2 / 4 s median (m) |",
          "|:--|--:|--:|:--|--:|--:|--:|:--|--:|--:|--:|:--|"]
    for lab, e in out["labels"].items():
        z, c = e["ztypes"], e.get("coverage") or {}
        md.append(f"| {lab} | {e['mean']:.4f} | {e['zeros']} | {z.get('collision', 0)} / {z.get('offroad', 0)} / {z.get('corridor', 0)} / {z.get('sanity', 0)} / {z.get('other', 0)} | "
                  f"{e['gt']:.3f} | {e['lat_gt']:.3f} | {e['progress_rel']:.4f} | {e['dist_m']:.1f} / {e['gt_dist_m']:.1f} | "
                  f"{c.get('served_share', float('nan')):.3f} | {c.get('mean_cov', float('nan')):.2f} | {c.get('gap_median') if c.get('gap_median') is not None else '-'} | "
                  f"{' / '.join(map(str, c.get('dlat_abs_median_1_2_4s') or ['-']))} |")
    for lab, e in out["labels"].items():
        if e.get("coverage"):
            md += ["", f"{lab}: decisions {e['coverage']['decisions']}, kept-plan reasons {e['coverage']['why']}, served share by decision index {e['coverage']['served_share_by_k']}"]
    (o / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
