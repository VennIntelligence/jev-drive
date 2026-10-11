#!/usr/bin/env python3
"""Three-way paired read on all 1491 public nuPlan scenes: P2H10-F, APY10m10-AB (seeds 0,1) and P2H10S-F (seeds 0-3), per-scene seed means,
log-clustered 95% CI (cf1_report.py convention). Descriptive; nothing here promotes a checkpoint.

  nuplan1491_report.py --out DIR     (box; plain python3 + numpy)
Reads ot3/a, cf1/a manifests (P2H10-F, APY10m10-AB), body1/cl/shape-cl and nuplan1491/a manifests (P2H10S-F). Writes nuplan1491_pick.md,
nuplan1491_stats.json and nuplan1491_per_scene.csv into --out.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c0b_report as R  # noqa: E402
from ot3_report import ci2, f2  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RA = DATA / "runs/alpasim"
SEEDS = {"P2H10-F": (0, 1), "APY10m10-AB": (0, 1), "P2H10S-F": (0, 1, 2, 3)}
SHORT = {"P2H10-F": "P2H10", "APY10m10-AB": "APY10m10", "P2H10S-F": "P2H10S"}


def man(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    o3, cf = man(RA / "ot3/a/manifest.json"), man(RA / "cf1/a/manifest.json")
    sh, nw = man(DATA / "runs/body1/cl/shape-cl/manifest.json"), man(RA / "nuplan1491/a/manifest.json")
    L5 = RA / "c0b/lists"
    S7 = set((L5 / "all.txt").read_text().split())
    NEW = set((L5 / "cf1fresh.txt").read_text().split()) | set((L5 / "ot3new.txt").read_text().split())
    dirs = {}
    for r in ("P2H10-F", "APY10m10-AB"):
        for s in SEEDS[r]:
            k = f"{r}-s{s}"
            dirs[k] = o3.get(k, []) + o3.get(k + "@new", []) + cf.get(k + "@new", []) + cf.get(k + "@fresh", [])
    for s in SEEDS["P2H10S-F"]:
        k = f"P2H10S-F-s{s}"
        dirs[k] = sh.get(k, []) + nw.get(k + "@cf1fresh", []) + nw.get(k + "@ot3new", [])
    D = {k: R.load_driver(v) for k, v in dirs.items()}
    scenes = sorted(S7 | NEW)
    assert len(scenes) == 1491
    cl = np.array([R.log_of(s) for s in scenes])
    f = lambda fn, d=np.nan: {k: np.array([fn(D[k][0][s]) if s in D[k][0] else d for s in scenes]) for k in D}  # noqa: E731
    sc = f(lambda r: r["score"])
    ev = f(lambda r: float((r.get("metrics") or {}).get("offroad_or_collision_at_fault", 0) or 0))
    zc = f(R.zclass, "")
    miss = {k: int(np.isnan(sc[k]).sum()) for k in D}
    rec = {SHORT[r]: [f"{r}-s{s}" for s in SEEDS[r]] for r in SEEDS}
    gs = {t: np.mean([sc[x] for x in v], 0) for t, v in rec.items()}
    gs01 = np.mean([sc["P2H10S-F-s0"], sc["P2H10S-F-s1"]], 0)
    sets = {"All (1491)": np.ones(len(scenes), bool), "New (791: BODY1 read did not cover)": np.array([s in NEW for s in scenes])}
    st, Lm = {"missing_scenes_per_run": miss}, []
    for n, m in sets.items():
        have = m & np.all([~np.isnan(gs[t]) for t in rec], 0)
        Lm += [f"## {n}: {int(have.sum())} of {int(m.sum())} scenes scored by all three, {len(set(cl[have]))} logs", "",
               "| recipe | seeds (mean) | mean [95% CI, logs] | zeros | collision | offroad | corridor | slow (0<s<1) | mean progress |", "|:--|:--|:--|--:|--:|--:|--:|--:|--:|"]
        st[n] = dict(n=int(have.sum()), recipes={}, diffs={})
        for t, v in rec.items():
            h = have
            mean = R.ci(gs[t][h], cl[h])
            cnt = lambda fn: float(np.mean([fn(x) for x in v]))  # noqa: E731
            z = {c: cnt(lambda x: (zc[x][h] == c).sum()) for c in R.FAIL}
            zeros, slow = cnt(lambda x: (sc[x][h] == 0).sum()), cnt(lambda x: ((sc[x][h] > 0) & (sc[x][h] < 1)).sum())
            prog = cnt(lambda x: _prog(D[x][0], scenes, h))
            st[n]["recipes"][t] = dict(mean=mean, seeds=[float(sc[x][h].mean()) for x in v], zeros=zeros, slow=slow, progress=prog, **z)
            Lm.append(f"| {t} | {' / '.join(f'{sc[x][h].mean():.4f}' for x in v)} | {R.fmt(mean)} | {zeros:g} | {z['collision_at_fault']:g} | {z['offroad']:g} | "
                      f"{z['left_corridor_laterally']:g} | {slow:g} | {prog:.4f} |")
        Lm += ["", "| paired difference (per-scene seed means) | point | 95% CI scenes | 95% CI logs | per-seed-index differences |", "|:--|--:|:--|:--|:--|"]
        for x, y in (("P2H10S", "P2H10"), ("APY10m10", "P2H10"), ("P2H10S", "APY10m10")):
            d = ci2(gs[x][have] - gs[y][have], cl[have])
            sd = [float((sc[p][have] - sc[q][have]).mean()) for p, q in zip(rec[x], rec[y])]
            st[n]["diffs"][f"{x} - {y}"] = dict(**d, per_seed=sd)
            Lm.append(f"| {x} (4 seeds) - {y} | {f2(d)} | " + " / ".join(f"{v:+.4f}" for v in sd) + " |")
        for y in ("P2H10", "APY10m10"):
            d = ci2(gs01[have] - gs[y][have], cl[have])
            st[n]["diffs"][f"P2H10S s0-1 - {y}"] = d
            Lm.append(f"| P2H10S (seeds 0-1 only) - {y} | {f2(d)} | |")
        Lm.append("")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "nuplan1491_per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "new"] + [f"{k}|{c}" for k in D for c in ("score", "zero_class")])
        for i, s in enumerate(scenes):
            w.writerow([s, cl[i], int(s in NEW)] + [x for k in D for x in (sc[k][i], zc[k][i])])
    (out / "nuplan1491_stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "nuplan1491_pick.md").write_text("\n".join(Lm) + "\n")
    print("\n".join(Lm), "\nmissing per run:", miss)


def _prog(Rr, scenes, h):
    v = []
    for s, keep in zip(scenes, h):
        r = Rr.get(s)
        if keep and r is not None:
            v.append(_progress(r))
    return float(np.mean(v)) if v else float("nan")


def _progress(r):
    m = r.get("metrics") or {}
    for k in ("progress", "progress_ratio", "route_progress"):
        if k in m:
            return float(m[k])
    return float("nan")


if __name__ == "__main__":
    main()
