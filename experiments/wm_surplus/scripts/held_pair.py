#!/usr/bin/env python3
"""Score-only paired table of stored AlpaSim nuPlan held 460 @18 reads (P2, plans/2026-10-22-p2-match-prereg.md).

jevdrive.facts table needs our driver's decision logs (native_gap extract), which third-party drivers (WA-JEPA) do not write; this
reads only aggregate/results-summary.json of each chunk: the first rollout per scene, its score and failure_reason. Scenes where
either read failed the route sanity check ("route folds back", a scene / map defect, not driving) are dropped. Units = scenes,
bootstrap clustered by the scene's log (id up to the last '-'), jevdrive.stats.paired.

  python held_pair.py --read LABEL=RUNGLOB ... --refs LABEL ... --out results/p2_match/alpasim_held460.md
  RUNGLOB: a glob of AlpaSim run dirs, e.g. '$DATA_DIR/runs/route_mem/read/STANDFMXH-L18-s1/c*/run'
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import stats  # noqa: E402


def load(pattern: str) -> dict:
    """scene -> (score, failure class)."""
    out = {}
    for run in sorted(glob.glob(os.path.expandvars(pattern))):
        for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]:
            if r["clipgt_id"] not in out:
                fr = str(r.get("failure_reason") or "")
                out[r["clipgt_id"]] = (float(r["score"]), "route-fail" if "sanity" in fr else (fr or ("pass" if r["score"] > 0 else "?")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", nargs="+", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X = dict((lab, load(g)) for lab, g in (r.split("=", 1) for r in a.read))
    L = ["# P2: AlpaSim nuPlan held 460 @18, stored reads, score-only", "",
         "Scenes read by both, route-fail scenes (either read) dropped; paired = arm - ref, 95 % CI clustered by log (jevdrive.stats.paired).", "",
         "| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref |", "|:--|:--|--:|--:|--:|--:|"]
    res = {}
    for t in X:
        for r in a.refs:
            if r == t:
                continue
            ids = sorted(i for i in set(X[t]) & set(X[r]) if X[t][i][1] != "route-fail" and X[r][i][1] != "route-fail")
            x, y = (np.array([X[k][i][0] for i in ids]) for k in (t, r))
            p = stats.paired(x, y, groups=np.array([i.rsplit("-", 1)[0] for i in ids]))
            res[f"{t}|{r}"] = dict(n=len(ids), a=float(x.mean()), b=float(y.mean()), **{k: float(p[k]) for k in ("mean", "lo", "hi")},
                                   zeros_a=int((x == 0).sum()), zeros_b=int((y == 0).sum()))
            L.append(f"| {t} | {r} | {len(ids)} | {x.mean():.4f} / {y.mean():.4f} | {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] | "
                     f"{(x == 0).sum()} / {(y == 0).sum()} |")
    C = {t: Counter(c for s, c in F.values() if s == 0 or c == "route-fail") for t, F in X.items()}
    L += ["", "Zero classes (all scenes of each read; route-fail counted separately):", "", "| class | " + " | ".join(X) + " |",
          "|:--|" + "--:|" * len(X)]
    for k in sorted({k for c in C.values() for k in c}, key=lambda k: -sum(c.get(k, 0) for c in C.values())):
        L.append(f"| {k} | " + " | ".join(str(C[t].get(k, 0)) for t in X) + " |")
    L.append("\nScenes per read: " + ", ".join(f"{t} {len(F)}" for t, F in X.items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
