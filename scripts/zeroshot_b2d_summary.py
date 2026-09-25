#!/usr/bin/env python3
"""Headline numbers of a zero-shot Bench2Drive run: DS mean with a route bootstrap 95% CI, SR with a Wilson 95% CI
(Bench2Drive tools/merge_route_json.py: status Completed and no infraction other than min_speed_infractions), RC, and
the routes that never finished. Standard library only.

    python3 scripts/zeroshot_b2d_summary.py <run dir> [--expected 220] [--out summary.json]
"""
import argparse
import json
import math
import random
from pathlib import Path


def records(run):
    out = {}
    for rdir in sorted((run / "attempts").iterdir()):
        for a in sorted(rdir.iterdir(), key=lambda p: int(p.name), reverse=True):
            try:
                rec = json.loads((a / "results.json").read_text())["_checkpoint"]["records"][0]
                rr = json.loads((a / "route_result.json").read_text())
            except (OSError, KeyError, IndexError, ValueError):
                continue
            if rr.get("status") == "finished":
                out[rdir.name] = rec
                break
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def boot(xs, n=10000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return (m[int(0.025 * n)], m[int(0.975 * n) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--expected", type=int, default=220)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    recs = records(a.run)
    ds = [r["scores"]["score_composed"] for r in recs.values()]
    rc = [r["scores"]["score_route"] for r in recs.values()]
    ok = [r["status"] == "Completed" and all(not v for k, v in r["infractions"].items() if k != "min_speed_infractions")
          for r in recs.values()]
    n, k = len(ds), sum(ok)
    res = {"run": str(a.run), "n_finished": n, "n_never_finished": a.expected - n,
           "DS": round(sum(ds) / n, 2), "DS_CI95": [round(x, 2) for x in boot(ds)],
           "DS_missing_as_0": round(sum(ds) / a.expected, 2),
           "SR": round(100 * k / n, 2), "SR_k": k, "SR_CI95": [round(100 * x, 2) for x in wilson(k, n)],
           "RC": round(sum(rc) / n, 2)}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
