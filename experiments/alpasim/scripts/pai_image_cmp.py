#!/usr/bin/env python3
"""Scene-by-scene and decision-by-decision comparison of two PAI runs of the same scene list (results/pai_image.md): the submission image
in the official containerised stack against a native run (box pai_box.sh chunk, or Tokyo's mounted-driver pai_run.sh). stdlib only.
Per run: <dir>/results-summary.json (or <dir>/sim/aggregate/...) and the driver's drive.jsonl (<dir>/drive.jsonl, <dir>/driver-logs/, <dir>/driver/ (pai_run.sh),
<dir>/sim/driver-logs/). Per scene: both scores, the first decision k whose served plan (`poses`) differs by more than --tol m, the plan
difference at k = 0 (same rendered first frame -> the driver's own numerics), and the score difference; then paired mean, bootstrap CI,
zero agreement, and how many scenes are bit-identical over every decision.
  pai_image_cmp.py --a image=<dir> --b native=<dir> [--tol 1e-4] [--out cmp.md]
"""
import argparse
import json
import random
from pathlib import Path


def find(d: Path, name: str) -> Path:
    for p in (d / name, d / "aggregate" / name, d / "sim/aggregate" / name, d / "driver-logs" / name, d / "sim/driver-logs" / name, d / "driver" / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} under {d}")


def load(d: Path):
    sc = {}
    for r in json.load(open(find(d, "results-summary.json")))["rollouts"]:
        m = r.get("score_metrics") or {}
        why = r.get("failure_reason") or ",".join(k for k in ("collision_at_fault", "offroad", "left_corridor_laterally") if m.get(k)) or ""
        sc[r["clipgt_id"]] = (float(r["score"] or 0.0), str(why)[:40])
    plans = {}
    for line in open(find(d, "drive.jsonl")):
        r = json.loads(line)
        if r.get("kind") == "drive" and r.get("infer"):
            plans.setdefault(r["scene"], {})[r["k"]] = r["poses"]
    return sc, plans


def dmax(p, q) -> float:
    return max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(p, q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out")
    a = ap.parse_args()
    (la, da), (lb, db) = (x.split("=", 1) for x in (a.a, a.b))
    (sa, pa), (sb, pb) = load(Path(da)), load(Path(db))
    both = sorted(set(sa) & set(sb))
    rows, ident = [], 0
    for s in both:
        A, B = pa.get(s, {}), pb.get(s, {})
        ks = sorted(set(A) & set(B))
        first = next((k for k in ks if dmax(A[k], B[k]) > a.tol), None)
        d0 = dmax(A[0], B[0]) if 0 in A and 0 in B else float("nan")
        ident += first is None and len(ks) > 0 and len(A) == len(B)
        rows.append((s, sa[s], sb[s], len(ks), first, d0))
    diff = [r[1][0] - r[2][0] for r in rows]
    n = len(diff)
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(diff) for _ in range(n)) / n for _ in range(10000))
    za = sum((r[1][0] == 0) == (r[2][0] == 0) for r in rows)
    k0 = sorted(r[5] for r in rows if r[5] == r[5])
    L = [f"# {la} vs {lb}: {n} scenes", "",
         f"- mean {la} {sum(r[1][0] for r in rows) / n:.4f} ({sum(r[1][0] == 0 for r in rows)} zeros), {lb} {sum(r[2][0] for r in rows) / n:.4f} "
         f"({sum(r[2][0] == 0 for r in rows)} zeros); paired diff {sum(diff) / n:+.4f} [{bs[250]:+.4f}, {bs[9750]:+.4f}] (bootstrap 95 %)",
         f"- scores equal {sum(abs(x) < 1e-9 for x in diff)}, within 0.01 {sum(abs(x) <= 0.01 for x in diff)}, within 0.05 {sum(abs(x) <= 0.05 for x in diff)}; "
         f"zero / non-zero agree {za} of {n}",
         f"- served plans identical (all decisions, tol {a.tol} m) in {ident} of {n} scenes; plan difference at decision 0: median "
         f"{k0[len(k0) // 2] if k0 else float('nan'):.4f} m, max {max(k0) if k0 else float('nan'):.4f} m", "",
         f"| scene | {la} | {lb} | diff | decisions | first k differing | plan diff k=0 (m) |", "|---|--:|--:|--:|--:|--:|--:|"]
    for s, x, y, nk, f, d0 in rows:
        L.append(f"| {s[7:15]} | {x[0]:.4f} {x[1]} | {y[0]:.4f} {y[1]} | {x[0] - y[0]:+.4f} | {nk} | {'-' if f is None else f} | {d0:.4f} |")
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
