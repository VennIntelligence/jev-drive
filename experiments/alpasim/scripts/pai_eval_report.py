"""Read-out of pai_eval.sh / pai_run.sh runs that are cut into chunks: every label is a set of run dirs (globs), merged by scene.

  pai_eval_report.py --rows s1='/data/runs/alpasim/pai_s1/runs/*' s0='/data/runs/alpasim/pai_full/runs/p2h10s-f-s0_*' [--pair s1-s0 ...]
                     [--mean both=s0,s1] [--scenes tsv] [--out md]        (plain python3 + numpy, on the Tokyo box)
--mean adds a row whose scene score is the mean of the named rows (their common scenes; it has no zero kinds).
--only <file of scene ids> restricts the scene set of the row table and the pairs (a subset read). --detail appends, after everything else, a
per-row table (scenes scored 1, slow scenes, zeros by kind, driver errors of the run dirs) and one table of the --pair differences with CI,
better / worse counts and the decomposition of each difference into its zero -> scored, scored -> zero and both-scored parts (a mean row is
zero on a scene only if every seed is zero). Without the two options the output is as before.
Tables: mean scene score and zeros by kind per label on the scenes every label scored; per --pair the paired difference with a bootstrap
95 % CI over scenes (10000 draws, seed 0); throughput of every run dir (scenes, CONC, card, wall, scenes per hour, render / drive share of
the summed RPC time, per-container VRAM peaks, card peak). --scenes writes one row per scene (scores of every label).
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pai_report as P  # noqa: E402


def load(pat: str) -> tuple:
    dirs = [Path(d) for p in pat.split(",") for d in sorted(glob.glob(p)) if (Path(d) / "sim/aggregate/results-summary.json").exists()]
    R = {}
    for d in dirs:
        R |= P.rollouts(d)
    return dirs, R


def rpc(d: Path) -> dict:
    out = {}
    for ln in (d / "sim/telemetry/metrics.prom").read_text().splitlines():
        m = re.match(r'rpc_duration_seconds_sum\{method="(\w+)",service="(\w+)"[^}]*\} (\S+)', ln)
        if m:
            out[f"{m[2]}.{m[1]}"] = float(m[3])
    return out


def usage(d: Path) -> tuple:
    peak, card = {}, 0
    for ln in (d / "usage.jsonl").read_text().splitlines():
        r = json.loads(ln)
        for k, v in r["vram_mib"].items():
            k = "driver" if k.endswith("-drv") else k.split("-")[-3]
            peak[k] = max(peak.get(k, 0), v)
        card = max(card, max(map(int, r["cards_mib"]), default=0))
    return peak, card


ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True, help="label=glob[,glob...]")
ap.add_argument("--pair", nargs="*", default=[], help="a-b: paired difference a minus b on their common scenes")
ap.add_argument("--mean", nargs="*", default=[], help="label=a,b[,...]: the per-scene mean of rows a, b, ...")
ap.add_argument("--scenes"), ap.add_argument("--out")
ap.add_argument("--only", help="file of scene ids (first column, a scene_id header is skipped): restrict the scenes of the tables and pairs")
ap.add_argument("--detail", action="store_true", help="append the per-row detail and the decomposition of every pair")
a = ap.parse_args()
D, R = {}, {}
for x in a.rows:
    k, pat = x.split("=", 1)
    D[k], R[k] = load(pat)
for x in a.mean:
    k, ks = x.split("=", 1)
    ks = ks.split(",")
    R[k] = {q: {"score": float(np.mean([R[j][q]["score"] for j in ks])), "score_metrics": {}} for q in set.intersection(*(set(R[j]) for j in ks))}
keep = {ln.split("\t")[0].strip() for ln in Path(a.only).read_text().splitlines() if ln.strip() and not ln.startswith("scene_id")} if a.only else None
sc = sorted(set.intersection(*(set(v) for v in R.values())) & (keep if keep is not None else set.union(*(set(v) for v in R.values()))))
L = [f"Scenes scored by every row: {len(sc)} (" + ", ".join(f"{k} {len(v)}" for k, v in R.items()) + ").", "",
     "| Row | Mean | Zeros | at-fault collision | offroad | left corridor | other zero |", "|---|--:|--:|--:|--:|--:|--:|"]
for k, v in R.items():
    z = [x for x in sc if v[x]["score"] == 0]
    cnt = [sum(bool(v[x]["score_metrics"].get(f)) for x in z) for f in P.FAIL]
    kinds = " | ".join(map(str, cnt)) + f" | {sum(not any(v[x]['score_metrics'].get(f) for f in P.FAIL) for x in z)}" if k in D else " | | | "
    L.append(f"| {k} | {np.mean([v[x]['score'] for x in sc]):.4f} | {len(z)} | {kinds} |")
PS = []
for pr in a.pair:
    ka, kb = pr.split("-", 1)
    s = sorted(set(R[ka]) & set(R[kb]) & (keep if keep is not None else set(R[ka])))
    d = np.array([R[ka][x]["score"] - R[kb][x]["score"] for x in s])
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    PS.append((ka, kb, s, d, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    L += ["", f"Paired difference {ka} - {kb} on {len(s)} scenes: {d.mean():+.4f}, bootstrap 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]; "
          f"means {np.mean([R[ka][x]['score'] for x in s]):.4f} / {np.mean([R[kb][x]['score'] for x in s]):.4f}; {ka} better on {(d > 0.01).sum()}, "
          f"worse on {(d < -0.01).sum()}, within 0.01 on {(abs(d) <= 0.01).sum()} (max |difference| {abs(d).max():.4f})."]
L += ["", "| Run | Scenes | CONC | Card | Wall s | Scenes / h | Render share | Drive share | Driver ms / call | VRAM peaks MiB | Card peak MiB |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|"]
for k, dirs in D.items():
    for d in dirs:
        if not (d / "run.json").exists():
            continue
        j, t, (pk, card) = json.loads((d / "run.json").read_text()), rpc(d), usage(d)
        tot = sum(t.values())
        dr = [x["total_ms"] for x in P.drive_log(d)[0] if x.get("infer")]
        L.append(f"| {k}: {d.name} | {j['scenes']} | {j['conc']} | {j['gpu']} | {j['wall_s']} | {j['scenes'] * 3600 / j['wall_s']:.0f} | "
                 f"{t.get('sensorsim.render_rgb', 0) / tot:.2f} | {t.get('driver.drive', 0) / tot:.2f} | {np.median(dr) if dr else float('nan'):.0f} | "
                 + ", ".join(f"{q} {v}" for q, v in sorted(pk.items())) + f" | {card} |")
if a.detail:
    err = lambda dirs: sum(x.get("inference_error", 0) + x.get("input_error", 0) for d in dirs for x in P.drive_log(d)[1])  # noqa: E731
    L += ["", f"Detail on the {len(sc)} scenes scored by every row. Score 1 = perfect scene, slow = 0 < score < 1; zero kinds are the scorer's flags on "
          "zero-score scenes (two flags can sit on one scene); driver errors = inference_error + input_error counters over all scenes of the row's "
          "run dirs; a mean row has no flags and is zero on a scene only if every seed is zero.", "",
          "| Row | Scenes | Mean | Score 1 | Slow | Zeros | at-fault collision | offroad | left corridor | other zero | Driver errors |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, v in R.items():
        x = np.array([v[q]["score"] for q in sc])
        z = [q for q in sc if v[q]["score"] == 0]
        kinds = " | ".join(str(sum(bool(v[q]["score_metrics"].get(f)) for q in z)) for f in P.FAIL) + \
            f" | {sum(not any(v[q]['score_metrics'].get(f) for f in P.FAIL) for q in z)} | {err(D[k])}" if k in D else " | | | | "
        L.append(f"| {k} | {len(sc)} | {x.mean():.4f} | {(x >= 1).sum() if k in D else ''} | {((x > 0) & (x < 1)).sum() if k in D else ''} | {len(z)} | {kinds} |")
    L += ["", "Every pair a - b with its decomposition by zero / scored status. Fixed = b zero and a scored, broken = b scored and a zero; a part is the "
          "summed difference of those scenes divided by the scenes of the pair, so the three parts add up to the mean difference (check column). "
          "Better / worse / within = scenes where a is above b by more than 0.01 / below by more than 0.01 / within 0.01.", "",
          "| Pair (a - b) | Scenes | Mean difference | Bootstrap 95 % CI | Better / worse / within | Fixed: scenes | part | Broken: scenes | part | "
          "Both scored: scenes | part | Both zero: scenes | Check |", "|---|--:|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for ka, kb, s, d, lo, hi in PS:
        xa, xb = np.array([R[ka][q]["score"] for q in s]), np.array([R[kb][q]["score"] for q in s])
        m = [(xb == 0) & (xa > 0), (xb > 0) & (xa == 0), (xb > 0) & (xa > 0), (xb == 0) & (xa == 0)]
        part = [d[q].sum() / len(d) for q in m]
        L.append(f"| {ka} - {kb} | {len(s)} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {(d > 0.01).sum()} / {(d < -0.01).sum()} / {(abs(d) <= 0.01).sum()} | "
                 f"{m[0].sum()} | {part[0]:+.4f} | {m[1].sum()} | {part[1]:+.4f} | {m[2].sum()} | {part[2]:+.4f} | {m[3].sum()} | {sum(part) - d.mean():+.0e} |")
print("\n".join(L))
if a.out:
    Path(a.out).write_text("\n".join(L) + "\n")
if a.scenes:
    ks = list(R)
    allsc = sorted(set.union(*(set(v) for v in R.values())))
    Path(a.scenes).write_text("scene_id\t" + "\t".join(ks) + "\n" + "".join(
        x + "\t" + "\t".join(f"{R[k][x]['score']:.4f}" if x in R[k] else "" for k in ks) + "\n" for x in allsc))
