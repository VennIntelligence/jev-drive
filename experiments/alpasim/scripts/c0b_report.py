#!/usr/bin/env python3
"""C0b read-out: five drivers on the same public AlpaSim scenes -> per-scene table, paired differences with log-clustered
bootstrap CIs, best-of-k, per-shard breakdown, OT30 line, cold start. Plain python3 + numpy on the box.

  c0b_report.py --manifest c0b/manifest.json --shards c0b/lists/shards.tsv --out c0b/report
manifest = {driver name: [run dir, ...]}; a scene found in several run dirs of one driver must agree (checked, reported).
Writes c0b_report.md, c0b_per_scene.csv, c0b_per_scene.json, c0b_stats.json into --out.
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")
SH, AP, WA, O0, O1 = "SH30-F-s0", "AP2-AB-s0", "WA-JEPA (reference)", "OT30-F-s0", "OT30-F-s1"
OURS = [SH, AP, O0, O1]
B = 10000


def log_of(s):
    return "_".join(s.split("_")[:2])


def load_driver(dirs):
    """scene -> record (first run dir wins; disagreements are listed), plus the driver-log facts per scene."""
    R, mism, first = {}, [], {}
    ndrive = defaultdict(int)
    for D in map(Path, dirs):
        d = json.loads((D / "aggregate/results-summary.json").read_text())
        for r in d["rollouts"]:
            s = r["clipgt_id"]
            if s in R:
                if abs(R[s]["score"] - r["score"]) > 1e-9:
                    mism.append(s)
            else:
                R[s] = r
        f = D / "driver-logs/drive.jsonl"
        if f.exists():
            for line in f.open():
                if '"drive"' not in line:
                    continue
                x = json.loads(line)
                if x["kind"] != "drive":
                    continue
                if x["k"] == 0:
                    first.setdefault(x["scene"], x["poses"][-1][:2])
                ndrive[x["scene"]] = max(ndrive[x["scene"]], x["k"] + 1)
    return R, mism, first, ndrive


def zclass(r):
    m = r["score_metrics"]
    for f in FAIL:
        if m.get(f):
            return f
    return "other" if r["score"] == 0 else ""


def ci(vals, clusters, rng_seed=0):
    """Mean of per-scene `vals` with a 95% bootstrap CI resampling whole clusters (logs)."""
    v = np.asarray(vals, float)
    u, inv = np.unique(clusters, return_inverse=True)
    S = np.bincount(inv, v, len(u))
    N = np.bincount(inv, minlength=len(u)).astype(float)
    idx = np.random.default_rng(rng_seed).integers(0, len(u), (B, len(u)))
    boot = S[idx].sum(1) / N[idx].sum(1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def fmt(t, p=4, sign=False):
    m, lo, hi = t
    return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]" if sign else f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"), ap.add_argument("--shards"), ap.add_argument("--out")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = dict(l.split("\t") for l in Path(a.shards).read_text().strip().split("\n"))
    shard = {k: v.strip() for k, v in shard.items()}
    D = {k: load_driver(v) for k, v in man.items()}
    names = [k for k in (SH, AP, O0, O1, WA) if k in D]
    scenes = sorted(set.intersection(*(set(D[k][0]) for k in names)) & set(shard))
    cl = np.array([log_of(s) for s in scenes])
    sc = {k: np.array([D[k][0][s]["score"] for s in scenes]) for k in names}
    zc = {k: [zclass(D[k][0][s]) for s in scenes] for k in names}
    ev = {k: np.array([float((D[k][0][s].get("metrics") or {}).get("offroad_or_collision_at_fault", 0) or 0) for s in scenes]) for k in names}
    prog = {k: np.array([D[k][0][s]["score_metrics"].get("progress_clipped_rel") or 0 for s in scenes])   # null (no progress value) counts as 0 for k in names}
    nd = {k: np.array([D[k][3].get(s, 0) for s in scenes]) for k in names}
    first = {k: np.array([D[k][2].get(s, (np.nan, np.nan)) for s in scenes], float) for k in names}
    L = [f"Scenes common to all {len(names)} drivers: {len(scenes)} from {len(set(cl))} logs, {len(set(shard[s] for s in scenes))} shards. "
         "Scene score = AlpaSim scene score. CIs: 95% bootstrap resampling whole logs (nuPlan `date_vehicle`), 10 000 draws, seed 0.", ""]
    for k in names:
        L.append(f"- {k}: {len(D[k][0])} scored scenes, {len(man[k])} run dirs, disagreeing duplicate scenes: {len(D[k][1])}")
    st = {}
    L += ["", "## Scores", "",
          "| driver | scenes | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | zero, other | slow (0 < score < 1) | at-fault events | mean progress |",
          "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in names:
        z = zc[k]
        c = {f: sum(x == f for x in z) for f in (*FAIL, "other")}
        slow = int(((sc[k] > 0) & (sc[k] < 1)).sum())
        st[k] = dict(n=len(scenes), mean=ci(sc[k], cl), n1=int((sc[k] >= 1).sum()), zeros=int((sc[k] == 0).sum()), **c, slow=slow,
                     events=float(ev[k].sum()), progress=float(prog[k].mean()))
        s = st[k]
        L.append(f"| {k} | {s['n']} | {fmt(s['mean'])} | {s['n1']} | {s['zeros']} | {c['collision_at_fault']} | {c['offroad']} | "
                 f"{c['left_corridor_laterally']} | {c['other']} | {slow} | {s['events']:.0f} | {s['progress']:.3f} |")
    # ---- paired differences
    ot = (sc[O0] + sc[O1]) / 2 if O0 in sc and O1 in sc else None
    coll = lambda k: np.array([x == "collision_at_fault" for x in zc[k]], float)
    cands = {k: sc[k] for k in names}
    if ot is not None:
        cands["OT30 mean of 2 seeds"] = ot
    L += ["", "## Paired differences (per scene, log-clustered CI)", "",
          "| A - B | mean scene score difference [95% CI] | zeros A / B | at-fault collision zeros A / B |", "|:--|:--|--:|--:|"]
    pd = {}
    pairs = [(k, SH) for k in cands if k not in (SH, WA)] + ([(k, WA) for k in cands if k != WA] if WA in sc else [])
    for A, Bn in pairs:
        d = cands[A] - cands[Bn]
        za = lambda k: (st[k]["zeros"] if k in st else (st[O0]["zeros"] + st[O1]["zeros"]) / 2)
        ca = lambda k: (st[k]["collision_at_fault"] if k in st else (st[O0]["collision_at_fault"] + st[O1]["collision_at_fault"]) / 2)
        pd[f"{A} - {Bn}"] = ci(d, cl)
        L.append(f"| {A} - {Bn} | {fmt(pd[f'{A} - {Bn}'], sign=True)} | {za(A):g} / {za(Bn):g} | {ca(A):g} / {ca(Bn):g} |")
    # ---- OT30 line
    if ot is not None:
        t = pd[f"OT30 mean of 2 seeds - {SH}"]
        c_ot = (st[O0]["collision_at_fault"] + st[O1]["collision_at_fault"]) / 2
        ok1, ok2 = t[0] >= 0.010 and t[1] > 0, c_ot <= st[SH]["collision_at_fault"]
        L += ["", "## OT30 line (pre-registered, plans/2026-10-09-ot30-closedloop-prereg.md)", "",
              f"- mean of the two OT30 seeds minus SH30-F-s0: {fmt(t, sign=True)}; line: >= +0.010 and CI lower bound > 0 -> **{'met' if ok1 else 'not met'}**",
              f"- at-fault collision zeros, mean of the two seeds {c_ot:g} vs SH30 {st[SH]['collision_at_fault']}: line: not higher -> **{'met' if ok2 else 'not met'}**",
              f"- verdict: **{'candidate' if ok1 and ok2 else 'not a candidate'}**",
              f"- each seed alone minus SH30: s0 {fmt(pd[f'{O0} - {SH}'], sign=True)}, s1 {fmt(pd[f'{O1} - {SH}'], sign=True)}",
              f"- offroad + left-corridor zeros: SH30 {st[SH]['offroad'] + st[SH]['left_corridor_laterally']}, OT30 s0 {st[O0]['offroad'] + st[O0]['left_corridor_laterally']}, "
              f"s1 {st[O1]['offroad'] + st[O1]['left_corridor_laterally']}"]
    # ---- best-of-k among our own drivers (oracle selector)
    L += ["", "## Best-of-k among our own drivers (per-scene maximum = an oracle selector, an upper bound)", "",
          "| set | oracle mean [95% CI] | best single in the set (mean) | oracle minus best single [95% CI] |", "|:--|:--|--:|:--|"]
    bo = {}
    for r in (2, 3, 4):
        for combo in itertools.combinations([k for k in OURS if k in sc], r):
            m = np.max([sc[k] for k in combo], 0)
            best = max(combo, key=lambda k: sc[k].mean())
            g = ci(m - sc[best], cl)
            bo[" + ".join(combo)] = dict(oracle=ci(m, cl), best=best, gain=g)
            L.append(f"| {' + '.join(combo)} | {fmt(ci(m, cl))} | {best} {sc[best].mean():.4f} | {fmt(g, sign=True)} |")
    if O0 in sc and O1 in sc:
        L += ["", "Reference for the selector rows: `OT30-F-s0 + OT30-F-s1` is two seeds of one recipe, i.e. what an oracle gains from training-seed noise alone."]
    # ---- per shard
    shards = sorted(set(shard[s] for s in scenes))
    sh = np.array([shard[s] for s in scenes])
    L += ["", "## Per shard (mean scene score / zeros)", "", "| shard | scenes | " + " | ".join(names) + " |", "|:--|--:|" + "--:|" * len(names)]
    for p in shards:
        m = sh == p
        L.append(f"| {p} | {int(m.sum())} | " + " | ".join(f"{sc[k][m].mean():.4f} / {int((sc[k][m] == 0).sum())}" for k in names) + " |")
    m1, mr = sh == "part001", sh != "part001"
    L += ["", "Part001 against the rest (mean scene score; difference with a log-clustered CI):", "",
          "| driver | part001 | other shards | part001 - others [95% CI] |", "|:--|--:|--:|:--|"]
    for k in names:
        v = sc[k]
        # unpaired: bootstrap each side separately and subtract
        rng = np.random.default_rng(1)
        def bs(mask):
            u, inv = np.unique(cl[mask], return_inverse=True)
            S = np.bincount(inv, v[mask], len(u)); N = np.bincount(inv, minlength=len(u)).astype(float)
            ix = rng.integers(0, len(u), (B, len(u)))
            return S[ix].sum(1) / N[ix].sum(1)
        diff = bs(m1) - bs(mr)
        L.append(f"| {k} | {v[m1].mean():.4f} | {v[mr].mean():.4f} | {v[m1].mean() - v[mr].mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}] |")
    # ---- zero overlap
    if WA in sc:
        zs = {k: set(np.array(scenes)[sc[k] == 0]) for k in names}
        L += ["", f"## Where the zeros are", "",
              f"WA-JEPA (reference) zeros: {len(zs[WA])}; of them also zero for SH30 {len(zs[WA] & zs[SH])}, AP2 {len(zs[WA] & zs[AP])}"
              + (f", OT30 s0 {len(zs[WA] & zs[O0])}, s1 {len(zs[WA] & zs[O1])}" if O0 in zs and O1 in zs else "")
              + f". Zero for none of our drivers but for WA-JEPA: {len(zs[WA] - set().union(*[zs[k] for k in OURS if k in zs]))}; "
              f"zero for all of our drivers: {len(set.intersection(*[zs[k] for k in OURS if k in zs]))}.", "",
              "| WA-JEPA zero scene | shard | class | " + " | ".join(k for k in OURS if k in zs) + " |", "|:--|:--|:--|" + "--|" * len([k for k in OURS if k in zs])]
        idx = {s: i for i, s in enumerate(scenes)}
        for s in sorted(zs[WA]):
            i = idx[s]
            L.append(f"| {s} | {shard[s]} | {zc[WA][i]} | " + " | ".join((zc[k][i] or f"{sc[k][i]:.2f}") for k in OURS if k in zs) + " |")
    # ---- cold start
    L += ["", "## Cold start (decision 0: identical simulator state for every driver)", "",
          "Plan endpoint (4 s, x forward / y left in the ego frame at decision 0). Difference to SH30 = distance between the two endpoints.", "",
          "| driver | mean x (m) | mean abs y (m) | scenes with abs y > 2 m | distance to SH30: mean / p90 / max (m) | scenes > 3 m from SH30 | zeros ending within 3 decisions |",
          "|:--|--:|--:|--:|:--|--:|--:|"]
    cs = {}
    for k in names:
        f = first[k]
        okf = ~np.isnan(f[:, 0])
        row = [f"{np.nanmean(f[:, 0]):.2f}" if okf.any() else "n/a", f"{np.nanmean(np.abs(f[:, 1])):.2f}" if okf.any() else "n/a",
               str(int((np.abs(f[okf, 1]) > 2).sum())) if okf.any() else "n/a"]
        if k != SH and okf.any() and not np.isnan(first[SH][:, 0]).all():
            dd = np.linalg.norm(f - first[SH], axis=1)
            dd = dd[~np.isnan(dd)]
            row += [f"{dd.mean():.2f} / {np.percentile(dd, 90):.2f} / {dd.max():.2f}", str(int((dd > 3).sum()))]
            cs[k] = dict(mean=float(dd.mean()), p90=float(np.percentile(dd, 90)), max=float(dd.max()), over3=int((dd > 3).sum()))
        else:
            row += ["-", "-"]
        early = int(((sc[k] == 0) & (nd[k] > 0) & (nd[k] <= 3)).sum())
        L.append(f"| {k} | " + " | ".join(row) + f" | {early} |")
    L += ["", "Scenes whose rollout has no driver-log decisions are not counted in the last column (nd = 0: " +
          ", ".join(f"{k} {int((nd[k] == 0).sum())}" for k in names) + ")."]
    # ---- files
    cols = ["scene", "log", "shard"] + [f"{k}|{c}" for k in names for c in ("score", "zero_class", "progress", "atfault_events", "n_decisions", "plan0_x", "plan0_y")]
    rows = []
    for i, s in enumerate(scenes):
        r = dict(scene=s, log=cl[i], shard=shard[s])
        for k in names:
            r.update({f"{k}|score": round(float(sc[k][i]), 6), f"{k}|zero_class": zc[k][i], f"{k}|progress": round(float(prog[k][i]), 4),
                      f"{k}|atfault_events": float(ev[k][i]), f"{k}|n_decisions": int(nd[k][i]),
                      f"{k}|plan0_x": None if np.isnan(first[k][i, 0]) else round(float(first[k][i, 0]), 3),
                      f"{k}|plan0_y": None if np.isnan(first[k][i, 1]) else round(float(first[k][i, 1]), 3)})
        rows.append(r)
    with (out / "c0b_per_scene.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader(), w.writerows(rows)
    (out / "c0b_per_scene.json").write_text(json.dumps(dict(drivers=names, columns=["score", "zero_class", "progress", "atfault_events", "n_decisions", "plan0_x", "plan0_y"], rows=rows)))
    (out / "c0b_stats.json").write_text(json.dumps(dict(scenes=len(scenes), stats=st, paired=pd, best_of=bo, cold=cs), indent=1, default=float))
    (out / "c0b_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
