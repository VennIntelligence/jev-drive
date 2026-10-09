#!/usr/bin/env python3
"""Lane OT3 read-out (decisions 212 / 213): drivers on the same public AlpaSim scenes and chunking -> scores, zero classes, seed-mean recipes,
and the pre-registered line against the baseline recipe (plans/2026-10-09-ot3-lambda10-prereg.md): two-seed mean minus the baseline's two-seed
mean, paired per scene, with a scene-resampled and a log-clustered 95% bootstrap CI (the log-clustered one decides).

  ot3_report.py --manifest ot3/a/manifest.json [more ...] --shards c0b/lists/shards.tsv --out DIR --name a --base P2H10=P2H10-F-s0+P2H10-F-s1 \
      --group AP2H10=AP2H10-AB-s0+AP2H10-AB-s1 ... [--ref SH30=SH30-F-s0+SH30-F-s1 ...] [--alt P2H10-F-s0=<run dir>,<run dir> ...] [--groups groups.json] [--scenes list.txt]
manifest = {driver label: [run dirs]} (later files win). --group = a candidate recipe (the line is applied); --ref = a recipe shown and paired
without the line; --alt LABEL=dirs = the same checkpoint run under another scene list / chunking (list-composition noise, decision 211);
--groups = {scene group: [scenes]} (labels from the logged path, e.g. m1/lists/groups_diag.json) for zeros by group.
Plain python3 + numpy on the box. Writes <name>_report.md, <name>_per_scene.csv, <name>_stats.json into --out.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c0b_report as R  # noqa: E402

LINE, SLOW = 0.010, 10


def ci2(d, logs):
    """Mean of d with the scene-resampled and the log-clustered 95% bootstrap CI (10 000 draws, seed 0 each)."""
    d = np.asarray(d, float)
    a = d[np.random.default_rng(0).integers(0, len(d), (R.B, len(d)))].mean(1)
    return dict(mean=float(d.mean()), scene=[float(x) for x in np.percentile(a, [2.5, 97.5])], log=list(R.ci(d, logs)[1:]))


def f2(c):
    return f"{c['mean']:+.4f} | [{c['scene'][0]:+.4f}, {c['scene'][1]:+.4f}] | [{c['log'][0]:+.4f}, {c['log'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True), ap.add_argument("--shards", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ot3"), ap.add_argument("--base", required=True), ap.add_argument("--groups"), ap.add_argument("--scenes")
    for k in ("group", "ref", "alt", "pair"):
        ap.add_argument(f"--{k}", action="append", default=[])
    a = ap.parse_args()
    man = {}
    for m in a.manifest:
        man |= json.loads(Path(m).read_text())
    spec = lambda g: (g.split("=")[0], g.split("=")[1].split("+"))  # noqa: E731
    bname, bmem = spec(a.base)
    G = {bname: bmem} | dict(spec(g) for g in a.group) | dict(spec(g) for g in a.ref)
    cand = [spec(g)[0] for g in a.group]
    miss = sorted({x for v in G.values() for x in v if x not in man})
    G = {g: v for g, v in G.items() if all(x in man for x in v)}
    assert bname in G, f"baseline drivers missing from the manifests: {miss}"
    names = list(dict.fromkeys(x for v in G.values() for x in v))
    D = {k: R.load_driver(man[k]) for k in names}
    shard = {l.split("\t")[0]: l.split("\t")[1].strip() for l in Path(a.shards).read_text().strip().split("\n")} if Path(a.shards).exists() else {}
    scenes = sorted(set.intersection(*(set(D[k][0]) for k in names)))
    if a.scenes:
        scenes = sorted(set(scenes) & set(Path(a.scenes).read_text().split()))
    cl = np.array([R.log_of(s) for s in scenes])
    sc = {k: np.array([D[k][0][s]["score"] for s in scenes]) for k in names}
    zc = {k: np.array([R.zclass(D[k][0][s]) for s in scenes]) for k in names}
    ev = {k: np.array([float((D[k][0][s].get("metrics") or {}).get("offroad_or_collision_at_fault", 0) or 0) for s in scenes]) for k in names}
    prog = {k: np.array([(D[k][0][s].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0 for s in scenes]) for k in names}
    L = [f"Scenes common to the {len(names)} drivers: {len(scenes)} from {len(set(cl))} logs, {len(set(shard.get(s, '?') for s in scenes))} shards. Bootstrap: {R.B} draws, "
         f"seed 0; `scenes` resamples scenes, `logs` resamples whole nuPlan logs (`date_vehicle`). Missing from the manifests: {miss or 'none'}.", ""]
    L += [f"- {k}: {len(D[k][0])} scored scenes, {len(man[k])} run dirs, disagreeing duplicate scenes {len(D[k][1])}" for k in names]
    st = {}
    L += ["", "## Drivers", "",
          "| driver | mean scene score [95% CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | at-fault events | mean progress |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in names:
        c = {f: int((zc[k] == f).sum()) for f in (*R.FAIL, "other")}
        st[k] = dict(mean=R.ci(sc[k], cl), n1=int((sc[k] >= 1).sum()), zeros=int((sc[k] == 0).sum()), **c, slow=int(((sc[k] > 0) & (sc[k] < 1)).sum()),
                     events=float(ev[k].sum()), progress=float(prog[k].mean()))
        s = st[k]
        L.append(f"| {k} | {R.fmt(s['mean'])} | {s['n1']} | {s['zeros']} | {c['collision_at_fault']} | {c['offroad']} | {c['left_corridor_laterally']} | {s['slow']} | "
                 f"{s['events']:.0f} | {s['progress']:.3f} |")
    gs = {g: np.mean([sc[x] for x in v], 0) for g, v in G.items()}
    gm = lambda g, q: float(np.mean([st[x][q] for x in G[g]]))  # noqa: E731
    L += ["", "## Recipes (per-scene mean over the training seeds; counts are seed means)", "",
          "| recipe | seeds | mean [95% CI, logs] | zeros | at-fault collision | offroad | left corridor | slow | at-fault events | seed 0 - seed 1 | 95% CI scenes | 95% CI logs | "
          "scenes where the seeds differ in zero / non-zero |", "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|:--|--:|"]
    for g, v in G.items():
        st[g] = dict(mean=R.ci(gs[g], cl), **{q: gm(g, q) for q in ("zeros", *R.FAIL, "slow", "events")})
        sd = f2(ci2(sc[v[0]] - sc[v[1]], cl)) if len(v) == 2 else " | | "
        flip = int(((sc[v[0]] == 0) != (sc[v[1]] == 0)).sum()) if len(v) == 2 else ""
        if len(v) == 2:
            st[g]["seed_diff"], st[g]["flips"] = ci2(sc[v[0]] - sc[v[1]], cl), flip
        L.append(f"| {g} | {' / '.join(f'{sc[x].mean():.4f}' for x in v)} | {R.fmt(st[g]['mean'])} | {gm(g, 'zeros'):g} | {gm(g, 'collision_at_fault'):g} | {gm(g, 'offroad'):g} | "
                 f"{gm(g, 'left_corridor_laterally'):g} | {gm(g, 'slow'):g} | {gm(g, 'events'):g} | {sd} | {flip} |")
    pd = {}
    L += ["", f"## Against {bname} (pre-registered line for the candidates: difference >= +{LINE} with the log-clustered CI lower bound > 0; at-fault events not higher; "
              f"slow scenes not more than {bname}'s + {SLOW})", "",
          f"| recipe - {bname} | difference | 95% CI scenes | 95% CI logs (decides) | score line | at-fault events (recipe / base) | events line | slow (recipe / base) | slow line | "
          "zero -> non-zero / non-zero -> zero (seed-mean scores) | candidate |", "|:--|--:|:--|:--|:--|--:|:--|--:|:--|:--|:--|"]
    for g in G:
        if g == bname:
            continue
        d = ci2(gs[g] - gs[bname], cl)
        o1, o2, o3 = d["mean"] >= LINE and d["log"][0] > 0, st[g]["events"] <= st[bname]["events"], st[g]["slow"] <= st[bname]["slow"] + SLOW
        fx, br = int(((gs[bname] == 0) & (gs[g] > 0)).sum()), int(((gs[bname] > 0) & (gs[g] == 0)).sum())
        pd[f"{g} - {bname}"] = dict(diff=d, score=bool(o1), events=bool(o2), slow=bool(o3), candidate=bool(o1 and o2 and o3) if g in cand else None, registered=g in cand)
        L.append(f"| {g} - {bname} | {f2(d)} | {'met' if o1 else 'not met'} | {st[g]['events']:g} / {st[bname]['events']:g} | {'met' if o2 else 'not met'} | "
                 f"{st[g]['slow']:g} / {st[bname]['slow']:g} | {'met' if o3 else 'not met'} | {fx} / {br} | "
                 + (f"**{'yes' if o1 and o2 and o3 else 'no'}**" if g in cand else "(reference)") + " |")
    L += ["", "Single checkpoints against the baseline recipe:", "", f"| driver - {bname} | difference | 95% CI scenes | 95% CI logs |", "|:--|--:|:--|:--|"]
    for k in names:
        if k not in bmem:
            pd[f"{k} - {bname}"] = ci2(sc[k] - gs[bname], cl)
            L.append(f"| {k} - {bname} | {f2(pd[f'{k} - {bname}'])} |")
    val = lambda k: gs[k] if k in gs else sc[k]  # noqa: E731
    prs = [p.split(":") for p in a.pair if all((x in gs or x in sc) for x in p.split(":"))]
    if prs:
        L += ["", "## Other paired differences", "", "| A - B | difference | 95% CI scenes | 95% CI logs |", "|:--|--:|:--|:--|"]
        for x, y in prs:
            pd[f"{x} - {y}"] = ci2(val(x) - val(y), cl)
            L.append(f"| {x} - {y} | {f2(pd[f'{x} - {y}'])} |")
    if a.groups:
        GR = json.loads(Path(a.groups).read_text())
        L += ["", "## Zeros and mean score by scene group (labels from the logged path; seed means)", "",
              "| recipe | " + " | ".join(f"{g} ({len(set(ss) & set(scenes))}): mean, zeros" for g, ss in GR.items()) + " |", "|:--|" + ":--|" * len(GR)]
        for g, v in G.items():
            cells = []
            for ss in GR.values():
                m = np.isin(scenes, ss)
                cells.append(f"{gs[g][m].mean():.4f}, {np.mean([(sc[x][m] == 0).sum() for x in v]):g}" if m.any() else "")
            L.append(f"| {g} | " + " | ".join(cells) + " |")
    if a.alt:
        L += ["", "## List-composition noise (the same checkpoint under another scene list / chunking)", "",
              "| checkpoint | scenes in both | mean here / other | scenes with a different score | zero / non-zero flips | mean abs difference |", "|:--|--:|:--|--:|--:|--:|"]
        for sp in a.alt:
            k, dirs = sp.split("=", 1)
            if k not in sc:
                continue
            A = R.load_driver([d for d in dirs.split(",") if (Path(d) / "aggregate/results-summary.json").exists()])[0]
            idx = [i for i, s in enumerate(scenes) if s in A]
            x, y = sc[k][idx], np.array([A[scenes[i]]["score"] for i in idx])
            pd[f"alt {k}"] = dict(n=len(idx), here=float(x.mean()), other=float(y.mean()), differ=int((np.abs(x - y) > 1e-9).sum()), flips=int(((x == 0) != (y == 0)).sum()))
            L.append(f"| {k} | {len(idx)} | {x.mean():.4f} / {y.mean():.4f} | {(np.abs(x - y) > 1e-9).sum()} | {((x == 0) != (y == 0)).sum()} | {np.abs(x - y).mean():.4f} |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{a.name}_per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "shard"] + [f"{k}|{c}" for k in names for c in ("score", "zero_class", "progress", "atfault_events")])
        for i, s in enumerate(scenes):
            w.writerow([s, cl[i], shard.get(s, "")] + [x for k in names for x in (round(float(sc[k][i]), 6), zc[k][i], round(float(prog[k][i]), 4), float(ev[k][i]))])
    (out / f"{a.name}_stats.json").write_text(json.dumps(dict(scenes=len(scenes), logs=len(set(cl)), stats=st, lines=pd, missing=miss), indent=1, default=float))
    (out / f"{a.name}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
