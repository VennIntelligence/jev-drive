#!/usr/bin/env python3
"""Lane OT2 read-out (decisions 210 / 211): any set of drivers on the same public AlpaSim scenes -> scores, zero classes, seed-mean groups,
paired differences with log-clustered bootstrap CIs, the pre-registered lines of pieces B and C, the dose-response table.

  ot2_report.py --manifest c0b/manifest.json ot2/b/manifest.json --shards c0b/lists/shards.tsv --out DIR --name b \
      --group AP2=AP2-AB-s0+AP2-AB-s1 --group APO-a05m10=APO-a05m10-s0+APO-a05m10-s1 ... --cand APO-a05m10:AP2 ... --ens ENS-OT30:OT30-F-s0+OT30-F-s1
manifest = {driver label: [run dirs]} (later files win); --group NAME=label+label = per-scene mean of training seeds; --cand G:BASE = piece B's
line for group G against group BASE; --ens LABEL:member+member = piece C's line; --pair A:B = extra paired differences (labels or groups).
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

LINE_B, SLOW_B, LINE_C = 0.015, 10, 0.008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True), ap.add_argument("--shards", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ot2"), ap.add_argument("--drivers", nargs="*", default=[])
    for k in ("group", "cand", "ens", "pair"):
        ap.add_argument(f"--{k}", action="append", default=[])
    a = ap.parse_args()
    man = {}
    for m in a.manifest:
        man |= json.loads(Path(m).read_text())
    groups = {g.split("=")[0]: g.split("=")[1].split("+") for g in a.group}
    want = list(dict.fromkeys(a.drivers + [x for v in groups.values() for x in v] + [e.split(":")[0] for e in a.ens] + [x for e in a.ens for x in e.split(":")[1].split("+")]
                              + [x for p in a.pair for x in p.split(":") if x not in groups]))
    names = [k for k in want if k in man]
    miss = [k for k in want if k not in man]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = {l.split("\t")[0]: l.split("\t")[1].strip() for l in Path(a.shards).read_text().strip().split("\n")}
    D = {k: R.load_driver(man[k]) for k in names}
    scenes = sorted(set.intersection(*(set(D[k][0]) for k in names)) & set(shard))
    cl = np.array([R.log_of(s) for s in scenes])
    sc = {k: np.array([D[k][0][s]["score"] for s in scenes]) for k in names}
    zc = {k: [R.zclass(D[k][0][s]) for s in scenes] for k in names}
    ev = {k: np.array([float((D[k][0][s].get("metrics") or {}).get("offroad_or_collision_at_fault", 0) or 0) for s in scenes]) for k in names}
    prog = {k: np.array([D[k][0][s]["score_metrics"].get("progress_clipped_rel", 0) for s in scenes]) for k in names}
    L = [f"Scenes common to the {len(names)} drivers: {len(scenes)} from {len(set(cl))} logs, {len(set(shard[s] for s in scenes))} shards. CIs: 95% bootstrap "
         f"resampling whole logs (nuPlan `date_vehicle`), {R.B} draws, seed 0. Missing from the manifests: {miss or 'none'}.", ""]
    L += [f"- {k}: {len(D[k][0])} scored scenes, {len(man[k])} run dirs, disagreeing duplicate scenes {len(D[k][1])}" for k in names]
    st = {}
    L += ["", "## Scores", "",
          "| driver | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | slow (0 < score < 1) | at-fault events | mean progress |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in names:
        c = {f: sum(x == f for x in zc[k]) for f in (*R.FAIL, "other")}
        st[k] = dict(mean=R.ci(sc[k], cl), n1=int((sc[k] >= 1).sum()), zeros=int((sc[k] == 0).sum()), **c, slow=int(((sc[k] > 0) & (sc[k] < 1)).sum()),
                     events=float(ev[k].sum()), progress=float(prog[k].mean()))
        s = st[k]
        L.append(f"| {k} | {R.fmt(s['mean'])} | {s['n1']} | {s['zeros']} | {c['collision_at_fault']} | {c['offroad']} | {c['left_corridor_laterally']} | {s['slow']} | "
                 f"{s['events']:.0f} | {s['progress']:.3f} |")
    # ---- seed-mean groups
    G = {g: v for g, v in groups.items() if all(x in sc for x in v)}
    gs = {g: np.mean([sc[x] for x in v], 0) for g, v in G.items()}
    gm = lambda g, q: float(np.mean([st[x][q] for x in G[g]]))  # noqa: E731
    if G:
        L += ["", "## Seed means (per-scene mean over the training seeds of a recipe; counts are seed means)", "",
              "| recipe | seeds | mean scene score [95% CI] | zeros | at-fault collision | offroad + corridor | slow | at-fault events | seed difference [95% CI] | scenes where the seeds differ in zero / non-zero |",
              "|:--|:--|:--|--:|--:|--:|--:|--:|:--|--:|"]
        for g, v in G.items():
            sd = R.fmt(R.ci(sc[v[0]] - sc[v[1]], cl), sign=True) if len(v) == 2 else ""
            flip = int(((sc[v[0]] == 0) != (sc[v[1]] == 0)).sum()) if len(v) == 2 else ""
            st[g] = dict(mean=R.ci(gs[g], cl), zeros=gm(g, "zeros"), collision_at_fault=gm(g, "collision_at_fault"), slow=gm(g, "slow"), events=gm(g, "events"))
            L.append(f"| {g} | {' / '.join(f'{sc[x].mean():.4f}' for x in v)} | {R.fmt(st[g]['mean'])} | {gm(g, 'zeros'):g} | {gm(g, 'collision_at_fault'):g} | "
                     f"{gm(g, 'offroad') + gm(g, 'left_corridor_laterally'):g} | {gm(g, 'slow'):g} | {gm(g, 'events'):g} | {sd} | {flip} |")
    val = lambda k: gs[k] if k in gs else sc[k]  # noqa: E731
    pd = {}
    # ---- piece B lines
    cands = [c.split(":") for c in a.cand if c.split(":")[0] in gs and c.split(":")[1] in gs]
    if cands:
        L += ["", f"## Piece B lines (pre-registered: two-seed mean minus the baseline's two-seed mean >= +{LINE_B} with CI lower bound > 0; at-fault events not higher; "
                  f"slow scenes not more than the baseline's + {SLOW_B})", "",
              "| recipe - baseline | difference [95% CI] | score line | at-fault events (recipe / baseline) | events line | slow (recipe / baseline) | slow line | candidate |",
              "|:--|:--|:--|--:|:--|--:|:--|:--|"]
        for g, b in cands:
            d = R.ci(gs[g] - gs[b], cl)
            o1, o2, o3 = d[0] >= LINE_B and d[1] > 0, st[g]["events"] <= st[b]["events"], st[g]["slow"] <= st[b]["slow"] + SLOW_B
            pd[f"{g} - {b}"] = dict(diff=d, score=o1, events=o2, slow=o3, candidate=o1 and o2 and o3)
            L.append(f"| {g} - {b} | {R.fmt(d, sign=True)} | {'met' if o1 else 'not met'} | {st[g]['events']:g} / {st[b]['events']:g} | {'met' if o2 else 'not met'} | "
                     f"{st[g]['slow']:g} / {st[b]['slow']:g} | {'met' if o3 else 'not met'} | **{'yes' if o1 and o2 and o3 else 'no'}** |")
    # ---- dose response (amplitude x share), when the four recipes are there
    cell = {(am, m): next((g for g in gs if g.endswith(f"{am}{m}")), None) for am in ("a05", "a15") for m in ("m10", "m25")}
    if all(cell.values()) and cands:
        b = cands[0][1]
        L += ["", f"## Dose response (two-seed means minus {b}; rows = amplitude, columns = batch share)", "", "| amplitude | 10 % | 25 % | row mean |", "|:--|:--|:--|:--|"]
        for am in ("a05", "a15"):
            L.append(f"| {am} | " + " | ".join(R.fmt(R.ci(gs[cell[am, m]] - gs[b], cl), sign=True) for m in ("m10", "m25")) + " | "
                     + R.fmt(R.ci((gs[cell[am, 'm10']] + gs[cell[am, 'm25']]) / 2 - gs[b], cl), sign=True) + " |")
        L.append("| column mean | " + " | ".join(R.fmt(R.ci((gs[cell['a05', m]] + gs[cell['a15', m]]) / 2 - gs[b], cl), sign=True) for m in ("m10", "m25")) + " | "
                 + R.fmt(R.ci(np.mean([gs[c] for c in cell.values()], 0) - gs[b], cl), sign=True) + " |")
        amp = R.ci((gs[cell["a15", "m10"]] + gs[cell["a15", "m25"]] - gs[cell["a05", "m10"]] - gs[cell["a05", "m25"]]) / 2, cl)
        shr = R.ci((gs[cell["a05", "m25"]] + gs[cell["a15", "m25"]] - gs[cell["a05", "m10"]] - gs[cell["a15", "m10"]]) / 2, cl)
        pd["dose"] = dict(amplitude=amp, share=shr)
        L += ["", f"Main effects: amplitude (1.5 m - 0.5 m) {R.fmt(amp, sign=True)}; share (25 % - 10 %) {R.fmt(shr, sign=True)}.", "",
              "| recipe | zeros | at-fault collision | offroad + corridor | slow | at-fault events |", "|:--|--:|--:|--:|--:|--:|"]
        L += [f"| {g} | {gm(g, 'zeros'):g} | {gm(g, 'collision_at_fault'):g} | {gm(g, 'offroad') + gm(g, 'left_corridor_laterally'):g} | {gm(g, 'slow'):g} | {gm(g, 'events'):g} |"
              for g in [b, *cell.values()]]
    # ---- piece C lines
    for e in a.ens:
        lab, mem = e.split(":")[0], e.split(":")[1].split("+")
        if lab not in sc or not all(m in sc for m in mem):
            continue
        best = max(mem, key=lambda m: sc[m].mean())
        d, orc = R.ci(sc[lab] - sc[best], cl), np.max([sc[m] for m in mem], 0)
        o1, o2 = d[0] >= LINE_C and d[1] > 0, st[lab]["events"] <= st[best]["events"]
        zm = [sc[m] == 0 for m in mem]
        any0, all0 = np.any(zm, 0), np.all(zm, 0)
        pd[f"{lab} - {best}"] = dict(diff=d, score=o1, events=o2, adopted=o1 and o2)
        L += ["", f"## Piece C line: {lab} (pre-registered: ensemble minus its better member >= +{LINE_C} with CI lower bound > 0; at-fault events not higher)", "",
              f"- members {', '.join(f'{m} {sc[m].mean():.4f}' for m in mem)}; better member {best}; ensemble {sc[lab].mean():.4f}; per-scene best-of-members "
              f"{orc.mean():.4f} ({R.fmt(R.ci(orc - sc[best], cl), sign=True)} over the better member)",
              f"- ensemble minus better member: {R.fmt(d, sign=True)} -> score line **{'met' if o1 else 'not met'}**; at-fault events {st[lab]['events']:.0f} vs "
              f"{st[best]['events']:.0f} -> **{'met' if o2 else 'not met'}**; verdict: **{'adopted' if o1 and o2 else 'not adopted'}**",
              f"- ensemble minus the mean of its members: {R.fmt(R.ci(sc[lab] - np.mean([sc[m] for m in mem], 0), cl), sign=True)}",
              f"- scenes where exactly some members are zero ({int((any0 & ~all0).sum())}): ensemble zero in {int(((sc[lab] == 0) & any0 & ~all0).sum())}; "
              f"all members zero ({int(all0.sum())}): ensemble zero in {int(((sc[lab] == 0) & all0).sum())}; no member zero ({int((~any0).sum())}): ensemble zero in "
              f"{int(((sc[lab] == 0) & ~any0).sum())}",
              f"- zero classes of the ensemble: collision {st[lab]['collision_at_fault']}, offroad {st[lab]['offroad']}, corridor {st[lab]['left_corridor_laterally']}; "
              f"slow {st[lab]['slow']} (members {', '.join(str(st[m]['slow']) for m in mem)})"]
    # ---- extra pairs
    prs = [p.split(":") for p in a.pair if all((x in gs or x in sc) for x in p.split(":"))]
    if prs:
        L += ["", "## Paired differences", "", "| A - B | mean scene score difference [95% CI] |", "|:--|:--|"]
        for x, y in prs:
            pd[f"{x} - {y}"] = R.ci(val(x) - val(y), cl)
            L.append(f"| {x} - {y} | {R.fmt(pd[f'{x} - {y}'], sign=True)} |")
    with (out / f"{a.name}_per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "shard"] + [f"{k}|{c}" for k in names for c in ("score", "zero_class", "progress", "atfault_events")])
        for i, s in enumerate(scenes):
            w.writerow([s, cl[i], shard[s]] + [x for k in names for x in (round(float(sc[k][i]), 6), zc[k][i], round(float(prog[k][i]), 4), float(ev[k][i]))])
    (out / f"{a.name}_stats.json").write_text(json.dumps(dict(scenes=len(scenes), stats=st, lines=pd, missing=miss), indent=1, default=float))
    (out / f"{a.name}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
