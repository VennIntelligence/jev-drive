#!/usr/bin/env python3
"""ONPOL-LONG round 1 read-out (plans/2026-10-12-ol1-prereg.md, "Read" and "Registered lines"). AlpaSim's venv (pandas, polars).

  ol_report.py --family Y|P --runs <tag>@18=<run dir> ... --out DIR
      collects with alpasim_long's long_read.collect (the evaluator's truncation; per-decision distance to the logged path), then for the family:
      arm <F>SKC-L18-s0/s1, control <F>NONE-L18-s0/s1, base <base>-s0 (and -s1 for the null and CL-1's base mean), all on the same held scenes.
      -> DIR/per_scene.csv, per_scene_k.parquet (long_read), DIR/report_<F>.md, report_<F>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/alpasim_long/scripts"))
import long_read as LR  # noqa: E402

BASE = {"Y": "YR10m10-F", "P": "P2H10S-F"}
B = 10000


def main(a):
    import pandas as pd
    out = Path(a.out)
    LR.collect(argparse.Namespace(runs=a.runs, out=str(out), jobs=a.jobs, scenes=None))
    S = pd.read_csv(out / "per_scene.csv")
    K = pd.read_parquet(out / "per_scene_k.parquet")
    S["tag"] = S.label.str.split("@").str[0]
    f, base = a.family, BASE[a.family]
    arm, ctl, bas = [f"{f}{a.arm}-L18-s{s}" for s in (0, 1)], [f"{f}{a.ctl}-L18-s{s}" for s in (0, 1)], [f"{base}-s{s}" for s in (0, 1)]
    if a.base_tags:                                                              # Amendment 5: the previous round's checkpoint as the reference
        bas = a.base_tags.split(",")
        base = bas[0].rsplit("-s", 1)[0]
    have = set(S.tag)
    assert set(arm + ctl + bas[:1]) <= have, f"missing reads: {sorted(set(arm + ctl + bas[:1]) - have)}"
    bas = [t for t in bas if t in have]
    scenes = sorted(set.intersection(*[set(S[S.tag == t].scene) for t in arm + ctl + bas]))
    fold = sorted(set(S[S.scene.isin(scenes) & S.offroad_or_collision_at_fault.isna()].scene))   # no rollout (route fold): not a scene of the read
    scenes = [x for x in scenes if x not in set(fold)]
    S = S[S.scene.isin(scenes)].copy()
    S["log"] = S.scene.str.rsplit("-", n=1).str[0]
    z = S.score == 0
    S["zero"], S["z_col"] = z.astype(int), (z & (S.collision_at_fault > 0)).astype(int)
    S["z_off"], S["z_cor"] = (z & (S.offroad > 0)).astype(int), (z & (S.left_corridor_laterally > 0)).astype(int)
    S["z_atf"] = (z & (S.offroad_or_collision_at_fault > 0)).astype(int)
    S["z_oth"] = (z & ~(S.collision_at_fault > 0) & ~(S.offroad > 0) & ~(S.left_corridor_laterally > 0)).astype(int)
    S["gt"] = S.dist_to_gt_trajectory
    piv = {c: S.pivot(index="scene", columns="tag", values=c) for c in ("score", "zero", "z_col", "z_off", "z_cor", "z_oth", "z_atf", "gt")}
    logs = S.drop_duplicates("scene").set_index("scene").log.reindex(scenes)
    L = np.array(sorted(logs.unique()))
    li = np.searchsorted(L, logs.to_numpy())
    W = np.stack([np.bincount(d, minlength=len(L)) for d in np.random.default_rng(0).integers(0, len(L), (B, len(L)))])[:, li]   # (B, scenes)

    def slope(tags):
        kk = K[K.label.isin([f"{t}@{a.length}" for t in tags]) & K.scene.isin(scenes) & (K.k >= 4)].groupby("k").dist_gt.mean()
        if len(kk) < 2:
            return float("nan")
        x = kk.index.to_numpy(float) - kk.index.to_numpy(float).mean()
        return float(((kk.to_numpy() - kk.mean()) * x).sum() / (x * x).sum())

    def fam(tags):
        r = {c: float(piv[c][tags].mean(1).mean() if c in ("score", "gt") else piv[c][tags].sum(0).mean()) for c in piv}   # gt: nan-skipping
        return r | dict(seeds={t: dict(score=float(piv["score"][t].mean()), zeros=int(piv["zero"][t].sum()), atf=int(piv["z_atf"][t].sum()),
                                       gt=float(piv["gt"][t].mean())) for t in tags}, slope=slope(tags))

    def diff(A, Bt, c, total=False):
        d = (piv[c][A].mean(1) - piv[c][Bt].mean(1)).to_numpy() * (len(scenes) if total else 1)
        ok = np.isfinite(d)                                                      # GT distance: scenes where every label has one
        bs = (W[:, ok] @ d[ok]) / W[:, ok].sum(1)
        return dict(diff=float(d[ok].mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(ok.sum()))

    def fb(A, Bt):   # fixed = zero in every seed of B, in no seed of A; broken = the reverse
        za, zb = piv["zero"][A], piv["zero"][Bt]
        return int(((zb.min(1) == 1) & (za.max(1) == 0)).sum()), int(((za.min(1) == 1) & (zb.max(1) == 0)).sum())
    null = [abs(x - y) for p, q in ((ctl[0], ctl[1]),) + (((bas[0], bas[1]),) if len(bas) > 1 else ()) for x, y in [fb([p], [q])]]
    R = dict(family=f, base=base, scenes=len(scenes), dropped_no_rollout=fold, logs=len(L), arm=fam(arm), control=fam(ctl), base_s0=fam(bas[:1]), base_seeds=fam(bas))
    cols = ("score", "zero", "z_atf", "z_col", "z_off", "z_cor", "z_oth", "gt")
    R["pairs"] = {f"{n1} - {n2}": {c: diff(A, Bt, c, c.startswith("z")) for c in cols} | {"fixed_broken": fb(A, Bt)}
                  for n1, A, n2, Bt in (("arm", arm, "control", ctl), ("arm", arm, f"reference {bas[0]}", bas[:1]), ("control", ctl, f"reference {bas[0]}", bas[:1]))}
    R["null_max"] = max(null)
    ac, cc = R["arm"], R["control"]
    fx, br = R["pairs"]["arm - control"]["fixed_broken"]
    R["adopt"] = dict(fewer_zeros=ac["zero"] < cc["zero"], atfault_not_above=ac["z_atf"] <= cc["z_atf"], gt_not_larger=ac["gt"] <= cc["gt"])
    R["adopt"]["adopted"] = all(R["adopt"].values())
    bc = R["base_s0"]                                                           # closed-loop.md round gate (Amendment 5): against the reference
    R["round_gate"] = dict(fewer_zeros=ac["zero"] < bc["zero"], atfault_not_above=ac["z_atf"] <= bc["z_atf"], gt_not_larger=ac["gt"] <= bc["gt"])
    R["round_gate"]["passed"] = all(R["round_gate"].values())
    sc = R["pairs"]["arm - control"]["score"]["ci"]
    R["cl1_strict"] = dict(score_ci_low_gt0=sc[0] > 0, atfault_not_above_base_mean=ac["z_atf"] <= R["base_seeds"]["z_atf"],
                           gt_lower_by_010=cc["gt"] - ac["gt"] >= 0.10, fixed_minus_broken_over_null=(fx - br) > R["null_max"])
    R["cl1_strict"]["met"] = all(R["cl1_strict"].values())
    (out / f"report_{f}{a.arm}.json").write_text(json.dumps(R, indent=1) + "\n")
    row = lambda n, x: (f"| {n} | {x['score']:.4f} | {x['zero']:.1f} | {x['z_col']:.1f} / {x['z_off']:.1f} / {x['z_cor']:.1f} / {x['z_oth']:.1f} | {x['z_atf']:.1f} | {x['gt']:.3f} | "  # noqa: E731
                        f"{x['slope']:.4f} | " + " ; ".join(f"{t} {v['score']:.4f} / {v['zeros']} / {v['atf']} / {v['gt']:.3f}" for t, v in x["seeds"].items()) + " |")
    md = [f"# ONPOL-LONG round 1, family {f} (base {base}-s0): {len(scenes)} scenes, length `{a.length}`, {len(L)} logs", "",
          f"Dropped for every label alike: {len(fold)} scenes without a rollout in at least one label (route generation failed: route folds back).", "",
          "| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |",
          "|:--|--:|--:|:--|--:|--:|--:|:--|", row(f"arm {f}{a.arm}-L18 ({ {'SKC': 'S + K on-policy rows (C dropped by K0)', 'S': 'S hinges only on the same rows (Amendment 3)', 'S2': 'round 2: S rows of YS-L18 rollouts, from YS-L18-s1'}.get(a.arm, a.arm)})", ac), row(f"control {f}NONE-L18 (same continuation, no rows)", cc),
          row(f"reference {bas[0]}", R["base_s0"]), row(f"reference {', '.join(bas)}", R["base_seeds"]), "",
          "Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:", "",
          "| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |", "|:--|:--|:--|:--|:--|:--|:--|"]
    ci = lambda v, p=4: f"{v['diff']:+.{p}f} [{v['ci'][0]:+.{p}f}, {v['ci'][1]:+.{p}f}]"  # noqa: E731
    for n, p in R["pairs"].items():
        md.append(f"| {n} | {ci(p['score'])} | {ci(p['zero'], 1)} | {ci(p['z_atf'], 1)} | {p['z_col']['diff']:+.1f} / {p['z_off']['diff']:+.1f} / {p['z_cor']['diff']:+.1f} / {p['z_oth']['diff']:+.1f} | "
                  f"{ci(p['gt'], 3)} ({p['gt']['n']} scenes) | {p['fixed_broken'][0]} / {p['fixed_broken'][1]} |")
    md += ["", f"Null of fixed - broken: max |fixed - broken| of the single-seed comparisons {f}NONE-L18 s0 / s1" + (f" and {base} s0 / s1" if len(bas) > 1 else "")
           + f" = {R['null_max']}.", "",
           f"**Adoption line (registered): {'ADOPTED' if R['adopt']['adopted'] else 'NOT adopted'}** - fewer zeros than the control {R['adopt']['fewer_zeros']} "
           f"({ac['zero']:.1f} vs {cc['zero']:.1f}); at-fault zeros not above the control's (hard gate) {R['adopt']['atfault_not_above']} ({ac['z_atf']:.1f} vs {cc['z_atf']:.1f}); "
           f"GT distance not larger {R['adopt']['gt_not_larger']} ({ac['gt']:.3f} vs {cc['gt']:.3f} m).", "",
           f"Round gate (closed-loop.md; registered for round 2 in Amendment 5) against {bas[0]}: {'PASSED' if R['round_gate']['passed'] else 'not passed'} - "
           f"{json.dumps({k: bool(v) for k, v in R['round_gate'].items()})} (zeros {ac['zero']:.1f} vs {bc['zero']:.1f}, at-fault {ac['z_atf']:.1f} vs {bc['z_atf']:.1f}, GT {ac['gt']:.3f} vs {bc['gt']:.3f}).", "",
           f"CL-1 as written in closed-loop.md (reported beside it): {'met' if R['cl1_strict']['met'] else 'not met'} - {json.dumps({k: bool(v) for k, v in R['cl1_strict'].items()})}.", ""]
    (out / f"report_{f}{a.arm}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(BASE)), ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True), ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--ctl", default="NONE", help="control family suffix (S2NONE for round 2)")
    ap.add_argument("--base-tags", default="", help="reference tags, first = the paired one (default <family base>-s0, -s1)")
    ap.add_argument("--arm", default="SKC", help="arm family suffix: SKC (round 1) or S (Amendment 3)")
    ap.add_argument("--length", default="18", help="label suffix after @ (18 for the held read, pai for the Tokyo transfer read)")
    main(ap.parse_args())
