#!/usr/bin/env python3
"""BODY1, prereg Amendment 6 note (h): the base's own seed spread in the AlpaSim nuPlan closed loop (700 scenes, P2H10-F seeds 0 to 3).

Per seed: mean scene score, zeros by class, slow scenes, mean progress, the > 45 deg bucket. Per ordered seed pair (one seed read as if it
were an arm against another as the base): the quantities of the lines L1a / L1b / L2 / L3 of Amendment 4 item 6. No line, a description of
what two runs of the same recipe differ by.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/base_seeds.py --man <manifest.json> [<manifest.json> ...] --out experiments/body1/results/shape/base_seeds
Manifests are ot2_loop.py's {label: [run dirs]}; labels P2H10-F-s<k>. Scores from aggregate/results-summary.json through c0b_report.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import c0b_report as R  # noqa: E402
import stop_report as SR  # noqa: E402

CLS = ("collision", "offroad", "corridor")


def main(a):
    from jevdrive import stats
    from jevdrive.run import Run
    with Run("body1", "base-seeds", config=vars(a)) as run:
        man = {}
        for m in a.man:
            man |= {k: v for k, v in json.loads(_pl.Path(m).read_text()).items() if k.startswith(a.arm + "-s")}
        names = sorted(man)
        Rs = {k: R.load_driver(man[k])[0] for k in names}
        scenes = sorted(set.intersection(*(set(Rs[k]) for k in names)))
        logs = np.array([R.log_of(s) for s in scenes])
        T = SR.turns()
        gt45 = np.abs(np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])) > 45
        S = {k: SR.summary(Rs[k], scenes) for k in names}
        sc, zc = {k: S[k][0] for k in names}, {k: np.array([SR.SHORT[z] for z in S[k][1]]) for k in names}
        prog = {k: np.array([(Rs[k][s].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0 for s in scenes]) for k in names}
        slow = lambda x: (x > 0) & (x < 1)  # noqa: E731
        L = [f"Scenes common to the {len(names)} runs: {len(scenes)} from {len(set(logs))} logs; > 45 deg: {int(gt45.sum())}.", "",
             "| driver | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | collision + offroad | slow | mean progress | > 45 deg mean | > 45 deg zeros |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        st = dict(scenes=len(scenes), drivers={}, pairs={})
        for k in names:
            s = S[k][2]
            st["drivers"][k] = s | dict(mean_ci=R.ci(sc[k], logs), progress=float(prog[k].mean()), gt45_mean=float(sc[k][gt45].mean()), gt45_zeros=int((sc[k][gt45] == 0).sum()))
            L.append(f"| {k} | {R.fmt(R.ci(sc[k], logs))} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['collision'] + s['offroad']} | {s['slow']} | "
                     f"{prog[k].mean():.3f} | {sc[k][gt45].mean():.4f} | {int((sc[k][gt45] == 0).sum())} |")
        col = lambda f: [st["drivers"][k][f] for k in names]  # noqa: E731
        tz = [st["drivers"][k]["collision"] + st["drivers"][k]["offroad"] + st["drivers"][k]["corridor"] for k in names]
        L += ["", f"Range over the seeds: mean {min(col('mean')):.4f} to {max(col('mean')):.4f}; collision + offroad + corridor zeros {min(tz)} to {max(tz)}; at-fault collision "
              f"{min(col('collision'))} to {max(col('collision'))}; offroad {min(col('offroad'))} to {max(col('offroad'))}; corridor {min(col('corridor'))} to {max(col('corridor'))}; "
              f"slow {min(col('slow'))} to {max(col('slow'))}; progress {min(col('progress')):.3f} to {max(col('progress')):.3f}.", "",
              "## One seed read as an arm against another as the base (same recipe; the lines' quantities)", "",
              "| \"arm\" vs \"base\" | L1a zeros (collision / offroad / corridor) | L1b collision + offroad | L2 mean difference [95 % CI by log] | L3 slow vs 1.1 x base | removed / new zeros | score 1 -> slow / slow -> 1 | lines that would read as met |",
              "|:--|:--|:--|:--|:--|:--|:--|:--|"]
        for x, y in itertools.permutations(names, 2):
            ta, tb = [int((zc[x] == f).sum()) for f in CLS], [int((zc[y] == f).sum()) for f in CLS]
            r = stats.paired(sc[x], sc[y], groups=logs)
            sa, sb = int(slow(sc[x]).sum()), int(slow(sc[y]).sum())
            ok = dict(L1a=sum(ta) < sum(tb), L1b=sum(ta[:2]) < sum(tb[:2]), L2=bool(r["mean"] >= 0 and r["lo"] > -0.005), L3=sa <= 1.1 * sb)
            rem, new = int(((sc[y] == 0) & (sc[x] > 0)).sum()), int(((sc[y] > 0) & (sc[x] == 0)).sum())
            ts, fs = int(((sc[y] == 1) & slow(sc[x])).sum()), int((slow(sc[y]) & (sc[x] == 1)).sum())
            st["pairs"][f"{x} vs {y}"] = dict(zeros_arm=ta, zeros_base=tb, diff=r, slow_arm=sa, slow_base=sb, removed=rem, new=new, to_slow=ts, from_slow=fs, lines=ok)
            L.append(f"| {x[-2:]} vs {y[-2:]} | {sum(ta)} ({' / '.join(map(str, ta))}) vs {sum(tb)} ({' / '.join(map(str, tb))}) | {sum(ta[:2])} vs {sum(tb[:2])} | "
                     f"{r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {sa} vs {1.1 * sb:.1f} | {rem} / {new} | {ts} / {fs} | {', '.join(q for q, v in ok.items() if v) or '-'} |")
        P = st["pairs"].values()
        d = np.array([p["diff"]["mean"] for p in P])
        zero_any = np.array([(sc[k] == 0) for k in names])
        st["summary"] = dict(abs_mean_diff_max=float(np.abs(d).max()), ci_halfwidth_mean=float(np.mean([(p["diff"]["hi"] - p["diff"]["lo"]) / 2 for p in P])),
                             flips_per_pair_mean=float(np.mean([p["removed"] + p["new"] for p in P])), zero_in_all=int(zero_any.all(0).sum()), zero_in_any=int(zero_any.any(0).sum()),
                             to_slow_mean=float(np.mean([p["to_slow"] for p in P])), all_four_lines=int(sum(all(p["lines"].values()) for p in P)), l2_met=int(sum(p["lines"]["L2"] for p in P)),
                             l3_met=int(sum(p["lines"]["L3"] for p in P)), pairs=len(d))
        m = st["summary"]
        L += ["", f"Over the {m['pairs']} ordered pairs: largest |mean difference| {m['abs_mean_diff_max']:.4f}, mean CI half-width {m['ci_halfwidth_mean']:.4f}; zero flips per pair "
              f"{m['flips_per_pair_mean']:.1f} on average; scenes with a zero in all seeds {m['zero_in_all']}, in any seed {m['zero_in_any']}; score 1 -> slow per pair {m['to_slow_mean']:.1f}; "
              f"L2 would read as met in {m['l2_met']}, L3 in {m['l3_met']}, all four lines in {m['all_four_lines']}."]
        out = R.REPO / a.out if hasattr(R, "REPO") else _H.parents[3] / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text("\n".join(L) + "\n")
        out.with_suffix(".json").write_text(json.dumps(st, indent=1, default=float) + "\n")
        run.info("\n" + "\n".join(L))
        run.summary.update(m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--man", nargs="+", required=True)
    ap.add_argument("--arm", default="P2H10-F")
    ap.add_argument("--out", default="experiments/body1/results/shape/base_seeds")
    main(ap.parse_args())
