#!/usr/bin/env python3
"""BODY1 arm 4.3 closed-loop read (plans/2026-10-10-body1-prereg.md Amendment 4 item 6, Amendment 5): P2H10B-F (loss arm) against P2H10-F.

  check   --run <dir>[,<dir>] --base <dir>[,<dir>] --out DIR [--expect N]     the one-chunk checklist (items 1, 2, 4; the heading item is read by replan_heading.py)
  report  --base-man M --man M [M ...] --out DIR [--arm P2H10B-F] --dev <scene list>:<seed> [<scene list>:<seed> ...]
          the full read on both readings: all scenes x seeds, and the (seed, scene) pairs not in the development lists -> report.md, per_scene.csv, flips.csv, stats.json
Manifests are ot2_loop.py's {label: [run dirs]}; arm labels <arm>-s0 / <arm>-s1, baseline labels P2H10-F-s0 / P2H10-F-s1.
Box, envs/op-train python. Scores from aggregate/results-summary.json. Zero class = first failing flag (collision_at_fault, offroad, left_corridor_laterally).
> 45 deg = |heading change of the logged 4 s future of the scene's navtest token| (bench TURN_BINS; op_parity cache lb_navtest).
Lines (stricter reading decides): L1a at-fault collision + offroad + corridor zeros down in the two-seed total and in neither seed up; L1b the same for collision + offroad;
L2 mean per-scene difference >= 0 with the log-clustered lower bound > -0.005; L3 slow scenes (0 < score < 1) <= 1.1 x base.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import c0b_report as R  # noqa: E402
import stop_report as SR  # noqa: E402

BASE, SHORT = SR.BASE, SR.SHORT
CLS = ("collision", "offroad", "corridor")
slow = lambda x: (x > 0) & (x < 1)  # noqa: E731


def cmd_check(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run, base = a.run.split(","), a.base.split(",")
    Rr, Rb = R.load_driver(run)[0], R.load_driver(base)[0]
    scenes = sorted(set(Rr) & set(Rb))
    sc, zc, sr = SR.summary(Rr, scenes)
    scb, zcb, sbs = SR.summary(Rb, scenes)
    rem, new = (scb == 0) & (sc > 0), (scb > 0) & (sc == 0)
    taught = lambda z: np.isin(z, ["collision_at_fault", "offroad"])  # noqa: E731
    newco = new & taught(zc)
    remco = rem & taught(zcb)
    chk = [("rollouts complete", f"{len(set(Rr))} / {a.expect}", len(set(Rr)) == a.expect),
           ("taught-class zeros (collision + offroad) not above the baseline's", f"{int(taught(zc).sum())} against {int(taught(zcb).sum())}", int(taught(zc).sum()) <= int(taught(zcb).sum())),
           ("baseline-clean scenes turning into a collision / offroad zero not more than zeros removed", f"new {int(newco.sum())} against removed (collision / offroad) {int(remco.sum())}", int(newco.sum()) <= int(remco.sum()))]
    L = ["| check | value | verdict |", "|:--|:--|:--|"] + [f"| {n} | {v} | {'pass' if p else 'FAIL'} |" for n, v, p in chk]
    L += ["", "| run | n | mean | score 1 | zeros | collision | offroad | corridor | slow |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for n, s in (("new", sr), ("baseline run", sbs)):
        L.append(f"| {n} | {s['n']} | {s['mean']:.4f} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['slow']} |")
    L += ["", "Removed zeros: " + (", ".join(f"{s[-16:]} ({SHORT[zcb[i]]})" for i, s in enumerate(scenes) if rem[i]) or "-"),
          "New zeros: " + (", ".join(f"{s[-16:]} ({SHORT[zc[i]]})" for i, s in enumerate(scenes) if new[i]) or "-")]
    (out / "check.md").write_text("\n".join(L) + "\n")
    (out / "check.json").write_text(json.dumps(dict(checks=[(n, v, p) for n, v, p in chk], run=sr, base=sbs), indent=1, default=float))
    print("\n".join(L))
    if any(not p for _, _, p in chk):
        raise SystemExit("checklist FAILED")


def cmd_report(a):
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.base_man).read_text())
    for m in a.man:
        for k, v in json.loads(Path(m).read_text()).items():
            man[k] = (man[k] if k in man and k not in BASE else []) + v
    arm, base = [f"{a.arm}-s0", f"{a.arm}-s1"], list(BASE)
    names = base + arm
    Rs = {k: R.load_driver(man[k])[0] for k in names}
    scenes = sorted(set.intersection(*(set(Rs[k]) for k in names)))
    n = len(scenes)
    logs = np.array([R.log_of(s) for s in scenes])
    T = SR.turns()
    turn = np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])
    gt45 = np.abs(turn) > 45
    S = {k: SR.summary(Rs[k], scenes) for k in names}
    sc, zc = {k: S[k][0] for k in names}, {k: np.array([SHORT[z] for z in S[k][1]]) for k in names}
    prog = {k: np.array([(Rs[k][s].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0 for s in scenes]) for k in names}
    fresh = np.ones((2, n), bool)                                   # (seed, scene) pairs not in a development list
    for spec in a.dev:
        f, sd = spec.split(":")
        dev = set(Path(f).read_text().split())
        fresh[int(sd)] &= np.array([s not in dev for s in scenes])
    st = dict(scenes=n, logs=len(set(logs)), gt45=int(gt45.sum()), turn_missing=int(np.isnan(turn).sum()), fresh_pairs=int(fresh.sum()), drivers={k: S[k][2] | dict(mean_ci=R.ci(sc[k], logs)) for k in names})
    L = [f"Scenes common to all runs: {n} from {len(set(logs))} logs; logged 4 s turn > 45 deg: {int(gt45.sum())} (token not in lb_navtest: {int(np.isnan(turn).sum())}). "
         f"(seed, scene) pairs not in a development list: {int(fresh.sum())} of {2 * n}.", "", "## Drivers", "",
         "| driver | n | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | mean progress |", "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for k in names:
        s = S[k][2]
        L.append(f"| {k} | {s['n']} | {R.fmt(R.ci(sc[k], logs))} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['slow']} | {prog[k].mean():.3f} |")
    L += ["", "## Lines (the stricter of the two readings decides)", "",
          "| reading | pairs | L1a collision + offroad + corridor zeros, arm vs base (collision / offroad / corridor); per seed | L1a | L1b collision + offroad, per seed | L1b | L2 mean difference [95 % CI by log] | L2 | L3 slow arm vs 1.1 x base | L3 | all |",
          "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    st["lines"] = {}
    for name, M in (("A: all", np.ones((2, n), bool)), ("B: never-switched pairs", fresh)):
        w = M.sum(0)
        use = w > 0
        d = np.where(use, sum(np.where(M[i], sc[arm[i]] - sc[base[i]], 0.0) for i in range(2)) / np.maximum(w, 1), 0.0)[use]
        z = lambda ks, cl: [int((np.isin(zc[ks[i]], cl) & M[i]).sum()) for i in range(2)]  # noqa: E731
        ta, tb = z(arm, CLS), z(base, CLS)
        ca, cb = z(arm, CLS[:2]), z(base, CLS[:2])
        l1a = sum(ta) < sum(tb) and all(x <= y for x, y in zip(ta, tb))
        l1b = sum(ca) < sum(cb) and all(x <= y for x, y in zip(ca, cb))
        cls = lambda ks: " / ".join(str(sum(int(((zc[ks[i]] == f) & M[i]).sum()) for i in range(2))) for f in CLS)  # noqa: E731
        p_log, p_sc = stats.paired(d, np.zeros_like(d), groups=logs[use]), stats.paired(d, np.zeros_like(d))
        l2 = p_log["mean"] >= 0 and p_log["lo"] > -0.005
        sa, sb_ = sum(int((slow(sc[arm[i]]) & M[i]).sum()) for i in range(2)), sum(int((slow(sc[base[i]]) & M[i]).sum()) for i in range(2))
        l3 = sa <= 1.1 * sb_
        st["lines"][name] = dict(pairs=int(M.sum()), L1a=bool(l1a), L1b=bool(l1b), taught_arm=ta, taught_base=tb, co_arm=ca, co_base=cb, L2=bool(l2), diff_log=p_log, diff_scene=p_sc, L3=bool(l3),
                                 slow_arm=sa, slow_base=sb_, all=bool(l1a and l1b and l2 and l3))
        yn = lambda b: "met" if b else "not met"  # noqa: E731
        L.append(f"| {name} | {int(M.sum())} | {sum(ta)} ({cls(arm)}) vs {sum(tb)} ({cls(base)}); s0 {ta[0]} vs {tb[0]}, s1 {ta[1]} vs {tb[1]} | {yn(l1a)} | {sum(ca)} vs {sum(cb)}; s0 {ca[0]} vs {cb[0]}, s1 {ca[1]} vs {cb[1]} | {yn(l1b)} | "
                 f"{p_log['mean']:+.4f} [{p_log['lo']:+.4f}, {p_log['hi']:+.4f}] (by scene [{p_sc['lo']:+.4f}, {p_sc['hi']:+.4f}]) | {yn(l2)} | {sa} vs {1.1 * sb_:.1f} (base {sb_}) | {yn(l3)} | **{'yes' if l1a and l1b and l2 and l3 else 'no'}** |")
    st["verdict"] = bool(all(v["all"] for v in st["lines"].values()))
    ps = [stats.paired(sc[arm[i]], sc[base[i]], groups=logs) for i in range(2)]
    st["per_seed"] = ps
    L += ["", f"Counts are sums over the (seed, scene) pairs of the reading. Verdict: **{'all lines met on both readings' if st['verdict'] else 'not met'}**. Per seed (paired by scene, CI by log): "
          + "; ".join(f"s{i} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" for i, r in enumerate(ps)) + "."]
    flips = []
    L += ["", "## Zero changes per seed (all scenes)", "", "| seed | removed (collision / offroad / corridor) | new (collision / offroad / corridor) | base zero, still zero | new, mean score in scenes both ran | slow base -> arm |", "|:--|:--|:--|--:|--:|:--|"]
    for i in range(2):
        k, b = arm[i], base[i]
        rem, new = (sc[b] == 0) & (sc[k] > 0), (sc[b] > 0) & (sc[k] == 0)
        cnt = lambda m, z: " / ".join(str(int((m & (z == f)).sum())) for f in CLS)  # noqa: E731
        L.append(f"| s{i} | {int(rem.sum())} ({cnt(rem, zc[b])}) | {int(new.sum())} ({cnt(new, zc[k])}) | {int(((sc[b] == 0) & (sc[k] == 0)).sum())} | {sc[k].mean():.4f} vs {sc[b].mean():.4f} | {int(slow(sc[b]).sum())} -> {int(slow(sc[k]).sum())} |")
        for j in np.flatnonzero(rem | new | (sc[b] == 0) | (sc[k] == 0)):
            flips.append(dict(seed=i, scene=scenes[j], log=logs[j], fresh=bool(fresh[i, j]), turn4s_deg=round(float(turn[j]), 1), base_score=round(float(sc[b][j]), 4), base_zero=zc[b][j], arm_score=round(float(sc[k][j]), 4),
                              arm_zero=zc[k][j], change="removed" if rem[j] else "new" if new[j] else "same", base_progress=round(float(prog[b][j]), 3), arm_progress=round(float(prog[k][j]), 3)))
    if flips:
        with (out / "flips.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(flips[0]))
            w.writeheader(), w.writerows(flips)
    L += ["", "Every flip (seed, scene tail, base -> arm, zero class, turn deg):", ""]
    for f in flips:
        if f["change"] != "same":
            L.append(f"- s{f['seed']} `{f['scene'][-16:]}` {f['change']}: {f['base_score']} ({f['base_zero'] or '-'}) -> {f['arm_score']} ({f['arm_zero'] or '-'}); turn {f['turn4s_deg']}; {'fresh' if f['fresh'] else 'dev'}")
    rec = {"base": (sc[base[0]] + sc[base[1]]) / 2, a.arm: (sc[arm[0]] + sc[arm[1]]) / 2}
    L += ["", f"## Scenes whose logged 4 s future turns > 45 deg ({int(gt45.sum())} scenes, {len(set(logs[gt45]))} logs)", "", "| recipe | mean | zeros (sum of the two seeds) | collision / offroad / corridor | difference to base [95 % CI by log] |", "|:--|--:|--:|:--|:--|"]
    st["gt45_read"] = {}
    for g, v in (("base", base), (a.arm, arm)):
        zz = [int((sc[k][gt45] == 0).sum()) for k in v]
        cls = " / ".join(str(sum(int((zc[k][gt45] == f).sum()) for k in v)) for f in CLS)
        d = stats.paired(rec[g][gt45], rec["base"][gt45], groups=logs[gt45]) if g != "base" else None
        st["gt45_read"][g] = dict(mean=float(rec[g][gt45].mean()), zeros=zz, diff=d)
        L.append(f"| {g} | {rec[g][gt45].mean():.4f} | {sum(zz)} | {cls} | " + (f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] |" if d else "- |"))
    with (out / "per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "turn4s_deg", "fresh_s0", "fresh_s1"] + [f"{k}|{c}" for k in names for c in ("score", "zero_class", "progress")])
        for j, s in enumerate(scenes):
            w.writerow([s, logs[j], round(float(turn[j]), 1), int(fresh[0, j]), int(fresh[1, j])] + [x for k in names for x in (round(float(sc[k][j]), 6), zc[k][j], round(float(prog[k][j]), 4))])
    (out / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--run", required=True), p.add_argument("--base", required=True), p.add_argument("--out", required=True), p.add_argument("--expect", type=int, default=233)
    p = sub.add_parser("report")
    p.add_argument("--base-man", required=True), p.add_argument("--man", nargs="+", required=True), p.add_argument("--arm", default="P2H10B-F"), p.add_argument("--out", required=True)
    p.add_argument("--dev", nargs="*", default=[])
    a = ap.parse_args()
    {"check": cmd_check, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
