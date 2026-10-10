#!/usr/bin/env python3
"""BODY1, prereg Amendment 6 note (i): the DESCRIPTIVE 4-seed closed-loop read of the shape-only arm (P2H10S-F-s0..3) on the 700 AlpaSim nuPlan scenes.

Not a registered read: the arm stopped at G3 (b) and cannot be promoted by anything here. Reads the existing runs only.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/shape_cl_report.py --base-man <tr1 manifest> <base23 manifest> --arm-man <manifest> [--loss-man M] --out DIR

Manifests are ot2_loop.py's {label: [run dirs]}; labels P2H10-F-s<k> (base), P2H10S-F-s<k> (arm), P2H10B-F-s<k> (earlier loss arm, optional).
Statistics of a set of (arm run, base run) pairs, computed as loss_report.py does:
  L1a  at-fault collision + offroad + corridor zeros, arm minus base, summed over the pairs (met: total down and no pair up)
  L1b  collision + offroad zeros, the same
  L2   mean per-scene score difference (averaged over the pairs), CI by log (jevdrive.stats.paired), met: mean >= 0 and lower bound > -0.005
  L3   slow scenes (0 < score < 1), arm against 1.1 x base, summed over the pairs
Null distributions of the same statistics (same recipe, only the seed differs), the base's own seed spread:
  N1  the 12 ordered pairs of base seeds, one pair each
  N2  the 12 disjoint 2-vs-2 splits (6 choices of the two "arm" seeds x 2 pairings), two pairs each
Per-pair normalised values (divided by the number of pairs) are compared across sets of different size. "Outside the base's seed spread" is claimed only for a
value beyond every member of N1 and of N2 (so beyond all 12 of each).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import c0b_report as R  # noqa: E402
import stop_report as SR  # noqa: E402

CLS = ("collision", "offroad", "corridor")
slow = lambda x: (x > 0) & (x < 1)  # noqa: E731
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
BASE = [f"P2H10-F-s{i}" for i in range(4)]
ARM = [f"P2H10S-F-s{i}" for i in range(4)]
LOSS = [f"P2H10B-F-s{i}" for i in range(2)]


class Ctx:
    """Per-run arrays on the common scenes."""

    def __init__(self, man, names, mask_fn=None):
        Rs = {k: R.load_driver(man[k])[0] for k in names}
        self.scenes = sorted(set.intersection(*(set(Rs[k]) for k in names)))
        s = self.scenes
        self.logs = np.array([R.log_of(x) for x in s])
        T = SR.turns()
        self.turn = np.array([T.get(x.rsplit("-", 1)[1], np.nan) for x in s])
        self.gt45 = np.abs(self.turn) > 45
        S = {k: SR.summary(Rs[k], s) for k in names}
        self.sc = {k: S[k][0] for k in names}
        self.zc = {k: np.array([SR.SHORT[z] for z in S[k][1]]) for k in names}
        self.info = {k: S[k][2] for k in names}
        self.prog = {k: np.array([(Rs[k][x].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0 for x in s]) for k in names}
        self.names = names


def pair_stats(C, pairs, mask=None):
    """Statistics of a set of (arm, base) pairs on the scenes in `mask` (default all)."""
    from jevdrive import stats
    m = np.ones(len(C.scenes), bool) if mask is None else mask
    n = len(pairs)
    d = sum(C.sc[a] - C.sc[b] for a, b in pairs)[m] / n
    p = stats.paired(d, np.zeros_like(d), groups=C.logs[m])
    z = lambda k, cl: int(np.isin(C.zc[k][m], cl).sum())  # noqa: E731
    za = [z(a, CLS) for a, _ in pairs]
    zb = [z(b, CLS) for _, b in pairs]
    ca = [z(a, CLS[:2]) for a, _ in pairs]
    cb = [z(b, CLS[:2]) for _, b in pairs]
    sa = sum(int(slow(C.sc[a][m]).sum()) for a, _ in pairs)
    sb = sum(int(slow(C.sc[b][m]).sum()) for _, b in pairs)
    pr = sum(C.prog[a][m].mean() - C.prog[b][m].mean() for a, b in pairs) / n
    cl = lambda ks: [sum(int((C.zc[k][m] == f).sum()) for k in ks) for f in CLS]  # noqa: E731
    return dict(pairs=n, scenes=int(m.sum()), mean=p["mean"], lo=p["lo"], hi=p["hi"], zeros_arm=za, zeros_base=zb, co_arm=ca, co_base=cb,
                dz1a=(sum(za) - sum(zb)) / n, dz1b=(sum(ca) - sum(cb)) / n, slow_arm=sa, slow_base=sb, dslow=(sa - sb) / n, dprog=float(pr),
                cls_arm=cl([a for a, _ in pairs]), cls_base=cl([b for _, b in pairs]),
                L1a=bool(sum(za) < sum(zb) and all(x <= y for x, y in zip(za, zb))), L1b=bool(sum(ca) < sum(cb) and all(x <= y for x, y in zip(ca, cb))),
                L2=bool(p["mean"] >= 0 and p["lo"] > -0.005), L3=bool(sa <= 1.1 * sb))


def nulls(C, mask=None):
    n1 = {f"{a[-2:]} vs {b[-2:]}": pair_stats(C, [(a, b)], mask) for a, b in itertools.permutations(BASE, 2)}
    n2 = {}
    for arm in itertools.combinations(BASE, 2):
        rest = [b for b in BASE if b not in arm]
        for perm in (rest, rest[::-1]):
            n2[f"{arm[0][-2:]},{arm[1][-2:]} vs {perm[0][-2:]},{perm[1][-2:]}"] = pair_stats(C, list(zip(arm, perm)), mask)
    return n1, n2


KEYS = (("mean", "L2 mean difference", "{:+.4f}"), ("lo", "L2 CI lower bound", "{:+.4f}"), ("dz1a", "L1a zeros per pair (arm - base)", "{:+.2f}"),
        ("dz1b", "L1b zeros per pair (arm - base)", "{:+.2f}"), ("dslow", "slow scenes per pair (arm - base)", "{:+.1f}"), ("dprog", "mean progress per pair (arm - base)", "{:+.4f}"))


def vs_null(v, n1, n2):
    """Where a statistic lies against N1 and N2: min..max of each, rank, and 'outside' only if beyond all members of both."""
    out = {}
    for k, _, _ in KEYS:
        a, b = np.array([x[k] for x in n1.values()]), np.array([x[k] for x in n2.values()])
        x = v[k]
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        out[k] = dict(value=x, n1=[float(a.min()), float(a.max())], n2=[float(b.min()), float(b.max())], below_n1=int((a < x).sum()), below_n2=int((b < x).sum()),
                      outside="below" if x < lo else "above" if x > hi else "no")
    return out


def fmtnull(k, d, f):
    return f"{f.format(d['n1'][0])} to {f.format(d['n1'][1])} / {f.format(d['n2'][0])} to {f.format(d['n2'][1])}; {d['below_n1']} of 12 / {d['below_n2']} of 12 below; **{d['outside']}**"


def main(a):
    from jevdrive import stats
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = {}
    for m in [*a.base_man, a.arm_man, *([a.loss_man] if a.loss_man else [])]:
        man |= json.loads(Path(m).read_text())
    loss = [k for k in LOSS if k in man]
    names = BASE + ARM + loss
    C = Ctx(man, names)
    n = len(C.scenes)
    st = dict(scenes=n, logs=len(set(C.logs)), gt45=int(C.gt45.sum()), drivers={})
    L = [f"Scenes common to all {len(names)} runs: {n} from {len(set(C.logs))} logs; logged 4 s turn > 45 deg: {int(C.gt45.sum())} (token missing: {int(np.isnan(C.turn).sum())}).",
         "", "## 1. Per seed", "",
         "| driver | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | mean progress | > 45 deg mean | > 45 deg zeros |",
         "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in names:
        s = C.info[k]
        st["drivers"][k] = s | dict(mean_ci=R.ci(C.sc[k], C.logs), progress=float(C.prog[k].mean()), gt45_mean=float(C.sc[k][C.gt45].mean()), gt45_zeros=int((C.sc[k][C.gt45] == 0).sum()))
        L.append(f"| {k} | {R.fmt(R.ci(C.sc[k], C.logs))} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['slow']} | {C.prog[k].mean():.3f} | "
                 f"{C.sc[k][C.gt45].mean():.4f} | {int((C.sc[k][C.gt45] == 0).sum())} |")
    d = st["drivers"]
    rng = lambda ks, f, fm="{}": f"{fm.format(min(d[k][f] for k in ks))} to {fm.format(max(d[k][f] for k in ks))}"  # noqa: E731
    L += ["", "Ranges over the four seeds, base against arm: mean " + f"{rng(BASE, 'mean', '{:.4f}')} / {rng(ARM, 'mean', '{:.4f}')}; zeros {rng(BASE, 'zeros')} / {rng(ARM, 'zeros')}; "
          f"at-fault collision {rng(BASE, 'collision')} / {rng(ARM, 'collision')}; offroad {rng(BASE, 'offroad')} / {rng(ARM, 'offroad')}; corridor {rng(BASE, 'corridor')} / {rng(ARM, 'corridor')}; "
          f"slow {rng(BASE, 'slow')} / {rng(ARM, 'slow')}; progress {rng(BASE, 'progress', '{:.3f}')} / {rng(ARM, 'progress', '{:.3f}')}."]
    for k in ("mean", "zeros", "collision", "offroad", "corridor", "slow", "ones", "progress"):
        st.setdefault("means", {})[k] = dict(base=float(np.mean([d[b][k] for b in BASE])), arm=float(np.mean([d[b][k] for b in ARM])))
    L += ["", "Mean over the four seeds, base / arm: " + "; ".join(f"{k} {st['means'][k]['base']:.4g} / {st['means'][k]['arm']:.4g}" for k in st["means"]) + "."]

    # --- nulls and reads
    n1, n2 = nulls(C)
    st["null_n1"], st["null_n2"] = n1, n2
    reads = {"S 2 seeds (s0-1) vs TR1 base s0-1": [(ARM[i], BASE[i]) for i in (0, 1)], "S 4 seeds paired by seed index": [(ARM[i], BASE[i]) for i in range(4)]}
    if len(loss) == 2:
        reads["B (loss arm, Amendment 5) 2 seeds, for comparison"] = [(LOSS[i], BASE[i]) for i in (0, 1)]
    L += ["", "## 2. The four lines (descriptive; no promotion)", "",
          "| read | pairs | L1a zeros arm vs base (collision / offroad / corridor) | L1b | L2 mean difference [95 % CI by log] | L3 slow arm vs 1.1 x base | lines that would read as met |", "|:--|--:|:--|:--|:--|:--|:--|"]
    st["reads"], st["vs_null"] = {}, {}
    yn = lambda b: "met" if b else "not met"  # noqa: E731
    for name, pr in reads.items():
        s = pair_stats(C, pr)
        st["reads"][name] = s
        st["vs_null"][name] = vs_null(s, n1, n2)
        ok = [q for q in ("L1a", "L1b", "L2", "L3") if s[q]]
        L.append(f"| {name} | {s['pairs'] * n} | {sum(s['zeros_arm'])} ({' / '.join(map(str, s['cls_arm']))}) vs {sum(s['zeros_base'])} ({' / '.join(map(str, s['cls_base']))}); per pair {s['zeros_arm']} vs {s['zeros_base']} ({yn(s['L1a'])}) | "
                 f"{sum(s['co_arm'])} vs {sum(s['co_base'])}; per pair {s['co_arm']} vs {s['co_base']} ({yn(s['L1b'])}) | {s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] ({yn(s['L2'])}) | "
                 f"{s['slow_arm']} vs {1.1 * s['slow_base']:.1f} (base {s['slow_base']}) ({yn(s['L3'])}) | {', '.join(ok) or '-'} |")
    L += ["", "Per pair, paired by scene (CI by log): " + "; ".join(f"{ARM[i][-2:]} vs base {BASE[i][-2:]} {(p := stats.paired(C.sc[ARM[i]], C.sc[BASE[i]], groups=C.logs))['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]" for i in range(4)) + ".", "",
          "## 3. Each statistic against the base's seed spread", "",
          "N1 = the 12 ordered pairs of base seeds (one pair each), N2 = the 12 disjoint 2-vs-2 splits (two pairs each); per-pair normalised values. Cells: arm value; N1 range / N2 range; "
          "number of null members below the value (of 12 / of 12); **outside** only if beyond every member of both.", "",
          "| statistic | " + " | ".join(reads) + " |", "|:--|" + ":--|" * len(reads)]
    for k, lab, f in KEYS:
        L.append(f"| {lab} | " + " | ".join(f"{f.format(st['vs_null'][nm][k]['value'])}; " + fmtnull(k, st['vs_null'][nm][k], f) for nm in reads) + " |")
    L += ["", "L2 / L3 / L1 verdict of the null members (how often a same-recipe pair would read as met): "
          f"N1 L1a {sum(v['L1a'] for v in n1.values())}, L1b {sum(v['L1b'] for v in n1.values())}, L2 {sum(v['L2'] for v in n1.values())}, L3 {sum(v['L3'] for v in n1.values())}, all four {sum(all((v['L1a'], v['L1b'], v['L2'], v['L3'])) for v in n1.values())} of 12; "
          f"N2 L1a {sum(v['L1a'] for v in n2.values())}, L1b {sum(v['L1b'] for v in n2.values())}, L2 {sum(v['L2'] for v in n2.values())}, L3 {sum(v['L3'] for v in n2.values())}, all four {sum(all((v['L1a'], v['L1b'], v['L2'], v['L3'])) for v in n2.values())} of 12."]
    # --- flips by the 3-of-4 / 1-of-4 rule
    zb = np.array([C.sc[k] == 0 for k in BASE])
    za = np.array([C.sc[k] == 0 for k in ARM])
    nb, na = zb.sum(0), za.sum(0)
    rem, new = (nb >= 3) & (na <= 1), (na >= 3) & (nb <= 1)
    both = (nb >= 3) & (na >= 3)
    mode = lambda ks, j: max(CLS, key=lambda f: sum(C.zc[k][j] == f for k in ks))  # noqa: E731
    flips = []
    for j in np.flatnonzero(rem | new | both):
        flips.append(dict(scene=C.scenes[j], log=C.logs[j], turn4s_deg=round(float(C.turn[j]), 1), change="removed" if rem[j] else "new" if new[j] else "zero in both",
                          base_zeros=int(nb[j]), arm_zeros=int(na[j]), base_class=mode(BASE, j) if nb[j] else "", arm_class=mode(ARM, j) if na[j] else "",
                          base_mean=round(float(np.mean([C.sc[k][j] for k in BASE])), 3), arm_mean=round(float(np.mean([C.sc[k][j] for k in ARM])), 3),
                          **{f"{k}": round(float(C.sc[k][j]), 3) for k in BASE + ARM}))
    with (out / "flips.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(flips[0]))
        w.writeheader(), w.writerows(flips)
    cnt = lambda m, ks: " / ".join(str(sum(int(mode(ks, j) == f) for j in np.flatnonzero(m))) for f in CLS)  # noqa: E731
    # null for flips: a same-recipe "arm" of the 2 other seeds is not definable with the 3-of-4 rule; use leave-one-out on the base itself: seed k as "arm" (1 run) is not comparable either
    st["flips"] = dict(removed=int(rem.sum()), new=int(new.sum()), both=int(both.sum()), removed_cls=cnt(rem, BASE), new_cls=cnt(new, ARM),
                       zero_in_all_base=int((nb == 4).sum()), zero_in_any_base=int((nb >= 1).sum()), zero_in_any_arm=int((na >= 1).sum()), zero_in_all_arm=int((na == 4).sum()))
    L += ["", "## 4. Zero flips per scene against all four base seeds", "",
          "Rule: removed = zero in at least 3 of 4 base seeds and in at most 1 of 4 arm seeds; new = the mirror image.", "",
          "| | scenes | by class (collision / offroad / corridor; class by majority over the seeds that are zeros) |", "|:--|--:|:--|",
          f"| removed | {int(rem.sum())} | {cnt(rem, BASE)} |", f"| new | {int(new.sum())} | {cnt(new, ARM)} |", f"| zero in at least 3 of 4 seeds of both | {int(both.sum())} | - |", "",
          f"Scenes with a zero in all four base seeds {st['flips']['zero_in_all_base']}, in any {st['flips']['zero_in_any_base']}; in all four arm seeds {st['flips']['zero_in_all_arm']}, in any {st['flips']['zero_in_any_arm']}. "
          "Flips (scene tail, base zeros of 4 -> arm zeros of 4, class, turn):", ""]
    for f in flips:
        if f["change"] != "zero in both":
            L.append(f"- `{f['scene'][-16:]}` {f['change']}: {f['base_zeros']}/4 ({f['base_class'] or '-'}) -> {f['arm_zeros']}/4 ({f['arm_class'] or '-'}); turn {f['turn4s_deg']}; scores base {[f[k] for k in BASE]} arm {[f[k] for k in ARM]}")
    # --- lead / open split
    import prog_ol
    T = pd.read_parquet(DATA / "runs/body1/prog/ol_navtest.parquet")
    T["grp"] = prog_ol.group(T)
    T = T.set_index("name")
    tok = [x.rsplit("-", 1)[1] for x in C.scenes]
    grp = np.array([T.grp.get(t, "") for t in tok])
    st["groups"] = {}
    L += ["", "## 5. Lead / open split (proximity group of the base plan at the scene's navtest token, decision 230)", "",
          "| group | scenes | base mean progress | arm mean progress | difference per pair [95 % CI by log] | slow base -> arm (sum of 4 pairs) | N1 progress range | N2 progress range | slow per pair N1 range | outside |", "|:--|--:|--:|--:|:--|:--|:--|:--|:--|:--|"]
    pairs4 = reads["S 4 seeds paired by seed index"]
    for g in ("lead", "open", "contact", "near obj", "near edge"):
        m = grp == g
        if m.sum() < 5:
            continue
        s = pair_stats(C, pairs4, m)
        dp = sum(C.prog[a][m] - C.prog[b][m] for a, b in pairs4) / 4
        pp = stats.paired(dp, np.zeros_like(dp), groups=C.logs[m])
        g1, g2 = nulls(C, m)
        a1, a2 = np.array([v["dprog"] for v in g1.values()]), np.array([v["dprog"] for v in g2.values()])
        s1, s2 = np.array([v["dslow"] for v in g1.values()]), np.array([v["dslow"] for v in g2.values()])
        out_p = "progress " + ("below" if s["dprog"] < min(a1.min(), a2.min()) else "above" if s["dprog"] > max(a1.max(), a2.max()) else "no")
        out_s = "slow " + ("below" if s["dslow"] < min(s1.min(), s2.min()) else "above" if s["dslow"] > max(s1.max(), s2.max()) else "no")
        st["groups"][g] = dict(scenes=int(m.sum()), base_prog=float(np.mean([C.prog[b][m].mean() for b in BASE])), arm_prog=float(np.mean([C.prog[a][m].mean() for a in ARM])), dprog=pp, slow_arm=s["slow_arm"], slow_base=s["slow_base"],
                               null_prog=[float(a1.min()), float(a1.max()), float(a2.min()), float(a2.max())], null_slow=[float(s1.min()), float(s1.max()), float(s2.min()), float(s2.max())], outside=[out_p, out_s])
        L.append(f"| {g} | {int(m.sum())} | {st['groups'][g]['base_prog']:.3f} | {st['groups'][g]['arm_prog']:.3f} | {pp['mean']:+.4f} [{pp['lo']:+.4f}, {pp['hi']:+.4f}] | {s['slow_base']} -> {s['slow_arm']} | "
                 f"{a1.min():+.4f} to {a1.max():+.4f} | {a2.min():+.4f} to {a2.max():+.4f} | {s1.min():+.1f} to {s1.max():+.1f} | {out_p}; {out_s} |")
    # --- > 45 deg
    m45 = C.gt45
    s45 = pair_stats(C, pairs4, m45)
    g1, g2 = nulls(C, m45)
    L += ["", f"## 6. Scenes turning more than 45 deg ({int(m45.sum())} scenes, {len(set(C.logs[m45]))} logs; logged 4 s future of the scene's navtest token)", "",
          "| driver | mean | zeros | collision / offroad / corridor |", "|:--|--:|--:|:--|"]
    for k in BASE + ARM + loss:
        z = C.zc[k][m45]
        L.append(f"| {k} | {C.sc[k][m45].mean():.4f} | {int((C.sc[k][m45] == 0).sum())} | {' / '.join(str(int((z == f).sum())) for f in CLS)} |")
    a1, a2 = np.array([v["mean"] for v in g1.values()]), np.array([v["mean"] for v in g2.values()])
    z1, z2 = np.array([v["dz1a"] for v in g1.values()]), np.array([v["dz1a"] for v in g2.values()])
    st["gt45_read"] = dict(s4=s45, null_mean=[float(a1.min()), float(a1.max()), float(a2.min()), float(a2.max())], null_dz=[float(z1.min()), float(z1.max()), float(z2.min()), float(z2.max())])
    out45 = lambda x, a, b: "below" if x < min(a.min(), b.min()) else "above" if x > max(a.max(), b.max()) else "no"  # noqa: E731
    L += ["", f"Arm minus base, four seeds paired by seed index: mean difference {s45['mean']:+.4f} [{s45['lo']:+.4f}, {s45['hi']:+.4f}] (CI by log); zeros per pair {s45['dz1a']:+.2f} (arm {sum(s45['zeros_arm'])} vs base {sum(s45['zeros_base'])}, "
          f"classes {' / '.join(map(str, s45['cls_arm']))} vs {' / '.join(map(str, s45['cls_base']))}). Null, same bucket: mean difference N1 {a1.min():+.4f} to {a1.max():+.4f}, N2 {a2.min():+.4f} to {a2.max():+.4f}; "
          f"zeros per pair N1 {z1.min():+.2f} to {z1.max():+.2f}, N2 {z2.min():+.2f} to {z2.max():+.2f}. Outside the base's spread: mean {out45(s45['mean'], a1, a2)}, zeros {out45(s45['dz1a'], z1, z2)}."]
    # --- tables
    with (out / "per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "turn4s_deg", "group"] + [f"{k}|{c}" for k in names for c in ("score", "zero_class", "progress")])
        for j, s in enumerate(C.scenes):
            w.writerow([s, C.logs[j], round(float(C.turn[j]), 1), grp[j]] + [x for k in names for x in (round(float(C.sc[k][j]), 6), C.zc[k][j], round(float(C.prog[k][j]), 4))])
    (out / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-man", nargs="+", required=True)
    ap.add_argument("--arm-man", required=True)
    ap.add_argument("--loss-man", default="")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
